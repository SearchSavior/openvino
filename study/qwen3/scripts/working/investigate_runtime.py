"""Compares baseline (no rewrites, no .so) vs v3 (full fusion + .so) with
PERF_COUNT=True and reports:

  1. Per-(op_type|exec_type) prefill time aggregated by layer bucket.
  2. Runtime-model node counts per bucket.
  3. Activation bytes on a bound shape (T_q=128) bucketed by sub-arch,
     plus a per-(op_type, shape, dtype) rollup so individual tensor
     classes are visible.
  4. Per-call exec_time for the GatedDeltaRule custom op.

Findings and interpretation live in DISCUSSION.md, not here.

Run:
    cd study/qwen3
    QWEN3_USE_C=1 python3 scripts/working/investigate_runtime.py
"""
import sys, time, json, shutil
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import numpy as np
import openvino as ov
from fused_linear_attn import (
    register as rla,
    replace_gated_delta_rule_loops_v3,
)
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = str(Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so")

ETYPE = {"f32":4,"f16":2,"bf16":2,"i64":8,"i32":4,"i8":1,"u8":1,"boolean":1}
WK = ("embed_tokens","lm_head","_proj/",".weight",".bias","rotary","inv_freq",
      "A_log","ssm_dt","ssm_a","ssm_beta","ssm_alpha","ssm_norm","ssm_conv1d",
      "norm.weight","input_layernorm","k_norm","q_norm","attn_norm")


def shape_bytes(ps, et):
    if not all(d.is_static for d in ps): return 0
    n = 1
    for d in ps: n *= d.get_length()
    return n * ETYPE.get(et, 4)


def bucket(name, type_name=""):
    if "linear_attn" in name: return "linear_attn"
    if "self_attn"   in name: return "self_attn"
    if "mlp"         in name: return "mlp"
    # Custom fused ops carry generic names (GatedDeltaRule/V2/V3_<id>). They
    # *are* linear-attn work; bucket them there so the comparison is honest.
    if type_name.startswith("GatedDeltaRule"): return "linear_attn"
    return "other"


def build_baseline_xml():
    """Slice lm_head only (same prefill output shape as v3). No GDR rewrite."""
    out = "/tmp/lm_baseline.xml"
    m = ov.Core().read_model(str(ORIG / "openvino_language_model.xml"))
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace(".xml", ".bin"))
    return out


def build_v3_xml():
    out = "/tmp/lm_v3.xml"
    c = ov.Core(); rla(c)
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))
    replace_gated_delta_rule_loops_v3(m)
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace(".xml", ".bin"))
    return out


def compile_and_analyze(label, xml, with_ext, T_q=128):
    core = ov.Core()
    if with_ext:
        core.add_extension(SO)
    lm = core.read_model(xml)

    # Bind shapes so activation bytes are computable.
    bind = {}
    for p in lm.get_parameters():
        n = p.get_friendly_name(); ps = p.get_partial_shape()
        if   n == "inputs_embeds":   bind[n] = ov.PartialShape([1, T_q, ps[2]])
        elif n == "attention_mask":  bind[n] = ov.PartialShape([1, T_q])
        elif n == "position_ids":    bind[n] = ov.PartialShape([ps[0], 1, T_q])
        elif n == "beam_idx":        bind[n] = ov.PartialShape([1])
    lm_anal = lm.clone(); lm_anal.reshape(bind)

    # Activation accounting on the reshaped graph (compiled with stub options
    # — we won't infer through this one because state-variable Assigns stay
    # dynamic under reshape; we just need get_runtime_model()).
    anal = core.compile_model(lm_anal, "CPU",
                              {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})
    rt = anal.get_runtime_model()
    del anal

    # Real inference + profiling on the original dynamic graph.
    compiled = core.compile_model(lm, "CPU",
                                  {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})

    # 1. activation bytes, bucketed; also keep top-N tensors per bucket.
    act_bytes = defaultdict(int); act_total = 0
    top_tensors = defaultdict(list)  # bucket -> [(bytes, name, type, shape, et)]
    for op in rt.get_ops():
        n = op.get_friendly_name()
        if any(k in n for k in WK): continue
        tn = op.get_type_name()
        for i in range(op.get_output_size()):
            ps = op.get_output_partial_shape(i)
            et = op.get_output_element_type(i).get_type_name()
            b = shape_bytes(ps, et)
            if b < 1024: continue
            bk = bucket(n, tn)
            act_bytes[bk] += b
            act_total += b
            top_tensors[bk].append((b, n, tn, str(ps), et))

    # 2. node-count + exec_type distribution by bucket
    node_count = defaultdict(int)
    impl_count = defaultdict(lambda: defaultdict(int))
    for op in rt.get_ops():
        n = op.get_friendly_name()
        if any(k in n for k in WK): continue
        type_name = op.get_type_name()
        bk = bucket(n, type_name)
        node_count[bk] += 1
        try:
            impl = str(op.get_rt_info()["primitiveType"])
        except Exception:
            impl = "?"
        impl_count[bk][f"{type_name}|{impl}"] += 1

    # 3. one inference for profiling — random-ish prompt, single chunk
    rng = np.random.default_rng(0)
    ids = rng.integers(1, 200000, size=(1, T_q), dtype=np.int64)
    # for profiling we don't have the embed model loaded; fabricate the embeds
    # at the right hidden size.
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    embeds = rng.standard_normal((1, T_q, hidden), dtype=np.float32) * 0.01
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    req = compiled.create_infer_request()
    feeds = {
        "inputs_embeds":  ov.Tensor(embeds),
        "attention_mask": ov.Tensor(np.ones((1, T_q), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T_q, dtype=np.int64).reshape(1,1,T_q),
                                            (pid_b,1,1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req.infer(feeds); req.infer(feeds)  # warm + measure
    t0 = time.time(); req.infer(feeds); t_infer = time.time() - t0

    # 4. profiling info: aggregate exec_time_us by (bucket, exec_type)
    prof = req.get_profiling_info()
    by_impl = defaultdict(lambda: defaultdict(float))   # bucket -> impl -> us
    by_op_us = defaultdict(float)                       # bucket -> total us
    custom_gdr_us = []
    for p in prof:
        if p.status != ov.ProfilingInfo.Status.EXECUTED:
            continue
        n = p.node_name
        if any(k in n for k in WK): continue
        b = bucket(n, p.node_type)
        us = float(p.real_time.total_seconds() * 1e6)
        by_impl[b][f"{p.node_type}|{p.exec_type}"] += us
        by_op_us[b] += us
        if "GatedDeltaRule" in p.node_type:
            custom_gdr_us.append((n, us, p.exec_type))

    print(f"\n{'='*78}\n## {label}    (one prefill @ T_q={T_q}: {t_infer*1000:.1f} ms)\n{'='*78}")

    print(f"  activation bytes (bound T_q={T_q}):  TOTAL {act_total/(1<<20):.1f} MiB")
    for b in ("linear_attn", "self_attn", "mlp", "other"):
        print(f"    {b:<12s}  {act_bytes[b]/(1<<20):>7.1f} MiB   (runtime nodes: {node_count[b]})")

    # Per-tensor-shape rollup for linear_attn: groups by (op_type, shape, dtype)
    # so we see "we have 18 tensors of shape [1,128,6144] f32 totalling X MiB".
    print(f"\n  linear_attn: activation tensors grouped by (op_type, shape, dtype):")
    rollup = defaultdict(lambda: [0, 0])  # key -> [count, total_bytes]
    for b, n, tn, sh, et in top_tensors["linear_attn"]:
        k = f"{tn:<20s} {sh:<30s} {et}"
        rollup[k][0] += 1
        rollup[k][1] += b
    for k, (cnt, tot) in sorted(rollup.items(), key=lambda x: -x[1][1])[:14]:
        print(f"    {tot/(1<<20):>6.1f} MiB   x{cnt:>3d}   {k}")

    print(f"\n  per-bucket prefill time (us, from PERF_COUNT):")
    tot = sum(by_op_us.values())
    for b in ("linear_attn", "self_attn", "mlp", "other"):
        share = 100 * by_op_us[b] / tot if tot else 0
        print(f"    {b:<12s}  {by_op_us[b]/1000:>9.2f} ms   ({share:5.1f} %)")

    print(f"\n  top 12 (op_type | exec_type) by exec time per bucket:")
    for b in ("linear_attn", "self_attn", "mlp"):
        rows = sorted(by_impl[b].items(), key=lambda x: -x[1])[:12]
        print(f"  --- {b}:")
        for k, us in rows:
            print(f"      {us/1000:>7.2f} ms   {k}")

    if custom_gdr_us:
        print(f"\n  GatedDeltaRule custom-op per-call exec time:")
        tots = sum(us for _, us, _ in custom_gdr_us)
        avg = tots / len(custom_gdr_us)
        print(f"    {len(custom_gdr_us)} instances, total {tots/1000:.2f} ms, avg {avg/1000:.2f} ms each")
        print(f"    exec_type sample: {custom_gdr_us[0][2]}")

    return dict(label=label, act_total=act_total, t_infer=t_infer,
                act_bytes=dict(act_bytes), by_op_us=dict(by_op_us))


def main():
    print("[1] building baseline (lm_head_slice only)…")
    xml_b = build_baseline_xml()
    print("[2] building v3 fusion IR…")
    xml_v = build_v3_xml()

    rb = compile_and_analyze("BASELINE (stock plugin, no .so)", xml_b, with_ext=False)
    rv = compile_and_analyze("V3 (custom .so, full fusion)", xml_v, with_ext=True)

    print(f"\n{'='*78}\n## DELTA  v3 − baseline\n{'='*78}")
    print(f"  total activation:  {(rv['act_total']-rb['act_total'])/(1<<20):+.1f} MiB")
    for b in ("linear_attn", "self_attn", "mlp", "other"):
        d = (rv['act_bytes'].get(b,0) - rb['act_bytes'].get(b,0)) / (1<<20)
        print(f"    {b:<12s}  {d:+7.1f} MiB")
    print(f"  one-prefill time:  {(rv['t_infer']-rb['t_infer'])*1000:+.1f} ms")
    for b in ("linear_attn", "self_attn", "mlp", "other"):
        d = (rv['by_op_us'].get(b,0) - rb['by_op_us'].get(b,0)) / 1000
        print(f"    {b:<12s}  {d:+7.2f} ms")


if __name__ == "__main__":
    main()
