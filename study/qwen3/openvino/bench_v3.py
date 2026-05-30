"""pp512+tg32 comparing v1 / v2 / v3 with serialize/reload so the C++
extension actually wins evaluate().

v3 absorbs the conv1d-with-state + SiLU + Transposes chain in addition to
what v2 absorbed (split / reshape / L2-norm / Q-scale / transpose). Goal:
eliminate ~21 IR-level edge buffers per linear-attn layer, saving another
~250-300 MiB of addressable activation memory beyond v2.

See bench_v2.py for the serialize/reload pattern: the Python `Op` subclass
takes priority over the .so when both register the same op name, so we
serialize the IR after Python-class construction and re-load it with a
fresh ov.Core() that has only the .so registered.

Last measured output (this commit, INFERENCE_NUM_THREADS=4, chunk=128):

    === A. v1 (C++ ext) ===
      activation budget @ T_q=128: 687.4 MiB total
        linear_attn        464.2 MiB
        self_attn           33.1 MiB
        mlp                 21.0 MiB
        other              169.1 MiB
      pp512: 5.07s (100.90 tok/s)
      tg32:  2.04s (15.66 tok/s)

    === B. v2 (+ split/L2/scale/transpose) ===
      activation budget @ T_q=128: 579.1 MiB total
        linear_attn        355.9 MiB
        self_attn           33.1 MiB
        mlp                 21.0 MiB
        other              169.1 MiB
      pp512: 2.09s (245.17 tok/s)
      tg32:  2.02s (15.87 tok/s)

    === C. v3 (+ conv1d/SiLU/Transposes) ===
      activation budget @ T_q=128: 390.7 MiB total
        linear_attn        164.2 MiB
        self_attn           33.1 MiB
        mlp                 21.0 MiB
        other              172.5 MiB
      pp512: 2.76s (185.59 tok/s)
      tg32:  2.08s (15.39 tok/s)

    SUMMARY (pp512 + tg32, chunk=128, threads=4, all via serialize/reload)
    metric                              A v1            B v2            C v3
      activation budget          687.35 MiB      579.07 MiB      390.70 MiB
        linear_attn              464.20 MiB      355.92 MiB      164.18 MiB
      pp512 throughput           100.90 tok/s     245.17 tok/s     185.59 tok/s
      tg32 throughput             15.66 tok/s      15.87 tok/s      15.39 tok/s
      pp512 duration               5.07 s          2.09 s          2.76 s
      tg32 duration                2.04 s          2.02 s          2.08 s

v3 vs v1:    -296 MiB activation (-43 %), -300 MiB linear_attn (-65 %),
             +84 % prefill throughput, ~parity decode.
v3 vs v2:    -188 MiB activation (-33 %), -192 MiB linear_attn (-54 %),
             -24 % prefill (the absorbed conv1d kernel is less optimised than
             the plugin's blocked GroupConvolution), parity decode.

v3's 391 MiB total activation budget is *below* llama.cpp's measured 491 MiB
compute buffer for the same workload. Per linear-attn layer: 25.8 MiB
(v1) -> 9.1 MiB (v3), a ~65 % per-layer reduction.

Generation identical to baseline: "I am Qwen3.5, the latest large language
model developed by Tongyi Lab. I am a text-based AI model, so I don't".
"""
import sys, time
import openvino as ov
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
from fused_linear_attn import (
    register as rla,
    replace_gated_delta_rule_loops,
    replace_gated_delta_rule_loops_v2,
    replace_gated_delta_rule_loops_v3,
)
from lm_head_slice import slice_lm_head_to_last_token

MODEL = "/tmp/qwen3-work/qwen35-0.8b-int8"
SO = "/home/user/openvino/study/qwen3/cpp_ext/build/libqwen3_ov_ext.so"

ETYPE = {"f32":4,"f16":2,"bf16":2,"i64":8,"i32":4,"i8":1,"u8":1,"boolean":1}
WK = ("embed_tokens","lm_head","_proj/",".weight",".bias","rotary","inv_freq",
      "A_log","ssm_dt","ssm_a","ssm_beta","ssm_alpha","ssm_norm","ssm_conv1d",
      "norm.weight","input_layernorm","k_norm","q_norm","attn_norm")


def shape_bytes(ps, et):
    if not all(d.is_static for d in ps): return 0
    n = 1
    for d in ps: n *= d.get_length()
    return n * ETYPE.get(et, 4)


def build_and_serialize(label, version, xml_path):
    """Build IR with Python classes, serialize. `version` in {1, 2, 3}."""
    core_build = ov.Core(); rla(core_build)
    lm = core_build.read_model(f"{MODEL}/openvino_language_model.xml")
    if version == 1:   replace_gated_delta_rule_loops(lm)
    elif version == 2: replace_gated_delta_rule_loops_v2(lm)
    elif version == 3: replace_gated_delta_rule_loops_v3(lm)
    slice_lm_head_to_last_token(lm)
    ov.serialize(lm, xml_path, xml_path.replace(".xml", ".bin"))


def measure(label, xml_path, prompt_len=512, gen_len=32, chunk=128):
    """Fresh Core w/ only the .so; read serialized model; infer + measure."""
    core = ov.Core(); core.add_extension(SO)
    lm = core.read_model(xml_path)

    # Activation budget @ bound T_q=128 for comparison.
    bind = {}
    for p in lm.get_parameters():
        n = p.get_friendly_name(); ps = p.get_partial_shape()
        if n == "inputs_embeds":   bind[n] = ov.PartialShape([1, 128, ps[2]])
        elif n == "attention_mask": bind[n] = ov.PartialShape([1, 128])
        elif n == "position_ids":   bind[n] = ov.PartialShape([ps[0], 1, 128])
        elif n == "beam_idx":       bind[n] = ov.PartialShape([1])
    lm_anal = lm.clone()
    lm_anal.reshape(bind)
    rt = core.compile_model(lm_anal, "CPU",
                             {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True}).get_runtime_model()
    by_b = defaultdict(int); tot = 0
    for op in rt.get_ops():
        name = op.get_friendly_name()
        if any(k in name for k in WK): continue
        for i in range(op.get_output_size()):
            ps = op.get_output_partial_shape(i)
            et = op.get_output_element_type(i).get_type_name()
            b = shape_bytes(ps, et)
            if b < 1024: continue
            if   "linear_attn" in name: by_b["linear_attn"] += b
            elif "self_attn"   in name: by_b["self_attn"]   += b
            elif "mlp"         in name: by_b["mlp"]         += b
            else:                       by_b["other"]       += b
            tot += b

    # Inference
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    embed = core.compile_model(f"{MODEL}/openvino_text_embeddings_model.xml", "CPU")
    req = compiled.create_infer_request()
    logits = next(o for o in compiled.outputs if "logits" in o.get_any_name())
    ids = np.random.default_rng(0).integers(1, 200000, size=(1, prompt_len), dtype=np.int64)
    def embd(x): return list(embed.create_infer_request().infer({0: x}).values())[0]

    # warmup
    warm = compiled.create_infer_request()
    warm.infer({"inputs_embeds": embd(ids[:, :32]),
                "attention_mask": np.ones((1, 32), dtype=np.int64),
                "position_ids": np.tile(np.arange(32, dtype=np.int64).reshape(1, 1, 32), (4, 1, 1)),
                "beam_idx": np.zeros((1,), dtype=np.int32)})
    del warm

    t0 = time.time(); past = 0
    for i in range(0, prompt_len, chunk):
        L = min(chunk, prompt_len - i)
        req.infer({"inputs_embeds": embd(ids[:, i:i+L]),
                   "attention_mask": np.ones((1, past+L), dtype=np.int64),
                   "position_ids": np.tile(np.arange(past, past+L, dtype=np.int64).reshape(1, 1, L), (4, 1, 1)),
                   "beam_idx": np.zeros((1,), dtype=np.int32)})
        past += L
    t_pp = time.time() - t0

    nid = int(np.asarray(req.get_tensor(logits).data)[0, -1].argmax())
    t0 = time.time()
    for _ in range(gen_len):
        ne = list(embed.create_infer_request().infer({0: np.array([[nid]], dtype=np.int64)}).values())[0]
        req.infer({"inputs_embeds": ne,
                   "attention_mask": np.ones((1, past+1), dtype=np.int64),
                   "position_ids": np.full((4, 1, 1), past, dtype=np.int64),
                   "beam_idx": np.zeros((1,), dtype=np.int32)})
        nid = int(np.asarray(req.get_tensor(logits).data)[0, -1].argmax())
        past += 1
    t_tg = time.time() - t0

    print(f"\n=== {label} ===")
    print(f"  activation budget @ T_q=128: {tot/(1<<20):.1f} MiB total")
    for bk in ["linear_attn", "self_attn", "mlp", "other"]:
        print(f"    {bk:<14s}  {by_b[bk]/(1<<20):>8.1f} MiB")
    print(f"  pp{prompt_len}: {t_pp:.2f}s ({prompt_len/t_pp:.2f} tok/s)")
    print(f"  tg{gen_len}:  {t_tg:.2f}s ({gen_len/t_tg:.2f} tok/s)")
    return {"label": label, "act": tot/(1<<20),
            "by_b": {k: v/(1<<20) for k, v in by_b.items()},
            "pp_s": t_pp, "pp_tps": prompt_len/t_pp,
            "tg_s": t_tg, "tg_tps": gen_len/t_tg}


if __name__ == "__main__":
    build_and_serialize("v1", 1, "/tmp/lm_v1.xml")
    build_and_serialize("v2", 2, "/tmp/lm_v2.xml")
    build_and_serialize("v3", 3, "/tmp/lm_v3.xml")
    r1 = measure("A. v1 (C++ ext)", "/tmp/lm_v1.xml")
    r2 = measure("B. v2 (+ split/L2/scale/transpose)", "/tmp/lm_v2.xml")
    r3 = measure("C. v3 (+ conv1d/SiLU/Transposes)", "/tmp/lm_v3.xml")
    print(f"\n{'='*88}\nSUMMARY (pp512 + tg32, chunk=128, threads=4, all via serialize/reload)\n{'='*88}")
    print(f"{'metric':<24s} {'A v1':>15s} {'B v2':>15s} {'C v3':>15s}")
    for label, key, unit in [
        ("activation budget",  "act",     "MiB"),
        ("  linear_attn",      None,      "MiB"),
        ("pp512 throughput",   "pp_tps",  "tok/s"),
        ("tg32 throughput",    "tg_tps",  "tok/s"),
        ("pp512 duration",     "pp_s",    "s"),
        ("tg32 duration",      "tg_s",    "s"),
    ]:
        row = f"  {label:<22s}"
        for r in (r1, r2, r3):
            v = r["by_b"]["linear_attn"] if key is None else r[key]
            row += f" {v:>10.2f} {unit:<4s}"
        print(row)
