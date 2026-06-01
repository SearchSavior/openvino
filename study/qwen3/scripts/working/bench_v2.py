"""Build v1 + v2 fused IRs via the Python Op subclasses, serialize each,
reload with a fresh Core+.so so the C++ implementation wins evaluate(),
and run a chunked prefill + decode loop.

Both versions are serialized then re-loaded by a fresh ov.Core() that has
only the .so registered, severing the Python-class binding so the C++
implementation handles evaluate.

Performance numbers and analysis live in DISCUSSION.md, not here.
"""
import sys, time, os, subprocess, json
import openvino as ov
import numpy as np
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
from fused_linear_attn import register as rla, replace_gated_delta_rule_loops, replace_gated_delta_rule_loops_v2
from lm_head_slice import slice_lm_head_to_last_token

MODEL = "/tmp/qwen3-work/qwen35-0.8b-int8"
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


def build_and_serialize(label, apply_v1, apply_v2, xml_path):
    """Step 1 of the C++-takes-over pattern: build IR with Python classes, serialize."""
    core_build = ov.Core(); rla(core_build)
    lm = core_build.read_model(f"{MODEL}/openvino_language_model.xml")
    if apply_v1: replace_gated_delta_rule_loops(lm)
    if apply_v2: replace_gated_delta_rule_loops_v2(lm)
    slice_lm_head_to_last_token(lm)
    ov.serialize(lm, xml_path, xml_path.replace(".xml", ".bin"))


def measure(label, xml_path, prompt_len=512, gen_len=32, chunk=128):
    """Step 2: fresh Core w/ only .so, read serialized model, infer + measure."""
    core = ov.Core(); core.add_extension(SO)
    lm = core.read_model(xml_path)

    # also bind for activation analysis
    bind = {}
    for p in lm.get_parameters():
        n=p.get_friendly_name(); ps=p.get_partial_shape()
        if n=="inputs_embeds":   bind[n]=ov.PartialShape([1,128,ps[2]])
        elif n=="attention_mask":bind[n]=ov.PartialShape([1,128])
        elif n=="position_ids":  bind[n]=ov.PartialShape([ps[0],1,128])
        elif n=="beam_idx":      bind[n]=ov.PartialShape([1])
    lm_anal = lm.clone()
    lm_anal.reshape(bind)
    rt = core.compile_model(lm_anal, "CPU", {"INFERENCE_NUM_THREADS":4, "PERF_COUNT":True}).get_runtime_model()
    by_b = defaultdict(int); tot=0
    for op in rt.get_ops():
        name=op.get_friendly_name()
        if any(k in name for k in WK): continue
        for i in range(op.get_output_size()):
            ps=op.get_output_partial_shape(i); et=op.get_output_element_type(i).get_type_name()
            b=shape_bytes(ps,et)
            if b<1024: continue
            if "linear_attn" in name: by_b["linear_attn"]+=b
            elif "self_attn" in name: by_b["self_attn"]+=b
            elif "mlp" in name: by_b["mlp"]+=b
            else: by_b["other"]+=b
            tot+=b

    # Inference
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS":4})
    embed = core.compile_model(f"{MODEL}/openvino_text_embeddings_model.xml", "CPU")
    req = compiled.create_infer_request()
    logits = next(o for o in compiled.outputs if "logits" in o.get_any_name())
    ids = np.random.default_rng(0).integers(1, 200000, size=(1, prompt_len), dtype=np.int64)
    def embd(x): return list(embed.create_infer_request().infer({0:x}).values())[0]

    # warmup
    warm = compiled.create_infer_request()
    warm.infer({"inputs_embeds": embd(ids[:, :32]),
                "attention_mask": np.ones((1,32), dtype=np.int64),
                "position_ids": np.tile(np.arange(32, dtype=np.int64).reshape(1,1,32), (4,1,1)),
                "beam_idx": np.zeros((1,), dtype=np.int32)})
    del warm

    t0 = time.time(); past = 0
    for i in range(0, prompt_len, chunk):
        L = min(chunk, prompt_len-i)
        req.infer({"inputs_embeds": embd(ids[:, i:i+L]),
                   "attention_mask": np.ones((1, past+L), dtype=np.int64),
                   "position_ids": np.tile(np.arange(past, past+L, dtype=np.int64).reshape(1,1,L), (4,1,1)),
                   "beam_idx": np.zeros((1,), dtype=np.int32)})
        past += L
    t_pp = time.time() - t0

    nid = int(np.asarray(req.get_tensor(logits).data)[0,-1].argmax())
    t0 = time.time()
    for _ in range(gen_len):
        ne = list(embed.create_infer_request().infer({0: np.array([[nid]], dtype=np.int64)}).values())[0]
        req.infer({"inputs_embeds": ne, "attention_mask": np.ones((1, past+1), dtype=np.int64),
                   "position_ids": np.full((4,1,1), past, dtype=np.int64),
                   "beam_idx": np.zeros((1,), dtype=np.int32)})
        nid = int(np.asarray(req.get_tensor(logits).data)[0,-1].argmax())
        past += 1
    t_tg = time.time() - t0

    print(f"\n=== {label} ===")
    print(f"  activation budget @ T_q=128: {tot/(1<<20):.1f} MiB total")
    for bk in ["linear_attn","self_attn","mlp","other"]:
        print(f"    {bk:<14s}  {by_b[bk]/(1<<20):>8.1f} MiB")
    print(f"  pp{prompt_len}: {t_pp:.2f}s ({prompt_len/t_pp:.2f} tok/s)")
    print(f"  tg{gen_len}:  {t_tg:.2f}s ({gen_len/t_tg:.2f} tok/s)")
    return {"label":label, "act":tot/(1<<20), "by_b":{k:v/(1<<20) for k,v in by_b.items()},
            "pp_s":t_pp, "pp_tps":prompt_len/t_pp, "tg_s":t_tg, "tg_tps":gen_len/t_tg}


if __name__ == "__main__":
    build_and_serialize("v1 baseline", True, False, "/tmp/lm_v1.xml")
    build_and_serialize("v2 fused prep", False, True, "/tmp/lm_v2.xml")
    r1 = measure("A. v1 (C++ ext)", "/tmp/lm_v1.xml")
    r2 = measure("B. v2 (C++ ext, mixed_qkv input)", "/tmp/lm_v2.xml")
    print(f"\n{'='*70}\nSUMMARY (both via serialize/reload -> C++ ext)\n{'='*70}")
    print(f"{'metric':<24s} {'A v1':>15s} {'B v2':>15s} {'delta':>15s}")
    for label, key, unit in [
        ("activation budget", "act",    "MiB"),
        ("  linear_attn",     None,     "MiB"),
        ("pp512 throughput",  "pp_tps", "tok/s"),
        ("tg32 throughput",   "tg_tps", "tok/s"),
        ("pp512 duration",    "pp_s",   "s"),
        ("tg32 duration",     "tg_s",   "s"),
    ]:
        if key is None:
            a, b = r1["by_b"]["linear_attn"], r2["by_b"]["linear_attn"]
        else:
            a, b = r1[key], r2[key]
        print(f"  {label:<22s} {a:>10.2f} {unit:<4s} {b:>10.2f} {unit:<4s} {b-a:+10.2f} {unit:<4s}")
