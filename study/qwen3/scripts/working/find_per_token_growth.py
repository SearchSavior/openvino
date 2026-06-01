"""Compare runtime-model output bytes at T_q=32 vs T_q=770 to find which
nodes actually scale per-token. Sort by Δbytes/(770-32) descending."""
import subprocess, sys
from pathlib import Path
from collections import defaultdict

import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
ETYPE = {"f32":4,"f16":2,"bf16":2,"i64":8,"i32":4,"i8":1,"u8":1,"boolean":1}
WK = ("embed_tokens","lm_head","_proj/",".weight",".bias","rotary","inv_freq",
      "A_log","ssm_dt","ssm_a","ssm_beta","ssm_alpha","ssm_norm","ssm_conv1d",
      "norm.weight","input_layernorm","k_norm","q_norm","attn_norm")


def _prep():
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    out = f"{WORK}/raw_pertoken.xml"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{out}', '{out.replace('.xml','.bin')}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    return out


def shape_bytes(ps, et):
    if not all(d.is_static for d in ps): return 0
    n = 1
    for d in ps: n *= d.get_length()
    return n * ETYPE.get(et, 4)


def bucket(name, tn=""):
    if "linear_attn" in name: return "linear_attn"
    if "self_attn"   in name: return "self_attn"
    if "mlp"         in name: return "mlp"
    return "other"


def runtime_bytes(xml, T_q):
    core = ov.Core()
    lm = core.read_model(xml)
    bind = {}
    for p in lm.get_parameters():
        n = p.get_friendly_name(); ps = p.get_partial_shape()
        if   n == "inputs_embeds":   bind[n] = ov.PartialShape([1, T_q, ps[2]])
        elif n == "attention_mask":  bind[n] = ov.PartialShape([1, T_q])
        elif n == "position_ids":    bind[n] = ov.PartialShape([ps[0], 1, T_q])
        elif n == "beam_idx":        bind[n] = ov.PartialShape([1])
    lm.reshape(bind)
    anal = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})
    rt = anal.get_runtime_model()
    per_node = {}
    for op in rt.get_ops():
        n = op.get_friendly_name()
        if any(k in n for k in WK): continue
        tn = op.get_type_name()
        b = 0
        for i in range(op.get_output_size()):
            ps = op.get_output_partial_shape(i)
            et = op.get_output_element_type(i).get_type_name()
            b += shape_bytes(ps, et)
        if b > 0:
            shapes = [str(op.get_output_partial_shape(i)) for i in range(op.get_output_size())]
            per_node[n] = (b, tn, " ".join(shapes), bucket(n, tn))
    return per_node


def main():
    xml = _prep()
    print("[T=32] walking…")
    r32 = runtime_bytes(xml, 32)
    print("[T=770] walking…")
    r770 = runtime_bytes(xml, 770)

    # Compute per-token growth
    growth = []
    for name, (b770, tn, sh, bk) in r770.items():
        b32 = r32.get(name, (0, tn, sh, bk))[0]
        per_tok = (b770 - b32) / (770 - 32)  # bytes/token
        if per_tok > 0:
            growth.append((per_tok, b32, b770, name, tn, sh, bk))
    growth.sort(reverse=True)

    print(f"\n=== top 25 nodes by Δbytes/token ===")
    print(f"{'B/tok':>10s}  {'T=32 MiB':>10s}  {'T=770 MiB':>10s}  {'op_type':<18s}  bucket  name")
    for per_tok, b32, b770, name, tn, sh, bk in growth[:25]:
        print(f"  {per_tok/1024:>7.2f} KiB  {b32/(1<<20):>8.2f}   {b770/(1<<20):>8.2f}   "
              f"{tn:<18s}  {bk:<11s}  {name[:60]}")

    # roll-up by bucket and shape pattern
    rollup = defaultdict(lambda: [0, 0, 0])  # bucket -> [per_tok_total, b32_total, b770_total]
    for per_tok, b32, b770, name, tn, sh, bk in growth:
        rollup[bk][0] += per_tok
        rollup[bk][1] += b32
        rollup[bk][2] += b770
    print(f"\n=== per-bucket roll-up ===")
    print(f"{'bucket':<14s}  {'Σ B/tok':>14s}  {'T=32 total':>14s}  {'T=770 total':>14s}")
    for bk, (pt, t32, t770) in sorted(rollup.items(), key=lambda x: -x[1][0]):
        print(f"  {bk:<12s}  {pt/(1<<20):>10.2f} MiB  {t32/(1<<20):>10.1f} MiB  {t770/(1<<20):>10.1f} MiB")
    total_pt = sum(g[0] for g in growth)
    print(f"\nTotal Δ per token across ALL nodes (paper budget): {total_pt/(1<<20):.2f} MiB/token")
    print(f"Actual measured RSS per token (from earlier sweep):  ~0.73 MiB/token")


if __name__ == "__main__":
    main()
