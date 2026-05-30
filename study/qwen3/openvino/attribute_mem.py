"""
Memory attribution from the post-compile runtime graph + oneDNN verbose.

For a compiled model:
  1. Walk get_runtime_model() and for each node, compute its output bytes at
     a chosen runtime shape (uses the dynamic-dim substitution dict).
  2. Bucket by:
        - op type (Convolution, MatMul, SDPA, Concat, ReadValue, ...)
        - "growth class": activation (scales with T_q), persistent state (scales
          with T_full), weights/Constant (fixed), other.
  3. Tally Constant tensor bytes by element type -> total weight footprint.
  4. Print top-N nodes by output bytes.

Pair with ONEDNN_VERBOSE=all to capture the actual oneDNN primitive
allocations during inference (writes to stderr, parse out memory descriptors).

Usage:
    python attribute_mem.py --config baseline --T_q 128 --T_full 2048
    python attribute_mem.py --config int8_sdpa --T_q 128 --T_full 2048 --dump-runtime /tmp/exec.xml
    ONEDNN_VERBOSE=all python attribute_mem.py --config baseline --T_q 128 --T_full 2048 --infer 2> /tmp/onednn.log
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/home/user/openvino/study/qwen3/kernels")
import numpy as np
import openvino as ov

from fused_linear_attn  import register as rla, replace_gated_delta_rule_loops
from lm_head_slice      import slice_lm_head_to_last_token
from quantized_kv       import register as rqkv, replace_kv_with_int8
from quantized_int8_sdpa import register as rqi,  replace_kv_with_int8_sdpa

MODEL = "/tmp/qwen3-work/qwen35-0.8b-int8"
SO    = "/home/user/openvino/study/qwen3/cpp_ext/build/libqwen3_ov_ext.so"

ETYPE_BYTES = {
    "f32": 4, "f16": 2, "bf16": 2,
    "i64": 8, "i32": 4, "i16": 2, "i8": 1,
    "u64": 8, "u32": 4, "u16": 2, "u8": 1,
    "boolean": 1, "string": 0, "dynamic": 0,
}


def dim_bytes(ps, T_q, T_full, B=1):
    """Estimate the total bytes implied by a partial shape, substituting:
       -1 / dynamic dims with T_q if 'T_q' label is the canonical one, else T_full.
       Heuristic: if a node has TWO dynamic dims, the larger one is T_full (state) and the smaller is T_q.
    """
    dims = []
    dyn_count = 0
    for d in ps:
        if d.is_static:
            dims.append(d.get_length())
        else:
            dyn_count += 1
            dims.append(None)
    if dyn_count == 0:
        prod = 1
        for d in dims: prod *= d
        return prod
    # Fill dynamic dims. First dim is B (1). If exactly one dynamic remaining,
    # assume it's T_q for short tensors, T_full for longer state tensors.
    out = []
    other_static_max = max((d for d in dims if d is not None), default=1)
    dyn_filled = 0
    for i, d in enumerate(dims):
        if d is not None:
            out.append(d); continue
        if i == 0:
            out.append(B)
        elif dyn_count == 1:
            # one dyn dim; depends on context. If the static max is large (>=1024) likely activations -> T_q.
            # If static max is small (<256) and we're in KV path -> T_full.
            out.append(T_q if other_static_max >= 1024 else T_full)
        else:
            # Multi-dyn: heuristic. Use T_full for first dyn after B, T_q for rest.
            out.append(T_full if dyn_filled == 0 else T_q)
        dyn_filled += 1
    prod = 1
    for d in out: prod *= d
    return prod


def build_model(config):
    core = ov.Core()
    core.add_extension(SO)
    rla(core); rqkv(core); rqi(core)
    lm = core.read_model(f"{MODEL}/openvino_language_model.xml")
    replace_gated_delta_rule_loops(lm)
    slice_lm_head_to_last_token(lm)
    if config == "int8_kv_dequant":
        replace_kv_with_int8(lm)
    elif config == "int8_sdpa":
        replace_kv_with_int8_sdpa(lm)
    elif config != "baseline":
        sys.exit(f"unknown config {config}")
    return lm, core


def build_compiled(config, reshape_to=None):
    lm, core = build_model(config)
    if reshape_to is not None:
        B, T_q = reshape_to
        bind = {}
        for p in lm.get_parameters():
            n = p.get_friendly_name()
            ps = p.get_partial_shape()
            if n == "inputs_embeds":
                bind[n] = ov.PartialShape([B, T_q, ps[2]])
            elif n == "attention_mask":
                bind[n] = ov.PartialShape([B, T_q])
            elif n == "position_ids":
                bind[n] = ov.PartialShape([ps[0], B, T_q])
            elif n == "beam_idx":
                bind[n] = ov.PartialShape([B])
        lm.reshape(bind)
    cfg = {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True}
    return core.compile_model(lm, "CPU", cfg), core


def ir_weight_breakdown(model):
    """Walk Constant nodes in the SOURCE IR and bucket by dtype + role."""
    by_dtype = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_role = defaultdict(lambda: {"count": 0, "bytes": 0})
    for op in model.get_ops():
        if op.get_type_name() != "Constant":
            continue
        ps = op.get_output_partial_shape(0)
        if not ps.is_static:
            continue
        elems = 1
        for d in ps:
            elems *= d.get_length()
        et = op.get_output_element_type(0).get_type_name()
        b = elems * ETYPE_BYTES.get(et, 4)
        by_dtype[et]["count"] += 1
        by_dtype[et]["bytes"] += b

        # Guess role by friendly name + shape.
        name = op.get_friendly_name().lower()
        if "embed_tokens" in name or "lm_head" in name:
            role = "embed/lm_head"
        elif "linear_attn" in name and ("in_proj_qkv" in name or "qkv" in name):
            role = "linear_attn.in_proj_qkv"
        elif "linear_attn" in name and ("in_proj_z" in name or "in_proj_a" in name or "in_proj_b" in name):
            role = "linear_attn.in_proj_{z,a,b}"
        elif "linear_attn" in name and "out_proj" in name:
            role = "linear_attn.out_proj"
        elif "linear_attn" in name and "conv1d" in name:
            role = "linear_attn.conv1d"
        elif "self_attn" in name and "_proj" in name:
            role = "self_attn.qkv_out_proj"
        elif "mlp" in name and ("gate_proj" in name or "up_proj" in name or "down_proj" in name):
            role = "mlp"
        elif "norm" in name:
            role = "norm"
        elif "rotary" in name or "inv_freq" in name:
            role = "rotary"
        elif b < 256:
            role = "small_const"
        else:
            role = f"other_{ps.to_shape() if ps.rank.get_length() else '0d'}"
        by_role[role]["count"] += 1
        by_role[role]["bytes"] += b
    return by_dtype, by_role


def classify_growth(node, ps):
    """Tag this node's output by whether its byte size scales with T_q, T_full, or is fixed."""
    if node.get_type_name() == "Constant":
        return "weight"
    dyn = sum(1 for d in ps if not d.is_static)
    static_max = max((d.get_length() for d in ps if d.is_static), default=1)
    name = node.get_friendly_name().lower()
    if any(k in name for k in ("cache", "state", "readvalue", "assign", "concat")):
        return "state"
    if dyn == 0:
        return "fixed"
    return "activation"


def analyse(rt_model, T_q, T_full, B=1):
    by_type = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_growth = defaultdict(lambda: {"count": 0, "bytes": 0})
    by_dtype_fixed = defaultdict(lambda: {"count": 0, "bytes": 0})
    big_nodes = []

    weights_by_dtype = defaultdict(lambda: {"count": 0, "bytes": 0})

    for op in rt_model.get_ops():
        tn = op.get_type_name()
        # Pull plugin runtime info: primitiveType, execType, originalLayersNames.
        rt = op.get_rt_info()
        prim = ""
        for k in ("primitiveType", "execType", "originalLayersNames"):
            if k in rt:
                prim = str(rt[k]); break
        for i in range(op.get_output_size()):
            ps = op.get_output_partial_shape(i)
            et = op.get_output_element_type(i).get_type_name()
            elems = dim_bytes(ps, T_q, T_full, B)
            bytes_ = elems * ETYPE_BYTES.get(et, 4)

            by_type[tn]["count"] += 1
            by_type[tn]["bytes"] += bytes_

            growth = classify_growth(op, ps)
            by_growth[growth]["count"] += 1
            by_growth[growth]["bytes"] += bytes_
            if growth == "fixed":
                by_dtype_fixed[et]["count"] += 1
                by_dtype_fixed[et]["bytes"] += bytes_

            if tn == "Constant":
                weights_by_dtype[et]["count"] += 1
                weights_by_dtype[et]["bytes"] += bytes_

            big_nodes.append((bytes_, tn, op.get_friendly_name()[:70], str(ps), et, prim[:40]))

    big_nodes.sort(reverse=True)
    return by_type, by_growth, weights_by_dtype, by_dtype_fixed, big_nodes


def fmt(b):
    for unit, scale in (("GiB", 1 << 30), ("MiB", 1 << 20), ("KiB", 1 << 10)):
        if b >= scale:
            return f"{b/scale:.2f} {unit}"
    return f"{int(b)} B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=["baseline", "int8_kv_dequant", "int8_sdpa"], default="baseline")
    ap.add_argument("--T_q", type=int, default=128)
    ap.add_argument("--T_full", type=int, default=2048)
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--dump-runtime", default=None, help="serialize runtime model XML to this path")
    ap.add_argument("--infer", action="store_true",
                    help="also run a single chunk infer so ONEDNN_VERBOSE prints get triggered")
    args = ap.parse_args()

    print(f"=== config={args.config} T_q={args.T_q} T_full={args.T_full} ===")

    # Source IR weight analysis (Constants are unambiguous here).
    src_model, _ = build_model(args.config)
    by_dt, by_role = ir_weight_breakdown(src_model)
    print(f"\nSOURCE IR weights (Constant nodes):")
    wtotal = sum(v["bytes"] for v in by_dt.values())
    for et, v in sorted(by_dt.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  Constant({et:<6s}) {v['count']:>5d}  {fmt(v['bytes']):>10s} "
              f"({100*v['bytes']/wtotal:>5.1f}%)")
    print(f"  weights total in IR: {fmt(wtotal)}")
    print(f"\nweights by role:")
    for role, v in sorted(by_role.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {role:<35s} {v['count']:>5d}  {fmt(v['bytes']):>10s} "
              f"({100*v['bytes']/wtotal:>5.1f}%)")

    compiled, core = build_compiled(args.config, reshape_to=(1, args.T_q))
    rt = compiled.get_runtime_model()
    if args.dump_runtime:
        ov.serialize(rt, args.dump_runtime)
        print(f"runtime model -> {args.dump_runtime}")

    by_type, by_growth, weights, by_dtype_fixed, big = analyse(rt, args.T_q, args.T_full)
    total = sum(v["bytes"] for v in by_type.values())

    print(f"\nRuntime graph: {sum(v['count'] for v in by_type.values())} output tensors, "
          f"total addressable bytes (if all live simultaneously): {fmt(total)}\n")

    print(f"growth class:")
    for cls, v in sorted(by_growth.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {cls:<10s} {v['count']:>6d} tensors  {fmt(v['bytes']):>12s} {100*v['bytes']/total:>7.1f}%")

    print(f"\n'fixed' tensors by dtype (post-compile, includes weight reorders):")
    ft = sum(v["bytes"] for v in by_dtype_fixed.values())
    for et, v in sorted(by_dtype_fixed.items(), key=lambda kv: -kv[1]["bytes"]):
        print(f"  {et:<6s} {v['count']:>6d} tensors  {fmt(v['bytes']):>12s} "
              f"({100*v['bytes']/ft:>5.1f}% of fixed)")

    print(f"\ntop {args.top} nodes by single-output byte size:")
    for sz, tn, name, ps, et, prim in big[:args.top]:
        print(f"  {fmt(sz):>10s}  {et:<6s} {ps:<28s} prim={prim:<26s} {name}")

    if args.infer:
        # Run a single chunk so ONEDNN_VERBOSE has something to print.
        import time, numpy as np
        embed = core.compile_model(f"{MODEL}/openvino_text_embeddings_model.xml", "CPU")
        req = compiled.create_infer_request()
        ids = np.random.default_rng(0).integers(1, 200000, size=(1, args.T_q), dtype=np.int64)
        ne = list(embed.create_infer_request().infer({0: ids}).values())[0]
        req.infer({"inputs_embeds": ne,
                   "attention_mask": np.ones((1, args.T_q), dtype=np.int64),
                   "position_ids": np.tile(np.arange(args.T_q, dtype=np.int64).reshape(1, 1, -1), (4, 1, 1)),
                   "beam_idx": np.zeros((1,), dtype=np.int32)})
        prof = req.get_profiling_info()
        prof = sorted(prof, key=lambda p: p.real_time, reverse=True)
        print(f"\ntop 15 nodes by real_time:")
        print(f"  {'real_us':>10s} {'cpu_us':>10s}  {'type':<22s} {'exec_type':<22s} name")
        for p in prof[:15]:
            print(f"  {p.real_time.total_seconds()*1e6:>10.0f} "
                  f"{p.cpu_time.total_seconds()*1e6:>10.0f}  "
                  f"{p.node_type:<22s} {p.exec_type[:22]:<22s} {p.node_name[:60]}")


if __name__ == "__main__":
    main()
