"""Trace a Slice node in v3's IR: who is upstream (the source feeding it),
who is downstream (the consumers). Helps locate where to break the chain."""
import sys
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import openvino as ov
from fused_linear_attn import register, replace_gated_delta_rule_loops_v3
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")


def main():
    c = ov.Core(); register(c)
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))
    replace_gated_delta_rule_loops_v3(m)
    slice_lm_head_to_last_token(m)

    # Pre-compile (not runtime) graph
    target_names = [
        "__module.model.model.language_model.layers.0.linear_attn/aten::slice/Slice_4",
    ]
    for op in m.get_ops():
        if op.get_friendly_name() in target_names:
            print(f"\n=== {op.get_friendly_name()}  type={op.get_type_name()} ===")
            for i in range(op.get_input_size()):
                src = op.input(i).get_source_output()
                sn = src.get_node()
                print(f"  in[{i}]: {sn.get_type_name():<22s} {sn.get_friendly_name()} "
                      f" shape={src.get_partial_shape()} dtype={src.get_element_type()}")
            for i in range(op.get_output_size()):
                out = op.output(i)
                print(f"  out[{i}]: shape={out.get_partial_shape()} dtype={out.get_element_type()}")
                print(f"           consumers:")
                for t in out.get_target_inputs():
                    cn = t.get_node()
                    print(f"             {cn.get_type_name():<22s} {cn.get_friendly_name()}  in[{t.get_index()}]")

    # Compact upstream walk: limit fanout to skip weight chains.
    SKIP_TYPES = {"Constant", "Convert", "Subtract", "Multiply", "ShapeOf"}
    seen = set()
    def walk_up(out, depth, max_d=6):
        if depth > max_d: return
        n = out.get_node()
        if n.get_friendly_name() in seen: return
        seen.add(n.get_friendly_name())
        ps = out.get_partial_shape()
        print(f"  {'  '*depth}{n.get_type_name():<22s} {n.get_friendly_name()}  ps={ps}")
        if n.get_type_name() in SKIP_TYPES: return
        for i in range(n.get_input_size()):
            walk_up(n.input(i).get_source_output(), depth + 1, max_d)

    target = None
    for op in m.get_ops():
        if op.get_friendly_name() == target_names[0]:
            target = op
            break

    if target is not None:
        slice_in0 = target.input(0).get_source_output()
        concat = slice_in0.get_node()
        print(f"\n=== Concat feeding Slice_4: {concat.get_friendly_name()}  type={concat.get_type_name()}")
        print(f"    out ps={slice_in0.get_partial_shape()}")
        for i in range(concat.get_input_size()):
            inp = concat.input(i).get_source_output()
            sn = inp.get_node()
            print(f"  Concat in[{i}]: {sn.get_type_name():<22s} {sn.get_friendly_name()}  ps={inp.get_partial_shape()}")

        # walk back from each concat input
        for i in range(concat.get_input_size()):
            print(f"\n--- upstream from Concat in[{i}]:")
            seen.clear()
            walk_up(concat.input(i).get_source_output(), 0)

        # downstream walk: what does the Slice feed?
        def walk_down(out, depth, max_d=6):
            if depth > max_d: return
            for t in out.get_target_inputs():
                cn = t.get_node()
                ops_in_chain = cn.get_friendly_name()
                if ops_in_chain in seen: continue
                seen.add(ops_in_chain)
                print(f"  {'  '*depth}{cn.get_type_name():<22s} {cn.get_friendly_name()}  in[{t.get_index()}]  cn_out_ps={cn.get_output_partial_shape(0) if cn.get_output_size() else '-'}")
                if cn.get_type_name() in {"Constant"}: continue
                for j in range(cn.get_output_size()):
                    walk_down(cn.output(j), depth + 1, max_d)

        print("\n--- downstream from Slice_4 (where does the sliced data go?):")
        seen.clear()
        walk_down(target.output(0), 0)

    # Also find the constants used as start/stop/step/axis on these Slices
    print("\n=== One layer's slice constants ===")
    for op in m.get_ops():
        n = op.get_friendly_name()
        if "layers.0.linear_attn/aten::slice/Slice_4" not in n:
            continue
        if op.get_type_name() != "Slice":
            continue
        for i, label in zip(range(1, 5), ("start", "stop", "step", "axis")):
            src = op.input(i).get_source_output().get_node()
            print(f"  {label}: {src.get_type_name()} {src.get_friendly_name()}")
            try:
                arr = src.get_data().tolist() if hasattr(src, "get_data") else None
                print(f"         value={arr}")
            except Exception as e:
                print(f"         (no const data: {e})")


if __name__ == "__main__":
    main()
