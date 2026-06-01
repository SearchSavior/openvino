"""Print every node in linear_attn.0 of the BASELINE LM IR (no rewrites)
in topological order, with op_type, shape, and key inputs. Used to find
fusion targets that would collapse [1, T, 6144] f32 intermediates."""
import sys
from pathlib import Path
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WK = ("embed_tokens", "lm_head", ".weight", ".bias")


def main():
    c = ov.Core()
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))

    # Collect every node whose friendly_name starts with the layer 0 linear_attn path.
    layer_prefix = "__module.model.model.language_model.layers.0.linear_attn"
    nodes = []
    for op in m.get_ops():
        n = op.get_friendly_name()
        if not n.startswith(layer_prefix): continue
        if any(k in n for k in WK): continue
        if op.get_type_name() in ("Constant", "Convert", "Subtract", "Multiply") \
                and "fq_weights" in n: continue
        if op.get_type_name() == "Constant" and op.get_output_partial_shape(0).rank.get_length() <= 1:
            continue
        nodes.append(op)

    # Order by toposort. OV's get_ordered_ops handles this.
    ordered = [op for op in m.get_ordered_ops() if op in set(nodes)]

    print(f"{len(ordered)} nodes in linear_attn.0:\n")
    for op in ordered:
        n = op.get_friendly_name()
        tn = op.get_type_name()
        outs = [str(op.get_output_partial_shape(i)) for i in range(op.get_output_size())]
        ins = []
        for i in range(op.get_input_size()):
            src = op.input(i).get_source_output()
            sn = src.get_node()
            ins.append(f"{sn.get_type_name()}({sn.get_friendly_name().split('/')[-1] if '/' in sn.get_friendly_name() else sn.get_friendly_name()})")
        # short name
        sn = n.replace(layer_prefix + "/", "").replace(layer_prefix, "<self>")
        print(f"  {tn:<22s}  out={','.join(outs):<30s}  {sn[:60]}")
        if len(ins) <= 6:
            for j, inp in enumerate(ins):
                print(f"      in[{j}]={inp[:90]}")


if __name__ == "__main__":
    main()
