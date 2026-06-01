"""Identify which Slice nodes in v3's linear_attn bucket eat ~7 ms at T_q=1.
Reports per-node Slice exec time, friendly name, and i/o shapes."""
import sys, time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import numpy as np
import openvino as ov
from fused_linear_attn import register, replace_gated_delta_rule_loops_v3
from lm_head_slice import slice_lm_head_to_last_token

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = str(Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so")


def build_v3(out):
    c = ov.Core(); register(c)
    m = c.read_model(str(ORIG / "openvino_language_model.xml"))
    replace_gated_delta_rule_loops_v3(m)
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace(".xml", ".bin"))


def main():
    xml = "/tmp/lm_v3_slices.xml"
    print("[build]"); build_v3(xml)
    core = ov.Core(); core.add_extension(SO)
    lm = core.read_model(xml)
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4, "PERF_COUNT": True})

    # T_q=1 decode-shape probe
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((1, 1, hidden), dtype=np.float32) * 0.01
    feeds = {
        "inputs_embeds":  ov.Tensor(embeds),
        "attention_mask": ov.Tensor(np.ones((1, 1), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.zeros((pid_b, 1, 1), dtype=np.int64)),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds); req.infer(feeds); req.infer(feeds)

    prof = req.get_profiling_info()
    rt = compiled.get_runtime_model()
    rt_by_name = {n.get_friendly_name(): n for n in rt.get_ops()}

    slices = []
    for p in prof:
        if p.status != ov.ProfilingInfo.Status.EXECUTED:
            continue
        if p.node_type != "Slice":
            continue
        us = float(p.real_time.total_seconds() * 1e6)
        n = rt_by_name.get(p.node_name)
        shapes = []
        if n is not None:
            for i in range(n.get_input_size()):
                shapes.append(f"in{i}={n.get_input_partial_shape(i)}")
            for i in range(n.get_output_size()):
                shapes.append(f"out{i}={n.get_output_partial_shape(i)}")
        slices.append((us, p.node_name, p.exec_type, " ".join(shapes)))

    slices.sort(reverse=True)
    print(f"\n{'us':>10s}  {'exec_type':<22s}  name  /  shapes")
    tot = 0
    for us, name, ex, sh in slices:
        tot += us
        print(f"{us:>10.1f}  {ex:<22s}  {name}    {sh}")
    print(f"\nTotal Slice time: {tot/1000:.2f} ms across {len(slices)} executed Slice nodes")

    # Group by exec_type + output shape for a roll-up
    buckets = defaultdict(lambda: [0, 0])
    for us, name, ex, sh in slices:
        # extract shorthand: the most relevant attribute is exec_type and a marker
        out_shape = sh.split("out0=")[-1].split()[0] if "out0=" in sh else "?"
        k = f"{ex}  out0={out_shape}"
        buckets[k][0] += 1; buckets[k][1] += us
    print("\nGrouped by (exec_type, out_shape):")
    for k, (c, u) in sorted(buckets.items(), key=lambda x: -x[1][1]):
        print(f"  {u/1000:>8.2f} ms  x{c:>3d}   {k}")


if __name__ == "__main__":
    main()
