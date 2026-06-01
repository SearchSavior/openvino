"""Confirm whether the model .bin is mmap'd into our process during raw
ov.Core() inference, and whether toggling ENABLE_MMAP changes that.
Reads /proc/self/maps after compile and after warmup, greps for the bin
file path, and reports VmHWM."""
import os, sys, threading, time, subprocess
from pathlib import Path
import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
BIN_NAME = "openvino_language_model.bin"


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): return int(line.split()[1])
    return 0


def vmhwm_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"): return int(line.split()[1])
    return 0


def find_mappings_for(path_substring):
    """Yield (range, perms, offset, path) for every /proc/self/maps line
    whose backing path contains path_substring."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.rstrip().split(None, 5)
            if len(parts) < 6: continue
            mapped_path = parts[5]
            if path_substring in mapped_path:
                rng, perms, offset = parts[0], parts[1], parts[2]
                lo, hi = (int(x, 16) for x in rng.split("-"))
                out.append({"size_mib": (hi - lo) / (1 << 20),
                            "perms": perms, "offset": offset, "path": mapped_path})
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmap", choices=["default", "true", "false"], default="default")
    args = ap.parse_args()

    rss_pre = rss_kb()
    # Prep in a subprocess so the parent starts clean
    code = f"""
import sys; sys.path.insert(0, '{Path(__file__).resolve().parents[2] / "kernels"}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/probe_mmap.xml', '{WORK}/probe_mmap.bin')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0: sys.exit(r.stderr)

    props = {"INFERENCE_NUM_THREADS": 4}
    if args.mmap == "true":  props["ENABLE_MMAP"] = True
    if args.mmap == "false": props["ENABLE_MMAP"] = False
    print(f"\n=== probe_mmap_check  mmap={args.mmap}  props={props} ===")

    core = ov.Core()
    lm = core.read_model(f"{WORK}/probe_mmap.xml")
    compiled = core.compile_model(lm, "CPU", props)
    rss_post_compile = rss_kb()
    print(f"  RSS after compile: {rss_post_compile/1024:7.1f} MiB")

    maps_post_compile = find_mappings_for("probe_mmap.bin")
    print(f"  mmap'd ranges of probe_mmap.bin after compile: {len(maps_post_compile)}")
    for m in maps_post_compile[:6]:
        print(f"    {m['size_mib']:7.1f} MiB  perms={m['perms']}  offset={m['offset']}")

    # one prefill at T=128 to exercise the graph
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, 128, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, 128), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(128, dtype=np.int64).reshape(1, 1, 128), (pid_b, 1, 1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds)

    rss_post_warm = rss_kb()
    maps_post_warm = find_mappings_for("probe_mmap.bin")
    print(f"  RSS after warmup:  {rss_post_warm/1024:7.1f} MiB")
    print(f"  mmap'd ranges of probe_mmap.bin after warmup: {len(maps_post_warm)}")
    for m in maps_post_warm[:6]:
        print(f"    {m['size_mib']:7.1f} MiB  perms={m['perms']}  offset={m['offset']}")

    print(f"  VmHWM (peak): {vmhwm_kb()/1024:7.1f} MiB")


if __name__ == "__main__":
    main()
