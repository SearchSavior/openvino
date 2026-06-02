"""Show that L1 + compiled_model.release_memory() + malloc_trim brings RSS
to llama.cpp parity at T=1024. Findings live in DISCUSSION.md.

WARNING: release_memory currently CRASHES on subsequent infer when the
graph contains TensorIterator/Loop nodes (e.g. gated-delta-net linear
attention). This means release_memory works for a final 'shut down'
sequence but not for continuous serving until that bug is fixed.
"""
import ctypes, subprocess, sys
from pathlib import Path
import openvino as ov
import numpy as np

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")

libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def prep():
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/probe_rel.xml', '{WORK}/probe_rel.bin')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=1024)
    args = ap.parse_args()
    T = args.T

    prep()
    print(f"pre-load:                {rss():7.1f} MiB")
    core = ov.Core()
    lm = core.read_model(f"{WORK}/probe_rel.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    print(f"after compile:           {rss():7.1f} MiB")

    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (pid_b, 1, 1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds)
    print(f"after warmup (T={T}):    {rss():7.1f} MiB   <- peak (also VmHWM)")

    # Drop request handle so model isn't 'busy'
    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    print(f"after release+trim:      {rss():7.1f} MiB   <- post-session steady state")

    # Try re-infer — will crash on models with Loop nodes due to upstream bug
    try:
        req = compiled.create_infer_request()
        req.infer(feeds)
        print(f"after re-infer:          {rss():7.1f} MiB")
    except RuntimeError as e:
        msg = str(e).split("\n")[0][:120]
        print(f"after re-infer: CRASHED — {msg}")


if __name__ == "__main__":
    main()
