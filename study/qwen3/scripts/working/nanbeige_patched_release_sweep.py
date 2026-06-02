"""Sweep depths with the PATCHED release_memory (m_socketWeights.clear)
to measure the new post-release floor. Writes a CSV mergeable with
study/qwen3/data/nanbeige_with_reset.csv.
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
PP_LEN = 128
TG_LEN = 32
DEPTHS = [128, 256, 512, 768, 1024]
THREADS = 4
N_REPS = 2

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def feeds(T, past):
    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=166000, size=(1, T), dtype=np.int64)
    return {
        "input_ids":      ov.Tensor(ids),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.arange(past, past + T, dtype=np.int64).reshape(1, T)),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def main():
    print(f"OV version: {ov.__version__}")
    core = ov.Core()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    req = compiled.create_infer_request()
    req.infer(feeds(PP_LEN, past=0)); req.reset_state()
    print(f"warmup done. RSS={rss_kb()/1024:.1f} MiB")

    rows = []
    for d in DEPTHS:
        peak_max = 0
        for _ in range(N_REPS):
            req.reset_state()
            if d > 0:
                req.infer(feeds(d, past=0))
            req.infer(feeds(PP_LEN, past=d))
            past = d + PP_LEN
            for _ in range(TG_LEN):
                req.infer(feeds(1, past=past)); past += 1
            peak_max = max(peak_max, rss_kb())

        del req
        t0 = time.time()
        compiled.release_memory()
        libc.malloc_trim(0)
        t_release = (time.time() - t0) * 1000
        floor_patched = rss_kb()
        req = compiled.create_infer_request()
        print(f"  d={d:>4}  peak={peak_max/1024:7.1f}  "
              f"patched_floor={floor_patched/1024:7.1f}  release_cost={t_release:.0f}ms",
              flush=True)
        rows.append((d, peak_max / 1024, floor_patched / 1024, t_release))

    out_csv = "/home/user/openvino/study/qwen3/data/nanbeige_patched_release.csv"
    with open(out_csv, "w") as f:
        f.write("depth,ov_peak_rss,ov_patched_floor_rss,release_cost_ms\n")
        for d, peak, floor, t in rows:
            f.write(f"{d},{peak:.1f},{floor:.1f},{t:.0f}\n")
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
