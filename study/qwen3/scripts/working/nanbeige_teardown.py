"""Try the nuclear 'tear down compiled_model entirely between conversations'
approach for memory reset. Costs the recompile time (~2s for Nanbeige) but
should reach a much lower floor than release_memory + reset_state can.

Tests:
  1. Build context to d=1024 + pp + tg, measure peak
  2. release_memory + reset_state (the existing API combo we tested)  → floor1
  3. del request, del compiled, malloc_trim                            → floor2
  4. Re-create compiled and a fresh request                            → cost?
  5. Run a fresh request to confirm it works correctly                 → working?
"""
import ctypes
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
DEPTH = 1024
PP_LEN = 128
TG_LEN = 32
THREADS = 4

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def feeds(lm, T, past):
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
    print(f"pre-everything RSS: {rss_kb()/1024:.1f} MiB")

    core = ov.Core()
    t0 = time.time()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    print(f"read_model:      {time.time()-t0:5.2f}s   RSS={rss_kb()/1024:.1f} MiB")
    t0 = time.time()
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    print(f"compile_model:   {time.time()-t0:5.2f}s   RSS={rss_kb()/1024:.1f} MiB")
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()
    print(f"warmup:                  RSS={rss_kb()/1024:.1f} MiB")

    # Do work at depth
    if DEPTH > 0:
        req.infer(feeds(lm, DEPTH, past=0))
    req.infer(feeds(lm, PP_LEN, past=DEPTH))
    past = DEPTH + PP_LEN
    for _ in range(TG_LEN):
        req.infer(feeds(lm, 1, past=past)); past += 1
    peak = rss_kb()
    print(f"\nafter work at d={DEPTH}:    RSS={peak/1024:7.1f} MiB")

    # Floor 1: release_memory + reset_state (current best API)
    req.reset_state()
    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    floor1 = rss_kb()
    print(f"floor1 (release+reset):  RSS={floor1/1024:7.1f} MiB"
          f"  delta {(peak-floor1)/1024:+.1f} MiB")

    # Floor 2: nuclear — del compiled, keep core+model
    del compiled
    libc.malloc_trim(0)
    floor2 = rss_kb()
    print(f"floor2 (del compiled):   RSS={floor2/1024:7.1f} MiB"
          f"  delta vs peak {(peak-floor2)/1024:+.1f} MiB")

    # Floor 3: also del lm (model)
    del lm
    libc.malloc_trim(0)
    floor3 = rss_kb()
    print(f"floor3 (also del model): RSS={floor3/1024:7.1f} MiB"
          f"  delta vs peak {(peak-floor3)/1024:+.1f} MiB")

    # Compare to llama.cpp at d=1024: ~4007 MiB peak
    print(f"\n=== vs llama.cpp peak at d=1024 (4007 MiB) ===")
    for name, val in [("floor1", floor1), ("floor2", floor2), ("floor3", floor3)]:
        delta = val/1024 - 4007
        marker = "BELOW" if delta < 0 else "above"
        print(f"  {name}: {val/1024:7.1f}  {marker} by {abs(delta):.0f} MiB")

    # === Now rebuild and verify correctness + measure cost ===
    print(f"\n=== rebuild from floor3 (simulating new conversation) ===")
    t0 = time.time()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    t_read = time.time() - t0
    print(f"  read_model:    {t_read:5.2f}s   RSS={rss_kb()/1024:.1f} MiB")
    t0 = time.time()
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    t_comp = time.time() - t0
    print(f"  compile_model: {t_comp:5.2f}s   RSS={rss_kb()/1024:.1f} MiB")
    req = compiled.create_infer_request()
    t0 = time.time()
    req.infer(feeds(lm, PP_LEN, past=0))
    t_pp = time.time() - t0
    print(f"  pp{PP_LEN} (cold):    {t_pp*1000:5.0f} ms  RSS={rss_kb()/1024:.1f} MiB")
    t0 = time.time()
    req.infer(feeds(lm, 1, past=PP_LEN))
    t_tg = time.time() - t0
    print(f"  one decode:    {t_tg*1000:5.0f} ms  RSS={rss_kb()/1024:.1f} MiB")
    print(f"\nTotal cost of nuclear reset: {(t_read+t_comp)*1000:.0f} ms")


if __name__ == "__main__":
    main()
