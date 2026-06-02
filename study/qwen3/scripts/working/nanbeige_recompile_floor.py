"""Practical between-conversation reset: del+rebuild compiled_model from
an already-loaded ov.Model. Keeps read_model cost out of the cycle.

Measures: floor RSS after each cycle, recompile time, first-cold-infer time,
steady-state infer time. If this gives llama.cpp-parity RSS at acceptable
latency cost, this is the immediate practical recipe before any patches.

Compares directly:
  cycle A: release_memory only
  cycle B: del compiled, recreate
  cycle C: del compiled + del model + reread + recompile (full teardown)
"""
import argparse
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


def do_workload(req, lm, depth):
    req.reset_state()
    if depth > 0:
        req.infer(feeds(lm, depth, past=0))
    req.infer(feeds(lm, PP_LEN, past=depth))
    past = depth + PP_LEN
    for _ in range(TG_LEN):
        req.infer(feeds(lm, 1, past=past)); past += 1


def main():
    print(f"OV version: {ov.__version__}")
    core = ov.Core()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    print(f"after initial compile:   RSS={rss_kb()/1024:7.1f} MiB")

    # warmup
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()
    print(f"after warmup:            RSS={rss_kb()/1024:7.1f} MiB")

    # === Cycle A: release_memory ===
    print("\n=== Cycle A: release_memory only ===")
    do_workload(req, lm, DEPTH)
    peak_a = rss_kb()
    req.reset_state()
    del req
    t0 = time.time()
    compiled.release_memory()
    libc.malloc_trim(0)
    t_release = time.time() - t0
    floor_a = rss_kb()
    print(f"  peak={peak_a/1024:.1f}  floor={floor_a/1024:.1f}  cycle_cost={t_release*1000:.0f}ms")
    t0 = time.time()
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0))
    t_cold = time.time() - t0
    print(f"  next pp{PP_LEN} cold: {t_cold*1000:.0f}ms")

    # === Cycle B: del compiled, recompile ===
    print("\n=== Cycle B: del compiled, keep model, recompile ===")
    do_workload(req, lm, DEPTH)
    peak_b = rss_kb()
    del req
    del compiled
    t0 = time.time()
    libc.malloc_trim(0)
    floor_b = rss_kb()
    print(f"  peak={peak_b/1024:.1f}  floor_after_del_compiled={floor_b/1024:.1f}")
    t0 = time.time()
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    t_recompile = time.time() - t0
    after_compile = rss_kb()
    print(f"  recompile: {t_recompile*1000:.0f}ms  RSS_after_compile={after_compile/1024:.1f}")
    req = compiled.create_infer_request()
    t0 = time.time()
    req.infer(feeds(lm, PP_LEN, past=0))
    t_cold_b = time.time() - t0
    print(f"  next pp{PP_LEN} cold: {t_cold_b*1000:.0f}ms")
    print(f"  total cycle cost: {(t_recompile + t_cold_b)*1000:.0f}ms")

    # === Cycle C: full teardown ===
    print("\n=== Cycle C: full teardown (del compiled + del lm + reread + recompile) ===")
    do_workload(req, lm, DEPTH)
    peak_c = rss_kb()
    del req
    del compiled
    del lm
    libc.malloc_trim(0)
    floor_c = rss_kb()
    print(f"  peak={peak_c/1024:.1f}  floor_after_full_teardown={floor_c/1024:.1f}")
    t0 = time.time()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    t_read = time.time() - t0
    t0 = time.time()
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    t_full_recompile = time.time() - t0
    print(f"  read: {t_read*1000:.0f}ms  recompile: {t_full_recompile*1000:.0f}ms"
          f"  RSS={rss_kb()/1024:.1f}")
    req = compiled.create_infer_request()
    t0 = time.time()
    req.infer(feeds(lm, PP_LEN, past=0))
    t_cold_c = time.time() - t0
    print(f"  next pp{PP_LEN} cold: {t_cold_c*1000:.0f}ms")
    print(f"  total cycle cost: {(t_read + t_full_recompile + t_cold_c)*1000:.0f}ms")

    # Summary
    print("\n" + "=" * 60)
    print(f"{'cycle':<10}  {'peak':>8}  {'floor':>8}  {'cycle ms':>9}  {'vs llama 4007':>15}")
    print("-" * 60)
    print(f"{'A: release':<10}  {peak_a/1024:8.0f}  {floor_a/1024:8.0f}  "
          f"{t_release*1000:9.0f}  {floor_a/1024 - 4007:+15.0f}")
    print(f"{'B: del comp':<10}  {peak_b/1024:8.0f}  {floor_b/1024:8.0f}  "
          f"{(t_recompile+t_cold_b)*1000:9.0f}  {floor_b/1024 - 4007:+15.0f}")
    print(f"{'C: full':<10}  {peak_c/1024:8.0f}  {floor_c/1024:8.0f}  "
          f"{(t_read+t_full_recompile+t_cold_c)*1000:9.0f}  {floor_c/1024 - 4007:+15.0f}")


if __name__ == "__main__":
    main()
