"""Aggressively close the memory gap to llama.cpp on Nanbeige.

After reset_state + release_memory leaves OV at ~5.3 GiB while llama.cpp
sits at ~4.0 GiB, walk /proc/self/maps and madvise(MADV_DONTNEED) every
anonymous mapping >5 MiB one at a time. Measure RSS drop per mapping and
verify the next infer still produces correct(ish) output. Anything we
can drop without breaking the next infer is recoverable; the surviving
chunks are 'true OV overhead'.

Then re-infer to see how much pages back in. Net resident after one
post-release inference cycle is the realistic 'between-conversation
floor'.
"""
import argparse
import ctypes
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
T_DEPTH = 1024
PP_LEN = 128
TG_LEN = 32
THREADS = 4

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int
MADV_DONTNEED = 4


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def get_anon_mappings(min_kb=5120):
    """Returns sorted list of (lo, size_kb, path, perms) for r-w private
    anonymous mappings above min_kb, sorted by size desc."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.rstrip().split(None, 5)
            if len(parts) < 5: continue
            perms = parts[1]
            path = parts[5] if len(parts) > 5 else ""
            if path:  # only anonymous
                continue
            if perms[0] != 'r' or perms[1] != 'w' or perms[3] != 'p':
                continue  # only private anon read-write
            try:
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
            except ValueError:
                continue
            size_kb = (hi - lo) >> 10
            if size_kb < min_kb:
                continue
            out.append((lo, size_kb, path or "(anon)", perms))
    out.sort(key=lambda x: -x[1])
    return out


def feeds(lm, T, past):
    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=166000, size=(1, T), dtype=np.int64)
    return {
        "input_ids":      ov.Tensor(ids),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.arange(past, past + T, dtype=np.int64).reshape(1, T)),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def time_one_infer(req, lm, T, past):
    t0 = time.perf_counter()
    req.infer(feeds(lm, T, past))
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=T_DEPTH)
    ap.add_argument("--min-mib", type=float, default=5.0)
    ap.add_argument("--max-evict-passes", type=int, default=4)
    args = ap.parse_args()

    print(f"OV version: {ov.__version__}")
    core = ov.Core()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    print(f"after compile:           {rss_kb()/1024:7.1f} MiB")
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()
    print(f"after warmup:            {rss_kb()/1024:7.1f} MiB")

    # Build to depth + small pp + tg, like the prior sweep at d=1024
    req.reset_state()
    if args.depth > 0:
        req.infer(feeds(lm, args.depth, past=0))
    req.infer(feeds(lm, PP_LEN, past=args.depth))
    past = args.depth + PP_LEN
    for _ in range(TG_LEN):
        req.infer(feeds(lm, 1, past=past)); past += 1
    print(f"after work at d={args.depth}:    {rss_kb()/1024:7.1f} MiB")

    # Standard cleanup we already know about
    req.reset_state()
    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    print(f"after release+trim:      {rss_kb()/1024:7.1f} MiB  <-- post-API floor")

    # === AGGRESSIVE: madvise every large anon mapping ===
    min_kb = int(args.min_mib * 1024)
    print(f"\n=== anon mappings >= {args.min_mib:.0f} MiB before madvise ===")
    maps = get_anon_mappings(min_kb)
    for lo, sz, _, perms in maps[:30]:
        print(f"  {sz/1024:7.1f} MiB  {perms}  addr=0x{lo:x}")

    print(f"\n=== madvise(MADV_DONTNEED) per-mapping ===")
    pre = rss_kb()
    print(f"  RSS pre: {pre/1024:.1f} MiB")
    deltas = []
    for lo, sz, _, perms in maps:
        rss_before = rss_kb()
        ret = libc.madvise(lo, sz << 10, MADV_DONTNEED)
        if ret != 0:
            err = ctypes.get_errno()
            print(f"  {sz/1024:7.1f} MiB  FAIL errno={err}")
            continue
        rss_after = rss_kb()
        drop_kb = rss_before - rss_after
        deltas.append((sz, drop_kb))
        if drop_kb > 1024:  # >1 MiB drop
            print(f"  {sz/1024:7.1f} MiB  drop_actual={drop_kb/1024:7.1f} MiB  addr=0x{lo:x}")
    post = rss_kb()
    print(f"  RSS after all madvise: {post/1024:.1f} MiB  (total drop = {(pre-post)/1024:.1f} MiB)")

    # Verify next infer still works correctness-roughly (pages will fault in)
    print(f"\n=== verify next infer still runs (correctness via no crash) ===")
    try:
        req = compiled.create_infer_request()
        t = time_one_infer(req, lm, PP_LEN, past=0)
        print(f"  cold prefill {PP_LEN} after mass-evict: {t*1000:.0f} ms  RSS={rss_kb()/1024:.1f} MiB")
        t = time_one_infer(req, lm, PP_LEN, past=PP_LEN)
        print(f"  second prefill (warmer):              {t*1000:.0f} ms  RSS={rss_kb()/1024:.1f} MiB")
        t = time_one_infer(req, lm, 1, past=2*PP_LEN)
        print(f"  one decode token:                     {t*1000:.0f} ms  RSS={rss_kb()/1024:.1f} MiB")
    except Exception as e:
        print(f"  CRASHED: {str(e)[:200]}")


if __name__ == "__main__":
    main()
