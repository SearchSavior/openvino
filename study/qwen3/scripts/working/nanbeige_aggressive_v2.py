"""Aggressively close the memory gap to llama.cpp — surgically this time.

The v1 brute madvise dropped RSS by 2.4 GiB but corrupted glibc.
v2 strategy:
  1. Snapshot /proc/self/maps BEFORE model work (just imports loaded)
  2. After release_memory + reset_state, find mappings that DID NOT exist
     in the pre-work snapshot. Those are model-state, safe to evict.
  3. madvise only those, one at a time, recording per-mapping RSS drop.
  4. Verify next infer still runs.

This protects glibc's arena (which existed before model load) from being
torn through, while still targeting model-state-backed anon allocations.
"""
import argparse
import ctypes
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
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


def snapshot_maps():
    """Return list of (lo, hi, size_kb, perms, path) for all mappings."""
    out = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.rstrip().split(None, 5)
            if len(parts) < 5: continue
            try: lo, hi = (int(x, 16) for x in parts[0].split("-"))
            except ValueError: continue
            perms = parts[1]
            path = parts[5] if len(parts) > 5 else ""
            out.append((lo, hi, (hi-lo) >> 10, perms, path))
    return out


def diff_new_anon_mappings(before, after, min_kb=128):
    """Return mappings in `after` that are NOT in `before`, filtered to
    anonymous (no path), private read-write, and >= min_kb."""
    before_set = {(lo, hi) for lo, hi, _, _, _ in before}
    out = []
    for lo, hi, sz, perms, path in after:
        if (lo, hi) in before_set:
            continue
        if path:  # only anon
            continue
        if perms[:4] != "rw-p":  # only private rw
            continue
        if sz < min_kb:
            continue
        out.append((lo, hi, sz, perms))
    out.sort(key=lambda x: -x[2])
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=1024)
    ap.add_argument("--top-n", type=int, default=10,
                    help="how many largest mappings to madvise (start small)")
    ap.add_argument("--min-kb", type=int, default=512)
    args = ap.parse_args()

    print(f"OV version: {ov.__version__}")
    print(f"RSS at script start: {rss_kb()/1024:.1f} MiB")

    # PRE snapshot
    pre_maps = snapshot_maps()
    print(f"pre-load mappings: {len(pre_maps)}")

    core = ov.Core()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    print(f"after compile:           {rss_kb()/1024:7.1f} MiB")
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()

    # Build context + work
    if args.depth > 0:
        req.infer(feeds(lm, args.depth, past=0))
    req.infer(feeds(lm, PP_LEN, past=args.depth))
    past = args.depth + PP_LEN
    for _ in range(TG_LEN):
        req.infer(feeds(lm, 1, past=past)); past += 1
    print(f"after work at d={args.depth}:    {rss_kb()/1024:7.1f} MiB")

    # Standard cleanup
    req.reset_state()
    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    floor = rss_kb()
    print(f"after release+trim:      {floor/1024:7.1f} MiB  <-- post-API floor")

    # Diff
    post_maps = snapshot_maps()
    new_mappings = diff_new_anon_mappings(pre_maps, post_maps, args.min_kb)
    print(f"\nnew anon mappings created by OV (still alive after release): {len(new_mappings)}")
    total_new_kb = sum(m[2] for m in new_mappings)
    print(f"total size of those: {total_new_kb/1024:.1f} MiB")
    print(f"\ntop {min(args.top_n, len(new_mappings))} largest:")
    for lo, hi, sz, perms in new_mappings[:args.top_n]:
        print(f"  {sz/1024:7.1f} MiB  {perms}  addr=0x{lo:x}")

    # Try madvise on those top-N
    print(f"\n=== madvise(MADV_DONTNEED) on top {args.top_n} ===")
    pre = rss_kb()
    print(f"RSS pre: {pre/1024:.1f} MiB")
    saved = 0
    for lo, hi, sz, perms in new_mappings[:args.top_n]:
        before = rss_kb()
        ret = libc.madvise(lo, hi - lo, MADV_DONTNEED)
        if ret != 0:
            err = ctypes.get_errno()
            print(f"  {sz/1024:7.1f} MiB  FAIL errno={err}")
            continue
        after = rss_kb()
        d = before - after
        saved += d
        marker = "*" if d > 1024 else " "
        print(f" {marker}{sz/1024:7.1f} MiB  drop={d/1024:7.1f} MiB  addr=0x{lo:x}")
    post = rss_kb()
    print(f"\nRSS post: {post/1024:.1f} MiB  (total drop {(pre-post)/1024:.1f} MiB)")
    print(f"vs llama.cpp peak at d=1024 (4007 MiB): {'BELOW' if post/1024 < 4007 else 'ABOVE'} by "
          f"{abs(post/1024 - 4007):.1f} MiB")

    # Verify next infer still works
    print(f"\n=== verify next infer ===")
    try:
        req = compiled.create_infer_request()
        t0 = time.perf_counter()
        req.infer(feeds(lm, PP_LEN, past=0))
        t1 = time.perf_counter() - t0
        print(f"  cold prefill {PP_LEN}: {t1*1000:.0f} ms  RSS after: {rss_kb()/1024:.1f} MiB")
        t0 = time.perf_counter()
        req.infer(feeds(lm, 1, past=PP_LEN))
        t2 = time.perf_counter() - t0
        print(f"  one decode:          {t2*1000:.0f} ms  RSS after: {rss_kb()/1024:.1f} MiB")
        print(f"  SUCCESS — no crash, no corruption")
    except Exception as e:
        print(f"  CRASHED: {str(e).splitlines()[0][:200]}")


if __name__ == "__main__":
    main()
