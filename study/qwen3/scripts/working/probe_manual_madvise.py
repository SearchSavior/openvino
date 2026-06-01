"""Empirically test what the L1 'CPU plugin calls hint_evict' patch would
do, by walking /proc/self/maps after compile_model and calling
madvise(MADV_DONTNEED) on every mmap'd range of the model .bin file.

If VmHWM (or post-evict RSS) drops by ~700 MiB, the L1 patch is exactly
this call wrapped in OV's API."""
import ctypes, os, sys, subprocess
from pathlib import Path
import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")

MADV_DONTNEED = 4

# bind madvise(addr, length, advice) -> int via libc
libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
libc.madvise.restype = ctypes.c_int


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): return int(line.split()[1]) / 1024.0
    return 0


def smaps_rollup():
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                if ":" not in line: continue
                k, _, rest = line.partition(":")
                rest = rest.strip()
                if rest.endswith(" kB"):
                    out[k.strip()] = int(rest[:-3]) / 1024.0
    except FileNotFoundError:
        pass
    return out


def show_split(label):
    s = smaps_rollup()
    print(f"  {label}:")
    print(f"    Rss            = {s.get('Rss',0):7.1f} MiB")
    print(f"    Anonymous      = {s.get('Anonymous',0):7.1f} MiB")
    print(f"    Shared_Clean   = {s.get('Shared_Clean',0):7.1f} MiB  (file-backed)")
    print(f"    Private_Clean  = {s.get('Private_Clean',0):7.1f} MiB  (file-backed)")
    print(f"    Private_Dirty  = {s.get('Private_Dirty',0):7.1f} MiB  (anonymous + CoW)")
    print(f"    Pss_File       = {s.get('Pss_File',0):7.1f} MiB")


def find_bin_mappings(name_substr):
    rngs = []
    with open("/proc/self/maps") as f:
        for line in f:
            parts = line.rstrip().split(None, 5)
            if len(parts) < 6: continue
            if name_substr not in parts[5]: continue
            lo, hi = (int(x, 16) for x in parts[0].split("-"))
            rngs.append((lo, hi - lo, parts[1]))
    return rngs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=770)
    args = ap.parse_args()
    T = args.T
    code = f"""
import sys; sys.path.insert(0, '{Path(__file__).resolve().parents[2] / "kernels"}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/probe_madv.xml', '{WORK}/probe_madv.bin')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)

    print(f"T = {T}")
    print(f"pre-load:       RSS = {rss():7.1f} MiB")
    core = ov.Core()
    lm = core.read_model(f"{WORK}/probe_madv.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    print(f"after compile:  RSS = {rss():7.1f} MiB")

    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T, dtype=np.int64).reshape(1,1,T), (pid_b,1,1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds)
    print(f"after warmup:   RSS = {rss():7.1f} MiB")
    show_split("after warmup")

    rngs = find_bin_mappings("probe_madv.bin")
    total = sum(s for _, s, _ in rngs)
    print(f"\nFound {len(rngs)} mmap ranges of probe_madv.bin, totaling {total/(1<<20):.1f} MiB")

    # Now madvise DONTNEED on each range to simulate hint_evict.
    failed = 0
    for addr, size, perms in rngs:
        if libc.madvise(addr, size, MADV_DONTNEED) != 0:
            err = ctypes.get_errno()
            failed += 1
    if failed:
        print(f"  madvise failed on {failed}/{len(rngs)} ranges (last errno={err})")
    else:
        print(f"  madvise DONTNEED applied to all {len(rngs)} ranges OK")

    print(f"after madvise:  RSS = {rss():7.1f} MiB")
    show_split("after madvise")

    # Re-run a prefill -- weights get paged in again as needed. Lets us see
    # whether this would actually break correctness for subsequent infer.
    req.infer(feeds)
    print(f"after 2nd infer: RSS = {rss():7.1f} MiB")
    show_split("after 2nd infer")


if __name__ == "__main__":
    main()
