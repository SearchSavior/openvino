"""Attribute the resident memory at T=1024 to its components.

Combines four data sources:
  1. /proc/self/status VmRSS at stages (load, compile, warmup, after L1)
  2. /proc/self/smaps_rollup breakdown (Anonymous vs file-backed splits)
  3. /proc/self/maps grouped by mapping kind (private anon, mmap'd .bin,
     mmap'd .so library code, ovh CACHE_DIR cached blobs, etc.)
  4. mallinfo2() — glibc heap usage (arena, mmap, total allocated)

  Also dumps OV's CPU plugin memory statistics via OV_CPU_MEMORY_STATISTICS_PATH
  if available (debug-caps build only). Output written to /tmp/ov_mem_stats.csv."""
import ctypes, os, sys, subprocess, threading, time
from pathlib import Path
import openvino as ov
import numpy as np

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
T = 1024

os.environ.setdefault("OV_CPU_MEMORY_STATISTICS_PATH", "/tmp/ov_mem_stats.csv")


libc = ctypes.CDLL("libc.so.6", use_errno=True)
# struct mallinfo2 (size_t-wide, returned by-value)
class _MallInfo2(ctypes.Structure):
    _fields_ = [("arena", ctypes.c_size_t),
                ("ordblks", ctypes.c_size_t),
                ("smblks", ctypes.c_size_t),
                ("hblks", ctypes.c_size_t),
                ("hblkhd", ctypes.c_size_t),
                ("usmblks", ctypes.c_size_t),
                ("fsmblks", ctypes.c_size_t),
                ("uordblks", ctypes.c_size_t),
                ("fordblks", ctypes.c_size_t),
                ("keepcost", ctypes.c_size_t)]
libc.mallinfo2.restype = _MallInfo2


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): return int(line.split()[1]) / 1024
    return 0


def smaps_rollup():
    out = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if ":" not in line: continue
            k, _, rest = line.partition(":")
            rest = rest.strip()
            if rest.endswith(" kB"):
                out[k.strip()] = int(rest[:-3]) / 1024
    return out


def categorize_maps():
    """Walk /proc/self/maps and classify every RESIDENT region by kind.
    Each mapping's resident bytes come from /proc/self/smaps (per-VMA detail).
    """
    BUCKETS = {
        "anon_rw":  "writable anonymous (heap + per-thread + scratchpads)",
        "anon_rx":  "executable anonymous (JIT code)",
        "bin_shared": "model .bin mmap (shared, read-only)",
        "so_text":  "library text (.so r-xp)",
        "so_data":  "library data (.so rw-p)",
        "stack":    "thread stacks",
        "heap":     "[heap] brk arena",
        "vdso":     "[vdso]/[vvar]",
        "other":    "other",
    }
    rss_by_bucket = {k: 0 for k in BUCKETS}
    count_by_bucket = {k: 0 for k in BUCKETS}
    samples = {k: [] for k in BUCKETS}

    cur_kind = None
    cur_size = 0
    with open("/proc/self/smaps") as f:
        for line in f:
            if not line: continue
            # A header line starts with hex addresses. All other lines (Size:,
            # Rss:, Pss:, VmFlags:, etc.) have a key followed by ':'. The
            # header has no colon in column 0..first-space.
            parts = line.rstrip().split(None, 5)
            if not parts: continue
            first = parts[0]
            is_header = ("-" in first and first.replace("-", "").replace(",","").isalnum()
                         and not first.endswith(":"))
            if is_header:
                if len(parts) < 5: continue
                perms = parts[1]
                path = parts[5] if len(parts) > 5 else ""
                if path == "[stack]" or path.startswith("[stack:"):
                    kind = "stack"
                elif path == "[heap]":
                    kind = "heap"
                elif path in ("[vdso]", "[vvar]", "[vvar_vclock]"):
                    kind = "vdso"
                elif path.endswith(".bin") and "s" in perms:
                    kind = "bin_shared"
                elif path.endswith(".so") or ".so." in path or path.endswith(".so.0"):
                    if "x" in perms: kind = "so_text"
                    else: kind = "so_data"
                elif not path:
                    kind = "anon_rx" if "x" in perms else "anon_rw"
                else:
                    kind = "other"
                cur_kind = kind
                try:
                    lo, hi = (int(x, 16) for x in parts[0].split("-"))
                except ValueError:
                    cur_kind = None
                    continue
                cur_size = (hi - lo) >> 10  # KiB
                samples[kind].append((path or "(anon)", cur_size))
                count_by_bucket[kind] += 1
            else:
                if cur_kind and line.lstrip().startswith("Rss:"):
                    parts2 = line.split()
                    if len(parts2) >= 2 and parts2[1].isdigit():
                        rss_by_bucket[cur_kind] += int(parts2[1])
    return rss_by_bucket, count_by_bucket, samples


def show(label):
    s = smaps_rollup()
    print(f"\n=== {label} ===")
    print(f"  VmRSS = {s.get('Rss',0):8.1f} MiB")
    print(f"    Anonymous     = {s.get('Anonymous',0):8.1f} MiB")
    print(f"    Pss_File      = {s.get('Pss_File',0):8.1f} MiB")
    print(f"    Shared_Clean  = {s.get('Shared_Clean',0):8.1f} MiB")
    print(f"    Private_Clean = {s.get('Private_Clean',0):8.1f} MiB")
    print(f"    Private_Dirty = {s.get('Private_Dirty',0):8.1f} MiB")
    mi = libc.mallinfo2()
    print(f"  mallinfo2: arena={mi.arena/(1<<20):.1f} MiB  hblkhd={mi.hblkhd/(1<<20):.1f} MiB  uordblks={mi.uordblks/(1<<20):.1f} MiB")


def main():
    # rebuild light IR via subprocess so parent RSS starts low
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))
    subprocess.run([sys.executable, "-c", f"""
import sys; sys.path.insert(0, '{Path(__file__).resolve().parents[2] / 'kernels'}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/attr.xml', '{WORK}/attr.bin')
"""], check=True, capture_output=True)

    show("pre-load")
    core = ov.Core()
    lm = core.read_model(f"{WORK}/attr.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    show("after compile_model")

    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1,T,hidden), dtype=np.float32)*0.01),
        "attention_mask": ov.Tensor(np.ones((1,T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T, dtype=np.int64).reshape(1,1,T),(pid_b,1,1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds)
    show(f"after warmup (T={T})")

    # Categorize maps after warmup
    rss_b, cnt_b, samples = categorize_maps()
    print("\n=== /proc/self/maps breakdown by mapping kind (sum Rss) ===")
    total = sum(rss_b.values())
    for k in sorted(rss_b, key=lambda x: -rss_b[x]):
        if rss_b[k] == 0: continue
        print(f"  {rss_b[k]/1024:>7.1f} MiB ({rss_b[k]*100/total:5.1f}%)  x{cnt_b[k]:>3d}  {k}")
    print(f"  ----")
    print(f"  {total/1024:>7.1f} MiB  total resident")

    # Top 10 individual mappings by SIZE (not RSS — to confirm what's biggest)
    print("\n=== top 10 individual mappings by virtual size ===")
    flat = [(s, p, k) for k, lst in samples.items() for p, s in lst]
    flat.sort(reverse=True)
    for s, p, k in flat[:10]:
        print(f"  {s/1024:>7.1f} MiB  [{k:<11s}] {p[:80]}")


if __name__ == "__main__":
    main()
