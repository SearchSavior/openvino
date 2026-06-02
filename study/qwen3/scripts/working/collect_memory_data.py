"""Collect comprehensive memory attribution data across configurations.
Runs one config per invocation; output is a JSON snapshot of every stage.

Configs:
  loop      - default model with TensorIterator Loop nodes (linear-attn intact)
  customop  - Loops replaced with v1 GatedDeltaRule custom op
  noattn    - sanity: just the 6 self-attn layers compute (no L1 patch needed)

At each stage we capture:
  - VmRSS, VmHWM, VmPeak
  - smaps_rollup full breakdown
  - mallinfo2 fields (arena, hblkhd, uordblks, fordblks, etc.)
  - /proc/self/maps grouped by mapping kind
  - Top-20 individual mappings by virtual size
  - Top-20 individual mappings by RSS
  - OV's CPU plugin stats CSV (written by ~CompiledModel — we re-read at end)
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
STATS_CSV_PREFIX = "/tmp/ov_mem"

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in
                ("arena", "ordblks", "smblks", "hblks", "hblkhd",
                 "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


libc.mallinfo2.restype = _MallInfo2


def read_status_kb(field):
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return 0


def smaps_rollup():
    out = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            if ":" not in line:
                continue
            k, _, rest = line.partition(":")
            rest = rest.strip()
            if rest.endswith(" kB"):
                out[k.strip()] = int(rest[:-3])
    return out


def categorize_maps():
    rss_by = {}
    size_by = {}
    top_size = []
    top_rss = []
    cur_kind = None
    cur_path = None
    cur_size_kb = 0
    cur_rss_kb = 0

    def kind_of(perms, path):
        if path == "[stack]" or path.startswith("[stack:"):
            return "stack"
        if path == "[heap]":
            return "heap"
        if path in ("[vdso]", "[vvar]", "[vvar_vclock]"):
            return "vdso"
        if path.endswith(".bin") and "s" in perms:
            return "bin_shared"
        if path.endswith(".so") or ".so." in path:
            return "so_text" if "x" in perms else "so_data"
        if not path:
            return "anon_rx" if "x" in perms else "anon_rw"
        return "other"

    def commit():
        nonlocal cur_kind, cur_path, cur_size_kb, cur_rss_kb
        if cur_kind is None:
            return
        rss_by[cur_kind] = rss_by.get(cur_kind, 0) + cur_rss_kb
        size_by[cur_kind] = size_by.get(cur_kind, 0) + cur_size_kb
        top_size.append((cur_size_kb, cur_path or "(anon)", cur_kind))
        top_rss.append((cur_rss_kb, cur_path or "(anon)", cur_kind))

    with open("/proc/self/smaps") as f:
        for line in f:
            parts = line.rstrip().split(None, 5)
            if not parts:
                continue
            first = parts[0]
            is_header = ("-" in first and not first.endswith(":")
                         and all(c in "0123456789abcdefABCDEF-" for c in first))
            if is_header:
                commit()
                if len(parts) < 5:
                    cur_kind = None
                    continue
                perms = parts[1]
                path = parts[5] if len(parts) > 5 else ""
                try:
                    lo, hi = (int(x, 16) for x in first.split("-"))
                except ValueError:
                    cur_kind = None
                    continue
                cur_kind = kind_of(perms, path)
                cur_path = path
                cur_size_kb = (hi - lo) >> 10
                cur_rss_kb = 0
            elif cur_kind is not None:
                if line.lstrip().startswith("Rss:"):
                    p2 = line.split()
                    if len(p2) >= 2 and p2[1].isdigit():
                        cur_rss_kb = int(p2[1])
        commit()

    top_size.sort(reverse=True)
    top_rss.sort(reverse=True)
    return rss_by, size_by, top_size[:25], top_rss[:25]


def snapshot(label):
    s = smaps_rollup()
    rss_by_kind, size_by_kind, top_size, top_rss = categorize_maps()
    mi = libc.mallinfo2()
    snap = {
        "label": label,
        "VmRSS_kb": read_status_kb("VmRSS"),
        "VmHWM_kb": read_status_kb("VmHWM"),
        "VmPeak_kb": read_status_kb("VmPeak"),
        "VmSize_kb": read_status_kb("VmSize"),
        "smaps_rollup_kb": s,
        "mallinfo2": {f: getattr(mi, f) for f, _ in _MallInfo2._fields_},
        "maps_rss_by_kind_kb": rss_by_kind,
        "maps_size_by_kind_kb": size_by_kind,
        "top_25_by_virtual_size": [{"size_kb": s, "path": p, "kind": k}
                                    for s, p, k in top_size],
        "top_25_by_rss": [{"rss_kb": r, "path": p, "kind": k}
                          for r, p, k in top_rss],
    }
    return snap


def prep(version):
    """Build the right model: 'loop' (default) or 'customop' (v1 GatedDeltaRule)."""
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    out = f"{WORK}/data_{version}.xml"
    if version == "loop":
        code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{out}', '{out.replace(".xml",".bin")}')
"""
    else:
        code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from fused_linear_attn import register as rc, replace_gated_delta_rule_loops
from lm_head_slice import slice_lm_head_to_last_token
c = ov.Core(); rc(c)
m = c.read_model('{ORIG}/openvino_language_model.xml')
n = replace_gated_delta_rule_loops(m)
slice_lm_head_to_last_token(m)
print(f'replaced loops: {{n}}')
ov.serialize(m, '{out}', '{out.replace(".xml",".bin")}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout); print(r.stderr)
        sys.exit("prep failed")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["loop", "customop"])
    ap.add_argument("--T", type=int, default=1024)
    ap.add_argument("--out", default="/tmp/mem_snapshot.json")
    args = ap.parse_args()

    os.environ["OV_CPU_MEMORY_STATISTICS_PATH"] = \
        f"{STATS_CSV_PREFIX}_{args.version}.csv"

    stages = []
    stages.append(snapshot("pre-load"))

    xml = prep(args.version)

    import numpy as np
    import openvino as ov

    core = ov.Core()
    if args.version == "customop":
        core.add_extension(str(SO))
    lm = core.read_model(xml)
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    stages.append(snapshot("after compile_model"))

    T = args.T
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    feeds = lambda: {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(T, dtype=np.int64).reshape(1, 1, T), (pid_b, 1, 1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }
    req = compiled.create_infer_request()
    req.infer(feeds())
    stages.append(snapshot(f"after warmup T={T} (infer #1)"))

    # Second infer at same T (steady state)
    req.infer(feeds())
    stages.append(snapshot(f"after infer #2 same T"))

    # release_memory cycle
    del req
    libc.malloc_trim(0)
    stages.append(snapshot("after del req + trim"))

    compiled.release_memory()
    libc.malloc_trim(0)
    stages.append(snapshot("after release_memory + trim"))

    # Re-infer (will succeed for customop, crash for loop)
    try:
        req = compiled.create_infer_request()
        req.infer(feeds())
        stages.append(snapshot(f"after re-infer post-release"))
        del req
    except RuntimeError as e:
        stages.append({"label": "re-infer crashed",
                       "error": str(e).split("\n")[0][:200]})

    # destroy compiled_model to flush the OV CSV
    del compiled
    libc.malloc_trim(0)
    stages.append(snapshot("after del compiled + trim"))

    # Try to read OV's CSV
    csv_paths = []
    for p in Path("/tmp").glob(f"ov_mem_{args.version}*.csv"):
        csv_paths.append({"path": str(p), "content": p.read_text()})

    out = {
        "version": args.version,
        "T": args.T,
        "stages": stages,
        "ov_stats_csv": csv_paths,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"snapshot written to {args.out}")

    # quick summary to stdout
    print(f"\n=== {args.version} T={args.T} summary ===")
    for s in stages:
        if "VmRSS_kb" not in s:
            print(f"  {s['label']:<40s} {s.get('error','')}")
        else:
            print(f"  {s['label']:<40s}  RSS={s['VmRSS_kb']/1024:7.1f} MiB  "
                  f"HWM={s['VmHWM_kb']/1024:7.1f}  "
                  f"arena={s['mallinfo2']['arena']/(1<<20):6.1f}  "
                  f"hblkhd={s['mallinfo2']['hblkhd']/(1<<20):6.1f}")


if __name__ == "__main__":
    main()
