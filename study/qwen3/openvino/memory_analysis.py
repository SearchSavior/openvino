"""
Empirical analysis of memory consumption: unfused vs fused (C++ custom op).

Hypotheses TESTED (no assumptions, all checked against runtime data):
  (A) Full-sequence output materialization: the C++ custom op writes the
      entire [B,H,T,D] / [B,C,KS+T-K+1] output of each layer in one shot.
  (B) OpenMP parallelism in kernels.c is allocating per-thread arenas.

Findings recorded in commit log:
  - Hypothesis B REJECTED: at OMP_NUM_THREADS=1 the fused peak RSS is
    identical to the OMP-default fused peak (3056 vs 3058 MB at seq=512;
    2395 vs 2393 at seq=1024). OpenMP is NOT the cause.
  - Hypothesis A is real at single-shot prefill (fused 3055 MB vs unfused
    4357 MB at seq=1024 in raw ov.Core) — fused actually saves memory there.
  - The genai regression seen through VLMPipeline is caused by VLMPipeline
    doing CHUNKED prefill for the unfused IR but apparently not for the
    fused IR (see probe_structural.py). At a matching chunk size (4x256)
    both converge to ~2100-2350 MB; at 16x64 chunks both reach a 1838 MB
    floor regardless of which IR is used.

The actual root cause discovery: the CPU plugin fuses the unfused IR's
`Loop` into its own internal primitive `GatedDeltaNet` (visible in
get_runtime_model() output). Our custom op shows up as `Reference`. Both
implementations work but the plugin's interpretation of chunked prefill
differs between them — that's why the genai VLM peak diverges.

Tools used (all available WITHOUT a debug-caps rebuild of OpenVINO):
  - /proc/self/status (VmRSS, VmPeak)  — process resident set + ever-peak
  - /proc/self/smaps_rollup            — anon vs file-backed bytes
  - glibc mallinfo2 via ctypes         — heap arena stats:
        uordblks  bytes in use,       fordblks  free in arena,
        hblkhd    mmap'd big allocs,  arena     sbrk total,
        keepcost  top of arena
  - compiled_model.get_runtime_model() — post-compile execution graph;
        Node.get_type_name() reveals plugin primitive choices
        (this is how we discovered the GatedDeltaNet primitive)
  - infer_request.get_profiling_info() — per-node CPU plugin timing
  - OMP_NUM_THREADS env var            — pin OpenMP fan-out to 1
  - model.reshape({...})               — probe static-shape compile path
  - sequential req.infer() w/ stateful KV — probe chunked-prefill behaviour
        (see probe_structural.py — this is the deciding experiment)

Modes:
  unfused / fused / fused-omp1 — raw ov.Core, single-shot prefill
  vlm-unfused / vlm-fused       — VLMPipeline, native chunking applied
"""
import argparse
import ctypes
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SO = HERE.parent / "cpp_ext/build/libqwen3_ov_ext.so"
ORIG_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
FUSED_XML = "/tmp/qwen3-work/qwen35-0.8b-int8-fused/openvino_language_model.xml"

HIDDEN = 1024
PREFILL_LEN = 512        # smaller than the genai run, fits in subprocess time budget


# ---------------------------------------------------------------------------
# Memory introspection helpers
# ---------------------------------------------------------------------------
class Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


_libc = ctypes.CDLL("libc.so.6")
_libc.mallinfo2.restype = Mallinfo2


def mallinfo() -> dict:
    m = _libc.mallinfo2()
    return {n: getattr(m, n) for n, _ in Mallinfo2._fields_}


def proc_status(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return -1


def rss_kb() -> int:
    return proc_status("VmRSS")


def vmpeak_kb() -> int:
    return proc_status("VmPeak")


def smaps_rollup() -> dict:
    out = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            for k in ("Rss", "Pss", "Anonymous", "File", "Shared_Clean",
                      "Shared_Dirty", "Private_Clean", "Private_Dirty"):
                if line.startswith(k + ":"):
                    out[k] = int(line.split()[1])
                    break
    return out


# ---------------------------------------------------------------------------
# Workload
# ---------------------------------------------------------------------------
def prefill_inputs(seq):
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def worker(mode: str, seq: int):
    import openvino as ov

    via_vlm = mode.startswith("vlm-")
    core_mode = mode.replace("vlm-", "")

    snapshots = []

    def snap(label: str):
        snapshots.append({
            "label": label,
            "t": time.time(),
            "rss_mb": rss_kb() / 1024,
            "vmpeak_mb": vmpeak_kb() / 1024,
            "smaps": {k: v / 1024 for k, v in smaps_rollup().items()},
            "malloc": {k: v / 1024 / 1024 for k, v in mallinfo().items()},
        })

    snap("start")

    if via_vlm:
        import openvino_genai as ov_genai
        path = "/tmp/qwen3-work/qwen35-0.8b-int8-fused" if core_mode == "fused" else "/tmp/qwen3-work/qwen35-0.8b-int8"
        kwargs = {"extensions": [str(SO)]} if core_mode == "fused" else {}
        pipe = ov_genai.VLMPipeline(path, "CPU", **kwargs)
        snap("after_compile")
        snap("after_create_request")
        snap("after_input_alloc")

        # Use prefill-only by setting max_new_tokens=1
        cfg = ov_genai.GenerationConfig()
        cfg.max_new_tokens = 1
        prompt = ("Computers process information through electrical signals. " * (seq // 6)).strip()

        sampled = []
        stop_flag = threading.Event()
        def sampler():
            while not stop_flag.is_set():
                sampled.append((time.time(), rss_kb()))
                time.sleep(0.005)
        samp_t = threading.Thread(target=sampler, daemon=True); samp_t.start()
        t0 = time.time()
        out = pipe.generate(prompt, generation_config=cfg)
        t_inf = time.time() - t0
        stop_flag.set(); samp_t.join()
        snap("after_infer")

        peak_rss = max((r for _, r in sampled), default=rss_kb())
        peak_t = next((t - t0 for t, r in sampled if r == peak_rss), 0)
        result = {
            "mode": mode, "seq": seq, "infer_s": t_inf,
            "peak_rss_during_infer_mb": peak_rss / 1024,
            "peak_rss_t_rel": peak_t,
            "snapshots": snapshots,
            "rt_node_count": -1,
            "rt_types_top10": {},
            "rt_primitives_top10": {},
            "prof_top": [],
            "prof_by_type": [],
            "n_samples": len(sampled),
        }
        print("@@RESULT@@", json.dumps(result))
        return

    core = ov.Core()
    if core_mode in {"fused", "fused-omp1"}:
        core.add_extension(str(SO))
        xml = FUSED_XML
    else:
        xml = ORIG_XML

    model = core.read_model(xml)
    snap("after_read_model")

    compiled = core.compile_model(model, "CPU", {
        "INFERENCE_NUM_THREADS": 4,
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": True,
    })
    snap("after_compile")

    req = compiled.create_infer_request()
    snap("after_create_request")

    inp = prefill_inputs(seq)
    snap("after_input_alloc")

    # Background RSS sampler during infer
    sampled = []
    stop_flag = threading.Event()

    def sampler():
        while not stop_flag.is_set():
            sampled.append((time.time(), rss_kb()))
            time.sleep(0.005)

    samp_t = threading.Thread(target=sampler, daemon=True)
    samp_t.start()
    t0 = time.time()
    req.infer(inp)
    t_inf = time.time() - t0
    stop_flag.set()
    samp_t.join()
    snap("after_infer")

    peak_rss = max((r for _, r in sampled), default=rss_kb())
    peak_t = next(t - t0 for t, r in sampled if r == peak_rss)

    # Runtime graph inspection
    rt = compiled.get_runtime_model()
    rt_ops = rt.get_ops()
    rt_type_count = Counter(op.get_type_name() for op in rt_ops)
    rt_prim_count = Counter()
    for op in rt_ops:
        info = op.get_rt_info()
        prim = ""
        for k in ("primitiveType", "execType"):
            if k in info:
                prim = str(info[k]); break
        rt_prim_count[prim] += 1

    # Per-node profiling — top by real_time
    prof = req.get_profiling_info()
    prof_sorted = sorted(prof, key=lambda p: p.real_time, reverse=True)[:15]
    prof_top = [{
        "name": p.node_name,
        "type": p.node_type,
        "exec_type": p.exec_type,
        "real_us": p.real_time.total_seconds() * 1e6,
        "cpu_us": p.cpu_time.total_seconds() * 1e6,
    } for p in prof_sorted]

    # Total real time across ALL nodes, by node_type
    prof_by_type = {}
    for p in prof:
        prof_by_type.setdefault(p.node_type, [0, 0.0])
        prof_by_type[p.node_type][0] += 1
        prof_by_type[p.node_type][1] += p.real_time.total_seconds() * 1e6
    prof_by_type_sorted = sorted(prof_by_type.items(), key=lambda x: x[1][1], reverse=True)[:10]

    result = {
        "mode": mode,
        "seq": seq,
        "infer_s": t_inf,
        "peak_rss_during_infer_mb": peak_rss / 1024,
        "peak_rss_t_rel": peak_t,
        "snapshots": snapshots,
        "rt_node_count": len(rt_ops),
        "rt_types_top10": dict(rt_type_count.most_common(10)),
        "rt_primitives_top10": dict(rt_prim_count.most_common(10)),
        "prof_top": prof_top,
        "prof_by_type": [{"type": t, "count": c, "real_us": rt}
                          for t, (c, rt) in prof_by_type_sorted],
        "n_samples": len(sampled),
    }
    print("@@RESULT@@", json.dumps(result))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one(mode: str, seq: int):
    env = dict(os.environ)
    if mode == "fused-omp1":
        env["OMP_NUM_THREADS"] = "1"
    print(f"\n→ {mode} ({'OMP_NUM_THREADS=1' if mode == 'fused-omp1' else 'OMP default'})")
    cmd = [sys.executable, str(__file__), "--worker", "--mode", mode, "--seq", str(seq)]
    res = subprocess.run(cmd, capture_output=True, env=env, text=True)
    if res.returncode != 0:
        print(res.stdout); print(res.stderr)
        sys.exit(f"worker {mode} failed (exit {res.returncode})")
    line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
    return json.loads(line[len("@@RESULT@@ "):])


def render_snapshots(rows):
    # use the row with most snapshots (vlm path is shorter)
    base = max(rows, key=lambda r: len(r["snapshots"]))
    keys = [s["label"] for s in base["snapshots"]]

    def snap_for(r, label):
        for s in r["snapshots"]:
            if s["label"] == label:
                return s
        return None

    def fmt(v):
        return f"{v:>14.0f}" if v is not None else f"{'-':>14s}"

    print(f"\n{'phase':<24s}  " + "  ".join(f"{r['mode']:>14s}" for r in rows))
    for k in keys:
        rss = [snap_for(r, k)["rss_mb"] if snap_for(r, k) else None for r in rows]
        print(f"  RSS {k:<20s}  " + "  ".join(fmt(v) for v in rss))
    print()
    for k in keys:
        for stat in ("uordblks", "hblkhd", "fordblks"):
            vals = [snap_for(r, k)["malloc"][stat] if snap_for(r, k) else None for r in rows]
            print(f"  mallinfo {stat:<10s} @ {k:<18s}  " +
                  "  ".join(f"{v:>14.1f}" if v is not None else f"{'-':>14s}" for v in vals))


def render_top_ops(r):
    print(f"\n  {r['mode']} runtime graph: {r['rt_node_count']} nodes")
    print(f"    top node types: {r['rt_types_top10']}")
    print(f"    top primitives: {r['rt_primitives_top10']}")
    print(f"  {r['mode']} top 10 by total time:")
    print(f"    {'type':<28s} {'count':>5s} {'total_us':>10s}")
    for e in r['prof_by_type']:
        print(f"    {e['type']:<28s} {e['count']:>5d} {e['real_us']:>10.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode",
                    choices=["unfused", "fused", "fused-omp1", "vlm-unfused", "vlm-fused"])
    ap.add_argument("--seq", type=int, default=PREFILL_LEN)
    ap.add_argument("--modes", nargs="+",
                    default=["unfused", "fused", "fused-omp1"],
                    help="which modes to run (subset of choices)")
    args = ap.parse_args()

    if args.worker:
        worker(args.mode, args.seq)
        return

    if not SO.exists():
        sys.exit(f"missing {SO} — build cpp_ext first")

    rows = [run_one(m, args.seq) for m in args.modes]

    print("\n" + "=" * 88)
    print(f"prefill seq={args.seq}, INFERENCE_NUM_THREADS=4 throughout\n")
    print(f"{'mode':<14s} {'infer(s)':>10s} {'peak_rss(MB)':>14s} {'peak@t(s)':>12s}")
    for r in rows:
        print(f"{r['mode']:<14s} {r['infer_s']:>10.2f} "
              f"{r['peak_rss_during_infer_mb']:>14.0f} "
              f"{r['peak_rss_t_rel']:>12.3f}")

    print("\n" + "-" * 88 + "\nphase-by-phase RSS and heap breakdown (MB)")
    render_snapshots(rows)

    for r in rows:
        print("\n" + "-" * 88)
        render_top_ops(r)

    print("\n=== JSON ===")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
