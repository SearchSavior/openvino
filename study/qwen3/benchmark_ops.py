"""
Head-to-head benchmark: our fused custom ops (C++ via libqwen3_ov_ext.so) vs
the CPU plugin's built-in primitives.

Background: the unfused IR's `Loop` is rewritten by the CPU plugin at compile
time into a dedicated `GatedDeltaNet` primitive (discovered via
compiled_model.get_runtime_model()). The unfused conv1d-state chain remains
as `GroupConvolution` + `Slice` (no fusion at plugin level).

What we compare per layer:
  linear-attention recurrence:
    unfused:  GatedDeltaNet                           — plugin primitive
    fused:    Reference (GatedDeltaRule, named …/Fused)
  conv1d with state:
    unfused:  GroupConvolution + Slice                — generic primitives
    fused:    Reference (FusedCausalConv1d, named …/Fused)

For each (config, seq) we:
  - run a warmup infer (skips JIT-compile artifacts in the timing)
  - run N (=5) measurement infers
  - extract per-op timing from infer_request.get_profiling_info()
  - group ops by friendly_name patterns and report:
      total time (sum across 18 layers × N runs) and avg per-call (us)

All runs in clean subprocesses per (config, seq) so heap state doesn't bleed.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SO = HERE / "cpp_ext/build/libqwen3_ov_ext.so"
ORIG_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
# fused XML path is overridable so we can profile fused-light (no conv1d) too.
FUSED_XML = os.environ.get(
    "QWEN3_FUSED_XML",
    "/tmp/qwen3-work/qwen35-0.8b-int8-fused/openvino_language_model.xml")
HIDDEN = 1024
WARMUP_RUNS = 1
MEASURE_RUNS = 5

# friendly_name fingerprints. The unfused IR's linear_attn Loop got mapped
# to GatedDeltaNet by the plugin; the conv1d chain stays as GroupConvolution
# inside the linear_attn module.
RE_LINATTN = re.compile(r"linear_attn")
RE_CONV_FUSED = re.compile(r"linear_attn/aten::cat/Concat/Fused")  # FusedCausalConv1d
RE_GDR_FUSED = re.compile(r"Loop_\d+/Fused")                         # GatedDeltaRule


def prefill_inputs(seq):
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def classify(node_type: str, name: str, mode: str) -> str | None:
    """Return a bucket label for ops we care about, else None.

    Works for both `fused` (gdr+conv1d) and `fused-light` (gdr only).
    """
    if mode == "unfused":
        if node_type == "GatedDeltaNet":
            return "linear_attn(GatedDeltaNet)"
        if node_type == "GroupConvolution" and RE_LINATTN.search(name):
            return "conv1d(GroupConvolution)"
        if node_type == "Slice" and "linear_attn/aten::cat/Concat" in name:
            return "conv1d_state(Slice)"
    else:
        # In fused IRs, the gdr replacement is always present. The conv1d
        # replacement is present only if it was applied at export time;
        # otherwise we still see the plugin's GroupConvolution.
        if node_type == "Reference":
            if RE_GDR_FUSED.search(name):
                return "linear_attn(GatedDeltaRule[fused])"
            if RE_CONV_FUSED.search(name):
                return "conv1d(FusedCausalConv1d[fused])"
        if node_type == "GroupConvolution" and RE_LINATTN.search(name):
            return "conv1d(GroupConvolution)"
        if node_type == "Slice" and "linear_attn/aten::cat/Concat" in name:
            return "conv1d_state(Slice)"
    return None


def worker(mode: str, seq: int):
    import openvino as ov

    core = ov.Core()
    if mode == "fused":
        core.add_extension(str(SO))
        xml = FUSED_XML
    else:
        xml = ORIG_XML

    model = core.read_model(xml)
    compiled = core.compile_model(model, "CPU", {
        "INFERENCE_NUM_THREADS": 4,
        "PERFORMANCE_HINT": "LATENCY",
        "PERF_COUNT": True,
    })
    req = compiled.create_infer_request()
    inp = prefill_inputs(seq)

    # Warmup
    for _ in range(WARMUP_RUNS):
        req.infer(inp)

    # Measure
    bucket_count = defaultdict(int)        # tag -> total invocations across runs
    bucket_time_us = defaultdict(float)    # tag -> total real_us across runs
    total_infer_s = 0.0
    for _ in range(MEASURE_RUNS):
        t0 = time.time()
        req.infer(inp)
        total_infer_s += time.time() - t0
        for p in req.get_profiling_info():
            tag = classify(p.node_type, p.node_name, mode)
            if tag is None:
                continue
            bucket_count[tag] += 1
            bucket_time_us[tag] += p.real_time.total_seconds() * 1e6

    result = {
        "mode": mode,
        "seq": seq,
        "measure_runs": MEASURE_RUNS,
        "total_infer_s": total_infer_s,
        "per_bucket": {
            tag: {
                "count": bucket_count[tag],
                "total_us": bucket_time_us[tag],
                "avg_us_per_call": bucket_time_us[tag] / bucket_count[tag],
            }
            for tag in bucket_count
        },
    }
    print("@@RESULT@@", json.dumps(result))


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        worker(sys.argv[i + 1], int(sys.argv[i + 2]))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", nargs="+", type=int, default=[128, 512, 1024, 2048])
    ap.add_argument("--fused-xml",
                    help="path to fused IR xml (default: env QWEN3_FUSED_XML or full fused)")
    ap.add_argument("--label", default="fused")
    args = ap.parse_args()
    if args.fused_xml:
        os.environ["QWEN3_FUSED_XML"] = args.fused_xml

    if not SO.exists():
        sys.exit(f"missing {SO} — build cpp_ext first")

    rows = []
    for seq in args.seqs:
        for mode in ("unfused", "fused"):
            print(f"\n→ {mode}  seq={seq}  (warmup={WARMUP_RUNS}, measure={MEASURE_RUNS})",
                  flush=True)
            cmd = [sys.executable, str(__file__), "--worker", mode, str(seq)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(res.stdout); print(res.stderr)
                sys.exit(f"worker {mode}/{seq} failed")
            line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
            rows.append(json.loads(line[len("@@RESULT@@ "):]))

    # ----------------- linear-attention head-to-head -----------------
    print("\n" + "=" * 80)
    print("LINEAR-ATTENTION RECURRENCE  (per-call avg, 18 layers × 5 runs = 90 samples)")
    print("=" * 80)
    print(f"{'seq':>6s}  {'plugin GatedDeltaNet':>26s}  {'our GatedDeltaRule':>26s}  {'speedup':>9s}")
    for seq in args.seqs:
        u = next(r for r in rows if r["mode"] == "unfused" and r["seq"] == seq)
        f = next(r for r in rows if r["mode"] == "fused" and r["seq"] == seq)
        plug = u["per_bucket"].get("linear_attn(GatedDeltaNet)", {}).get("avg_us_per_call")
        ours = f["per_bucket"].get("linear_attn(GatedDeltaRule[fused])", {}).get("avg_us_per_call")
        if plug is None or ours is None:
            continue
        ratio = plug / ours
        print(f"{seq:>6d}  {plug/1000:>21.2f} ms  {ours/1000:>21.2f} ms  "
              f"{ratio:>8.2f}x")

    # ----------------- conv1d head-to-head -----------------
    print("\n" + "=" * 80)
    print("CONV1D WITH STATE  (per-call avg)")
    print("=" * 80)
    print(f"{'seq':>6s}  {'unfused: GroupConv + Slice':>32s}  {'our FusedCausalConv1d':>26s}  {'speedup':>9s}")
    for seq in args.seqs:
        u = next(r for r in rows if r["mode"] == "unfused" and r["seq"] == seq)
        f = next(r for r in rows if r["mode"] == "fused" and r["seq"] == seq)
        gc = u["per_bucket"].get("conv1d(GroupConvolution)", {}).get("avg_us_per_call", 0)
        sl = u["per_bucket"].get("conv1d_state(Slice)", {}).get("avg_us_per_call", 0)
        plug = gc + sl
        ours = f["per_bucket"].get("conv1d(FusedCausalConv1d[fused])", {}).get("avg_us_per_call")
        if ours is None or plug == 0:
            continue
        ratio = plug / ours
        print(f"{seq:>6d}  {gc/1000:>10.2f}+{sl/1000:.2f}={plug/1000:>7.2f} ms"
              f"  {ours/1000:>21.2f} ms  {ratio:>8.2f}x")

    # ----------------- full infer time -----------------
    print("\n" + "=" * 80)
    print(f"END-TO-END PREFILL (avg of {MEASURE_RUNS} infer calls)")
    print("=" * 80)
    print(f"{'seq':>6s}  {'unfused (s)':>14s}  {'fused (s)':>14s}  {'speedup':>9s}")
    for seq in args.seqs:
        u = next(r for r in rows if r["mode"] == "unfused" and r["seq"] == seq)
        f = next(r for r in rows if r["mode"] == "fused" and r["seq"] == seq)
        us = u["total_infer_s"] / MEASURE_RUNS
        fs = f["total_infer_s"] / MEASURE_RUNS
        print(f"{seq:>6d}  {us:>14.3f}  {fs:>14.3f}  {us/fs:>8.2f}x")

    print("\n=== JSON ===")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
