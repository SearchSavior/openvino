"""
Sweep over {baseline, fused-py, fused-c} × prefill lengths.

Per config × length:
  - launch a clean subprocess (so heap state doesn't bleed across runs)
  - compile the LM (fused rewrites applied where applicable)
  - run prefill at the given length
  - run a fixed number of decode steps
  - report prefill latency / decode tok/s / peak RSS / resting RSS

  fused-py:  custom ops dispatch to numpy evaluate()
  fused-c:   QWEN3_USE_C=1 → custom ops dispatch to libqwen3_kernels.so

Usage:
    python sweep_kernels.py                       # default sweep
    python sweep_kernels.py --prefill-lens 1024
    python sweep_kernels.py --modes baseline fused-c
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_LENS = [256, 1024]
DEFAULT_MODES = ["baseline", "fused-py", "fused-c"]
DEFAULT_DECODE_STEPS = 32


# ---------------------------------------------------------------------------
# Worker — runs ONE (mode, prefill_len) measurement in its own process.
# ---------------------------------------------------------------------------
def _worker(mode: str, prefill_len: int, decode_steps: int):
    import threading
    import numpy as np
    import openvino as ov

    sys.path.insert(0, str(HERE))
    from fused_linear_attn import (
        register as register_la, replace_gated_delta_rule_loops)
    from fused_conv1d import (
        register as register_cv, replace_causal_conv1d_chains)
    from lm_head_slice import slice_lm_head_to_last_token

    MODEL_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
    HIDDEN = 1024

    def rss_kb():
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
        return -1

    class Sampler:
        def __init__(self):
            self.peak = 0
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)

        def _run(self):
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_kb())
                time.sleep(0.010)

        def __enter__(self):
            self.peak = rss_kb(); self._t.start(); return self

        def __exit__(self, *a):
            self._stop.set(); self._t.join()

    apply_fusions = mode != "baseline"
    core = ov.Core()
    if apply_fusions:
        register_la(core); register_cv(core)
    model = core.read_model(MODEL_XML)
    if apply_fusions:
        n1 = replace_gated_delta_rule_loops(model)
        n2 = replace_causal_conv1d_chains(model)
        ok = slice_lm_head_to_last_token(model)
    else:
        n1 = n2 = 0; ok = False
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 4})
    req = compiled.create_infer_request()

    rss_after_compile = rss_kb()

    rng = np.random.default_rng(0)
    def prefill_inputs(seq):
        return {
            "inputs_embeds": (rng.standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32),
            "attention_mask": np.ones((1, seq), dtype=np.int64),
            "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
            "beam_idx": np.zeros((1,), dtype=np.int32),
        }

    def decode_inputs(past_len):
        return {
            "inputs_embeds": (rng.standard_normal((1, 1, HIDDEN)) * 0.02).astype(np.float32),
            "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
            "position_ids": np.full((4, 1, 1), past_len, dtype=np.int64),
            "beam_idx": np.zeros((1,), dtype=np.int32),
        }

    # Prefill
    t0 = time.monotonic()
    with Sampler() as s_pre:
        req.infer(prefill_inputs(prefill_len))
    t_prefill = time.monotonic() - t0
    rss_after_prefill = rss_kb()

    # Decode
    past = prefill_len
    decode_times = []
    decode_peak = rss_after_prefill
    for _ in range(decode_steps):
        t1 = time.monotonic()
        with Sampler() as s_dec:
            req.infer(decode_inputs(past))
        decode_times.append(time.monotonic() - t1)
        decode_peak = max(decode_peak, s_dec.peak)
        past += 1

    rss_end = rss_kb()
    median_step = sorted(decode_times)[len(decode_times) // 2]
    median_first_5 = sorted(decode_times[:5])[len(decode_times[:5]) // 2] if decode_steps >= 5 else median_step

    result = {
        "mode": mode,
        "prefill_len": prefill_len,
        "decode_steps": decode_steps,
        "rewrites": {"linear_attn": n1, "conv1d": n2, "lm_head_slice": ok},
        "use_c": os.environ.get("QWEN3_USE_C", "0") == "1",
        "prefill_s": t_prefill,
        "decode_median_ms": median_step * 1000,
        "decode_median_first_5_ms": median_first_5 * 1000,
        "decode_tok_s": decode_steps / sum(decode_times),
        "rss_compile_mb": rss_after_compile / 1024,
        "rss_prefill_peak_mb": s_pre.peak / 1024,
        "rss_after_prefill_mb": rss_after_prefill / 1024,
        "rss_decode_peak_mb": decode_peak / 1024,
        "rss_end_mb": rss_end / 1024,
    }
    print("@@RESULT@@", json.dumps(result))


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_one(mode: str, prefill_len: int, decode_steps: int) -> dict:
    env = dict(os.environ)
    env["QWEN3_USE_C"] = "1" if mode == "fused-c" else "0"
    cmd = [sys.executable, str(__file__), "--worker", mode, str(prefill_len), str(decode_steps)]
    print(f"\n  → mode={mode:<10s}  prefill_len={prefill_len}  decode_steps={decode_steps}")
    res = subprocess.run(cmd, capture_output=True, env=env, text=True)
    if res.returncode != 0:
        print(res.stdout)
        print(res.stderr)
        raise RuntimeError(f"worker {mode} failed (exit {res.returncode})")
    line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
    return json.loads(line[len("@@RESULT@@ "):])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    ap.add_argument("--prefill-lens", nargs="+", type=int, default=DEFAULT_LENS)
    ap.add_argument("--decode-steps", type=int, default=DEFAULT_DECODE_STEPS)
    ap.add_argument("rest", nargs="*", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        mode, prefill_len, decode_steps = args.rest
        _worker(mode, int(prefill_len), int(decode_steps))
        return

    so_path = HERE / "libqwen3_kernels.so"
    if "fused-c" in args.modes and not so_path.exists():
        print(f"libqwen3_kernels.so not built — run {HERE}/build_kernels.sh first")
        sys.exit(1)

    rows = []
    for L in args.prefill_lens:
        for m in args.modes:
            rows.append(run_one(m, L, args.decode_steps))

    # Summary table
    print("\n" + "=" * 110)
    print(f"{'mode':<10s} {'pref_len':>8s} {'pref(s)':>8s} {'med_ms':>8s} {'med5_ms':>8s} "
          f"{'tok/s':>7s} {'pre_pk':>8s} {'pre_rest':>9s} {'dec_pk':>8s} {'end_mb':>8s}")
    print("-" * 110)
    for r in rows:
        print(f"{r['mode']:<10s} {r['prefill_len']:>8d} "
              f"{r['prefill_s']:>8.2f} "
              f"{r['decode_median_ms']:>8.1f} "
              f"{r['decode_median_first_5_ms']:>8.1f} "
              f"{r['decode_tok_s']:>7.1f} "
              f"{r['rss_prefill_peak_mb']:>8.0f} "
              f"{r['rss_after_prefill_mb']:>9.0f} "
              f"{r['rss_decode_peak_mb']:>8.0f} "
              f"{r['rss_end_mb']:>8.0f}")

    # Speedup comparisons within each prefill_len
    print("\n" + "=" * 110)
    print("speedups (relative to baseline at each prefill_len, decode median ms):")
    by_len = {}
    for r in rows:
        by_len.setdefault(r["prefill_len"], {})[r["mode"]] = r
    for L, modes in by_len.items():
        if "baseline" not in modes:
            continue
        b = modes["baseline"]["decode_median_ms"]
        line = f"  prefill_len={L:<5d}  baseline={b:.1f}ms"
        for name, r in modes.items():
            if name == "baseline":
                continue
            ratio = b / r["decode_median_ms"]
            line += f"   {name}={r['decode_median_ms']:.1f}ms ({ratio:.2f}x)"
        print(line)

    print("\nresults JSON:")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
