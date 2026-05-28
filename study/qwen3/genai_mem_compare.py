"""
Compare RSS during inference: unfused vs fused, both through ov_genai.VLMPipeline.

Two configs:
  - unfused: original IR, no custom ops, no extensions= needed
  - fused:   fused IR + extensions=[libqwen3_ov_ext.so]

Each runs in its own subprocess so heap state doesn't bleed across.
A long synthetic prompt is built by repeating a phrase to hit the requested
token count. A custom streamer captures time-to-first-token (TTFT, ≈ prefill
latency) and per-token decode timing; a background sampler tracks peak RSS
through the whole generate() call.
"""
import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SO = HERE / "cpp_ext/build/libqwen3_ov_ext.so"
ORIG = "/tmp/qwen3-work/qwen35-0.8b-int8"
# The "fused" path defaults to the all-3-rewrites dir; override via --fused-dir.
FUSED = os.environ.get("QWEN3_FUSED_DIR", "/tmp/qwen3-work/qwen35-0.8b-int8-fused")

PHRASE = (
    "Computers are remarkable machines that process information through "
    "carefully orchestrated electrical signals. Each operation in a CPU "
    "depends on billions of transistors switching in precise patterns. "
)


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return -1


def worker(mode: str, prompt_repeats: int, max_new: int):
    import openvino_genai as ov_genai

    state = {"start_t": None, "first_token_t": None}

    class PhaseSampler:
        """Sample RSS at 10ms; track peak per phase boundaries we set externally."""
        def __init__(self):
            self.samples = []  # list of (t, rss_kb)
            self._stop = threading.Event()
            self._t = threading.Thread(target=self._run, daemon=True)

        def _run(self):
            while not self._stop.is_set():
                self.samples.append((time.time(), rss_kb()))
                time.sleep(0.010)

        def __enter__(self):
            self.samples.append((time.time(), rss_kb()))
            self._t.start(); return self

        def __exit__(self, *a):
            self._stop.set(); self._t.join()
            self.samples.append((time.time(), rss_kb()))

    class TimerStreamer(ov_genai.StreamerBase):
        def __init__(self):
            super().__init__()
            self.first_t = None
            self.last_t = None
            self.ts = []

        def write(self, token):
            now = time.time()
            if self.first_t is None:
                self.first_t = now
                state["first_token_t"] = now
            self.last_t = now
            self.ts.append(now)
            return ov_genai.StreamingStatus.RUNNING

        def end(self):
            pass

    if mode == "fused":
        kwargs = {"extensions": [str(SO)]}
        path = FUSED
    else:
        kwargs = {}
        path = ORIG

    rss_start = rss_kb()
    t0 = time.time()
    pipe = ov_genai.VLMPipeline(path, "CPU", **kwargs)
    t_load = time.time() - t0
    rss_loaded = rss_kb()

    prompt = (PHRASE * prompt_repeats).strip()

    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new

    # Warmup: a tiny generate so the plugin JIT-compiles the dynamic-shape kernels
    # before we measure memory. Otherwise the first run's peak conflates
    # compile-time arenas with real working memory.
    print(f"  warming up...", file=sys.stderr)
    warm_cfg = ov_genai.GenerationConfig()
    warm_cfg.max_new_tokens = 4
    pipe.generate("Hello there.", generation_config=warm_cfg)
    rss_after_warmup = rss_kb()

    streamer = TimerStreamer()
    state["start_t"] = time.time()
    with PhaseSampler() as s:
        out = pipe.generate(prompt, generation_config=cfg, streamer=streamer)
    t_gen = time.time() - state["start_t"]
    rss_end = rss_kb()

    # Partition samples into prefill phase (t < first_token_t) vs decode phase
    first_t = state["first_token_t"]
    start_t = state["start_t"]
    pre_samples = [(t, r) for t, r in s.samples if first_t is None or t <= first_t]
    dec_samples = [(t, r) for t, r in s.samples if first_t is not None and t > first_t]
    pre_peak = max((r for _, r in pre_samples), default=rss_loaded)
    dec_peak = max((r for _, r in dec_samples), default=rss_loaded)
    overall_peak = max(pre_peak, dec_peak)

    ttft = (first_t - start_t) if first_t else None
    if streamer.first_t and streamer.last_t and len(streamer.ts) > 1:
        decode_total = streamer.last_t - streamer.first_t
        decode_tok_s = (len(streamer.ts) - 1) / decode_total if decode_total > 0 else None
    else:
        decode_tok_s = None

    out_text = str(out)

    # Optional: write RSS trace to /tmp for inspection
    trace_path = os.environ.get("RSS_TRACE_OUT")
    if trace_path:
        with open(trace_path, "w") as fh:
            fh.write(f"# {mode}\n")
            fh.write(f"# start_t={start_t}\n")
            fh.write(f"# first_token_t={first_t}\n")
            for t, r in s.samples:
                fh.write(f"{t - start_t:.3f}\t{r/1024:.1f}\n")

    result = {
        "mode": mode,
        "prompt_chars": len(prompt),
        "prompt_repeats": prompt_repeats,
        "max_new_tokens": max_new,
        "n_generated_tokens": len(streamer.ts),
        "load_s": t_load,
        "ttft_s": ttft,
        "decode_tok_s": decode_tok_s,
        "generate_s": t_gen,
        "rss_start_mb": rss_start / 1024,
        "rss_after_load_mb": rss_loaded / 1024,
        "rss_after_warmup_mb": rss_after_warmup / 1024,
        "rss_prefill_peak_mb": pre_peak / 1024,
        "rss_decode_peak_mb": dec_peak / 1024,
        "rss_peak_mb": overall_peak / 1024,
        "rss_end_mb": rss_end / 1024,
        "output_first_140": out_text[:140],
    }
    print("@@RESULT@@", json.dumps(result))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=["unfused", "fused"])
    ap.add_argument("--prompt-repeats", type=int, default=60,
                    help="how many times to repeat PHRASE (≈21 tokens each)")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--fused-dir",
                    help="model dir for the 'fused' arm (default: env QWEN3_FUSED_DIR or "
                         "/tmp/qwen3-work/qwen35-0.8b-int8-fused)")
    ap.add_argument("--label", default="fused",
                    help="label to print for the fused arm (e.g. fused-light)")
    args = ap.parse_args()
    if args.fused_dir:
        os.environ["QWEN3_FUSED_DIR"] = args.fused_dir

    if args.worker:
        worker(args.mode, args.prompt_repeats, args.max_new_tokens)
        return

    if not SO.exists():
        sys.exit(f"missing {SO} — build cpp_ext first")
    if not Path(ORIG).exists() or not Path(FUSED).exists():
        sys.exit(f"missing model dirs {ORIG} or {FUSED}")

    rows = []
    label_for = {"unfused": "unfused", "fused": args.label}
    for mode in ["unfused", "fused"]:
        print(f"\n  → {label_for[mode]}  (repeats={args.prompt_repeats}, "
              f"max_new={args.max_new_tokens})", flush=True)
        if mode == "fused":
            print(f"      fused dir: {FUSED}")
        cmd = [
            sys.executable, str(__file__), "--worker", "--mode", mode,
            "--prompt-repeats", str(args.prompt_repeats),
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        sub_env = dict(os.environ)
        res = subprocess.run(cmd, capture_output=True, text=True, env=sub_env)
        if res.returncode != 0:
            print(res.stdout); print(res.stderr)
            sys.exit(f"worker {mode} failed (exit {res.returncode})")
        line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
        rows.append(json.loads(line[len("@@RESULT@@ "):]))

    print("\n" + "=" * 124)
    print(f"{'mode':<12s} {'load(s)':>8s} {'TTFT(s)':>8s} {'gen(s)':>8s} {'tok/s':>7s} "
          f"{'loaded':>8s} {'warmed':>8s} {'pre_pk':>8s} {'dec_pk':>8s} {'peak':>8s} {'end':>8s} {'n_tok':>6s}")
    print("-" * 124)
    for r in rows:
        tok_s = f"{r['decode_tok_s']:.1f}" if r['decode_tok_s'] else "n/a"
        ttft = f"{r['ttft_s']:.2f}" if r['ttft_s'] else "n/a"
        display = label_for[r["mode"]]
        print(f"{display:<12s} "
              f"{r['load_s']:>8.2f} {ttft:>8s} {r['generate_s']:>8.2f} {tok_s:>7s} "
              f"{r['rss_after_load_mb']:>8.0f} {r['rss_after_warmup_mb']:>8.0f} "
              f"{r['rss_prefill_peak_mb']:>8.0f} {r['rss_decode_peak_mb']:>8.0f} "
              f"{r['rss_peak_mb']:>8.0f} {r['rss_end_mb']:>8.0f} {r['n_generated_tokens']:>6d}")

    if len(rows) == 2:
        u, f = rows[0], rows[1]
        print()
        for label, key in [
            ("loaded     ", "rss_after_load_mb"),
            ("warmed     ", "rss_after_warmup_mb"),
            ("prefill pk ", "rss_prefill_peak_mb"),
            ("decode pk  ", "rss_decode_peak_mb"),
            ("overall pk ", "rss_peak_mb"),
            ("end        ", "rss_end_mb"),
        ]:
            d = u[key] - f[key]
            base = u[key]
            print(f"  fused vs unfused at {label}: {d:>+7.0f} MB  ({d/base*100:+5.1f}%)")

    print("\n  unfused output (first 140 chars):")
    print(f"    {rows[0]['output_first_140']!r}")
    print("\n  fused output (first 140 chars):")
    print(f"    {rows[1]['output_first_140']!r}")


if __name__ == "__main__":
    main()
