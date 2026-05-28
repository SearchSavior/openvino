"""
Test the chunked-prefill hypothesis using ContinuousBatchingPipeline.

VLMPipeline does chunked prefill internally and we can't toggle it. CB
pipeline exposes the knob: SchedulerConfig.dynamic_split_fuse (default True,
chunk size = max_num_batched_tokens default 256).

Matrix:
  IR ∈ {unfused, fused}
  chunking ∈ {on (default), off (dynamic_split_fuse=False, big batch)}

Hypothesis from prior runs:
  - unfused, chunk-ON:  ~2.2 GB peak  (VLMPipeline-equivalent)
  - unfused, chunk-OFF: ~4.3 GB peak  (raw ov.Core baseline at seq=1024)
  - fused,   chunk-ON:  ~3.4 GB peak  (chunking blocked by custom op)
  - fused,   chunk-OFF: ~3.0 GB peak  (same as ON — chunking already off-by-effect)

If chunk-OFF brings the unfused peak up to match fused, that confirms
chunked prefill alone explains the genai memory gap. If it doesn't move,
the cause is something else inside CB / VLM pipeline.

Tooling:
  /proc/self/status VmRSS, glibc mallinfo2, openvino_genai.StreamerBase
  (callable form) for token-by-token timing.
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
IR_DIRS = {
    "unfused": "/tmp/qwen3-work/qwen35-0.8b-int8",
    "slice-only": "/tmp/qwen3-work/qwen35-0.8b-int8-slice-only",
    "fused-light": "/tmp/qwen3-work/qwen35-0.8b-int8-fused-light",
    "fused": "/tmp/qwen3-work/qwen35-0.8b-int8-fused",
}
NEEDS_EXT = {"fused", "fused-light"}

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


def worker(ir: str, chunk_on: bool, prompt_repeats: int, max_new: int):
    import openvino_genai as ov_genai

    state = {"start_t": None, "first_t": None, "last_t": None, "n_tok": 0}

    def streamer_cb(text):
        now = time.time()
        if state["first_t"] is None:
            state["first_t"] = now
        state["last_t"] = now
        state["n_tok"] += 1
        return False  # keep streaming

    # Scheduler config
    sc = ov_genai.SchedulerConfig()
    sc.cache_size = 2  # GB
    if chunk_on:
        sc.dynamic_split_fuse = True
        sc.max_num_batched_tokens = 256  # default — chunked prefill
    else:
        sc.dynamic_split_fuse = False
        # Make batched tokens big enough to fit the full prompt + decode in one go.
        sc.max_num_batched_tokens = 8192
    sc.max_num_seqs = 8

    path = IR_DIRS[ir]
    props = {"extensions": [str(SO)]} if ir in NEEDS_EXT else {}

    rss_start = rss_kb()
    t0 = time.time()
    pipe = ov_genai.ContinuousBatchingPipeline(path, sc, "CPU", props)
    t_load = time.time() - t0
    rss_loaded = rss_kb()

    prompt = (PHRASE * prompt_repeats).strip()
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = max_new

    # Warmup
    pipe.generate("Hello there.", ov_genai.GenerationConfig(max_new_tokens=4))
    rss_warmed = rss_kb()

    # Sampler thread for peak RSS through generate()
    sampled = []
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            sampled.append((time.time(), rss_kb()))
            time.sleep(0.005)

    th = threading.Thread(target=sampler, daemon=True); th.start()
    state["start_t"] = time.time()
    out = pipe.generate(prompt, cfg, streamer_cb)
    t_gen = time.time() - state["start_t"]
    stop.set(); th.join()
    rss_end = rss_kb()

    first_t = state["first_t"]; last_t = state["last_t"]
    start_t = state["start_t"]
    pre_samples = [r for t, r in sampled if first_t is None or t <= first_t]
    dec_samples = [r for t, r in sampled if first_t is not None and t > first_t]
    pre_peak = max(pre_samples) if pre_samples else rss_loaded
    dec_peak = max(dec_samples) if dec_samples else rss_loaded
    overall_peak = max(pre_peak, dec_peak)

    ttft = (first_t - start_t) if first_t else None
    n_tok = state["n_tok"]
    if first_t and last_t and n_tok > 1:
        dec_tok_s = (n_tok - 1) / max(last_t - first_t, 1e-9)
    else:
        dec_tok_s = None

    out_text = str(out[0]) if isinstance(out, list) and out else str(out)
    result = {
        "ir": ir,
        "chunk_on": chunk_on,
        "scheduler_max_num_batched_tokens": sc.max_num_batched_tokens,
        "scheduler_dynamic_split_fuse": sc.dynamic_split_fuse,
        "prompt_repeats": prompt_repeats,
        "max_new_tokens": max_new,
        "n_generated_tokens": n_tok,
        "load_s": t_load,
        "ttft_s": ttft,
        "decode_tok_s": dec_tok_s,
        "generate_s": t_gen,
        "rss_start_mb": rss_start / 1024,
        "rss_loaded_mb": rss_loaded / 1024,
        "rss_warmed_mb": rss_warmed / 1024,
        "rss_prefill_peak_mb": pre_peak / 1024,
        "rss_decode_peak_mb": dec_peak / 1024,
        "rss_peak_mb": overall_peak / 1024,
        "rss_end_mb": rss_end / 1024,
        "output_first_140": out_text[:140],
    }
    print("@@RESULT@@", json.dumps(result))


def main():
    if "--worker" in sys.argv:
        i = sys.argv.index("--worker")
        ir, chunk = sys.argv[i + 1], sys.argv[i + 2]
        worker(ir, chunk == "on",
               int(sys.argv[i + 3]), int(sys.argv[i + 4]))
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-repeats", type=int, default=60)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--irs", nargs="+", default=["unfused", "slice-only", "fused-light", "fused"])
    ap.add_argument("--results-out", default="/tmp/cb_test_results.jsonl",
                    help="persist each row as a jsonl line so a later failure doesn't lose data")
    args = ap.parse_args()

    if not SO.exists():
        sys.exit(f"missing {SO} — build cpp_ext first")

    open(args.results_out, "w").close()

    rows = []
    for ir in args.irs:
        for chunk in ("on", "off"):
            print(f"\n→ ir={ir}  chunk={chunk}  repeats={args.prompt_repeats}", flush=True)
            cmd = [sys.executable, str(__file__), "--worker", ir, chunk,
                   str(args.prompt_repeats), str(args.max_new_tokens)]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                fail_msg = (res.stderr.splitlines()[-1] if res.stderr else "no stderr")[:200]
                row = {"ir": ir, "chunk_on": chunk == "on", "FAILED": fail_msg}
                rows.append(row)
                with open(args.results_out, "a") as fh:
                    fh.write(json.dumps(row) + "\n")
                print(f"  FAILED: {fail_msg}")
                continue
            line = next(L for L in res.stdout.splitlines() if L.startswith("@@RESULT@@"))
            row = json.loads(line[len("@@RESULT@@ "):])
            rows.append(row)
            with open(args.results_out, "a") as fh:
                fh.write(json.dumps(row) + "\n")

    print("\n" + "=" * 104)
    print(f"{'ir':<12s} {'chunk':>6s} {'TTFT(s)':>8s} {'dec(tok/s)':>11s} "
          f"{'pre_pk(MB)':>11s} {'dec_pk(MB)':>11s} {'peak(MB)':>9s} {'end(MB)':>8s} {'n_tok':>6s}")
    print("-" * 104)
    for r in rows:
        chunk = "on" if r.get('chunk_on') else "off"
        if "FAILED" in r:
            print(f"{r['ir']:<12s} {chunk:>6s}   FAILED: {r['FAILED'][:75]}")
            continue
        tok_s = f"{r['decode_tok_s']:.1f}" if r['decode_tok_s'] else "n/a"
        ttft = f"{r['ttft_s']:.2f}" if r['ttft_s'] else "n/a"
        print(f"{r['ir']:<12s} {chunk:>6s} {ttft:>8s} {tok_s:>11s} "
              f"{r['rss_prefill_peak_mb']:>11.0f} {r['rss_decode_peak_mb']:>11.0f} "
              f"{r['rss_peak_mb']:>9.0f} {r['rss_end_mb']:>8.0f} {r['n_generated_tokens']:>6d}")

    print("\nhypothesis check — does chunk-OFF lift the unfused peak?")
    by_key = {(r["ir"], r.get("chunk_on")): r for r in rows if "FAILED" not in r}
    for ir in args.irs:
        if (ir, True) in by_key and (ir, False) in by_key:
            on = by_key[(ir, True)]["rss_peak_mb"]
            off = by_key[(ir, False)]["rss_peak_mb"]
            print(f"  {ir:<12s}  chunk-ON peak: {on:>5.0f} MB   chunk-OFF peak: {off:>5.0f} MB"
                  f"   (delta: {off-on:+.0f} MB)")

    print("\n=== JSON ===")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
