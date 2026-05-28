"""
Compare decode-time memory usage: unfused vs fully fused.

Pipeline:
  1. Prefill at PREFILL_LEN tokens (synthetic, fixed seed).
  2. Run DECODE_STEPS decode iterations.
  3. Sample RSS at 10 ms intervals across the entire run.
  4. Report:
     - peak RSS during prefill
     - resting RSS right after prefill
     - peak RSS across the whole decode loop (= worst decode step)
     - resting RSS at end of decode
     - per-step working memory (peak above resting at each decode step)

Run as a separate subprocess for each config so heap state doesn't leak.
"""
import argparse, sys, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import openvino as ov
from fused_linear_attn import register as register_la, replace_gated_delta_rule_loops
from fused_conv1d import register as register_cv, replace_causal_conv1d_chains
from lm_head_slice import slice_lm_head_to_last_token

MODEL_XML = "/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml"
HIDDEN = 1024
SAMPLE_S = 0.010


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
            time.sleep(SAMPLE_S)

    def __enter__(self):
        self.peak = rss_kb(); self._t.start(); return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join()


def prefill_inputs(seq):
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, HIDDEN)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def decode_inputs(past_len):
    return {
        "inputs_embeds": np.random.randn(1, 1, HIDDEN).astype(np.float32) * 0.02,
        "attention_mask": np.ones((1, past_len + 1), dtype=np.int64),
        "position_ids": np.full((4, 1, 1), past_len, dtype=np.int64),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def fmt(kb):
    return f"{kb/1024:8.1f} MB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["unfused", "fused"], required=True)
    ap.add_argument("--prefill-len", type=int, default=4096)
    ap.add_argument("--decode-steps", type=int, default=64)
    args = ap.parse_args()

    print(f"mode={args.mode}  prefill={args.prefill_len}  decode_steps={args.decode_steps}")
    core = ov.Core()
    if args.mode == "fused":
        register_la(core); register_cv(core)
    model = core.read_model(MODEL_XML)
    if args.mode == "fused":
        n1 = replace_gated_delta_rule_loops(model)
        n2 = replace_causal_conv1d_chains(model)
        ok = slice_lm_head_to_last_token(model)
        print(f"  rewrites: linear_attn={n1}  conv1d={n2}  lm_head_slice={ok}")
    compiled = core.compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 4})
    req = compiled.create_infer_request()

    print(f"\n  RSS after compile:        {fmt(rss_kb())}")

    # Prefill
    t0 = time.monotonic()
    with Sampler() as s_pre:
        req.infer(prefill_inputs(args.prefill_len))
    t_prefill = time.monotonic() - t0
    rss_after_prefill = rss_kb()
    print(f"  prefill time:             {t_prefill:6.1f} s")
    print(f"  RSS after prefill:        {fmt(rss_after_prefill)}")
    print(f"  PEAK RSS during prefill:  {fmt(s_pre.peak)}  (+{fmt(s_pre.peak - rss_after_prefill)} above rest)")

    # Decode
    past = args.prefill_len
    decode_peaks = []
    decode_rests = []
    decode_times = []
    overall_peak = rss_after_prefill
    for i in range(args.decode_steps):
        t1 = time.monotonic()
        with Sampler() as s_dec:
            req.infer(decode_inputs(past))
        decode_times.append(time.monotonic() - t1)
        post = rss_kb()
        decode_peaks.append(s_dec.peak)
        decode_rests.append(post)
        overall_peak = max(overall_peak, s_dec.peak)
        past += 1

    median_step_ms = sorted(decode_times)[len(decode_times)//2] * 1000
    print(f"\n  decode median step: {median_step_ms:.0f} ms  "
          f"({args.decode_steps/sum(decode_times):.1f} tok/s)")
    print(f"  RSS at end of decode:     {fmt(decode_rests[-1])}  "
          f"(growth across decode: {fmt(decode_rests[-1] - rss_after_prefill)})")
    print(f"  PEAK RSS across decode:   {fmt(max(decode_peaks))}")
    print(f"  PEAK RSS across the run:  {fmt(overall_peak)}")

    # Per-step working memory (peak above next-step resting)
    work_kb = [p - r for p, r in zip(decode_peaks, decode_rests)]
    print(f"  per-step working mem (peak above post-step rest):")
    print(f"     min={fmt(min(work_kb))}  median={fmt(sorted(work_kb)[len(work_kb)//2])}  max={fmt(max(work_kb))}")


if __name__ == "__main__":
    main()
