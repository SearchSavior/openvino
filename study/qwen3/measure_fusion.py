"""
Memory and latency comparison: original Loop fallback vs fused GatedDeltaRule.

Runs 4096-token prefill through both versions of the Qwen3.5-VL 0.8B language
model. Samples /proc/self/status at 10 ms intervals to catch peak resident set,
records the post-call resting RSS, and times the prefill call.

Usage:
    python measure_fusion.py [--model <ir-dir>] [--prompt-len 4096]
"""
import argparse
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).parent))
from fused_linear_attn import register, replace_gated_delta_rule_loops  # noqa: E402

HIDDEN = 1024
SAMPLE_INTERVAL_S = 0.010


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
            time.sleep(SAMPLE_INTERVAL_S)

    def __enter__(self):
        self.peak = rss_kb()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def prefill_inputs(seq, hidden):
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, hidden)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def run_one(label, model_xml, prompt_len, apply_fusion):
    print(f"\n=== {label} ===")

    rss_before = rss_kb()
    print(f"  RSS at process entry: {rss_before/1024:7.1f} MB")

    core = ov.Core()
    if apply_fusion:
        register(core)
    model = core.read_model(model_xml)
    if apply_fusion:
        n = replace_gated_delta_rule_loops(model)
        print(f"  replaced {n} Loop(s) with GatedDeltaRule")

    compiled = core.compile_model(model, "CPU", {
        "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4,
    })
    rss_after_compile = rss_kb()
    print(f"  RSS after compile:    {rss_after_compile/1024:7.1f} MB"
          f"  (+{(rss_after_compile-rss_before)/1024:6.1f} MB)")

    req = compiled.create_infer_request()
    inp = prefill_inputs(prompt_len, HIDDEN)

    t0 = time.monotonic()
    with Sampler() as s:
        req.infer(inp)
    elapsed = time.monotonic() - t0
    rss_after_prefill = rss_kb()

    print(f"  prefill time:         {elapsed:7.1f} s")
    print(f"  RSS after prefill:    {rss_after_prefill/1024:7.1f} MB"
          f"  (+{(rss_after_prefill-rss_after_compile)/1024:6.1f} MB vs compile)")
    print(f"  PEAK RSS in prefill:  {s.peak/1024:7.1f} MB"
          f"  (+{(s.peak-rss_after_compile)/1024:6.1f} MB transient)")

    return {
        "after_compile_mb": rss_after_compile / 1024,
        "after_prefill_mb": rss_after_prefill / 1024,
        "peak_prefill_mb": s.peak / 1024,
        "prefill_s": elapsed,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml")
    ap.add_argument("--prompt-len", type=int, default=4096)
    ap.add_argument("--mode", choices=["baseline", "fused"], required=True,
                    help="run only one mode at a time so OOM from the other doesn't pollute the measurement")
    args = ap.parse_args()

    print(f"Qwen3.5-VL 0.8B INT8, prefill={args.prompt_len} tokens, 4 threads, CPU plugin, mode={args.mode}")
    if args.mode == "baseline":
        run_one("BASELINE (Loop fallback)", args.model, args.prompt_len, apply_fusion=False)
    else:
        run_one("FUSED (GatedDeltaRule)", args.model, args.prompt_len, apply_fusion=True)


if __name__ == "__main__":
    main()
