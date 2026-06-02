"""Measure throughput impact of release_memory + malloc_trim between requests.

Uses the v1 GatedDeltaRule custom op model (release_memory works there;
Loop model crashes on re-infer because of the upstream TensorIterator bug).

Three timing modes per workload:
  A. Single warmup + N timed runs (no release between)
  B. Single warmup + N timed runs, each preceded by release_memory + trim
  C. Each timed run from scratch (recompile_model — control for cold cache)

Workloads:
  prefill: one infer at T=PROMPT_LEN (default 512)
  decode:  one infer at T=1 (32 of them in sequence to amortize jitter)

Reports best/median/worst per (mode, workload).
"""
import argparse
import ctypes
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return 0


def prep():
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from fused_linear_attn import register as rc, replace_gated_delta_rule_loops
from lm_head_slice import slice_lm_head_to_last_token
c = ov.Core(); rc(c)
m = c.read_model('{ORIG}/openvino_language_model.xml')
n = replace_gated_delta_rule_loops(m)
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/throughput.xml', '{WORK}/throughput.bin')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def make_feeds(lm, T, past=0):
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds":  ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.tile(np.arange(past, past + T, dtype=np.int64).reshape(1, 1, T),
                                            (pid_b, 1, 1))),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def time_one_prefill(req, lm, T):
    t0 = time.perf_counter()
    req.infer(make_feeds(lm, T, past=0))
    return time.perf_counter() - t0


def time_one_decode_block(req, lm, n_tokens, past_start=0):
    """Time n_tokens individual T=1 infers (must call reset_state externally)."""
    past = past_start
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        req.infer(make_feeds(lm, 1, past=past))
        past += 1
    return time.perf_counter() - t0


def fmt(times):
    if not times:
        return ""
    best = min(times) * 1000
    med = statistics.median(times) * 1000
    worst = max(times) * 1000
    return f"best={best:6.0f}ms  med={med:6.0f}ms  worst={worst:6.0f}ms"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--decode-len", type=int, default=32)
    ap.add_argument("--n-runs", type=int, default=5)
    args = ap.parse_args()

    prep()
    core = ov.Core()
    core.add_extension(str(SO))
    lm = core.read_model(f"{WORK}/throughput.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})
    print(f"compiled. RSS={rss():.1f} MiB")

    req = compiled.create_infer_request()

    # warmup
    req.infer(make_feeds(lm, args.prompt_len))
    req.reset_state()
    print(f"warmup done. RSS={rss():.1f} MiB")

    # === A. No release between runs ===
    print(f"\n=== A. No release between runs (n={args.n_runs}) ===")
    pp_no_release = []
    tg_no_release = []
    for i in range(args.n_runs):
        req.reset_state()
        pp = time_one_prefill(req, lm, args.prompt_len)
        tg = time_one_decode_block(req, lm, args.decode_len, past_start=args.prompt_len)
        pp_no_release.append(pp); tg_no_release.append(tg)
    print(f"  prefill ({args.prompt_len}tok):  {fmt(pp_no_release)}")
    print(f"  decode  ({args.decode_len}tok):  {fmt(tg_no_release)}")
    print(f"  RSS after section: {rss():.1f} MiB")

    # === B. release_memory + trim before EVERY run ===
    print(f"\n=== B. release_memory + trim before every run (n={args.n_runs}) ===")
    pp_with_release = []
    tg_with_release = []
    for i in range(args.n_runs):
        del req
        compiled.release_memory()
        libc.malloc_trim(0)
        req = compiled.create_infer_request()
        req.reset_state()
        pp = time_one_prefill(req, lm, args.prompt_len)
        tg = time_one_decode_block(req, lm, args.decode_len, past_start=args.prompt_len)
        pp_with_release.append(pp); tg_with_release.append(tg)
    print(f"  prefill ({args.prompt_len}tok):  {fmt(pp_with_release)}")
    print(f"  decode  ({args.decode_len}tok):  {fmt(tg_with_release)}")
    print(f"  RSS after section: {rss():.1f} MiB")

    # === Deltas ===
    print(f"\n=== B - A delta (cost of release_memory per request) ===")
    if pp_with_release and pp_no_release:
        a_pp = statistics.median(pp_no_release) * 1000
        b_pp = statistics.median(pp_with_release) * 1000
        print(f"  prefill median:  A {a_pp:6.0f}ms  B {b_pp:6.0f}ms  delta {b_pp-a_pp:+6.0f}ms  "
              f"({(b_pp-a_pp)/a_pp*100:+.1f}%)")
        a_tg = statistics.median(tg_no_release) * 1000
        b_tg = statistics.median(tg_with_release) * 1000
        print(f"  decode  median:  A {a_tg:6.0f}ms  B {b_tg:6.0f}ms  delta {b_tg-a_tg:+6.0f}ms  "
              f"({(b_tg-a_tg)/a_tg*100:+.1f}%)")

    # === C. Cost of just the first prefill after release (warmup vs steady) ===
    print(f"\n=== C. First-vs-subsequent prefill cost after release ===")
    del req
    compiled.release_memory()
    libc.malloc_trim(0)
    req = compiled.create_infer_request()
    first = time_one_prefill(req, lm, args.prompt_len)
    print(f"  prefill #1 (right after release): {first*1000:6.0f}ms")
    req.reset_state()
    second = time_one_prefill(req, lm, args.prompt_len)
    print(f"  prefill #2 (after #1):           {second*1000:6.0f}ms   "
          f"(#1 - #2 = {(first-second)*1000:+.0f}ms one-shot regrow cost)")
    req.reset_state()
    third = time_one_prefill(req, lm, args.prompt_len)
    print(f"  prefill #3 (warm steady):        {third*1000:6.0f}ms")


if __name__ == "__main__":
    main()
