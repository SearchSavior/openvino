"""Sweep KV-cache depth and measure pp128 + tg32 with and without
release_memory between requests. Compare to llama-bench at the same
shape via its -d flag. Emit a CSV and a chart.

Usage:
  python3 depth_sweep.py --plot /tmp/depth_chart.png
"""
import argparse
import ctypes
import json
import re
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
LLAMA_BIN = "/tmp/llama.cpp/build/bin/llama-bench"
GGUF = "/tmp/qwen35-0.8b-Q8_0.gguf"

PP_LEN = 128
TG_LEN = 32
DEPTHS = [128, 256, 512, 768, 1024]
N_REPS = 3
THREADS = 4

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


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
ov.serialize(m, '{WORK}/depthsweep.xml', '{WORK}/depthsweep.bin')
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def feeds(lm, T, past):
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


def time_at_depth(req, lm, depth, pp, tg):
    """Build KV to `depth`, then time pp prefill and tg decode."""
    if depth > 0:
        req.infer(feeds(lm, depth, past=0))  # build context, untimed
    t0 = time.perf_counter()
    req.infer(feeds(lm, pp, past=depth))
    t_pp = time.perf_counter() - t0
    past = depth + pp
    t0 = time.perf_counter()
    for _ in range(tg):
        req.infer(feeds(lm, 1, past=past))
        past += 1
    t_tg = time.perf_counter() - t0
    return t_pp, t_tg


def bench_ov(release_each):
    """Returns: dict[depth] = (pp_tok_per_s, tg_tok_per_s) using best of N_REPS."""
    prep()
    core = ov.Core()
    core.add_extension(str(SO))
    lm = core.read_model(f"{WORK}/depthsweep.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    req = compiled.create_infer_request()

    # warmup
    req.infer(feeds(lm, PP_LEN, past=0))
    req.reset_state()

    out = {}
    for d in DEPTHS:
        pp_times, tg_times = [], []
        for _ in range(N_REPS):
            if release_each:
                del req
                compiled.release_memory()
                libc.malloc_trim(0)
                req = compiled.create_infer_request()
            req.reset_state()
            t_pp, t_tg = time_at_depth(req, lm, d, PP_LEN, TG_LEN)
            pp_times.append(t_pp); tg_times.append(t_tg)
        # best (fastest) iteration -> highest tok/s
        out[d] = (PP_LEN / min(pp_times), TG_LEN / min(tg_times))
        print(f"  d={d:>4}  pp={out[d][0]:6.1f} tok/s  tg={out[d][1]:5.2f} tok/s")
    del req
    return out


def bench_llama():
    """llama-bench -d <depths> reports pp128/tg32 at each depth.
    We call it once per depth for clean output."""
    out = {}
    for d in DEPTHS:
        cmd = [LLAMA_BIN, "-m", GGUF, "-p", str(PP_LEN), "-n", str(TG_LEN),
               "-d", str(d), "-t", str(THREADS), "-r", str(N_REPS)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        # parse lines like '| ... pp128 @ d128 |  ...  tok/s |'
        pp_tps = None; tg_tps = None
        for line in r.stdout.splitlines():
            m_pp = re.search(rf"pp{PP_LEN}\s*@\s*d{d}\s*\|\s*([0-9.]+)", line)
            m_tg = re.search(rf"tg{TG_LEN}\s*@\s*d{d}\s*\|\s*([0-9.]+)", line)
            if m_pp: pp_tps = float(m_pp.group(1))
            if m_tg: tg_tps = float(m_tg.group(1))
        out[d] = (pp_tps or 0.0, tg_tps or 0.0)
        print(f"  d={d:>4}  pp={out[d][0]:6.1f} tok/s  tg={out[d][1]:5.2f} tok/s")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="/tmp/depth_chart.png")
    ap.add_argument("--csv", default="/tmp/depth_sweep.csv")
    args = ap.parse_args()

    print(f"\n=== OV with custom-op model, NO release between runs ===")
    ov_keep = bench_ov(release_each=False)
    print(f"\n=== OV with custom-op model, release_memory before each depth ===")
    ov_rel = bench_ov(release_each=True)
    print(f"\n=== llama.cpp Q8_0 ===")
    llama = bench_llama()

    # CSV
    with open(args.csv, "w") as f:
        f.write("depth,ov_no_release_pp,ov_no_release_tg,ov_release_pp,ov_release_tg,llama_pp,llama_tg\n")
        for d in DEPTHS:
            f.write(f"{d},{ov_keep[d][0]:.2f},{ov_keep[d][1]:.2f},"
                    f"{ov_rel[d][0]:.2f},{ov_rel[d][1]:.2f},"
                    f"{llama[d][0]:.2f},{llama[d][1]:.2f}\n")
    print(f"\nCSV: {args.csv}")

    # Chart
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        ds = list(DEPTHS)
        ax1.plot(ds, [ov_keep[d][0] for d in ds], "o-", label="OV (no release)", lw=2)
        ax1.plot(ds, [ov_rel[d][0]  for d in ds], "s--", label="OV (release each)", lw=2)
        ax1.plot(ds, [llama[d][0]   for d in ds], "^-",  label="llama.cpp Q8_0", lw=2)
        ax1.set_xlabel("KV-cache depth (tokens)")
        ax1.set_ylabel("Prefill throughput (tok/s)")
        ax1.set_title(f"Prefill ({PP_LEN} tok) vs context depth")
        ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.plot(ds, [ov_keep[d][1] for d in ds], "o-", label="OV (no release)", lw=2)
        ax2.plot(ds, [ov_rel[d][1]  for d in ds], "s--", label="OV (release each)", lw=2)
        ax2.plot(ds, [llama[d][1]   for d in ds], "^-",  label="llama.cpp Q8_0", lw=2)
        ax2.set_xlabel("KV-cache depth (tokens)")
        ax2.set_ylabel("Decode throughput (tok/s)")
        ax2.set_title(f"Decode ({TG_LEN} tok) vs context depth")
        ax2.grid(True, alpha=0.3); ax2.legend()

        fig.suptitle(f"Qwen3.5-0.8B INT8/Q8_0 — {THREADS} threads, best of {N_REPS}", y=1.02)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"Chart: {args.plot}")
    except ImportError:
        print("matplotlib not available; install with `pip install matplotlib`")


if __name__ == "__main__":
    main()
