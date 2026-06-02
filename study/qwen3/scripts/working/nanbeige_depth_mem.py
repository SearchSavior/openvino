"""Reproduce the activation-pool + release_memory finding on a vanilla
Llama-architecture model where it should just work on stock OV — no
patches, no custom ops.

Model: Nanbeige4.1-3B (LlamaForCausalLM, 32 layers, GQA, INT4-AWQ OV;
no Loop/TensorIterator nodes — SDPA is pre-fused).

Sweeps KV depth in {128, 256, 512, 768, 1024}; pp=128, tg=32.
Measures per depth: prefill tok/s, decode tok/s, peak RSS during the
request, and floor RSS after release_memory + malloc_trim.

Llama.cpp comparison at the same shape via -d flag.
"""
import argparse
import ctypes
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
GGUF = "/tmp/nanbeige-gguf/nanbeige4.1-3b-q4_k_m.gguf"
LLAMA_BIN = "/tmp/llama.cpp/build/bin/llama-bench"

PP_LEN = 128
TG_LEN = 32
DEPTHS = [128, 256, 512, 768, 1024]
THREADS = 4
N_REPS = 3

libc = ctypes.CDLL("libc.so.6", use_errno=True)
libc.malloc_trim.argtypes = [ctypes.c_int]
libc.malloc_trim.restype = ctypes.c_int


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


class RSSSampler:
    def __init__(self, interval=0.02):
        self.interval = interval; self.peak = 0
        self._stop = threading.Event(); self._t = None
    def start(self):
        self.peak = rss_kb(); self._stop.clear()
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()
    def _loop(self):
        while not self._stop.is_set():
            r = rss_kb()
            if r > self.peak: self.peak = r
            time.sleep(self.interval)
    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=1)
        return self.peak


def feeds(lm, T, past):
    """Nanbeige uses input_ids, not inputs_embeds. attention_mask + position_ids
    are 2D, position_ids has the standard [B, T] shape (no extra axis)."""
    pid_b = lm.input("position_ids").get_partial_shape()
    # Random plausible token ids (vocab=166144)
    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=166000, size=(1, T), dtype=np.int64)
    return {
        "input_ids":      ov.Tensor(ids),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.arange(past, past + T, dtype=np.int64).reshape(1, T)),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def run_ov():
    print(f"OV version: {ov.__version__}")
    core = ov.Core()
    print("read_model…", flush=True); t0 = time.time()
    lm = core.read_model(str(OV_DIR / "openvino_model.xml"))
    print(f"  done in {time.time()-t0:.1f}s")
    print("compile_model…", flush=True); t0 = time.time()
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    print(f"  done in {time.time()-t0:.1f}s. RSS={rss_kb()/1024:.1f} MiB")

    req = compiled.create_infer_request()
    # warmup
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()
    print(f"warmup done. RSS={rss_kb()/1024:.1f} MiB")

    out = {}
    for d in DEPTHS:
        pp_best = tg_best = 0.0
        peak_max = 0
        for _ in range(N_REPS):
            req.reset_state()
            sampler = RSSSampler(); sampler.start()
            if d > 0:
                req.infer(feeds(lm, d, past=0))
            t0 = time.perf_counter()
            req.infer(feeds(lm, PP_LEN, past=d))
            t_pp = time.perf_counter() - t0
            past = d + PP_LEN
            t0 = time.perf_counter()
            for _ in range(TG_LEN):
                req.infer(feeds(lm, 1, past=past)); past += 1
            t_tg = time.perf_counter() - t0
            peak = sampler.stop()
            pp_best = max(pp_best, PP_LEN / t_pp)
            tg_best = max(tg_best, TG_LEN / t_tg)
            peak_max = max(peak_max, peak)

        # floor after release between requests
        del req
        compiled.release_memory()
        libc.malloc_trim(0)
        floor = rss_kb()
        req = compiled.create_infer_request()

        out[d] = (pp_best, tg_best, peak_max / 1024, floor / 1024)
        print(f"  d={d:>4}  pp={pp_best:6.1f}  tg={tg_best:5.2f}  "
              f"peak={peak_max/1024:7.1f} MiB  floor={floor/1024:7.1f} MiB", flush=True)

    del req; del compiled
    return out


def run_llama():
    out = {}
    for d in DEPTHS:
        cmd = [LLAMA_BIN, "-m", GGUF, "-p", str(PP_LEN), "-n", str(TG_LEN),
               "-d", str(d), "-t", str(THREADS), "-r", str(N_REPS)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        peak_kb = 0
        while p.poll() is None:
            try:
                with open(f"/proc/{p.pid}/status") as f:
                    for line in f:
                        if line.startswith("VmHWM:"):
                            peak_kb = max(peak_kb, int(line.split()[1]))
            except FileNotFoundError:
                break
            time.sleep(0.02)
        stdout, _ = p.communicate()
        pp_tps = tg_tps = 0.0
        for line in stdout.decode(errors="replace").splitlines():
            m_pp = re.search(rf"pp{PP_LEN}\s*@\s*d{d}\s*\|\s*([0-9.]+)", line)
            m_tg = re.search(rf"tg{TG_LEN}\s*@\s*d{d}\s*\|\s*([0-9.]+)", line)
            if m_pp: pp_tps = float(m_pp.group(1))
            if m_tg: tg_tps = float(m_tg.group(1))
        out[d] = (pp_tps, tg_tps, peak_kb / 1024)
        print(f"  [llama] d={d:>4}  pp={pp_tps:6.1f}  tg={tg_tps:5.2f}  peak={peak_kb/1024:7.1f} MiB",
              flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="/tmp/nanbeige_chart.png")
    ap.add_argument("--csv", default="/tmp/nanbeige.csv")
    args = ap.parse_args()

    print("=== OV stock 2026.3 — Nanbeige4.1-3B INT4-AWQ ===")
    ov_res = run_ov()
    print("\n=== llama.cpp — Nanbeige4.1-3B Q4_K_M ===")
    llama = run_llama()

    with open(args.csv, "w") as f:
        f.write("depth,ov_pp,ov_tg,ov_peak_rss,ov_floor_rss,llama_pp,llama_tg,llama_peak_rss\n")
        for d in DEPTHS:
            pp, tg, peak, floor = ov_res[d]
            lpp, ltg, lrss = llama[d]
            f.write(f"{d},{pp:.2f},{tg:.2f},{peak:.1f},{floor:.1f},{lpp:.2f},{ltg:.2f},{lrss:.1f}\n")
    print(f"\nCSV: {args.csv}")

    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(17, 5))
        ds = list(DEPTHS)

        axes[0].plot(ds, [ov_res[d][0] for d in ds], "o-", lw=2, label="OV stock 2026.3 (INT4-AWQ)")
        axes[0].plot(ds, [llama[d][0] for d in ds], "^-", color="black", lw=2.5, label="llama.cpp Q4_K_M")
        axes[0].set_title("Prefill throughput (tok/s)")

        axes[1].plot(ds, [ov_res[d][1] for d in ds], "o-", lw=2, label="OV stock")
        axes[1].plot(ds, [llama[d][1] for d in ds], "^-", color="black", lw=2.5, label="llama.cpp")
        axes[1].set_title("Decode throughput (tok/s)")

        axes[2].plot(ds, [ov_res[d][2] for d in ds], "o-", color="tab:red", lw=2,
                     label="OV peak (during request)")
        axes[2].plot(ds, [ov_res[d][3] for d in ds], "s--", color="tab:green", lw=2,
                     label="OV floor (after release_memory)")
        axes[2].plot(ds, [llama[d][2] for d in ds], "^-", color="black", lw=2.5,
                     label="llama.cpp peak")
        axes[2].set_title("Resident memory (MiB)")

        for ax in axes:
            ax.set_xlabel("KV-cache depth (tokens)")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"Nanbeige4.1-3B (LlamaForCausalLM)  pp{PP_LEN}+tg{TG_LEN}  "
                     f"{THREADS} threads  (stock OV, no patches)", y=1.02)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"Chart: {args.plot}")
    except ImportError:
        print("matplotlib missing")


if __name__ == "__main__":
    main()
