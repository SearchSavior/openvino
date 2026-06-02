"""Add a third memory measurement to the Nanbeige sweep: floor RSS after
reset_state() + release_memory() + malloc_trim(0). This isolates whether
the depth-scaling floor we saw is the KV state Variables (which reset_state
clears) or something else.

Reads the existing /tmp/nan.csv, adds an `ov_reset_floor_rss` column,
re-emits the chart with all four memory series (OV peak, OV release-only
floor, OV reset+release floor, llama.cpp peak).
"""
import argparse
import csv
import ctypes
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

OV_DIR = Path("/tmp/nanbeige-ov")
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


def feeds(lm, T, past):
    pid_b = lm.input("position_ids").get_partial_shape()
    rng = np.random.default_rng(0)
    ids = rng.integers(low=1, high=166000, size=(1, T), dtype=np.int64)
    return {
        "input_ids":      ov.Tensor(ids),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids":   ov.Tensor(np.arange(past, past + T, dtype=np.int64).reshape(1, T)),
        "beam_idx":       ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-csv",  default="/tmp/nan.csv")
    ap.add_argument("--out-csv", default="/tmp/nan_v2.csv")
    ap.add_argument("--plot",    default="/tmp/nan_v2.png")
    args = ap.parse_args()

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

    reset_floors = {}
    for d in DEPTHS:
        # Do one full request at this depth to populate KV state to it
        req.reset_state()
        if d > 0:
            req.infer(feeds(lm, d, past=0))
        req.infer(feeds(lm, PP_LEN, past=d))
        past = d + PP_LEN
        for _ in range(TG_LEN):
            req.infer(feeds(lm, 1, past=past)); past += 1

        # Now: reset_state + del req + release_memory + trim
        req.reset_state()
        del req
        compiled.release_memory()
        libc.malloc_trim(0)
        floor = rss_kb()
        reset_floors[d] = floor / 1024
        print(f"  d={d:>4}  floor after reset_state+release: {floor/1024:7.1f} MiB", flush=True)
        # recreate request for next depth
        req = compiled.create_infer_request()

    del req; del compiled

    # Merge with existing CSV
    rows = list(csv.DictReader(open(args.in_csv)))
    fieldnames = list(rows[0].keys()) + ["ov_reset_floor_rss"]
    with open(args.out_csv, "w") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            d = int(r["depth"])
            r["ov_reset_floor_rss"] = f"{reset_floors[d]:.1f}"
            w.writerow(r)
    print(f"\nCSV: {args.out_csv}")

    # Re-chart with the new series
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = list(csv.DictReader(open(args.out_csv)))
    ds = [int(r["depth"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    axes[0].plot(ds, [float(r["ov_pp"])     for r in rows], "o-", lw=2, label="OV stock 2026.3")
    axes[0].plot(ds, [float(r["llama_pp"])  for r in rows], "^-", color="black", lw=2.5, label="llama.cpp Q4_K_M")
    axes[0].set_title("Prefill throughput (tok/s)")

    axes[1].plot(ds, [float(r["ov_tg"])     for r in rows], "o-", lw=2, label="OV stock")
    axes[1].plot(ds, [float(r["llama_tg"])  for r in rows], "^-", color="black", lw=2.5, label="llama.cpp")
    axes[1].set_title("Decode throughput (tok/s)")

    axes[2].plot(ds, [float(r["ov_peak_rss"])        for r in rows], "o-",  color="tab:red",    lw=2,
                 label="OV peak (during request)")
    axes[2].plot(ds, [float(r["ov_floor_rss"])       for r in rows], "s--", color="tab:orange", lw=2,
                 label="OV floor (release only)")
    axes[2].plot(ds, [float(r["ov_reset_floor_rss"]) for r in rows], "D-.", color="tab:green",  lw=2,
                 label="OV floor (reset_state + release)")
    axes[2].plot(ds, [float(r["llama_peak_rss"])     for r in rows], "^-",  color="black",      lw=2.5,
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


if __name__ == "__main__":
    main()
