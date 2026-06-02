"""Depth sweep with MEMORY as a third axis.

Question: does releasing memory between requests pull OV's RSS toward
llama.cpp's, and at what throughput cost?

release_memory frees the activation pool, which is rebuilt on the next
infer. It can only be called BETWEEN requests (calling it mid-decode would
wipe KV state). So there are two distinct memory numbers per request:

  peak_rss  - high-water mark DURING the prefill+decode (pool fully grown)
  floor_rss - resident set AFTER release_memory+malloc_trim between requests

For each KV depth in {128,256,512,768,1024} on the OV custom-op model we
measure pp128 tok/s, tg32 tok/s, peak_rss, and floor_rss. Two OV policies:

  never_release  - floor == peak (pool kept between requests)
  release_each   - release after each request -> low floor, peak unchanged

llama.cpp (Q8_0) is measured at the same shape via its -d flag; its peak
RSS is sampled from /proc. llama.cpp mmaps weights so its peak ~= floor.

Emits CSV + a 3-panel chart: prefill tok/s, decode tok/s, and RSS
(showing OV peak, OV floor-after-release, and llama.cpp) all vs depth.
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

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
LLAMA_BIN = "/tmp/llama.cpp/build/bin/llama-bench"
GGUF = "/tmp/qwen35-0.8b-Q8_0.gguf"

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


def prep():
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from fused_linear_attn import register as rc, replace_gated_delta_rule_loops
from lm_head_slice import slice_lm_head_to_last_token
c = ov.Core(); rc(c)
m = c.read_model('{ORIG}/openvino_language_model.xml')
replace_gated_delta_rule_loops(m)
slice_lm_head_to_last_token(m)
ov.serialize(m, '{WORK}/depthmem.xml', '{WORK}/depthmem.bin')
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


def run_ov():
    """dict[depth] = (pp_tps, tg_tps, peak_rss_mib, floor_rss_mib)."""
    prep()
    core = ov.Core(); core.add_extension(str(SO))
    lm = core.read_model(f"{WORK}/depthmem.xml")
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    req = compiled.create_infer_request()
    req.infer(feeds(lm, PP_LEN, past=0)); req.reset_state()  # warmup

    out = {}
    for d in DEPTHS:
        pp_best = tg_best = 0.0
        peak_max = 0
        for _ in range(N_REPS):
            req.reset_state()
            sampler = RSSSampler(); sampler.start()
            if d > 0:
                req.infer(feeds(lm, d, past=0))           # build context (untimed)
            t0 = time.perf_counter()
            req.infer(feeds(lm, PP_LEN, past=d))           # timed prefill
            t_pp = time.perf_counter() - t0
            past = d + PP_LEN
            t0 = time.perf_counter()
            for _ in range(TG_LEN):                        # timed decode
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
              f"peak={peak_max/1024:7.1f} MiB  floor_after_release={floor/1024:7.1f} MiB")
    del req
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
        print(f"  [llama] d={d:>4}  pp={pp_tps:6.1f}  tg={tg_tps:5.2f}  peak={peak_kb/1024:7.1f} MiB")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", default="/tmp/depth_mem_chart.png")
    ap.add_argument("--csv", default="/tmp/depth_mem.csv")
    args = ap.parse_args()

    print("=== OV custom-op model ===")
    ov_res = run_ov()
    print("=== llama.cpp Q8_0 ===")
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

        axes[0].plot(ds, [ov_res[d][0] for d in ds], "o-", lw=2, label="OV")
        axes[0].plot(ds, [llama[d][0] for d in ds], "^-", color="black", lw=2.5, label="llama.cpp")
        axes[0].set_title("Prefill throughput (tok/s)")

        axes[1].plot(ds, [ov_res[d][1] for d in ds], "o-", lw=2, label="OV")
        axes[1].plot(ds, [llama[d][1] for d in ds], "^-", color="black", lw=2.5, label="llama.cpp")
        axes[1].set_title("Decode throughput (tok/s)")

        axes[2].plot(ds, [ov_res[d][2] for d in ds], "o-", color="tab:red", lw=2,
                     label="OV peak (during request)")
        axes[2].plot(ds, [ov_res[d][3] for d in ds], "s--", color="tab:green", lw=2,
                     label="OV floor (after release)")
        axes[2].plot(ds, [llama[d][2] for d in ds], "^-", color="black", lw=2.5,
                     label="llama.cpp peak")
        axes[2].set_title("Resident memory (MiB)")

        for ax in axes:
            ax.set_xlabel("KV-cache depth (tokens)")
            ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
        fig.suptitle(f"Qwen3.5-0.8B — pp{PP_LEN}+tg{TG_LEN}, {THREADS} threads "
                     f"(OV=Debug+L1, custom-op model; best of {N_REPS})", y=1.02)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=120, bbox_inches="tight")
        print(f"Chart: {args.plot}")
    except ImportError:
        print("matplotlib missing")


if __name__ == "__main__":
    main()
