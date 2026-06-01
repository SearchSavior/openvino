"""Single-config raw ov.Core() probe with RSS sampling. Matches probe_pa_path.py
in shape (770-token one-shot prefill + 32-token decode, INFERENCE_NUM_THREADS=4)
but bypasses ov_genai entirely. Rewrite + serialize happens in a subprocess so
the measurement process starts with ~70 MiB resident.

Driven by run_probe_raw.sh. Findings live in DISCUSSION.md, not here."""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
WORK = Path("/tmp/qwen3-work")

PROMPT_LEN = 770
GEN_LEN = 32
THREADS = 4


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def vmhwm_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


class RSSSampler:
    def __init__(self, interval_s=0.02):
        self.interval = interval_s; self.peak = 0
        self._stop = threading.Event(); self._thr = None
    def start(self):
        self.peak = rss_kb(); self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True); self._thr.start()
    def _loop(self):
        while not self._stop.is_set():
            r = rss_kb()
            if r > self.peak: self.peak = r
            time.sleep(self.interval)
    def stop(self):
        self._stop.set()
        if self._thr is not None: self._thr.join(timeout=1.0)
        return self.peak


def _prep_in_subprocess(version):
    """Rewrite + serialize per-version LM in a subprocess. Returns the path
    to the prepared LM xml. For 'baseline' we still apply slice_lm_head_to_last_token
    so the output shape matches v1/v2/v3."""
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    code = f"""
import sys
sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token

orig = '{ORIG}/openvino_language_model.xml'
out = '{WORK}/raw_probe_{version}.xml'

if '{version}' == 'baseline':
    c = ov.Core()
    m = c.read_model(orig)
    slice_lm_head_to_last_token(m)
    ov.serialize(m, out, out.replace('.xml', '.bin'))
    print('serialized: baseline (lm_head_slice only)')
else:
    from fused_linear_attn import register as rc, replace_gated_delta_rule_loops, \\
        replace_gated_delta_rule_loops_v2, replace_gated_delta_rule_loops_v3
    REW = {{'v1': replace_gated_delta_rule_loops,
            'v2': replace_gated_delta_rule_loops_v2,
            'v3': replace_gated_delta_rule_loops_v3}}['{version}']
    c = ov.Core(); rc(c)
    m = c.read_model(orig)
    n = REW(m)
    slice_lm_head_to_last_token(m)
    print(f'rewrite: replaced={{n}}')
    ov.serialize(m, out, out.replace('.xml', '.bin'))
    print(f'serialized: {version}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print("PREP STDOUT:", r.stdout); print("PREP STDERR:", r.stderr)
        raise RuntimeError(f"prep subprocess failed for {version}")
    for ln in r.stdout.strip().splitlines():
        print(f"  prep: {ln}")
    return f"{WORK}/raw_probe_{version}.xml"


def feeds_for(lm, T, past=0):
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    embeds = rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01
    return {
        "inputs_embeds": ov.Tensor(embeds),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids": ov.Tensor(np.tile(np.arange(past, past + T, dtype=np.int64).reshape(1, 1, T),
                                          (pid_b, 1, 1))),
        "beam_idx": ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["baseline", "v1", "v2", "v3"])
    args = ap.parse_args()
    print(f"openvino={ov.__version__}")
    print(f"version={args.version}")

    rss_baseline = rss_kb()
    xml = _prep_in_subprocess(args.version)
    rss_after_prep = rss_kb()

    core = ov.Core()
    if args.version != "baseline":
        if not SO.exists():
            sys.exit(f"missing {SO}; build cpp_ext first")
        core.add_extension(str(SO))

    t0 = time.time()
    lm = core.read_model(xml)
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": THREADS})
    load = time.time() - t0
    rss_after_load = rss_kb()

    req = compiled.create_infer_request()
    logits = next(o for o in compiled.outputs if "logits" in o.get_any_name())

    # Warmup: one prefill + one decode step
    sampler_warm = RSSSampler(); sampler_warm.start()
    req.infer(feeds_for(lm, PROMPT_LEN, past=0))
    req.infer(feeds_for(lm, 1, past=PROMPT_LEN))
    rss_peak_warm = sampler_warm.stop()

    # Reset state for the timed run by recreating the request.
    del req
    req = compiled.create_infer_request()

    # TIMED prefill (one-shot, like VLMPipeline)
    sampler_pp = RSSSampler(); sampler_pp.start()
    t0 = time.time()
    req.infer(feeds_for(lm, PROMPT_LEN, past=0))
    t_pp = time.time() - t0
    rss_peak_pp = sampler_pp.stop()

    # TIMED decode: GEN_LEN steps
    sampler_tg = RSSSampler(); sampler_tg.start()
    past = PROMPT_LEN
    t0 = time.time()
    for _ in range(GEN_LEN):
        req.infer(feeds_for(lm, 1, past=past))
        past += 1
    t_tg = time.time() - t0
    rss_peak_tg = sampler_tg.stop()
    rss_end = rss_kb()
    hwm = vmhwm_kb()

    def mib(x): return x / 1024.0
    print(f"\n=== raw  {args.version} ===")
    print(f"  load (read+compile):       {load:6.2f}s")
    print(f"  prefill {PROMPT_LEN}:         {t_pp*1000:6.0f} ms   ({PROMPT_LEN/t_pp:6.1f} tok/s)")
    print(f"  decode  {GEN_LEN}:           {t_tg*1000:6.0f} ms   ({GEN_LEN/t_tg:6.2f} tok/s)")
    print(f"  RSS pre-prep:              {mib(rss_baseline):8.1f} MiB")
    print(f"  RSS after subprocess prep: {mib(rss_after_prep):8.1f} MiB")
    print(f"  RSS after compile_model:   {mib(rss_after_load):8.1f} MiB")
    print(f"  RSS peak during warmup:    {mib(rss_peak_warm):8.1f} MiB")
    print(f"  RSS peak during prefill:   {mib(rss_peak_pp):8.1f} MiB")
    print(f"  RSS peak during decode:    {mib(rss_peak_tg):8.1f} MiB")
    print(f"  RSS at end:                {mib(rss_end):8.1f} MiB")
    print(f"  VmHWM (kernel peak):       {mib(hwm):8.1f} MiB")


if __name__ == "__main__":
    main()
