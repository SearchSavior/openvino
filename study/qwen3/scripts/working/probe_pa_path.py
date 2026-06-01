"""Run one (config, attention_backend) pair per process; results go to stdout.

Driven by run_probe_pa_path.sh. Re-uses the fused-dir convention from
bench_genai.py."""
import argparse
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "kernels"))

import numpy as np
import openvino as ov
import openvino_genai as ov_genai
from PIL import Image

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
SO = Path(__file__).resolve().parents[2] / "cpp_ext/build/libqwen3_ov_ext.so"
WORK = Path("/tmp/qwen3-work")
CACHE_DIR = WORK / "genai-cache-probe"
IMG = "/tmp/llama.cpp/media/llama1-logo.png"


def load_image(path):
    img = Image.open(path).convert("RGB")
    return ov.Tensor(np.asarray(img, dtype=np.uint8)[None])


def rss_kb():
    """Return current VmRSS (resident set size) in KiB by reading /proc/self/status."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def vmhwm_kb():
    """Peak resident-set high-water-mark recorded by the kernel for this process."""
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    return 0


class RSSSampler:
    """Background thread that polls VmRSS every `interval_s` and keeps the max."""
    def __init__(self, interval_s=0.02):
        self.interval = interval_s
        self.peak = 0
        self._stop = threading.Event()
        self._thr = None

    def start(self):
        self.peak = rss_kb()
        self._stop.clear()
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _loop(self):
        while not self._stop.is_set():
            r = rss_kb()
            if r > self.peak:
                self.peak = r
            time.sleep(self.interval)

    def stop(self):
        self._stop.set()
        if self._thr is not None:
            self._thr.join(timeout=1.0)
        return self.peak


def _prep_in_subprocess(version):
    """Run the rewrite + serialize in a subprocess so its peak RSS does not
    contaminate the measurement of VLMPipeline-only memory usage."""
    code = f"""
import sys
from pathlib import Path
sys.path.insert(0, '{Path(__file__).resolve().parents[2] / "kernels"}')
import openvino as ov
from fused_linear_attn import register as rc, replace_gated_delta_rule_loops, \\
    replace_gated_delta_rule_loops_v2, replace_gated_delta_rule_loops_v3
from lm_head_slice import slice_lm_head_to_last_token

REWRITES = {{
    'v1': replace_gated_delta_rule_loops,
    'v2': replace_gated_delta_rule_loops_v2,
    'v3': replace_gated_delta_rule_loops_v3,
}}

orig = Path('{ORIG}')
fused = Path('{WORK}') / 'qwen35-0.8b-int8-fused-{version}probe'
import shutil
if fused.exists():
    shutil.rmtree(fused)
fused.mkdir(parents=True, exist_ok=True)
LM = {{'openvino_language_model.xml', 'openvino_language_model.bin'}}
for f in orig.iterdir():
    if f.name in LM: continue
    (fused / f.name).symlink_to(f, target_is_directory=f.is_dir())
c = ov.Core(); rc(c)
m = c.read_model(str(orig / 'openvino_language_model.xml'))
n = REWRITES['{version}'](m)
slice_lm_head_to_last_token(m)
print(f'rewrite: replaced={{n}}')
ov.serialize(m, str(fused / 'openvino_language_model.xml'),
             str(fused / 'openvino_language_model.bin'))
print(f'serialized -> {{fused}}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        print("PREP STDOUT:", r.stdout)
        print("PREP STDERR:", r.stderr)
        raise RuntimeError(f"prep subprocess failed for {version}")
    for ln in r.stdout.strip().splitlines():
        print(f"  prep: {ln}")
    return WORK / f"qwen35-0.8b-int8-fused-{version}probe"


def run(model_dir, props, label):
    if CACHE_DIR.exists():
        shutil.rmtree(CACHE_DIR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    props = dict(props)
    props["CACHE_DIR"] = str(CACHE_DIR)
    props["INFERENCE_NUM_THREADS"] = 4
    image = load_image(IMG)
    cfg = ov_genai.GenerationConfig()
    cfg.max_new_tokens = 32
    cfg.do_sample = False

    rss_baseline = rss_kb()

    t0 = time.time()
    vlm = ov_genai.VLMPipeline(str(model_dir), "CPU", **props)
    load = time.time() - t0
    rss_after_load = rss_kb()

    # Warmup. Capture peak RSS during warmup -- this is when the plugin
    # actually allocates per-shape compute buffers for prefill + decode.
    sampler = RSSSampler(interval_s=0.02)
    sampler.start()
    vlm.generate("Describe this image.", images=[image], generation_config=cfg)
    rss_peak_warmup = sampler.stop()
    rss_after_warmup = rss_kb()

    # Timed run. We also peak-sample here in case warmup didn't hit max.
    sampler2 = RSSSampler(interval_s=0.02)
    sampler2.start()
    out = vlm.generate("Describe this image.", images=[image], generation_config=cfg)
    rss_peak_timed = sampler2.stop()
    rss_after_timed = rss_kb()
    pm = out.perf_metrics
    ttft = pm.get_ttft().mean / 1000.0
    tput = pm.get_throughput().mean
    n_in = pm.get_num_input_tokens()
    text = str(out).strip().replace("\n", " ")[:80]

    # Kernel-recorded peak across the whole process.
    hwm = vmhwm_kb()

    def mib(x): return x / 1024.0
    print(f"\n=== {label} ===")
    print(f"  load:                  {load:6.2f}s")
    print(f"  in tokens:             {n_in}")
    print(f"  TTFT:                  {ttft:6.3f}s   ({n_in/ttft:6.1f} tok/s)")
    print(f"  decode:                {tput:6.2f} tok/s")
    print(f"  RSS baseline (pre-load): {mib(rss_baseline):8.1f} MiB")
    print(f"  RSS after load:          {mib(rss_after_load):8.1f} MiB   (Δ {mib(rss_after_load - rss_baseline):+.1f})")
    print(f"  RSS peak during warmup:  {mib(rss_peak_warmup):8.1f} MiB")
    print(f"  RSS peak during timed:   {mib(rss_peak_timed):8.1f} MiB")
    print(f"  RSS at end of timed:     {mib(rss_after_timed):8.1f} MiB")
    print(f"  VmHWM (kernel peak):     {mib(hwm):8.1f} MiB")
    print(f"  output:                {text!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, choices=["baseline", "v1", "v2", "v3"])
    ap.add_argument("--backend", required=True, choices=["pa", "sdpa"])
    args = ap.parse_args()
    print(f"openvino={ov.__version__}  openvino_genai={ov_genai.__version__}")
    print(f"version={args.version}  backend={args.backend}")

    if args.version == "baseline":
        model_dir = ORIG
        ext_props = {}
    else:
        model_dir = _prep_in_subprocess(args.version)
        ext_props = {"extensions": [str(SO)]}

    if args.backend == "sdpa":
        ext_props["ATTENTION_BACKEND"] = "SDPA"

    run(model_dir, ext_props, f"{args.version}_{args.backend}")


if __name__ == "__main__":
    main()
