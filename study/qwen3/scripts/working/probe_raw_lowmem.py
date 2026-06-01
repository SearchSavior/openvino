"""Sweep OV config knobs trying to bring raw baseline VmHWM toward llama.cpp's
953 MiB on the same Qwen3.5-0.8B int8 / 770pp + 32tg workload.

Each invocation runs ONE config combo and reports peak RSS, prefill tok/s,
decode tok/s. Drive from a shell loop.
"""
import argparse, os, shutil, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")
PROMPT_LEN = 770
GEN_LEN = 32


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


def _prep_baseline_in_subproc():
    """Apply lm_head_slice only to keep parity with our other measurements."""
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    out = f"{WORK}/raw_probe_lowmem.xml"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{out}', '{out.replace('.xml','.bin')}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return out


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
    ap.add_argument("--label", required=True)
    ap.add_argument("--props", required=True,
                    help="comma-separated key=value list; supported keys are passed to compile_model")
    args = ap.parse_args()

    print(f"label={args.label}")
    print(f"props={args.props}")
    rss_pre = rss_kb()
    xml = _prep_baseline_in_subproc()

    # Parse props string into an ov AnyMap-like dict.
    props = {}
    for kv in args.props.split(","):
        kv = kv.strip()
        if not kv: continue
        k, v = kv.split("=", 1)
        # convert booleans + integers
        if v.lower() in ("true", "false"):
            v = (v.lower() == "true")
        else:
            try: v = int(v)
            except ValueError:
                pass
        props[k] = v
    # always pin threads
    props.setdefault("INFERENCE_NUM_THREADS", 4)

    core = ov.Core()
    t0 = time.time()
    lm = core.read_model(xml)
    compiled = core.compile_model(lm, "CPU", props)
    load = time.time() - t0
    rss_after_load = rss_kb()

    req = compiled.create_infer_request()

    sampler = RSSSampler(); sampler.start()
    req.infer(feeds_for(lm, PROMPT_LEN, past=0))
    req.infer(feeds_for(lm, 1, past=PROMPT_LEN))
    rss_warm = sampler.stop()

    del req
    req = compiled.create_infer_request()
    t0 = time.time()
    req.infer(feeds_for(lm, PROMPT_LEN, past=0))
    t_pp = time.time() - t0

    past = PROMPT_LEN
    t0 = time.time()
    for _ in range(GEN_LEN):
        req.infer(feeds_for(lm, 1, past=past))
        past += 1
    t_tg = time.time() - t0

    hwm = vmhwm_kb()
    def mib(x): return x / 1024.0
    print(f"\n=== {args.label} ===")
    print(f"  load:                  {load:6.2f}s")
    print(f"  prefill {PROMPT_LEN}:     {t_pp*1000:6.0f} ms  ({PROMPT_LEN/t_pp:6.1f} tok/s)")
    print(f"  decode  {GEN_LEN}:       {t_tg*1000:6.0f} ms  ({GEN_LEN/t_tg:6.2f} tok/s)")
    print(f"  RSS pre:               {mib(rss_pre):8.1f} MiB")
    print(f"  RSS after load:        {mib(rss_after_load):8.1f} MiB")
    print(f"  RSS peak warmup:       {mib(rss_warm):8.1f} MiB")
    print(f"  VmHWM (peak):          {mib(hwm):8.1f} MiB")


if __name__ == "__main__":
    main()
