"""Break OV peak RSS into 'weights paged in' vs 'compute buffer' contributions
by sweeping prompt length and reading /proc/self/maps before+after warmup."""
import argparse, subprocess, sys, threading, time
from pathlib import Path
import numpy as np
import openvino as ov

ORIG = Path("/tmp/qwen3-work/qwen35-0.8b-int8")
WORK = Path("/tmp/qwen3-work")


def rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"): return int(line.split()[1])
    return 0


def vmhwm_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmHWM:"): return int(line.split()[1])
    return 0


def maps_summary():
    """Sum private (anonymous) pages vs file-backed pages from /proc/self/smaps_rollup."""
    out = {}
    try:
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                k, _, rest = line.partition(":")
                rest = rest.strip()
                if rest.endswith(" kB"):
                    out[k.strip()] = int(rest[:-3])
    except FileNotFoundError:
        pass
    return out


def _prep():
    kdir = Path(__file__).resolve().parents[2] / "kernels"
    out = f"{WORK}/raw_breakdown.xml"
    code = f"""
import sys; sys.path.insert(0, '{kdir}')
import openvino as ov
from lm_head_slice import slice_lm_head_to_last_token
m = ov.Core().read_model('{ORIG}/openvino_language_model.xml')
slice_lm_head_to_last_token(m)
ov.serialize(m, '{out}', '{out.replace('.xml','.bin')}')
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0: raise RuntimeError(r.stderr)
    return out


def feeds(lm, T, past=0):
    pid_b = lm.input("position_ids").get_partial_shape()[0].get_length()
    hidden = lm.input("inputs_embeds").get_partial_shape()[2].get_length()
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": ov.Tensor(rng.standard_normal((1, T, hidden), dtype=np.float32) * 0.01),
        "attention_mask": ov.Tensor(np.ones((1, past + T), dtype=np.int64)),
        "position_ids": ov.Tensor(np.tile(np.arange(past, past + T, dtype=np.int64).reshape(1, 1, T),
                                          (pid_b, 1, 1))),
        "beam_idx": ov.Tensor(np.zeros((1,), dtype=np.int32)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, required=True)
    args = ap.parse_args()
    rss_pre = rss_kb()
    xml = _prep()
    print(f"\n--- T_prefill={args.T} ---")
    print(f"  RSS pre-prep:        {rss_pre/1024:8.1f} MiB")

    core = ov.Core()
    lm = core.read_model(xml)
    compiled = core.compile_model(lm, "CPU", {"INFERENCE_NUM_THREADS": 4})

    rss_load = rss_kb()
    m_load = maps_summary()
    print(f"  RSS after compile:   {rss_load/1024:8.1f} MiB")
    print(f"  smaps after compile: Rss={m_load.get('Rss', 0)/1024:.1f}  "
          f"Pss={m_load.get('Pss', 0)/1024:.1f}  "
          f"Anon={m_load.get('Anonymous', 0)/1024:.1f}  "
          f"File={m_load.get('File', 0)/1024:.1f} MiB")

    req = compiled.create_infer_request()
    req.infer(feeds(lm, args.T, past=0))     # prefill
    req.infer(feeds(lm, 1, past=args.T))     # one decode

    rss_warm = rss_kb()
    hwm = vmhwm_kb()
    m_warm = maps_summary()
    print(f"  RSS after warmup:    {rss_warm/1024:8.1f} MiB")
    print(f"  smaps after warmup:  Rss={m_warm.get('Rss',0)/1024:.1f}  "
          f"Pss={m_warm.get('Pss',0)/1024:.1f}  "
          f"Anon={m_warm.get('Anonymous',0)/1024:.1f}  "
          f"File={m_warm.get('File',0)/1024:.1f} MiB")
    print(f"  Δ from compile->warmup: total +{(rss_warm-rss_load)/1024:.1f} MiB  "
          f"of which Anon +{(m_warm.get('Anonymous',0)-m_load.get('Anonymous',0))/1024:.1f}  "
          f"File +{(m_warm.get('File',0)-m_load.get('File',0))/1024:.1f}")
    print(f"  VmHWM:               {hwm/1024:8.1f} MiB")


if __name__ == "__main__":
    main()
