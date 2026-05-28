"""
Probe where the CPU plugin's resident memory actually lives.

For a single prompt length (passed via --prompt-len), reports:
  - RSS at process start
  - RSS after read_model    (model graph + weight constants)
  - RSS after compile_model (plugin-allocated state)
  - RSS just before infer   (idle state)
  - PEAK RSS during infer   (working memory)
  - RSS after infer         (resting after release)
  - RSS after gc + sleep    (anything Python could give back)
  - glibc mallinfo2 breakdown of heap, mmaps, and freed-but-uncommitted regions
  - /proc/self/status anon vs file-backed bytes

Run multiple times with different --prompt-len to see what scales with sequence.
"""
import argparse
import ctypes
import gc
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).parent))
from fused_linear_attn import register as register_la, replace_gated_delta_rule_loops  # noqa: E402
from fused_conv1d import register as register_cv, replace_causal_conv1d_chains  # noqa: E402
from lm_head_slice import slice_lm_head_to_last_token  # noqa: E402

HIDDEN = 1024
SAMPLE_INTERVAL_S = 0.010


class Mallinfo2(ctypes.Structure):
    _fields_ = [
        ("arena", ctypes.c_size_t),
        ("ordblks", ctypes.c_size_t),
        ("smblks", ctypes.c_size_t),
        ("hblks", ctypes.c_size_t),
        ("hblkhd", ctypes.c_size_t),
        ("usmblks", ctypes.c_size_t),
        ("fsmblks", ctypes.c_size_t),
        ("uordblks", ctypes.c_size_t),
        ("fordblks", ctypes.c_size_t),
        ("keepcost", ctypes.c_size_t),
    ]


_libc = ctypes.CDLL("libc.so.6", use_errno=True)
_libc.mallinfo2.restype = Mallinfo2


def mallinfo():
    """glibc heap stats.

    arena   = total bytes in arenas (sbrk)
    hblkhd  = total bytes in mmap'd regions (large allocs)
    uordblks = bytes currently in use (allocated to caller)
    fordblks = bytes free in arenas (allocated to glibc, not given back to kernel)
    keepcost = top-of-arena that could be released via malloc_trim
    """
    m = _libc.mallinfo2()
    return {k: getattr(m, k) for k, _ in Mallinfo2._fields_}


def malloc_trim():
    """Force glibc to release free top-of-heap to the kernel."""
    _libc.malloc_trim.restype = ctypes.c_int
    _libc.malloc_trim.argtypes = [ctypes.c_size_t]
    return _libc.malloc_trim(0)


def status_bytes(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1]) * 1024  # kB → bytes
    return -1


def rss():
    return status_bytes("VmRSS")


def anon_rss():
    return status_bytes("RssAnon")


def file_rss():
    return status_bytes("RssFile")


def fmt(b):
    return f"{b/1024/1024:9.1f} MB"


class Sampler:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            self.peak = max(self.peak, rss())
            time.sleep(SAMPLE_INTERVAL_S)

    def __enter__(self):
        self.peak = rss()
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set(); self._t.join()


def prefill_inputs(seq, hidden=HIDDEN):
    rng = np.random.default_rng(0)
    return {
        "inputs_embeds": (rng.standard_normal((1, seq, hidden)) * 0.02).astype(np.float32),
        "attention_mask": np.ones((1, seq), dtype=np.int64),
        "position_ids": np.tile(np.arange(seq, dtype=np.int64).reshape(1, 1, seq), (4, 1, 1)),
        "beam_idx": np.zeros((1,), dtype=np.int32),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/tmp/qwen3-work/qwen35-0.8b-int8/openvino_language_model.xml")
    ap.add_argument("--prompt-len", type=int, default=512)
    ap.add_argument("--fuse", action="store_true", help="apply fused-linear-attn rewrite")
    ap.add_argument("--fuse-conv1d", action="store_true", help="apply fused causal-conv1d rewrite")
    ap.add_argument("--lm-head-slice", action="store_true", help="slice lm_head input to last token only")
    args = ap.parse_args()

    def snapshot(label):
        r = rss(); ar = anon_rss(); fr = file_rss(); mi = mallinfo()
        print(f"\n[{label}]")
        print(f"  RSS        = {fmt(r)}  (anon={fmt(ar)}  file={fmt(fr)})")
        print(f"  mallinfo2  uordblks(in-use)={fmt(mi['uordblks'])}  "
              f"fordblks(free-in-arena)={fmt(mi['fordblks'])}  "
              f"hblkhd(mmap)={fmt(mi['hblkhd'])}")
        return r

    print(f"Model={args.model}  prompt_len={args.prompt_len}  "
          f"fuse_linear_attn={args.fuse}  fuse_conv1d={args.fuse_conv1d}  "
          f"lm_head_slice={args.lm_head_slice}")
    snapshot("process start")

    core = ov.Core()
    if args.fuse:
        register_la(core)
    if args.fuse_conv1d:
        register_cv(core)
    model = core.read_model(args.model)
    if args.fuse:
        n = replace_gated_delta_rule_loops(model)
        print(f"  → fused-linear-attn applied to {n} Loops")
    if args.fuse_conv1d:
        n = replace_causal_conv1d_chains(model)
        print(f"  → fused-conv1d applied to {n} chains")
    if args.lm_head_slice:
        ok = slice_lm_head_to_last_token(model)
        print(f"  → lm_head_slice applied: {ok}")
    r_after_read = snapshot("after read_model")

    compiled = core.compile_model(model, "CPU", {
        "PERFORMANCE_HINT": "LATENCY", "INFERENCE_NUM_THREADS": 4,
    })
    r_after_compile = snapshot("after compile_model")

    req = compiled.create_infer_request()
    r_after_req = snapshot("after create_infer_request")

    t0 = time.monotonic()
    with Sampler() as s:
        req.infer(prefill_inputs(args.prompt_len))
    elapsed = time.monotonic() - t0
    print(f"\n  prefill {args.prompt_len} tokens: {elapsed:.1f} s")
    r_after_infer = snapshot("after infer (resting)")
    print(f"  PEAK during infer: {fmt(s.peak)}  (+{fmt(s.peak - r_after_req)} above pre-infer)")

    # Force glibc to give back what it can
    gc.collect()
    trimmed = malloc_trim()
    time.sleep(0.5)
    r_after_trim = snapshot(f"after gc + malloc_trim (returned={trimmed})")

    print("\n=== one-line summary ===")
    print(f"prompt_len={args.prompt_len} fuse={args.fuse}: "
          f"compile={fmt(r_after_compile)}  peak={fmt(s.peak)}  resting={fmt(r_after_infer)}  "
          f"after_trim={fmt(r_after_trim)}")


if __name__ == "__main__":
    main()
