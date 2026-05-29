"""
Low-memory plain-OpenVINO inference for Qwen3.5 (Qwen3-Next hybrid arch).

Goal: drive runtime memory toward the llama.cpp reference (~200 MB of runtime
state on top of weights for a 2048-token prompt with a quantized KV cache),
using nothing but `ov.Core` and our own fused kernels — no openvino_genai, no
paged attention, no pre-exported fused dir.

What this harness controls explicitly (the "memory architecture"):

  1. lm_head slice — only the last position is projected to the 248320-wide
     vocab. Removes the [T, vocab] fp32 logits tensor (~2 GB at T=2048).

  2. Chunked prefill — the prompt is fed to the *stateful* model in chunks of
     --chunk tokens instead of one [1, T, H] shot. The recurrent (gated-delta)
     state and the conv state carry across chunks for free; the full-attention
     KV cache grows in the infer_request. This bounds the per-call activation
     footprint to chunk_size rather than T.

  3. KV-cache precision — optional u8 / f16 KV for the 6 full-attention layers
     via the CPU plugin hint.

  4. Our kernels — the gated-delta-rule recurrence and the causal conv1d run as
     our custom ops (Python op -> C kernel via ctypes when QWEN3_USE_C=1),
     swapped in over the exported `Loop` / conv chains.

Everything is measured: peak RSS (background sampler), heap (mallinfo2),
file-backed vs anon (smaps_rollup), and the persistent state bytes by category
(query_state).

Usage:
    QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 256
    QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 256 --kv-precision u8
    QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --no-chunk   # single-shot baseline
"""
from __future__ import annotations

import argparse
import ctypes
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernels"))
from fused_linear_attn import register as register_la, replace_gated_delta_rule_loops  # noqa: E402
from fused_conv1d import register as register_cv, replace_causal_conv1d_chains  # noqa: E402
from lm_head_slice import slice_lm_head_to_last_token  # noqa: E402

MODEL_DIR = "/tmp/qwen3-work/qwen35-0.8b-int8"
HIDDEN = 1024
ROPE_SECTIONS = 4  # mrope position_ids leading dim


# ---------------------------------------------------------------------------
# Memory introspection
# ---------------------------------------------------------------------------
class Mallinfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd",
        "usmblks", "fsmblks", "uordblks", "fordblks", "keepcost")]


_libc = ctypes.CDLL("libc.so.6")
_libc.mallinfo2.restype = Mallinfo2


def _proc_status(field: str) -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith(field + ":"):
                return int(line.split()[1])
    return -1


def rss_mb() -> float:
    return _proc_status("VmRSS") / 1024.0


def smaps_mb() -> dict:
    out = {}
    with open("/proc/self/smaps_rollup") as f:
        for line in f:
            for k in ("Rss", "Anonymous", "Private_Dirty"):
                if line.startswith(k + ":"):
                    out[k] = int(line.split()[1]) / 1024.0
    return out


class PeakSampler:
    """Background thread that records peak VmRSS while a block runs."""

    def __init__(self, period=0.004):
        self.period = period
        self.peak = 0.0
        self._stop = threading.Event()
        self._t = None

    def __enter__(self):
        self.peak = rss_mb()

        def loop():
            while not self._stop.is_set():
                self.peak = max(self.peak, rss_mb())
                time.sleep(self.period)

        self._t = threading.Thread(target=loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()


# ---------------------------------------------------------------------------
# State accounting (query_state) — persistent KV + recurrent + conv state
# ---------------------------------------------------------------------------
def classify_state(shape):
    if len(shape) == 4 and shape[1] == 16 and shape[2] == 128 and shape[3] == 128:
        return "linear_attn_state"
    if len(shape) == 3 and shape[1] == 6144:
        return "conv1d_state"
    if len(shape) == 4 and shape[1] == 2 and shape[3] == 256:
        return "full_attn_kv"
    return f"other_{tuple(shape)}"


def state_bytes(req):
    by_cat = defaultdict(lambda: [0, 0])  # cat -> [count, bytes]
    total = 0
    for s in req.query_state():
        t = s.state
        sh = tuple(t.shape)
        nbytes = int(np.prod(sh)) * t.element_type.size
        cat = classify_state(sh)
        by_cat[cat][0] += 1
        by_cat[cat][1] += nbytes
        total += nbytes
    return by_cat, total


def fmt(b):
    if b >= 1 << 20:
        return f"{b / (1 << 20):8.2f} MB"
    if b >= 1 << 10:
        return f"{b / (1 << 10):8.2f} KB"
    return f"{b:6d}  B"


# ---------------------------------------------------------------------------
# Model setup
# ---------------------------------------------------------------------------
def build_model(kv_precision: str | None):
    core = ov.Core()
    register_la(core)
    register_cv(core)

    lm = core.read_model(f"{MODEL_DIR}/openvino_language_model.xml")
    n_gdr = replace_gated_delta_rule_loops(lm)
    n_cv = replace_causal_conv1d_chains(lm)
    n_slice = slice_lm_head_to_last_token(lm)
    print(f"  rewrites: gdr={n_gdr}  conv1d={n_cv}  lm_head_slice={n_slice}")

    cfg = {"INFERENCE_NUM_THREADS": 4, "PERFORMANCE_HINT": "LATENCY"}
    if kv_precision:
        cfg["KV_CACHE_PRECISION"] = kv_precision
    try:
        compiled = core.compile_model(lm, "CPU", cfg)
    except Exception as e:
        if kv_precision:
            print(f"  KV_CACHE_PRECISION={kv_precision} rejected ({str(e)[:80]}); retrying default")
            cfg.pop("KV_CACHE_PRECISION")
            compiled = core.compile_model(lm, "CPU", cfg)
        else:
            raise

    embed = core.compile_model(f"{MODEL_DIR}/openvino_text_embeddings_model.xml", "CPU")
    return compiled, embed


def embed_ids(embed, ids):
    return list(embed.create_infer_request().infer({0: ids}).values())[0]


def position_ids(start, length):
    pos = np.arange(start, start + length, dtype=np.int64).reshape(1, 1, length)
    return np.tile(pos, (ROPE_SECTIONS, 1, 1))


# ---------------------------------------------------------------------------
# Prefill (chunked or single-shot) + a few decode steps
# ---------------------------------------------------------------------------
def run(seq, chunk, kv_precision, decode_steps):
    print(f"\n{'=' * 72}")
    print(f"seq={seq}  chunk={chunk or 'single-shot'}  "
          f"kv_precision={kv_precision or 'default(f32)'}")
    print('=' * 72)

    s_rss0 = rss_mb()
    compiled, embed = build_model(kv_precision)
    logits_out = next(o for o in compiled.outputs if "logits" in o.get_any_name())
    s_compiled = smaps_mb()
    print(f"  after compile: RSS {rss_mb():8.1f} MB  "
          f"(file-backed {s_compiled['Rss'] - s_compiled['Anonymous']:.0f}, "
          f"anon {s_compiled['Anonymous']:.0f})")

    req = compiled.create_infer_request()

    # Synthetic prompt embeddings (random ids); deterministic.
    rng = np.random.default_rng(0)
    all_ids = rng.integers(1, 200000, size=(1, seq), dtype=np.int64)

    chunks = [(0, seq)] if not chunk else [
        (i, min(chunk, seq - i)) for i in range(0, seq, chunk)]

    peak = PeakSampler()
    with peak:
        t0 = time.time()
        past = 0
        for (start, length) in chunks:
            ids = all_ids[:, start:start + length]
            req.infer({
                "inputs_embeds": embed_ids(embed, ids),
                "attention_mask": np.ones((1, past + length), dtype=np.int64),
                "position_ids": position_ids(past, length),
                "beam_idx": np.zeros((1,), dtype=np.int32),
            })
            past += length
        t_prefill = time.time() - t0
    prefill_peak = peak.peak

    by_cat, total_state = state_bytes(req)
    next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())

    # Decode
    peak_d = PeakSampler()
    with peak_d:
        t1 = time.time()
        for _ in range(decode_steps):
            req.infer({
                "inputs_embeds": embed_ids(embed, np.array([[next_id]], dtype=np.int64)),
                "attention_mask": np.ones((1, past + 1), dtype=np.int64),
                "position_ids": position_ids(past, 1),
                "beam_idx": np.zeros((1,), dtype=np.int32),
            })
            next_id = int(np.asarray(req.get_tensor(logits_out).data)[0, -1].argmax())
            past += 1
        t_decode = time.time() - t1
    decode_peak = peak_d.peak

    _, total_state_after = state_bytes(req)

    # ----- report -----
    print(f"\n  prefill: {t_prefill:6.2f}s  ({seq} tok in {len(chunks)} chunk(s))")
    print(f"  decode:  {t_decode:6.2f}s  ({decode_steps} tok)")
    print(f"\n  PEAK RSS  prefill={prefill_peak:8.1f} MB   decode={decode_peak:8.1f} MB")
    print(f"  start RSS {s_rss0:8.1f} MB  ->  end RSS {rss_mb():8.1f} MB")
    print(f"\n  persistent state @ {past} tokens:")
    for cat in sorted(by_cat):
        c, b = by_cat[cat]
        print(f"    {cat:<22s} x{c:<3d} {fmt(b)}")
    print(f"    {'TOTAL state':<22s}     {fmt(total_state_after)}")

    return {
        "seq": seq, "chunk": chunk, "kv_precision": kv_precision,
        "prefill_peak_mb": prefill_peak, "decode_peak_mb": decode_peak,
        "state_mb": total_state_after / (1 << 20),
        "prefill_s": t_prefill,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--no-chunk", action="store_true", help="single-shot prefill")
    ap.add_argument("--kv-precision", choices=["u8", "f16"], default=None)
    ap.add_argument("--decode-steps", type=int, default=8)
    args = ap.parse_args()

    chunk = None if args.no_chunk else args.chunk
    res = run(args.seq, chunk, args.kv_precision, args.decode_steps)

    print(f"\n{'-' * 72}")
    print(f"SUMMARY  prefill_peak={res['prefill_peak_mb']:.0f} MB  "
          f"state={res['state_mb']:.1f} MB  "
          f"(target: ~200 MB runtime over weights)")


if __name__ == "__main__":
    main()
