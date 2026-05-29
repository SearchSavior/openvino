# Memory optimization — plain OpenVINO + our kernels

Goal: drive Qwen3.5-0.8B (Qwen3-Next hybrid arch) inference memory toward the
llama.cpp reference — **~200 MB of runtime state on top of weights** for a
2048-token prompt with a quantized KV cache — using only `ov.Core` and our own
fused kernels (no openvino_genai, no paged attention).

Driver: [`openvino/lowmem_infer.py`](openvino/lowmem_infer.py). It reads the
stock optimum IR, swaps in our custom ops (gated-delta-rule recurrence + causal
conv1d, dispatching to the C kernels via ctypes when `QWEN3_USE_C=1`), slices
the lm_head, and feeds the prompt to the *stateful* model in chunks.

All numbers below are **measured** on this container (seq=2048,
`INFERENCE_NUM_THREADS=4`, `QWEN3_USE_C=1`, lm_head sliced). Reproduce with the
commands at the bottom.

## Measured — prefill peak RSS vs chunking

| prefill mode | peak RSS (prefill) | persistent state |
|---|---:|---:|
| single-shot (`--no-chunk`)  | 4487 MB | 67.8 MB |
| chunked, chunk=256          | 2246 MB | 67.8 MB |
| chunked, chunk=128          | 2051 MB | 67.8 MB |
| chunked, chunk=256 + u8 KV hint | 2246 MB | 67.8 MB (unchanged) |

After `compile_model`, before any infer: **166 MB** RSS (69 file-backed +
97 anon). The 754 MB int8 weight `.bin` is mmap'd lazily — it is **not** all
resident until the first infer touches it.

## Persistent state at 2048 tokens (all fp32)

| category | tensors | bytes |
|---|---:|---:|
| `full_attn_kv` (6 layers × K/V) | 12 | 48.0 MB  — grows with context |
| `linear_attn_state` (18 layers) | 18 | 18.0 MB  — fixed (recurrent state) |
| `conv1d_state` (18 layers)      | 18 |  1.7 MB  — fixed |
| **TOTAL** | 48 | **67.8 MB** |

The full-attn KV is the only part that scales with context: 3.0 MB at 128
tokens → 48.0 MB at 2048. At fp32 that is 4× the q8_0 llama.cpp reference
(~12 MB for the same 6 layers). Quantizing it is the main lever left on state.

## What worked

1. **lm_head slice** — only the last position is projected to the 248320-wide
   vocab. Removes the `[T, vocab]` fp32 logits tensor (~2 GB at T=2048). Applied
   in every row above.

2. **Chunked prefill** — feeding the stateful model in chunks bounds the
   per-call activation footprint. Prefill peak drops **4487 → 2051 MB**
   (chunk=128). Pure plain-`ov.Core`: the recurrent and conv state carry across
   chunks for free; the full-attn KV grows in the infer_request. No accuracy or
   state-size change (state is 67.8 MB regardless of chunking).

## What did not work / open items

- **`KV_CACHE_PRECISION` hint is inert here.** u8 vs default gives byte-identical
  state (67.8 MB) and the same peak. The hint targets the plugin's internal
  SDPA/PagedAttention KV path; this IR exposes the full-attn KV as a stateful
  `ReadValue/Assign` pair the hint does not touch. Getting the 48 MB KV to q8_0
  parity needs our own KV-quantizing op / explicit KV handling.

- **`validate_fusion.py` does not pass numerically.** Baseline `Loop` vs our
  fused `GatedDeltaRule` gives max abs logit diff ≈ 2.0 (top-10 token overlap
  10/10, so generations match, but it is not bit-equivalent). Must be root-caused
  before trusting the fused path for correctness work — see "Architecture
  correctness" below.

- **Prefill peak (2 GB) is still ~10× the state (68 MB).** The gap is infer-time
  activations + first-touch of the int8 weights (and likely a plugin
  decompression of int8 → fp32/bf16 for the matmuls). Untested hypothesis;
  needs `get_profiling_info` / runtime-graph attribution to confirm where the
  ~1.9 GB over the 166 MB compile baseline goes.

## Architecture correctness (the precondition)

Per the HF reference (`transformers.models.qwen3_5`), the linear-attention layer
is gated-delta-net with: q/k optionally L2-normed, `q *= 1/sqrt(D)`, decay
`g = -softplus(a + dt_bias) * exp(A_log)`, then the recurrence

```
S = S * exp(g_t)
kv_mem = (S * k_t).sum(-2)
delta  = (v_t - kv_mem) * beta_t
S = S + k_t ⊗ delta
out_t = (S * q_t).sum(-2)
```

Our `gdr_kernel` implements exactly this recurrence and assumes the scale and
L2-norm are applied as graph ops *before* the op. The 2.0 logit gap suggests one
of those pre-ops is not where we think it is in the exported IR (candidates:
the `1/sqrt(D)` scale, the q/k L2-norm, or the order of `exp(g)` vs the gate).
Next step is a per-layer activation diff (stock Loop output vs our op output) to
localize it.

## Next steps (ordered by expected payoff)

1. Root-cause the `validate_fusion` 2.0 gap with a per-layer activation diff —
   correctness gates everything else.
2. Our own quantized full-attn KV op (48 → ~12 MB q8_0) — the only path to KV
   parity, since the plugin hint is inert on this IR.
3. fp16 recurrent + conv state (18 → ~9 MB) — small, low risk at these
   magnitudes; needs the state tensors typed fp16.
4. Attribute the 2 GB prefill peak: confirm int8-weight decompression and
   whether the chunk activations can be bounded further.

## Reproduce

```bash
cd study/qwen3
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --no-chunk
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 256
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 256 --kv-precision u8
python openvino/validate_fusion.py --prompt-len 16   # currently FAILS (2.0 gap)
```
