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
`INFERENCE_NUM_THREADS=4`, `QWEN3_USE_C=1`). Reproduce with the commands at
the bottom.

## Where the memory was going (attribution)

After `compile_model`, before any infer: **166 MB** RSS (69 file-backed +
97 anon). The 754 MB int8 weight `.bin` is mmap'd lazily.

The single biggest cost was the **CPU plugin pre-decompressing every int8
weight constant into a bf16/f32 in-memory buffer for matmul throughput**.
At first infer this added ~866 MB of anon RSS, fixed regardless of sequence
width (W=1 and W=32 produced identical weight_expand). The IR stores 717 MB
of u8 weight Constants across 187 MatMuls (dequant chain
`u8 → Convert(f16) → Subtract(zp u8) → Multiply(scale f16) → Convert(f32) → MatMul`).
The plugin's pre-expanded copy was ~1.2× that footprint sitting alongside the
file-backed Constants.

## The kernels

Three custom ops, each dispatching to a C kernel via ctypes when
`QWEN3_USE_C=1`:

1. **`GatedDeltaRule`** ([fused_linear_attn.py](kernels/fused_linear_attn.py)) —
   replaces the unfused `Loop` body of the 18 linear-attention layers with a
   single op that implements the recurrence. Numerically validated to ≈1e-8
   against the HF `torch_recurrent_gated_delta_rule` reference.
2. **`FusedCausalConv1d`** ([fused_conv1d.py](kernels/fused_conv1d.py)) — the
   conv1d-with-state for those same layers, avoiding the `Concat(prev_state,
   current)` materialization.
3. **`QuantizedMatMul`** ([quantized_matmul.py](kernels/quantized_matmul.py)) —
   takes `(act, u8_weight, scale_f16, zp_u8)` as four inputs and streams the
   dequant in the inner loop. **Never materializes a bf16/f32 weight buffer.**
   Math: `y[..., n] = scale[n] · (act · u8[n] − zp[n] · sum(act))`. The C
   kernel parallelizes across output rows N with OpenMP.

Plus a graph-only rewrite, [`lm_head_slice`](kernels/lm_head_slice.py): insert
`Slice(axis=1, last token)` before the lm_head MatMul.

## Measured — seq = 2048, chunk = 128

| config                  | peak RSS | Δ vs baseline | prefill | persistent state |
|-------------------------|---------:|--------------:|--------:|-----------------:|
| stock IR (no fusions)   |  ~3.8 GB |             — |     —   |             —    |
| + all 4 rewrites, qmm=none      | 2051 MB |              — |  11.2s  |    67.9 MB |
| + qmm=lm_head only      |    **1805 MB** |   **−246 MB** |  11.7s  |    67.9 MB |
| + qmm=all (187 matmuls) |    **1332 MB** |   **−719 MB** |    112s |    67.9 MB |

After compile, before first infer: 166 MB RSS (qmm=none/lm_head) or 154 MB
(qmm=all — the smaller graph drops some plugin scratch).

### Persistent state at 2048 tokens (all fp32)

| category | tensors | bytes |
|---|---:|---:|
| `full_attn_kv` (6 layers × K/V) | 12 | 48.0 MB  — grows with context |
| `linear_attn_state` (18 layers) | 18 | 18.0 MB  — fixed (recurrent state) |
| `conv1d_state` (18 layers)      | 18 |  1.7 MB  — fixed |
| **TOTAL** | 48 | **67.8 MB** |

The full-attn KV is the only piece that scales with context: 3.0 MB at 128
tokens → 48.0 MB at 2048. At fp32 that is 4× the q8_0 llama.cpp reference
(~12 MB for the same 6 layers).

## What worked

1. **lm_head slice** — only the last position is projected to the 248320-wide
   vocab. Removes the `[T, vocab]` fp32 logits tensor (~2 GB at T=2048).
2. **Chunked prefill** — feeding the stateful model in chunks bounds the
   per-call activation footprint. Pure plain-`ov.Core`: recurrent and conv
   state carry across chunks for free; the full-attn KV grows in the
   infer_request.
3. **`QuantizedMatMul` streaming dequant** — eliminates the plugin's bf16
   weight repack. lm_head alone is 242 MB u8 → ~485 MB bf16; killing that
   single repack saves 246 MB at peak. The full sweep (`qmm=all`) saves
   719 MB but is 10× slower at prefill (the C kernel is a straight scalar
   loop, no VNNI/AVX-512 int8 dot product — the right speedup target).

## What did not work

- **`KV_CACHE_PRECISION` hint is inert here.** u8 vs default gives
  byte-identical state (67.8 MB) and the same peak. The hint targets the
  plugin's internal SDPA/PagedAttention KV path; this IR exposes the full-attn
  KV as a stateful `ReadValue/Assign` pair the hint does not touch.
- **`INFERENCE_PRECISION_HINT=bf16` and `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
  don't move the needle.** Same ~865 MB weight_expand as default.

## Correctness

- `kernels/test_kernels.py` — `gdr_kernel` matches HF
  `torch_recurrent_gated_delta_rule` to **1.3e-8** max diff.
- `openvino/validate_fusion.py` — with real (tokenized + embedded) inputs,
  top-10 logit overlap is 10/10 between stock and fused; max abs diff ~0.4.
  Random `inputs_embeds` is not a valid test — it amplifies sub-ULP ordering
  differences in the 64-step multiplicative recurrence.
- `openvino/generate.py` — end-to-end greedy decode with all rewrites
  applied produces coherent text ("I am Qwen3.5, the latest large language
  model developed by Tongyi Lab...").

## Next steps (ordered by expected payoff)

1. **Quantize the full-attn KV cache** to u8 in a custom KV op (48 → ~12 MB,
   saves ~36 MB). The only path to KV parity with llama.cpp on this IR.
2. **Speed up `QuantizedMatMul`** with AVX-512 VNNI int8 dot product, then
   `qmm=all` becomes the default with no time penalty — saves ~700 MB.
3. fp16 recurrent + conv state (18 → ~9 MB) — small, low risk.

## Reproduce

```bash
cd study/qwen3
# Baseline
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128
# Streaming int8 lm_head (the practical sweet spot)
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128 --qmm lm_head
# Memory ceiling (slow)
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128 --qmm all
# End-to-end generation (verify correctness)
QWEN3_USE_C=1 python openvino/generate.py
```
