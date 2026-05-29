# Memory optimization — plain OpenVINO + our kernels

Goal: drive Qwen3.5-0.8B (Qwen3-Next hybrid arch) inference memory toward the
llama.cpp reference — **~200 MB of runtime state on top of weights** for a
2048-token prompt with a quantized KV cache — using only `ov.Core` and our own
fused kernels (no openvino_genai, no paged attention).

Driver: [`openvino/lowmem_infer.py`](openvino/lowmem_infer.py). It reads the
stock optimum IR, swaps in our custom ops (gated-delta-rule recurrence + causal
conv1d, dispatching to the C kernels via ctypes when `QWEN3_USE_C=1`), slices
the lm_head, and feeds the prompt to the *stateful* model in chunks.

## Architecture is correct (not the problem)

`openvino/validate_fusion.py` shows our gated-delta-rule op matches the stock
`Loop` to **1.8e-5**. The exported graph applies the `q * 1/sqrt(D)` scale and
the q/k L2-norm as graph ops *before* the Loop, so our kernel — which does
neither internally — is a faithful drop-in. The bottleneck is the **memory
architecture** (how prefill and state are laid out), not the math.

## Measured (seq = 2048, INFERENCE_NUM_THREADS=4)

After compile, RSS is **1219 MB** = 478 file-backed + 741 anon. The 741 MB anon
is the CPU plugin repacking the int8 weights into its own matmul layout
(comparable to llama.cpp's ~800 MB mmap'd weights).

| prefill mode | peak RSS | runtime over weights | persistent state |
|---|---:|---:|---:|
| single-shot       | 1863 MB | +644 MB | 45.1 MB |
| chunked (256)     | 1425 MB | +205 MB | 45.1 MB |
| chunked (128)     | 1375 MB | +155 MB | 45.1 MB |
| chunked (256) + u8 KV hint | 1425 MB | +205 MB | 45.1 MB (no change) |

Persistent state at 2048 tokens (all fp32):
- `full_attn_kv`     6 layers × K/V  — **25.2 MB** (grows with context)
- `linear_attn_state` 18 layers       — **18.9 MB** (fixed, recurrent state)
- `conv1d_state`      18 layers       —  **1.1 MB** (fixed)

## What worked

1. **lm_head slice** — only the last position is projected to the 248320-wide
   vocab. Removes the `[T, vocab]` fp32 logits tensor (~2 GB at T=2048). This is
   the difference between the 3811 MB stock-IR peak and the ~1.8 GB sliced peak.

2. **Chunked prefill** — feeding the stateful model in chunks bounds the
   per-call activation footprint to `chunk_size` instead of `T`. Runtime over
   weights drops **644 → 155 MB** (chunk=128). This is the main win and it is
   pure plain-`ov.Core`: the recurrent and conv state carry across chunks for
   free; the full-attention KV grows in the infer_request.

## What did not work / open items

- **`KV_CACHE_PRECISION` hint has no effect here.** It targets the plugin's
  internal SDPA/PagedAttention KV path; this IR exposes the full-attn KV as a
  stateful `ReadValue/Assign` pair, which the hint does not touch. Quantizing
  the 25 MB full-attn KV to u8 (~6 MB) needs our own KV op / explicit KV
  handling.

- **Weight footprint: 1219 MB resident vs llama.cpp's ~800 MB.** The 478 MB
  file-backed mapping appears to survive after the plugin repacks weights into
  741 MB of anon — i.e. weights are double-resident. Releasing the original
  mapping post-compile, or avoiding the repack, is the next ~400 MB.

## Next steps (ordered by expected payoff)

1. Kill the weight double-mapping (~400 MB): investigate `ov::cache_dir` /
   `MMAP` properties so the repacked weights are the only resident copy.
2. fp16 recurrent + conv state (18.9 → ~9.5 MB) — small, free, no accuracy risk
   at these magnitudes; needs the state tensors typed fp16.
3. Our own quantized full-attn KV op (25 → ~6 MB) — the only way to get the KV
   cache to q8_0 parity, since the plugin hint is inert on this IR.
4. Decode-time memory: confirm the per-step peak stays flat as context grows.
