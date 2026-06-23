# llama.cpp PR #24423 — DiffusionGemma decode, optimizations to steal

Source: [`ggml-org/llama.cpp#24423`](https://github.com/ggml-org/llama.cpp/pull/24423)
— a from-scratch C++/CUDA implementation of **DiffusionGemma** decode. This is the
single most directly comparable reference to our SYCL/oneDNN Xe2 plan: same model,
GPU decode, written against a low-level tensor/kernel API (ggml) rather than a
PyTorch framework. (The git relay blocked cloning `ggml-org`; this is read from the
raw PR diff, ~3,900 lines.) It **confirms nearly every conclusion in our other
reports** and adds a few new, concrete techniques.

Key files: `examples/diffusion/diffusion.cpp` (denoise loop), `src/models/diffusion-gemma.cpp`
(graph + masks + SC), `ggml/src/ggml-cuda/diffusion-sampling.cu` (on-GPU sampler),
`src/llama-context.cpp` (sparse output rows).

---

## Decode optimizations (and how each maps to us)

### 1. Exact prompt-KV prefix cache — PREFILL once, DECODE only the canvas
`params.kv_cache`: prefill the prompt once into a per-layer K/V store
(`pkv_k[il]`/`pkv_v[il]`), then each denoising step decodes **only the canvas**,
reading the cached prefix — instead of re-decoding `[prompt|canvas]` every step.
The DECODE mask is rectangular `[P+C keys, C queries]` over cached prompt + fresh
canvas K/V. **This is exactly our "exact prefix cache + per-step canvas recompute"
conclusion, in a working impl.** No dual/vicinity/approximate cache anywhere —
confirming those are LLaDA-specific and unnecessary for DiffusionGemma.

### 2. One unified forward reproduces the two-pass (encoder/decoder) via region splits
Rather than literally run encoder-then-decoder, the graph does a single forward
over `[prompt | canvas]` split at `P = n_tokens - canvas_length`, region-aware in
three places (`diffusion-gemma.cpp` header comment):
1. **input embeddings**: prompt = `embed*sqrt(n_embd)`; canvas = `rms_norm_noscale(...)`
2. **per-layer scalar**: prompt = `enc_out_scale`; canvas = `out_scale` (two learned
   per-layer scalars — a model detail to handle at weight load)
3. **attention mask**: prompt causal (SWA-clipped); canvas bidirectional over all
   prompt+canvas

Useful design point for us: one graph with region masks/scalars is simpler than two
passes and still exact.

### 3. Region-aware additive mask = mixed causal/bidirectional **with SWA in the mask**
The mask classes (`llm_graph_input_attn_diffusion*`) build the additive mask
directly: prompt queries causal over earlier prompt (never canvas); canvas queries
bidirectional; **sliding layers clip keys outside the window** — canvas→prompt
reach is bounded to the last `n_swa-1` prompt positions on sliding layers, "all" on
global. **This confirms our final position exactly**: sliding window is handled in
the **mask + the cached store**, per layer — not a special attention kernel, and no
`k0start`-style kernel patch.

### 4. Chunked causal prefill — flat memory, long context  *(NEW vs our reports)*
The prompt is prefilled in `n_ubatch`-sized chunks; each chunk writes its K/V to the
store and attends causally over `[0, off+n_tokens)` (prior chunks from the store +
this chunk). Keeps `n_tokens <= n_ubatch`, so the compute buffer tracks the chunk,
not the prompt → flat activation memory regardless of context (reported ~566 MiB,
enabling 60K+ prompts; avoids O(prompt²) overflow on 32-bit indices). **New lever
for our encoder pass**: chunk the prompt encode to bound memory and index range.

### 5. On-GPU entropy-bound sampler kernel  (`diffusion-sampling.cu`)
One CUDA block **per canvas position**; in shared memory: parallel max→argmax,
parallel `Z=Σe^d` and `T=Σ d·e^d`, then `entropy = logZ - T/Z`; multinomial draw via
a **slice-scan**: split vocab into `blockDim` contiguous slices, each thread sums its
slice, exclusive-scan the slice sums, and only the thread whose slice spans `r=u·Z`
walks its `~vocab/threads` elements — serial work drops from `O(vocab)` to one slice.
Device scratch (`u` + 3 outputs) is **grow-only/cached** (no steady-state malloc).
**This is our SYCL DiffusionSampler design, concretely**: workgroup-per-canvas-row,
fused argmax+softmax-Z+entropy, slice-scan categorical sampling, persistent scratch.

### 6. Device-resident self-conditioning — no host round-trip, one matmul/step
`sc_dev`: the graph persists this step's canvas logits (last `C` rows) into a device
buffer in-graph (`ggml_cpy` → `sc_dev`); the next step reads it on-device for the
canvas embedding. The SC soft-embedding weight (`tok_embd` dequantized + transposed
to `[n_vocab, n_embd]` F16) is built **once** (`dg_ensure_sc_embT`), so per-step SC
cost is **one matmul**, not a weight rebuild. Avoids a ~268 MB logits D2H per step.
**Maps to our SC plan**: keep SC on-device, prebuild the transposed dequant embed,
one matmul/step — and note this is the path that makes the DeepMind-paper "SC is
prunable" question testable cheaply.

### 7. Q8_0 KV-cache quantization on the prompt store
The K/V store type follows the attention path; quantizing to **Q8_0** roughly doubles
effective context (reported ~32K→65K) with per-block dequant caching and ~10 ms/step
overhead. **Confirms our "quantize the frozen prefix KV" lever** — read every step,
so the saving compounds over `S`.

### 8. Sparse output rows — logits only for the canvas
`llama-context.cpp`: memoryless/canvas models cap `n_outputs_max` and map flagged
token indices to output rows, so logits are materialized only for the canvas, not
the prompt. (Note: *all* canvas rows are needed — "no `inp_out_ids` row-selection
mid-stack" — unlike AR's last-token trick.) Confirms our sparse-logits point, scoped
to the canvas.

### 9. Entropy-bound accept + stability/confidence stop (matches the reference)
Sort positions by entropy, accept the lowest where `cumE - entropy[pos] <=
entropy_bound`, renoise the rest to fresh random; output = stable argmax canvas.
Adaptive stop = argmax stable for `stability_threshold` steps **and** mean entropy
`< confidence_threshold`. Identical to the HF `EntropyBoundSampler` we documented —
confirms the convergence logic and the `S`-cutting levers.

### 10. Final logit softcap as scale→tanh→scale
`ggml_scale(1/cap)` → `tanh` → `ggml_scale(cap)`. As we planned (oneDNN `eltwise_tanh`
post-op).

---

## What's new vs our prior reports

| New from this PR | Why it matters for Xe2/SYCL |
|---|---|
| **Chunked causal prefill** (§4) | Bound encoder activation memory + index range; chunk the prompt encode |
| **Unified forward + region splits** (§2) | One graph, region-aware embed/scalar/mask — simpler than two passes, still exact |
| **Two per-layer scalars** (`enc_out_scale` vs `out_scale`) | Weight-load detail: DiffusionGemma has distinct encoder/decoder per-layer scalars |
| **Slice-scan multinomial** in the sampler (§5) | Concrete kernel trick: `O(vocab)`→`O(vocab/threads)` serial categorical draw |
| **Device-resident SC with prebuilt transposed embed** (§6) | Keep SC on-device; one matmul/step; no D2H |

## Confirmations of earlier conclusions
Exact prefix cache (no approximate caching) · sliding window handled in mask+store,
no kernel patch · per-step canvas recompute · entropy-bound + stability/confidence
stop · sparse logits over canvas · KV quant on the prefix store · softcap as a
post-op · self-conditioning = `probs @ embed`.

## Reported decode throughput (PR description — treat as rough)
RTX 5090 ≈ 2.2–2.7 tok/s (256-canvas denoise); AMD 7900 XTX ≈ 364 ms/step; Strix
Halo ≈ 9 tok/s. The modest tok/s on a 5090 underlines that **`S` (denoising steps) ×
MoE is the bottleneck** — consistent with "cut `S`" being the master lever.

---

*Read from the raw PR diff (`pr24423.diff`, ~3.9k lines) via the HTTPS proxy; direct
`git clone` of `ggml-org` was blocked by the session's git egress policy.*
