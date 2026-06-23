# DiffusionGemma in vLLM: Architecture-Driven Optimization Strategies

> Study of the `DiffusionGemma` implementation in [`vllm-project/vllm`](https://github.com/vllm-project/vllm),
> focused on the optimization strategies that fall directly out of the model's
> architecture. File/line references point at the upstream `main` tree studied
> for this report.
>
> Primary sources:
> - `vllm/model_executor/models/diffusion_gemma.py`
> - `vllm/model_executor/models/gemma4.py`
> - `vllm/model_executor/models/config.py`
> - `vllm/transformers_utils/configs/diffusion_gemma.py`
> - `vllm/config/diffusion.py`
> - `vllm/v1/attention/ops/triton_unified_attention.py`

---

## 1. Executive Summary

DiffusionGemma is a **discrete diffusion language model (dLLM)**: instead of
generating tokens left-to-right one at a time, it iteratively *denoises* a
fixed-length block of tokens (a "canvas") in parallel. The vLLM port is
interesting precisely because it reuses almost the entire autoregressive serving
stack — KV cache, attention kernels, the speculative-decoding data path, CUDA
graphs — by mapping diffusion semantics onto existing machinery rather than
building a parallel runtime.

The architecture has four load-bearing properties, and every optimization in the
port is downstream of one of them:

| Architectural property | Optimization it unlocks |
|---|---|
| **One backbone, two modes (YOCO-style)** — same weights run as a causal "encoder" that writes KV, and a bidirectional "decoder" that only reads it | No duplicated weights; prompt KV computed once and reused across all denoising steps |
| **Mixed causal / bidirectional attention** — encoder spans are causal, denoising canvases are bidirectional, *in the same batch* | A single fused attention kernel with a per-sequence causal flag; no batch splitting |
| **Block (canvas) generation** reuses speculative-decoding semantics | Draft-token slots, scheduler spec path, and sampler output plumbing are reused with overloaded meaning |
| **Parallel denoising over a `[seqs, canvas, vocab]` tensor** | One fused `torch.compile` sampler step; entropy-bound parallel acceptance; confidence-based early exit |

The net effect: DiffusionGemma "behaves like every other model" to the engine
(`config.py:161-173`), which is the whole optimization thesis.

---

## 2. Architecture Overview

### 2.1 Single backbone, two modes (YOCO pattern)

The model docstring states the design directly
(`diffusion_gemma.py:5-11`):

```
Single Gemma4 backbone run in two modes (like YOCO):
- encoder mode: causal attention, writes KV cache
- decoder mode: bidirectional attention, reads encoder KV, doesn't write
Same weights, same layers. The only decoder-unique component is a
self-conditioning MLP.
```

There is exactly **one** `Gemma4Model` instance
(`diffusion_gemma.py:230-233`). The HF checkpoint ships separate
`model.encoder.language_model.*` and `model.decoder.*` weight trees, but
`load_weights` folds both into the single backbone and *skips the duplicate
decoder copy* (`diffusion_gemma.py:419-429`). The only genuinely
decoder-unique parameter is the self-conditioning MLP
(`DiffusionGemmaSelfConditioning`, `diffusion_gemma.py:66-92`).

Mode dispatch is data-driven, not a code branch: `DiffusionGemmaModelState`
sets a per-request causal flag, and `num_draft_tokens == 0` means
encoder/prefill while `> 0` means a denoising decode step
(`diffusion_gemma.py:744-750`, `1217-1218`).

### 2.2 The canvas and the denoising loop

Per-request diffusion state lives in pre-allocated GPU buffers
(`DiffusionGemmaRequestStates`, `diffusion_gemma.py:645-741`): a `canvas`
of token ids `[max_num_reqs, canvas_length]`, a `step` counter, an
`argmax_canvas` (the deterministic best-guess that is eventually emitted), a
circular `accepted_canvas_history` for stability detection, a `confident`
flag, and the self-conditioning soft embeddings. The denoising loop runs
encoder (commit) → repeated denoise steps → convergence → commit, all encoded
as flags flipped in-place by the sampler.

### 2.3 `k_eq_v` (K = V sharing)

DiffusionGemma's full-attention layers have **no `v_proj`**: V is taken from
the K projection output (`diffusion_gemma.py:185-190`). The backbone supports
this as the `attention_k_eq_v` variant; the unified QKV path loads K weights
into both K and V slots so `V == K` automatically
(`gemma4.py:511-515`, `567-589`). The config forces this flag on
unconditionally (`diffusion_gemma.py:190`,
`transformers_utils/configs/diffusion_gemma.py:9-16`).

---

## 3. Optimization Strategies That Leverage the Architecture

### 3.1 YOCO-style KV reuse: compute prompt KV once, read it every step

**What the architecture enables.** Because encoder and decoder are the *same
weights* and the decoder is *read-only* against the KV cache, the expensive
prompt context only has to be encoded a single time. Every one of the (up to
`max_denoising_steps`, default 48) denoising passes over the canvas attends
into that already-materialized KV without re-projecting or re-writing it.

**How vLLM exploits it.** The decoder mode is explicitly "reads encoder KV,
doesn't write" (`diffusion_gemma.py:7`). This is the diffusion analogue of
YOCO / cross-decoder KV sharing: the prompt is the "encoder", the canvas is the
"decoder", and the KV cache is the shared bridge. The backbone already supports
KV-shared layers (`kv_sharing_target_layer_name`, `gemma4.py:480-503`), so the
mechanism is reused rather than reinvented.

**Why it matters.** For a dLLM the denoising loop is the dominant cost. Pinning
prompt KV and only paying canvas-sized attention per step turns an `O(steps ×
(prompt+canvas))` projection cost into `O(prompt) + O(steps × canvas)`.

**Further opportunity.** With canvases pinned per request, an explicit
prefix-cache / KV-dedup across requests sharing a system prompt would compound
the saving — the read-only decoder makes this safe by construction.

### 3.2 Mixed causal/bidirectional attention in one kernel, one batch

**What the architecture enables.** Diffusion needs *bidirectional* attention
across the canvas (every noisy token sees every other) while the prompt stays
*causal*. Naively this forces two attention regimes — and if requests at
different phases are batched together, two different masks in the same step.

**How vLLM exploits it.** Rather than split the batch, the port threads a
**per-sequence causal flag** through the unified Triton attention kernel.
`prepare_attn` builds a `_causal_buf` GPU tensor — `True` for encoder-phase
requests, `False` for denoising ones — and passes it as `causal`
(`diffusion_gemma.py:997-1025`). The kernel branches on
`USE_PER_SEQ_CAUSAL`, loading `per_seq_causal_ptr[seq_idx]` and applying either
the causal or the symmetric (bidirectional) sliding-window mask per sequence
(`triton_unified_attention.py:545-555`, `822-825`, `1058-1060`). A persistent
buffer updated in place is what makes this compatible with FULL CUDA graph
replay (`diffusion_gemma.py:813-817`).

**Backend gating.** Because the mask is heterogeneous within a batch,
FlashInfer is excluded and `use_non_causal` is auto-set; FA / Triton are the
supported backends (`config.py:122-137`).

**Why it matters.** Encoder and denoising requests coexist in one scheduler
step with **zero batch splitting** and a single kernel launch — the masking
cost is one extra `tl.load` per sequence.

**Further opportunity.** The padded/phantom canvas tail (§3.6) still occupies
mask lanes; a compaction pass for canvases truncated near `max_model_len`
would reclaim that attention work.

### 3.3 Reuse the speculative-decoding data path for block diffusion

**What the architecture enables.** Canvas/block generation emits *many* tokens
per step, exactly like speculative decoding verifies many draft tokens per
step. The two have isomorphic data shapes.

**How vLLM exploits it.** The `DiffusionConfig` docstring is explicit: dLLMs
"reuse the speculative-decoding data path (draft token ids, scheduled spec
decode tokens) with overloaded semantics for block-based generation"
(`config/diffusion.py:11-18`). `canvas_length` doubles as the number of "spec
tokens" scheduled per step. The canvas is published into
`req_states.draft_tokens` (`diffusion_gemma.py:639-641`, `1142`), and the
sampler returns `num_sampled` / `num_rejected` via the ordinary
`SamplerOutput` (`diffusion_gemma.py:1196-1202`). `num_draft_tokens` is the
encoder-vs-decode discriminator (`diffusion_gemma.py:1217`).

**Why it matters.** No new scheduler, no new worker loop, no new output
plumbing — the most invasive parts of a serving engine are reused unchanged.
The `_compute_num_rejected` helper even reinterprets "rejected" as "renoised
this step" (`diffusion_gemma.py:454-463`).

**Further opportunity.** Because draft slots are sized to `canvas_length`,
variable/adaptive canvas sizing could be expressed purely through the existing
spec-token count without touching the scheduler.

### 3.4 One fused `torch.compile` sampler step

**What the architecture enables.** A denoising step is a fixed, branch-light
pipeline applied uniformly to a `[num_decode, canvas, vocab]` tensor:
temperature schedule → Gumbel-max sample → softmax/entropy → accept/renoise →
stability/convergence → write-back. There is no token-by-token dependency
*within* a step.

**How vLLM exploits it.** The entire step is a single
`@torch.compile(dynamic=True)` function, `_compiled_sample_step`
(`diffusion_gemma.py:466-642`), operating on pre-allocated buffers with **no
GPU→CPU syncs on the hot path** (`diffusion_gemma.py:1038-1042`). All
per-request state (canvas, step, history, confidence, SC embeds, draft tokens)
is mutated in place inside the compiled region across seven labeled phases.
CPU/NumPy work (splitting decode vs prefill, staging slot indices) is kept
*outside* compile and moved through UVA-backed tensors
(`diffusion_gemma.py:1087-1090`, `1220-1240`).

**Why it matters.** The whole sampler collapses to effectively one kernel
sequence per step, fully vectorized over requests and canvas positions —
critical because the sampler runs once per denoising iteration (tens of times
per generated block).

### 3.5 Store self-conditioning as `[..,hidden]`, not `[..,vocab]` (~170× buffer cut)

**What the architecture enables.** Self-conditioning feeds the *previous*
step's predicted distribution back in. But the SC MLP only ever consumes
`probs @ embed_weight` — a hidden-sized soft embedding — never the raw
`[.., vocab]` distribution (`diffusion_gemma.py:268-277`, `87-92`).

**How vLLM exploits it.** The sampler computes the matmul at denoise time and
stores only the `[max_num_reqs, canvas, hidden]` soft embedding
(`self_conditioning_embeds`), explicitly noting this "shrinks this buffer by
vocab/hidden (~170×)" versus persisting full probabilities
(`diffusion_gemma.py:711-718`, `623-631`). On the next step
`_apply_self_conditioning` just runs the MLP over the stored soft embed — one
MLP call per request, CPU metadata only, no GPU syncs
(`diffusion_gemma.py:904-925`). The sampler also pre-masks SC embeds to zero
for slots that will *not* denoise next, so the consumer reads them directly
(`diffusion_gemma.py:629-631`).

**Why it matters.** A persistent `[seqs, canvas, vocab]` fp32 buffer would be
enormous (vocab ≈ 256k). Folding the matmul forward shrinks persistent memory
by the vocab/hidden ratio and moves the cost to where it's already paid.

### 3.6 Uniform-canvas math via phantom padding (branch-free sampler)

**What the architecture enables.** Almost every canvas is exactly
`canvas_length`; only the final block near `max_model_len` is shorter. Special-
casing that in the hot loop would add branches to an otherwise uniform kernel.

**How vLLM exploits it.** Truncated canvases are **padded back to `CL`** with
zeroed logits before the compiled step (`diffusion_gemma.py:1250-1259`). Zeroed
rows are uniform → maximum entropy (never trigger premature convergence) and
argmax 0 (stable), and are simply never committed because `num_sampled` carries
the real `valid_canvas_len` (`diffusion_gemma.py:537-543`, `601-605`). The
uniform-`CL` assumption lets the entire sampler stay branch-free and shape-
static.

**Why it matters.** Static shapes are friendlier to `torch.compile` and CUDA
graphs, and the correctness argument (phantom rows are inert) means no runtime
guards.

### 3.7 Entropy-bound parallel acceptance + confidence-based early convergence

**What the architecture enables.** Because all canvas positions are predicted
simultaneously, the sampler can decide *per step* how many positions to "lock
in" and when the whole block has converged — there is no sequential commit
order to respect.

**How vLLM exploits it.**
- **Entropy-bound acceptance** sorts per-position entropies, takes a cumulative
  mask under `entropy_bound`, and accepts the confident positions while
  renoising the rest in one vectorized pass
  (`diffusion_gemma.py:545-572`). A required `EntropyBound` sampler config is
  validated up front (`diffusion_gemma.py:833-855`).
- **Confidence + stability convergence**: a block is done when mean entropy is
  below `confidence_threshold` *and* the last `stability_threshold` argmax
  canvases are identical *and* enough history exists — or `max_denoising_steps`
  is hit (`diffusion_gemma.py:537-543`, `607-621`). On convergence it overwrites
  the stochastic canvas with the deterministic argmax
  (`diffusion_gemma.py:633-637`).

**Why it matters.** Both mechanisms cut the number of denoising iterations,
which is the single biggest runtime lever for a dLLM — fewer full backbone
passes per generated block.

**Further opportunity.** `max_denoising_steps` (default 48) and
`entropy_bound`/`confidence_threshold` are the throughput/quality dial;
per-request adaptive step budgets would let easy prompts exit even earlier.

### 3.8 Fused fp32 logits softcap

**What the architecture enables.** Gemma applies a final-logit softcap in fp32
before any other processing; this is a pure elementwise map over
`[num_tokens, vocab]`.

**How vLLM exploits it.** `_softcap_logits` is `@torch.compile`d so the
cast/div/tanh/mul fuse into **one** elementwise kernel instead of four passes
over the large logits tensor (`diffusion_gemma.py:123-129`). It's applied
manually in `compute_logits`, leaving the `LogitsProcessor` to do only the
`lm_head` GEMM (`diffusion_gemma.py:244-252`, `330-334`).

### 3.9 Memory-bound concurrency tuning (`max_num_seqs = 8`)

**What the architecture enables/forces.** The diffusion sampler materializes
`[num_seqs, canvas_length, vocab]` fp32 transients, so concurrency is
**memory-bound**, not compute-bound (`config.py:149-159`).

**How vLLM exploits it.** When the user doesn't override `--max-num-seqs`, the
config caps it at 8 ("\>8 OOMs a single H200"), detected via the engine's
default sentinel (`config.py:151-159`). It also strips the
`generation_config.json` `max_new_tokens=256` cap so the model behaves like any
other (`config.py:161-173`).

**Further opportunity.** The fp32 `[seqs, canvas, vocab]` transient is the
binding constraint; a chunked/streamed softmax over the vocab dimension, or
bf16 intermediates where the tanh-stability argument allows, would raise the
concurrency ceiling.

### 3.10 FULL CUDA graphs via persistent, address-stable buffers

**What the architecture enables.** Denoising steps are shape-static (fixed
canvas, fixed concurrency cap), which is ideal for FULL CUDA graph capture —
*if* every runtime tensor lives at the same address the graph captured.

**How vLLM exploits it.** Two persistent buffers are central: `_inputs_embeds_buf`
and `_causal_buf`, both written **in place** so captured graphs and runtime
point at identical memory (`diffusion_gemma.py:813-828`). `prepare_dummy_inputs`
(capture) and `prepare_inputs` (runtime) deliberately return *slices of the same
buffer* (`diffusion_gemma.py:927-972`). The per-sequence causal flag (§3.2)
is likewise a persistent buffer for the same reason.

**Why it matters.** FULL graphs remove per-step launch overhead from the
denoising loop, which is otherwise dominated by many small kernel launches.

### 3.11 Logprobs: stash on convergence, emit on commit

**What the architecture enables.** The "real" distribution for a block exists
on the *converging* denoise step, but vLLM's output contract emits tokens on the
*commit* step. These are different steps.

**How vLLM exploits it.** Logprobs are computed once, on the converging step
(detected via the `is_encoder_phase False→True` flip), stashed per slot, and
reassembled/emitted on the subsequent commit step
(`diffusion_gemma.py:1092-1095`, `1304-1354`). Stale stashes from aborted
requests are purged on slot reuse (`diffusion_gemma.py:1103-1106`). This avoids
recomputing top-k logprobs over `[canvas, vocab]` twice.

---

## 4. Cross-Cutting Theme: "Behave Like Every Other Model"

The recurring design principle is to **map diffusion onto existing
abstractions** so the rest of vLLM needs no diffusion-awareness:

- Two modes → one backbone + a per-request flag (no separate encoder model).
- Bidirectional canvas → a per-sequence causal bit in the *existing* attention
  kernel (no new backend).
- Block emission → the *existing* speculative-decoding draft/accept plumbing
  (no new scheduler or output path).
- Per-step compute → *existing* `torch.compile` + FULL CUDA graph machinery,
  enabled by address-stable persistent buffers.

This is why the port can lean on KV cache management, batching, CUDA graphs, and
quantization (`SupportsQuant`, vision-tower quant gating at
`diffusion_gemma.py:193-221`) essentially for free.

---

## 5. Summary Table of Optimizations

| # | Optimization | Architectural enabler | Key location |
|---|---|---|---|
| 3.1 | Prompt KV computed once, read every step | YOCO single backbone, read-only decoder | `diffusion_gemma.py:5-11`, `419-429` |
| 3.2 | Mixed causal/bidirectional in one kernel | per-request attention regime | `diffusion_gemma.py:997-1025`; `triton_unified_attention.py:545-555` |
| 3.3 | Reuse spec-decode data path | block emission ≈ draft verification | `config/diffusion.py:11-18`; `diffusion_gemma.py:1142,1217` |
| 3.4 | Single fused `torch.compile` sampler step | uniform parallel denoising | `diffusion_gemma.py:466-642` |
| 3.5 | SC stored as `[..,hidden]` (~170× cut) | SC consumes `probs @ embed` only | `diffusion_gemma.py:711-718`, `623-631` |
| 3.6 | Phantom padding → branch-free sampler | near-uniform canvas length | `diffusion_gemma.py:1250-1259`, `537-543` |
| 3.7 | Entropy-bound accept + early convergence | all positions predicted in parallel | `diffusion_gemma.py:545-572`, `607-621` |
| 3.8 | Fused fp32 logits softcap | elementwise vocab-wide map | `diffusion_gemma.py:123-129` |
| 3.9 | `max_num_seqs=8` default | fp32 `[seqs,canvas,vocab]` transient | `config.py:149-159` |
| 3.10 | FULL CUDA graphs, persistent buffers | shape-static denoising steps | `diffusion_gemma.py:813-828`, `927-972` |
| 3.11 | Stash-on-converge / emit-on-commit logprobs | distribution ≠ emission step | `diffusion_gemma.py:1304-1354` |
| — | `k_eq_v` K=V sharing | no `v_proj` on full-attn layers | `diffusion_gemma.py:185-190`; `gemma4.py:567-589` |

---

## 6. Highest-Leverage Future Directions

1. **Fewer denoising steps** dominate everything. Adaptive per-request step
   budgets and tuned `entropy_bound`/`confidence_threshold` directly cut full
   backbone passes (§3.7).
2. **Raise the concurrency ceiling** by attacking the fp32
   `[seqs, canvas, vocab]` transient — chunked/streamed vocab softmax or
   reduced-precision intermediates (§3.4, §3.9).
3. **Cross-request prompt KV sharing**, made safe by the read-only decoder, to
   amortize encoding across requests with shared prefixes (§3.1).
4. **Canvas compaction** for truncated/phantom-padded blocks to reclaim wasted
   attention and sampler lanes (§3.2, §3.6).

---

*Report generated from a fresh clone of `vllm-project/vllm` (`main`). All
line references reflect that snapshot and may drift as the file evolves.*
