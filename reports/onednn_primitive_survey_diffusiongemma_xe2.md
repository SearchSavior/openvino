# oneDNN Primitive Survey for a DiffusionGemma Backend on Xe2 dGPUs

> Companion to `diffusiongemma_optimization_strategies.md`. Goal: pin down, for
> an Intel **Xe2 dGPU** target, which parts of the DiffusionGemma compute graph
> can **reuse** existing oneDNN primitives and which must be **authored in
> SYCL**, so the backend can match vLLM's serving approach (single backbone,
> mixed causal/bidirectional attention, fused parallel-denoising sampler).
>
> Surveyed tree: fresh clone of [`oneapi-src/oneDNN`](https://github.com/oneapi-src/oneDNN) (`main`).
> Key source roots:
> - `src/common/` — primitive descriptors (matmul, softmax, layer_normalization, eltwise, binary, reduction, **sdpa**, **gated_mlp**)
> - `src/gpu/intel/` — the **JIT/OpenCL GPU backend** (the optimized Xe2 path: `gemm/jit`, `sdpa/micro`, `gated_mlp/micro_horz`, `lnorm`, `softmax`, …)
> - `src/gpu/generic/sycl/` — the **portable SYCL reference backend** (ref kernels for eltwise, binary, reduction, layernorm, softmax, matmul, …)
> - `src/graph/backend/dnnl/patterns/` — **Graph API fusion patterns** (sdp, mlp, layernorm)

---

## 1. Executive Summary

oneDNN already covers the **dense linear-algebra and normalization core** of the
Gemma4 backbone with Xe2-tuned kernels, and — importantly — it ships two
*fused, attention-grade* internal primitives that map almost 1:1 onto
DiffusionGemma's hot path:

- **`sdpa`** (`src/gpu/intel/sdpa/micro.*`) — a micro-kernel fused
  scaled-dot-product-attention with scale, **attention-mask buffer**, GQA, KV
  quantization, and (via the Graph API) **logit soft-capping**.
- **`gated_mlp`** (`src/gpu/intel/gated_mlp/micro_horz.*`) — a fused
  gate/up/down MLP with a configurable activation, i.e. exactly Gemma's
  `down(gelu_tanh(gate(x)) * up(x))`.

What oneDNN does **not** have is anything resembling the **diffusion sampler**:
no sort, argmax, top-k, cumsum/scan, scatter/gather, RNG/Gumbel, or
entropy/convergence reductions as primitives (confirmed: none appear in
`dnnl_types.h`). The entire `_compiled_sample_step` from the vLLM port —
temperature schedule → Gumbel-max → entropy-bound acceptance →
stability/convergence → canvas write-back — has **no oneDNN equivalent and must
be authored in SYCL**. Likewise RoPE and token-embedding gather are not
primitives.

### Reuse / Author ledger

| DiffusionGemma op | oneDNN coverage | Verdict | Where |
|---|---|---|---|
| Token embedding lookup (gather) | none | **AUTHOR SYCL** | — |
| RMSNorm (pre/post norms) | `layer_normalization` + `dnnl_rms_norm` flag | **REUSE** | `dnnl_types.h:2311-2320` |
| QKV / O / gate / up / down projections (GEMM) | `matmul` (Xe2 systolic/DPAS JIT) | **REUSE** | `src/gpu/intel/gemm/jit` |
| `k_eq_v` (K reused as V) | weight-layout/packing choice, no V GEMM | **REUSE (host-side wiring)** | — |
| RoPE | none | **AUTHOR SYCL** | — |
| Attention (SDPA) | internal `sdpa` micro-kernel | **REUSE w/ gaps** | `src/gpu/intel/sdpa/micro.*` |
| ↳ `full_attention` layers (~1 in 6) | causal block-skip (`k0end`) + GQA | **REUSE (efficient)** | `micro.cl:536-545` |
| ↳ `sliding_attention` layers (majority) | present only ~`W` keys from our SYCL KV cache | **REUSE** — our KV cache bounds the window; primitive sees `W` keys | `kv_cache_interface.py:472` |
| ↳ mixed causal/bidirectional per-seq mask | `attn_mask` *buffer* (float additive) | **REUSE via mask buffer** | `sdpa_types.hpp:44-55` |
| ↳ logit soft-capping inside attn | Graph API `optional_soft_capping` | **REUSE via Graph API** | `patterns/sdp.cpp:148`, `utils.hpp:453` |
| MoE / gated MLP | internal `gated_mlp` (gate+act+mul+down) | **REUSE w/ gaps** | `gated_mlp/ref.hpp:49-59` |
| ↳ MoE expert routing / top-k gating | none | **AUTHOR SYCL** | — |
| Softmax / log-softmax (sampler probs) | `softmax_accurate` / `softmax_log` | **REUSE** | `dnnl_types.h` |
| Elementwise gelu_tanh / tanh / exp | `eltwise_gelu_tanh`, `eltwise_tanh`, `eltwise_exp` | **REUSE** | `dnnl_types.h` |
| Binary add/mul/sub (residual, masking, SC) | `binary_*` | **REUSE** | `dnnl_types.h` |
| Reductions (entropy mean/sum) | `reduction_mean/sum`, `norm_lp_*` | **REUSE** | `dnnl_types.h` |
| Final logit softcap (LM head) | `eltwise_tanh` + scale as matmul post-op | **REUSE (post-op)** | `matmul/gemm.hpp:55` |
| **Diffusion sampler** (Gumbel-max, entropy-bound accept, sort, cumsum, scatter, stability, convergence) | **none** | **AUTHOR SYCL** | — |
| RNG (uniform → Gumbel, random renoise tokens) | none (stochastic-rounding only) | **AUTHOR SYCL** | — |
| Self-conditioning soft-embed (`probs @ embed`) | `matmul` | **REUSE** | — |

**Bottom line:** ~70% of the *FLOPs* (all GEMMs, attention, MLP, norms) are
reusable oneDNN primitives with Xe2-optimized kernels; the *new SYCL code* is
concentrated in (a) the diffusion sampler, (b) RoPE, (c) embedding gather, and
(d) the per-sequence attention-mask construction that encodes
encoder-vs-denoise phase.

> **The core thesis (§1A):** unlike an autoregressive model, DiffusionGemma
> keeps the **XMX/DPAS systolic arrays saturated during *decode* as well as
> prefill**, because every denoising step is a fat GEMM over the whole canvas,
> not a skinny GEMV over one token. This is *the* reason the model is an unusually
> good fit for Xe2, and it reorders the optimization priorities.

---

## 1A. Why Xe2/XMX Is an Unusually Good Fit (the core thesis)

Xe2 dGPUs derive most of their FLOPs from **XMX** (Xe Matrix Extensions — the
DPAS systolic arrays), reached in oneDNN through the `src/gpu/intel/gemm/jit`
systolic path. XMX is only efficient on **GEMM-shaped** work with a large enough
`M` (rows) to fill the systolic tiles; on **GEMV-shaped** work (`M≈1`) it is
starved and the kernel becomes **memory-bandwidth-bound** on weight reads.

That distinction is exactly where autoregressive (AR) and diffusion decoding
diverge:

| | AR decode | DiffusionGemma decode |
|---|---|---|
| Tokens processed per forward pass | **1 per request** (`M = num_reqs`, ≤8) | **whole canvas per request** (`M = num_decode × canvas_length`) |
| GEMM shape | skinny **GEMV** | fat **GEMM** |
| Weight reuse per pass | each weight used ~once → low arithmetic intensity | each weight reused `canvas_length×` → high arithmetic intensity |
| Bottleneck | **memory bandwidth** (XMX idle) | **XMX compute** (arrays saturated) |
| Sequential passes to emit N tokens | **N** (one per token) | **(N/canvas_length) × (S+1)** (blocks × denoise+commit) |

**Grounded in the code.** A decode step runs the backbone over the full canvas:
`per_req_nlogits == canvas_length` and the model processes
`num_decode × canvas_length` tokens (`diffusion_gemma.py:474, 603`,
`prepare_inputs`). The canvas default is **256** (`diffusion_gemma.py` /
`configs/diffusion_gemma.py:31`, `models/config.py:144`). So with
`max_num_seqs=8` the decode-time GEMM `M` is up to **8 × 256 = 2048** — squarely
in the XMX-efficient regime — versus `M ≤ 8` for AR decode.

### What is and isn't parallel (the limit of the speedup)

The XMX win comes from **spatial** parallelism only. There are three dependency
axes, and bidirectional attention removes just one of them:

1. **Across canvas positions (spatial): fully parallel.** All `canvas_length`
   positions of the *current* block are refined together — this is what makes
   the decode matmuls fat GEMMs. This is the entire XMX advantage.
2. **Across denoising steps (temporal): sequential.** Step `t+1` consumes step
   `t`'s renoised canvas plus self-conditioning embeds, so the `S` steps of a
   block cannot be parallelized. `S` is *reducible* (entropy-bound acceptance +
   confidence/stability early convergence, `diffusion_gemma.py:545-621`) but
   bounded below by however many refinements the block needs.
3. **Across canvas blocks (autoregressive): sequential and IRREDUCIBLE.** This
   is "block diffusion" (the registry name is `DiffusionGemmaForBlockDiffusion`):
   block `k+1`'s context is block `k`'s committed tokens, made visible only after
   the commit step writes them to KV via the causal/encoder pass
   (`is_encoder_phase` cycle: `:732` → `:1146` → `:619` → commit `:603`). Block
   `k+1` cannot begin until block `k` commits. **Bidirectional attention does
   not weaken this** — it is confined to the live canvas; past blocks are frozen
   KV attended to *causally*.

So we **cannot** reduce sequentiality below the block structure. To emit `N`
tokens we pay `⌈N / canvas_length⌉` sequential blocks, each `S+1` sequential
passes ≈ `(N/canvas_length)·(S+1)` total. This beats AR's `N` sequential passes
**only if `S+1 < canvas_length`** (e.g. `S≈10–48`, `CL=256` → ~5–25× fewer
*passes*) — and even when the pass count isn't lower, each pass does `CL`
positions of XMX-efficient work instead of one GEMV. The two real levers are
therefore: **shrink `S`** (axis 2) and **grow `canvas_length`** to cut the block
count (axis 3) — the latter bounded by XMX occupancy, the fp32 sampler
transient, and whether a larger block needs more steps to converge.

### Bounding per-step cost: tile the canvas, freeze the past in KV

The reason this is tractable at all is the YOCO encoder/decoder split, which is
exactly a "compute only the current canvas, keep previous canvases as frozen KV"
scheme:

- **Commit step = encoder pass** (`is_encoder_phase=True`, causal): the finalized
  canvas is written to the paged KV cache — now **frozen K/V**.
- **Denoise steps = decoder pass** (`is_encoder_phase=False`, bidirectional,
  *read-only* against the cache, `diffusion_gemma.py:7`): the **only query tile
  computed is the current canvas** (`num_decode × canvas_length`,
  `:474, 603`). It attends to `[frozen past KV from cache] +
  [its own freshly-computed canvas K/V]`. `prepare_attn` feeds the kernel the
  full-context `block_tables`/`slot_mappings`/`seq_lens` — standard paged KV,
  identical to AR; the past is encoded **once**, not re-encoded per step.

Per-denoise-step cost under this scheme:

| Work | Cost / step | Scales with seq len? |
|---|---|---|
| Projection + MLP GEMMs (qkv/o/gate/up/down) | `O(canvas_length · d)`, **M = CL fixed** | **No** |
| Attention — **global** layers | `O(canvas_length · context · d)` | Yes (KV grows) |
| Attention — **sliding** layers | `O(canvas_length · min(context, W) · d)` | Bounded by `W` |

The key point: the **XMX-bound bulk (projections/MLP) is pinned at `M =
canvas_length` regardless of total sequence length** — only attention's KV
dimension grows, and on Gemma's majority sliding layers it is capped at `W`.
Without the KV cache, every denoise step would re-encode the prefix at `O(seq)`
— the cache is what keeps decode at canvas-bounded compute. (The live canvas's
own K/V change every step as it is refined, so they are recomputed each step for
the in-canvas bidirectional self-attention and only the committed result is
persisted — but that is only `CL` wide, so it is cheap.) For sliding layers, our
KV cache presents only the last `W` keys, so most of the frozen past is simply
never read (§3.2).

### Consequences (these reorder the priorities)

1. **The matmul / gated_mlp / SDPA reuse is even more valuable than the FLOP
   share suggests.** Those *are* the XMX kernels, and here they run hot in
   **both** phases. Targeting the DPAS systolic GEMM path for *every* projection
   (qkv/o/gate/up/down/lm_head) pays off in decode, not just prefill.

2. **Efficiency no longer depends on cross-request batching.** AR needs many
   concurrent requests to fill the GEMM `M`; DiffusionGemma fills `M` from the
   canvas *within a single request*. This is fortunate, because the fp32
   `[seqs, canvas, vocab]` sampler transient caps `max_num_seqs` at 8
   (`models/config.py:149-159`) — we don't need high concurrency for XMX
   occupancy.

3. **The optimization target shifts from "bandwidth/GEMV latency" to "denoising
   steps `S` × XMX tile occupancy".** The AR playbook (KV-cache bandwidth,
   flash-decoding GEMV kernels, speculative decode to dodge GEMV) is largely
   irrelevant. What matters is: (a) minimize `S` — entropy-bound acceptance and
   confidence/stability early convergence (`diffusion_gemma.py:545-621`); and
   (b) keep each step's canvas GEMM in the efficient DPAS regime.

4. **Attention is also GEMM-shaped in decode.** Because a step has
   `canvas_length` queries (not 1), the SDPA `ugemm` KQ/VS matmuls are
   DPAS-friendly — unlike AR flash-decoding, which is a batch-1 GEMV. This makes
   the `sdpa` micro-kernel a strong fit in decode too. Sliding-window efficiency
   falls out of **our own KV cache presenting only ~`W` keys** for sliding layers
   (§3.2) — so the primitive sees only `CL` queries × `min(context, W)` keys.

5. **More raw FLOPs, but at far higher utilization — *not* unconditionally
   fewer sequential steps.** Diffusion does ~`(S+1)×` the token-passes of AR per
   block, yet runs them as saturated GEMMs instead of starved GEMVs. The
   *sequential-pass* advantage exists only when `S+1 < canvas_length` (see the
   parallelism section); the block-autoregressive floor `⌈N/canvas_length⌉` is
   irreducible. The robust win is the **utilization** one (GEMM vs GEMV), which
   holds regardless; the latency/pass-count win is conditional on `S`.

**Net:** the backend should treat decode like a second prefill — same XMX GEMM
kernels, same tiling concerns — and spend its optimization budget on cutting `S`
(within-block) and maximizing DPAS occupancy of the canvas GEMM, not on the
bandwidth-oriented tricks an AR engine needs. The block-to-block autoregression
is a hard sequential floor, so don't bank on parallelizing across blocks.

### Optimizing KV across denoising steps

Within one block, the `S` denoise steps see **two KV populations** that behave
oppositely:

| | Frozen KV (prompt + committed blocks) | Canvas K/V (current block) |
|---|---|---|
| Changes across the `S` steps? | **No** — identical every step | **Yes** — canvas tokens are re-sampled each step |
| Computed | incrementally at commit → cache | recomputed every denoise step |
| Persisted? | yes (read-only thereafter) | no — regenerated until commit |

This is confirmed directly by the HF reference (transformers 5.12.1). The
decoder attention is read-only against the encoder cache and recomputes + concats
canvas K/V every step:
`DiffusionGemmaDecoderTextAttention` "doesn't update the KV cache in the forward
pass … read-only encoder KV cache" (`modeling_diffusion_gemma.py:374-377`);
`key_states = self.k_proj(hidden_states)` … `torch.cat([encoder_key_states,
key_states], dim=2)` with no `.update()` (`:436-452`); `is_causal = False`
(`:384`). The generation loop is block-autoregressive — per block it "encode[s]
all previous tokens using the encoder, to get the KV cache" then runs the inner
denoising loop against that cache (`generation_diffusion_gemma.py:558-574`), and
the live canvas is not cached (`:670`). The per-block encode is *incremental*
(only `unprocessed_input_ids`, `:721`). Note `k_eq_v` is on the **global** layers
(`v_proj = ... if self.is_sliding else None`, `:402-406`).

**Reframe — frozen KV is read `S×` but that is *not* the bottleneck.** The
tempting target is the `S×` re-read of frozen KV per block. But a denoise step
issues **`CL` queries at once**, so each frozen-KV read is amortized across the
whole canvas: arithmetic intensity ≈ `CL · gqa_ratio / dtype_bytes` (~hundreds
of FLOP/byte at `CL=256`), well above Xe2's ridge point. So **frozen-KV
attention is compute-bound**, not bandwidth-bound — diffusion has none of AR
decode's KV-bandwidth wall (where intensity ≈ 1–2 FLOP/byte, `CL=1`). The real
cross-step waste is **redundant FLOPs** (`S` full forward passes over `CL`
tokens, including already-locked positions) and, secondarily, HBM traffic/power.

Optimizations, ranked (exactness noted):

1. **Canvas K/V on-chip, never in the paged HBM cache (exact; design).** During
   denoise the canvas K/V are read-only-this-step and discarded; only the commit
   pass writes the canonical *causal* KV. Compute canvas K/V in SLM/registers per
   step (flash-style) and allocate no paged slots for them — no HBM churn for the
   ephemeral block.
2. **Frozen-KV L2 residency across the `S` back-to-back steps (exact;
   scheduling).** The `S` steps run consecutively over identical frozen KV.
   Schedule the per-block loop to keep frozen KV hot in L2 → `1×` HBM + `S×` L2
   instead of `S×` HBM. Cuts traffic/power even though we are compute-bound.
3. **Frozen-KV quantization (near-exact; bandwidth/footprint).** oneDNN `sdpa`
   already supports KV scales/zero-points (`kq_scales`/`vs_scales`,
   `sdpa_types.hpp:82-85`). int8/fp8 frozen KV is *more* valuable than in AR
   because the saving multiplies over the `S×` reads and the long residency.
4. **Sliding-window cache bound (exact).** Local layers present only `W` keys
   (§3.2), capping frozen KV and its `S×` reads regardless of context length.
5. **Progressive freezing / query-set pruning (APPROXIMATE — biggest FLOP
   save; must validate accuracy).** The only lever on the dominant cost. The
   sampler already knows which positions are locked (entropy-bound accept mask +
   stability/convergence, `diffusion_gemma.py:545-621`). Once a position locks,
   its committed token is fixed; if we freeze its K/V and stop recomputing its
   query/MLP, the active query set `M` shrinks from `CL` toward 0 as the block
   converges, cutting the canvas projection/MLP GEMMs and the self-attention.
   *Catch:* bidirectional coupling means a locked position's true K/V would still
   drift as neighbors change, so this is an approximation — tolerable only if it
   holds up against full-recompute accuracy. Validate before adopting.
6. **Skip embed + self-conditioning recompute for unchanged positions (exact;
   minor).** Gate the embedding lookup and SC MLP on the sampler's change mask.

What is **not** possible:
- *Cache frozen-KV attention scores across steps* — `Q_canvas` changes every
  step, so `Q·K_frozen` cannot be reused even though `K_frozen` is constant.
- *Batch the `S` steps* — strictly sequential (step `t+1` consumes step `t`'s
  renoised canvas + self-conditioning).
- *Exactly skip locked-position recompute* — bidirectional attention couples all
  positions; only the approximate freezing in (5) gets around it.
- *Reuse the last denoise step's canvas K/V for the commit write* — denoise
  computes them under *bidirectional* attention; the commit needs *causal* KV
  for future blocks, a genuinely different result.

---

## 2. oneDNN Surface Area on Xe2

### 2.1 Two GPU backends, pick per op

1. **`src/gpu/intel/`** — the **performance** backend: JIT (nGEN) and OpenCL
   kernels, systolic/DPAS GEMM, micro-kernel SDPA and gated_mlp. This is where
   the Xe2 speed lives. Xe2 is a first-class arch in the dispatcher
   (`gpu_arch_t::xe2`, `compute/device_info.hpp:47-88`), with per-arch
   subgroup size, GRF, EU/thread queries.
2. **`src/gpu/generic/sycl/`** — a **portable SYCL reference** backend: `ref_*`
   kernels for eltwise, binary, reduction, layer/group norm, softmax, matmul,
   reorder, etc. (`src/gpu/generic/sycl/`). Useful as a correctness reference and
   as a template for *authoring our own SYCL kernels* in the same idiom.

For Xe2 we want the `intel/` kernels for the heavy ops; the `generic/sycl/`
backend is the model for the custom kernels we must write.

### 2.2 Primitive API vs Graph API — a crucial distinction

- `matmul`, `softmax`, `layer_normalization`, `eltwise`, `binary`,
  `reduction`, `reorder`, `sum`, `concat` are **public primitives**
  (`include/oneapi/dnnl/dnnl.hpp`) — instantiate directly.
- **`sdpa` and `gated_mlp` are *internal* primitives** — they have
  `src/common/*_pd.hpp` descriptors and `src/gpu/intel/...` implementations but
  **no public header** (confirmed: nothing in `dnnl.hpp`). They are reached
  through the **Graph API** fusion patterns
  (`src/graph/backend/dnnl/patterns/sdp.cpp`, `mlp.cpp`). So to reuse fused
  attention/MLP we express the subgraph through `dnnl::graph` and let the
  backend fuse-and-dispatch to the micro-kernels, *or* call the internal PDs
  directly if we vendor into the build.

This shapes the integration: **the backbone should be built as a oneDNN Graph**
so SDPA + gated_mlp fusion (and softcap) are picked up automatically, with our
custom SYCL kernels (RoPE, sampler, gather) slotted in between graph partitions.

---

## 3. Op-by-Op Mapping of the DiffusionGemma Graph

### 3.1 REUSE — covered by Xe2-optimized primitives

**RMSNorm — reuse `layer_normalization` with the RMS flag.**
oneDNN added a dedicated RMS path: `dnnl_rms_norm = 0x20` "the mean is
considered zero, and RMS norm is used instead of variance"
(`dnnl_types.h:2311-2320`). This covers `pre_norm`, `post_norm`,
`q_norm`/`k_norm`/`v_norm`, and the final model norm. Note the SC `post_norm`
uses `has_weight=False` (`diffusion_gemma.py:78`) — supported by simply not
binding a scale argument.

**All projections — reuse `matmul` on the Xe2 systolic/DPAS path.**
qkv_proj, o_proj, gate/up/down, lm_head, and the SC matmuls are GEMMs. The
`intel/gemm/jit` backend provides DPAS systolic kernels; `matmul` exposes
**post-ops** (`matmul/gemm.hpp:55`) so we fuse bias, residual `binary_add`, and
activation `eltwise` directly into the GEMM epilogue — matching the vLLM port's
fusion intent.

**`k_eq_v` — reuse, host-side.** The "no v_proj, V = K" trick
(`diffusion_gemma.py:185-190`) is purely a weight-loading/packing decision: we
skip the V GEMM and alias K's output as V into SDPA. No new kernel.

**Final logit softcap — reuse `eltwise_tanh` as a matmul post-op.**
`tanh(logits/cap)*cap` = scale → `eltwise_tanh` → scale, appendable to the
lm_head matmul post-op chain (same fusion vLLM does with `_softcap_logits`,
`diffusion_gemma.py:123-129`).

**Sampler elementwise/reduction sub-steps — reuse primitives.** Inside the
custom sampler we can still call primitives for the heavy vectorized pieces:
`softmax_log` for `log_softmax`, `eltwise_exp` for probs, `reduction_mean`/`sum`
for entropy aggregation, `binary_mul`/`add` for masking and residual SC.

### 3.2 REUSE WITH GAPS — fused attention & MLP

**SDPA — reuse `sdpa` micro-kernel; the gaps are mask construction & softcap wiring.**
The internal `sdpa` PD (`src/common/sdpa_types.hpp`, `src/gpu/intel/sdpa/micro.*`)
supports everything the backbone needs:
- attention **scale** (`with_attn_scale`, `invert_scale`),
- **GQA** (4D Q/K/V, separate KV heads) — needed for Gemma's grouped KV and the
  `k_eq_v` layers,
- **KV quantization** (`kq_scales`, `vs_scales`, zero-points) for future INT KV
  cache,
- a **`d_max` head-dim limit** (`micro.hpp:347`) — must verify Gemma's
  `global_head_dim` fits the micro-kernel's supported range on Xe2.

The three gaps:
0. **Sliding-window (local) attention — handled by our own KV cache, not a
   concern for the oneDNN call.** Gemma4 is *hybrid*: most layers are
   `sliding_attention` (local window `W`), interleaved with a few
   `full_attention` layers (`gemma4.py:435-437, 560-562`; typically a 5:1 ratio).
   Since we author the KV cache from scratch in SYCL, **we decide which keys to
   present to the attention primitive.** For a sliding layer we store/present
   only the last ~`W` keys (the well-known per-layer sliding-window KV design;
   vLLM's `SlidingWindowSpec` allocates `cdiv(W, block_size)+1` blocks per layer,
   `vllm/v1/kv_cache_interface.py:472`, `simple_kv_offload/manager.py:217-218`).
   Whatever attention primitive we then call — oneDNN `sdpa`, or our own
   matmul+softmax+matmul in SYCL — sees only those ~`W` keys, so attention stays
   `O(canvas_length · W · d)` and **KV memory is bounded to `W`** (the larger
   prize). Global layers present the full context. This is purely a
   data-presentation decision in our KV cache; no attention-kernel windowing
   logic is required. Canvas tiling separately bounds the *query* side to
   `M = canvas_length` (§1A); cache-level windowing bounds the *key* side.

1. **Mixed causal / bidirectional per request.** SDPA's built-in masks are
   `top_left` / `bottom_right` causal or a **buffer mask**
   (`dnnl_attn_mask_buffer`, `sdpa_types.hpp:44-55`; 4D, `mask_q/k_index`).
   It has **no per-sequence boolean "causal flag"** like vLLM's `_causal_buf`
   threaded into the Triton kernel (`diffusion_gemma.py:997-1025`). The clean
   reuse path on Xe2 is to **materialize a float additive `attn_mask` buffer**:
   `0` where attended, `-inf` where masked, built per batch so encoder rows are
   causal and denoise rows are bidirectional. **Building that mask buffer is
   custom SYCL** (cheap, `[batch, 1, q, k]`), but the attention math itself is
   reused. (Alternatively: patch the micro-kernel to accept a per-seq causal
   predicate — more work, closer to vLLM.)
2. **Logit soft-capping inside attention** (Gemma `attn_logit_softcapping`).
   Present only through the **Graph API** `optional_soft_capping`
   (`patterns/sdp.cpp:148-151`, `patterns/utils.hpp:453-464`;
   `kernels/sdp_decomp.cpp:237`). So to get in-attention softcap we must drive
   SDPA through the Graph API rather than the raw internal PD.

**Gated MLP — reuse `gated_mlp`; gap is MoE routing.**
`gated_mlp` fuses exactly Gemma's MLP: gate matmul → `eltwise(activation)` →
`binary_mul` with up → down matmul (`gated_mlp/ref.hpp:49-59`,
intel `micro_horz.cl`). `activation()` is configurable, so `gelu_tanh`
(`diffusion_gemma.py:90`) is a parameter, not new code. **Gap:** DiffusionGemma
is **MoE** (`enable_moe_block`, `.moe.experts.`,
`diffusion_gemma.py:163-166`, `gemma4.py:631`). `gated_mlp` is a *dense*
gated MLP — it has no expert **router / top-k gating / token dispatch**. That
routing layer (argmax/top-k over expert logits, scatter tokens to experts,
combine) is **custom SYCL**, after which each expert can reuse `gated_mlp` or
plain `matmul`.

### 3.3 AUTHOR IN SYCL — no oneDNN coverage

**1. The diffusion sampler (`_compiled_sample_step`).** This is the single
biggest authoring task and the model's defining hot loop
(`diffusion_gemma.py:466-642`). None of its primitives exist in oneDNN:
- **RNG → Gumbel noise** and **random renoise tokens** (oneDNN only has
  stochastic-rounding bias, not general RNG). Need a SYCL Philox/counter-based
  RNG.
- **Gumbel-max argmax** over `[decode, canvas, vocab]` — no `argmax` primitive.
- **Per-position entropy**, then **sort + cumsum + cummax** for the
  entropy-bound acceptance mask (`diffusion_gemma.py:545-551`) — no sort/scan
  primitives.
- **scatter** of the accept mask, **circular history writes**, **stability
  mismatch reduction**, **convergence flag** logic
  (`diffusion_gemma.py:579-637`).
- **canvas / argmax_canvas / draft_tokens write-back** (in-place state).

  This should be authored as a **small number of fused SYCL kernels** over the
  pre-allocated per-request state buffers, mirroring the vLLM design of one
  fused step with no host syncs. The `generic/sycl` `reduction`/`simple_reduction`
  kernels are good structural templates for the entropy/stability reductions.

**2. RoPE** (`gemma4.py:485-534`) — rotary position embedding has no oneDNN
primitive; custom SYCL applied to Q/K before SDPA.

**3. Token-embedding gather** (`embed_input_ids`) and **multimodal embed merge**
— gather/scatter by id; custom SYCL (or a `reorder`-based indirection).

**4. Per-sequence attention-mask buffer builder** (§3.2 gap 1) — custom SYCL to
fill the float additive mask from the per-request `is_encoder_phase` flags and
`query_start_loc`.

**5. MoE router/dispatch** (§3.2 gap 2).

---

## 4. Matching the vLLM Approach

vLLM's wins (from the companion report) translate to this oneDNN/Xe2 plan:

| vLLM strategy | oneDNN/Xe2 realization |
|---|---|
| Single backbone, encoder/decoder modes | One graph; **phase = the attn-mask buffer** we build, not two models |
| Mixed causal/bidirectional in one batch | **float additive `attn_mask` buffer** consumed by the `sdpa` micro-kernel (custom builder, reused attention) |
| Reuse spec-decode data path | Lives in the serving/runtime layer (our backend, à la vLLM), not oneDNN; oneDNN sees fixed `[seqs, canvas, ...]` tensors |
| One fused `torch.compile` sampler step | **One/few fused SYCL kernels** over pre-allocated state buffers (no oneDNN primitive) |
| SC stored as `[..,hidden]` | `matmul` (`probs @ embed`) + `gated_mlp`/`matmul`; storage is host buffer policy |
| Fused fp32 softcap | `eltwise_tanh` matmul **post-op** |
| Persistent buffers for CUDA graphs | Pre-allocated USM + oneDNN primitive caching / persistent scratchpad; Xe2 immutable command lists |
| GEMM/MLP/attention throughput | DPAS systolic `matmul`, `sdpa` micro, `gated_mlp` micro — all Xe2-tuned |

The division of labor is clean: **oneDNN owns the FLOP-heavy dense graph
(GEMM + SDPA + gated_mlp + norms), we own the diffusion control plane (sampler,
RoPE, gather, mask/router builders) in SYCL.**

---

## 5. Open Verification Items (Xe2-specific)

1. **SDPA `d_max` vs Gemma head dims.** Confirm `global_head_dim` /
   sliding-vs-full head dims fit the Xe2 micro-kernel's `d_max`
   (`sdpa/micro.hpp:347`, `configs.cpp`). If not, fall back to
   matmul+softmax+matmul (all reusable) for the oversized layers.
1b. **Sliding window is a KV-cache design task in our SYCL code.** Author the
   per-layer sliding-window KV cache (mirror vLLM's `SlidingWindowSpec`) so
   sliding layers store/present only ~`W` keys — this bounds both attention
   compute and KV memory, and whatever attention primitive we call sees only `W`
   keys. No oneDNN kernel change is involved.
2. **Bidirectional masking cost.** Validate that a float additive mask buffer
   on Xe2 doesn't defeat the micro-kernel's causal-skip optimization for the
   *encoder* rows; if it does, consider a per-seq causal predicate patch.
3. **In-attention softcap requires Graph API.** Decide between Graph-API-driven
   SDPA (gets softcap + fusion for free) vs. direct internal-PD calls (more
   control, must re-add softcap as a post-step).
4. **MoE.** Confirm whether the target checkpoint actually activates
   `enable_moe_block`; if dense-only, `gated_mlp` covers the whole MLP with no
   router needed.
5. **gated_mlp activation = gelu_tanh** availability on the Xe2 `micro_horz`
   path (the algo enum exists; verify the JIT kernel supports it).

---

## 6. Suggested Authoring Order

1. **Mask-buffer builder** (SYCL) — unblocks reusing `sdpa` for the
   mixed-phase batch; smallest custom kernel, highest leverage.
2. **RoPE + embedding gather** (SYCL) — standard, well-understood.
3. **Backbone as a oneDNN Graph** — wire `matmul`/`gated_mlp`/`sdpa`/RMSNorm,
   get softcap + fusion via the Graph API.
4. **Diffusion sampler** (SYCL) — the large, model-defining piece; build it as
   a fused step over persistent USM state buffers, leaning on `softmax_log` /
   `reduction` primitives where the shapes are regular.
5. **MoE router** (SYCL) — only if the checkpoint is MoE.

---

*Surveyed from a fresh clone of `oneapi-src/oneDNN` (`main`); line references
reflect that snapshot. Target: Intel Xe2 dGPUs via the `src/gpu/intel` JIT/OpenCL
backend with `src/gpu/generic/sycl` as the reference/authoring template.*
