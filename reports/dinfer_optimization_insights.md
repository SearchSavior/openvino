# dInfer — Optimization Insights for dLLM Decode

Source: [`inclusionAI/dInfer`](https://github.com/inclusionAI/dInfer) (cloned `main`),
an efficient inference framework for diffusion LLMs (LLaDA / LLaDA-MoE / LLaDA2
block-diffusion). Technical report: arXiv 2510.08666. Headline results: >1,100
TPS at batch 1 on HumanEval (LLaDA-MoE, 8×H800); "10× faster than Fast-dLLM with
accuracy maintained." This is the most directly relevant prior art to our
DiffusionGemma/Xe2 work — same four components (model / diffusion iteration
manager / decoder / **KV-cache manager**). What follows is what transfers.

Caveat on architecture fit: dInfer targets the **LLaDA family**, where the entire
`gen_length` is materialized as `[MASK]` tokens up front and denoised block-by-block
left-to-right. DiffusionGemma is pure block-AR (it materializes only the current
canvas and commits via an encoder pass). The **prefix-cache** and
**recompute-only-the-active-block** insights transfer directly; the **suffix-cache**
piece is LLaDA-specific (it caches the not-yet-generated mask region).

---

## 1. NFE is the headline metric

dInfer counts `nfe` (number of forward evaluations) everywhere
(`generate_fastdllm.py`), and the entire framework is built to (a) reduce `nfe`
and (b) make each `nfe` cheaper. `nfe` is exactly our `S` (denoising steps per
block). This confirms the priority from our reports: **cutting denoising steps is
the master throughput lever**, and it is the number a dLLM framework optimizes
first.

## 2. Two KV-cache modes — exact vs approximate (the core lesson)

`DiffusionKVCacheManager` (`utils.py:383`) docstring states the problem outright:
"Because diffusion LLM uses bidirectional attention, the KV-cache has to be
updated frequently in the diffusion iterations. This class defines the … policy
[for] where keys and values can be updated and the frequency."

**Prefix cache (exact)** — `generate_with_prefix_cache`:
- One full forward at block start with `use_cache=True`, then **truncate the
  cache to `current_block_start`** (`past_key_values[...][:, :, :current_block_start]`).
- Denoising steps feed only `x[:, current_block_start:]` against the cached
  prefix. The committed prefix KV is computed once and reused across all steps of
  the block; the active region is recomputed each step.
- This is exactly our "frozen KV reused, canvas KV recomputed" model, and it is
  exact (the prefix is clean/committed).

**Dual cache (approximate, the speedup)** — `generate_with_dual_cache` +
`KVCache.update(replace_position=...)`:
- Caches KV for the prefix **and** the suffix. Each denoising step recomputes
  **only the current block** (`x[:, current_block_start:current_block_end]`) and
  **splices it into the cached full-sequence KV** via `slice_scatter` at
  `replace_position` (`utils.py:366-369`).
- Per-step forward cost drops to `O(block_length)` instead of `O(seq_len)`.
- Because bidirectional attention means the cached prefix/suffix KV technically
  drift as the block changes, the cache is **periodically refreshed**
  (`cache_update_freq`, `DiffusionKVCacheManager.require_update`) — a
  controlled-staleness approximation.

**This is the production answer to the open question in our KV-across-steps
section (the "progressive freezing" lever #5, flagged as needing validation):**
approximate cross-step KV reuse *works* and delivers the headline speedup, *if*
you bound the error with a refresh-frequency knob. dInfer ships exactly that knob.

## 3. `replace_position` / `slice_scatter` IS the "paged-attention variant"

The block-splice mechanism — compute only the active block's K/V, scatter them
into the cached sequence KV at the block position, attend against the rest from
cache (`utils.py:366-376`) — is precisely the "compute only the current block,
present cached KV for everything else" structure. It is how each `nfe` is made
cheap: the model forward sees `block_length` tokens, not the whole sequence.
Confirms that the right kernel structure for our decode is *active-block compute +
cached-KV attention*, with the block spliced in-place.

## 4. Parallel decoders = the cut-S algorithms (`parallel_strategy.py`)

dInfer offers a family of "how many tokens to accept per step" strategies. All
accept **multiple** tokens per step (more accepted ⇒ fewer steps), with a
progress guarantee:

- **Threshold** (`get_transfer_index_threshold`): accept every masked position
  with confidence ≥ `threshold`; the threshold is clamped to
  `max(confidence) − 1e-5` so **at least the argmax always transfers** (no
  zero-progress step). Same idea as DiffusionGemma's entropy-bound accept, using
  a confidence threshold instead of an entropy budget. Default 0.9.
- **Hierarchy** (`get_transfer_index_hierarchy_fast_v2`): accept the
  highest-confidence token in **each contiguous masked span**, so acceptances are
  spatially spread across the block (one per region) rather than clustered — plus
  any above `threshold`. Helps convergence/quality on long blocks.
- **Credit** (`CreditThresholdParallelDecoder._apply_credit_fusion`): accumulate
  confidence "credit" across steps before committing — reduces premature accepts.
- **Dynamic** (`get_transfer_index_dynamic`): factor-driven adaptive count.

Takeaway for us: the accept rule is a pluggable strategy; a confidence threshold
with an at-least-one-per-step (or one-per-span) guarantee is the simple, effective
baseline for cutting `S`.

## 5. Other concrete choices

- **Block length is small and a first-class knob**: 32–64 in their benchmarks
  (vs DiffusionGemma's canvas 256). Smaller block ⇒ cheaper per-step forward but
  more blocks (more sequential). A real `CL`-vs-`S` tradeoff to sweep, not assume.
- **Batched inference + tensor parallel**: batch 32 gives ~3–4× over batch 1
  (LLaDA2-flash table). Aggregate-throughput lever, orthogonal to per-request `S`.
- **Backend-pluggable** (vLLM / SGLang): the KV-cache manager abstracts the
  attention backend, with `backend='vllm'` paths using `slice_scatter`. Validates
  building the cache manager as a thin layer over whatever attention primitive we
  call.
- **Quantized model variants** supported (LLaDA2 quant) — precision reduction is
  in their throughput toolkit, matching our "lower precision / weight quant"
  lever.

---

## How this updates our reports

| Our prior claim | dInfer evidence |
|---|---|
| Cut `S` is the master lever (§ "what improves decode throughput") | Confirmed — `nfe` is *the* optimized metric |
| Frozen-KV reuse across steps (exact) | Confirmed — `prefix` cache |
| "Progressive freezing" approximate KV reuse #5 — *needs validation* | **Validated in production** — `dual` cache + `cache_update_freq` refresh; 10× over Fast-dLLM with accuracy held |
| User's "paged-attention variant: compute current block, present cached KV" | Confirmed — `replace_position` + `slice_scatter` block-splice |
| Confidence/entropy multi-token accept | Confirmed + extended — threshold / hierarchy / credit / dynamic, all with a progress guarantee |
| `CL` is a tunable (grow to amortize) | Confirmed it's a first-class knob; they run *smaller* (32–64), so sweep both directions |

**Net new insight:** the single highest-value thing dInfer demonstrates that we
had only hypothesized is that **approximate cross-denoising-step KV caching (recompute
only the active block, hold prefix+suffix stale, refresh on a frequency) is
production-viable and is where the order-of-magnitude speedup lives** — provided
the refresh frequency is exposed as an accuracy/speed dial. That should move from
"speculative #5, validate first" to a primary design element of our KV-cache
manager, with `cache_update_freq` as a first-class parameter.
