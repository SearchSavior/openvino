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

---

## 6. More riches (second pass — `generate_uniform.py`, `diffusion_runner.py`, `generate_dist.py`)

The first pass covered the Fast-dLLM-derived path. The `generate_uniform.py`
pipeline (the larger, composable engine: `DiffusionIteration` × `Decoder` ×
`Cache` × `BlockRunner`) and the runner/distributed code carry several more
optimizations worth stealing.

### 6.1 Vicinity cache — distance-aware approximate KV (the refined dual cache)

`VicinityCacheIteration` (`generate_uniform.py:671`) sharpens the dual cache:
instead of holding *all* non-block KV stale, it **refreshes KV only in a window
around the active block** — `[block_start − prefix_look, block_end + after_look]`
— and recomputes only that window each step (`model(window_input,
replace_position=...)`, `:704-706`), holding far-away KV frozen. `warmup_steps`
full forwards run first to seed the cache. The window can be wider than the block.

This is the principled version of our "progressive freezing": KV drift from
bidirectional attention is **local**, so refresh near neighbors (which actually
drift) and freeze distant tokens (which don't). `prefix_look`/`after_look` are the
accuracy/speed dial — directly more targeted than a uniform `cache_update_freq`.
For us: the SYCL KV-cache manager should support a *windowed refresh*, not just
all-or-nothing.

### 6.2 Iteration smoothing — self-conditioning + annealed acceptance

`IterationSmooth` (`:558`) does two things per iteration:
- **Self-conditioning via continuation weight:** next step's input embeds blend
  the previous step's logits back in through a learned `h2e`, weighted by
  `iter_cont_weight = min(cont_weight_init + growth·iter_no, cont_weight)` — the
  weight **ramps up** over iterations (`:582, 592`). This is DiffusionGemma's
  self-conditioning, but with a *scheduled* feedback strength.
- **Annealed acceptance threshold:** `iter_threshold = max(1 −
  iter_no·threshold_decay, base_threshold)` (`:583`) — strict early (accept only
  very confident), relaxing as iterations progress. This actively shrinks `S` by
  loosening acceptance over time.

The annealed threshold is a lever DiffusionGemma's fixed entropy bound does not
have. Cheap to add to any confidence-threshold decoder; a likely `S` win.

### 6.3 Dynamic loop unroll (`expected_tpf`, `maximum_unroll`)

`BlockRunner.decode` (`:102-105`) computes
`unroll_k = clamp(num_masked // expected_tpf, 1, maximum_unroll)` and runs that
many denoising forwards before re-checking the block. `expected_tpf` = expected
**tokens accepted per forward** (i.e. `CL/(S+1)` from our analysis, made an
explicit target); unrolling amortizes Python/launch overhead and fits a fixed
number of forwards into a captured graph. For us: drive the denoise loop by an
expected-acceptance estimate and unroll to fill a CUDA-graph/command-list.

### 6.4 Per-sequence early exit within a batch

`select_undecoded(..., writeback=True)` (`:99-115`) filters out sequences whose
block has fully decoded, so the active compute set shrinks mid-block and finished
sequences don't waste forwards. Important for batched throughput — the batch's
effective `M` tracks only unfinished sequences.

### 6.5 CUDA-graph capture with shape bucketing

`ModelRunner` (`diffusion_runner.py:127`) captures device graphs over discrete
shape buckets: `prefill_lengths=[64,96,128]`, `decoding_lengths=[32]`, and
**`cache_lengths` buckets** (+ `align_exp2` to round shapes). Because the cache
grows block-by-block, they capture a graph per cache-length bucket and replay the
matching one (`forward` → `graph_runner.can_run(...)` → `replay`). It also uses
`torch.compile` with `fx_graph_cache`. For us: the growing-context decode needs
graphs/command-lists captured per cache-length bucket, not a single static shape.

### 6.6 Sequence-parallel diffusion forward (`generate_dist.py`)

Distributed decode splits the **sequence dimension** across ranks: each rank runs
`model(x[:, rank*part:(rank+1)*part])` and the shards are combined with
`all_gather_into_tensor(logits)` (`generate_dist.py:136-140, 263-267`). This is a
dLLM-specific parallelism: a denoise step has many tokens to score at once, so you
can shard them across devices — something AR decode (one token) cannot do. It
composes with the prefix/block caches (`generate_with_cache`,
`generate_block_cache`). Relevant if we ever scale one request across multiple
Xe2 tiles/cards.

### 6.7 Cross-block KV update at seams (`use_cross_block`)

At a block boundary the previous block's KV — computed while it was still being
denoised bidirectionally — is refreshed before the next block proceeds
(`need_cross_block_update`, `cross_block_*`, `generate_uniform.py:243-277`). This
fixes the staleness at the seam between a just-finalized block and the new one.
A correctness detail any windowed/approximate cache must handle.

### 6.8 Composable architecture + analytic cost model

The engine is built as orthogonal, swappable parts — `DiffusionIteration`
(plain / shift / smooth / vicinity / smooth+vicinity), `ParallelDecoder`
(threshold / credit / hierarchy / fixed), `KVCache`/`DiffusionKVCacheManager`
(prefix / dual), `BlockRunner` — so e.g. `IterSmoothWithVicinityCache` just
composes 6.1 + 6.2. There is also an analytic op-count model
(`calculate_op_num(..., cache_length=...)`, `utils.py:41`) used to reason about
cost including the growing cache. Worth mirroring: keep iteration/decoder/cache as
independent, swappable components in our backend.

### Updated takeaways for our backend

| dInfer mechanism | Action for our SYCL/oneDNN Xe2 backend |
|---|---|
| Vicinity cache (6.1) | KV-cache manager must support **windowed refresh** (`prefix_look`/`after_look`), not just full vs none |
| Annealed threshold (6.2) | Add a step-scheduled acceptance threshold to the decoder — cheap `S` win |
| Dynamic unroll (6.3) | Drive denoise loop by expected accepts; unroll to fill a captured graph |
| Per-seq early exit (6.4) | Compact the active batch mid-block |
| Cache-length-bucketed graphs (6.5) | Capture command-lists per cache-length bucket (context grows) |
| Sequence-parallel forward (6.6) | Multi-tile scaling option: shard the canvas across Xe2 devices, all-gather logits |
| Cross-block update (6.7) | Refresh the previous block's KV at the seam in any approximate-cache design |
| Composable parts (6.8) | Keep iteration/decoder/cache swappable |

