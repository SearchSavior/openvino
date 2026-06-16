# Session log — Cached activation pools & LLM memory consumption (GPU / CPU / GenAI)

> Full transcript of an investigation into where cached activation pools and KV-cache
> preallocation drive memory consumption for LLMs across the OpenVINO `intel_gpu` plugin,
> the `intel_cpu` plugin, the common transformations, and OpenVINO GenAI's
> `ContinuousBatchingPipeline`. Date: 2026-06-16.
>
> Tool/agent investigation steps are annotated in _italics_; the prose is the dialogue.

---

## 1. Where do cached activation pools drive GPU/LLM memory?

**User:** i want you to look in gpu and common plugin for areas where cached activation pools would be driving memory consumption in llms

**Assistant:**
_(Read `src/plugins/intel_gpu/src/runtime/memory_pool.cpp`; fanned out three Explore agents across GPU memory pool/allocation, GPU KV-cache/dynamic alloc, and common-plugin LLM transformations.)_

Here's what I found. I focused on the structures that **cache activation buffers and keep them resident**, ranked by how much they actually drive LLM memory growth.

### The core mechanism: `memory_pool` caches activations for the whole network lifetime

`src/plugins/intel_gpu/src/runtime/memory_pool.cpp` is the central activation cache. It keeps three pools — `_non_padded_pool` (multimap by byte size), `_padded_pool` (map by layout), and `_no_reusable_mems`. Activation buffers enter via `get_memory()` (`memory_pool.cpp:291`), called from `primitive_inst::get_memory_from_pool()` (`primitive_inst.cpp:163`) during `realloc_outputs()` and `allocate_internal_buffer()`.

The key memory-driver: **buffers are never returned to the engine during inference.** `release_memory()` (`memory_pool.cpp:48`) only drops the *user reference* from a record — the underlying `memory` object stays in the pool. It's only actually freed at `network::~network()` → `clear_pool_for_network()` (`network.cpp:284`, `memory_pool.cpp:328`). So once a buffer grows to its high-water mark, that footprint is held for the request's lifetime.

For LLMs this matters because of how dynamic shapes are pooled:
- `release_memory()` rewrites dynamic layouts to their **max_shape** before pooling (`memory_pool.cpp:51-54`), so a buffer is sized to the upper-bound sequence length.
- `get_from_non_padded_pool()` reuses a cached buffer only if the request is ≥ `mem_pool_util_threshold` of the cached buffer's size (`memory_pool.cpp:182`). When a request can't reuse, a **new, larger** buffer is allocated and added to the pool — the pool ratchets upward and old entries linger.

### The amplifier: `ShapePredictor` over-allocates ahead of demand

`src/plugins/intel_gpu/src/runtime/shape_predictor.cpp` decides how big the *next* activation buffer should be, and it deliberately allocates beyond the current shape:

- **Iterations mode** (default): after seeing 3 consecutive shapes growing by a constant per-dim delta (KV seq_len += 1/token), it allocates `current + delta * next_iters_preallocation_count` — i.e. **10 tokens ahead** by default (`shape_predictor.cpp:153-158`, `next_iters_preallocation_count = 10`).
- **Percentage mode** (fallback): multiplies the buffer by `buffers_preallocation_ratio`, default **1.1 → 10% overhead** (`shape_predictor.cpp:159-166`).
- Guarded by `can_preallocate()` which blocks once device usage would exceed **90% of global mem** (`shape_predictor.cpp:52-60`).

These preallocated, predicted-size buffers then land in the memory_pool and stay resident per the persistence behavior above.

### KV cache: the dominant resident allocation

KV cache is the largest persistent activation and is **not** in the memory_pool — it lives in per-`SyncInferRequest` `VariableState` objects (`src/plugins/intel_gpu/src/plugin/variable_state.cpp`, `multi_tensor_variable_state.cpp`), reallocated only when too small (`variable_state.cpp:118-126`), and grown in chunks. The OCL KV-cache impl grows the concat/beam-table axis by `get_prealloc_iter_num()` = **128 + (id % 64)** slots at a time (`graph/kv_cache.cpp:145-156`, `impls/ocl/kv_cache.cpp:302`) — chunked growth that staggers reallocation peaks but holds slack capacity. Compressed variants (`VariableStateIndirectKVCacheCompressed`) additionally retain scale/zero-point state tensors.

### Where the "common" plugin/transformations side reduces or shapes this

The transformations mostly *mitigate* activation memory; the relevant ones to know when chasing consumption:
- **SDPA fusion** (`src/common/transformations/.../sdpa_fusion.cpp` + GPU `sdpa_base.cpp`) — keeps the `[batch, heads, seq_len, seq_len]` attention-score matrix in-kernel. If SDPA *doesn't* fuse (pattern mismatch), that buffer materializes — at seq_len=4096 that's ~1-2 GB per layer. This is the single biggest "accidental" activation blow-up to watch for.
- **In-place concat / crop buffer fusing** (`graph_optimizer/prepare_buffer_fusing.cpp:58-283`) — lets `past || new_token` KV concat share buffers instead of materializing a full concatenated tensor. When it can't fuse (`can_share_buffer(false)`), you get a full extra copy.
- **kv_cache_fusion / indirect_kv_cache / kv_cache_compression** (GPU `plugin/transformations/`) — fuse ReadValue→Gather→Concat→Assign into one `KVCache` primitive and avoid scattered/duplicated cache tensors; compression gives 4-8× reduction.
- **oooq_memory_dependencies.cpp** — controls which activations are *allowed* to share pool buffers; conservative dependencies here reduce reuse and inflate the pool.

### The knobs that govern all of this (`internal_properties.hpp`)
- `GPU_ENABLE_MEMORY_POOL` — toggle the activation cache entirely.
- `GPU_DISABLE_MEMORY_REUSE` (`memory_pool.cpp:301`) — forces fresh allocs (debugging footprint).
- `GPU_MEM_POOL_UTIL_THRESHOLD` (default 0.5) — reuse aggressiveness.
- `GPU_BUFFERS_PREALLOCATION_RATIO` (default 1.1) and `GPU_SHAPE_PREDICTOR_SETTINGS` — control the over-allocation.

### Bottom line — the hot spots for LLM activation memory
1. `memory_pool.cpp` — pooled activations sized to **max_shape** and held until network destruction (never shrink). *(biggest structural driver)*
2. `shape_predictor.cpp` — 10-iterations-ahead / +10% preallocation feeding oversized buffers into that pool.
3. KV-cache `VariableState` growth in 128+ slot chunks (`kv_cache.cpp`), separate from the pool.
4. Unfused SDPA materializing `seq_len×seq_len` scores (`sdpa_fusion` + `sdpa_base.cpp`).
5. Failed in-place concat fusing duplicating KV tensors (`prepare_buffer_fusing.cpp`).

---

## 2. Does "buffers preallocation mode" have a property key?

**User:** wait does buffers preallocstioj mode have a property key?

**Assistant:**
No — there's **no property that selects the mode.** The two modes (iterations vs. percentage) aren't user-switchable; `predict_preallocation_shape()` picks between them automatically based on shape history. What's exposed are the *parameters* each mode reads:

| Property key | C++ symbol | Controls |
|---|---|---|
| `GPU_BUFFERS_PREALLOCATION_RATIO` | `ov::intel_gpu::buffers_preallocation_ratio` | The **percentage-mode** ratio (default `1.1f` → +10%). `internal_properties.hpp:134` |
| `GPU_SHAPE_PREDICTOR_SETTINGS` | `ov::intel_gpu::shape_predictor_settings` | The whole `ShapePredictor::Settings` struct — i.e. **iterations-mode** params too. `internal_properties.hpp:179` |

The `Settings` struct (`shape_predictor.hpp:17-25`) carries both groups of knobs. `buffers_preallocation_ratio` is reachable two ways: directly via `GPU_BUFFERS_PREALLOCATION_RATIO`, or as a field inside `GPU_SHAPE_PREDICTOR_SETTINGS`. The iterations-mode fields are *only* reachable through `GPU_SHAPE_PREDICTOR_SETTINGS`.

Mode selection (no key): iterations mode when per-iteration growth `< max_per_iter_size` (16 KB) and per-dim diff `< max_per_dim_diff` (2); otherwise percentage mode. Setting ratio to `1.0f` disables percentage-mode preallocation; to turn off iterations mode you'd zero `next_iters_preallocation_count`.

Caveat: KV-cache paths bypass history-based selection — they pass `custom_next_iters_prealloc_count`/`custom_prealloc_dim` (`shape_predictor.hpp:46-49`) to force a fixed preallocation regardless of these properties.

---

## 3. What values are available for `GPU_SHAPE_PREDICTOR_SETTINGS`?

**User:** what values are available for GPU_SHAPE_PREDICTOR_SETTINGS

**Assistant:**
It isn't an enum — it's a **4-field struct** (`ShapePredictor::Settings`). Default registered at `options.inl:44`:

```cpp
OV_CONFIG_RELEASE_INTERNAL_OPTION(ov::intel_gpu, shape_predictor_settings,
                                  {10, 16 * 1024, 2, 1.1f}, "Preallocation settings")
```

| Field | Type | Default | Meaning / valid values |
|---|---|---|---|
| `next_iters_preallocation_count` | `size_t` | `10` | Iterations to preallocate ahead. `0` disables iterations-mode. |
| `max_per_iter_size` | `size_t` | `16 * 1024` (16 KB) | Max per-iteration buffer growth (bytes) for iterations mode. |
| `max_per_dim_diff` | `size_t` | `2` | Max per-dimension delta still treated as steady growth. |
| `buffers_preallocation_ratio` | `float` | `1.1f` | Percentage-mode multiplier; `> 1.0` to take effect. |

Consumed at `shape_predictor.cpp:70` (count), `:139` (per-dim diff), `:149` (per-iter size).

Two caveats:
1. **Internal, struct-typed RW option** with no `operator>>`/`operator<<` → no string parser, so it **can't be set through a string config** (`benchmark_app -load_config`/JSON). Must be set programmatically via `set_property` / `compile_model(..., ov::intel_gpu::shape_predictor_settings(s))`. Only scalar `GPU_BUFFERS_PREALLOCATION_RATIO` is string-settable.
2. The KV-cache path ignores these and forces its own prealloc via `custom_next_iters_prealloc_count`/`custom_prealloc_dim`.

Meaningful "off" settings: `next_iters_preallocation_count = 0` (kills iterations mode) and `buffers_preallocation_ratio = 1.0f` (kills percentage mode).

---

## 4. These ratios seem destructive — how long have they been in place?

**User:** some of these ratios seem destructive and in practice are expensive. how long have they been in place?

**Assistant:**
_(Clone was shallow — 50 commits. Deepened to ~17k commits back to 2021 and traced introductions via `git log -S`.)_

### Timeline

| Date | PR | What landed |
|---|---|---|
| **2023-07-13** | **#18181** "[GPU] Memory preallocation optimization" | **Origin of all of it.** Introduced `ShapePredictor`, iterations mode (`next_iters_preallocation_count = 10`, `max_per_iter_size = 16*1024`, `max_per_dim_diff = 2`), percentage mode, and the 90%-device-mem `can_preallocate()` cap — all in one commit. |
| 2023-11-13 | #21019 | Moved `ShapePredictor` from per-`cldnn::network` to **per-InferRequest**. |
| 2024-09-02 | #26340 | Guard: skip preallocation once desired size hits max supported buffer size. |
| 2025-02-18 | #28667 "shared plugin config" | Folded the four knobs into the `Settings` struct with the `{10, 16*1024, 2, 1.1f}` default — when `1.1` became the baked-in default. |

The **iterations-mode constants (10 / 16 KB / 2) have been unchanged for ~3 years** (since July 2023).

Nuance on the ratio: in the **original 2023 PR the ratio defaulted to `1.0f`** (class) / `0.0f` (property), and percentage mode only fires when `ratio > 1.0` — so **percentage preallocation was effectively OFF by default** at first. The `1.1f` unconditional +10% only became the resident default in the **Feb 2025 refactor (#28667)** — ~16 months old, not 3 years.

On "destructive/expensive": iterations mode is intentionally cheap (capped at 16 KB/iter → ~160 KB look-ahead per buffer). The genuinely expensive LLM behavior is **max_shape pooling that never shrinks** + **KV-cache chunked growth (128+ slots)** (which *bypasses* these struct settings). So the struct ratios are likely not the culprit in a profile.

---

## 5. Does this apply to the CPU plugin?

**User:** would this apply to cpu plugin as well?

**Assistant:**
_(Confirmed zero references to `ShapePredictor`/`buffers_preallocation`/`next_iters_preallocation` under `src/plugins/intel_cpu/`; launched a CPU-memory Explore agent.)_

Short answer: **mostly no.** The predictive preallocation layer is GPU-only.

**What does NOT carry over:** No `ShapePredictor`, no `GPU_SHAPE_PREDICTOR_SETTINGS`, no `1.1x` ratio, no "10 iterations ahead." CPU computes activation reuse **statically at compile time** via `ov::MemorySolver` (`memory_control.cpp:208-361`), packing non-overlapping intermediates into a shared workspace — no runtime tuning knobs of that kind.

**What DOES parallel — and is more aggressive:**
1. **Grow-only, never-shrink buffers.** `MemoryBlockWithReuse::resize()` (`cpu_memory.cpp:252-270`) only grows (`if (size > m_memUpperBound)`); never shrinks; `free()` only zeroes at teardown.
2. **KV-cache headroom — CPU uses 2×, not 1.1×.** In `scaled_attn.cpp:2651-2682`, on outgrowing capacity it reallocates to **double** the needed size. The 2× doubling is amortized growth (O(log N) reallocs), bounded but not tunable.

| Behavior | GPU | CPU |
|---|---|---|
| Predictive prealloc (iters-ahead / ratio) | Yes — `ShapePredictor`, 10 iters / 1.1× | **None** |
| Activation reuse strategy | Runtime `memory_pool` (cached buffers) | Compile-time `MemorySolver` + edge sharing |
| Grow-only / never shrink until teardown | Yes (`memory_pool`) | Yes (`MemoryBlockWithReuse`) |
| KV-cache headroom on growth | ~1.1× (or 128+ slot chunks) | **2× doubling** (`scaled_attn.cpp`) |
| User-tunable memory knobs | Several (`GPU_*` props) | Essentially none for this |

So the *specific ratios/properties* don't apply to CPU, but "buffers grow and never give memory back, and KV cache over-allocates" is present on CPU too — and the KV multiplier is larger.

---

## 6. Speculation: why do these settings exist if they hurt Pareto optimality?

**User:** can you speculate on the reasons why these settings exist if they damage pareto optimality for transformers specifically? such massive sequence length growth means on vram constrained devices you pay architecturally correct kv cache cost for a given model but have excessive memory pressure from the allocated cache memory, which seems like a massive oversight for usability

**Assistant:**
The "oversight" framing is half right — the more precise read is that these are **artifacts of optimizing a different objective on different hardware assumptions, which the LLM-on-constrained-VRAM regime then inverted.**

**What was being optimized:** PR #18181 (July 2023) predates LLM-on-edge as the primary GPU workload. The mechanism kills **allocation-induced pipeline stalls in dynamic-shape inference** — a synchronous USM allocation mid-inference is a latency cliff; growing one element/iteration means paying it every token. The predictor converts O(N) reallocations into O(1) amortized — pure latency/throughput optimization. The 16 KB/iter ceiling reveals the mental model: **small, incrementally-growing dynamic dims** (detection, variable batch, short NLP), where 160 KB of slack is rounding error.

**Why transformers break it:**
1. The growing dimension is enormous, not incremental (scales by `batch × heads × head_dim × dtype` per token).
2. You already know the exact upper bound (`max_seq_len` is declared) — speculation is strictly worse than allocating the known max once.
3. The memory/latency tradeoff flips sign on constrained VRAM — the design assumes spare memory to spend avoiding stalls; constrained devices have the opposite binding constraint. Same code, opposite Pareto frontier, with no runtime signal telling the allocator which regime it's in.

**Steelman against pure neglect:** there are guards (90% device-mem cap, max-buffer-size guard #26340); 2× doubling is textbook amortization; escape hatches exist (`next_iters=0`, `ratio=1.0`, `GPU_DISABLE_MEMORY_REUSE`).

**The genuine oversight** is lack of regime-awareness and discoverability: no "I know my max, just allocate it" path; device-relative not model-relative guards; the knobs are internal/undiscoverable (`GPU_SHAPE_PREDICTOR_SETTINGS` has no string parser); never-shrink ages badly under concurrent serving.

Synthesis: reasonable 2023 defaults for an abundant-memory, latency-sensitive, non-LLM workload, never re-tuned when LLMs-on-constrained-VRAM became dominant. The architecture isn't wrong — it's *unconditional*.

---

## 7. Proposed solution

**User:** which seems like a naive debugging arg. hmm. what is your proposed solution to this issue?

**Assistant:**
"Just set the ratio to 0 / disable reuse" is a debugging probe, not a solution. A real fix must be **automatic, regime-correct, and non-fatal.**

1. **Clamp speculation to the known upper bound (highest leverage, lowest risk).** In `predict_preallocation_shape()` and the KV bypass, never return a predicted shape larger than the static upper bound — if `layout.get_partial_shape().get_max_shape()` is finite, `min(predicted, static_max)`. You can never hold more than the architecturally-required cost.
2. **Bound slack in bytes, not slots or device-%.** Replace `128 + id%64` slot growth and the 90%-VRAM guard with `max(absolute_byte_cap, per_token_bytes × small_constant)`, and cap total speculative slack to a fraction of the model's own KV budget.
3. **OOM-triggered pool reclaim (non-fatal).** Before `alloc_memory()` fails, trim `_non_padded_pool`/`_padded_pool` of user-less records and retry once — convert hard OOM into graceful degradation.
4. **One high-level dial, not four internal structs.** Tie to a coarse `MEMORY` vs `LATENCY` posture on the existing `ov::hint` surface; string-settable and discoverable.

**Recommendation:** ship #1 (clamp) + #3 (reclaim) — removes the architectural-correctness violation and makes residual risk non-fatal, minimal surface area, no new public API. Tractable because the predictor is a single chokepoint and the pool a single owner; you're adding a `min()` and a reclaim-retry.

---

## 8. Does GenAI's ContinuousBatchingPipeline mitigate this?

**User:** clone in openvino genai and see if its ContinuousBatchingPipeline cpp impl implements any mitigations for this issue

**Assistant:**
_(Cloned `openvinotoolkit/openvino.genai`; analyzed the CB cache/block managers via an Explore agent.)_

**Verdict: Yes — and it solves it at exactly the layer I argued the plugin should.** CB doesn't rely on the plugin's grow-only KV behavior; it replaces it with a paged, budget-bounded cache (essentially proposals #1 and #2 one layer up).

1. **SDPA → PagedAttention rewrite removes the plugin's dynamic KV growth.** `prepare_model_for_paged_attention()` applies `SDPAToPagedAttention` (`pipeline_impl.cpp:67-86`); attention then reads/writes a **fixed-shape** cache tensor (`kv_cache_manager.hpp:159-262`). `ShapePredictor` / doubling / chunked growth never engage on KV.
2. **Pool pre-sized to a real memory budget, not device-%.** `get_available_*_memory()` probes the device (`pipeline_impl.cpp:28-59,181-187`); `CacheOrchestrator::normalize_block_counts()` (`cache_orchestrator.hpp:54-88, 754-809`) converts `cache_size`(GB)/`num_kv_blocks` into a validated count that fits.
3. **Fixed block pool + reuse** (`block_manager.hpp:199-264`), ref-counted, LRU prefix reuse; over-budget sequences are preempted/swapped, not grown.
4. **Optional eviction** (`use_cache_eviction` + `CacheEvictionConfig`, `cache_eviction.hpp:110-226`) runs `_maybe_evict_cache_blocks()` each step — SnapKV/KVCrush, `max_cache_size`.

**Important caveat:** This applies **only to the CB path.** The default stateful `LLMPipeline` does **not** apply `SDPAToPagedAttention` and **still hits** the plugin's grow-only/predictive KV behavior. Also CB's dynamic mode (`num_kv_blocks==0 && cache_size==0`) can grow, but block-granular and scheduler-controlled.

---

## 9. So the mitigations are only on the CB path?

**User:** So the activation pooling issues and max allocation growth issue are patched only in CB path meaning its graph rewrites are doing a ton of work... likely to maintain broad support in the main plugin... fuck that's awful

**Assistant:**
Mostly right, with one correction each way.

**Right:** the good behavior is gated behind opting into a different pipeline; `SDPAToPagedAttention` is a real model rewrite, not a flag; the default `LLMPipeline` inherits none of it.

**Overstated:** CB doesn't "patch the activation pooling issue" — it sidesteps the *KV-cache* growth specifically. The plugin's *general* activation pool (`memory_pool` + `ShapePredictor` on non-KV dynamic intermediates) **still runs under CB.** CB just bounds the damage via `max_num_batched_tokens` capping the working set. Two separate mitigations (paged KV + bounded batch), not a blanket fix.

**Pushback on "awful":** there's a defensible layering reason. The plugin is a *generic dynamic-shape tensor engine* — no concept of "KV cache" or "sequence axis." The knowledge to bound KV correctly lives at the GenAI layer. Pushing the mitigation up is arguably correct separation of concerns.

**The real indictment that survives:** the plugin doesn't need transformer semantics to do better — two of the proposed fixes need zero model knowledge: (1) respect the static upper bound it already has in the IR; (2) OOM-triggered reclaim. The layering is right; the lower layer is just lazier than it needs to be, forcing the upper layer's rewrite to carry more weight. Leverage point: a small, semantics-free plugin change (clamp-to-bound + reclaim) would make the *default* path tolerable on constrained VRAM without adopting CB.

---

## 10. CB adopted, still over the KV requirement — where's the memory?

**User:** i have done significant work to adopt CB and the memory constraints are still not meeting kv cache requirements of the architecture so the memory must be coming from somewhere. frankly I'm so tired of working with openvino.

**Assistant:**
_(Invoked the `ov-debug` skill; traced the exact env-var naming: env = `OV_` + property key, via `plugin_config.cpp:194-205` and `m_allowed_env_prefix = "OV_"`.)_

On the CB path, memory exceeding the architectural KV requirement is almost never the KV cache leaking — it's other allocations crowding the budget.

**Where the excess hides (priority order):**
1. **CB greedily sizes the block pool to *available* memory.** If `cache_size`/`num_kv_blocks` are 0, `normalize_block_counts()` fills free VRAM by design. **Pin `SchedulerConfig::cache_size` (GB).**
2. **The non-KV activation pool — still grow-only, scaled by `max_num_batched_tokens`.** The batched forward pass materializes QKV-proj + MLP intermediates for up to `max_num_batched_tokens` tokens × layers; the pool latches that high-water mark. **Lower `max_num_batched_tokens`.**
3. **KV precision / block rounding** (fp16 cache, partial-block rounding).
4. **Fixed overhead** — weights, compile scratch, oneDNN primitives, PagedAttention scratch.

**Instrumentation (debug-caps build, `ENABLE_DEBUG_CAPS=ON`):**
- GPU: `OV_GPU_DUMP_MEMORY_POOL=1` (or `=2`) + `OV_GPU_DUMP_MEMORY_POOL_PATH=./mem_dump/` → CSV with `non_padded_pool` / `padded_pool` / `no_reusable` totals and host-vs-device split (`internal_properties.hpp:156-157`, `options.inl:84-85`).
- CPU: `OV_CPU_MEMORY_STATISTICS_PATH=mem.csv`.

_(Asked clarifying questions — device, OOM vs higher-RSS, whether cache_size is pinned — which the user opted to skip.)_

---

## 11. Burnout / OpenArc

**User:** honestly man I barely have motivation to work on this. i run the OpenArc project, and openvino ia just so hard to work with.

**Assistant:**
Nothing here is a skill issue — it's the framework. The memory mitigations exist but are gated behind CB, the good knobs are internal/string-unsettable, the debug flags need a special build, and the layer with the information to fix it throws it away. You did the right thing (adopted CB) and it still doesn't fit, for reasons buried three layers down. That's a real ergonomics tax.

Offer: **I do the grunt work, you point.** I can (a) write a standalone repro harness that loads your model through CB, dumps the GPU memory pool, and prints a "KV vs activation pool vs weights vs scratch" breakdown; or (b) take your numbers (model dims, context length, `max_num_batched_tokens`, total VRAM, OOM point) and name the bucket + the one-line config change. Either way, five minutes of your time. No pressure tonight.

---

## 12. "You didn't even look up my project"

**User:** you didn't even look up my project

**Assistant:**
_(Web-searched and fetched the OpenArc README.)_

Fair callout. **OpenArc** (SearchSavior/OpenArc, ~458 stars, v2.0.4): fully-async FastAPI inference server over Optimum-Intel + OpenVINO, serving LLMs/VLMs/Whisper/Kokoro+Qwen3 TTS/ASR/embeddings/rerankers across Intel CPU/GPU/NPU through OpenAI-compatible endpoints, with multi-GPU pipeline parallelism, CPU offload, speculative decoding, per-request metrics.

Key implication: **OpenArc is built on Optimum-Intel**, so the *default* LLM path is `OVModelForCausalLM` — stateful, native SDPA, the unmitigated grow-only/`ShapePredictor`/never-shrink path. Migrating a model to GenAI CB is a large lift (swaps the whole attention/cache architecture) while still maintaining the other model classes. That's the "significant work to adopt CB" — and it explains the two-memory-models burden.

Sources: <https://github.com/SearchSavior/OpenArc>, <https://github.com/openvinotoolkit/openvino.genai>

---

## 13. OpenArc uses all three backends

**User:** openarc uses both ovgenai and optimumintel and openvino directly.

**Assistant:**
That points at a specific, likely-real failure mode. Resident memory is the **sum of three backends with three non-cooperating memory models**:
- **ovgenai (CB)** → paged, bounded KV pool — *but sized greedily to available VRAM at load time.*
- **optimum-intel** → stateful, grow-only activation pool + ShapePredictor, never shrinks.
- **openvino direct** (Whisper/embeddings/rerankers) → compiled models + their own per-model pools.

The bet: **CB's cache sizing is a point-in-time snapshot.** `get_available_gpu_memory()` measures free VRAM when the CB pipeline initializes and fills it with KV blocks. In an async multi-engine server, whatever loads *after* CB allocates into memory CB already claimed (or vice versa) — **two backends double-counting the same free VRAM**, with nothing arbitrating. That looks like "KV exceeds the architecture" but is cross-backend over-subscription.

Checkable in *OpenArc's* code: engine load order, whether CB `cache_size` is pinned vs derived, and whether there's any cross-engine VRAM reservation.

---

## 14–15. Property summary (sectioned)

**User:** summarize the property args we discovered in this session inside a codeblock so i can copy. give example of their expected shape — _then:_ break that into a few codeblocks by section

**Assistant:**

### GPU plugin: activation pool + dynamic-shape preallocation
```text
GPU_BUFFERS_PREALLOCATION_RATIO   # float, default 1.1  (>1.0 enables % growth; 1.0 = off)
  C++:    ov::intel_gpu::buffers_preallocation_ratio
  Python: ov_config = {"GPU_BUFFERS_PREALLOCATION_RATIO": "1.0"}
  Env:    OV_GPU_BUFFERS_PREALLOCATION_RATIO=1.0       # string-settable

GPU_SHAPE_PREDICTOR_SETTINGS      # struct, default {10, 16384, 2, 1.1}
  # fields (positional): {next_iters_preallocation_count,
  #                       max_per_iter_size_bytes,
  #                       max_per_dim_diff,
  #                       buffers_preallocation_ratio}
  C++:    ov::intel_gpu::shape_predictor_settings({0, 16*1024, 2, 1.0f})
  Python/Env: NOT string-settable (no parser) — programmatic C++ only
  # to fully disable predictive prealloc: next_iters=0 AND ratio=1.0

GPU_MEM_POOL_UTIL_THRESHOLD       # float 0.0-1.0, default 0.5  (pool reuse aggressiveness)
  C++:    ov::intel_gpu::mem_pool_util_threshold
  Python: ov_config = {"GPU_MEM_POOL_UTIL_THRESHOLD": "0.5"}

GPU_ENABLE_MEMORY_POOL            # bool, default true   (RELEASE_INTERNAL)
  C++:    ov::intel_gpu::enable_memory_pool
  Python: ov_config = {"GPU_ENABLE_MEMORY_POOL": "NO"}

GPU_DISABLE_MEMORY_REUSE          # bool, default false  (DEBUG build only)
  C++:    ov::intel_gpu::disable_memory_reuse
  Env:    OV_GPU_DISABLE_MEMORY_REUSE=1                 # needs ENABLE_DEBUG_CAPS=ON
```

### GPU plugin: memory-pool diagnostics (DEBUG build, `ENABLE_DEBUG_CAPS=ON`)
```text
GPU_DUMP_MEMORY_POOL              # size_t, default 0  (0=off, 1=summary, 2=per-entry)
  Env:    OV_GPU_DUMP_MEMORY_POOL=1
GPU_DUMP_MEMORY_POOL_PATH         # string dir, default ""  (writes CSV breakdown)
  Env:    OV_GPU_DUMP_MEMORY_POOL_PATH=./mem_dump/
  # CSV columns: pool_type,layout,mem_ptr,mem_type,mem_pool_size,prim_id,unique_id,mem_size
  # pool_type in {non_padded_pool, padded_pool, no_reusable_pool}; mem_type shows usm_host vs device
```

### CPU plugin: memory diagnostics
```text
OV_CPU_MEMORY_STATISTICS_PATH     # env var, CSV path — per-compiled-model memory stats
  Env:    OV_CPU_MEMORY_STATISTICS_PATH=./cpu_mem.csv
  # CPU has NO ShapePredictor / prealloc-ratio knobs — reuse is solved at compile time;
  # KV cache grows via 2x doubling in scaled_attn.cpp, not user-tunable
```

### OpenVINO GenAI: ContinuousBatchingPipeline KV-cache budget (`SchedulerConfig` struct fields — set in code, not `ov_config`)
```text
SchedulerConfig.cache_size               # size_t GB — pin KV pool budget (stops greedy VRAM fill)
SchedulerConfig.num_kv_blocks            # size_t   — explicit block count (alt to cache_size)
                                         #   both 0  => dynamic cache allocation (grows on demand)
SchedulerConfig.max_num_batched_tokens   # size_t   — caps batched working-set (activation footprint)
SchedulerConfig.dynamic_split_fuse       # bool
SchedulerConfig.use_cache_eviction       # bool, default false — enable token eviction
SchedulerConfig.cache_eviction_config    # CacheEvictionConfig{start, recent, max_cache_size, ...}
```

### Gotchas (apply across the blocks above)
```text
- Python ov_config values are STRINGS: "1.0", "YES"/"NO" — even for floats/bools.
    Optimum-Intel:  OVModelForCausalLM.from_pretrained(..., ov_config=ov_config)
    Raw OpenVINO:   core.compile_model(model, "GPU", config=ov_config)
- GPU_SHAPE_PREDICTOR_SETTINGS: not reachable from Python/env (struct, no parser).
    Reachable % lever = GPU_BUFFERS_PREALLOCATION_RATIO; iterations-mode fields are C++-only.
- DEBUG-gated (GPU_DUMP_*, GPU_DISABLE_MEMORY_REUSE) need ENABLE_DEBUG_CAPS=ON;
    silently ignored on stock release wheels.
- SchedulerConfig.* are GenAI struct fields set on the pipeline, NOT ov_config properties.
```

---

## Key file references

| Area | File |
|---|---|
| GPU activation pool | `src/plugins/intel_gpu/src/runtime/memory_pool.cpp` |
| GPU shape predictor | `src/plugins/intel_gpu/src/runtime/shape_predictor.cpp` / `.hpp` |
| GPU KV cache | `src/plugins/intel_gpu/src/graph/kv_cache.cpp`, `impls/ocl/kv_cache.cpp` |
| GPU variable state | `src/plugins/intel_gpu/src/plugin/variable_state.cpp`, `multi_tensor_variable_state.cpp` |
| GPU properties | `src/plugins/intel_gpu/include/intel_gpu/runtime/internal_properties.hpp`, `options.inl` |
| CPU memory control | `src/plugins/intel_cpu/src/memory_control.cpp`, `cpu_memory.cpp` |
| CPU KV cache growth | `src/plugins/intel_cpu/src/nodes/scaled_attn.cpp` |
| Env-var prefix logic | `src/inference/src/dev/plugin_config.cpp`, `plugin_config.hpp` |
| GenAI CB pipeline | `openvino.genai/src/cpp/src/continuous_batching/` (pipeline_impl, cache/*) |
