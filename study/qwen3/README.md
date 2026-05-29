# Qwen3.5-VL custom-kernel study

A study branch investigating whether fused custom kernels can be added to an
OpenVINO model **through the public extension API** — no fork, no upstream PR —
for the Qwen3.5-VL 0.8B language model (which is the Qwen3-Next architecture:
hybrid linear-attention + full-attention layers, gated delta rule recurrence,
causal conv1d with state).

This file summarizes **what was done and what we learned**, so we can decide
what is worth keeping. It is deliberately light on performance numbers — the
point here is the process and the conclusions, not the benchmark tables (those
live in the individual commit messages if needed).

---

## The question we started with

Can we attach hand-written fused kernels to an exported OV model via
`core.add_extension(...)` and have them participate in real inference — and
does that buy us anything over the stock pipeline?

The investigation walked through four phases, each of which left behind a
script. Reading them in this order tells the story:

### Phase 1 — Profiling & baseline (can we even see what's happening?)
- `profile_prefill.py` / `profile_decode.py` — dump the post-fusion execution
  graph + per-node timing for any optimum-intel-exported causal LM. Generic,
  reusable, model-agnostic.
- `probe_memory.py` — measures where the CPU plugin's resident memory actually
  lives (heap vs mmap, scaling with sequence length). Built to *disprove* an
  early wrong guess that the plugin held a fixed memory pool.

### Phase 2 — Building the custom ops (does the extension API work at all?)
- `fused_linear_attn.py` — Python `Op` subclass for the gated delta rule, plus
  the graph rewrite that finds the unfused `Loop` and swaps it in.
- `fused_conv1d.py` — Python `Op` for the causal conv1d-with-state, plus its
  rewrite (matches the `Concat → GroupConvolution → Slice` chain).
- `lm_head_slice.py` — pure graph rewrite (no custom op): slice the final
  hidden state to the last token before `lm_head`.
- `kernels.c` / `kernels.py` / `build_kernels.sh` — the same two kernels in C,
  callable from the Python ops via ctypes (toggle with `QWEN3_USE_C=1`). This
  let us compare a numpy reference vs compiled C without touching the op.
- `test_kernels.py` — numerical check of the C kernels vs the numpy reference.
- `validate_fusion.py` — numerical check that the rewritten model still
  produces the same logits as the stock model.
- `generate.py` — end-to-end greedy generation through the fully-fused model.

**Result:** yes, the extension API works. Custom ops load, run, and produce
correct output end to end.

### Phase 3 — Productionizing into a real extension library
- `cpp_ext/` — the two kernels wrapped as a proper C++ OpenVINO extension
  (`libqwen3_ov_ext.so`), loadable via `core.add_extension("…​.so")` or the
  `openvino_genai` `extensions=` property. `kernels.c` is shared with the
  ctypes path.
- `export_fused.py` — serialize a fused IR variant with any subset of the three
  rewrites applied (`gdr`, `conv1d`, `lm_head_slice`), so each can be tested in
  isolation.
- `genai_vlm_pipeline.py` — drive the fused model through `genai.VLMPipeline`
  using the C++ extension.

**Result:** the extension library loads through `genai` and generates correctly.

### Phase 4 — Measurement & the key finding
A family of harnesses, each isolating one variable in its own subprocess:
- `measure_fusion.py` — first cut: fused vs unfused prefill memory/latency.
- `compare_kv.py` — confirms the rewrites don't change the persistent KV state.
- `compare_decode_mem.py` — decode-time memory, unfused vs fused.
- `sweep_kernels.py` — numpy-eval vs C-eval vs baseline across sequence lengths.
- `benchmark_ops.py` — head-to-head: our op vs the CPU plugin's own primitives.
- `genai_mem_compare.py` — fused vs unfused **through `genai.VLMPipeline`**.
- `probe_compile_config.py` — sweeps compile-time properties to rule them out.
- `probe_structural.py` — chunked vs single-shot prefill.
- `cb_chunking_test.py` — toggles chunked prefill explicitly via
  `ContinuousBatchingPipeline` + `SchedulerConfig.dynamic_split_fuse`.
- `memory_analysis.py` — the consolidated analysis: uses `get_runtime_model()`,
  `get_profiling_info()`, glibc `mallinfo2`, and an `OMP_NUM_THREADS` toggle to
  attribute memory phase by phase.

**What we learned (qualitatively):**

1. **The CPU plugin already fuses these patterns itself.** The stock `Loop`
   gets rewritten by the plugin into its own `GatedDeltaNet` primitive at
   compile time. Our custom op is competing against an existing native fusion,
   not against an unoptimized fallback.

2. **`genai` does much more than we do.** `SDPAToPagedAttention` (run at
   pipeline construction, not baked into the IR on disk) applies a stack of
   passes — paged-attention conversion, `PagedGatedDeltaNetFusion`,
   `PagedCausalConv1DFusion`, state-table conversion — and enables chunked
   prefill at the scheduler level. These are the real source of `genai`'s
   memory efficiency.

3. **A custom op registered as a generic `Reference` node opts out of all of
   that.** It is opaque to the matchers, so it (a) blocks the paged-attention
   conversion, (b) loses the chunked-prefill benefit, and (c) breaks
   `ContinuousBatchingPipeline` outright (a `beam_idx` cleanup assertion fires
   because the matcher can't see through our op).

So the headline conclusion: **the extension API absolutely works for adding a
custom kernel, but on CPU through `genai` the stock pipeline's compile-time
transformations already capture the wins we were chasing, and an opaque custom
op forfeits them.**

---

## Approach A (in progress, uncommitted): subclass the internal op

The follow-up experiment: instead of a standalone op classified as `Reference`,
make `GatedDeltaRule` a **subclass of `ov::op::internal::GatedDeltaNet`** so the
OV/`genai` pattern matchers (which use `is_castable`, i.e. subclass-aware) can
see through it.

Current state of this work (committed as an explicit WIP in
`cpp_ext/gated_delta_rule.{hpp,cpp}`, `cpp_ext/CMakeLists.txt`,
`fused_linear_attn.py`):
- The subclass compiles and runs correctly end to end via raw `ov.Core`.
- It survives `compile_model` (the L2-norm fusion happens not to match this
  model's graph, so we aren't replaced).
- It gets **past** the `beam_idx` assertion that killed the plain custom op in
  `ContinuousBatchingPipeline` — progress.
- It now hits a **layout mismatch**: `PagedGatedDeltaNetFusion` fires on our op
  but OV's paged form expects `[B, T, H, D]` while our IR is `[B, H, T, D]`.
  Not yet resolved.

This branch of work is unfinished and should be treated as a spike, not a
result.

---

## File inventory

| File | Role | Notes |
|---|---|---|
| `profile_prefill.py`, `profile_decode.py` | Generic OV profiling | Model-agnostic, reusable |
| `probe_memory.py` | Where RSS lives | Diagnostic |
| `fused_linear_attn.py` | GatedDeltaRule op + rewrite | Core artifact |
| `fused_conv1d.py` | Causal conv1d op + rewrite | Core artifact |
| `lm_head_slice.py` | Last-token slice rewrite | Pure rewrite, no custom op |
| `kernels.c` / `kernels.py` / `build_kernels.sh` | C kernels + ctypes | Shared with `cpp_ext` |
| `cpp_ext/` | C++ extension library | The "real" deliverable |
| `export_fused.py` | Serialize fused IR variants | Tooling |
| `generate.py`, `genai_vlm_pipeline.py` | End-to-end runners | Demos |
| `test_kernels.py`, `validate_fusion.py` | Correctness checks | Keep |
| `measure_fusion.py`, `compare_kv.py`, `compare_decode_mem.py` | Early memory harnesses | Superseded by `memory_analysis.py` |
| `sweep_kernels.py`, `benchmark_ops.py` | Latency benchmarks | |
| `genai_mem_compare.py`, `probe_compile_config.py`, `probe_structural.py`, `cb_chunking_test.py` | genai/chunking investigation | |
| `memory_analysis.py` | Consolidated analysis | Best single entry point |

---

## Suggested keep / drop (for discussion)

- **Keep — the reusable tooling:** `profile_prefill.py`, `profile_decode.py`,
  `memory_analysis.py`. These are model-agnostic and useful beyond this study.
- **Keep — the demonstrated capability:** `cpp_ext/` + `fused_linear_attn.py` +
  `fused_conv1d.py` + `lm_head_slice.py` + `generate.py`. This is the proof that
  the extension path works.
- **Consider folding together — the measurement harnesses:** there are ~8
  overlapping memory/latency scripts. `memory_analysis.py` and `benchmark_ops.py`
  largely cover what the earlier ones (`measure_fusion.py`, `compare_*.py`,
  `sweep_kernels.py`) do. Candidates to drop or merge.
- **Decide — Approach A:** finish the subclass spike (resolve the layout
  mismatch) or drop it. As-is it is half-done and shouldn't ship in that state.
