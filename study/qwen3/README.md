# Qwen3.5-VL custom-kernel study

A study branch investigating whether fused custom kernels can be added to an
OpenVINO model **through the public extension API** — no fork, no upstream PR —
for the Qwen3.5-VL 0.8B language model (the Qwen3-Next architecture: hybrid
linear-attention + full-attention layers, gated delta rule recurrence, causal
conv1d with state).

This file summarizes **what was done and what we learned**, so the artifacts
are easy to navigate. It is deliberately light on performance numbers — the
point is the process and the conclusions, not the benchmark tables (those live
in the individual commit messages).

---

## Layout

```
kernels/     the two fused kernels, in C and in Python, + their unit test
cpp_ext/     the C++ OpenVINO extension library (wraps kernels/kernels.c)
openvino/    tests / runners that use the raw ov.Core API
genai/       tests / runners that use the openvino_genai pipelines
```

### `kernels/` — the kernels (C + Python)
- `kernels.c` / `kernels.h` — the canonical C implementations of the gated
  delta rule recurrence (`gdr_kernel`) and the causal conv1d-with-state
  (`conv1d_kernel`). Built two ways: as a standalone `.so` for the Python
  ctypes path (here) and compiled directly into the C++ extension (`cpp_ext/`).
- `build_kernels.sh` — builds `libqwen3_kernels.so` for the ctypes path.
- `kernels.py` — ctypes bindings + a `QWEN3_USE_C=1` toggle.
- `fused_linear_attn.py` — the `GatedDeltaRule` OpenVINO custom op. Its
  `evaluate()` has **both** a NumPy reference and a ctypes-to-C path; this is
  the "Python kernel" next to the C one. Also contains the graph rewrite that
  finds the unfused `Loop` and swaps the op in.
- `fused_conv1d.py` — the `FusedCausalConv1d` custom op (NumPy + C) and its
  rewrite (matches `Concat → GroupConvolution → Slice`).
- `lm_head_slice.py` — a pure graph rewrite (no custom op): slice the final
  hidden state to the last token before `lm_head`.
- `export_fused.py` — serialize a fused IR with any subset of the three
  rewrites applied (`gdr`, `conv1d`, `lm_head_slice`).
- `test_kernels.py` — numerical unit test of the C kernels vs the NumPy
  reference. No OpenVINO needed.

### `cpp_ext/` — the C++ extension library
The two kernels wrapped as a proper OpenVINO extension (`libqwen3_ov_ext.so`),
loadable via `core.add_extension("…​.so")` or the `openvino_genai`
`extensions=` property. It compiles `../kernels/kernels.c` directly, so there
is a single canonical copy of the C source.

```
cd cpp_ext && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
```

### `openvino/` — raw `ov.Core` entry points
- `profile_prefill.py` / `profile_decode.py` — generic, model-agnostic OV
  profiling (execution graph + per-node timing). Reusable beyond this study.
- `generate.py` — end-to-end greedy generation through the fully-fused model.
- `validate_fusion.py` — confirms the rewritten model produces the same logits
  as the stock model.
- `compare_kv.py` — confirms the rewrites don't change the persistent KV state.
- `benchmark_ops.py` — head-to-head per-op timing: our op vs the CPU plugin's
  own primitives.
- `memory_analysis.py` — the consolidated memory attribution: `get_runtime_model()`,
  `get_profiling_info()`, glibc `mallinfo2`, and an `OMP_NUM_THREADS` toggle,
  with raw and `vlm-*` comparison modes. Best single entry point.

### `genai/` — `openvino_genai` entry points
- `genai_vlm_pipeline.py` — drives the fused model through `VLMPipeline` using
  the C++ extension.
- `cb_chunking_test.py` — uses `ContinuousBatchingPipeline` +
  `SchedulerConfig.dynamic_split_fuse` to toggle chunked prefill explicitly.

---

## The question we started with

Can we attach hand-written fused kernels to an exported OV model via
`core.add_extension(...)` and have them participate in real inference — and
does that buy us anything over the stock pipeline?

The investigation walked through four phases:

1. **Profiling & baseline** (`openvino/profile_*.py`) — make the execution
   graph and per-node timing visible.
2. **Building the custom ops** (`kernels/`) — the Python ops + rewrites + C
   kernels + correctness checks.
3. **Productionizing** (`cpp_ext/`, `genai/genai_vlm_pipeline.py`) — the same
   kernels as a real C++ extension, loadable through `genai`.
4. **Measurement** (`openvino/memory_analysis.py`, `openvino/benchmark_ops.py`,
   `genai/cb_chunking_test.py`) — attribute memory and latency phase by phase.

### What we learned (qualitatively)

1. **The extension API works.** Custom ops — Python or compiled C++ — load,
   run, and produce correct output end to end, including through
   `genai.VLMPipeline`. No fork, no upstream PR.

2. **The CPU plugin already fuses these patterns itself.** The stock `Loop`
   gets rewritten by the plugin into its own `GatedDeltaNet` primitive at
   compile time. Our custom op competes against an existing native fusion, not
   an unoptimized fallback.

3. **`genai` does much more than we do.** `SDPAToPagedAttention` (run at
   pipeline construction, not baked into the IR on disk) applies a stack of
   passes — paged-attention conversion, `PagedGatedDeltaNetFusion`,
   `PagedCausalConv1DFusion`, state-table conversion — and enables chunked
   prefill at the scheduler level. These are the real source of `genai`'s
   memory efficiency.

4. **An opaque custom op (a generic `Reference` node) opts out of all of
   that.** It is invisible to the matchers, so it (a) blocks the paged-attention
   conversion, (b) loses the chunked-prefill benefit, and (c) breaks
   `ContinuousBatchingPipeline` outright (a `beam_idx` cleanup assertion fires
   because the matcher can't see through our op).

**Headline:** the extension API absolutely works for adding a custom kernel,
but on CPU through `genai` the stock pipeline's compile-time transformations
already capture the wins we were chasing, and an opaque custom op forfeits them.

---

## Tried and reverted: subclassing the internal op

A follow-up spike made `GatedDeltaRule` a **subclass of
`ov::op::internal::GatedDeltaNet`** so the `is_castable`-based matchers would
see through it. It got past the `beam_idx` assertion that breaks the plain
custom op in `ContinuousBatchingPipeline`, but then hit a layout mismatch:
`PagedGatedDeltaNetFusion` fires but OV's paged form expects `[B, T, H, D]`
while our IR is `[B, H, T, D]`. The spike was **reverted** (commit history
preserves it); finishing it would mean reconciling that layout convention.

---

## Status of the artifacts

- **Reusable tooling:** `openvino/profile_*.py`, `openvino/memory_analysis.py`
  are model-agnostic and useful beyond this study.
- **The capability proof:** `cpp_ext/` + `kernels/` + `openvino/generate.py` +
  `genai/genai_vlm_pipeline.py` demonstrate the extension path end to end.
- **Consolidated:** several overlapping early memory/latency probes
  (`probe_*.py`, `measure_fusion.py`, `sweep_kernels.py`, `compare_decode_mem.py`,
  `genai_mem_compare.py`) were removed; `memory_analysis.py`, `benchmark_ops.py`,
  and `cb_chunking_test.py` cover the same ground.
