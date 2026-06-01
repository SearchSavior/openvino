# Performance discussion — Qwen3.5-VL custom GatedDeltaRule ops

This file is the **only** place perf numbers and interpretation live for
the v1/v2/v3 work. Scripts, headers, kernels, and rewrites stay number-
free; their comments only describe what the code does. Re-run the
underlying scripts to refresh the tables here.

---

## 2026-06-01: corrected user-facing comparison (genai VLMPipeline, one-shot prefill)

Source: `scripts/working/run_probe_pa_path.sh` (drives
`probe_pa_path.py --version X --backend Y` once per cell, fresh process,
wiped genai compile cache).

Setup: Qwen3.5-VL-0.8B INT8, `INFERENCE_NUM_THREADS=4`, image
`/tmp/llama.cpp/media/llama1-logo.png` (770 input tokens after vision
encoder), greedy 32-token generation.

| version  | backend                          | TTFT (s) | prefill (tok/s) | decode (tok/s) |
|----------|----------------------------------|---------:|----------------:|---------------:|
| baseline | PA  (genai default)              |  3.09    |       **249.1** |          17.80 |
| baseline | SDPA                             |  9.77    |          78.8   |          15.72 |
| v1       | PA  (silently falls back to SDPA)|  6.74    |         114.3   |          12.18 |
| v1       | SDPA                             |  6.60    |         116.7   |          11.76 |
| v2       | PA  (silently falls back to SDPA)|  6.88    |         112.0   |          11.35 |
| v2       | SDPA                             |  6.46    |         119.1   |          12.09 |
| v3       | PA  (silently falls back to SDPA)|  9.45    |          81.5   |          11.96 |
| v3       | SDPA                             |  9.53    |          80.8   |          11.52 |

### Reading the table

1. **baseline PA vs baseline SDPA: ~3× prefill** (249 vs 79 tok/s). The default
   `VLMPipeline` path is `PA_BACKEND` → `VLMContinuousBatchingAdapter`
   (paged attention + continuous batching). Forcing `ATTENTION_BACKEND=SDPA`
   downgrades to `VLMPipelineImpl` (stateful KV cache) and removes that
   speedup.

2. **All custom-op configs are on the SDPA path.** For v1/v2/v3 the PA and
   SDPA rows match within noise. The paged-attention transformation pass
   cannot rewrite our `GatedDeltaRule` op, the call throws, genai catches
   it in `log_paged_attention_fallback` (`pipeline.cpp:875`), and the
   pipeline re-loads through `VLMPipelineImpl`. So a user requesting
   default `VLMPipeline` on the fused model never gets the PA fast path,
   regardless of `ATTENTION_BACKEND`.

3. **Under same backend (SDPA on both sides):**
   - Prefill: v1 / v2 *exceed* baseline by ~48 % / 51 %. v3 is ~+3 %
     vs baseline_sdpa.
   - Decode: v1 / v2 / v3 are ~25–30 % slower than baseline_sdpa
     (11.4–12.2 vs 15.7 tok/s).
   - The v3 prefill flatness vs v1/v2 is the absorbed conv1d being on
     the critical path of our scalar C kernel; v1/v2 leave the conv to
     the plugin.

4. **The earlier framing in `latest_memory.md` was misleading**:
   - The chunked `pp512` numbers there (101 → 245 → 186 tok/s) were
     `chunk=128` × 4 infer calls, not what any user-facing app does.
     The honest user-facing number is the one above.
   - The "v3 below llama.cpp compute buffer" claim mixed our paper-
     budget walk (`get_runtime_model` × shape × dtype) with llama.cpp's
     resident compute buffer. These aren't the same axis; until we
     RSS-sample the running process we don't know the resident drop.
   - The doc compared baseline (PA fast path) to custom configs (SDPA
     fallback path), labelling the gap as kernel slowness when most of
     it was the path difference.

### What the gap actually is, today

Under the only comparison we can do honestly (same backend, same prompt,
same pipeline), the kernels we wrote are:

- Faster than the stock CPU plugin's stateful linear-attn path on
  prefill (v1/v2 substantially, v3 ~flat).
- Slower on decode by ~25–30 %.
- Locking the model onto the SDPA fallback by virtue of being present.

The largest single performance lever remaining is **getting the custom
op past `paged_attention_transformations`** so v1/v2/v3 can ride the
PA path the stock model gets for free. That alone is a ~3× prefill
swing; nothing in the kernel itself comes close.

---

## Memory: paper budget vs resident set (still unresolved)

`scripts/working/investigate_runtime.py` walks `get_runtime_model()`,
multiplies output shape × dtype, sums per bucket. This is **addressable**
bytes — the sum of live tensors if storage couldn't be shared. The
OpenVINO MemMgr pools and overlaps; resident memory is generally lower.

The headline numbers in `latest_memory.md` (391 vs 687 MiB linear-attn
activation) are paper budget. They are correct on that axis and v3
absorbs the 162-MiB `[1,6144,128]` Transposed-mixed_qkv tensors, the
111-MiB `[1,6144,132]` conv-concat tensors, and the 54-MiB `[1,6144,129]`
conv outputs outright. That part is real.

What is not real — without an RSS sampler — is the comparison to
llama.cpp's 491 MiB **resident** compute buffer. To make that
apples-to-apples we need:

- Process-level RSS sampled around one prefill.
- A `pmap` or `/proc/self/maps` rollup of allocations attributable to
  inference vs weights.
- Or oneDNN scratchpad accounting (`ONEDNN_VERBOSE=2` shows scratchpad
  sizes per primitive).

Until then, treat the −296 MiB / −383.5 MiB deltas as upper bounds on
the resident savings.

---

## Open levers (not yet attempted)

- Make `GatedDeltaRule{,V2,V3}` survive `paged_attention_transformations`,
  or document the failure and propose a skip pattern.
- The v3 decode cost is dominated by the per-call C kernel overhead
  (state read + 18 invocations × one token). The fixed cost matters
  more at T=1 than at T=770. Plausible mitigations:
  - SIMD-blocked inner loop on the recurrence matmul.
  - Hoist the state matmul out of the per-call C side and use OV
    primitives for the state·k / state·q steps.
- The `Slice|ref_f32` regression we measured in `investigate_runtime`
  (9.8 → 18.3 ms in v3's linear_attn at T_q=128) deserves its own
  investigation — the v3 rewrite must have introduced or grown some
  dynamic slices on a hot path.

Update: chasing these moved to the next session entry.

---

## 2026-06-01 (later): close the decode kernel gap via state-Assign bypass

`scripts/working/find_hot_slices.py` revealed that 18 nodes named
`layers.X.linear_attn/aten::slice/Slice_4` (one per linear-attn layer)
were each spending ~398 µs in `ref_f32` slicing a flat `[264192]` tensor
down to `[262144]` — totalling **7.17 ms = 78 % of v3 linear_attn time
at decode**.

Tracing the IR with `scripts/working/find_slice_source.py` exposed why:
the PyTorch export of the gated-delta Loop packs both outputs (per-T
result + new state) into a single flat tensor via
`Reshape -> Concat -> Slice -> Reshape -> Assign`. My v1/v2/v3 rewrites
replaced `loop.output(1)` with `v3.output(1)` but left this whole chain
alive, because the terminal `Assign` was reading through it.

Fix in `kernels/fused_linear_attn.py`: a new helper
`_find_gd_state_assign(loop.output(1))` walks the chain forward to the
`Assign`, then each rewrite rewires `assign.input(0)` directly to
`fused.output(1)`. The dead Reshape/Concat/Slice/Reshape nodes get
pruned at compile time.

Re-run of `run_probe_pa_path.sh` after the fix:

| version  | backend | TTFT (s) | prefill (tok/s) | decode (tok/s) |
|----------|---------|---------:|----------------:|---------------:|
| baseline | PA      |   2.99   |       **257.2** |          17.56 |
| baseline | SDPA    |  10.11   |          76.2   |          14.33 |
| v1       | PA      |   7.02   |         109.7   |          14.76 |
| v1       | SDPA    |   6.54   |         117.7   |          13.67 |
| v2       | PA      |   6.79   |         113.3   |          15.08 |
| v2       | SDPA    |   6.54   |         117.8   |    **15.79**   |
| v3       | PA      |   9.16   |          84.0   |          14.51 |
| v3       | SDPA    |   9.08   |          84.8   |          13.90 |

### Same-backend (SDPA on both sides) comparison vs baseline_sdpa

| version | prefill Δ | decode Δ |
|---------|----------:|---------:|
| v1      | **+54 %** |    −4.6 %|
| v2      | **+55 %** |   **+10 %** |
| v3      | **+11 %** |    −3.0 %|

**The decode kernel gap is closed.** Before the fix all three custom
configs were 23–28 % behind baseline_sdpa on decode. After: v2 beats
baseline_sdpa, v1 and v3 are within ~5 % of it. v2 is the throughput
sweet spot at both prefill and decode under the same backend.

The PA fast-path gap (baseline_pa 257 vs v3_pa 84 tok/s prefill)
remains — that lever is still about making the custom op survive
`paged_attention_transformations`, not about kernel quality.

### Lesson

The "kernel gap" between baseline and v3 was mostly an IR-leftover
problem, not a kernel-quality problem. A two-line fix in the rewrite
(bypass a single Concat→Slice that an old PyTorch export emitted to
pack two outputs into one Assign) reclaimed +21 %, +31 %, +21 % of
decode for v1, v2, v3. Future fusion work should grep for similar
"packed-output unpack" chains before assuming a perf gap is in C code.

---

## 2026-06-01 (later still): the memory finding the earlier benches missed

The probe was missing a memory column entirely. `probe_pa_path.py` now
samples `/proc/self/status` (VmRSS via a background thread, VmHWM at
end) around `VLMPipeline.generate()`. The rewrite + serialize step is
forked into a subprocess so it does not leave its own peak RSS in the
parent process.

VmHWM (resident high-water mark) at 770 input tokens, 32-token greedy:

| version  | backend | VmHWM (MiB) |
|----------|---------|------------:|
| baseline | PA      |   **2967**  |
| baseline | SDPA    |     3229    |
| v1       | PA      |     3473    |
| v1       | SDPA    |     3670    |
| v2       | PA      |     3652    |
| v2       | SDPA    |     3853    |
| v3       | PA      |     3408    |
| v3       | SDPA    |     3608    |

### Same-backend comparison (SDPA on both sides) — vs baseline_sdpa

| version | VmHWM Δ |
|---------|--------:|
| v1      | +441 MiB |
| v2      | +624 MiB |
| v3      | +379 MiB |

**This contradicts the paper-budget headline of `latest_memory.md`.**
Under apples-to-apples backend, every custom-op config uses *more*
resident memory than stock. The earlier "v3 saves 296 MiB activation
vs v1, beats llama.cpp's 491 MiB compute buffer" framing compared a
`get_runtime_model()` shape-by-dtype walk to llama.cpp's resident
compute buffer. Different axes. The actual resident-set winner is
**baseline_pa at 2967 MiB**.

Among the custom configs v3 is the smallest (−62 MiB vs v1, −245 MiB
vs v2), so the v1→v3 progression *does* monotonically improve our
own footprint. But that progression never crossed under stock's curve.

### Where the regression likely lives

- The `.so` itself, its libgomp/jit globals, and the ctypes plumbing.
- Per-call heap scratch in `gdr_kernel_v3` (~3 MiB × 18 layers ≈ 54 MiB,
  not enough on its own).
- OpenVINO's MemMgr can pool tensors aggressively around its own
  primitives but treats the boundary of a custom op as opaque — the
  state and output tensors of our op cannot share storage with
  surrounding compute the way plugin nodes do. That is the most
  plausible source of the rest.

### PA path is also more memory-efficient

- baseline: PA 2967 vs SDPA 3229 → **−262 MiB**
- v3:       PA 3408 vs SDPA 3608 → **−200 MiB**

So getting the custom op past `paged_attention_transformations` would
buy both the prefill throughput AND a ~200 MiB resident win.

### Lessons (compounding)

1. **Measure RSS, not paper budget.** A `get_runtime_model()` shape
   sum is what the plugin would allocate if it couldn't share. It can.
   The two numbers can move in opposite directions.
2. **Always fork the prep.** Any model-rewrite step that allocates
   inside the measurement process inflates the baseline; subprocess
   isolation gives a number you can trust.
3. The earlier `latest_memory.md` 391-vs-491-MiB claim should be
   retracted; the actual peak resident memory comparison is the one
   above and stock wins on its own backend.
