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

---

## 2026-06-01 (still later): raw ov.Core() probe — is the regression genai?

`scripts/working/probe_raw.py` + `run_probe_raw.sh` repeat the
770-token one-shot prefill + 32-token decode at the raw `ov.Core()`
layer, no `ov_genai.VLMPipeline` involved. Subprocess prep for v1/v2/v3,
same RSS sampler, same THREADS=4.

| version  | prefill (tok/s) | decode (tok/s) | VmHWM (MiB) |
|----------|----------------:|---------------:|------------:|
| baseline |          113.6  |         14.01  |  **2248**   |
| v1       |          189.6  |         15.19  |    2706     |
| v2       |          186.6  |         15.72  |    2885     |
| v3       |          114.4  |         15.27  |    2648     |

### Compute (raw layer, same backend on both sides)

| version | prefill Δ vs baseline | decode Δ vs baseline |
|---------|----------------------:|---------------------:|
| v1      | **+67 %**             | **+8.4 %**           |
| v2      | **+64 %**             | **+12.2 %**          |
| v3      | +0.7 %                | **+9.0 %**           |

All three fusions are now strictly faster than stock at both prefill and
decode at the raw layer. v3 trades almost all prefill back for the
absorbed conv1d but holds the decode win.

### Memory (raw layer)

| version | VmHWM Δ vs baseline |
|---------|--------------------:|
| v1      |             +458 MiB |
| v2      |             +637 MiB |
| v3      |             +400 MiB |

Compared to the genai-SDPA matrix (+441 / +624 / +379), the raw and
genai deltas are within ±20 MiB of each other. **The memory regression
is structural to the custom-op boundary, not a genai-pipeline artifact.**

### Genai's own overhead, as a side product

| metric          | raw baseline | genai baseline_pa | genai baseline_sdpa |
|-----------------|-------------:|------------------:|--------------------:|
| VmHWM           |    2248 MiB  |          2967 MiB |            3229 MiB |
| genai overhead  |       —      |         **+719**  |          **+981**   |

A genai user pays ~720 MiB just for the VLM machinery on top of the LM
(tokenizer + detokenizer + vision encoder + merger + embed + sampler +
PA block manager). The custom-op fusion adds another ~400 MiB on top.

### Honest end-to-end summary

- **Compute (kernels):** v1/v2/v3 are faster than stock at the raw
  layer on both prefill and decode. v2 wins.
- **Memory (resident):** v1/v2/v3 are +400–637 MiB worse than stock at
  both raw and pipeline layers. v3 is the smallest of the customs.
- **Pipeline overhead:** ~+720–980 MiB for genai's VLM stack on top of
  whatever the LM weighs, independent of fusion choice.

The interesting open question is not "are our kernels fast" — they are
— but "why can't the plugin pool storage around our custom op the way
it pools around its own primitives", and "can a partial absorption
(keep more of the fused chain as plugin ops) keep the speed and recover
the memory".

---

## 2026-06-01 (still later): llama.cpp parity attempt

Goal: get the OV-side resident memory down to llama.cpp's footprint on
the same workload (Qwen3.5-0.8B int8-equivalent, 770pp + 32tg, 4 threads).

`/usr/bin/time -v llama-bench -m /tmp/qwen35-0.8b-Q8_0.gguf -p 770 -n 32 -t 4 -r 1`
gives `Maximum resident set size: 976396 KiB` → **953 MiB total RSS**.

For comparison the raw-OV baseline measured earlier is **2248 MiB**.
Gap = **1295 MiB**, larger than anything fusion can move.

### Knob sweep (raw ov.Core(), baseline LM, prompt_len=770)

`scripts/working/run_probe_raw_lowmem.sh` drives one config per fresh
process, each with the same RSS sampler. Best results:

| config                       | VmHWM (MiB) | prefill (tok/s) | decode (tok/s) |
|------------------------------|------------:|----------------:|---------------:|
| default                      |       2250  |          116.4  |          15.35 |
| INFERENCE_PRECISION_HINT=bf16|       2060  |           92.9  |          13.41 |
| bf16 + u8 KV + latency + …   |   **2058**  |          105.9  |          13.49 |
| (v3 fusion, default knobs)   |       2648  |          114.4  |          15.27 |
| **llama.cpp Q8_0 reference** |     **953** |              —  |              — |

`INFERENCE_PRECISION_HINT=bf16` is the only knob worth ~190 MiB. KV-cache
u8, single stream, latency hint, dynamic-quant groups all move VmHWM by
≤4 MiB. The combo of every memory-leaning knob lands at 2058 MiB —
still **+1105 MiB over llama.cpp**.

### Why the gap is structural

`scripts/working/probe_raw_breakdown.py` reads `/proc/self/smaps_rollup`
before and after warmup at T ∈ {32, 128, 512, 770}:

|  T  | VmHWM (MiB) | smaps File-backed | smaps Anonymous |
|-----|------------:|------------------:|----------------:|
|  32 |       1670  |             0 MiB |       886 MiB   |
| 128 |       1746  |             0 MiB |       962 MiB   |
| 512 |       2033  |             0 MiB |      1248 MiB   |
| 770 |       2212  |             0 MiB |      1427 MiB   |

Two facts that close the explanation:

1. **`File-backed = 0` everywhere.** OV's `ENABLE_MMAP=True` is on by
   default and does mmap the IR for *reading*, but compile_model copies
   the weights into anonymous memory (for weight repack / prepack /
   format conversion). llama.cpp mmaps the GGUF and keeps it
   file-backed through inference, so those ~720 MiB of weights sit in
   the page cache (shared, not counted against RSS).
2. **The compute buffer scales linearly with T** at ~0.73 MiB/token, on
   top of a ~1500 MiB fixed cost at T=32. llama.cpp's total RSS at
   T=770 is 953 MiB — *less than OV's compute buffer alone at T=32*.

Roughly attributing the gap:
- ~720 MiB: OV's anonymous-copy of weights vs llama.cpp's mmap'd GGUF.
- ~400 MiB: larger fixed compute buffer (oneDNN scratchpads, runtime
  graph structures, JIT cache).
- compute buffer growth with T diverges further at larger prompts.

### Implications for the v1/v2/v3 fusion work

Fusion was a kernel/activation lever. It is not the right tool for this
gap — at best it touches a single-digit-percent of the compute buffer,
and v3 in particular shows up as a +400 MiB regression (custom-op
boundary defeats the pool) on top of an already-+1295 MiB-over-llama.cpp
baseline.

If the goal is llama.cpp parity, the levers are:
- Get OV to keep weights file-backed through compile (engine-level work
  in the CPU plugin's transformation passes — non-trivial).
- Use a backend with smaller fixed compute buffer (NPU? GPU has its own
  memory; CPU plugin is the largest).
- Switch engines for this workload.

The fusion work delivers real **compute** wins (v1/v2/v3 all faster
than stock on prefill and decode at the raw layer) and a +400 MiB
**memory** regression. It is not in the path of llama.cpp parity.

### Open question

Whether OV exposes a per-`compile_model` flag that prevents weight
repack (forcing native-int8 kernels at some perf cost) hasn't been
found. The `SUPPORTED_PROPERTIES` enumeration on CPU includes
`CPU_SPARSE_WEIGHTS_DECOMPRESSION_RATE`, `ENABLE_WEIGHTLESS`,
`WEIGHTS_PATH` — none obviously match. If such a flag does not exist,
closing the gap requires patching the CPU plugin.

### Where the 0.73 MiB/token actually comes from

A flag of `0.73 MiB/token` is too high for this architecture
(6 self-attn + 18 linear-attn layers, hidden=1024, GQA). Architecturally
the KV cache should grow at ~0.025 MiB/token (6 self-attn layers ×
2 × kv_heads × head_dim × 4 bytes), and the 18 linear-attn layers
should be O(1) per token (recurrent state has no T axis).

`scripts/working/find_per_token_growth.py` walks `get_runtime_model()`
at T=32 and T=770 and reports per-node Δbytes/token, bucketed:

| bucket      | Σ paper Δ/token | T=770 paper | T=32 paper |
|-------------|----------------:|------------:|-----------:|
| linear_attn |        4.64 MiB |     3580 MiB|    152 MiB |
| other       |        0.92 MiB |      711 MiB|     32 MiB |
| self_attn   |        0.52 MiB |      397 MiB|     17 MiB |
| mlp         |        0.33 MiB |      253 MiB|     11 MiB |
| total paper |    **6.41 MiB** |   **4942**  |   **212**  |

Versus actual measured RSS Δ/token: **0.73 MiB**. So the plugin's pool
recovers ~89 % of the paper budget. The remaining 0.73 is dominated by
the linear_attn bucket.

Per linear-attn layer the IR contains ~18 separate `[1, T, 6144]` f32
buffers — every Transpose / Multiply / Reshape / Split / L2-norm
output in PyTorch's eager-style decomposition is a distinct IR tensor.
24 KiB/token × 18 nodes × 18 layers = ~7.6 MiB/token paper, close to
the measured 4.64 (some nodes alias / share). After the pool: 0.73.

**The architecture does not demand 0.73 MiB/token; the IR representation
does.** The lever is to fuse more of the per-layer ops into a single
primitive so the IR-level intermediate set collapses. But our v3
fusion's custom-op boundary defeats the plugin's pool on its own
outputs (worth +400 MiB at T=770), and that lost-pool cost is larger
than the absorbed paper budget reclaim. Net: v3 RSS is worse than
baseline.

### What it would take to actually shrink resident memory

- Fuse the entire linear-attn module into one custom op (eliminate the
  18 × `[1, T, 6144]` intermediates from the IR), AND
- Get the plugin's MemMgr to pool through that op's I/O the way it
  pools through its own primitives. This is engine-level work in the
  CPU plugin's memory allocator — declaring our op's outputs as
  in-place-reusable, or marking input/output tensor descriptors as
  alias-eligible.

Without (2), absorbing more into the kernel makes resident memory
*worse* by trading 89 %-poolable IR tensors for 0 %-poolable custom-op
boundary tensors.
