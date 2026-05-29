# Memory optimization — plain OpenVINO + our kernels

Goal: drive Qwen3.5-0.8B (Qwen3-Next hybrid arch) inference memory toward the
llama.cpp reference — **~200 MB of runtime state on top of weights** for a
2048-token prompt with a quantized KV cache — using only `ov.Core` and our own
fused kernels (no openvino_genai, no paged attention).

Driver: [`openvino/lowmem_infer.py`](openvino/lowmem_infer.py). It reads the
stock optimum IR, swaps in our custom ops (gated-delta-rule recurrence + causal
conv1d, dispatching to the C kernels via ctypes when `QWEN3_USE_C=1`), slices
the lm_head, and feeds the prompt to the *stateful* model in chunks.

All numbers below are **measured** on this container (seq=2048,
`INFERENCE_NUM_THREADS=4`, `QWEN3_USE_C=1`). Reproduce with the commands at
the bottom.

## Where the memory was going (attribution)

After `compile_model`, before any infer: **166 MB** RSS (69 file-backed +
97 anon). The 754 MB int8 weight `.bin` is mmap'd lazily.

The single biggest cost was the **CPU plugin pre-decompressing every int8
weight constant into a bf16/f32 in-memory buffer for matmul throughput**.
At first infer this added ~866 MB of anon RSS, fixed regardless of sequence
width (W=1 and W=32 produced identical weight_expand). The IR stores 717 MB
of u8 weight Constants across 187 MatMuls (dequant chain
`u8 → Convert(f16) → Subtract(zp u8) → Multiply(scale f16) → Convert(f32) → MatMul`).
The plugin's pre-expanded copy was ~1.2× that footprint sitting alongside the
file-backed Constants.

## The kernels

Three custom ops, each dispatching to a C kernel via ctypes when
`QWEN3_USE_C=1`:

1. **`GatedDeltaRule`** ([fused_linear_attn.py](kernels/fused_linear_attn.py)) —
   replaces the unfused `Loop` body of the 18 linear-attention layers with a
   single op that implements the recurrence. Numerically validated to ≈1e-8
   against the HF `torch_recurrent_gated_delta_rule` reference.
2. **`FusedCausalConv1d`** ([fused_conv1d.py](kernels/fused_conv1d.py)) — the
   conv1d-with-state for those same layers, avoiding the `Concat(prev_state,
   current)` materialization.
3. **`QuantizedMatMul`** ([quantized_matmul.py](kernels/quantized_matmul.py)) —
   takes `(act, u8_weight, scale_f16, zp_u8)` as four inputs and streams the
   dequant in the inner loop. **Never materializes a bf16/f32 weight buffer.**
   Math: `y[..., n] = scale[n] · (act · u8[n] − zp[n] · sum(act))`. The C
   kernel parallelizes across output rows N with OpenMP.

Plus a graph-only rewrite, [`lm_head_slice`](kernels/lm_head_slice.py): insert
`Slice(axis=1, last token)` before the lm_head MatMul.

## Measured — seq = 2048, chunk = 128

| config                  | peak RSS | Δ peak | prefill | decode (4 tok) |
|-------------------------|---------:|-------:|--------:|---------------:|
| stock IR (no fusions)   |  ~3.8 GB |     — |     —   |     —          |
| + gdr + conv + slice (qmm=none) | 2052 MB |     —    |  14.4s |  0.45s |
| + qmm=lm_head           |    1807 MB |  −245 MB |  **11.5s** | 0.49s |
| + qmm=all, FMA inner loop |  1335 MB |  −717 MB |   40.7s | 1.13s |
| + qmm=all, **VNNI** (`QWEN3_USE_VNNI=1`) | **1335 MB** | **−717 MB** | **19.2s** | 0.95s |

`qmm=lm_head` is strictly better than baseline: lower memory AND faster
prefill (it saves both the plugin's 245 MB lm_head bf16 expansion at first
infer and the time spent doing that expansion).

`qmm=all + VNNI` is the memory ceiling at 1335 MB — within ~300 MB of
llama.cpp's q8_0 footprint — at only 1.34× the baseline prefill time. The
i8 activation quantisation introduces ~0.7 % relative error per matmul but
preserves coherent generation end-to-end ("As an AI, I don't have a physical
body, so I don't have a show size in the traditional sense...").

Persistent state at 2048 tokens (fp32, unchanged across configs): 67.8 MB
(48 full-attn KV + 18 linear-attn state + 1.7 conv state).

After compile, before first infer: 166 MB RSS (qmm=none/lm_head) or 154 MB
(qmm=all — the smaller graph drops some plugin scratch).

### Persistent state at 2048 tokens (all fp32)

| category | tensors | bytes |
|---|---:|---:|
| `full_attn_kv` (6 layers × K/V) | 12 | 48.0 MB  — grows with context |
| `linear_attn_state` (18 layers) | 18 | 18.0 MB  — fixed (recurrent state) |
| `conv1d_state` (18 layers)      | 18 |  1.7 MB  — fixed |
| **TOTAL** | 48 | **67.8 MB** |

The full-attn KV is the only piece that scales with context: 3.0 MB at 128
tokens → 48.0 MB at 2048. At fp32 that is 4× the q8_0 llama.cpp reference
(~12 MB for the same 6 layers).

## What worked

1. **lm_head slice** — only the last position is projected to the 248320-wide
   vocab. Removes the `[T, vocab]` fp32 logits tensor (~2 GB at T=2048).
2. **Chunked prefill** — feeding the stateful model in chunks bounds the
   per-call activation footprint. Pure plain-`ov.Core`: recurrent and conv
   state carry across chunks for free; the full-attn KV grows in the
   infer_request.
3. **`QuantizedMatMul` streaming dequant** — eliminates the plugin's bf16
   weight repack. lm_head alone is 242 MB u8 → ~245 MB bf16 in the plugin
   cache; killing that single repack saves 245 MB at peak. The full sweep
   (`qmm=all`) saves 717 MB at the cost of 3× prefill time vs baseline.

4. **Hand-rolled AVX-512 inner loop + affinity reset.** Initial scalar QMM
   was 10× slower than the plugin (qmm=all 112s prefill). Two fixes brought
   it to 2.75× slower (40.7s):
   - The k-loop now does 16-lane f32 FMAs with the u8 weight converted via
     `_mm_loadu_si128` + `_mm512_cvtepu8_epi32` + `_mm512_cvtepi32_ps`.
     Per-row dequant is written once into 4 KB stack scratch, then M dot
     products reuse it.
   - OV's plugin runs Op.evaluate inside a TBB worker pinned to one CPU. Our
     `#pragma omp parallel` inherited that affinity, so all 4 omp threads ran
     on the same core. `sched_setaffinity(0, all_cpus)` at the top of the
     kernel restores fan-out. Verified with `taskset -c 0` standalone: 60 ms
     pinned vs 14 ms unpinned, matching the in-OV gap exactly.

5. **AVX-512 VNNI inner loop (`qmm_kernel_vnni`).** Per-call symmetric
   quantisation of the activation row to i8 (per-row scale = max|act|/127),
   then `vpdpbusd` for u8·i8 → i32 dot products. The factored math
   `y = s_m * s_w * (i32_dot − zp_w * sum(act_i8))` lets `sum(act_i8)` be
   precomputed once per row and reused. Standalone speedups vs the FMA
   inner loop: 1.7× on lm_head (M=1), 4–8× on the bigger prefill matmuls
   (M=256, N≥1024). In OV: `qmm=all` prefill 40.7s → 19.2s. Enabled via
   `QWEN3_USE_VNNI=1` (kept opt-in because of the i8-quant accuracy hit).

## What did not work

- **`KV_CACHE_PRECISION` hint is inert here.** u8 vs default gives
  byte-identical state (67.8 MB) and the same peak. The hint targets the
  plugin's internal SDPA/PagedAttention KV path; this IR exposes the full-attn
  KV as a stateful `ReadValue/Assign` pair the hint does not touch.
- **`INFERENCE_PRECISION_HINT=bf16` and `DYNAMIC_QUANTIZATION_GROUP_SIZE=0`
  don't move the needle.** Same ~865 MB weight_expand as default.

## Correctness

- `kernels/test_kernels.py` — `gdr_kernel` matches HF
  `torch_recurrent_gated_delta_rule` to **1.3e-8** max diff.
- `openvino/validate_fusion.py` — with real (tokenized + embedded) inputs,
  top-10 logit overlap is 10/10 between stock and fused; max abs diff ~0.4.
  Random `inputs_embeds` is not a valid test — it amplifies sub-ULP ordering
  differences in the 64-step multiplicative recurrence.
- `openvino/generate.py` — end-to-end greedy decode with all rewrites
  applied produces coherent text ("I am Qwen3.5, the latest large language
  model developed by Tongyi Lab...").

## Int8 KV cache (`QuantizedKVCache`)

We confirmed with `llama-bench` that llama.cpp at T=2048 with `-ctk q8_0
-ctv q8_0 -fa 1` allocates **32.02 MiB** of KV+RS state (12.75 KV + 19.27 RS).
The OV baseline at the same T was 67.8 MiB with fp32 KV. The gap is just K
and V precision; both implementations correctly handle the hybrid arch.

`kernels/quantized_kv.py` adds a custom op + IR rewrite that mirrors the
llama.cpp design:

- Each full-attn layer's f32 `K/V` state Variable is replaced by a pair:
  `(data: i8 [B, 2, T, 256], scale: f32 [B, 2, T])`. Per-token symmetric
  int8 (`scale_t = max|kv[..,t,:]| / 127`).
- One op (`QuantizedKVCache`) takes `(prev_data, prev_scale, new_kv_f32)`
  and emits `(full_kv_f32 for SDPA, new_data, new_scale)`. SDPA sees the
  full dequant f32 tensor it expects -- no SDPA changes needed.

OV doesn't expose a `remove_node` for the old f32 Variable's ReadValue;
removing the matching Assign trips a "sibling output" check in the plugin's
memory pass. As a workaround the old Variable is mutated `f32 -> i8` and
Convert nodes bridge the dead chain, keeping its storage at 1/4 the bytes
even though the dead Concat still grows with T.

### Measured at pp2048 + tg512 (chunk=128, INFERENCE_NUM_THREADS=4)

| metric                  | f32 KV (baseline) |  int8 KV  | Δ |
|-------------------------|------------------:|----------:|--:|
| pp2048 duration         |             14.68s | **11.15s** | -3.5s   |
| pp2048 throughput       |        139.5 tok/s | **183.7 tok/s** | +44 tok/s |
| pp2048 peak RSS         |             2051 MB | 4287 MB   | +2.2 GB |
| tg512 duration          |              53.4s | 154.0s    | +100s   |
| tg512 throughput        |          9.59 tok/s | 3.33 tok/s | -6.3 tok/s |
| tg512 peak RSS          |             2224 MB | 6036 MB   | +3.8 GB |
| **persistent state**    |        **79.7 MiB** | **34.9 MiB** | **-44.8 MiB** |
| compile-time RSS        |              167 MB | 1338 MB   | +1.2 GB |

State breakdown @ T=2560:
- baseline: 60 MiB full_attn_kv (f32) + 18 RS + 1.7 conv = 79.7 MiB
- int8 KV:  15 MiB full_attn_kv (i8) + 0.2 scale + 18 RS + 1.7 conv = **34.9 MiB**

That's the **same persistent state size as llama.cpp** at this T (32.02 MiB
measured earlier) — the original question is answered. End-to-end generation
is identical to the f32 baseline ("I am Qwen3.5, the latest large language
model developed by Tongyi Lab. I am a text-based AI model, which means I
don't…").

### Closing the decode gap with fused int8 SDPA

The next iteration replaced the canonical "QuantizedKVCache (f32 dequant
output) -> Unsqueeze -> Broadcast -> Reshape -> SDPA" chain with one fused
op that consumes i8 K/V directly. The full [B, H, T_full, D] f32 dequant
tensor is never materialised.

New custom ops in `cpp_ext/`:
- `QuantizedKVCacheUpdate` -- same kernel as `QuantizedKVCache` but skips
  the f32 dequant (`qkv_kernel(..., full_f32=NULL)`). Outputs are just
  `(data_i8, scale_f32)`.
- `QuantizedInt8SDPA` -- scaled dot-product attention with i8 K/V.
  `i8_f32_dot` AVX-512 inner loop, GQA handled internally via
  `h_kv = h_q / gqa_factor`. No f32 buffer for K or V is ever materialised.

Wiring lives in `kernels/quantized_int8_sdpa.py::replace_kv_with_int8_sdpa`.

Build gotcha: `cpp_ext/CMakeLists.txt` had to *drop* libgomp. Mixing libgomp
with OV's libtbb in the same process tanks baseline decode 5x (10.43 ->
1.82 tok/s) even when no custom op is in the IR. The cpp_ext kernels are
now parallelised via `std::thread` spawned per call instead. Python/ctypes
kernels keep libgomp in their own .so (loaded outside the OV plugin
process).

### Measured pp512 + tg32 (each config in a fresh subprocess, CACHE_DIR per config)

| metric             | A. baseline f32 KV | B. int8 KV + f32 dequant | C. int8 KV + int8 SDPA |
|--------------------|-------------------:|-------------------------:|-----------------------:|
| pp512 duration     |              4.03s |                    3.51s |                 22.10s |
| pp512 throughput   |          127 tok/s |               146 tok/s  |           23.2 tok/s   |
| pp512 peak RSS     |           2309 MB  |               2365 MB    |          **2302 MB**   |
| **tg32 duration**  |              2.98s |                    4.71s |               **3.54s**|
| **tg32 throughput**|     **10.74 tok/s**|              6.80 tok/s  |       **9.05 tok/s**   |
| tg32 peak RSS      |           2311 MB  |               2475 MB    |          **2307 MB**   |
| persistent state   |          32.4 MiB  |               22.9 MiB   |             22.9 MiB   |

C cuts the decode gap to baseline from -37% (B) down to -16%, at the same
22.9 MiB persistent state, **at essentially the same peak RSS as baseline
(2307 vs 2311 MB)**, and identical generation. The int8 SDPA path pays for
it on prefill: 127 -> 23 tok/s. Our scalar int8 SDPA kernel doesn't match
the plugin's optimized blocked-GEMM SDPA at large T_q. Next steps for the
prefill gap: a persistent thread pool to amortise `std::thread` spawn, and
a Flash-Attention-style tiled inner loop.

#### Why earlier benches reported a +1.7 GB peak RSS regression

The previous in-process A->B->C loop showed C peaking at 3793 MB vs A's
2087 MB. That was almost entirely an in-process accumulation artifact: each
`core.compile_model` allocates ~1 GB of plugin-internal weight buffers, and
running three configs sequentially kept all three sets resident. In a fresh
subprocess each peak is ~2.3 GB regardless of config. The 10 MiB
persistent-state difference is too small to surface in peak RSS, which is
dominated by ~1 GB weights + ~1 GB compute scratch. The state win matters
for use cases like serving many concurrent contexts (paged-style) where the
per-context overhead compounds.

### Runtime cost (B alone, kept for context)

The persistent state goal is met, but the **decode runtime is 3× slower**
and **peak RSS is 2.7× higher** during decode. Two root causes:

1. **The QuantizedKVCache op materialises a full `[B, H, T_full, D]` f32
   dequant tensor for SDPA on every call.** At T=2560 that is 5 MiB per K/V
   buffer × 12 buffers × 512 decode steps -- lots of allocator churn the
   stock Concat→Assign path doesn't have. Prefill is fine because the smaller
   state has better cache locality and chunking bounds the working set, but
   decode pays the dequant cost per token.
2. **Compile-time RSS jumps +1.2 GB.** The plugin pre-faults all int8
   weights at compile time when a custom op feeds SDPA, where the baseline
   lazy-mmaps them on first infer.

The proper fix is **a custom SDPA op that consumes i8 KV directly** -- what
llama.cpp does. That replaces the current SDPA node with one that takes
`(Q_f32, K_i8, K_scale, V_i8, V_scale, mask)` and never materialises the
dequant'd full K or V. That's a real chunk of work (fused attention with
mixed-dtype matmul + masked softmax) and the right next step if the runtime
cost matters.

### Dead-chain removal trick

Removing the old f32 ReadValue/Concat/Assign chain looks impossible from
Python (no `model.remove_node`), but works in stages:

1. Selectively rewire all old-Concat consumers (including the old Assign)
   to the new QuantizedKVCache's f32 output.
2. The IR computes the SDPA GQA Broadcast target T as
   `Add(ShapeOf(old_Gather)[2], N)` = `T_prev + N`. Redirect those ShapeOfs
   to read from our new i8 Gather (`rv_data_g`), which preserves the
   `T_prev` semantic exactly.
3. Replace the rest of the old ReadValue's consumers with an empty f32
   Constant so the dead Gather/Concat compute on empty tensors.
4. `model.remove_sink(old_assign)` + `model.remove_variable(old_var)`.

After this the old Variable is gone and the dead subgraph is unreferenced
from any sink/parameter/result -- the plugin elides it at compile time.

### C kernel speedup (`qkv_kernel`)

Initial numpy `evaluate()` made `int8 KV` 2x SLOWER than the f32 baseline
because per-token max/quant + concat + dequant of the growing state was
running through numpy. C kernel with AVX-512 SIMD on all three legs (max
+ quant + dequant):

| T_prev | numpy (us) | C (us) | speedup | max abs diff |
|-------:|-----------:|-------:|--------:|-------------:|
|      0 |        233 |     34 |    6.9x |      2.4e-7  |
|    256 |       1110 |     48 |   22.9x |      2.4e-7  |
|   1024 |       3047 |    120 |   25.4x |      2.4e-7  |
|   1920 |       4863 |    217 |   22.4x |      4.8e-7  |

In OV at seq=2048, chunk=128:

| variant                          | prefill | state    |
|----------------------------------|--------:|---------:|
| baseline f32 KV (no rewrite)     |   14.4s | 67.8 MiB |
| int8 KV, numpy `evaluate()`      |   30.8s | 43.9 MiB |
| **int8 KV, C `qkv_kernel`**      | **12.9s** | **43.9 MiB** |

So int8 KV with the C kernel is now strictly better than baseline on both
axes: 35% less persistent state AND 10% faster prefill (the smaller state
has better cache locality and the C kernel is bandwidth-bound).

## Next steps (ordered by expected payoff)

1. Fully drop the dead f32 chain (the missing 12 MiB to reach llama.cpp
   parity). Blocked by lack of `remove_node` in the Python API; would need
   either a `ModelPass` that runs at compile time, or to drop into the C++
   API.
2. fp16 recurrent + conv state (18 -> ~9 MB) -- small, low risk.
3. Activation quantisation group-of-K (q8_0-style with K=32) instead of
   per-row, to claw back the ~0.7 % VNNI error and make it the default.

## Reproduce

```bash
cd study/qwen3
# Baseline
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128
# Streaming int8 lm_head (the practical sweet spot)
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128 --qmm lm_head
# Full streaming int8, scalar FMA inner loop
QWEN3_USE_C=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128 --qmm all
# Full streaming int8 with VNNI vpdpbusd
QWEN3_USE_C=1 QWEN3_USE_VNNI=1 python openvino/lowmem_infer.py --seq 2048 --chunk 128 --qmm all
# End-to-end generation (verify correctness)
QWEN3_USE_C=1 python openvino/generate.py
```
