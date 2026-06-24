# llama.cpp – Q4_K Prefill Handling: CUDA vs Vulkan vs SYCL

Analysis based on llama.cpp `master` branch (June 2026).
Source: direct reads of raw GitHub files + PR/issue history.

---

## Background

**Prefill** is the phase where LLM inference processes the initial prompt.
Key characteristics:
- `ne11` (number of query tokens / batch rows) is large — hundreds to thousands
- All attention heads process the full prompt simultaneously
- Two operations dominate cost:
  1. **Weight matmul**: projection layers (Q/K/V/FFN) with q4_k-quantized weights
  2. **Flash attention**: `O = softmax(Q × Kᵀ / √d) × V`

**Q4_K** is llama.cpp's mixed 4/6-bit "k-quant": 256-element blocks, per-block
scale/min stored in 6 bits (`block_q4_K`).

---

## 1. CUDA Backend (reference)

### 1a. Matmul constants (confirmed from source)

```c
// mmvq.cuh
#define MMVQ_MAX_BATCH_SIZE 8

// mmq.cuh
#define MMQ_DP4A_MAX_BATCH_SIZE 64
```

### 1b. `ggml_cuda_should_use_mmvq` (verbatim, mmvq.cu)

For non-CDNA GPUs: `return ne11 <= MMVQ_MAX_BATCH_SIZE` (i.e. `ne11 <= 8`).

For CDNA1 specifically, Q4_K: `return ne11 <= 2`.
For CDNA2 specifically, Q4_K: `return ne11 <= 3`.

### 1c. `ggml_cuda_should_use_mmq` (verbatim, mmq.cu) — Q4_K path

```c
// Q4_K is in the mmq_supported list (mmq_supported = true)

if (turing_mma_available(cc)) {
    return true;  // always MMQ on Turing+ with MMA
}
if (GGML_CUDA_CC_IS_NVIDIA(cc)) {
    return !fp16_mma_hardware_available(cc) || ne11 < MMQ_DP4A_MAX_BATCH_SIZE;
    // → MMQ when ne11 < 64 (if FP16 MMA present) or always (if no FP16 MMA)
}
if (amd_mfma_available(cc)) {
    if (ne11 <= 256 && (type == GGML_TYPE_Q4_K || type == GGML_TYPE_Q5_K)) {
        return true;  // MMQ when ne11 ≤ 256
    }
    return false;
}
```

### 1d. CUDA matmul dispatch ladder for Q4_K

```
ne11 ≤ 8        →  MMVQ  (warp-based, minimal shared memory)
ne11 < 64       →  MMQ   (tile-based, DP4A integer dot)    [NVIDIA without Turing MMA]
ne11 ≥ 64       →  dequantize q4_k → FP16 → cuBLAS GEMM   [NVIDIA without Turing MMA]
```
On Turing+ with tensor-core MMA: MMQ is always preferred over cuBLAS for Q4_K.

### 1e. Flash attention — Q4_K KV cache (verbatim, fattn.cu)

```c
switch (K->type) {
    case GGML_TYPE_F32:  case GGML_TYPE_F16:
        break;
    case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
#ifndef GGML_CUDA_FA_ALL_QUANTS
        return BEST_FATTN_KERNEL_NONE;
#endif
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_BF16:
        break;
    default:
        return BEST_FATTN_KERNEL_NONE;   // ← Q4_K lands here
}
```

`ggml_cuda_flash_attn_ext_supported()` returns `false` for Q4_K.
`ggml_cuda_flash_attn_ext()` calls `GGML_ABORT("fatal error")` if dispatched
with an unsupported type. Callers must gate on `supported()` first.

### 1f. CUDA flash attention kernel hierarchy

| Q->ne[1] (query tokens) | Kernel |
|---|---|
| 1 (or ≤2 with GQA) | VEC — decode-optimised scalar |
| small (Volta) | TILE |
| >2 with MMA support | **MMA_F16** — tensor-core tiled flash attention |

The MMA path is Turing+ only and is the highest-priority prefill kernel.

---

## 2. Vulkan Backend

### 2a. Matmul — q4_k weights

Dedicated GLSL compute shaders, three size variants (large/medium/small) that
vary by batch size. Integer dot product pipeline (`mul_mat_l_int[]` etc.) when
the device supports it. Decodes Q4_K inline; no separate dequantization step.

### 2b. Flash attention — q4_k KV cache

Three shader paths: `FA_SCALAR`, `FA_COOPMAT1`, `FA_COOPMAT2`.

The `flash_attn_dequant.glsl` inline-decode type list (confirmed verbatim):
```
FA_TYPE_F32, FA_TYPE_F16, FA_TYPE_BF16
FA_TYPE_Q4_0, FA_TYPE_Q4_1
FA_TYPE_Q5_0, FA_TYPE_Q5_1
FA_TYPE_Q8_0, FA_TYPE_Q1_0
```

**Q4_K is absent.** The `flash_attn_cm2.comp` cooperative-matrix shader uses
the same type set. Q4_K KV cache is not supported in Vulkan flash attention.

---

## 3. SYCL Backend

### 3a. Kernel paths for mul_mat

SYCL has three distinct execution paths, unlike CUDA's two main paths:

| Path | File | Operation |
|---|---|---|
| DMMV | `dmmv.cpp` | Dequantize on-the-fly, then scalar mat-vec |
| MMVQ | `mmvq.cpp` | Quantized dot-product mat-vec (no dequant) |
| MMQ  | `mmq.cpp`  | Tile-based quantized mat-mat (batched) |

Q4_K is handled by all three paths. The MMVQ path has a **reorder (SoA)**
variant for Q4_K (added in PR #13109) which reorganises data for coalesced
memory access on Intel GPUs.

**Measured decode performance (Intel Arc Pro B70, issue #21517):**
- Q4_K_M via MMVQ+reorder: 20.56 t/s, 53% memory bandwidth utilisation
- Q4_K_M forced to DMMV: 12.38 t/s, ~38% bandwidth
- Q8_0 (stuck on DMMV, no reorder): 4.88 t/s, 21% bandwidth

The batch-size dispatch logic between DMMV, MMVQ, and MMQ lives in
`ggml_sycl_mul_mat` (~line 3526 of `ggml-sycl.cpp`). No `SYCL_MMVQ_MAX_BATCH_SIZE`
constant is publicly defined; exact ne11 thresholds were not extractable from
the untruncated source.

### 3b. Flash attention — Q4_K KV cache (verbatim, fattn.cpp)

```c
switch (K->type) {
    case GGML_TYPE_F32:  case GGML_TYPE_F16:
        break;
    case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
#ifndef GGML_SYCL_FA_ALL_QUANTS
        return BEST_FATTN_KERNEL_NONE;
#endif
    case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0:
        break;
    default:
        return BEST_FATTN_KERNEL_NONE;   // ← Q4_K lands here
}
```

Identical type rejection to CUDA. Dispatch:
```c
case BEST_FATTN_KERNEL_NONE:
    GGML_ABORT("Not support Flash-Attention");
```

### 3c. SYCL flash attention kernel hierarchy

| Q->ne[1] | Kernel selected |
|---|---|
| ≤1 (float KV, no GQA) or ≤2 (quantized KV) | VEC |
| otherwise | TILE |

**SYCL is missing the MMA/XMX kernel tier entirely.** CUDA has:
`VEC → TILE → WMMA → MMA_F16`; SYCL only has `VEC → TILE`.

There is no kernel in SYCL that uses Intel XMX (DPAS) matrix-multiply
instructions for flash attention. Both VEC and TILE use scalar or subgroup
shuffle-based reduction.

### 3d. SYCL gap summary

| Item | CUDA | Vulkan | SYCL |
|---|---|---|---|
| Q4_K matmul: all kernel paths | MMVQ / MMQ / cuBLAS | Shader variants | DMMV / MMVQ / MMQ |
| Q4_K matmul: reorder optimisation | — | — | Yes (PR #13109) |
| Q8_0 matmul: reorder optimisation | — | — | **No** (issue #21517) |
| Prefill (large ne11) → BLAS fallback | Yes (ne11 ≥ 64 on NVIDIA) | Via large shader | Not confirmed |
| Flash attention: implemented | Yes | Yes | Yes (`fattn.cpp`) |
| Flash attention: Q4_K KV support | No | No | No |
| Flash attention: tensor-core kernel | Yes (MMA_F16, Turing+) | Yes (COOPMAT2) | **No** |
| Flash attention: VEC kernel | Yes | Via scalar path | Yes |
| Flash attention: TILE kernel | Yes | Yes | Yes |

### 3e. Real gaps in SYCL for Q4_K prefill

1. **No XMX/DPAS flash attention kernel.** For large prefill on Intel GPUs,
   SYCL flash attention uses a TILE kernel without hardware matrix-multiply
   instructions. CUDA and Vulkan both have dedicated tensor-core/coopmat
   paths for this case.

2. **Q4_K KV cache unsupported in flash attention.** Same design limitation
   as CUDA and Vulkan — not a SYCL-specific gap, but worth noting as a
   system-level constraint.

3. **Q8_0 lacks the reorder path.** Q4_K got MMVQ+reorder in PR #13109;
   Q8_0 remains on DMMV at ~4x lower throughput (issue #21517). This is a
   SYCL-specific regression relative to what Q4_K achieves.

4. **Prefill batch dispatch thresholds unconfirmed.** CUDA's exact `ne11`
   breakpoints (8 / 64) are visible in constants and functions. SYCL's
   equivalent in `ggml_sycl_mul_mat` could not be read from source due to
   file size; it may mirror CUDA's thresholds or differ.

---

## References

| File | What it confirms |
|---|---|
| `ggml/src/ggml-cuda/mmvq.cuh` | `MMVQ_MAX_BATCH_SIZE = 8` |
| `ggml/src/ggml-cuda/mmq.cuh` | `MMQ_DP4A_MAX_BATCH_SIZE = 64` |
| `ggml/src/ggml-cuda/mmvq.cu` | `ggml_cuda_should_use_mmvq` verbatim |
| `ggml/src/ggml-cuda/mmq.cu` | `ggml_cuda_should_use_mmq` verbatim |
| `ggml/src/ggml-cuda/fattn.cu` | Q4_K → `BEST_FATTN_KERNEL_NONE` (verbatim) |
| `ggml/src/ggml-sycl/fattn.cpp` | SYCL flash attn type switch + dispatch (verbatim) |
| `ggml/src/ggml-sycl/dmmv.cpp` | DMMV path exists, handles Q4_K |
| `ggml/src/ggml-sycl/mmvq.cpp` | MMVQ path, reorder variant |
| `ggml/src/ggml-sycl/presets.hpp` | SYCL constants (no MMVQ batch threshold) |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` | Q4_K absent from type list |
| PR #12858, #13109 | SYCL reorder MMVQ for Q4_0, Q4_K |
| Issue #21517 | Q4_K vs Q8_0 SYCL bandwidth measurements |
