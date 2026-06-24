# llama.cpp – Q4_K Prefill Handling: CUDA vs Vulkan vs SYCL

Analysis based on llama.cpp `master` branch (June 2026).

---

## Background

**Prefill** is the phase where LLM inference processes the initial prompt.
Key characteristics:
- `ne11` (number of query tokens / batch rows) is large — hundreds to thousands
- All attention heads process the full prompt simultaneously
- Memory and compute efficiency are dominated by two operations:
  1. **Weight matmul**: projection layers (Q/K/V/FFN) with q4_k-quantized weights
  2. **Flash attention**: `O = softmax(Q × Kᵀ / √d) × V`

**Q4_K** is llama.cpp's mixed 4/6-bit "k-quant" format: 256-element blocks with
per-block scale/min stored in 6 bits, using the `block_q4_K` struct.

---

## 1. CUDA Backend (reference implementation)

### 1a. Matmul path — q4_k model weights × dense input

Controlled by `ggml_cuda_should_use_mmq()` (`ggml/src/ggml-cuda/mmq.cu`):

| Architecture | Q4_K MMQ threshold (`ne11`) | Fallback |
|---|---|---|
| NVIDIA (DP4A) | `ne11 < 64` (`MMQ_DP4A_MAX_BATCH_SIZE`) | dequantize → cuBLAS |
| AMD MFMA (CDNA) | `ne11 ≤ 256` | dequantize → BLAS |
| AMD WMMA RDNA3 | `ne11 ≤ 256` | dequantize → BLAS |
| AMD WMMA RDNA4 | always MMQ | — |

**Large prefill consequence on NVIDIA**: at `ne11 ≥ 64`, q4_k weights are
dequantized to FP16 and sent through cuBLAS. The quantized representation is
abandoned for the most compute-heavy phase of inference.

The MMQ kernel itself uses a Q8_1 intermediate: the dense input is quantized
to Q8_1 on-the-fly (`quantize_mmq_q8_1_cuda`), then the q4_k × Q8_1 kernel
fires with DP4A integer dot products.

### 1b. Flash attention path — q4_k KV cache

Dispatch: `ggml_cuda_get_best_fattn_kernel()` (`ggml/src/ggml-cuda/fattn.cu`).

```c
// Accepted K/V types
case GGML_TYPE_F32:  case GGML_TYPE_F16:
case GGML_TYPE_Q4_0: case GGML_TYPE_Q8_0: case GGML_TYPE_BF16:
    break;
// With GGML_CUDA_FA_ALL_QUANTS:
case GGML_TYPE_Q4_1: case GGML_TYPE_Q5_0: case GGML_TYPE_Q5_1:
    break;
default:
    return BEST_FATTN_KERNEL_NONE;   // ← Q4_K lands here
```

`ggml_cuda_flash_attn_ext_supported()` returns `false` for Q4_K.
`ggml_cuda_flash_attn_ext()` calls `GGML_ABORT("fatal error")` if invoked
directly with an unsupported type — so callers must gate on `supported()`.

**Result**: Q4_K KV cache is **incompatible** with CUDA flash attention.
llama.cpp's graph planner will not schedule flash attention when KV cache
type is Q4_K on CUDA; it falls back to F16 KV cache or a slower non-tiled
attention path.

### 1c. Flash attention kernel selection for prefill

When flash attention *is* usable (F16/Q4_0/Q8_0 KV), the kernel hierarchy is:

```
ne11 (query tokens)     Kernel selected
─────────────────────   ─────────────────────────────────────────
1–2                     BEST_FATTN_KERNEL_VEC (decode-optimised)
≤ 16 (Volta)            BEST_FATTN_KERNEL_TILE
> 2 with MMA support    BEST_FATTN_KERNEL_MMA_F16  ← prefill path
```

The MMA (tensor-core) flash attention kernel (`fattn-mma-f16.cuh`) is the
preferred prefill kernel on Turing+. It is fully tile-based with GQA support
and handles variable head dimensions (64–576).

---

## 2. Vulkan Backend

### 2a. Matmul path — q4_k weights

Vulkan compiles dedicated GLSL compute shaders per quantization type.
For Q4_K, it uses integer dot product (`GGML_TYPE_Q4_K` entry in
`pipeline_dequant_mul_mat_mat`). Three pipeline size variants are compiled:

| Pipeline | Activation threshold |
|---|---|
| `mul_mat_l` | large batch |
| `mul_mat_m` | medium batch |
| `mul_mat_s` | small batch |

The MMQ (integer DP) path uses the `mul_mat_l_int[]` family when integer
dot products are available. No on-the-fly dequantization to FP16 is required
for matmul; the shader decodes Q4_K blocks inline.

The vector-matmul threshold is `mul_mat_vec_max_cols = 8`, below which a
matrix–vector kernel fires (decode path).

### 2b. Flash attention path — q4_k KV cache

Three code paths: `FA_SCALAR`, `FA_COOPMAT1`, `FA_COOPMAT2`.

The cm2 cooperative-matrix shader (`vulkan-shaders/flash_attn_cm2.comp`)
decodes K/V inline via `faDecodeK`/`faDecodeV`. Supported types:

```
FA_TYPE_F32, FA_TYPE_F16, FA_TYPE_BF16
FA_TYPE_Q4_0, FA_TYPE_Q4_1
FA_TYPE_Q5_0, FA_TYPE_Q5_1
FA_TYPE_Q8_0
FA_TYPE_Q1_0
```

**Q4_K is absent.** The scalar and coopmat1 shaders
(`flash_attn_dequant.glsl`) cover the same set and also exclude Q4_K.

**Result**: Q4_K KV cache is unsupported in Vulkan flash attention — the same
limitation as CUDA, but without the `GGML_ABORT` guard. Vulkan's
`ggml_vk_flash_attn_ext` presumably falls back to a non-flash or dequantized
path when Q4_K is detected (exact fallback is in the pipeline selection map).

---

## 3. SYCL Backend — what is missing

### 3a. Matmul path — q4_k weights

`ggml_mul_mat_q4_K_q8_1_sycl()` (`ggml/src/ggml-sycl/mmq.cpp`) exists and
dispatches per Intel/AMD generation:

```cpp
if (compute_capability >= VER_GEN13) {       // RDNA2 / Arc Alchemist+
    mmq_x = 64; mmq_y = 128; nwarps = 8;
} else if (compute_capability >= VER_GEN9) { // Ampere-equivalent
    mmq_x = 64; mmq_y = 128; nwarps = 8;
} else {                                      // Pascal-equivalent
    mmq_x = 64; mmq_y = 64;  nwarps = 8;
}
```

**Gap 1 — missing prefill/large-batch threshold logic.**
CUDA has `ggml_cuda_should_use_mmq()` which switches from MMQ to
dequantize+GEMM once `ne11 ≥ 64` (NVIDIA). SYCL has no equivalent
function visible in the codebase. The backend appears to always invoke the
MMQ kernel regardless of batch size. For large prefill (ne11 ≫ 64) this
means:
- Running through an integer tile kernel sized for decode / small batch
- Missing the opportunity to use oneMKL/SYCL GEMM on the dequantized data
- Potential register pressure issues at large ne11 (the MMQ code itself
  comments on this)

### 3b. Flash attention path — q4_k KV cache

**Gap 2 — flash attention is not production-ready.**

```cpp
#ifdef SYCL_FLASH_ATTN
    g_ggml_sycl_enable_flash_attention =
        ggml_sycl_get_env("GGML_SYCL_ENABLE_FLASH_ATTN", 1);
#else
    g_ggml_sycl_enable_flash_attention = 0;   // disabled by default
#endif
```

- The compile flag `SYCL_FLASH_ATTN` is not set in standard builds.
- The implementation file (`ggml-sycl/flash_attn.cpp` / `.hpp`) was not found
  at the expected path — flash attention for SYCL is either absent or
  in-progress.
- Even when the flag is set, there is no evidence of Q4_K KV support; CUDA
  and Vulkan both reject Q4_K in flash attention.

**Gap 3 — no MMA/XMX tensor instruction path for prefill.**
CUDA uses a dedicated `BEST_FATTN_KERNEL_MMA_F16` kernel that leverages
NVIDIA tensor cores for prefill. Vulkan has `FA_COOPMAT2` with NV
cooperative matrix 2 extensions. SYCL has no analogous kernel using Intel
XMX matrix-multiply instructions for flash attention.

### 3c. Summary of SYCL gaps

| Area | CUDA | Vulkan | SYCL |
|---|---|---|---|
| Q4_K matmul kernel exists | Yes | Yes (shader) | Yes |
| Prefill threshold → dequantize+GEMM | Yes (ne11≥64) | Yes (size variants) | **No** |
| Flash attention implemented | Yes (MMA) | Yes (coopmat) | **Partial/flag-gated** |
| Q4_K KV in flash attention | No (by design) | No (by design) | No (not implemented) |
| Tensor-core/XMX flash attention | Yes (MMA F16) | Yes (coopmat2) | **No** |
| Large-prefill dequantize fallback | Yes | Yes | **Missing** |

---

## 4. Implications for OpenVINO Integration

If OpenVINO is to serve as an alternative SYCL/GPU backend for llama.cpp-style
workloads, the following should be addressed for correct Q4_K prefill:

1. **Matmul dispatch by batch size**: implement a decision function analogous
   to `ggml_cuda_should_use_mmq` that switches to a dequantize → oneDNN GEMM
   path for large ne11. The threshold likely sits around 64–128 depending on
   the Intel GPU generation.

2. **Flash attention for prefill**: a SYCL flash attention kernel leveraging
   Intel XMX (DPAS) instructions is the highest-impact missing component.
   It should handle F16/BF16/Q8_0 KV cache at minimum.

3. **Q4_K KV cache**: not supported in any GPU backend's flash attention path.
   This is a known limitation upstream; for now, F16 or Q8_0 KV cache should
   be recommended for SYCL prefill workloads.

4. **Register pressure in large MMQ tiles**: the SYCL MMQ kernel comments flag
   potential register spilling at large tile sizes. Profiling on Arc/Flex GPUs
   at ne11 ≥ 512 is recommended before production use.

---

## References

| File | Purpose |
|---|---|
| `ggml/src/ggml-cuda/mmq.cu` | CUDA MMQ dispatch, `ggml_cuda_should_use_mmq` |
| `ggml/src/ggml-cuda/fattn.cu` | CUDA flash attention dispatch |
| `ggml/src/ggml-cuda/fattn-mma-f16.cuh` | CUDA tensor-core flash attention |
| `ggml/src/ggml-sycl/mmq.cpp` | SYCL MMQ kernels (Q4_K) |
| `ggml/src/ggml-sycl/ggml-sycl.cpp` | SYCL backend entry, flash attention flag |
| `ggml/src/ggml-vulkan/ggml-vulkan.cpp` | Vulkan pipeline creation |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_cm2.comp` | Vulkan coopmat flash attn |
| `ggml/src/ggml-vulkan/vulkan-shaders/flash_attn_dequant.glsl` | Vulkan flash attn type list |
