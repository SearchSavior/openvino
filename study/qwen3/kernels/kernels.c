/*
 * C implementations of the two fused kernels, callable from Python via ctypes.
 *
 *   gdr_kernel     — gated delta rule linear-attention recurrence.
 *                    Mirrors GatedDeltaRule.evaluate() in fused_linear_attn.py.
 *
 *   conv1d_kernel  — causal depthwise conv1d with separate prev_state / current_input.
 *                    Mirrors FusedCausalConv1d.evaluate() in fused_conv1d.py.
 *
 * Build with:
 *   gcc -O3 -march=native -ffast-math -fopenmp -shared -fPIC kernels.c -o kernels.so -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif
#include <stdio.h>

/* Reset CPU affinity to all available CPUs. OV's plugin pins the calling
 * thread to a single CPU via TBB; without this, our omp fork inherits the
 * single-CPU affinity and serialises. _GNU_SOURCE must be defined before
 * <sched.h> to expose CPU_SET / cpu_set_t. */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>
#include <unistd.h>
static void qmm_unpin_self(void)
{
    cpu_set_t cs;
    CPU_ZERO(&cs);
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n <= 0) n = 4;
    for (long i = 0; i < n; ++i) CPU_SET(i, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);
}

/* ---------------------------------------------------------------------------
 * gdr_kernel — gated delta rule recurrence.
 *
 *   q, k       [B, H, T, D]   query / key
 *   v          [B, H, T, D]   value
 *   g          [B, H, T]      decay gate (raw; exp'd inside)
 *   beta       [B, H, T]      step size
 *   S          [B, H, D, D]   in/out — gated state, mutated in place
 *   out        [B, H, T, D]   per-token output
 *
 * Inner loop matches:
 *     g_t = exp(g[b,h,t]);  S *= g_t
 *     kv_mem[v] = sum_k S[k,v] * k_t[k]
 *     delta[v]  = (v_t[v] - kv_mem[v]) * beta_t
 *     S[k,v]   += k_t[k] * delta[v]
 *     out[v]    = sum_k S[k,v] * q_t[k]
 *
 * Parallelism: independent across the BH outer axis.
 * --------------------------------------------------------------------------- */
void gdr_kernel(
    const float * __restrict__ q,
    const float * __restrict__ k,
    const float * __restrict__ v,
    const float * __restrict__ g,
    const float * __restrict__ beta,
    float       * __restrict__ S,
    float       * __restrict__ out,
    int B, int H, int T, int D)
{
    const int BH = B * H;
    const size_t D2 = (size_t)D * (size_t)D;

    #pragma omp parallel for schedule(static)
    for (int bh = 0; bh < BH; ++bh) {
        const float *q_bh    = q    + (size_t)bh * T * D;
        const float *k_bh    = k    + (size_t)bh * T * D;
        const float *v_bh    = v    + (size_t)bh * T * D;
        const float *g_bh    = g    + (size_t)bh * T;
        const float *beta_bh = beta + (size_t)bh * T;
        float       *S_bh    = S    + (size_t)bh * D2;
        float       *out_bh  = out  + (size_t)bh * T * D;

        /* scratch — D=128 in practice, 512 is plenty of headroom on stack */
        float kv_mem[512];
        float delta[512];

        for (int t = 0; t < T; ++t) {
            const float *q_t = q_bh + (size_t)t * D;
            const float *k_t = k_bh + (size_t)t * D;
            const float *v_t = v_bh + (size_t)t * D;
            const float g_t  = expf(g_bh[t]);
            const float beta_t = beta_bh[t];

            /* S *= g_t */
            for (size_t i = 0; i < D2; ++i) S_bh[i] *= g_t;

            /* kv_mem[v] = sum_k S[k,v] * k_t[k] */
            for (int vv = 0; vv < D; ++vv) kv_mem[vv] = 0.0f;
            for (int kk = 0; kk < D; ++kk) {
                const float kv = k_t[kk];
                const float *Srow = S_bh + (size_t)kk * D;
                for (int vv = 0; vv < D; ++vv) {
                    kv_mem[vv] += Srow[vv] * kv;
                }
            }

            /* delta[v] = (v_t[v] - kv_mem[v]) * beta_t */
            for (int vv = 0; vv < D; ++vv) {
                delta[vv] = (v_t[vv] - kv_mem[vv]) * beta_t;
            }

            /* S[k,v] += k_t[k] * delta[v] */
            for (int kk = 0; kk < D; ++kk) {
                const float kv = k_t[kk];
                float *Srow = S_bh + (size_t)kk * D;
                for (int vv = 0; vv < D; ++vv) {
                    Srow[vv] += kv * delta[vv];
                }
            }

            /* out[v] = sum_k S[k,v] * q_t[k] */
            float *out_t = out_bh + (size_t)t * D;
            for (int vv = 0; vv < D; ++vv) out_t[vv] = 0.0f;
            for (int kk = 0; kk < D; ++kk) {
                const float qv = q_t[kk];
                const float *Srow = S_bh + (size_t)kk * D;
                for (int vv = 0; vv < D; ++vv) {
                    out_t[vv] += Srow[vv] * qv;
                }
            }
        }
    }
}


/* ---------------------------------------------------------------------------
 * conv1d_kernel — causal depthwise conv1d with explicit prev-state.
 *
 *   prev       [B, C, KS]              prior state (KS = kernel_size - 1 + ...)
 *   cur        [B, C, T]               current input
 *   w          [C, K]                  depthwise weights
 *   out        [B, C, KS + T - K + 1]  same shape as GroupConvolution output
 *   new_state  [B, C, KS]              last KS positions of (prev ++ cur)
 *
 * Parallelism: independent across BC outer axis.
 * --------------------------------------------------------------------------- */
void conv1d_kernel(
    const float * __restrict__ prev,
    const float * __restrict__ cur,
    const float * __restrict__ w,
    float       * __restrict__ out,
    float       * __restrict__ new_state,
    int B, int C, int KS, int T, int K)
{
    const int out_len = KS + T - K + 1;
    const int BC = B * C;

    #pragma omp parallel for schedule(static)
    for (int bc = 0; bc < BC; ++bc) {
        const int c = bc % C;
        const float *prev_bc = prev + (size_t)bc * KS;
        const float *cur_bc  = cur  + (size_t)bc * T;
        const float *wc      = w    + (size_t)c  * K;
        float *out_bc        = out  + (size_t)bc * out_len;
        float *ns_bc         = new_state + (size_t)bc * KS;

        memset(out_bc, 0, sizeof(float) * (size_t)out_len);

        for (int kk = 0; kk < K; ++kk) {
            int cutover = KS - kk;
            if (cutover < 0) cutover = 0;
            if (cutover > out_len) cutover = out_len;
            const float wk = wc[kk];

            for (int i = 0; i < cutover; ++i) {
                out_bc[i] += wk * prev_bc[kk + i];
            }
            const int rem = out_len - cutover;
            for (int i = 0; i < rem; ++i) {
                out_bc[cutover + i] += wk * cur_bc[i];
            }
        }

        /* new_state = last KS positions of (prev ++ cur) */
        if (T >= KS) {
            memcpy(ns_bc, cur_bc + (T - KS), sizeof(float) * (size_t)KS);
        } else {
            memcpy(ns_bc, prev_bc + T, sizeof(float) * (size_t)(KS - T));
            memcpy(ns_bc + (KS - T), cur_bc, sizeof(float) * (size_t)T);
        }
    }
}


/* fp16 (binary16) -> fp32. Avoids depending on _Float16 / __fp16 by reading
 * the raw bits. Matches IEEE 754 half precision. */
static inline float f16_to_f32(unsigned short h)
{
    unsigned int sign = (unsigned int)(h & 0x8000) << 16;
    unsigned int exp  = (h >> 10) & 0x1f;
    unsigned int mant = h & 0x3ff;
    unsigned int f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            /* subnormal */
            int e = -1;
            do { e++; mant <<= 1; } while ((mant & 0x400) == 0);
            f = sign | ((unsigned int)(e + 127 - 14) << 23) | ((mant & 0x3ff) << 13);
        }
    } else if (exp == 0x1f) {
        f = sign | 0x7f800000 | (mant << 13);
    } else {
        f = sign | ((unsigned int)(exp + 127 - 15) << 23) | (mant << 13);
    }
    union { unsigned int u; float f; } u; u.u = f; return u.f;
}

void qmm_kernel(
    const float * __restrict__ act,
    const unsigned char * __restrict__ u8,
    const unsigned short * __restrict__ scale,
    const unsigned char * __restrict__ zp,
    float       * __restrict__ out,
    int M, int N, int K)
{
    /* OV's plugin runs Op.evaluate() inside a TBB worker pinned to one CPU.
     * Reset affinity so our omp threads can spread across the 4 cores. */
    qmm_unpin_self();
#ifdef _OPENMP
    if (omp_get_max_threads() < 4) omp_set_num_threads(4);
#endif

    /* Precompute sum_k act[m, k] for each m. SIMD-vectorisable directly. */
    float *act_sum = (float *)malloc(sizeof(float) * (size_t)M);
    for (int m = 0; m < M; ++m) {
        const float *a_m = act + (size_t)m * K;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += a_m[k];
        act_sum[m] = s;
    }

#if defined(__AVX512F__)
    /* Per-thread scratch for the dequantised weight row.
     * Max K in this model is 3584 (MLP down_proj); 4096 leaves headroom. */
    #define QMM_KMAX 4096
    #pragma omp parallel
    {
        float w_f[QMM_KMAX] __attribute__((aligned(64)));

        #pragma omp for schedule(static)
        for (int n = 0; n < N; ++n) {
            const unsigned char *w_n = u8 + (size_t)n * K;
            const float sc_n = f16_to_f32(scale[n]);
            const float zp_n = (float)zp[n];

            /* Dequant w_n into stack scratch (u8 -> f32, no zp/scale here;
             * those fold into the factored math below). */
            int k = 0;
            for (; k + 16 <= K; k += 16) {
                __m128i u = _mm_loadu_si128((const __m128i *)(w_n + k));
                __m512i i = _mm512_cvtepu8_epi32(u);
                __m512  f = _mm512_cvtepi32_ps(i);
                _mm512_store_ps(w_f + k, f);
            }
            for (; k < K; ++k) w_f[k] = (float)w_n[k];

            /* M dot products against the same dequanted row. */
            for (int m = 0; m < M; ++m) {
                const float *a_m = act + (size_t)m * K;
                __m512 acc = _mm512_setzero_ps();
                int kk = 0;
                for (; kk + 16 <= K; kk += 16) {
                    __m512 av = _mm512_loadu_ps(a_m + kk);
                    __m512 wv = _mm512_load_ps(w_f + kk);
                    acc = _mm512_fmadd_ps(av, wv, acc);
                }
                float dot = _mm512_reduce_add_ps(acc);
                for (; kk < K; ++kk) dot += a_m[kk] * w_f[kk];

                out[(size_t)m * N + n] = sc_n * (dot - zp_n * act_sum[m]);
            }
        }
    }
#else
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const unsigned char *w_n = u8 + (size_t)n * K;
        const float sc_n = f16_to_f32(scale[n]);
        const float zp_n = (float)zp[n];
        for (int m = 0; m < M; ++m) {
            const float *a_m = act + (size_t)m * K;
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) dot += a_m[k] * (float)w_n[k];
            out[(size_t)m * N + n] = sc_n * (dot - zp_n * act_sum[m]);
        }
    }
#endif
    free(act_sum);
}


/* ---------------------------------------------------------------------------
 * qmm_kernel_vnni — same external semantics as qmm_kernel, but uses VNNI
 * `vpdpbusd` for the inner u8 * i8 -> i32 dot product.
 *
 * Per call we quantise each activation row to signed i8 (per-row symmetric):
 *     s_m       = max(|act[m, :]|) / 127
 *     act_i8[m, k] = round(act[m, k] / s_m)
 *     act_sum_i32[m] = sum_k act_i8[m, k]
 *
 * Then for each (m, n):
 *     i32_dot = sum_k act_i8[m, k] * u8[n, k]          # via vpdpbusd
 *     y[m, n] = s_m * scale_w[n] * (i32_dot - zp[n] * act_sum_i32[m])
 *
 * Storage during the call: act_i8 [M, K] bytes + small per-row scalars.
 * The (B, B) bf16 weight cache is still NOT materialised.
 * --------------------------------------------------------------------------- */
#if defined(__AVX512VNNI__)
void qmm_kernel_vnni(
    const float * __restrict__ act,
    const unsigned char * __restrict__ u8,
    const unsigned short * __restrict__ scale,
    const unsigned char * __restrict__ zp,
    float       * __restrict__ out,
    int M, int N, int K)
{
    qmm_unpin_self();
#ifdef _OPENMP
    if (omp_get_max_threads() < 4) omp_set_num_threads(4);
#endif

    /* Per-row activation quantisation. */
    signed char *act_i8 = (signed char *)aligned_alloc(64, (size_t)M * (size_t)K);
    float *act_scale = (float *)malloc(sizeof(float) * (size_t)M);
    int   *act_sum_i8 = (int   *)malloc(sizeof(int)   * (size_t)M);

    #pragma omp parallel for schedule(static)
    for (int m = 0; m < M; ++m) {
        const float *a_m = act + (size_t)m * K;
        /* max(|.|) via AVX-512 */
        __m512 vmax = _mm512_setzero_ps();
        const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
        int k = 0;
        for (; k + 16 <= K; k += 16) {
            __m512 v = _mm512_loadu_ps(a_m + k);
            v = _mm512_and_ps(v, abs_mask);
            vmax = _mm512_max_ps(vmax, v);
        }
        float max_abs = _mm512_reduce_max_ps(vmax);
        for (; k < K; ++k) {
            float ax = a_m[k]; if (ax < 0) ax = -ax;
            if (ax > max_abs) max_abs = ax;
        }
        float s = (max_abs > 0.0f) ? max_abs / 127.0f : 1.0f;
        float inv = (max_abs > 0.0f) ? 127.0f / max_abs : 0.0f;
        act_scale[m] = s;

        signed char *row = act_i8 + (size_t)m * K;
        int sum = 0;
        __m512i sum_v = _mm512_setzero_si512();
        const __m512 inv_v = _mm512_set1_ps(inv);
        k = 0;
        for (; k + 16 <= K; k += 16) {
            __m512 v = _mm512_loadu_ps(a_m + k);
            v = _mm512_mul_ps(v, inv_v);
            __m512i q = _mm512_cvtps_epi32(v);             /* rounds-to-nearest */
            /* clamp [-128, 127] */
            q = _mm512_max_epi32(q, _mm512_set1_epi32(-128));
            q = _mm512_min_epi32(q, _mm512_set1_epi32(127));
            /* sum and pack to i8 */
            sum_v = _mm512_add_epi32(sum_v, q);
            __m128i q8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128((__m128i *)(row + k), q8);
        }
        sum += _mm512_reduce_add_epi32(sum_v);
        for (; k < K; ++k) {
            int q = (int)lrintf(a_m[k] * inv);
            if (q > 127) q = 127; if (q < -128) q = -128;
            row[k] = (signed char)q;
            sum += q;
        }
        act_sum_i8[m] = sum;
    }

    /* Matmul using vpdpbusd: u8 * i8 -> i32 accumulator. */
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const unsigned char *w_n = u8 + (size_t)n * K;
        const float sc_n = f16_to_f32(scale[n]);
        const float zp_n = (float)zp[n];
        for (int m = 0; m < M; ++m) {
            const signed char *a_i8 = act_i8 + (size_t)m * K;
            __m512i acc = _mm512_setzero_si512();
            int k = 0;
            for (; k + 64 <= K; k += 64) {
                __m512i zw = _mm512_loadu_si512((const __m512i *)(w_n + k));
                __m512i za = _mm512_loadu_si512((const __m512i *)(a_i8 + k));
                acc = _mm512_dpbusd_epi32(acc, zw, za);
            }
            int dot = _mm512_reduce_add_epi32(acc);
            for (; k < K; ++k) dot += (int)w_n[k] * (int)a_i8[k];

            float dot_f = (float)dot - zp_n * (float)act_sum_i8[m];
            out[(size_t)m * N + n] = act_scale[m] * sc_n * dot_f;
        }
    }

    free(act_i8);
    free(act_scale);
    free(act_sum_i8);
}
#endif  /* __AVX512VNNI__ */


/* ---------------------------------------------------------------------------
 * qkv_kernel - int8 KV cache update.
 *
 *   prev_data    [B, H, T_prev, D]   i8     existing quantised state
 *   prev_scale   [B, H, T_prev]      f32    existing per-token scales
 *   new_kv       [B, H, N,     D]    f32    new K (or V) for this chunk
 *
 *   new_data     [B, H, T_full, D]   i8     concatenated quantised state out
 *   new_scale    [B, H, T_full]      f32    concatenated scales out
 *   full_f32     [B, H, T_full, D]   f32    dequantised KV for SDPA
 *
 *   T_full = T_prev + N
 *
 * Per-token symmetric int8 quant on the new tokens:
 *     scale_t = max|kv[..,t,:]| / 127
 *     q_t[d]  = clip(round(kv[..,t,d] / scale_t), -128, 127)
 * Then concatenate (prev, new) and dequantise the whole thing for SDPA.
 * --------------------------------------------------------------------------- */
void qkv_kernel(
    const signed char  * __restrict__ prev_data,
    const float        * __restrict__ prev_scale,
    const float        * __restrict__ new_kv,
    signed char        * __restrict__ new_data,
    float              * __restrict__ new_scale,
    float              * __restrict__ full_f32,
    int B, int H, int T_prev, int N, int D)
{
    qmm_unpin_self();
#ifdef _OPENMP
    if (omp_get_max_threads() < 4) omp_set_num_threads(4);
#endif
    const int T_full = T_prev + N;
    const size_t BH   = (size_t)B * (size_t)H;

    /* Copy prev_data + prev_scale into the head of the new buffers. */
    for (size_t bh = 0; bh < BH; ++bh) {
        memcpy(new_data  + bh * T_full * D,
               prev_data + bh * T_prev * D, (size_t)T_prev * D);
        memcpy(new_scale + bh * T_full,
               prev_scale + bh * T_prev,    (size_t)T_prev * sizeof(float));
    }

    /* Quantise the N new tokens, appending to the tail of (new_data, new_scale). */
#if defined(__AVX512F__)
    const __m512 abs_mask = _mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff));
#endif
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < BH * (size_t)N; ++idx) {
        const size_t bh = idx / (size_t)N;
        const int    t  = (int)(idx % (size_t)N);
        const float *kv_t = new_kv + bh * (size_t)N * D + (size_t)t * D;

        float mx = 0.0f;
#if defined(__AVX512F__)
        __m512 vmax = _mm512_setzero_ps();
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            __m512 v = _mm512_loadu_ps(kv_t + d);
            v = _mm512_and_ps(v, abs_mask);
            vmax = _mm512_max_ps(vmax, v);
        }
        mx = _mm512_reduce_max_ps(vmax);
        for (; d < D; ++d) {
            float a = fabsf(kv_t[d]); if (a > mx) mx = a;
        }
#else
        for (int d = 0; d < D; ++d) {
            float a = fabsf(kv_t[d]); if (a > mx) mx = a;
        }
#endif
        float scale = mx > 1e-12f ? mx / 127.0f : 1e-12f;
        float inv   = mx > 1e-12f ? 127.0f / mx : 0.0f;

        signed char *out_t = new_data + bh * (size_t)T_full * D + (size_t)(T_prev + t) * D;
#if defined(__AVX512F__)
        __m512 inv_v = _mm512_set1_ps(inv);
        const __m512i lo = _mm512_set1_epi32(-128);
        const __m512i hi = _mm512_set1_epi32( 127);
        int dd = 0;
        for (; dd + 16 <= D; dd += 16) {
            __m512 v = _mm512_loadu_ps(kv_t + dd);
            v = _mm512_mul_ps(v, inv_v);
            __m512i q = _mm512_cvtps_epi32(v);
            q = _mm512_max_epi32(q, lo);
            q = _mm512_min_epi32(q, hi);
            __m128i q8 = _mm512_cvtepi32_epi8(q);
            _mm_storeu_si128((__m128i *)(out_t + dd), q8);
        }
        for (; dd < D; ++dd) {
            int q = (int)lrintf(kv_t[dd] * inv);
            if (q > 127) q = 127; if (q < -128) q = -128;
            out_t[dd] = (signed char)q;
        }
#else
        for (int d = 0; d < D; ++d) {
            int q = (int)lrintf(kv_t[d] * inv);
            if (q > 127) q = 127; if (q < -128) q = -128;
            out_t[d] = (signed char)q;
        }
#endif
        new_scale[bh * (size_t)T_full + (size_t)(T_prev + t)] = scale;
    }

    /* Dequant the full state to f32 for SDPA. Pass full_f32 == NULL to skip
     * this step entirely (e.g. when the consumer is an int8-aware SDPA). */
    if (full_f32 == NULL) return;
    #pragma omp parallel for schedule(static)
    for (size_t idx = 0; idx < BH * (size_t)T_full; ++idx) {
        const size_t bh = idx / (size_t)T_full;
        const int    t  = (int)(idx % (size_t)T_full);
        const float  sc = new_scale[bh * (size_t)T_full + (size_t)t];
        const signed char *src = new_data + bh * (size_t)T_full * D + (size_t)t * D;
        float             *dst = full_f32 + bh * (size_t)T_full * D + (size_t)t * D;
#if defined(__AVX512F__)
        __m512 sc_v = _mm512_set1_ps(sc);
        int d = 0;
        for (; d + 16 <= D; d += 16) {
            __m128i i8 = _mm_loadu_si128((const __m128i *)(src + d));
            __m512i i32 = _mm512_cvtepi8_epi32(i8);
            __m512  f   = _mm512_cvtepi32_ps(i32);
            f = _mm512_mul_ps(f, sc_v);
            _mm512_storeu_ps(dst + d, f);
        }
        for (; d < D; ++d) dst[d] = (float)src[d] * sc;
#else
        for (int d = 0; d < D; ++d) dst[d] = (float)src[d] * sc;
#endif
    }
}


/* ---------------------------------------------------------------------------
 * int8_sdpa_kernel - scaled dot-product attention reading i8 K/V directly.
 *
 * Standard SDPA: out = softmax(Q K^T * scale + mask) V.
 * K/V live as (i8 data, f32 per-token scale). Dequant is fused into the dot
 * and weighted-sum loops; no full f32 K or V buffer is ever materialised.
 *
 * GQA: K/V have H_kv heads, Q has H_q = gqa * H_kv heads. Each q head h
 * indexes its KV head via h / gqa.
 *
 * Parallelism: across (B, H_q, T_q). Each thread allocates a per-row scores
 * buffer of T_full floats. T_max=4096 covers our model.
 * --------------------------------------------------------------------------- */
#define SDPA_TMAX 4096

static inline float i8_f32_dot(const float *q, const signed char *k, int D)
{
#if defined(__AVX512F__)
    __m512 acc = _mm512_setzero_ps();
    int d = 0;
    for (; d + 16 <= D; d += 16) {
        __m128i  k8  = _mm_loadu_si128((const __m128i *)(k + d));
        __m512i  ki  = _mm512_cvtepi8_epi32(k8);
        __m512   kf  = _mm512_cvtepi32_ps(ki);
        __m512   qv  = _mm512_loadu_ps(q + d);
        acc = _mm512_fmadd_ps(qv, kf, acc);
    }
    float s = _mm512_reduce_add_ps(acc);
    for (; d < D; ++d) s += q[d] * (float)k[d];
    return s;
#else
    float s = 0.0f;
    for (int d = 0; d < D; ++d) s += q[d] * (float)k[d];
    return s;
#endif
}

void int8_sdpa_kernel(
    const float        * __restrict__ q,
    const signed char  * __restrict__ k_data,
    const float        * __restrict__ k_scale,
    const signed char  * __restrict__ v_data,
    const float        * __restrict__ v_scale,
    const float        * __restrict__ mask,
    float                              scale,
    float              * __restrict__ out,
    int B, int H_q, int H_kv, int T_q, int T_full, int D)
{
    qmm_unpin_self();
#ifdef _OPENMP
    if (omp_get_max_threads() < 4) omp_set_num_threads(4);
#endif
    const int gqa = H_q / H_kv;

    #pragma omp parallel
    {
        float scores[SDPA_TMAX];

        #pragma omp for collapse(2) schedule(static)
        for (int b = 0; b < B; ++b) {
            for (int h = 0; h < H_q; ++h) {
                const int h_kv = h / gqa;
                const signed char *K  = k_data  + ((size_t)b*H_kv + h_kv) * (size_t)T_full * D;
                const float       *Ks = k_scale + ((size_t)b*H_kv + h_kv) * (size_t)T_full;
                const signed char *V  = v_data  + ((size_t)b*H_kv + h_kv) * (size_t)T_full * D;
                const float       *Vs = v_scale + ((size_t)b*H_kv + h_kv) * (size_t)T_full;

                for (int tq = 0; tq < T_q; ++tq) {
                    const float *q_row = q + (((size_t)b*H_q + h)*T_q + tq) * D;
                    float       *o_row = out + (((size_t)b*H_q + h)*T_q + tq) * D;
                    const float *m_row = mask
                        ? mask + ((size_t)b * T_q + tq) * (size_t)T_full
                        : NULL;

                    /* Scores: scale * (q . dequant(K[k])) + mask. */
                    float max_s = -INFINITY;
                    for (int k = 0; k < T_full; ++k) {
                        float s = i8_f32_dot(q_row, K + (size_t)k * D, D);
                        s = s * Ks[k] * scale;
                        if (m_row) s += m_row[k];
                        scores[k] = s;
                        if (s > max_s) max_s = s;
                    }

                    /* Softmax. */
                    float sum_e = 0.0f;
                    for (int k = 0; k < T_full; ++k) {
                        scores[k] = expf(scores[k] - max_s);
                        sum_e += scores[k];
                    }
                    const float inv_sum = 1.0f / sum_e;

                    /* out = sum_k softmax[k] * dequant(V[k]). */
                    for (int d = 0; d < D; ++d) o_row[d] = 0.0f;
#if defined(__AVX512F__)
                    for (int k = 0; k < T_full; ++k) {
                        const float w = scores[k] * inv_sum * Vs[k];
                        const signed char *v_row = V + (size_t)k * D;
                        __m512 wv = _mm512_set1_ps(w);
                        int d = 0;
                        for (; d + 16 <= D; d += 16) {
                            __m128i v8  = _mm_loadu_si128((const __m128i *)(v_row + d));
                            __m512i vi  = _mm512_cvtepi8_epi32(v8);
                            __m512  vf  = _mm512_cvtepi32_ps(vi);
                            __m512  ov  = _mm512_loadu_ps(o_row + d);
                            ov = _mm512_fmadd_ps(wv, vf, ov);
                            _mm512_storeu_ps(o_row + d, ov);
                        }
                        for (; d < D; ++d) o_row[d] += w * (float)v_row[d];
                    }
#else
                    for (int k = 0; k < T_full; ++k) {
                        const float w = scores[k] * inv_sum * Vs[k];
                        const signed char *v_row = V + (size_t)k * D;
                        for (int d = 0; d < D; ++d) o_row[d] += w * (float)v_row[d];
                    }
#endif
                }
            }
        }
    }
}
