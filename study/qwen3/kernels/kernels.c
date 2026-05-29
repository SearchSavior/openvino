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

#ifdef _OPENMP
#include <omp.h>
#endif

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


/* ---------------------------------------------------------------------------
 * qmm_kernel — dequant-on-the-fly int8 matmul, transpose_b semantics.
 *
 *   act        [M, K]            fp32
 *   u8         [N, K]            uint8
 *   scale      [N]               fp16 (cast inside)
 *   zp         [N]               uint8
 *   out        [M, N]            fp32
 *
 *   y[m, n] = scale[n] * (sum_k act[m, k] * (float)u8[n, k] - zp[n] * sum_k act[m, k])
 *
 * No bf16/f32 weight buffer is ever materialised. Per output row we read
 * K bytes of u8 directly from the IR Constant (file-backed) plus 4 bytes
 * scale + 1 byte zp.
 *
 * Parallelism: outer loop over N (output cols). Inner loop is K reductions.
 * For each m we precompute sum_k act[m, k].
 * --------------------------------------------------------------------------- */

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
    /* Precompute sum_k act[m, k] for each m. */
    /* For typical M up to a few thousand this fits on stack. Use heap if big. */
    float *act_sum = (float *)malloc(sizeof(float) * (size_t)M);
    for (int m = 0; m < M; ++m) {
        const float *a_m = act + (size_t)m * K;
        float s = 0.0f;
        for (int k = 0; k < K; ++k) s += a_m[k];
        act_sum[m] = s;
    }

    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; ++n) {
        const unsigned char *w_n = u8 + (size_t)n * K;
        const float sc_n = f16_to_f32(scale[n]);
        const float zp_n = (float)zp[n];
        for (int m = 0; m < M; ++m) {
            const float *a_m = act + (size_t)m * K;
            float dot = 0.0f;
            for (int k = 0; k < K; ++k) {
                dot += a_m[k] * (float)w_n[k];
            }
            out[(size_t)m * N + n] = sc_n * (dot - zp_n * act_sum[m]);
        }
    }
    free(act_sum);
}
