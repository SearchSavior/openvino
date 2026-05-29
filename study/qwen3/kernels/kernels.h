/* Forward declarations for kernels.c (built alongside the .cpp ops). */
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void gdr_kernel(const float *q, const float *k, const float *v,
                const float *g, const float *beta,
                float *S, float *out,
                int B, int H, int T, int D);

void conv1d_kernel(const float *prev, const float *cur, const float *w,
                   float *out, float *new_state,
                   int B, int C, int KS, int T, int K);

void qmm_kernel(const float *act,
                const unsigned char *u8,
                const unsigned short *scale,
                const unsigned char *zp,
                float *out,
                int M, int N, int K);

/* AVX-512 VNNI variant: per-row quantises act to i8 and uses vpdpbusd. */
void qmm_kernel_vnni(const float *act,
                     const unsigned char *u8,
                     const unsigned short *scale,
                     const unsigned char *zp,
                     float *out,
                     int M, int N, int K);

/* int8 KV cache update: dequant prev + quantise new + dequant full for SDPA. */
void qkv_kernel(const signed char *prev_data,
                const float       *prev_scale,
                const float       *new_kv,
                signed char       *new_data,
                float             *new_scale,
                float             *full_f32,
                int B, int H, int T_prev, int N, int D);

/* Fused SDPA reading i8 K/V directly (no f32 KV materialisation).
 *
 *   q         [B, H_q,  T_q,    D]    fp32  (post-RoPE, after GQA broadcast?)
 *   k_data    [B, H_kv, T_full, D]    i8
 *   k_scale   [B, H_kv, T_full]       fp32
 *   v_data    [B, H_kv, T_full, D]    i8
 *   v_scale   [B, H_kv, T_full]       fp32
 *   mask      [B, 1, T_q, T_full]     fp32 or NULL
 *   scale     1/sqrt(D)               fp32
 *
 *   out       [B, H_q, T_q, D]        fp32
 *
 * gqa_factor = H_q / H_kv. K/V are indexed by h_kv = h_q / gqa_factor.
 * Standard scaled dot-product attention:
 *   scores[t_q, t_k] = (Q[t_q, :] . dequant(K[h_kv, t_k, :])) * scale + mask[t_q, t_k]
 *   weights = softmax(scores, dim=-1)
 *   out[t_q, :] = sum_t_k weights[t_q, t_k] * dequant(V[h_kv, t_k, :])
 *
 * Dequant is fused into the dot/accumulate loops -- no full f32 K or V buffer
 * is ever materialised.
 */
void int8_sdpa_kernel(const float       *q,
                      const signed char *k_data,
                      const float       *k_scale,
                      const signed char *v_data,
                      const float       *v_scale,
                      const float       *mask,
                      float              scale,
                      float             *out,
                      int B, int H_q, int H_kv, int T_q, int T_full, int D);

/* Process a slice of (b * H_q + h) indices in [bh_start, bh_end).
 * Used by the cpp_ext path to parallelise via std::thread (no libgomp). */
void int8_sdpa_kernel_slice(const float       *q,
                            const signed char *k_data,
                            const float       *k_scale,
                            const signed char *v_data,
                            const float       *v_scale,
                            const float       *mask,
                            float              scale,
                            float             *out,
                            int B, int H_q, int H_kv, int T_q, int T_full, int D,
                            int bh_start, int bh_end);

#ifdef __cplusplus
}
#endif
