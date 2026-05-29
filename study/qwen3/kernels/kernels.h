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

#ifdef __cplusplus
}
#endif
