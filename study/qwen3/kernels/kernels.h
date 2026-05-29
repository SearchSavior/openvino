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

#ifdef __cplusplus
}
#endif
