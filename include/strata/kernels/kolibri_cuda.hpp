#pragma once

#include <cstdint>

namespace strata::kernels::kolibri_cuda {

void rms_norm(const float* x, const float* weight, float* y, int n, float eps, void* stream);
void rms_norm_residual(const float* branch, const float* weight, float* residual, int n, float eps, void* stream);
void qk_norm_rope_cache(float* q, float* k, const float* v, const float* qw, const float* kw,
                        float* kc, float* vc, const int* pos, bool rope, float theta, void* stream);
void attention(const float* q, const float* kc, const float* vc, float* out, const int* pos,
               bool sliding, int window, int max_context, void* stream);
void route(const float* x, const void* router_bf16, const float* bias, int32_t* ids, float* weights,
           float* raw, void* stream);
void add_norm_residual(const float* routed, const float* shared, const float* weight, float* residual,
                       int n, float eps, void* stream);

} // namespace strata::kernels::kolibri_cuda
