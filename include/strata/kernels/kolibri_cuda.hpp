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

// Q4_0 KV cache (see the note in kolibri_cuda.cu): 0.5625 bytes per element instead of 4.
void attention_q4(const float* q, const unsigned char* k_q4, const unsigned char* v_q4, float* out,
                  const int* pos, bool sliding, int window, int max_context, void* stream);
void qk_norm_rope_scratch(float* q, float* k, const float* v, const float* qw, const float* kw,
                          float* ks, float* vs, const int* pos, bool rope, float theta, void* stream);
void attention_q8(const float* q, const unsigned char* k_q8, const unsigned char* v_q8, float* out,
                  const int* pos, bool sliding, int window, int max_context, void* stream);
void kv_store_q4(unsigned char* k_q4, unsigned char* v_q4, const float* ks, const float* vs,
                 const int* pos, int max_context, void* stream);
void kv_store_q8(unsigned char* k_q8, unsigned char* v_q8, const float* ks, const float* vs,
                 const int* pos, int max_context, void* stream);
size_t kv_q4_cell_bytes();
size_t kv_q8_cell_bytes();

} // namespace strata::kernels::kolibri_cuda
