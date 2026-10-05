#include "strata/kernels/kolibri_cuda.hpp"

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>

namespace strata::kernels::kolibri_cuda {
namespace {
constexpr int E = 2560, H = 48, HK = 4, D = 128, X = 384, K = 6;

__global__ void norm_k(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d]; __syncthreads(); }
    const float r = rsqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * r * (w ? w[i] : 1.f);
}

__global__ void norm_res_k(const float* x, const float* w, float* residual, int n, float eps) {
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i] * x[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d]; __syncthreads(); }
    const float r = rsqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) residual[i] += x[i] * r * w[i];
}

__global__ void qkn_k(float* q, float* k, const float* v, const float* qw, const float* kw,
                      float* kc, float* vc, const int* pp, int rope, float theta) {
    const int h = blockIdx.x;
    float* z = h < H ? q + h * D : k + (h - H) * D;
    const float* w = h < H ? qw : kw;
    __shared__ float red[128];
    const int i = threadIdx.x;
    red[i] = z[i] * z[i];
    __syncthreads();
    for (int d = 64; d; d >>= 1) { if (i < d) red[i] += red[i + d]; __syncthreads(); }
    z[i] *= rsqrtf(red[0] / D + 1e-6f) * w[i];
    __syncthreads();
    if (rope && i < D / 2) {
        const int j = i + D / 2;
        const float ang = (*pp) * powf(theta, -2.f * i / D);
        float s, c; sincosf(ang, &s, &c);
        const float a = z[i], b = z[j]; z[i] = a*c - b*s; z[j] = a*s + b*c;
    }
    __syncthreads();
    if (h >= H) {
        const int kh = h - H, p = *pp;
        kc[((size_t)p * HK + kh) * D + i] = z[i];
        vc[((size_t)p * HK + kh) * D + i] = v[kh * D + i];
    }
}

__global__ void attn_k(const float* q, const float* kc, const float* vc, float* out,
                       const int* pp, int sliding, int window, int max_context) {
    const int h = blockIdx.x, i = threadIdx.x, p = min(*pp, max_context - 1), kh = h / (H / HK);
    const int first = sliding ? max(0, p - window + 1) : 0;
    const int nk = p - first + 1;
    extern __shared__ float score[];          // nk window scores, then the reduction scratch
    float* scratch = score + max_context;
    // Scores: the keys are strided across the block.  One thread per head doing every dot product
    // (65k serial FMAs for a 513-key window) was the whole cost of this kernel.
    float hi = -1.0e30f;
    for (int t = first + i; t <= p; t += blockDim.x) {
        float d = 0.f;
        const float* kk = kc + ((size_t)t * HK + kh) * D;
        for (int j = 0; j < D; ++j) d += q[h*D+j] * kk[j];
        score[t - first] = d * 0.08838834764831845f;
        hi = fmaxf(hi, score[t - first]);
    }
    scratch[i] = hi;
    __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) { if (i < d) scratch[i] = fmaxf(scratch[i], scratch[i + d]); __syncthreads(); }
    const float high = scratch[0];
    float s = 0.f;
    for (int t = i; t < nk; t += blockDim.x) { const float a = expf(score[t] - high); score[t] = a; s += a; }
    scratch[i] = s;
    __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) { if (i < d) scratch[i] += scratch[i + d]; __syncthreads(); }
    const float denom = scratch[0];
    float a = 0.f;
    for (int t = first; t <= p; ++t) a += score[t - first] * vc[((size_t)t * HK + kh) * D + i];
    out[h*D+i] = a / denom;
}

__global__ void route_k(const float* x, const __nv_bfloat16* router, const float* bias, int32_t* ids, float* weights) {
    __shared__ float raw[X], ranked[X];
    const int e = threadIdx.x;
    float v = 0.f;
    for (int i = 0; i < E; ++i) v += __bfloat162float(router[(size_t)e * E + i]) * x[i];
    raw[e] = v; ranked[e] = v + bias[e];
    __syncthreads();
    if (e == 0) {
        for (int j = 0; j < K; ++j) {
            int best = -1;
            for (int i = 0; i < X; ++i)
                if (best < 0 || ranked[i] > ranked[best] || (ranked[i] == ranked[best] && i < best)) best = i;
            ids[j] = best;
            weights[j] = 1.f / (1.f + expf(-raw[best]));
            ranked[best] = -1.0e30f;
        }
    }
}

__global__ void add_norm_res_k(const float* routed, const float* shared, const float* w,
                               float* residual, int n, float eps) {
    __shared__ float red[256];
    float s = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { float v = routed[i] + shared[i]; s += v*v; }
    red[threadIdx.x] = s;
    __syncthreads();
    for (int d = 128; d; d >>= 1) { if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x+d]; __syncthreads(); }
    const float r = rsqrtf(red[0] / n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) residual[i] += (routed[i] + shared[i]) * r * w[i];
}
}

void rms_norm(const float* x, const float* w, float* y, int n, float eps, void* s) { norm_k<<<1,256,0,(cudaStream_t)s>>>(x,w,y,n,eps); }
void rms_norm_residual(const float* x, const float* w, float* r, int n, float eps, void* s) { norm_res_k<<<1,256,0,(cudaStream_t)s>>>(x,w,r,n,eps); }
void qk_norm_rope_cache(float* q,float* k,const float* v,const float* qw,const float* kw,float* kc,float* vc,const int* p,bool rope,float theta,void* s) { qkn_k<<<H+HK,D,0,(cudaStream_t)s>>>(q,k,v,qw,kw,kc,vc,p,rope,theta); }
void attention(const float* q,const float* kc,const float* vc,float* out,const int* p,bool sliding,int window,int max_context,void* s) { attn_k<<<H,D,(size_t)(max_context + D)*sizeof(float),(cudaStream_t)s>>>(q,kc,vc,out,p,sliding,window,max_context); }
void route(const float* x,const void* router,const float* bias,int32_t* ids,float* weights,void* s) { route_k<<<1,X,0,(cudaStream_t)s>>>(x,(const __nv_bfloat16*)router,bias,ids,weights); }
void add_norm_residual(const float* routed,const float* shared,const float* w,float* residual,int n,float eps,void* s) { add_norm_res_k<<<1,256,0,(cudaStream_t)s>>>(routed,shared,w,residual,n,eps); }

} // namespace strata::kernels::kolibri_cuda
