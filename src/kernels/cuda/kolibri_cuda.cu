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

// The router was ONE block of X threads: each thread reduced a whole 2560-element bf16 row serially
// and thread 0 ran the top-k as a serial scan, 0.86 ms per layer and 68% of a decode token.  Now the
// dot products are one block per expert (384 blocks, a real GEMV) and the top-k is a parallel
// reduction.  The selection rule is unchanged and is a total order - ranked descending, lower expert
// index on a tie - so a tree reduction gives the same ids as the serial scan did.
__device__ __forceinline__ void better(float v, int i, float& bv, int& bi) {
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
}

__global__ void route_dots_k(const float* x, const __nv_bfloat16* router, const float* bias, float* raw) {
    const int e = blockIdx.x;
    float s = 0.f;
    for (int i = threadIdx.x; i < E; i += blockDim.x) s += __bfloat162float(router[(size_t)e * E + i]) * x[i];
    __shared__ float red[128];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int d = blockDim.x >> 1; d; d >>= 1) { if (threadIdx.x < d) red[threadIdx.x] += red[threadIdx.x + d]; __syncthreads(); }
    if (threadIdx.x == 0) raw[e] = red[0];
}

__global__ void route_topk_k(const float* raw, const float* bias, int32_t* ids, float* weights) {
    __shared__ float sv[512];
    __shared__ int si[512];
    const int e = threadIdx.x;
    float r = raw[e] + bias[e];
    for (int j = 0; j < K; ++j) {
        sv[e] = r; si[e] = e;
        __syncthreads();
        for (int d = 256; d; d >>= 1) {
            if (e < d && e + d < X) better(sv[e + d], si[e + d], sv[e], si[e]);
            __syncthreads();
        }
        if (e == 0) { ids[j] = si[0]; weights[j] = 1.f / (1.f + expf(-raw[si[0]])); }
        __syncthreads();
        const int w = si[0];
        if (e == w) r = -1.0e30f;      // the winner sits out the next round
        __syncthreads();
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
void route(const float* x,const void* router,const float* bias,int32_t* ids,float* weights,float* raw,void* s) {
    route_dots_k<<<X,128,0,(cudaStream_t)s>>>(x,(const __nv_bfloat16*)router,bias,raw);
    route_topk_k<<<1,X,0,(cudaStream_t)s>>>(raw,bias,ids,weights);
}
void add_norm_residual(const float* routed,const float* shared,const float* w,float* residual,int n,float eps,void* s) { add_norm_res_k<<<1,256,0,(cudaStream_t)s>>>(routed,shared,w,residual,n,eps); }

} // namespace strata::kernels::kolibri_cuda
