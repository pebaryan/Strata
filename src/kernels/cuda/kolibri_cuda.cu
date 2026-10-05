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

// ---- Q4_0 KV storage ------------------------------------------------------------------------------
//
// Same layout as strata::kernels::block_q4_0 (kv_q4.hpp): 18 bytes per 32 values, an fp16 scale then
// 32 four-bit codes, element j in the low nibble of qs[j] and element j+16 in the high one.
//
// The fp32 cache costs 4 bytes per element, 200 KiB per token over 50 layers (4 KV heads of 128 dims,
// K and V).  At 0.5625 bytes per element this is 28 KiB per token, so a 16384-token context costs
// ~28 MB per layer instead of ~200 MB.  It is lossy, and the gate measures how lossy rather than
// assuming it.  K is quantized after the qk-norm and RoPE and V raw, as llama.cpp's --cache-type-k/v
// q4_0 does.
// Two element formats, both with a 32-value block and an fp16 scale, matching the layouts the engine
// already uses elsewhere (kv_q4.hpp for 4 bits, the same header's INT8 mode for 8):
//
//   q4_0: 18 B per block, 0.5625 B/element  - 7.1x smaller than fp32, and TOO LOSSY here.  Measured
//         against the oracle through the gate: cosine 0.966-0.979 at positions 2-4, argmax wrong at
//         three of five.  A Hadamard rotation of K, V and q (which preserves <q,k> and is undone on
//         the attention output) does not rescue it, because a block scale is amax/7 and a typical
//         element therefore carries ~10% error.  Kept because it is the format llama.cpp's
//         --cache-type-k/v q4_0 uses, so the cost can be measured rather than argued about.
//   q8_0: 34 B per block, 1.0625 B/element - 3.8x smaller than fp32 with 16x smaller per-element
//         error (scale amax/127).  This is the one to use.
struct q4_block { unsigned short d; unsigned char qs[16]; };
struct q8_block { unsigned short d; signed char qs[32]; };
static_assert(sizeof(q4_block) == 18, "q4_0 block must be 18 bytes");
static_assert(sizeof(q8_block) == 34, "q8_0 block must be 34 bytes");
constexpr int Q4_STRIDE = 18, Q8_STRIDE = 34;

__device__ __forceinline__ float blk_scale(const unsigned char* blk) {
    return __half2float(__ushort_as_half(*(const unsigned short*) blk));
}

// Orthonormal Fast Walsh-Hadamard transform of one head's D=128 vector, in shared memory, one thread
// per element.  Outlier channels in K are spread over all dims before rounding, exactly what kv_q4.cu
// does at 256 points for the QSA layers.  H is orthonormal and self-inverse, so <Hq, Hk> = <q, k> and
// the attention output, a mix of rotated values, is rotated back with the same matrix.
__device__ __forceinline__ void fwht128(float* s) {
    for (int h = 1; h < D; h <<= 1) {
        const int i = threadIdx.x, j = i ^ h;
        if (i < D && j > i) { const float a = s[i], b = s[j]; s[i] = a + b; s[j] = a - b; }
        __syncthreads();
    }
    if (threadIdx.x < D) s[threadIdx.x] *= 0.08838834764831845f;   // 1/sqrt(128)
    __syncthreads();
}

template <bool Q4>
__device__ __forceinline__ float qkv_dot32(const unsigned char* blk, const float* q) {
    const float s = blk_scale(blk);
    float acc = 0.f;
    if (Q4) {
        const unsigned char* qs = blk + 2;
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            const unsigned char b = qs[j];
            acc += q[j] * (float) ((int) (b & 0x0F) - 8) + q[16 + j] * (float) ((int) (b >> 4) - 8);
        }
    } else {
        const signed char* qs = (const signed char*) (blk + 2);
        #pragma unroll
        for (int j = 0; j < 32; ++j) acc += q[j] * (float) qs[j];
    }
    return acc * s;
}

// Element `i` of one cell, whose blocks are contiguous.
template <bool Q4>
__device__ __forceinline__ float qkv_at(const unsigned char* cell, int i) {
    const int stride = Q4 ? Q4_STRIDE : Q8_STRIDE;
    const unsigned char* blk = cell + (i >> 5) * stride;
    const int li = i & 31;
    if (Q4) {
        const unsigned char byte = blk[2 + (li & 15)];
        const int code = (li < 16) ? (byte & 0x0F) : (byte >> 4);
        return blk_scale(blk) * (float) (code - 8);
    }
    return blk_scale(blk) * (float) ((const signed char*) (blk + 2))[li];
}

// Quantize the current position's (already rotated) K and V: one block per KV head, one warp per
// 32-value group, so a D=128 head is four groups and the launch is HK blocks of 128 threads.
template <bool Q4>
__global__ void kvq_store_k(unsigned char* kq, unsigned char* vq, const float* ks, const float* vs,
                            const int* pp, int n_kv, int nblk, int max_context) {
    const int kh = blockIdx.x, g = threadIdx.x >> 5, lane = threadIdx.x & 31;
    if (g >= nblk) return;
    const int stride = Q4 ? Q4_STRIDE : Q8_STRIDE;
    const int p = min(*pp, max_context - 1);
    const size_t cell = ((size_t) p * n_kv + kh) * (size_t) nblk * stride + (size_t) g * stride;
    #pragma unroll 1
    for (int which = 0; which < 2; ++which) {
        const float x = which ? vs[kh * D + g * 32 + lane] : ks[kh * D + g * 32 + lane];
        float am = fabsf(x);
        #pragma unroll
        for (int o = 16; o; o >>= 1) am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, o));
        unsigned char* dst = which ? vq : kq;
        if (Q4) {
            const float d = am / 7.0f;                    // ggml q4_0: codes -8..7 at scale amax/7
            int c = (d > 0.f) ? (int) lroundf(x / d) + 8 : 8;
            c = c < 0 ? 0 : (c > 15 ? 15 : c);
            const int c_hi = __shfl_sync(0xffffffffu, c, lane + 16);
            if (lane == 0) *(unsigned short*) (dst + cell) = __half_as_ushort(__float2half(d));
            if (lane < 16) dst[cell + 2 + lane] = (unsigned char) (c | (c_hi << 4));
        } else {
            const float d = am / 127.0f;
            int c = (d > 0.f) ? (int) lroundf(x / d) : 0;
            c = c < -128 ? -128 : (c > 127 ? 127 : c);
            if (lane == 0) *(unsigned short*) (dst + cell) = __half_as_ushort(__float2half(d));
            ((signed char*) (dst + cell + 2))[lane] = (signed char) c;
        }
    }
}

// qkn_k, with the cell written to a scratch cell at index 0 so the store kernel can quantize it: a
// cache is indexed by position and a scratch is not.
__global__ void qkn_scratch_k(float* q, float* k, const float* v, const float* qw, const float* kw,
                              float* ks, float* vs, const int* pp, int rope, float theta) {
    const int h = blockIdx.x, i = threadIdx.x;
    float* z = h < H ? q + h * D : k + (h - H) * D;
    const float* w = h < H ? qw : kw;
    __shared__ float red[128];
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
    __shared__ float rot[128];
    // The query is rotated here too: the scores come from <Hq, Hk>, which equals <q, k>.
    rot[i] = z[i];
    __syncthreads();
    fwht128(rot);
    z[i] = rot[i];
    if (h >= H) {
        const int kh = h - H;
        ks[kh * D + i] = z[i];
        rot[i] = v[kh * D + i];
        __syncthreads();
        fwht128(rot);
        vs[kh * D + i] = rot[i];
    }
}

// attn_k over a q4 cache: the keys are still strided across the block, one key per iteration, and
// the cost is four block unpacks per key.  The value pass reads one nibble per key per thread.
template <bool Q4>
__global__ void attn_q_k(const float* q, const unsigned char* kq, const unsigned char* vq, float* out,
                         const int* pp, int sliding, int window, int max_context) {
    const int h = blockIdx.x, i = threadIdx.x, p = min(*pp, max_context - 1), kh = h / (H / HK);
    const int first = sliding ? max(0, p - window + 1) : 0;
    const int nk = p - first + 1;
    const int nblk = D / 32;
    extern __shared__ float score[];
    float* scratch = score + max_context;
    float hi = -1.0e30f;
    for (int t = first + i; t <= p; t += blockDim.x) {
        const unsigned char* kb = kq + ((size_t) t * HK + kh) * (size_t) nblk * (Q4 ? Q4_STRIDE : Q8_STRIDE);
        float d = 0.f;
        for (int g = 0; g < nblk; ++g) d += qkv_dot32<Q4>(kb + g * (Q4 ? Q4_STRIDE : Q8_STRIDE), q + h * D + g * 32);
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
    for (int t = first; t <= p; ++t) a += score[t - first] * qkv_at<Q4>(vq + ((size_t) t * HK + kh) * (size_t) nblk * (Q4 ? Q4_STRIDE : Q8_STRIDE), i);
    __shared__ float obuf[128];
    obuf[i] = a / denom;
    __syncthreads();
    fwht128(obuf);
    out[h * D + i] = obuf[i];
}

}

void rms_norm(const float* x, const float* w, float* y, int n, float eps, void* s) { norm_k<<<1,256,0,(cudaStream_t)s>>>(x,w,y,n,eps); }
void rms_norm_residual(const float* x, const float* w, float* r, int n, float eps, void* s) { norm_res_k<<<1,256,0,(cudaStream_t)s>>>(x,w,r,n,eps); }
void qk_norm_rope_cache(float* q,float* k,const float* v,const float* qw,const float* kw,float* kc,float* vc,const int* p,bool rope,float theta,void* s) { qkn_k<<<H+HK,D,0,(cudaStream_t)s>>>(q,k,v,qw,kw,kc,vc,p,rope,theta); }
// Both attention kernels keep one float of score per key in dynamic shared memory, so context is
// capped by the SM's shared limit (96 KB here, about 23k tokens) unless a kernel tiles.  Past 48 KB
// the launch needs the attribute set or it fails with "invalid argument".
static size_t g_smem_set = 0;
static void ensure_smem(size_t bytes, const void* kernel) {
    if (bytes <= 48 * 1024 || bytes <= g_smem_set) return;
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) bytes);
    g_smem_set = bytes;
}
void attention(const float* q,const float* kc,const float* vc,float* out,const int* p,bool sliding,int window,int max_context,void* s) {
    const size_t sh = (size_t)(max_context + D) * sizeof(float);
    ensure_smem(sh, (const void*) attn_k);
    attn_k<<<H,D,sh,(cudaStream_t)s>>>(q,kc,vc,out,p,sliding,window,max_context);
}
void attention_q4(const float* q,const unsigned char* kq4,const unsigned char* vq4,float* out,const int* p,bool sliding,int window,int max_context,void* s) {
    const size_t sh = (size_t)(max_context + D) * sizeof(float);
    ensure_smem(sh, (const void*) attn_q_k<true>);
    attn_q_k<true><<<H,D,sh,(cudaStream_t)s>>>(q,kq4,vq4,out,p,sliding,window,max_context);
}
void attention_q8(const float* q,const unsigned char* kq8,const unsigned char* vq8,float* out,const int* p,bool sliding,int window,int max_context,void* s) {
    const size_t sh = (size_t)(max_context + D) * sizeof(float);
    ensure_smem(sh, (const void*) attn_q_k<false>);
    attn_q_k<false><<<H,D,sh,(cudaStream_t)s>>>(q,kq8,vq8,out,p,sliding,window,max_context);
}
void qk_norm_rope_scratch(float* q,float* k,const float* v,const float* qw,const float* kw,float* ks,float* vs,const int* p,bool rope,float theta,void* s) {
    qkn_scratch_k<<<H+HK,D,0,(cudaStream_t)s>>>(q,k,v,qw,kw,ks,vs,p,rope,theta);
}
void kv_store_q4(unsigned char* kq,unsigned char* vq,const float* ks,const float* vs,const int* p,int max_context,void* s) {
    const int nblk = D / 32;
    kvq_store_k<true><<<HK,nblk*32,0,(cudaStream_t)s>>>(kq,vq,ks,vs,p,HK,nblk,max_context);
}
void kv_store_q8(unsigned char* kq,unsigned char* vq,const float* ks,const float* vs,const int* p,int max_context,void* s) {
    const int nblk = D / 32;
    kvq_store_k<false><<<HK,nblk*32,0,(cudaStream_t)s>>>(kq,vq,ks,vs,p,HK,nblk,max_context);
}
size_t kv_q4_cell_bytes() { return (size_t)(D / 32) * Q4_STRIDE; }
size_t kv_q8_cell_bytes() { return (size_t)(D / 32) * Q8_STRIDE; }
void route(const float* x,const void* router,const float* bias,int32_t* ids,float* weights,float* raw,void* s) {
    route_dots_k<<<X,128,0,(cudaStream_t)s>>>(x,(const __nv_bfloat16*)router,bias,raw);
    route_topk_k<<<1,X,0,(cudaStream_t)s>>>(raw,bias,ids,weights);
}
void add_norm_residual(const float* routed,const float* shared,const float* w,float* residual,int n,float eps,void* s) { add_norm_res_k<<<1,256,0,(cudaStream_t)s>>>(routed,shared,w,residual,n,eps); }

} // namespace strata::kernels::kolibri_cuda
