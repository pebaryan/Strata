#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <mutex>

namespace strata::kernels::glm {
namespace {
struct Scratch {
    float *q=nullptr,*k=nullptr,*v=nullptr,*g=nullptr,*beta=nullptr,*state=nullptr,*attn=nullptr;
    size_t q_cap=0,k_cap=0,v_cap=0,g_cap=0,beta_cap=0,state_cap=0,attn_cap=0;
    std::mutex mutex;
    ~Scratch() { cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(g); cudaFree(beta); cudaFree(state); cudaFree(attn); }
    bool reserve(float*& p, size_t& cap, size_t need) {
        if (p && need <= cap) return true;
        if (p) cudaFree(p);
        p=nullptr; cap=0;
        if (cudaMalloc(&p, need*sizeof(float)) != cudaSuccess) return false;
        cap=need; return true;
    }
};
Scratch& scratch() { static Scratch s; return s; }

__global__ void kda_recur(const float* q, const float* k, const float* v, const float* g,
                          const float* beta, int tokens, int nh, int hd, float* state, float* attn) {
    const int h=(int)blockIdx.x, tid=(int)threadIdx.x;
    extern __shared__ float sh[];
    float* pred=sh; float* delta=sh+hd;
    const size_t state_off=(size_t)h*hd*hd;
    for (int t=0;t<tokens;++t) {
        const size_t row=((size_t)t*nh+h)*hd;
        if (tid<hd) {
            float sum=0.0f;
            for (int i=0;i<hd;++i) sum=fmaf(state[state_off+(size_t)i*hd+tid], k[row+i], sum);
            pred[tid]=sum;
        }
        __syncthreads();
        if (tid<hd) delta[tid]=(v[row+tid]-pred[tid])*beta[(size_t)t*nh+h];
        __syncthreads();
        for (int ix=tid;ix<hd*hd;ix+=blockDim.x) {
            const int i=ix/hd, j=ix%hd;
            const float decay=expf(g[row+i]);
            state[state_off+ix]=fmaf(k[row+i],delta[j],state[state_off+ix]*decay);
        }
        __syncthreads();
        if (tid<hd) {
            float sum=0.0f;
            for (int i=0;i<hd;++i) sum=fmaf(state[state_off+(size_t)i*hd+tid],q[row+i],sum);
            attn[row+tid]=sum*rsqrtf((float)hd);
        }
        __syncthreads();
    }
}
} // namespace

bool kda_recurrence_cuda(const float* q, const float* k, const float* v, const float* g, const float* beta,
                         int tokens, int nh, int hd, float* state, float* attn, char* error, size_t error_capacity) {
    auto fail=[&](const char* msg) { if (error && error_capacity) std::snprintf(error,error_capacity,"%s",msg); return false; };
    if (!q || !k || !v || !g || !beta || !attn || tokens<1 || nh<1 || hd<1)
        return fail("invalid KDA recurrence arguments");
    Scratch& s=scratch(); std::lock_guard<std::mutex> lock(s.mutex);
    const size_t seq=(size_t)tokens*nh*hd, st=(size_t)nh*hd*hd;
    if (!s.reserve(s.q,s.q_cap,seq) || !s.reserve(s.k,s.k_cap,seq) ||
        !s.reserve(s.v,s.v_cap,seq) || !s.reserve(s.g,s.g_cap,seq) ||
        !s.reserve(s.beta,s.beta_cap,(size_t)tokens*nh) || !s.reserve(s.state,s.state_cap,st) ||
        !s.reserve(s.attn,s.attn_cap,seq)) return fail("CUDA scratch allocation failed");
    cudaError_t e=cudaSuccess;
    if ((e=cudaMemcpy(s.q,q,seq*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess ||
        (e=cudaMemcpy(s.k,k,seq*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess ||
        (e=cudaMemcpy(s.v,v,seq*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess ||
        (e=cudaMemcpy(s.g,g,seq*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess ||
        (e=cudaMemcpy(s.beta,beta,(size_t)tokens*nh*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if (state) e=cudaMemcpy(s.state,state,st*sizeof(float),cudaMemcpyHostToDevice);
    else e=cudaMemset(s.state,0,st*sizeof(float));
    if (e!=cudaSuccess) return fail(cudaGetErrorString(e));
    kda_recur<<<nh,256,(size_t)2*hd*sizeof(float)>>>(s.q,s.k,s.v,s.g,s.beta,tokens,nh,hd,s.state,s.attn);
    if ((e=cudaGetLastError())!=cudaSuccess || (e=cudaDeviceSynchronize())!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if ((e=cudaMemcpy(attn,s.attn,seq*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if (state && (e=cudaMemcpy(state,s.state,st*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if (error && error_capacity) error[0]='\0';
    return true;
}
} // namespace strata::kernels::glm
