#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace strata::kernels::glm {
namespace {
struct Scratch {
    float *q=nullptr,*k=nullptr,*v=nullptr,*g=nullptr,*beta=nullptr,*state=nullptr,*attn=nullptr;
    float *x=nullptr,*fa=nullptr,*fb=nullptr,*beta_pre=nullptr,*gate_out=nullptr,*beta_out=nullptr;
    size_t q_cap=0,k_cap=0,v_cap=0,g_cap=0,beta_cap=0,state_cap=0,attn_cap=0;
    size_t x_cap=0,fa_cap=0,fb_cap=0,beta_pre_cap=0,gate_cap=0,beta_out_cap=0;
    std::unordered_map<const float*,std::pair<size_t,float*>> resident_weights;
    struct ResidentState {
        size_t count=0;
        float* device=nullptr;
        std::vector<float> host_shadow;
    };
    std::unordered_map<const float*,ResidentState> resident_states;
    std::mutex mutex;
    ~Scratch() {
        cudaFree(q); cudaFree(k); cudaFree(v); cudaFree(g); cudaFree(beta); cudaFree(state); cudaFree(attn);
        cudaFree(x); cudaFree(fa); cudaFree(fb); cudaFree(beta_pre); cudaFree(gate_out); cudaFree(beta_out);
        for (auto& kv:resident_weights) cudaFree(kv.second.second);
        for (auto& kv:resident_states) cudaFree(kv.second.device);
    }
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
            for (int i=0;i<hd;++i)
                sum=fmaf(state[state_off+(size_t)i*hd+tid]*expf(g[row+i]), k[row+i], sum);
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

__global__ void kda_rows(const float* w,const float* x,float* y,int tokens,int rows,int cols) {
    const int ix=(int)(blockIdx.x*blockDim.x+threadIdx.x), total=tokens*rows;
    if (ix>=total) return;
    const int t=ix/rows,r=ix%rows;
    const float* wr=w+(size_t)r*cols; const float* xr=x+(size_t)t*cols;
    float acc=0.0f;
    for (int c=0;c<cols;++c) acc=fmaf(wr[c],xr[c],acc);
    y[(size_t)t*rows+r]=acc;
}

__global__ void kda_apply_gates(const float* fb,const float* beta_pre,const float* ssm_a,const float* dt_bias,
                                int tokens,int nh,int hd,float* g,float* beta) {
    const int ix=(int)(blockIdx.x*blockDim.x+threadIdx.x), ng=tokens*nh*hd, nb=tokens*nh;
    if (ix<ng) {
        const int t=ix/(nh*hd), rem=ix%(nh*hd), h=rem/hd, i=rem%hd;
        const float z=-(ssm_a[h]*(fb[ix]+dt_bias[h*hd+i]));
        g[ix]=-5.0f/(1.0f+expf(-z));
    }
    if (ix<nb) beta[ix]=1.0f/(1.0f+expf(-beta_pre[ix]));
}

bool resident_weight(Scratch& s,const float* host,size_t n,float*& dev) {
    auto it=s.resident_weights.find(host);
    if (it!=s.resident_weights.end() && it->second.first==n) { dev=it->second.second; return true; }
    if (it!=s.resident_weights.end()) { cudaFree(it->second.second); s.resident_weights.erase(it); }
    float* d=nullptr;
    if (cudaMalloc(&d,n*sizeof(float))!=cudaSuccess) return false;
    if (cudaMemcpy(d,host,n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess) { cudaFree(d); return false; }
    s.resident_weights.emplace(host,std::make_pair(n,d)); dev=d; return true;
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
    float* state_device=s.state;
    if (state) {
        auto it=s.resident_states.find(state);
        if (it==s.resident_states.end() || it->second.count!=st) {
            if (it!=s.resident_states.end()) {
                cudaFree(it->second.device);
                s.resident_states.erase(it);
            }
            Scratch::ResidentState entry;
            entry.count=st;
            if (cudaMalloc(&entry.device,st*sizeof(float))!=cudaSuccess)
                return fail("CUDA resident state allocation failed");
            entry.host_shadow.assign(state,state+st);
            e=cudaMemcpy(entry.device,state,st*sizeof(float),cudaMemcpyHostToDevice);
            if (e!=cudaSuccess) { cudaFree(entry.device); return fail(cudaGetErrorString(e)); }
            it=s.resident_states.emplace(state,std::move(entry)).first;
        } else if (std::memcmp(it->second.host_shadow.data(),state,st*sizeof(float))!=0) {
            // The host buffer is authoritative: a request reset or external state edit invalidates residency.
            e=cudaMemcpy(it->second.device,state,st*sizeof(float),cudaMemcpyHostToDevice);
            if (e!=cudaSuccess) return fail(cudaGetErrorString(e));
            it->second.host_shadow.assign(state,state+st);
        }
        state_device=it->second.device;
    } else {
        e=cudaMemset(s.state,0,st*sizeof(float));
    }
    if (e!=cudaSuccess) return fail(cudaGetErrorString(e));
    kda_recur<<<nh,256,(size_t)2*hd*sizeof(float)>>>(s.q,s.k,s.v,s.g,s.beta,tokens,nh,hd,state_device,s.attn);
    if ((e=cudaGetLastError())!=cudaSuccess || (e=cudaDeviceSynchronize())!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if ((e=cudaMemcpy(attn,s.attn,seq*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if (state) {
        if ((e=cudaMemcpy(state,state_device,st*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
            return fail(cudaGetErrorString(e));
        s.resident_states.find(state)->second.host_shadow.assign(state,state+st);
    }
    if (error && error_capacity) error[0]='\0';
    return true;
}

bool kda_gates_cuda(const float* xn,const float* ssm_f_a,const float* ssm_f_b,const float* ssm_beta,
                    const float* ssm_a,const float* dt_bias,int tokens,int n_embd,int nh,int hd,
                    float* g,float* beta,char* error,size_t error_capacity) {
    auto fail=[&](const char* msg) { if(error&&error_capacity) std::snprintf(error,error_capacity,"%s",msg); return false; };
    if(!xn||!ssm_f_a||!ssm_f_b||!ssm_beta||!ssm_a||!dt_bias||!g||!beta||tokens<1||n_embd<1||nh<1||hd<1)
        return fail("invalid KDA gate arguments");
    Scratch& s=scratch(); std::lock_guard<std::mutex> lock(s.mutex);
    const int di=nh*hd; const size_t x_n=(size_t)tokens*n_embd,fa_n=(size_t)tokens*hd;
    const size_t fb_n=(size_t)tokens*di,beta_n=(size_t)tokens*nh,gate_n=fb_n;
    if(!s.reserve(s.x,s.x_cap,x_n)||!s.reserve(s.fa,s.fa_cap,fa_n)||!s.reserve(s.fb,s.fb_cap,fb_n)||
       !s.reserve(s.beta_pre,s.beta_pre_cap,beta_n)||!s.reserve(s.gate_out,s.gate_cap,gate_n)||
       !s.reserve(s.beta_out,s.beta_out_cap,beta_n)) return fail("KDA gate scratch allocation failed");
    float *d_fa_w=nullptr,*d_fb_w=nullptr,*d_beta_w=nullptr,*d_a=nullptr,*d_bias=nullptr;
    if(!resident_weight(s,ssm_f_a,(size_t)hd*n_embd,d_fa_w)||
       !resident_weight(s,ssm_f_b,(size_t)di*hd,d_fb_w)||
       !resident_weight(s,ssm_beta,(size_t)nh*n_embd,d_beta_w)||
       !resident_weight(s,ssm_a,(size_t)nh,d_a)||!resident_weight(s,dt_bias,(size_t)di,d_bias))
        return fail("KDA gate weight upload failed");
    cudaError_t e=cudaMemcpy(s.x,xn,x_n*sizeof(float),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess) return fail(cudaGetErrorString(e));
    kda_rows<<<(tokens*hd+255)/256,256>>>(d_fa_w,s.x,s.fa,tokens,hd,n_embd);
    kda_rows<<<(tokens*di+255)/256,256>>>(d_fb_w,s.fa,s.fb,tokens,di,hd);
    kda_rows<<<(tokens*nh+255)/256,256>>>(d_beta_w,s.x,s.beta_pre,tokens,nh,n_embd);
    const int count=(int)std::max(gate_n,beta_n);
    kda_apply_gates<<<(count+255)/256,256>>>(s.fb,s.beta_pre,d_a,d_bias,tokens,nh,hd,s.gate_out,s.beta_out);
    if((e=cudaGetLastError())!=cudaSuccess||(e=cudaDeviceSynchronize())!=cudaSuccess) return fail(cudaGetErrorString(e));
    if((e=cudaMemcpy(g,s.gate_out,gate_n*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess||
       (e=cudaMemcpy(beta,s.beta_out,beta_n*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
        return fail(cudaGetErrorString(e));
    if(error&&error_capacity) error[0]='\0';
    return true;
}
} // namespace strata::kernels::glm
