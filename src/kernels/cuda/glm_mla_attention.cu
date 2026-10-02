#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <unordered_map>

namespace strata::kernels::glm {
namespace {
struct Scratch {
    struct ResidentCache { float* device=nullptr; size_t capacity=0; int last_n=0; };
    float *q=nullptr,*attn=nullptr,*scores=nullptr;
    size_t q_cap=0,attn_cap=0,scores_cap=0;
    std::unordered_map<const float*,ResidentCache> resident_caches;
    std::mutex mutex;
    ~Scratch(){cudaFree(q);cudaFree(attn);cudaFree(scores);for(auto& kv:resident_caches)cudaFree(kv.second.device);}
    bool reserve(float*& p,size_t& cap,size_t n){
        if(p&&n<=cap)return true;
        if(p)cudaFree(p);
        p=nullptr;cap=0;
        if(cudaMalloc(&p,n*sizeof(float))!=cudaSuccess)return false;
        cap=n;return true;
    }
};
Scratch& scratch(){static Scratch s;return s;}

__global__ void mla_latent_attention(const float* q,const float* cache,int n_cache,int kv_lora,
                                     float scale,float* scores,float* out){
    const int h=(int)blockIdx.x,tid=(int)threadIdx.x;
    __shared__ float reduce[256];
    const float* qh=q+(size_t)h*kv_lora;
    float local_max=-1.0e30f;
    for(int t=tid;t<n_cache;t+=blockDim.x){
        const float* kt=cache+(size_t)t*kv_lora;float dot=0.0f;
        for(int i=0;i<kv_lora;++i)dot=fmaf(qh[i],kt[i],dot);
        const float s=dot*scale;scores[(size_t)h*n_cache+t]=s;local_max=fmaxf(local_max,s);
    }
    reduce[tid]=local_max;__syncthreads();
    for(int step=blockDim.x/2;step>0;step>>=1){if(tid<step)reduce[tid]=fmaxf(reduce[tid],reduce[tid+step]);__syncthreads();}
    const float max_score=reduce[0];float local_sum=0.0f;
    for(int t=tid;t<n_cache;t+=blockDim.x){
        float p=expf(scores[(size_t)h*n_cache+t]-max_score);scores[(size_t)h*n_cache+t]=p;local_sum+=p;
    }
    reduce[tid]=local_sum;__syncthreads();
    for(int step=blockDim.x/2;step>0;step>>=1){if(tid<step)reduce[tid]+=reduce[tid+step];__syncthreads();}
    const float denom=reduce[0];
    for(int i=tid;i<kv_lora;i+=blockDim.x){
        float acc=0.0f;
        for(int t=0;t<n_cache;++t)acc=fmaf(scores[(size_t)h*n_cache+t],cache[(size_t)t*kv_lora+i],acc);
        out[(size_t)h*kv_lora+i]=acc/denom;
    }
}
}

bool mla_attention_cuda(const float* qcur,const float* cache,int n_cache,int n_head,int head_dim,int kv_lora,
                        float* attn,char* error,size_t error_capacity){
    auto fail=[&](const char* msg){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",msg);return false;};
    if(!qcur||!cache||!attn||n_cache<1||n_cache>8192||n_head<1||head_dim<1||kv_lora<1)
        return fail("invalid MLA attention geometry or pointers");
    Scratch& s=scratch();std::lock_guard<std::mutex> lock(s.mutex);
    const size_t qn=(size_t)n_head*kv_lora,cn=(size_t)n_cache*kv_lora,on=qn,sn=(size_t)n_head*n_cache;
    if(!s.reserve(s.q,s.q_cap,qn)||!s.reserve(s.attn,s.attn_cap,on)||
       !s.reserve(s.scores,s.scores_cap,sn))return fail("MLA attention scratch allocation failed");
    auto it=s.resident_caches.find(cache);
    bool full_upload=false;
    if(it==s.resident_caches.end()){
        Scratch::ResidentCache entry;
        size_t cap=1;while(cap<cn)cap*=2;
        if(cudaMalloc(&entry.device,cap*sizeof(float))!=cudaSuccess)return fail("MLA resident cache allocation failed");
        entry.capacity=cap;entry.last_n=0;
        it=s.resident_caches.emplace(cache,entry).first;
        full_upload=true;
    }else if(cn>it->second.capacity){
        size_t cap=it->second.capacity?it->second.capacity:1;while(cap<cn)cap*=2;
        float* grown=nullptr;
        if(cudaMalloc(&grown,cap*sizeof(float))!=cudaSuccess)return fail("MLA resident cache growth failed");
        cudaFree(it->second.device);it->second.device=grown;it->second.capacity=cap;it->second.last_n=0;
        full_upload=true;
    }else if(n_cache!=it->second.last_n+1){
        // A repeated depth or a reset (depth moving backwards) means the host cache may have been rewritten.
        full_upload=true;
    }
    cudaError_t e=cudaMemcpy(s.q,qcur,qn*sizeof(float),cudaMemcpyHostToDevice);
    if(e==cudaSuccess){
        const size_t first=full_upload?0:(size_t)it->second.last_n;
        const size_t count=full_upload?cn:(size_t)kv_lora;
        e=cudaMemcpy(it->second.device+first,cache+first,count*sizeof(float),cudaMemcpyHostToDevice);
    }
    if(e!=cudaSuccess)return fail(cudaGetErrorString(e));
    it->second.last_n=n_cache;
    mla_latent_attention<<<n_head,256>>>(s.q,it->second.device,n_cache,kv_lora,1.0f/std::sqrt((float)head_dim),s.scores,s.attn);
    if((e=cudaGetLastError())!=cudaSuccess||(e=cudaDeviceSynchronize())!=cudaSuccess)return fail(cudaGetErrorString(e));
    if((e=cudaMemcpy(attn,s.attn,on*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)return fail(cudaGetErrorString(e));
    if(error&&error_capacity)error[0]='\0';return true;
}
} // namespace strata::kernels::glm
