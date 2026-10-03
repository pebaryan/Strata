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

// One warp per output row of a per-head matrix-vector product: y[h*rows+i] = sum_j m[(h*rows+i)*cols+j] * x[h*cols+j].
// Serves both the K absorption (rows=kv_lora, cols=head_dim) and the V un-absorption (rows=head_dim, cols=kv_lora).
__global__ void mla_head_matvec(const float* m,const float* x,float* y,int rows,int cols,int total_rows){
    const int warp=(int)(blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5)),lane=(int)(threadIdx.x&31);
    if(warp>=total_rows)return;
    const int h=warp/rows;
    const float* row=m+(size_t)warp*cols;const float* xh=x+(size_t)h*cols;
    float acc=0.0f;
    for(int j=lane;j<cols;j+=32)acc=fmaf(row[j],xh[j],acc);
    for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if(lane==0)y[warp]=acc;
}
}

namespace {
struct HeadMatvecState {
    std::unordered_map<const float*,float*> weights;   // host pointer -> resident device copy (uploaded once)
    float *x=nullptr,*y=nullptr;size_t x_cap=0,y_cap=0;
    std::mutex mutex;
    ~HeadMatvecState(){for(auto& kv:weights)cudaFree(kv.second);cudaFree(x);cudaFree(y);}
};
HeadMatvecState& head_state(){static HeadMatvecState s;return s;}
}

bool mla_head_matvec_cuda(const float* weights,const float* x,int n_head,int rows,int cols,float* y,
                          char* error,size_t error_capacity){
    auto fail=[&](const char* msg){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",msg);return false;};
    if(!weights||!x||!y||n_head<1||rows<1||cols<1)return fail("invalid MLA head matvec arguments");
    HeadMatvecState& s=head_state();std::lock_guard<std::mutex> lock(s.mutex);
    const size_t wn=(size_t)n_head*rows*cols,xn=(size_t)n_head*cols,yn=(size_t)n_head*rows;
    cudaError_t e=cudaSuccess;
    auto it=s.weights.find(weights);
    if(it==s.weights.end()){
        float* dev=nullptr;
        if((e=cudaMalloc(&dev,wn*sizeof(float)))!=cudaSuccess)return fail(cudaGetErrorString(e));
        if((e=cudaMemcpy(dev,weights,wn*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess){cudaFree(dev);return fail(cudaGetErrorString(e));}
        it=s.weights.emplace(weights,dev).first;
    }
    if(xn>s.x_cap){cudaFree(s.x);s.x=nullptr;if(cudaMalloc(&s.x,xn*sizeof(float))!=cudaSuccess)return fail("MLA matvec scratch");s.x_cap=xn;}
    if(yn>s.y_cap){cudaFree(s.y);s.y=nullptr;if(cudaMalloc(&s.y,yn*sizeof(float))!=cudaSuccess)return fail("MLA matvec scratch");s.y_cap=yn;}
    if((e=cudaMemcpy(s.x,x,xn*sizeof(float),cudaMemcpyHostToDevice))!=cudaSuccess)return fail(cudaGetErrorString(e));
    const int total=n_head*rows,blocks=(total+7)/8;
    mla_head_matvec<<<blocks,256>>>(it->second,s.x,s.y,rows,cols,total);
    if((e=cudaGetLastError())!=cudaSuccess)return fail(cudaGetErrorString(e));
    if((e=cudaMemcpy(y,s.y,yn*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)return fail(cudaGetErrorString(e));
    if(error&&error_capacity)error[0]='\0';
    return true;
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
