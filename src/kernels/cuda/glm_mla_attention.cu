#include <cuda_runtime.h>
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/glm_indexer_device.hpp"
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
        // `first` is a FLOAT offset: last_n counts rows of kv_lora floats.  (Used unscaled, every incremental row landed at
        // float index last_n instead of last_n*kv_lora, so past ~257 cached positions the kernel attended to stale data.)
        const size_t first=full_upload?0:(size_t)it->second.last_n*(size_t)kv_lora;
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

// ---- batched (prompt-chunk) MLA: absorb -> causal latent attention -> un-absorb, all tokens at once, on the device ----

namespace {
// y[idx] for idx = (t, h, i): sum_j W[(h*rows+i)*cols + j] * x[(t*n_head+h)*cols + j].  One warp per output element.
__global__ void mla_head_matvec_tokens(const float* m,const float* x,float* y,int rows,int cols,int n_head,size_t total){
    const size_t idx=(size_t)blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5);
    const int lane=(int)(threadIdx.x&31);
    if(idx>=total)return;
    const size_t per_token=(size_t)n_head*rows;
    const size_t t=idx/per_token,rem=idx%per_token;
    const int h=(int)(rem/rows);
    const float* row=m+rem*cols;const float* xh=x+(t*n_head+h)*cols;
    float acc=0.0f;
    for(int j=lane;j<cols;j+=32)acc=fmaf(row[j],xh[j],acc);
    for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if(lane==0)y[idx]=acc;
}

// Causal latent attention for a block of S tokens: token t sees cache rows [0, c_base+t].  One thread block per
// (token, head); the scores live in dynamic shared memory (<= 8192 positions = 32 KB), so nothing is materialised in
// global memory however long the chunk is.  Warp-per-position dots keep the cache reads coalesced.
__global__ void mla_latent_attention_batch(const float* qcur,const float* cache,int c_base,int kv_lora,float scale,float* out){
    extern __shared__ float sc[];
    __shared__ float qs[512];
    __shared__ float red[8];
    __shared__ float bcast;
    const int t=(int)blockIdx.x,h=(int)blockIdx.y,tid=(int)threadIdx.x,warp=tid>>5,lane=tid&31;
    const int n=c_base+t+1;
    const float* qh=qcur+((size_t)t*gridDim.y+h)*kv_lora;
    for(int i=tid;i<kv_lora;i+=blockDim.x)qs[i]=qh[i];
    __syncthreads();
    float local_max=-1.0e30f;
    for(int pos=warp;pos<n;pos+=8){
        const float* kt=cache+(size_t)pos*kv_lora;float acc=0.0f;
        for(int i=lane;i<kv_lora;i+=32)acc=fmaf(qs[i],kt[i],acc);
        for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
        if(lane==0){const float s=acc*scale;sc[pos]=s;local_max=fmaxf(local_max,s);}
    }
    // block max (only lane 0 of each warp holds a value; the others carry the -1e30 floor)
    for(int o=16;o>0;o>>=1)local_max=fmaxf(local_max,__shfl_down_sync(0xffffffffu,local_max,o));
    if(lane==0)red[warp]=local_max;
    __syncthreads();
    if(tid==0){float m=red[0];for(int w=1;w<8;++w)m=fmaxf(m,red[w]);bcast=m;}
    __syncthreads();
    const float max_score=bcast;
    float local_sum=0.0f;
    for(int pos=tid;pos<n;pos+=blockDim.x){const float p=expf(sc[pos]-max_score);sc[pos]=p;local_sum+=p;}
    for(int o=16;o>0;o>>=1)local_sum+=__shfl_down_sync(0xffffffffu,local_sum,o);
    __syncthreads();
    if(lane==0)red[warp]=local_sum;
    __syncthreads();
    if(tid==0){float s=0.0f;for(int w=0;w<8;++w)s+=red[w];bcast=s;}
    __syncthreads();
    const float denom=bcast;
    float* oh=out+((size_t)t*gridDim.y+h)*kv_lora;
    for(int i=tid;i<kv_lora;i+=blockDim.x){
        float acc=0.0f;
        for(int pos=0;pos<n;++pos)acc=fmaf(sc[pos],cache[(size_t)pos*kv_lora+i],acc);
        oh[i]=acc/denom;
    }
}


// The same attention over an explicit cell list (the indexer's selection): token t reads cells[t*IDX_CELL_STRIDE .. + ncells[t]).
// Scores live in shared memory (<= IDX_CELL_STRIDE floats).
__global__ void mla_latent_attention_cells(const float* qcur,const float* cache,const int* cells,const int* ncells,int kv_lora,float scale,float* out){
    extern __shared__ float sc[];
    __shared__ float qs[512];
    __shared__ float red[8];
    __shared__ float bcast;
    const int t=(int)blockIdx.x,h=(int)blockIdx.y,tid=(int)threadIdx.x,warp=tid>>5,lane=tid&31;
    const int n=ncells[t];
    const int* cl=cells+(size_t)t*IDX_CELL_STRIDE;
    const float* qh=qcur+((size_t)t*gridDim.y+h)*kv_lora;
    for(int i=tid;i<kv_lora;i+=blockDim.x)qs[i]=qh[i];
    __syncthreads();
    float local_max=-1.0e30f;
    for(int j=warp;j<n;j+=8){
        const float* kt=cache+(size_t)cl[j]*kv_lora;float acc=0.0f;
        for(int i=lane;i<kv_lora;i+=32)acc=fmaf(qs[i],kt[i],acc);
        for(int o=16;o>0;o>>=1)acc+=__shfl_down_sync(0xffffffffu,acc,o);
        if(lane==0){const float s=acc*scale;sc[j]=s;local_max=fmaxf(local_max,s);}
    }
    for(int o=16;o>0;o>>=1)local_max=fmaxf(local_max,__shfl_down_sync(0xffffffffu,local_max,o));
    if(lane==0)red[warp]=local_max;
    __syncthreads();
    if(tid==0){float m=red[0];for(int w=1;w<8;++w)m=fmaxf(m,red[w]);bcast=m;}
    __syncthreads();
    const float max_score=bcast;
    float local_sum=0.0f;
    for(int j=tid;j<n;j+=blockDim.x){const float p=expf(sc[j]-max_score);sc[j]=p;local_sum+=p;}
    for(int o=16;o>0;o>>=1)local_sum+=__shfl_down_sync(0xffffffffu,local_sum,o);
    __syncthreads();
    if(lane==0)red[warp]=local_sum;
    __syncthreads();
    if(tid==0){float s=0.0f;for(int w=0;w<8;++w)s+=red[w];bcast=s;}
    __syncthreads();
    const float denom=bcast;
    float* oh=out+((size_t)t*gridDim.y+h)*kv_lora;
    for(int i=tid;i<kv_lora;i+=blockDim.x){
        float acc=0.0f;
        for(int j=0;j<n;++j)acc=fmaf(sc[j],cache[(size_t)cl[j]*kv_lora+i],acc);
        oh[i]=acc/denom;
    }
}

struct BatchScratch{
    float *q=nullptr,*qcur=nullptr,*attn=nullptr,*v=nullptr,*xin=nullptr,*qrin=nullptr;
    size_t q_cap=0,qcur_cap=0,attn_cap=0,v_cap=0,xin_cap=0,qrin_cap=0;
    cudaStream_t stream=nullptr;   // the indexer's native kernels need an explicit stream; a blocking-flag one orders with the default stream
    std::mutex mutex;
    ~BatchScratch(){cudaFree(q);cudaFree(qcur);cudaFree(attn);cudaFree(v);cudaFree(xin);cudaFree(qrin);if(stream)cudaStreamDestroy(stream);}
    bool reserve(float*& p,size_t& cap,size_t n){
        if(p&&n<=cap)return true;
        if(p)cudaFree(p);
        p=nullptr;cap=0;
        if(cudaMalloc(&p,n*sizeof(float))!=cudaSuccess)return false;
        cap=n;return true;
    }
};
BatchScratch& batch_scratch(){static BatchScratch s;return s;}

float* resident_head_weights(const float* host,size_t n){
    HeadMatvecState& hs=head_state();std::lock_guard<std::mutex> lock(hs.mutex);
    auto it=hs.weights.find(host);
    if(it!=hs.weights.end())return it->second;
    float* dev=nullptr;
    if(cudaMalloc(&dev,n*sizeof(float))!=cudaSuccess)return nullptr;
    if(cudaMemcpy(dev,host,n*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess){cudaFree(dev);return nullptr;}
    hs.weights.emplace(host,dev);
    return dev;
}
}  // namespace

// The one implementation behind both entry points.  With `iw` (and the layer's attn-normed `x` and normed q_a output `qr`,
// [S][n_embd] / [S][q_lora] on the host) it also feeds the sparse indexer: the rows of these S tokens are written, and from
// IDX_SPARSE_FROM positions on each token attends only the cells the indexer selects.  Without it the attention is dense,
// which is the model's own behaviour only up to 2051 positions (and the device kernel's limit is 8192).
static bool attend_batch_impl(const MlaWeights* iw,const MlaGeometry* ig,const float* x_host,const float* qr_host,
                              const float* wk_b,const float* wv_b,const float* q,const float* cache,int c_base,int S,
                              int n_head,int head_dim,int kv_lora,float* v_out,char* error,size_t error_capacity){
    auto fail=[&](const char* msg){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",msg);return false;};
    const int n_total=c_base+S;
    const bool use_idx=iw&&ig&&x_host&&qr_host&&idx_available(*iw);
    if(!wk_b||!wv_b||!q||!cache||!v_out||S<1||c_base<0||(n_total>8192&&!use_idx)||n_head<1||head_dim<1||kv_lora<1||kv_lora>512)
        return fail("invalid batched MLA arguments");
    float* dk=resident_head_weights(wk_b,(size_t)n_head*kv_lora*head_dim);
    float* dv=resident_head_weights(wv_b,(size_t)n_head*head_dim*kv_lora);
    if(!dk||!dv)return fail("batched MLA weight upload failed");
    const size_t q_dim=(size_t)n_head*head_dim,lat=(size_t)n_head*kv_lora;
    BatchScratch& b=batch_scratch();std::lock_guard<std::mutex> blk(b.mutex);
    Scratch& s=scratch();std::lock_guard<std::mutex> lock(s.mutex);
    if(!b.reserve(b.q,b.q_cap,(size_t)S*q_dim)||!b.reserve(b.qcur,b.qcur_cap,(size_t)S*lat)||
       !b.reserve(b.attn,b.attn_cap,(size_t)S*lat)||!b.reserve(b.v,b.v_cap,(size_t)S*q_dim))
        return fail("batched MLA scratch allocation failed");
    // the device copy of the latent cache: rows [0, c_base) must already be there, rows [c_base, n_total) are new
    const size_t cn=(size_t)n_total*kv_lora;
    auto it=s.resident_caches.find(cache);
    size_t upload_from=(size_t)c_base;
    if(it==s.resident_caches.end()){
        Scratch::ResidentCache entry;size_t cap=1;while(cap<cn)cap*=2;
        if(cudaMalloc(&entry.device,cap*sizeof(float))!=cudaSuccess)return fail("MLA resident cache allocation failed");
        entry.capacity=cap;entry.last_n=0;
        it=s.resident_caches.emplace(cache,entry).first;
        upload_from=0;
    }else if(cn>it->second.capacity){
        size_t cap=it->second.capacity?it->second.capacity:1;while(cap<cn)cap*=2;
        float* grown=nullptr;
        if(cudaMalloc(&grown,cap*sizeof(float))!=cudaSuccess)return fail("MLA resident cache growth failed");
        cudaFree(it->second.device);it->second.device=grown;it->second.capacity=cap;
        upload_from=0;
    }else if(it->second.last_n!=c_base){
        upload_from=0;   // the host cache may have been rewritten (a reset, or a depth that does not follow on)
    }
    cudaError_t e=cudaMemcpy(it->second.device+upload_from*kv_lora,cache+upload_from*kv_lora,
                             (size_t)(n_total-(int)upload_from)*kv_lora*sizeof(float),cudaMemcpyHostToDevice);
    if(e==cudaSuccess)e=cudaMemcpy(b.q,q,(size_t)S*q_dim*sizeof(float),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess)return fail(cudaGetErrorString(e));
    it->second.last_n=n_total;
    const size_t tot_k=(size_t)S*n_head*kv_lora,tot_v=(size_t)S*n_head*head_dim;
    mla_head_matvec_tokens<<<(unsigned)((tot_k+7)/8),256>>>(dk,b.q,b.qcur,kv_lora,head_dim,n_head,tot_k);
    const float scale=1.0f/std::sqrt((float)head_dim);
    bool sparse=false;
    if(use_idx){
        char ie[160]="";
        if(!b.stream&&cudaStreamCreate(&b.stream)!=cudaSuccess)return fail("indexer stream creation failed");
        void* sv=(void*)b.stream;
        const size_t xn=(size_t)S*ig->n_embd,qn=(size_t)S*ig->q_lora;
        if(!b.reserve(b.xin,b.xin_cap,xn)||!b.reserve(b.qrin,b.qrin_cap,qn))return fail("indexer scratch allocation failed");
        e=cudaMemcpyAsync(b.xin,x_host,xn*sizeof(float),cudaMemcpyHostToDevice,b.stream);
        if(e==cudaSuccess)e=cudaMemcpyAsync(b.qrin,qr_host,qn*sizeof(float),cudaMemcpyHostToDevice,b.stream);
        if(e!=cudaSuccess)return fail(cudaGetErrorString(e));
        const int wr=idx_write_device(*iw,*ig,cache,b.xin,S,c_base,sv,ie,sizeof(ie));
        if(wr<0)return fail(ie);
        if(n_total>=IDX_SPARSE_FROM){
            sparse=wr==1;
            const int cmax=idx_max_tokens();
            for(int off=0;sparse&&off<S;off+=cmax){
                const int tc=std::min(cmax,S-off);
                int32_t *dc=nullptr,*dn=nullptr;
                const int rd=idx_cells_device(*iw,*ig,cache,b.xin+(size_t)off*ig->n_embd,b.qrin+(size_t)off*ig->q_lora,tc,c_base+off,&dc,&dn,sv,ie,sizeof(ie));
                if(rd<0)return fail(ie);
                if(rd!=1){sparse=false;break;}
                mla_latent_attention_cells<<<dim3((unsigned)tc,(unsigned)n_head),256,(size_t)IDX_CELL_STRIDE*sizeof(float),b.stream>>>(
                    b.qcur+(size_t)off*n_head*kv_lora,it->second.device,dc,dn,kv_lora,scale,b.attn+(size_t)off*n_head*kv_lora);
            }
            if(!sparse){
                static bool warned=false;
                if(!warned){warned=true;std::fprintf(stderr,"[mla] sparse indexer unavailable at %d positions (%s): attending densely\n",n_total,ie);}
                if(n_total>8192)return fail("no sparse indexer beyond 8192 positions");
            }
        }
    }
    if(!sparse)
        mla_latent_attention_batch<<<dim3((unsigned)S,(unsigned)n_head),256,(size_t)n_total*sizeof(float)>>>(
            b.qcur,it->second.device,c_base,kv_lora,scale,b.attn);
    mla_head_matvec_tokens<<<(unsigned)((tot_v+7)/8),256>>>(dv,b.attn,b.v,head_dim,kv_lora,n_head,tot_v);
    if((e=cudaGetLastError())!=cudaSuccess||(e=cudaDeviceSynchronize())!=cudaSuccess)return fail(cudaGetErrorString(e));
    if((e=cudaMemcpy(v_out,b.v,(size_t)S*q_dim*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)return fail(cudaGetErrorString(e));
    if(error&&error_capacity)error[0]='\0';
    return true;
}

bool mla_attend_batch_cuda(const float* wk_b,const float* wv_b,const float* q,const float* cache,int c_base,int S,
                           int n_head,int head_dim,int kv_lora,float* v_out,char* error,size_t error_capacity){
    return attend_batch_impl(nullptr,nullptr,nullptr,nullptr,wk_b,wv_b,q,cache,c_base,S,n_head,head_dim,kv_lora,v_out,error,error_capacity);
}

bool mla_attend_batch_idx_cuda(const MlaWeights& w,const MlaGeometry& g,const float* x,const float* qr,const float* q,const float* cache,
                               int c_base,int S,float* v_out,char* error,size_t error_capacity){
    return attend_batch_impl(&w,&g,x,qr,w.wk_b,w.wv_b,q,cache,c_base,S,g.n_head,g.head_dim,g.kv_lora,v_out,error,error_capacity);
}

// ---- the device-resident single-token MLA block (decode) ---------------------------------------------------------------
//
// One upload of the layer input, one download of its output (and of the new latent row, so the host cache stays complete):
// q_a -> norm -> q_b -> absorb, kv_a -> norm -> append to the resident latent cache, causal attention, un-absorb, wo - all on
// one stream.  The host path is ~7 synchronous round trips for the same work.

namespace {
__device__ __forceinline__ double mla_block_sum(double v,double* red){
    for(int o=16;o>0;o>>=1)v+=__shfl_down_sync(0xffffffffu,v,o);
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    if(lane==0)red[warp]=v;
    __syncthreads();
    double total=0.0;
    if(threadIdx.x==0){for(int w=0;w<(int)(blockDim.x>>5);++w)total+=red[w];red[0]=total;}
    __syncthreads();
    total=red[0];
    __syncthreads();
    return total;
}
// y = rms_norm(x) * w (double accumulation like the host)
__global__ void mla_rmsnorm1(const float* x,const float* w,float* y,int n,float eps){
    __shared__ double red[32];
    double acc=0.0;
    for(int i=threadIdx.x;i<n;i+=blockDim.x)acc+=(double)x[i]*(double)x[i];
    const double ss=mla_block_sum(acc,red);
    const float inv=1.0f/sqrtf((float)(ss/n)+eps);
    for(int i=threadIdx.x;i<n;i+=blockDim.x)y[i]=x[i]*inv*w[i];
}

struct DecodeScratch{
    int n_embd=0,q_lora=0,q_dim=0,lat=0,kv_lora=0;
    float *x=nullptr,*qr=nullptr,*q=nullptr,*kv=nullptr,*qcur=nullptr,*attn=nullptr,*v=nullptr,*out=nullptr;
    void *xq=nullptr,*qrq=nullptr,*vq=nullptr;
    cudaStream_t stream=nullptr;
    ~DecodeScratch(){release();if(stream)cudaStreamDestroy(stream);}
    void release(){
        float** all[]={&x,&qr,&q,&kv,&qcur,&attn,&v,&out};
        for(float** p:all){if(*p)cudaFree(*p);*p=nullptr;}
        if(xq)cudaFree(xq);if(qrq)cudaFree(qrq);if(vq)cudaFree(vq);
        xq=qrq=vq=nullptr;n_embd=0;
    }
    bool alloc(const MlaGeometry& g){
        const int qd=g.n_head*g.head_dim,lt=g.n_head*g.kv_lora;
        if(n_embd==g.n_embd&&q_lora==g.q_lora&&q_dim==qd&&lat==lt&&kv_lora==g.kv_lora&&x)return true;
        release();
        auto mk=[&](float*& p,size_t n){return cudaMalloc(&p,n*sizeof(float))==cudaSuccess;};
        if(!mk(x,g.n_embd)||!mk(qr,g.q_lora)||!mk(q,qd)||!mk(kv,g.kv_lora)||!mk(qcur,lt)||!mk(attn,lt)||!mk(v,qd)||!mk(out,g.n_embd)){release();return false;}
        if(cudaMalloc(&xq,strata::kernels::native_q8_1_bytes(g.n_embd,1))!=cudaSuccess||
           cudaMalloc(&qrq,strata::kernels::native_q8_1_bytes(g.q_lora,1))!=cudaSuccess||
           cudaMalloc(&vq,strata::kernels::native_q8_1_bytes(qd,1))!=cudaSuccess){release();return false;}
        if(!stream&&cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking)!=cudaSuccess){release();return false;}
        n_embd=g.n_embd;q_lora=g.q_lora;q_dim=qd;lat=lt;kv_lora=g.kv_lora;
        return true;
    }
};
DecodeScratch& decode_scratch(){static DecodeScratch d;return d;}
}  // namespace

// 1 done; 0 declined (nothing touched: the host path must run); -1 failed after work began.
// The one implementation behind both entry points: host-pointer mode (x_host/out_host: upload, run, download, synchronise) and
// device-pointer mode (d_x_ext/d_out_ext + ext_stream: input already on the device, result left there, no copies of either and
// no synchronisation, so further device work can be chained on the stream).  The new latent row is copied to the host cache in
// both modes (asynchronously in device mode; it lands by the caller's next synchronisation).
static int mla_block_impl(const MlaWeights& w,const MlaGeometry& g,const float* x_host,const float* d_x_ext,float* d_out_ext,
                          void* ext_stream,int n_cache,const float* cache_host,float* kv_row_host,float* out_host,
                          char* error,size_t error_capacity){
    auto decline=[&](const char* msg){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",msg);return 0;};
    auto fail=[&](const char* msg){if(error&&error_capacity)std::snprintf(error,error_capacity,"%s",msg);return -1;};
    using namespace strata::kernels;
    const bool host_io=x_host&&out_host;
    const bool dev_io=d_x_ext&&d_out_ext&&ext_stream;
    if((!host_io&&!dev_io)||!cache_host||!kv_row_host||n_cache<1||(n_cache>8192&&!idx_available(w)))return decline("arguments or context out of range");
    if(!w.wq_a_type||!w.wq_b_type||!w.kv_a_type||!w.wo_type||!native_mmvq_supported(w.wq_a_type)||
       !native_mmvq_supported(w.wq_b_type)||!native_mmvq_supported(w.kv_a_type)||!native_mmvq_supported(w.wo_type))
        return decline("projection types not native");
    const int ne=g.n_embd,kvl=g.kv_lora,ql=g.q_lora,nh=g.n_head,hdim=g.head_dim,qd=nh*hdim;
    if(kvl>512||kvl%32||ql%32||ne%32||qd%32||ne>16384||qd>16384)return decline("unsupported geometry");
    float* dk=resident_head_weights(w.wk_b,(size_t)nh*kvl*hdim);
    float* dv=resident_head_weights(w.wv_b,(size_t)nh*hdim*kvl);
    float* dqn=resident_head_weights(w.q_a_norm,(size_t)ql);
    float* dkn=resident_head_weights(w.kv_a_norm,(size_t)kvl);
    if(!dk||!dv||!dqn||!dkn)return decline("weight upload failed");
    Scratch& s=scratch();std::lock_guard<std::mutex> lock(s.mutex);
    DecodeScratch& d=decode_scratch();
    if(!d.alloc(g))return decline("scratch allocation failed");
    // the device latent cache: rows [0, n_cache-1) must already be there; this block appends the new row
    const size_t cn=(size_t)n_cache*kvl;
    auto it=s.resident_caches.find(cache_host);
    bool full=false;
    if(it==s.resident_caches.end()){
        Scratch::ResidentCache entry;size_t cap=1;while(cap<cn)cap*=2;
        if(cudaMalloc(&entry.device,cap*sizeof(float))!=cudaSuccess)return decline("latent cache allocation failed");
        entry.capacity=cap;entry.last_n=0;
        it=s.resident_caches.emplace(cache_host,entry).first;
        full=true;
    }else if(cn>it->second.capacity){
        size_t cap=it->second.capacity?it->second.capacity:1;while(cap<cn)cap*=2;
        float* grown=nullptr;
        if(cudaMalloc(&grown,cap*sizeof(float))!=cudaSuccess)return decline("latent cache growth failed");
        cudaFree(it->second.device);it->second.device=grown;it->second.capacity=cap;
        full=true;
    }else if(it->second.last_n!=n_cache-1){
        full=true;   // a reset, or a depth that does not follow on: the host cache is authoritative
    }
    cudaError_t e=cudaSuccess;
    if(full&&n_cache>1)e=cudaMemcpy(it->second.device,cache_host,(size_t)(n_cache-1)*kvl*sizeof(float),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess)return decline(cudaGetErrorString(e));
    it->second.last_n=n_cache;           // from here the device cache holds this position too
    float* cache_dev=it->second.device;

    cudaStream_t st=host_io?d.stream:(cudaStream_t)ext_stream;void* sv=(void*)st;
    const float* xin=host_io?d.x:d_x_ext;
    float* outp=host_io?d.out:d_out_ext;
    if(host_io)e=cudaMemcpyAsync(d.x,x_host,(size_t)ne*sizeof(float),cudaMemcpyHostToDevice,st);
    if(e!=cudaSuccess)return fail(cudaGetErrorString(e));
    native_quantize_q8_1(xin,d.xq,ne,1,sv);
    native_mmvq(w.wq_a_type,w.wq_a,d.xq,d.qr,ne,ql,1,sv);
    mla_rmsnorm1<<<1,512,0,st>>>(d.qr,dqn,d.qr,ql,MLA_RMS_EPS);
    native_quantize_q8_1(d.qr,d.qrq,ql,1,sv);
    native_mmvq(w.wq_b_type,w.wq_b,d.qrq,d.q,ql,qd,1,sv);
    native_mmvq(w.kv_a_type,w.kv_a,d.xq,d.kv,ne,kvl,1,sv);
    mla_rmsnorm1<<<1,512,0,st>>>(d.kv,dkn,d.kv,kvl,MLA_RMS_EPS);
    cudaMemcpyAsync(cache_dev+(size_t)(n_cache-1)*kvl,d.kv,(size_t)kvl*sizeof(float),cudaMemcpyDeviceToDevice,st);
    cudaMemcpyAsync(kv_row_host,d.kv,(size_t)kvl*sizeof(float),cudaMemcpyDeviceToHost,st);
    const size_t tot_k=(size_t)nh*kvl,tot_v=(size_t)nh*hdim;
    mla_head_matvec_tokens<<<(unsigned)((tot_k+7)/8),256,0,st>>>(dk,d.q,d.qcur,kvl,hdim,nh,tot_k);
    // The sparse indexer: its rows are written for EVERY position (they cannot be rebuilt later), and from IDX_SPARSE_FROM
    // positions on the layer reads only the cells it selects.  Where it cannot (rows missing, scratch), the dense kernel
    // below runs and says so once - beyond 8192 positions there is no dense fallback at all.
    int32_t *d_cells=nullptr,*d_ncells=nullptr;
    bool sparse=false;
    if(idx_available(w)){
        char ie[160]="";
        const int wr=idx_write_device(w,g,cache_host,xin,1,n_cache-1,sv,ie,sizeof(ie));
        if(wr<0)return fail(ie);
        if(n_cache>=IDX_SPARSE_FROM){
            const int rd=wr==1?idx_cells_device(w,g,cache_host,xin,d.qr,1,n_cache-1,&d_cells,&d_ncells,sv,ie,sizeof(ie)):0;
            if(rd<0)return fail(ie);
            sparse=rd==1;
            if(!sparse){
                static bool warned=false;
                if(!warned){warned=true;std::fprintf(stderr,"[mla] sparse indexer unavailable at %d positions (%s): attending densely\n",n_cache,ie);}
                if(n_cache>8192)return decline("no sparse indexer beyond 8192 positions");
            }
        }
    }
    if(sparse)
        mla_latent_attention_cells<<<dim3(1,(unsigned)nh),256,(size_t)IDX_CELL_STRIDE*sizeof(float),st>>>(
            d.qcur,cache_dev,d_cells,d_ncells,kvl,1.0f/std::sqrt((float)hdim),d.attn);
    else
        mla_latent_attention_batch<<<dim3(1,(unsigned)nh),256,(size_t)n_cache*sizeof(float),st>>>(
            d.qcur,cache_dev,n_cache-1,kvl,1.0f/std::sqrt((float)hdim),d.attn);
    mla_head_matvec_tokens<<<(unsigned)((tot_v+7)/8),256,0,st>>>(dv,d.attn,d.v,hdim,kvl,nh,tot_v);
    native_quantize_q8_1(d.v,d.vq,qd,1,sv);
    native_mmvq(w.wo_type,w.wo,d.vq,outp,qd,ne,1,sv);
    if(host_io){
        e=cudaMemcpyAsync(out_host,d.out,(size_t)ne*sizeof(float),cudaMemcpyDeviceToHost,st);
        if(e==cudaSuccess)e=cudaStreamSynchronize(st);
    }
    if(e==cudaSuccess)e=cudaGetLastError();
    if(e!=cudaSuccess)return fail(cudaGetErrorString(e));
    return 1;
}

int mla_block_decode_cuda(const MlaWeights& w,const MlaGeometry& g,const float* x_host,int n_cache,const float* cache_host,
                          float* kv_row_host,float* out_host,char* error,size_t error_capacity){
    return mla_block_impl(w,g,x_host,nullptr,nullptr,nullptr,n_cache,cache_host,kv_row_host,out_host,error,error_capacity);
}

int mla_block_launch_cuda(const MlaWeights& w,const MlaGeometry& g,const float* d_x,int n_cache,const float* cache_host,
                          float* kv_row_host,float* d_out,void* stream,char* error,size_t error_capacity){
    return mla_block_impl(w,g,nullptr,d_x,d_out,stream,n_cache,cache_host,kv_row_host,nullptr,error,error_capacity);
}
} // namespace strata::kernels::glm
