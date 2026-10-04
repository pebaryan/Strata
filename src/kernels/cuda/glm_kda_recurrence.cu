#include <cuda_runtime.h>
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <algorithm>
#include <cmath>
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
// Lazy state: the device copy is authoritative once uploaded; the host buffer goes stale and is only re-read after
// kda_invalidate_state() (the serve loop's reset).  Saves a 4 MB compare and a 4 MB readback per KDA layer per call.
bool g_lazy_state=false;

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
    // One warp per output element: lanes stride the dot product, so the weight row is read coalesced.
    const int ix=(int)(blockIdx.x*(blockDim.x>>5)+(threadIdx.x>>5)), lane=(int)(threadIdx.x&31), total=tokens*rows;
    if (ix>=total) return;
    const int t=ix/rows,r=ix%rows;
    const float* wr=w+(size_t)r*cols; const float* xr=x+(size_t)t*cols;
    float acc=0.0f;
    for (int c=lane;c<cols;c+=32) acc=fmaf(wr[c],xr[c],acc);
    for (int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    if (lane==0) y[(size_t)t*rows+r]=acc;
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
// ---- the device-resident single-token KDA block (decode) -------------------------------------------------------------
//
// One upload of the layer input and one download of its output; everything between stays on the device.  The host path
// (kda_forward) does ~8 synchronous host/GPU round trips per layer for the same work, which is what bounds decode.

// Block reductions in double, exactly as the host does its sums of squares.
__device__ __forceinline__ double block_sum_double(double v, double* red /*[32]*/) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    if (lane == 0) red[warp] = v;
    __syncthreads();
    double total = 0.0;
    if (threadIdx.x == 0) { for (int w = 0; w < (int) (blockDim.x >> 5); ++w) total += red[w]; red[0] = total; }
    __syncthreads();
    total = red[0];
    __syncthreads();
    return total;
}

// y = rms_norm(x) * w   (plain multiply, the model's eps)
__global__ void kda_rmsnorm1(const float* x, const float* w, float* y, int n, float eps) {
    __shared__ double red[32];
    double acc = 0.0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) acc += (double) x[i] * (double) x[i];
    const double ss = block_sum_double(acc, red);
    const float inv = 1.0f / sqrtf((float) (ss / n) + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = x[i] * inv * w[i];
}

// Causal depthwise conv + SiLU for ONE token, for q, k and v at once, with the [3][d_conv-1][d_inner] history shifted in place.
__global__ void kda_conv_step(const float* wq, const float* wk, const float* wv, const float* rq, const float* rk,
                              const float* rv, float* hist, float* cq, float* ck, float* cv, int di, int dc) {
    const int ch = (int) (blockIdx.x * blockDim.x + threadIdx.x), which = (int) blockIdx.y;
    if (ch >= di) return;
    const float* w = which == 0 ? wq : which == 1 ? wk : wv;
    const float* raw = which == 0 ? rq : which == 1 ? rk : rv;
    float* out = which == 0 ? cq : which == 1 ? ck : cv;
    float* h = hist + (size_t) which * (dc - 1) * di;
    float acc = 0.0f;
    for (int k = 0; k < dc; ++k) {
        const float v = (k < dc - 1) ? h[(size_t) k * di + ch] : raw[ch];
        acc += w[(size_t) k * di + ch] * v;
    }
    out[ch] = acc / (1.0f + expf(-acc));
    for (int j = 0; j < dc - 2; ++j) h[(size_t) j * di + ch] = h[(size_t) (j + 1) * di + ch];
    h[(size_t) (dc - 2) * di + ch] = raw[ch];
}

// q,k per-head l2 normalisation (double accumulation like the host): grid (nh, 2)
__global__ void kda_l2norm_qk(const float* cq, const float* ck, float* qn, float* kn, int hd, float eps) {
    __shared__ double red[32];
    const int h = (int) blockIdx.x;
    const float* src = blockIdx.y == 0 ? cq : ck;
    float* dst = blockIdx.y == 0 ? qn : kn;
    double acc = 0.0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += (double) src[(size_t) h * hd + i] * (double) src[(size_t) h * hd + i];
    const double nrm = sqrt(block_sum_double(acc, red));
    const double inv = 1.0 / fmax(nrm, (double) eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) dst[(size_t) h * hd + i] = (float) ((double) src[(size_t) h * hd + i] * inv);
}

// the gated norm: rms over head_dim per head, times o_norm, times sigmoid(gb): grid (nh)
__global__ void kda_outgate(const float* attn, const float* gb, const float* o_norm, float* o, int hd, float eps) {
    __shared__ double red[32];
    const int h = (int) blockIdx.x;
    double acc = 0.0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) acc += (double) attn[(size_t) h * hd + i] * (double) attn[(size_t) h * hd + i];
    const double ss = block_sum_double(acc, red);
    const float inv = 1.0f / sqrtf((float) (ss / hd) + eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        const size_t c = (size_t) h * hd + i;
        o[c] = attn[c] * inv * o_norm[i] * (1.0f / (1.0f + expf(-gb[c])));
    }
}

struct BlockScratch {
    int ne = 0, nh = 0, hd = 0;
    float *x = nullptr, *xn = nullptr, *rq = nullptr, *rk = nullptr, *rv = nullptr, *cq = nullptr, *ck = nullptr, *cv = nullptr;
    float *qn = nullptr, *kn = nullptr, *fa = nullptr, *fb = nullptr, *bp = nullptr, *gg = nullptr, *bb = nullptr;
    float *attn = nullptr, *ga = nullptr, *gb = nullptr, *o = nullptr, *out = nullptr;
    void *xq = nullptr, *oq = nullptr;
    cudaStream_t stream = nullptr;
    struct Conv { float* device = nullptr; size_t count = 0; bool ahead = false; };   // history, keyed by the host buffer
    std::unordered_map<const float*, Conv> conv;
    ~BlockScratch() {
        release();
        for (auto& kv : conv) cudaFree(kv.second.device);
        if (stream) cudaStreamDestroy(stream);
    }
    void release() {
        float** all[] = {&x, &xn, &rq, &rk, &rv, &cq, &ck, &cv, &qn, &kn, &fa, &fb, &bp, &gg, &bb, &attn, &ga, &gb, &o, &out};
        for (float** p : all) { if (*p) cudaFree(*p); *p = nullptr; }
        if (xq) cudaFree(xq);
        if (oq) cudaFree(oq);
        xq = oq = nullptr;
        ne = nh = hd = 0;
    }
    bool alloc(int ne_, int nh_, int hd_) {
        if (ne == ne_ && nh == nh_ && hd == hd_ && x) return true;
        release();
        const size_t di = (size_t) nh_ * hd_;
        auto mk = [&](float*& p, size_t n) { return cudaMalloc(&p, n * sizeof(float)) == cudaSuccess; };
        if (!mk(x, ne_) || !mk(xn, ne_) || !mk(rq, di) || !mk(rk, di) || !mk(rv, di) || !mk(cq, di) || !mk(ck, di) ||
            !mk(cv, di) || !mk(qn, di) || !mk(kn, di) || !mk(fa, hd_) || !mk(fb, di) || !mk(bp, nh_) || !mk(gg, di) ||
            !mk(bb, nh_) || !mk(attn, di) || !mk(ga, hd_) || !mk(gb, di) || !mk(o, di) || !mk(out, ne_)) { release(); return false; }
        if (cudaMalloc(&xq, strata::kernels::native_q8_1_bytes(ne_, 1)) != cudaSuccess ||
            cudaMalloc(&oq, strata::kernels::native_q8_1_bytes((int) di, 1)) != cudaSuccess) { release(); return false; }
        if (!stream && cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking) != cudaSuccess) { release(); return false; }
        ne = ne_; nh = nh_; hd = hd_;
        return true;
    }
};
BlockScratch& block_scratch() { static BlockScratch b; return b; }
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
        } else if (!g_lazy_state && std::memcmp(it->second.host_shadow.data(),state,st*sizeof(float))!=0) {
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
    if (state && !g_lazy_state) {
        if ((e=cudaMemcpy(state,state_device,st*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess)
            return fail(cudaGetErrorString(e));
        s.resident_states.find(state)->second.host_shadow.assign(state,state+st);
    }
    if (error && error_capacity) error[0]='\0';
    return true;
}

bool kda_rows_cuda(const float* weight,const float* x,int tokens,int rows,int cols,float* y,char* error,size_t error_capacity) {
    auto fail=[&](const char* msg) { if(error&&error_capacity) std::snprintf(error,error_capacity,"%s",msg); return false; };
    if(!weight||!x||!y||tokens<1||rows<1||cols<1) return fail("invalid KDA rows arguments");
    Scratch& s=scratch(); std::lock_guard<std::mutex> lock(s.mutex);
    const size_t xn=(size_t)tokens*cols,yn=(size_t)tokens*rows;
    if(!s.reserve(s.x,s.x_cap,xn)||!s.reserve(s.gate_out,s.gate_cap,yn)) return fail("KDA rows scratch allocation failed");
    float* dw=nullptr;
    if(!resident_weight(s,weight,(size_t)rows*cols,dw)) return fail("KDA rows weight upload failed");
    cudaError_t e=cudaMemcpy(s.x,x,xn*sizeof(float),cudaMemcpyHostToDevice);
    if(e!=cudaSuccess) return fail(cudaGetErrorString(e));
    kda_rows<<<(unsigned)(((size_t)tokens*rows+7)/8),256>>>(dw,s.x,s.gate_out,tokens,rows,cols);
    if((e=cudaGetLastError())!=cudaSuccess||(e=cudaDeviceSynchronize())!=cudaSuccess) return fail(cudaGetErrorString(e));
    if((e=cudaMemcpy(y,s.gate_out,yn*sizeof(float),cudaMemcpyDeviceToHost))!=cudaSuccess) return fail(cudaGetErrorString(e));
    if(error&&error_capacity) error[0]='\0';
    return true;
}

void kda_set_lazy_state(bool enabled) { std::lock_guard<std::mutex> lock(scratch().mutex); g_lazy_state=enabled; }
bool kda_lazy_state_enabled() { return g_lazy_state; }

void kda_invalidate_state() {
    Scratch& s=scratch(); std::lock_guard<std::mutex> lock(s.mutex);
    for (auto& kv:s.resident_states) cudaFree(kv.second.device);
    s.resident_states.clear();
    BlockScratch& b=block_scratch();
    for (auto& kv:b.conv) cudaFree(kv.second.device);
    b.conv.clear();
}

// The conv histories live on the device while the decode block runs (the host copy goes stale).  A host-side conv (prompt
// chunks, any fallback) must first pull the device copy back, and afterwards the device copy is stale and is dropped.
void kda_conv_sync_host(float* conv_state, size_t count) {
    if (!conv_state) return;
    BlockScratch& b=block_scratch();
    std::lock_guard<std::mutex> lock(scratch().mutex);
    auto it=b.conv.find(conv_state);
    if (it==b.conv.end() || !it->second.ahead || it->second.count!=count) return;
    cudaMemcpy(conv_state,it->second.device,count*sizeof(float),cudaMemcpyDeviceToHost);
    it->second.ahead=false;
}
void kda_conv_host_modified(const float* conv_state) {
    if (!conv_state) return;
    BlockScratch& b=block_scratch();
    std::lock_guard<std::mutex> lock(scratch().mutex);
    auto it=b.conv.find(conv_state);
    if (it==b.conv.end()) return;
    cudaFree(it->second.device);
    b.conv.erase(it);
}

// The one implementation behind both entry points.  Host-pointer mode (x_host/out_host): upload the input, run, download and
// synchronise.  Device-pointer mode (d_x_ext/d_out_ext + ext_stream): the input is already on the device and the result stays
// there; nothing is copied and nothing is synchronised, so the caller can chain further device work on the same stream.
// Returns 1 done, 0 declined (nothing touched: the caller runs the host path), -1 failed after work began.
static int kda_block_impl(const KdaWeights& w,const KdaGeometry& g,const float* x_host,float* out_host,const float* d_x_ext,
                          float* d_out_ext,void* ext_stream,float* state,float* conv_state,char* error,size_t error_capacity) {
    auto decline=[&](const char* msg){ if(error&&error_capacity) std::snprintf(error,error_capacity,"%s",msg); return 0; };
    auto fail=[&](const char* msg){ if(error&&error_capacity) std::snprintf(error,error_capacity,"%s",msg); return -1; };
    if (!g_lazy_state) return decline("device block needs lazy resident state (serve mode)");
    const bool host_io = x_host && out_host;
    const bool dev_io = d_x_ext && d_out_ext && ext_stream;
    if ((!host_io && !dev_io) || !state || !conv_state) return decline("null argument");
    using namespace strata::kernels;
    if (!w.wq_type||!w.wk_type||!w.wv_type||!w.wo_type||!native_mmvq_supported(w.wq_type)||!native_mmvq_supported(w.wk_type)||
        !native_mmvq_supported(w.wv_type)||!native_mmvq_supported(w.wo_type)) return decline("projection types not native");
    const int ne=g.n_embd,nh=g.nh,hd=g.hd,dc=g.d_conv,di=nh*hd;
    if (dc<3||dc>8||hd>1024||ne%32||di%32||ne>16384||di>16384) return decline("unsupported geometry");
    Scratch& s=scratch(); std::lock_guard<std::mutex> lock(s.mutex);
    BlockScratch& b=block_scratch();
    if (!b.alloc(ne,nh,hd)) return decline("block scratch allocation failed");
    float *d_an=nullptr,*d_cqw=nullptr,*d_ckw=nullptr,*d_cvw=nullptr,*d_fa=nullptr,*d_fb=nullptr,*d_beta=nullptr,*d_a=nullptr,
          *d_bias=nullptr,*d_ga=nullptr,*d_gb=nullptr,*d_on=nullptr;
    if(!resident_weight(s,w.attn_norm,(size_t)ne,d_an)||!resident_weight(s,w.conv_q,(size_t)dc*di,d_cqw)||
       !resident_weight(s,w.conv_k,(size_t)dc*di,d_ckw)||!resident_weight(s,w.conv_v,(size_t)dc*di,d_cvw)||
       !resident_weight(s,w.ssm_f_a,(size_t)hd*ne,d_fa)||!resident_weight(s,w.ssm_f_b,(size_t)di*hd,d_fb)||
       !resident_weight(s,w.ssm_beta,(size_t)nh*ne,d_beta)||!resident_weight(s,w.ssm_a,(size_t)nh,d_a)||
       !resident_weight(s,w.dt_bias,(size_t)di,d_bias)||!resident_weight(s,w.ssm_g_a,(size_t)hd*ne,d_ga)||
       !resident_weight(s,w.ssm_g_b,(size_t)di*hd,d_gb)||!resident_weight(s,w.o_norm,(size_t)hd,d_on))
        return decline("weight upload failed");
    // recurrence state: the same resident entry the prompt path's kernel uses
    const size_t st=(size_t)nh*hd*hd;
    float* state_dev=nullptr;
    {
        auto it=s.resident_states.find(state);
        if (it==s.resident_states.end()||it->second.count!=st) {
            if (it!=s.resident_states.end()) { cudaFree(it->second.device); s.resident_states.erase(it); }
            Scratch::ResidentState entry; entry.count=st;
            if (cudaMalloc(&entry.device,st*sizeof(float))!=cudaSuccess) return decline("state allocation failed");
            if (cudaMemcpy(entry.device,state,st*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess) { cudaFree(entry.device); return decline("state upload failed"); }
            it=s.resident_states.emplace(state,std::move(entry)).first;
        }
        state_dev=it->second.device;
    }
    // conv history: upload from the host copy the first time (or after a host-side conv invalidated it)
    const size_t conv_count=(size_t)3*(dc-1)*di;
    auto cit=b.conv.find(conv_state);
    if (cit==b.conv.end()||cit->second.count!=conv_count) {
        if (cit!=b.conv.end()) { cudaFree(cit->second.device); b.conv.erase(cit); }
        BlockScratch::Conv entry; entry.count=conv_count;
        if (cudaMalloc(&entry.device,conv_count*sizeof(float))!=cudaSuccess) return decline("conv history allocation failed");
        if (cudaMemcpy(entry.device,conv_state,conv_count*sizeof(float),cudaMemcpyHostToDevice)!=cudaSuccess) { cudaFree(entry.device); return decline("conv history upload failed"); }
        cit=b.conv.emplace(conv_state,entry).first;
    }
    float* hist=cit->second.device;
    cit->second.ahead=true;        // from here the device copy is the live one

    cudaStream_t st_=host_io ? b.stream : (cudaStream_t)ext_stream; void* sv=(void*)st_;
    const float* xin = host_io ? b.x : d_x_ext;
    float* outp = host_io ? b.out : d_out_ext;
    cudaError_t e=cudaSuccess;
    if (host_io) e=cudaMemcpyAsync(b.x,x_host,(size_t)ne*sizeof(float),cudaMemcpyHostToDevice,st_);
    if (e!=cudaSuccess) return fail(cudaGetErrorString(e));
    kda_rmsnorm1<<<1,1024,0,st_>>>(xin,d_an,b.xn,ne,KDA_RMS_EPS);
    native_quantize_q8_1(b.xn,b.xq,ne,1,sv);
    native_mmvq(w.wq_type,w.wq,b.xq,b.rq,ne,di,1,sv);
    native_mmvq(w.wk_type,w.wk,b.xq,b.rk,ne,di,1,sv);
    native_mmvq(w.wv_type,w.wv,b.xq,b.rv,ne,di,1,sv);
    kda_conv_step<<<dim3((di+255)/256,3),256,0,st_>>>(d_cqw,d_ckw,d_cvw,b.rq,b.rk,b.rv,hist,b.cq,b.ck,b.cv,di,dc);
    kda_l2norm_qk<<<dim3(nh,2),128,0,st_>>>(b.cq,b.ck,b.qn,b.kn,hd,KDA_L2_EPS);
    // gates, from the normed input
    kda_rows<<<(hd+7)/8,256,0,st_>>>(d_fa,b.xn,b.fa,1,hd,ne);
    kda_rows<<<(di+7)/8,256,0,st_>>>(d_fb,b.fa,b.fb,1,di,hd);
    kda_rows<<<(nh+7)/8,256,0,st_>>>(d_beta,b.xn,b.bp,1,nh,ne);
    kda_apply_gates<<<(di+255)/256,256,0,st_>>>(b.fb,b.bp,d_a,d_bias,1,nh,hd,b.gg,b.bb);
    kda_recur<<<nh,256,(size_t)2*hd*sizeof(float),st_>>>(b.qn,b.kn,b.cv,b.gg,b.bb,1,nh,hd,state_dev,b.attn);
    // output gate, then wo
    kda_rows<<<(hd+7)/8,256,0,st_>>>(d_ga,b.xn,b.ga,1,hd,ne);
    kda_rows<<<(di+7)/8,256,0,st_>>>(d_gb,b.ga,b.gb,1,di,hd);
    kda_outgate<<<nh,128,0,st_>>>(b.attn,b.gb,d_on,b.o,hd,KDA_RMS_EPS);
    native_quantize_q8_1(b.o,b.oq,di,1,sv);
    native_mmvq(w.wo_type,w.wo,b.oq,outp,di,ne,1,sv);
    if (host_io) {
        e=cudaMemcpyAsync(out_host,b.out,(size_t)ne*sizeof(float),cudaMemcpyDeviceToHost,st_);
        if (e==cudaSuccess) e=cudaStreamSynchronize(st_);
    }
    if (e==cudaSuccess) e=cudaGetLastError();
    if (e!=cudaSuccess) return fail(cudaGetErrorString(e));
    return 1;
}

int kda_block_decode_cuda(const KdaWeights& w,const KdaGeometry& g,const float* x_host,float* out_host,float* state,
                          float* conv_state,char* error,size_t error_capacity) {
    return kda_block_impl(w,g,x_host,out_host,nullptr,nullptr,nullptr,state,conv_state,error,error_capacity);
}

int kda_block_launch_cuda(const KdaWeights& w,const KdaGeometry& g,const float* d_x,float* d_out,float* state,
                          float* conv_state,void* stream,char* error,size_t error_capacity) {
    return kda_block_impl(w,g,nullptr,nullptr,d_x,d_out,stream,state,conv_state,error,error_capacity);
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
    kda_rows<<<(tokens*hd+7)/8,256>>>(d_fa_w,s.x,s.fa,tokens,hd,n_embd);
    kda_rows<<<(tokens*di+7)/8,256>>>(d_fb_w,s.fa,s.fb,tokens,di,hd);
    kda_rows<<<(tokens*nh+7)/8,256>>>(d_beta_w,s.x,s.beta_pre,tokens,nh,n_embd);
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
