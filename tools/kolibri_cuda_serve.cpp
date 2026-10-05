// CUDA graph Kolibri-1 serving path. Dense work and attention stay on device; each layer is
// split at its dynamic expert selection so Strata's resident expert-row cache can service it.
#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <functional>
#include <limits>
#include <map>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_row_cache.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm_expert_device.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/kolibri_cuda.hpp"
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace C = strata::core;
namespace K = strata::kernels;
namespace KC = strata::kernels::kolibri_cuda;
namespace KCPU = strata::kernels::cpu;

namespace {
constexpr int N_EMBD=2560, N_HEAD=48, N_KV=4, HD=128, FF=512, N_EXPERT=384, N_USED=6;

struct Layer {
    const C::WeightRef *q{}, *k{}, *v{}, *o{}, *sg{}, *su{}, *sd{};
    const float *an{}, *pan{}, *fn{}, *pfn{}, *qn{}, *kn{}, *bias{};
    const void* router{};
    float *kc{}, *vc{};                 // fp32 KV cache (used when the q4 path is off)
    unsigned char *kq{}, *vq{};         // quantized KV cache (q4_0 or q8_0 blocks)
    float *ks{}, *vs{};                 // the current cell, before quantization
    cudaGraphExec_t pre{}, post{};
    KCPU::NativeFmt fmt{};
    std::function<bool()> pre_fn, post_fn;   // the same op sequence the graphs encode, runnable
                                             // directly for the per-op attribution mode
};

struct CacheRuntime {
    C::ExpertRowCache cache;
    cudaStream_t stream{};
    float *x{}, *gate{}, *up{}, *out{}, *accum{};
    void *xq{}, *hq{};
    uint8_t* row_stage[N_USED]{};   // pinned, one per selected expert: the async H2D reads from these
    std::vector<void*> pinned;          // per layer: device base holding all 384 rows, nullptr if not pinned
    std::vector<size_t> pinned_stride;  // row stride for that layer's pinned block
    explicit CacheRuntime(size_t bytes) : cache([&] {
        C::ExpertRowCacheConfig c; c.budget_bytes=bytes; c.evict_ctx=this;
        c.on_evict=[](void*,const C::ExpertRowKey&,const C::ExpertRowEntry& e){ if(e.dev) cudaFree(e.dev); };
        return c;
    }()) {
        cudaStreamCreate(&stream);
        cudaMalloc((void**)&x,N_EMBD*4); cudaMalloc((void**)&gate,FF*4); cudaMalloc((void**)&up,FF*4);
        cudaMalloc((void**)&out,N_EMBD*4); cudaMalloc((void**)&accum,N_EMBD*4);
        cudaMalloc(&xq,K::native_q8_1_bytes(N_EMBD)); cudaMalloc(&hq,K::native_q8_1_bytes(FF));
        for(int i=0;i<N_USED;++i) cudaHostAlloc((void**)&row_stage[i],4u<<20,cudaHostAllocDefault);
    }
    ~CacheRuntime(){ cache.clear(); cudaFree(x);cudaFree(gate);cudaFree(up);cudaFree(out);cudaFree(accum);cudaFree(xq);cudaFree(hq);cudaStreamDestroy(stream); }
};

bool check(cudaError_t e, const char* where, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(where) + ": " + cudaGetErrorString(e); return false;
}

// ---- per-op attribution (STRATA_KOLIBRI_PROFILE=1).  Events cannot be read back from inside a
// capture, so ops are timed only when the graph path is off (STRATA_KOLIBRI_NO_GRAPH=1); in capture
// mode op_run() is a pass-through and the op is recorded into the graph as usual.
bool g_prof = false, g_capturing = false;
cudaEvent_t g_ev0 = nullptr, g_ev1 = nullptr;
struct OpStat { double ms = 0; long n = 0; };
std::map<std::string, OpStat> g_ops;

template <typename F>
void op_run(const char* name, F&& f) {
    if (!g_prof || g_capturing) { f(); return; }
    cudaEventRecord(g_ev0, 0);
    f();
    cudaEventRecord(g_ev1, 0);
    cudaEventSynchronize(g_ev1);
    float t = 0;
    cudaEventElapsedTime(&t, g_ev0, g_ev1);
    auto& s = g_ops[name]; s.ms += t; ++s.n;
}

void op_report(const char* phase, int tokens) {
    if (!g_prof || tokens <= 0) return;
    std::printf("  %s ops per token:", phase);
    for (const auto& kv : g_ops)
        std::printf(" %s %.2f", kv.first.c_str(), kv.second.ms / tokens);
    std::printf(" ms\n");
    g_ops.clear();
}

const C::WeightRef* need(const C::WeightTable& t, const std::string& n, std::string& err) {
    const C::WeightRef* r=t.find(n);
    if(!r || (!r->data && !r->native_data)) err="missing runtime tensor "+n;
    return r;
}

bool mmvq(const C::WeightRef* w, const void* xq, float* y, void* stream, std::string& err) {
    if(!w || !w->native_data || w->native_type < 0) { err="projection has no native GGUF weights"; return false; }
    try { K::native_mmvq(w->native_type,w->native_data,xq,y,(int)w->ne0,(int)w->ne1,1,stream); }
    catch(const std::exception& e){ err=e.what(); return false; }
    return true;
}

bool load_types(const std::string& pack, std::vector<std::pair<int,int>>& type, std::string& err) {
    std::FILE* f=std::fopen((pack+"/native_experts.txt").c_str(),"rb");
    if(!f){ err="cannot open native_experts.txt"; return false; }
    char line[1024];
    while(std::fgets(line,sizeof line,f)) {
        if(line[0]=='#') continue;
        int l=-1,g=0,d=0; if(std::sscanf(line,"%d %d %d",&l,&g,&d)==3 && l>=0 && l<(int)type.size()) type[l]={g,d};
    }
    std::fclose(f);
    for(size_t i=0;i<type.size();++i) if(!type[i].first){ err="missing expert type for layer "+std::to_string(i); return false; }
    return true;
}

// ---- assembling an expert row from the GGUF, ourselves.
//
// native_experts.txt gives, per layer, the file offsets of the three source tensors.  A row is
// [gate 512x2560 | up 512x2560 | down 2560x512] and the three slices are NOT adjacent in the GGUF,
// so a row has to be assembled.  FileExpertSource does that in gguf mode through its staging pool,
// but that pool is sized from layer_blob_bytes_ which a pack with no experts.bin never fills:
// stage_blob_ stays 0, the pool allocates zero-byte buffers, and copy_from_files writes 2.55 MB into
// them.  It also runs on every blob() call, cache hit or miss, so a warm token copied ~765 MB per
// token for rows already resident.  This assembler reads the three ranges straight from a read-only
// mmap, only when a row is actually missing.
struct RowGeom { size_t gate_off = 0, up_off = 0, down_off = 0, blob_bytes = 0; };
std::vector<RowGeom> g_row;
const uint8_t* g_gguf = nullptr;
int g_gguf_fd = -1;

bool load_row_geom(const std::string& pack, const std::string& model, int n_layers, std::string& err) {
    const int fd = ::open(model.c_str(), O_RDONLY);
    if (fd < 0) { err = "open " + model; return false; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { err = "stat " + model; ::close(fd); return false; }
    void* view = mmap(nullptr, (size_t) st.st_size, PROT_READ, MAP_SHARED, fd, 0);
    if (view == MAP_FAILED) { err = "mmap " + model; ::close(fd); return false; }
    g_gguf = (const uint8_t*) view;
    g_gguf_fd = fd;      // kept open: rows are read with pread, not by faulting the mapping

    std::FILE* f = std::fopen((pack + "/native_experts.txt").c_str(), "rb");
    if (f == nullptr) { err = "cannot open native_experts.txt"; return false; }
    g_row.assign((size_t) n_layers, RowGeom{});
    char line[1024];
    while (std::fgets(line, sizeof line, f)) {
        if (line[0] == '#') continue;
        long l = -1; int gt = 0, dt = 0;
        unsigned long long off = 0, blob = 0, go = 0, uo = 0, dow = 0;
        if (std::sscanf(line, "%ld %d %d %llu %llu %llu %llu %llu", &l, &gt, &dt, &off, &blob, &go, &uo, &dow) != 8) continue;
        if (l < 0 || l >= n_layers) continue;
        RowGeom& g = g_row[(size_t) l];
        g.gate_off = (size_t) go; g.up_off = (size_t) uo; g.down_off = (size_t) dow; g.blob_bytes = (size_t) blob;
    }
    std::fclose(f);
    for (int i = 0; i < n_layers; ++i)
        if (g_row[(size_t) i].blob_bytes == 0) { err = "native_experts.txt: no row for layer " + std::to_string(i); return false; }
    return true;
}

// Assemble row `e` of `layer` into `dst`: three preads, one per slice.
//
// This used to memcpy out of the mmap.  On a miss that meant ~640 minor page faults, each a 4 KB
// synchronous read of a DRAM-less SATA SSD, measured at ~70 ms per 2.4 MB row (34 MB/s) - and a
// generation that misses ~1.5 rows per token then runs at 135 ms/token no matter how fast the hits
// are.  pread hands each slice to the kernel as one large request; the mapping stays for the
// non-hot path and for validation.
inline bool read_row(int layer, int e, size_t gu, size_t down, uint8_t* dst, std::string& err) {
    const RowGeom& g = g_row[(size_t) layer];
    const off_t o0 = (off_t) (g.gate_off + (size_t) e * gu);
    const off_t o1 = (off_t) (g.up_off + (size_t) e * gu);
    const off_t o2 = (off_t) (g.down_off + (size_t) e * down);
    if (::pread(g_gguf_fd, dst, gu, o0) != (ssize_t) gu) { err = "pread gate"; return false; }
    if (::pread(g_gguf_fd, dst + gu, gu, o1) != (ssize_t) gu) { err = "pread up"; return false; }
    if (::pread(g_gguf_fd, dst + 2 * gu, down, o2) != (ssize_t) down) { err = "pread down"; return false; }
    return true;
}

bool run_experts(CacheRuntime& rt, int layer, const Layer& l,
                 const int32_t* ids, const float* weights, const float* x, float* out, std::string& err) {
    size_t gu=K::native_mmvq_weight_bytes(l.fmt.gu_type,N_EMBD,FF);
    size_t down=K::native_mmvq_weight_bytes(l.fmt.d_type,FF,N_EMBD);
    size_t up_off=gu, down_off=2*gu, bytes=down_off+down;
    if(bytes!=l.fmt.bytes){ err="expert native layout mismatch"; return false; }
    std::vector<const uint8_t*> rows(N_USED);
    std::vector<void*> temporary;
    bool rows_ok=true;
    op_run("expert_rows", [&]{
    for(int i=0;i<N_USED;++i) {
        if(rt.pinned[(size_t)layer]) {   // whole layer resident: the address is arithmetic, nothing else
            rows[i]=(const uint8_t*)rt.pinned[(size_t)layer]+(size_t)ids[i]*rt.pinned_stride[(size_t)layer];
            continue;
        }
        C::ExpertRowKey key{layer,ids[i]};
        auto state=rt.cache.lookup(key,bytes);          // cache first: a hit needs no host bytes at all
        void* dev=nullptr;
        if(state==C::ExpertRowState::resident) dev=rt.cache.find(key)->dev;
        else {
            if(!read_row(layer,ids[i],gu,down,rt.row_stage[i],err)){ rows_ok=false; return; }   // miss only
            // Reclaim before failing: a miss that cannot be allocated used to end the request AND the
            // engine ("expert allocation: out of memory" mid-prefill) once the LRU budget outgrew the
            // VRAM left after the pinned layers, the KV cache and the trunk.  Evict first, and only
            // report failure if the device is genuinely full.
            if(cudaMalloc(&dev,bytes)!=cudaSuccess){
                cudaGetLastError();
                // Give the cache's device memory back to this row: a miss that could not be allocated
                // used to end the request AND the engine ("expert allocation: out of memory" during a
                // prefill) as soon as the LRU outgrew the VRAM left after the pinned layers, the KV
                // cache and the trunk.  trim_one() exists for exactly this caller.
                size_t free_b=0,total_b=0;int trimmed=0;
                const size_t want=bytes+(size_t)(256u<<20);
                while(cudaMemGetInfo(&free_b,&total_b)==cudaSuccess&&free_b<want&&rt.cache.trim_one())++trimmed;
                cudaMalloc(&dev,bytes);
                if(dev==nullptr){ std::fprintf(stderr,"expert allocation: out of memory (%.0f MiB free after trimming %d cached rows)\n",(double)free_b/1048576.0,trimmed); err="expert allocation: out of memory"; rows_ok=false; return; }
                if(trimmed>0)std::fprintf(stderr,"expert allocation: trimmed %d cached rows to make room\n",trimmed);
            }
            if(!check(cudaMemcpyAsync(dev,rt.row_stage[i],bytes,cudaMemcpyHostToDevice,rt.stream),"expert upload",err)){ cudaFree(dev); rows_ok=false; return; }
            if(state==C::ExpertRowState::needs_upload){ rt.cache.insert(key,{dev,bytes}); dev=rt.cache.find(key)->dev; }
            else temporary.push_back(dev);
        }
        rows[i]=(const uint8_t*)dev;
    }
    });
    if(!rows_ok)return false;
    bool ok=check(cudaMemcpyAsync(rt.x,x,N_EMBD*4,cudaMemcpyHostToDevice,rt.stream),"expert activation upload",err) &&
            check(cudaMemsetAsync(rt.accum,0,N_EMBD*4,rt.stream),"expert accumulator",err);
    // STRATA_KOLIBRI_SKIP_KERNELS=1 keeps the row lookups/uploads and the final sync but drops the
    // mmvq chain, so the two runs differ only by the kernels' cost (wrong logits, timing only).
    static const bool skip_kernels=std::getenv("STRATA_KOLIBRI_SKIP_KERNELS")!=nullptr;
    if(ok && !skip_kernels) {
        op_run("x_quant", [&]{ K::native_quantize_q8_1(rt.x,rt.xq,N_EMBD,1,rt.stream); });
        op_run("x_mmvq", [&]{
        for(int i=0;i<N_USED;++i) {
            K::native_mmvq(l.fmt.gu_type,rows[i],rt.xq,rt.gate,N_EMBD,FF,1,rt.stream);
            K::native_mmvq(l.fmt.gu_type,rows[i]+up_off,rt.xq,rt.up,N_EMBD,FF,1,rt.stream);
            K::native_swiglu_quantize_q8_1(rt.gate,rt.up,rt.hq,FF,1,rt.stream);
            K::native_mmvq(l.fmt.d_type,rows[i]+down_off,rt.hq,rt.out,FF,N_EMBD,1,rt.stream);
            K::scaled_add_inplace(rt.accum,rt.out,N_EMBD,weights[i],rt.stream);
        }
        });
        op_run("x_d2h", [&]{
        ok=check(cudaMemcpyAsync(out,rt.accum,N_EMBD*4,cudaMemcpyDeviceToHost,rt.stream),"expert result copy",err) &&
           check(cudaStreamSynchronize(rt.stream),"expert synchronize",err);
        });
    }
    for(void* p:temporary) cudaFree(p);
    return ok;
}

int sample_logits(std::vector<float>& logits, const std::vector<int>& history, float temp, int top_k,
                  float top_p, float min_p, float repeat, float freq, float present, int last_n,
                  std::mt19937& rng) {
    int best=(int)(std::max_element(logits.begin(),logits.end())-logits.begin());
    if(temp<=0) return best;
    std::vector<int> count(logits.size());
    size_t begin=history.size()>(size_t)std::max(0,last_n)?history.size()-std::max(0,last_n):0;
    for(size_t i=begin;i<history.size();++i) if(history[i]>=0 && history[i]<(int)count.size()) ++count[history[i]];
    struct P{int id; double p;}; std::vector<P> c; c.reserve(logits.size()); double high=-INFINITY;
    for(int i=0;i<(int)logits.size();++i){ double v=logits[i]; if(count[i]){ if(repeat>0&&repeat!=1) v=v>=0?v/repeat:v*repeat; v-=freq*count[i]+present; } v/=temp; c.push_back({i,v}); high=std::max(high,v); }
    for(auto& p:c)p.p=std::exp(p.p-high);
    std::sort(c.begin(),c.end(),[](auto&a,auto&b){return a.p>b.p||(a.p==b.p&&a.id<b.id);});
    if(top_k>0&&top_k<(int)c.size())c.resize(top_k);
    if(min_p>0&&!c.empty()){double floor=c[0].p*min_p;c.erase(std::remove_if(c.begin(),c.end(),[&](auto&p){return p.p<floor;}),c.end());}
    double sum=0;for(auto&p:c)sum+=p.p;
    if(top_p>0&&top_p<1&&sum>0){double kept=0;size_t n=0;do{kept+=c[n++].p;}while(n<c.size()&&kept/sum<top_p);c.resize(n);sum=kept;}
    std::uniform_real_distribution<double>d(0,sum);double at=d(rng);for(auto&p:c)if((at-=p.p)<=0)return p.id;return c.back().id;
}
}

int main(int argc,char** argv){
    std::string pack,model,gate_prefix,gate_tokens,prefill_tokens; int max_context=4096,gate_gen=0,pin_layers=0; int kv_bits=32; bool serve=false;   // 32 = fp32 KV, 8 = q8_0, 4 = q4_0
    for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--serve")serve=true; else if(a=="--pack"&&i+1<argc)pack=argv[++i]; else if((a=="--native"||a=="--model")&&i+1<argc)model=argv[++i]; else if(a=="--max-context"&&i+1<argc)max_context=std::atoi(argv[++i]); else if(a=="--gate"&&i+2<argc){gate_prefix=argv[++i];gate_tokens=argv[++i];} else if(a=="--gen"&&i+1<argc)gate_gen=std::atoi(argv[++i]); else if(a=="--prefill"&&i+1<argc)prefill_tokens=argv[++i]; else if(a=="--pin-layers"&&i+1<argc)pin_layers=std::atoi(argv[++i]); else if(a=="--kv-q4")kv_bits=4; else if(a=="--kv-q8")kv_bits=8;}
    if(pack.empty()||model.empty()){std::fprintf(stderr,"usage: strata-kolibri --pack DIR --native MODEL [--serve] [--max-context N]\n       strata-kolibri --pack DIR --native MODEL --gate ORACLE-PREFIX t1,t2,... [--gen N]\n");return 2;}
    std::string err;
    strata::GgufModel gm=strata::GgufModel::open(model); strata::Kolibri1Geometry geo;
    if(!(err=strata::check_kolibri1_architecture(gm.meta(),geo,nullptr)).empty()){std::fprintf(stderr,"guard: %s\n",err.c_str());return 1;}
    max_context=std::min<int64_t>(max_context,geo.context_length); const int n_layers=(int)geo.block_count;
    const auto* ot=gm.find("output.weight"); const int vocab=ot?(int)ot->shape[1]:0; int eos=-1;
    if(const auto* e=gm.meta().get("tokenizer.ggml.eos_token_id"))eos=(int)e->u;

    std::vector<std::string> shards{model}; std::set<std::string> served;
    if(!C::NativeDense::served_names(shards,false,served,err)){std::fprintf(stderr,"served names: %s\n",err.c_str());return 1;}
    served.insert("token_embd.weight");
    served.insert("output.weight");
    for (const auto& tensor : gm.shard(0).tensors())
        if (tensor.name.find("_exps.weight") != std::string::npos) served.insert(tensor.name);
    C::WeightTable table; uint64_t arena_bytes=0;
    if(!table.pool_bytes(pack,arena_bytes,err,&served)){std::fprintf(stderr,"pool: %s\n",err.c_str());return 1;}
    void* arena=nullptr; if(!check(cudaMallocManaged(&arena,arena_bytes,cudaMemAttachGlobal),"weight arena",err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    if(!table.load(pack,arena,arena_bytes,err,&served)){std::fprintf(stderr,"weights: %s\n",err.c_str());return 1;}
    C::NativeDense dense; if(!dense.load(shards,table,err)){std::fprintf(stderr,"native dense: %s\n",err.c_str());return 1;}
    C::NativeEmbed embed; if(!embed.load(shards,N_EMBD,vocab,err)){std::fprintf(stderr,"embedding: %s\n",err.c_str());return 1;}
    C::NativeHead head; if(!head.load(shards,N_EMBD,vocab,err)){std::fprintf(stderr,"head: %s\n",err.c_str());return 1;}
    if(!KCPU::expert_layout_load(pack,n_layers,N_EXPERT,err,N_EMBD,FF)){std::fprintf(stderr,"expert layout: %s\n",err.c_str());return 1;}
    if(!load_row_geom(pack,model,n_layers,err)){std::fprintf(stderr,"row geometry: %s\n",err.c_str());return 1;}
    C::FileExpertSource src; src.set_gguf(model); if(!src.open(pack,n_layers,N_EXPERT,err)){std::fprintf(stderr,"experts: %s\n",err.c_str());return 1;}
    std::vector<std::pair<int,int>> types(n_layers); if(!load_types(pack,types,err)){std::fprintf(stderr,"types: %s\n",err.c_str());return 1;}

    cudaStream_t stream{}; check(cudaStreamCreate(&stream),"stream",err);
    float *x{},*norm{},*q{},*k{},*v{},*att{},*proj{},*gate{},*up{},*shared{},*routed{},*logits_dev{};
    int *pos_dev{}; int32_t *ids_dev{},*ids_host{}; float *weights_dev{},*weights_host{},*x_host{},*routed_host{},*raw_dev{};
    void *q8a{},*q8b{};
    auto alloc=[&](void**p,size_t n,const char*w){if(!check(cudaMalloc(p,n),w,err)){std::fprintf(stderr,"%s\n",err.c_str());std::exit(1);}};
    alloc((void**)&x,N_EMBD*4,"x");alloc((void**)&norm,N_EMBD*4,"norm");alloc((void**)&q,N_HEAD*HD*4,"q");alloc((void**)&k,N_KV*HD*4,"k");alloc((void**)&v,N_KV*HD*4,"v");alloc((void**)&att,N_HEAD*HD*4,"attention");alloc((void**)&proj,N_EMBD*4,"projection");alloc((void**)&gate,FF*4,"shared gate");alloc((void**)&up,FF*4,"shared up");alloc((void**)&shared,N_EMBD*4,"shared out");alloc((void**)&routed,N_EMBD*4,"routed out");alloc((void**)&logits_dev,(size_t)vocab*4,"logits");alloc((void**)&pos_dev,4,"position");alloc((void**)&ids_dev,N_USED*4,"route ids");alloc((void**)&weights_dev,N_USED*4,"route weights");alloc((void**)&raw_dev,N_EXPERT*4,"router logits");alloc(&q8a,K::native_q8_1_bytes(N_HEAD*HD),"q8 scratch a");alloc(&q8b,K::native_q8_1_bytes(N_EMBD),"q8 scratch b");
    cudaHostAlloc(&ids_host,N_USED*4,cudaHostAllocDefault);cudaHostAlloc(&weights_host,N_USED*4,cudaHostAllocDefault);cudaHostAlloc(&x_host,N_EMBD*4,cudaHostAllocDefault);cudaHostAlloc(&routed_host,N_EMBD*4,cudaHostAllocDefault);

    const bool prof=std::getenv("STRATA_KOLIBRI_PROFILE")!=nullptr;
    const bool no_graph=std::getenv("STRATA_KOLIBRI_NO_GRAPH")!=nullptr;
    g_prof=prof;if(prof){cudaEventCreate(&g_ev0);cudaEventCreate(&g_ev1);}
    std::vector<Layer> layers(n_layers);
    for(int b=0;b<n_layers;++b){auto& l=layers[b];std::string p="blk."+std::to_string(b)+".";
        l.q=need(table,p+"attn_q.weight",err);l.k=need(table,p+"attn_k.weight",err);l.v=need(table,p+"attn_v.weight",err);l.o=need(table,p+"attn_output.weight",err);l.sg=need(table,p+"ffn_gate_shexp.weight",err);l.su=need(table,p+"ffn_up_shexp.weight",err);l.sd=need(table,p+"ffn_down_shexp.weight",err);
        l.an=(const float*)need(table,p+"attn_norm.weight",err)->data;l.pan=(const float*)need(table,p+"post_attention_norm.weight",err)->data;l.fn=(const float*)need(table,p+"ffn_norm.weight",err)->data;l.pfn=(const float*)need(table,p+"post_ffw_norm.weight",err)->data;l.qn=(const float*)need(table,p+"attn_q_norm.weight",err)->data;l.kn=(const float*)need(table,p+"attn_k_norm.weight",err)->data;l.router=need(table,p+"ffn_gate_inp.weight",err)->data;l.bias=(const float*)need(table,p+"exp_probs_b.bias",err)->data;
        if(!err.empty()){std::fprintf(stderr,"bind: %s\n",err.c_str());return 1;}
        alloc((void**)&l.kc,(size_t)max_context*N_KV*HD*4,"K cache");alloc((void**)&l.vc,(size_t)max_context*N_KV*HD*4,"V cache");
        if(kv_bits<32){const size_t cell=(kv_bits==8)?KC::kv_q8_cell_bytes():KC::kv_q4_cell_bytes();
            alloc((void**)&l.kq,(size_t)max_context*N_KV*cell,"K quant cache");alloc((void**)&l.vq,(size_t)max_context*N_KV*cell,"V quant cache");
            cudaFree(l.kc);cudaFree(l.vc);l.kc=nullptr;l.vc=nullptr;                 // no fp32 copy alongside it
            alloc((void**)&l.ks,N_KV*HD*4,"K cell");alloc((void**)&l.vs,N_KV*HD*4,"V cell");}
        if(!KCPU::native_fmt(types[b].first,types[b].second,N_EMBD,FF,l.fmt,err)){std::fprintf(stderr,"expert fmt: %s\n",err.c_str());return 1;}
        l.pre_fn=[&,b]()->bool{
            const Layer& L=layers[b];
            op_run("p_rms",[&]{ KC::rms_norm(x,L.an,norm,N_EMBD,geo.rms_eps,stream); });
            op_run("p_q8",[&]{ K::native_quantize_q8_1(norm,q8b,N_EMBD,1,stream); });
            bool okq=true;
            op_run("p_qkv",[&]{ okq=mmvq(L.q,q8b,q,stream,err)&&mmvq(L.k,q8b,k,stream,err)&&mmvq(L.v,q8b,v,stream,err); });
            if(!okq)return false;
            op_run("p_qkn",[&]{ if(kv_bits<32){KC::qk_norm_rope_scratch(q,k,v,L.qn,L.kn,L.ks,L.vs,pos_dev,geo.swa_pattern[b]!=0,geo.rope_freq_base,stream);
                                 if(kv_bits==8)KC::kv_store_q8(L.kq,L.vq,L.ks,L.vs,pos_dev,max_context,stream);
                                 else KC::kv_store_q4(L.kq,L.vq,L.ks,L.vs,pos_dev,max_context,stream);}
                             else KC::qk_norm_rope_cache(q,k,v,L.qn,L.kn,L.kc,L.vc,pos_dev,geo.swa_pattern[b]!=0,geo.rope_freq_base,stream); });
            op_run("p_attn",[&]{ if(kv_bits==8)KC::attention_q8(q,L.kq,L.vq,att,pos_dev,geo.swa_pattern[b]!=0,(int)geo.swa_window,max_context,stream);
                             else if(kv_bits==4)KC::attention_q4(q,L.kq,L.vq,att,pos_dev,geo.swa_pattern[b]!=0,(int)geo.swa_window,max_context,stream);
                             else KC::attention(q,L.kc,L.vc,att,pos_dev,geo.swa_pattern[b]!=0,(int)geo.swa_window,max_context,stream); });
            op_run("p_o",[&]{ K::native_quantize_q8_1(att,q8a,N_HEAD*HD,1,stream);okq=mmvq(L.o,q8a,proj,stream,err); });
            if(!okq)return false;
            op_run("p_norms",[&]{ KC::rms_norm_residual(proj,L.pan,x,N_EMBD,geo.rms_eps,stream);KC::rms_norm(x,L.fn,norm,N_EMBD,geo.rms_eps,stream); });
            op_run("p_route",[&]{ KC::route(norm,L.router,L.bias,ids_dev,weights_dev,raw_dev,stream); });
            op_run("p_d2h",[&]{ cudaMemcpyAsync(ids_host,ids_dev,N_USED*4,cudaMemcpyDeviceToHost,stream);cudaMemcpyAsync(weights_host,weights_dev,N_USED*4,cudaMemcpyDeviceToHost,stream);cudaMemcpyAsync(x_host,norm,N_EMBD*4,cudaMemcpyDeviceToHost,stream); });
            return true;
        };
        l.post_fn=[&,b]()->bool{
            const Layer& L=layers[b];
            bool okq=true;
            op_run("q_h2d",[&]{ cudaMemcpyAsync(routed,routed_host,N_EMBD*4,cudaMemcpyHostToDevice,stream);K::native_quantize_q8_1(norm,q8b,N_EMBD,1,stream); });
            op_run("q_shexp",[&]{ okq=mmvq(L.sg,q8b,gate,stream,err)&&mmvq(L.su,q8b,up,stream,err); });
            if(!okq)return false;
            op_run("q_swiglu",[&]{ K::native_swiglu_quantize_q8_1(gate,up,q8a,FF,1,stream);okq=mmvq(L.sd,q8a,shared,stream,err); });
            if(!okq)return false;
            op_run("q_normres",[&]{ KC::add_norm_residual(routed,shared,L.pfn,x,N_EMBD,geo.rms_eps,stream); });
            return true;
        };
        auto capture=[&](bool post,cudaGraphExec_t* exec)->bool{
            g_capturing=true;
            cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
            const bool ok=post?l.post_fn():l.pre_fn();
            g_capturing=false;
            if(!ok)return false;
            cudaGraph_t g{};cudaError_t ce=cudaStreamEndCapture(stream,&g);if(ce!=cudaSuccess){err=cudaGetErrorString(ce);return false;}ce=cudaGraphInstantiate(exec,g,0);cudaGraphDestroy(g);return check(ce,"graph instantiate",err);
        };
        if(no_graph){ /* ops run directly per token; nothing to capture */ }
        else if(!capture(false,&l.pre)||!capture(true,&l.post)){std::fprintf(stderr,"capture layer %d: %s\n",b,err.c_str());return 1;}
    }
    const C::WeightRef* outnorm=need(table,"output_norm.weight",err); if(!outnorm){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    double cache_gb=8.0;if(const char* e=std::getenv("STRATA_KOLIBRI_EXPERT_CACHE_GB"))cache_gb=std::atof(e);
    if(pin_layers>(int)n_layers)pin_layers=(int)n_layers;
    size_t pinned_bytes=0;for(int b=0;b<pin_layers;++b)pinned_bytes+=g_row[(size_t)b].blob_bytes*(size_t)N_EXPERT;
    size_t total_bytes=(size_t)(cache_gb*1073741824.0);
    // The budget must fit the DEVICE, not just the expert cache: the KV cache (1.7 GB at a 4096
    // context), the trunk, the head and allocator slack live in the same VRAM.  Without this, a
    // 30 GB cache plus 24 pinned layers looked fine on paper and died mid-generation with
    // "expert allocation: out of memory" once the LRU grew into its budget.
    {
        size_t free_b=0,total_vram=0;
        if(cudaMemGetInfo(&free_b,&total_vram)==cudaSuccess){
            const size_t reserve=(size_t)2*1073741824;
            if(total_bytes+reserve>free_b){
                const size_t capped=(free_b>reserve)?free_b-reserve:free_b/2;
                std::printf("  budget %.1f GB needs %.1f GB free plus %.1f GB reserve; capping to %.1f GB\n",
                            (double)total_bytes/1073741824.0,(double)free_b/1073741824.0,(double)reserve/1073741824.0,(double)capped/1073741824.0);
                total_bytes=capped;
            }
        }
    }
    if(pinned_bytes>=total_bytes){std::fprintf(stderr,"pin-layers %d needs %.1f GB, budget is %.1f GB\n",pin_layers,(double)pinned_bytes/1073741824.0,cache_gb);return 1;}
    CacheRuntime cache(total_bytes-pinned_bytes);
    cache.pinned.assign(n_layers,nullptr);cache.pinned_stride.assign(n_layers,0);
    if(pin_layers>0){
        const auto p0=std::chrono::steady_clock::now();
        const size_t batch=8;
        uint8_t* hbuf=nullptr;
        size_t max_row=0;for(int b=0;b<pin_layers;++b)max_row=std::max(max_row,g_row[(size_t)b].blob_bytes);
        if(!check(cudaHostAlloc((void**)&hbuf,batch*max_row,cudaHostAllocDefault),"pin staging",err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
        for(int b=0;b<pin_layers;++b){
            const size_t row=g_row[(size_t)b].blob_bytes;
            const size_t gu=K::native_mmvq_weight_bytes(types[(size_t)b].first,N_EMBD,FF);
            const size_t down=K::native_mmvq_weight_bytes(types[(size_t)b].second,FF,N_EMBD);
            void* base=nullptr;
            if(!check(cudaMalloc(&base,row*(size_t)N_EXPERT),"pinned layer",err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
            cache.pinned[(size_t)b]=base;cache.pinned_stride[(size_t)b]=row;
            for(int e0=0;e0<N_EXPERT;e0+=(int)batch){
                const int n=std::min<int>((int)batch,N_EXPERT-e0);
                for(int j=0;j<n;++j)if(!read_row(b,e0+j,gu,down,hbuf+(size_t)j*row,err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
                if(!check(cudaMemcpyAsync((uint8_t*)base+(size_t)e0*row,hbuf,(size_t)n*row,cudaMemcpyHostToDevice,cache.stream),"pin upload",err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
                if(!check(cudaStreamSynchronize(cache.stream),"pin sync",err)){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
            }
            if((b+1)%8==0||b+1==pin_layers)std::printf("  pinned %d/%d layers\n",b+1,pin_layers);
        }
        cudaFreeHost(hbuf);
        std::printf("  pinned residency: %d layers, %.1f GB, in %.0f s; dynamic LRU gets %.1f GB\n",pin_layers,(double)pinned_bytes/1073741824.0,
                    std::chrono::duration<double>(std::chrono::steady_clock::now()-p0).count(),(double)(total_bytes-pinned_bytes)/1073741824.0);
        std::fflush(stdout);
    }
    std::vector<float> logits(vocab); int pos=0;
    // Per-phase attribution (STRATA_KOLIBRI_PROFILE=1): which phase actually owns the per-token time.
    // `pre` includes waiting on the PREVIOUS layer's post graph, because the sync that ends a layer
    // is what the next one blocks on; the totals are what matter.
    double ms_pre=0,ms_ex=0,ms_post=0,ms_head=0;int prof_tokens=0;
    auto run_token=[&](int token)->bool{
        if(token<0||token>=vocab||pos>=max_context)return false;cudaMemcpyAsync(pos_dev,&pos,4,cudaMemcpyHostToDevice,stream);embed.gather_one(token,x,stream);
        for(int b=0;b<n_layers;++b){
            const auto t0=std::chrono::steady_clock::now();
            if(no_graph){
                if(!layers[b].pre_fn())return false;
                if(!check(cudaStreamSynchronize(stream),"pre sync",err))return false;
            } else if(!check(cudaGraphLaunch(layers[b].pre,stream),"pre graph",err)||!check(cudaStreamSynchronize(stream),"pre sync",err))return false;
            const auto t1=std::chrono::steady_clock::now();
            if(!run_experts(cache,b,layers[b],ids_host,weights_host,x_host,routed_host,err))return false;
            const auto t2=std::chrono::steady_clock::now();
            if(no_graph){ if(!layers[b].post_fn())return false; }
            else if(!check(cudaGraphLaunch(layers[b].post,stream),"post graph",err))return false;
            const auto t3=std::chrono::steady_clock::now();
            if(prof){ms_pre+=std::chrono::duration<double,std::milli>(t1-t0).count();ms_ex+=std::chrono::duration<double,std::milli>(t2-t1).count();ms_post+=std::chrono::duration<double,std::milli>(t3-t2).count();}
        }
        const auto h0=std::chrono::steady_clock::now();
        KC::rms_norm(x,(const float*)outnorm->data,norm,N_EMBD,geo.rms_eps,stream);if(!head.run(norm,logits_dev,stream,err))return false;cudaMemcpyAsync(logits.data(),logits_dev,(size_t)vocab*4,cudaMemcpyDeviceToHost,stream);if(!check(cudaStreamSynchronize(stream),"token sync",err))return false;++pos;
        if(prof){ms_head+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-h0).count();++prof_tokens;}
        return true;
    };
    std::printf("INFO architecture=kolibri1 execution=cuda_graph expert_cache=gpu_lru cache_gb=%.2f pinned_layers=%d kv=%s vocab=%d layers=%d\n",cache_gb,pin_layers,kv_bits==8?"q8_0":(kv_bits==4?"q4_0":"fp32"),vocab,n_layers);std::printf("READY %d\n",max_context);std::fflush(stdout);

    // ---- --prefill: warm the row cache with a long prompt, no oracle comparison needed.  This is
    // the apples-to-apples setup against a llama-benchy/llama.cpp tg run, which always measures
    // decode AFTER a long pp: with the working set resident, decode reads nothing from the SSD.
    if(!prefill_tokens.empty()){
        std::vector<int> ids;{std::stringstream csv(prefill_tokens);std::string w;while(std::getline(csv,w,','))if(!w.empty())ids.push_back(std::stoi(w));}
        const auto w0=std::chrono::steady_clock::now();int ran=0;
        for(int t:ids){ if(t<0||t>=vocab||pos>=max_context){std::printf("FAIL prefill token %d out of range\n",t);return 1;} if(!run_token(t)){std::printf("FAIL prefill: %s\n",err.c_str());return 1;} ++ran; }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-w0).count();
        std::printf("  warm prefill: %d tokens in %.1f s (%.2f tok/s)\n",ran,ms/1000.0,ran/(ms/1000.0));
        // Report the prefill's own attribution, then zero the counters so the decode print below is
        // decode-only.  Averaging the two phases together is what made an earlier read wrong.
        if(prof&&prof_tokens>0){
            std::printf("  profile/prefill token: pre+sync %.1f ms, experts %.1f ms, post %.1f ms, head %.1f ms\n",
                        ms_pre/prof_tokens,ms_ex/prof_tokens,ms_post/prof_tokens,ms_head/prof_tokens);
            op_report("prefill",prof_tokens);
            ms_pre=ms_ex=ms_post=ms_head=0;prof_tokens=0;
        }
    }
    if(gate_prefix.empty()&&gate_gen>0){
        const auto g0=std::chrono::steady_clock::now();int made=0;
        for(int g=0;g<gate_gen;++g){
            const int nx=(int)(std::max_element(logits.begin(),logits.end())-logits.begin());
            if(nx==eos||pos>=max_context)break;
            if(!run_token(nx)){std::printf("FAIL generation: %s\n",err.c_str());return 1;}
            ++made;
        }
        const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-g0).count();
        if(made>0)std::printf("  greedy decode: %d tokens in %.0f ms (%.2f tok/s, %.0f ms/token)\n",made,ms,made/(ms/1000.0),ms/made);
        if(prof&&prof_tokens>0){std::printf("  profile/token: pre+sync %.1f ms, experts %.1f ms, post launch %.1f ms, head %.1f ms (sum %.1f) over %d tokens\n",ms_pre/prof_tokens,ms_ex/prof_tokens,ms_post/prof_tokens,ms_head/prof_tokens,(ms_pre+ms_ex+ms_post+ms_head)/prof_tokens,prof_tokens);op_report("decode",prof_tokens);}
        std::printf("  cache: %s\n",cache.cache.report().c_str());std::fflush(stdout);
        if(gate_prefix.empty())return 0;
    }

    // ---- the gate: one prefill through THIS code path, every position compared with the patched
    // llama.cpp oracle's per-position dumps (<prefix><t>, 128000 f32 each, in the oracle's vocab).
    // Argmax equality is the hard criterion, cosine >= 0.98 the numeric one, exactly as the CPU
    // trunk/multitoken gates use.  Without this the CUDA path had no parity evidence at all.
    if(!gate_prefix.empty()){
        std::vector<int> ids;{std::stringstream csv(gate_tokens);std::string w;while(std::getline(csv,w,','))if(!w.empty())ids.push_back(std::stoi(w));}
        if(ids.empty()){std::printf("GATE: no tokens\n");return 2;}
        int fails=0;pos=0;double prefill_ms=0;
        for(size_t t=0;t<ids.size();++t){
            const auto t0=std::chrono::steady_clock::now();
            if(!run_token(ids[t])){std::printf("FAIL position %zu: %s\n",t,err.c_str());return 1;}
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-t0).count();
            prefill_ms+=ms;
            char pf[512];std::snprintf(pf,sizeof pf,"%s%zu",gate_prefix.c_str(),t);
            std::FILE* f=std::fopen(pf,"rb");
            if(!f){std::printf("FAIL cannot open %s\n",pf);return 1;}
            std::vector<float> ref((size_t)vocab);const size_t got=std::fread(ref.data(),4,(size_t)vocab,f);std::fclose(f);
            if(got!=(size_t)vocab){std::printf("FAIL %s holds %zu values, expected %d (vocab mismatch?)\n",pf,got,vocab);return 1;}
            const int am_e=(int)(std::max_element(logits.begin(),logits.end())-logits.begin());
            const int am_o=(int)(std::max_element(ref.begin(),ref.end())-ref.begin());
            double dot=0,ne=0,no=0,maxerr=0;
            for(int i=0;i<vocab;++i){dot+=(double)logits[i]*ref[(size_t)i];ne+=(double)logits[i]*logits[i];no+=(double)ref[(size_t)i]*ref[(size_t)i];
                const double d=std::fabs((double)logits[i]-ref[(size_t)i]);if(d>maxerr)maxerr=d;}
            const double cos=dot/(std::sqrt(ne)*std::sqrt(no));
            const bool ok=(am_e==am_o)&&cos>=0.98;
            if(!ok)++fails;
            std::printf("  %s position %zu: argmax %d vs oracle %d, cos %.4f, max |logit err| %.3f (%.0f ms)\n",ok?"PASS":"FAIL",t,am_e,am_o,cos,maxerr,ms);
        }
        std::printf("  prefill: %zu tokens in %.1f s (%.2f tok/s)\n",ids.size(),prefill_ms/1000.0,ids.size()/(prefill_ms/1000.0));
        // Greedy decode through the same path (no sampler, so the rate is comparable run to run).
        if(gate_gen>0){
            const auto g0=std::chrono::steady_clock::now();int made=0;
            for(int g=0;g<gate_gen;++g){
                const int nx=(int)(std::max_element(logits.begin(),logits.end())-logits.begin());
                if(nx==eos||pos>=max_context)break;
                if(!run_token(nx)){std::printf("FAIL generation: %s\n",err.c_str());return 1;}
                ++made;
            }
            const double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-g0).count();
            if(made>0)std::printf("  greedy decode: %d tokens in %.0f ms (%.2f tok/s, %.0f ms/token)\n",made,ms,made/(ms/1000.0),ms/made);
        }
        std::printf("  cache: %s\n",cache.cache.report().c_str());
        std::printf("  every position argmax-identical to the oracle%s\n",fails?" NO":"");
        std::printf("\nCUDA KOLIBRI GATE: %s\n",fails==0?"PASS":"FAIL");std::fflush(stdout);
        return fails==0?0:1;
    }
    std::mt19937 rng(std::random_device{}());std::string line;
    while(std::getline(std::cin,line)){if(line=="QUIT")break;if(line.rfind("GEN ",0)!=0){std::printf("ERR expected GEN request\n");std::fflush(stdout);continue;}std::istringstream in(line.substr(4));int max_new=0;in>>max_new;float temp=0,top_p=1,min_p=0,repeat=1,freq=0,present=0;int top_k=64,last_n=64;std::string w,ids_text;while(in>>w){auto eq=w.find('=');if(eq==std::string::npos){ids_text=w;continue;}auto key=w.substr(0,eq),val=w.substr(eq+1);if(key=="temperature")temp=std::stof(val);else if(key=="top_p")top_p=std::stof(val);else if(key=="top_k")top_k=std::stoi(val);else if(key=="min_p")min_p=std::stof(val);else if(key=="penalty_repeat")repeat=std::stof(val);else if(key=="penalty_freq")freq=std::stof(val);else if(key=="penalty_present")present=std::stof(val);else if(key=="penalty_last_n")last_n=std::stoi(val);else if(key=="seed"&&std::stoul(val)>0)rng.seed((uint32_t)std::stoul(val));}
        std::vector<int> ids;std::stringstream csv(ids_text);while(std::getline(csv,w,','))if(!w.empty())ids.push_back(std::stoi(w));if(max_new<1||ids.empty()||ids.size()+max_new>(size_t)max_context){std::printf("ERR invalid request or context too long\n");std::fflush(stdout);continue;}pos=0;bool ok=true;std::vector<int> hist=ids;for(int t:ids){if(!(ok=run_token(t)))break;std::printf("PP %d %zu\n",pos,ids.size());std::fflush(stdout);}int next=ok?sample_logits(logits,hist,temp,top_k,top_p,min_p,repeat,freq,present,last_n,rng):-1;int made=0;while(ok&&made<max_new){std::printf("T %d\n",next);std::fflush(stdout);++made;if(next==eos||made==max_new)break;hist.push_back(next);ok=run_token(next);if(ok)next=sample_logits(logits,hist,temp,top_k,top_p,min_p,repeat,freq,present,last_n,rng);}if(ok)std::printf("DONE %d %zu 0.000 0.000 %s\n",made,ids.size(),next==eos?"eos":"length");else std::printf("ERR %s\n",err.c_str());std::fflush(stdout);
    }
    return 0;
}
