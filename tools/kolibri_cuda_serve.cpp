// CUDA graph Kolibri-1 serving path. Dense work and attention stay on device; each layer is
// split at its dynamic expert selection so Strata's resident expert-row cache can service it.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
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
    float *kc{}, *vc{};
    cudaGraphExec_t pre{}, post{};
    KCPU::NativeFmt fmt{};
};

struct CacheRuntime {
    C::ExpertRowCache cache;
    cudaStream_t stream{};
    float *x{}, *gate{}, *up{}, *out{}, *accum{};
    void *xq{}, *hq{};
    explicit CacheRuntime(size_t bytes) : cache([&] {
        C::ExpertRowCacheConfig c; c.budget_bytes=bytes; c.evict_ctx=this;
        c.on_evict=[](void*,const C::ExpertRowKey&,const C::ExpertRowEntry& e){ if(e.dev) cudaFree(e.dev); };
        return c;
    }()) {
        cudaStreamCreate(&stream);
        cudaMalloc((void**)&x,N_EMBD*4); cudaMalloc((void**)&gate,FF*4); cudaMalloc((void**)&up,FF*4);
        cudaMalloc((void**)&out,N_EMBD*4); cudaMalloc((void**)&accum,N_EMBD*4);
        cudaMalloc(&xq,K::native_q8_1_bytes(N_EMBD)); cudaMalloc(&hq,K::native_q8_1_bytes(FF));
    }
    ~CacheRuntime(){ cache.clear(); cudaFree(x);cudaFree(gate);cudaFree(up);cudaFree(out);cudaFree(accum);cudaFree(xq);cudaFree(hq);cudaStreamDestroy(stream); }
};

bool check(cudaError_t e, const char* where, std::string& err) {
    if (e == cudaSuccess) return true;
    err = std::string(where) + ": " + cudaGetErrorString(e); return false;
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

bool run_experts(CacheRuntime& rt, C::FileExpertSource& src, int layer, const Layer& l,
                 const int32_t* ids, const float* weights, const float* x, float* out, std::string& err) {
    size_t gu=K::native_mmvq_weight_bytes(l.fmt.gu_type,N_EMBD,FF);
    size_t down=K::native_mmvq_weight_bytes(l.fmt.d_type,FF,N_EMBD);
    size_t up_off=gu, down_off=2*gu, bytes=down_off+down;
    if(bytes!=l.fmt.bytes){ err="expert native layout mismatch"; return false; }
    std::vector<const uint8_t*> rows(N_USED);
    std::vector<void*> temporary;
    for(int i=0;i<N_USED;++i) {
        const uint8_t* blob=src.blob(layer,ids[i]);
        if(!blob){ err="expert source miss"; return false; }
        C::ExpertRowKey key{layer,ids[i]};
        auto state=rt.cache.lookup(key,bytes);
        void* dev=nullptr;
        if(state==C::ExpertRowState::resident) dev=rt.cache.find(key)->dev;
        else {
            if(!check(cudaMalloc(&dev,bytes),"expert allocation",err)) return false;
            // Async on rt.stream: the mmvq kernels below run on the same stream, so ordering is the
            // stream's job.  A blocking copy per row serialised 300 uploads per token against the GPU.
            if(!check(cudaMemcpyAsync(dev,blob,bytes,cudaMemcpyHostToDevice,rt.stream),"expert upload",err)){ cudaFree(dev); return false; }
            if(state==C::ExpertRowState::needs_upload){ rt.cache.insert(key,{dev,bytes}); dev=rt.cache.find(key)->dev; }
            else temporary.push_back(dev);
        }
        rows[i]=(const uint8_t*)dev;
    }
    bool ok=check(cudaMemcpyAsync(rt.x,x,N_EMBD*4,cudaMemcpyHostToDevice,rt.stream),"expert activation upload",err) &&
            check(cudaMemsetAsync(rt.accum,0,N_EMBD*4,rt.stream),"expert accumulator",err);
    if(ok) {
        K::native_quantize_q8_1(rt.x,rt.xq,N_EMBD,1,rt.stream);
        for(int i=0;i<N_USED;++i) {
            K::native_mmvq(l.fmt.gu_type,rows[i],rt.xq,rt.gate,N_EMBD,FF,1,rt.stream);
            K::native_mmvq(l.fmt.gu_type,rows[i]+up_off,rt.xq,rt.up,N_EMBD,FF,1,rt.stream);
            K::native_swiglu_quantize_q8_1(rt.gate,rt.up,rt.hq,FF,1,rt.stream);
            K::native_mmvq(l.fmt.d_type,rows[i]+down_off,rt.hq,rt.out,FF,N_EMBD,1,rt.stream);
            K::scaled_add_inplace(rt.accum,rt.out,N_EMBD,weights[i],rt.stream);
        }
        ok=check(cudaMemcpyAsync(out,rt.accum,N_EMBD*4,cudaMemcpyDeviceToHost,rt.stream),"expert result copy",err) &&
           check(cudaStreamSynchronize(rt.stream),"expert synchronize",err);
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
    std::string pack,model,gate_prefix,gate_tokens; int max_context=4096,gate_gen=0; bool serve=false;
    for(int i=1;i<argc;++i){std::string a=argv[i]; if(a=="--serve")serve=true; else if(a=="--pack"&&i+1<argc)pack=argv[++i]; else if((a=="--native"||a=="--model")&&i+1<argc)model=argv[++i]; else if(a=="--max-context"&&i+1<argc)max_context=std::atoi(argv[++i]); else if(a=="--gate"&&i+2<argc){gate_prefix=argv[++i];gate_tokens=argv[++i];} else if(a=="--gen"&&i+1<argc)gate_gen=std::atoi(argv[++i]);}
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
    C::FileExpertSource src; src.set_gguf(model); if(!src.open(pack,n_layers,N_EXPERT,err)){std::fprintf(stderr,"experts: %s\n",err.c_str());return 1;}
    std::vector<std::pair<int,int>> types(n_layers); if(!load_types(pack,types,err)){std::fprintf(stderr,"types: %s\n",err.c_str());return 1;}

    cudaStream_t stream{}; check(cudaStreamCreate(&stream),"stream",err);
    float *x{},*norm{},*q{},*k{},*v{},*att{},*proj{},*gate{},*up{},*shared{},*routed{},*logits_dev{};
    int *pos_dev{}; int32_t *ids_dev{},*ids_host{}; float *weights_dev{},*weights_host{},*x_host{},*routed_host{};
    void *q8a{},*q8b{};
    auto alloc=[&](void**p,size_t n,const char*w){if(!check(cudaMalloc(p,n),w,err)){std::fprintf(stderr,"%s\n",err.c_str());std::exit(1);}};
    alloc((void**)&x,N_EMBD*4,"x");alloc((void**)&norm,N_EMBD*4,"norm");alloc((void**)&q,N_HEAD*HD*4,"q");alloc((void**)&k,N_KV*HD*4,"k");alloc((void**)&v,N_KV*HD*4,"v");alloc((void**)&att,N_HEAD*HD*4,"attention");alloc((void**)&proj,N_EMBD*4,"projection");alloc((void**)&gate,FF*4,"shared gate");alloc((void**)&up,FF*4,"shared up");alloc((void**)&shared,N_EMBD*4,"shared out");alloc((void**)&routed,N_EMBD*4,"routed out");alloc((void**)&logits_dev,(size_t)vocab*4,"logits");alloc((void**)&pos_dev,4,"position");alloc((void**)&ids_dev,N_USED*4,"route ids");alloc((void**)&weights_dev,N_USED*4,"route weights");alloc(&q8a,K::native_q8_1_bytes(N_HEAD*HD),"q8 scratch a");alloc(&q8b,K::native_q8_1_bytes(N_EMBD),"q8 scratch b");
    cudaHostAlloc(&ids_host,N_USED*4,cudaHostAllocDefault);cudaHostAlloc(&weights_host,N_USED*4,cudaHostAllocDefault);cudaHostAlloc(&x_host,N_EMBD*4,cudaHostAllocDefault);cudaHostAlloc(&routed_host,N_EMBD*4,cudaHostAllocDefault);

    std::vector<Layer> layers(n_layers);
    for(int b=0;b<n_layers;++b){auto& l=layers[b];std::string p="blk."+std::to_string(b)+".";
        l.q=need(table,p+"attn_q.weight",err);l.k=need(table,p+"attn_k.weight",err);l.v=need(table,p+"attn_v.weight",err);l.o=need(table,p+"attn_output.weight",err);l.sg=need(table,p+"ffn_gate_shexp.weight",err);l.su=need(table,p+"ffn_up_shexp.weight",err);l.sd=need(table,p+"ffn_down_shexp.weight",err);
        l.an=(const float*)need(table,p+"attn_norm.weight",err)->data;l.pan=(const float*)need(table,p+"post_attention_norm.weight",err)->data;l.fn=(const float*)need(table,p+"ffn_norm.weight",err)->data;l.pfn=(const float*)need(table,p+"post_ffw_norm.weight",err)->data;l.qn=(const float*)need(table,p+"attn_q_norm.weight",err)->data;l.kn=(const float*)need(table,p+"attn_k_norm.weight",err)->data;l.router=need(table,p+"ffn_gate_inp.weight",err)->data;l.bias=(const float*)need(table,p+"exp_probs_b.bias",err)->data;
        if(!err.empty()){std::fprintf(stderr,"bind: %s\n",err.c_str());return 1;}
        alloc((void**)&l.kc,(size_t)max_context*N_KV*HD*4,"K cache");alloc((void**)&l.vc,(size_t)max_context*N_KV*HD*4,"V cache");
        if(!KCPU::native_fmt(types[b].first,types[b].second,N_EMBD,FF,l.fmt,err)){std::fprintf(stderr,"expert fmt: %s\n",err.c_str());return 1;}
        auto capture=[&](bool post,cudaGraphExec_t* exec)->bool{
            cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal);
            if(!post){
                KC::rms_norm(x,l.an,norm,N_EMBD,geo.rms_eps,stream);K::native_quantize_q8_1(norm,q8b,N_EMBD,1,stream);
                if(!mmvq(l.q,q8b,q,stream,err)||!mmvq(l.k,q8b,k,stream,err)||!mmvq(l.v,q8b,v,stream,err))return false;
                KC::qk_norm_rope_cache(q,k,v,l.qn,l.kn,l.kc,l.vc,pos_dev,geo.swa_pattern[b]!=0,geo.rope_freq_base,stream);
                KC::attention(q,l.kc,l.vc,att,pos_dev,geo.swa_pattern[b]!=0,(int)geo.swa_window,max_context,stream);
                K::native_quantize_q8_1(att,q8a,N_HEAD*HD,1,stream);if(!mmvq(l.o,q8a,proj,stream,err))return false;
                KC::rms_norm_residual(proj,l.pan,x,N_EMBD,geo.rms_eps,stream);KC::rms_norm(x,l.fn,norm,N_EMBD,geo.rms_eps,stream);
                KC::route(norm,l.router,l.bias,ids_dev,weights_dev,stream);
                cudaMemcpyAsync(ids_host,ids_dev,N_USED*4,cudaMemcpyDeviceToHost,stream);cudaMemcpyAsync(weights_host,weights_dev,N_USED*4,cudaMemcpyDeviceToHost,stream);cudaMemcpyAsync(x_host,norm,N_EMBD*4,cudaMemcpyDeviceToHost,stream);
            }else{
                cudaMemcpyAsync(routed,routed_host,N_EMBD*4,cudaMemcpyHostToDevice,stream);K::native_quantize_q8_1(norm,q8b,N_EMBD,1,stream);
                if(!mmvq(l.sg,q8b,gate,stream,err)||!mmvq(l.su,q8b,up,stream,err))return false;
                K::native_swiglu_quantize_q8_1(gate,up,q8a,FF,1,stream);if(!mmvq(l.sd,q8a,shared,stream,err))return false;
                KC::add_norm_residual(routed,shared,l.pfn,x,N_EMBD,geo.rms_eps,stream);
            }
            cudaGraph_t g{};cudaError_t ce=cudaStreamEndCapture(stream,&g);if(ce!=cudaSuccess){err=cudaGetErrorString(ce);return false;}ce=cudaGraphInstantiate(exec,g,0);cudaGraphDestroy(g);return check(ce,"graph instantiate",err);
        };
        if(!capture(false,&l.pre)||!capture(true,&l.post)){std::fprintf(stderr,"capture layer %d: %s\n",b,err.c_str());return 1;}
    }
    const C::WeightRef* outnorm=need(table,"output_norm.weight",err); if(!outnorm){std::fprintf(stderr,"%s\n",err.c_str());return 1;}
    double cache_gb=8.0;if(const char* e=std::getenv("STRATA_KOLIBRI_EXPERT_CACHE_GB"))cache_gb=std::atof(e);CacheRuntime cache((size_t)(cache_gb*1073741824.0));
    std::vector<float> logits(vocab); int pos=0;
    auto run_token=[&](int token)->bool{
        if(token<0||token>=vocab||pos>=max_context)return false;cudaMemcpyAsync(pos_dev,&pos,4,cudaMemcpyHostToDevice,stream);embed.gather_one(token,x,stream);
        for(int b=0;b<n_layers;++b){if(!check(cudaGraphLaunch(layers[b].pre,stream),"pre graph",err)||!check(cudaStreamSynchronize(stream),"pre sync",err))return false;if(!run_experts(cache,src,b,layers[b],ids_host,weights_host,x_host,routed_host,err))return false;if(!check(cudaGraphLaunch(layers[b].post,stream),"post graph",err))return false;}
        KC::rms_norm(x,(const float*)outnorm->data,norm,N_EMBD,geo.rms_eps,stream);if(!head.run(norm,logits_dev,stream,err))return false;cudaMemcpyAsync(logits.data(),logits_dev,(size_t)vocab*4,cudaMemcpyDeviceToHost,stream);if(!check(cudaStreamSynchronize(stream),"token sync",err))return false;++pos;return true;
    };
    std::printf("INFO architecture=kolibri1 execution=cuda_graph expert_cache=gpu_lru cache_gb=%.2f vocab=%d layers=%d\n",cache_gb,vocab,n_layers);std::printf("READY %d\n",max_context);std::fflush(stdout);

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
