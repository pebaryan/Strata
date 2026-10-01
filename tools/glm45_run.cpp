/// THE PORT'S LAST STEP: run the engine's OWN 45-block trunk and produce a token.
///
/// Everything this file assembles has been verified separately, and the point of writing it is to find the one thing
/// that cannot be verified separately - the assembly.  The acceptance test is a single number: argmax == 12089 for
/// prompt 785 6722 315 9621 374.  It is deliberately NOT a per-layer float comparison, because the engine's own
/// hidden state accumulates the quantization of 43 routed blocks; each routed layer differs from a float64 reference
/// by about 5% of rms at the worst element, measured, and that is the correct behaviour rather than a defect.
///
/// The plumbing is documented in tools/glm5_binding_contract.txt under "THE FINAL RUNNER".  Two traps are enforced in
/// code here because they are silent otherwise: the swiglu clamps are set EXPLICITLY (their defaults are 0, which
/// disables them) and expert_layout_load is called BEFORE FileExpertSource::open (open validates against the
/// process-wide layout, which is the canonical Q2_0 default until the pack is loaded).
///
/// usage: glm45_run <pack-dir> <gguf-shard1> [hc-init.bin] [w-output.bin] [w-output-norm.bin]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/expert_source.hpp"
#include "strata/core/glm_bind.hpp"
#include "strata/core/glm_layer.hpp"
#include "strata/core/glm_layer_weights.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/core/glm_trunk.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

namespace C = strata::core;
namespace K = strata::kernels::glm;
namespace KCPU = strata::kernels::cpu;

static const int N_LAYERS = 45;      // blocks 0..44; 45 is the MTP head and not part of the trunk
static const int N_EMBD = 4096;
static const int HC = 4;             // hyper-connection streams
static const int N_EXPERT = 288;
static const int N_USED = 8;
static const int FF_EXP = 2048;      // a routed expert / the shared expert
static const int FF_DENSE = 12288;   // the dense FFN of blocks 0..2
static const int NH = 64, HD = 128;  // KDA geometry
static const int KV_LORA = 512;      // MLA latent width
static const int TOKENS = 5;         // the prompt's length, which is also the MLA cache depth
static const float EPS = 1e-5f;
static const float CLAMP = 10.0f;    // swiglu_clamp_exp / _shexp: 10.0 for every layer of this model

struct StageCount {
    long calls = 0;
    std::map<std::string, long> by_name;
};

static void stage_cb(void* ctx, int layer, const char* name, const float* data, int n) {
    StageCount* sc = (StageCount*) ctx;
    ++sc->calls;
    sc->by_name[name] += 1;
}

static bool read_dump(const std::string& p, std::vector<float>& out, int ne[4]) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    unsigned hdr[5] = {0, 0, 0, 0, 0};
    f.read((char*) hdr, 20);
    if (!f) return false;
    for (int i = 0; i < 4; ++i) ne[i] = (int) hdr[i + 1];
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) (ne[i] > 0 ? ne[i] : 1);
    out.assign(n, 0.0f);
    f.read((char*) out.data(), (std::streamsize) (n * 4));
    return (bool) f;
}

static std::string layer_types(const std::string& pack, int layer, int& gu, int& dt) {
    std::ifstream f(pack + "/native_experts.txt");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        int64_t l = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0;
        int g = 0, d = 0;
        if (!(is >> l >> g >> d >> off >> nb >> go >> uo >> dob)) continue;
        if (l == layer) { gu = g; dt = d; return ""; }
    }
    return "layer " + std::to_string(layer) + " is not in native_experts.txt";
}

struct Provider {
    // Per-layer storage.  glm_fill_layer_weights assigns pointers INTO the bound block and into these, so all of it
    // must outlive the run - which is why they are sized once and never resized.
    std::vector<C::GlmBoundBlock> bound;
    std::vector<K::KdaWeights> kda;
    std::vector<K::MlaWeights> mla;
    std::vector<std::array<const float*, 3> > shexp_store;
    std::vector<C::glm::GlmTrunkLayerWeights> w;
    std::vector<KCPU::NativeFmt> fmt;
    std::vector<bool> fmt_ready;
    K::KdaGeometry kda_g;
    K::MlaGeometry mla_g;
    K::MoeGeometry dense_g, moe_g, shexp_g;
    C::ExpertSource* src = nullptr;
    std::vector<float*> staged;      ///< this layer's host floats; freed when the next layer is asked for
};

/// Dequantize one quantized bound tensor to HOST floats.  The kernels are device kernels (dequant_q5_K was gated
/// bit-exact against the oracle over 134 MB before being wired in), so the work happens on the device and the result is
/// copied back - what matters is that it ENDS UP IN HOST MEMORY, because every consumer in the trunk is a CPU kernel.
static float* dequant_to_host(const C::GlmBoundBlock::Tensor& t, std::string& err) {
    const size_t n = (size_t) t.ne0 * (size_t) (t.ne1 > 0 ? t.ne1 : 1);
    float* dev = nullptr;
    if (cudaMalloc(&dev, n * sizeof(float)) != cudaSuccess) {
        err = "dequant_to_host: cudaMalloc failed for " + t.name;
        return nullptr;
    }
    if (t.native_type == 8) {
        strata::kernels::dequant_q8_0((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else if (t.native_type == 13) {
        strata::kernels::dequant_q5_K((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else if (t.native_type == 14) {
        strata::kernels::dequant_q6_K((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else {
        err = "dequant_to_host: no dequantizer for type " + std::to_string(t.native_type) + " (" + t.name + ")";
        cudaFree(dev);
        return nullptr;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "dequant_to_host: the kernel failed for " + t.name + ": " + cudaGetErrorString(cudaGetLastError());
        cudaFree(dev);
        return nullptr;
    }
    float* host = (float*) std::malloc(n * sizeof(float));
    if (host == nullptr) { err = "dequant_to_host: host allocation failed for " + t.name; cudaFree(dev); return nullptr; }
    const cudaError_t cp = cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    if (cp != cudaSuccess) {
        std::free(host);
        err = "dequant_to_host: copy back failed for " + t.name + ": " + cudaGetErrorString(cp);
        return nullptr;
    }
    return host;
}

static const uint8_t* blob_adapter(void* ctx, int layer, int expert) {
    return ((C::ExpertSource*) ctx)->blob(layer, expert);
}

static bool provider(void* raw, int layer, C::glm::GlmTrunkLayerWeights& out, std::string& err) {
    Provider* p = (Provider*) raw;
    if (layer < 0 || layer >= N_LAYERS) { err = "layer out of range"; return false; }
    // ---- STAGE THIS LAYER'S WEIGHTS TO HOST, one layer at a time, releasing the previous layer's.
    //
    // The trunk's kernels are CPU-side, so every weight they read has to be host-resident floats - but materialising
    // all forty-five layers at once is 20-30 GB and OOM-kills the box (measured: exit -9, no output).  The provider is
    // the right place because glm_trunk_forward calls it once per layer, before that layer runs, which guarantees a
    // layer's weights are needed only while that layer runs.
    //
    // Every quantized tensor is staged EXCEPT the dense FFN's three on blocks 0..2: those are consumed by the GPU path,
    // which wants the quantized device blocks as they are, so they keep them.
    for (float* f : p->staged) std::free(f);
    p->staged.clear();
    {
        C::GlmBoundBlock& B = p->bound[(size_t) layer];
        const bool dense = (layer < 3);
        for (size_t i = 0; i < B.tensors.size(); ++i) {
            C::GlmBoundBlock::Tensor& t = B.tensors[i];
            if (!t.quantized || t.ptr == nullptr) continue;
            // NO EXCEPTIONS, and the measurement is why: I first skipped the dense FFN's three tensors on blocks
            // 0..2, on the assumption that a GPU path consumed them.  It does not - glm_stage_ffn calls the CPU
            // expert_ffn, so the crash simply moved from kda_forward to expert_ffn when the rest were staged.  Every
            // quantized tensor is staged, and the per-layer lifetime is what keeps it affordable.
            (void) dense;
            float* h = dequant_to_host(t, err);
            if (h == nullptr) return false;
            p->staged.push_back(h);
            t.ptr = h;
            t.quantized = false;      // never re-dequantize the same layer twice
        }
        // re-map: glm_fill_layer_weights is pure assignment, so doing it per layer costs nothing
        if (!C::glm::glm_fill_layer_weights(B, layer, p->kda_g, p->mla_g, p->kda[(size_t) layer],
                                           p->mla[(size_t) layer], p->shexp_store[(size_t) layer].data(),
                                           p->w[(size_t) layer], err)) {
            err = "provider: re-mapping layer " + std::to_string(layer) + ": " + err;
            return false;
        }
    }
    out = p->w[(size_t) layer];                      // assembled from the staged host weights

    // ---- LAYER 0 BESIDE LAYER 4, because layer 0 runs and layer 4 stops on "null argument".  A dump of the failing
    // layer says which field is NULL; a differential says which field DIFFERS, and the difference is the defect.
    if ((layer == 0 || layer == 4) && out.kda != nullptr) {
        const K::KdaWeights& kw = *out.kda;
        const char* nm[16] = {"attn_norm", "wq", "wk", "wv", "conv_q", "conv_k", "conv_v", "ssm_a",
                              "dt_bias", "ssm_f_a", "ssm_f_b", "ssm_beta", "ssm_g_a", "ssm_g_b", "o_norm", "wo"};
        const float* pp[16] = {kw.attn_norm, kw.wq, kw.wk, kw.wv, kw.conv_q, kw.conv_k, kw.conv_v, kw.ssm_a,
                               kw.dt_bias, kw.ssm_f_a, kw.ssm_f_b, kw.ssm_beta, kw.ssm_g_a, kw.ssm_g_b, kw.o_norm, kw.wo};
        std::printf("  layer %d KdaWeights:\n", layer);
        for (int i = 0; i < 16; ++i) {
            std::printf("    %-10s %p%s\n", nm[i], (const void*) pp[i], pp[i] == nullptr ? "   <-- NULL" : "");
        }
    }
    const bool is_dense = (layer < 3);
    if (is_dense) {
        out.moe_g = &p->dense_g;
        out.clamp_limit = CLAMP;
    } else {
        if (!p->fmt_ready[(size_t) layer]) {
            err = "no expert descriptor for layer " + std::to_string(layer);
            return false;
        }
        out.moe_g = &p->moe_g;
        out.shexp_g = &p->shexp_g;
        out.shexp = p->shexp_store[(size_t) layer].data();
        out.moe_fmt = &p->fmt[(size_t) layer];
        out.blob_fn = &blob_adapter;
        out.blob_ctx = p->src;
        out.shexp_clamp = CLAMP;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: glm45_run <pack-dir> <gguf-shard1> [hc-init.bin] [w-output.bin] [w-norm.bin]\n");
        return 2;
    }
    const std::string pack = argv[1];
    const std::string shard1 = argv[2];
    const std::string in_path = argc > 3 ? argv[3] : "/home/peb/moredata/glm5-oracle-input/hc_init.bin";
    const std::string wout_path = argc > 4 ? argv[4] : "/home/peb/moredata/glm5-head-gate/w_output.bin";
    const std::string wnorm_path = argc > 5 ? argv[5] : "/home/peb/moredata/glm5-head-gate/w_output_norm.bin";
    std::string err;

    // ---- all three GGUF shards: shard 1 is metadata-only, so a one-shard NativeDense sees an empty tensor list
    std::vector<std::string> shards;
    {
        const size_t slash = shard1.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "" : shard1.substr(0, slash + 1);
        const std::string base = slash == std::string::npos ? shard1 : shard1.substr(slash + 1);
        const size_t pos = base.find("-00001-of-");
        if (pos != std::string::npos) {
            const std::string stem = base.substr(0, pos);
            const std::string tail = base.substr(base.find("-of-") + 4);
            const int n = std::atoi(tail.substr(0, tail.find('.')).c_str());
            for (int i = 1; i <= n; ++i) {
                char buf[4096];
                std::snprintf(buf, sizeof buf, "%s%s-%05d-of-%05d.gguf", dir.c_str(), stem.c_str(), i, n);
                shards.emplace_back(buf);
            }
        } else {
            shards.push_back(shard1);
        }
    }
    std::printf("shards: %zu\n", shards.size());

    std::set<std::string> served;
    if (!C::NativeDense::served_names(shards, false, served, err)) {
        std::fprintf(stderr, "served_names: %s\n", err.c_str());
        return 1;
    }
    C::WeightTable table;
    uint64_t bytes = 0;
    if (!table.pool_bytes(pack, bytes, err, &served)) { std::fprintf(stderr, "pool_bytes: %s\n", err.c_str()); return 1; }
    // MANAGED, not device-only.  The trunk's hc and norm stages are CPU kernels that read their weights directly
    // (hc_pre / rms_norm on the host), while the same bound block feeds the CUDA kernels - so the arena has to be
    // readable from both sides.  With cudaMalloc the host dereference is a SIGSEGV, which is exactly what the first
    // run of this tool produced, and the backtrace named it: hc_pre <- glm_stage_hc_norm <- glm_trunk_forward.  On a
    // V100 there is no host mapping for cudaMalloc memory, so the run must use managed memory (page-fault migration,
    // slower, correct) - a production loader would instead keep the small hc/norm weights in host memory.
    void* arena = nullptr;
    if (cudaMallocManaged(&arena, bytes, cudaMemAttachGlobal) != cudaSuccess) {
        std::fprintf(stderr, "managed arena alloc of %llu failed\n", (unsigned long long) bytes);
        return 1;
    }
    if (!table.load(pack, arena, bytes, err, &served)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    C::NativeDense nd;
    if (!nd.load(shards, table, err)) { std::fprintf(stderr, "native: %s\n", err.c_str()); return 1; }
    std::printf("arena %.2f GB, %zu tensors served from the GGUF\n", bytes / 1073741824.0, served.size());

    // ---- geometry.  The clamps are set EXPLICITLY: their defaults are 0.0f, which DISABLES them, and the header
    // says so in as many words.  A routed pre-activation past 10 would then be passed through unclamped, silently.
    Provider P;
    P.kda_g.n_embd = N_EMBD; P.kda_g.nh = NH; P.kda_g.hd = HD; P.kda_g.d_conv = 4;
    P.mla_g.n_embd = N_EMBD;                                   // its defaults are this artifact's MLA geometry
    P.dense_g.n_embd = N_EMBD; P.dense_g.ff = FF_DENSE; P.dense_g.n_expert = N_EXPERT; P.dense_g.n_used = N_USED;
    P.moe_g.n_embd = N_EMBD; P.moe_g.ff = FF_EXP; P.moe_g.n_expert = N_EXPERT; P.moe_g.n_used = N_USED;
    P.shexp_g.n_embd = N_EMBD; P.shexp_g.ff = FF_EXP;
    for (K::MoeGeometry* g : {&P.dense_g, &P.moe_g, &P.shexp_g}) {
        g->clamp_exp = CLAMP;
        g->clamp_shexp = CLAMP;
    }

    // ---- THE EXPERTS.  The layout must be loaded BEFORE the source is opened, and with a layer count that covers
    // the highest layer the file holds (45) - 43 fails as "malformed", because a row's layer number must be < n_layers.
    if (!KCPU::expert_layout_load(pack, 46, N_EXPERT, err, N_EMBD, FF_EXP)) {
        std::fprintf(stderr, "expert_layout_load: %s\n", err.c_str());
        return 1;
    }
    C::FileExpertSource src;
    if (!src.open(pack, 46, N_EXPERT, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    P.src = &src;

    // ---- bind and map all 45 blocks
    P.bound.resize(N_LAYERS);
    P.kda.resize(N_LAYERS);
    P.mla.resize(N_LAYERS);
    P.shexp_store.resize(N_LAYERS);
    P.w.resize(N_LAYERS);
    P.fmt.resize(N_LAYERS);
    P.fmt_ready.assign(N_LAYERS, false);
    cudaStream_t stream = nullptr;
    for (int b = 0; b < N_LAYERS; ++b) {
        C::LayerView v(table, b);
        if (!bind_glm_block(v, b, 8192, 4, shards, P.bound[(size_t) b], (void*) stream, err)) {
            std::fprintf(stderr, "bind block %d: %s\n", b, err.c_str());
            return 1;
        }
        if (!C::glm::glm_fill_layer_weights(P.bound[(size_t) b], b, P.kda_g, P.mla_g, P.kda[(size_t) b],
                                            P.mla[(size_t) b], P.shexp_store[(size_t) b].data(),
                                            P.w[(size_t) b], err)) {
            std::fprintf(stderr, "fill block %d: %s\n", b, err.c_str());
            return 1;
        }
        if (b >= 3) {
            int gu = 0, dt = 0;
            const std::string te = layer_types(pack, b, gu, dt);
            if (!te.empty()) { std::fprintf(stderr, "%s\n", te.c_str()); return 1; }
            if (!KCPU::native_fmt(gu, dt, N_EMBD, FF_EXP, P.fmt[(size_t) b], err)) {
                std::fprintf(stderr, "native_fmt(layer %d, %d/%d): %s\n", b, gu, dt, err.c_str());
                return 1;
            }
            P.fmt_ready[(size_t) b] = true;
        }
    }
    std::printf("bound and mapped %d blocks\n", N_LAYERS);

    // ---- the hc weights need no workaround any more.
    //
    // This block used to copy hc_attn_fn and hc_ffn_fn to host, because the pack's dequantized weights came back as
    // device pointers while the trunk's kernels are CPU-side.  That copy moved the crash from hc_pre to kda_forward,
    // which proved the diagnosis; the fix now lives in the LOADER (glm_bind.cpp), where the dequantized floats land in
    // host memory for every tensor that needs it.  The runner no longer patches weights it did not load - and the
    // cudaMemcpy it used to do now fails with "invalid argument", because its source is host memory, which is the
    // signal that the loader is finally doing its job.

    // ---- DIAGNOSTIC: are the hc/norm weights the trunk's CPU stages read actually host-readable?
    //
    // Every previous caller of glm_trunk_forward passed FIXTURE arrays living in ordinary host memory, so this could
    // not come up before.  The bound pack puts those tensors somewhere, and hc_pre - a CPU kernel - segfaults on it.
    // An address inside the arena would mean the arena itself needs a host mapping; an address outside means the
    // binder allocated privately and that allocation is the thing to inspect.
    {
        const char* lo = (const char*) arena;
        const char* hi = lo + bytes;
        std::printf("arena range: %p .. %p\n", (const void*) lo, (const void*) hi);
        const int probe[2] = {0, 3};
        for (int k = 0; k < 2; ++k) {
            const int b = probe[k];
            const C::glm::GlmTrunkLayerWeights& ww = P.w[(size_t) b];
            const void* ptrs[4] = {(const void*) ww.hc_attn_fn, (const void*) ww.hc_attn_base,
                                   (const void*) ww.hc_attn_scale, (const void*) ww.attn_norm};
            const char* names[4] = {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "attn_norm"};
            std::printf("  layer %2d:", b);
            for (int j = 0; j < 4; ++j) {
                const char* where = "NULL";
                if (ptrs[j] != nullptr) {
                    const char* c = (const char*) ptrs[j];
                    where = (c >= lo && c < hi) ? "in-arena" : "OUTSIDE-arena";
                }
                std::printf(" %s=%p(%s)", names[j], ptrs[j], where);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    // ---- WHICH TENSORS CAN A CPU KERNEL NOT READ?
    //
    // This is the list the loader has to be fixed against: every tensor the pack serves QUANTIZED comes back as a
    // device pointer, and the trunk's hc and KDA stages are CPU kernels.  Printed per tensor rather than inferred,
    // because the fields the weight structs point at are exactly the ones that have to move to host memory.
    {
        const char* lo = (const char*) arena;
        const char* hi = lo + bytes;
        const int probe[2] = {0, 3};
        for (int k = 0; k < 2; ++k) {
            const int b = probe[k];
            size_t outside = 0, inside = 0;
            std::printf("  layer %d bound tensors outside the arena:", b);
            for (const C::GlmBoundBlock::Tensor& t : P.bound[(size_t) b].tensors) {
                const char* c = (const char*) t.ptr;
                if (t.ptr != nullptr && (c < lo || c >= hi)) {
                    ++outside;
                    std::printf(" %s", t.name.c_str());
                } else {
                    ++inside;
                }
            }
            std::printf("\n    (%zu outside, %zu inside the arena)\n", outside, inside);
            std::fflush(stdout);
        }
    }

    // ---- state: one KDA state per KDA layer, one MLA cache per MLA layer, both keyed by the ARTIFACT's layer number
    std::vector<std::vector<float> > kda_state(N_LAYERS);
    std::vector<std::vector<float> > mla_cache(N_LAYERS);
    float* kda_ptrs[N_LAYERS];
    float* mla_ptrs[N_LAYERS];
    int kda_len[N_LAYERS];
    int mla_len[N_LAYERS];
    int kda_index[N_LAYERS], mla_index[N_LAYERS];
    int n_kda = 0, n_mla = 0;
    for (int b = 0; b < N_LAYERS; ++b) {
        kda_index[b] = -1;
        mla_index[b] = -1;
        kda_len[b] = 0;
        mla_len[b] = 0;
        // FILLED BY SLOT, NOT BY LAYER, and that distinction is a real bug I had.
        //
        // The trunk reads state.kda_state[slot] and state.mla_cache[slot], where slot is kda_index[layer] /
        // mla_index[layer] - so those two arrays are keyed by SLOT.  This runner was filling them by LAYER.  For blocks
        // 0, 1 and 2 every block is KDA, so slot == layer and the two agree by accident; block 3 is MLA and takes no KDA
        // slot, so from block 4 onward the mapping shifts by one and kda_state[slot] lands on the index that block 3
        // would have had - never assigned, hence whatever the array held, and a fault named as a null argument.
        //
        // The same one-line shift applies to the MLA caches in the other direction, which is very likely what the
        // layer-3 pointer differential was showing when four device pointers came back identical across two layers.
        if (C::glm::glm_attention_is_mla(b) == 1) {
            mla_cache[(size_t) b].assign((size_t) TOKENS * KV_LORA, 0.0f);
            mla_index[b] = n_mla;
            mla_ptrs[n_mla] = mla_cache[(size_t) b].data();
            ++n_mla;
        } else {
            kda_state[(size_t) b].assign((size_t) NH * HD * HD, 0.0f);
            kda_index[b] = n_kda;
            kda_ptrs[n_kda] = kda_state[(size_t) b].data();
            ++n_kda;
        }
    }
    std::printf("state: %d KDA slots (%.0f MB), %d MLA caches (%d cells each)\n", n_kda,
                n_kda * (double) NH * HD * HD * 4.0 / 1048576.0, n_mla, TOKENS);

    C::glm::GlmTrunkState st;
    st.kda_state = kda_ptrs;
    st.kda_index = kda_index;
    st.mla_cache = mla_ptrs;
    st.mla_len = mla_len;
    st.mla_index = mla_index;

    // ---- the input: the oracle's own hc_init, five tokens of 16384 floats, ne0 fastest
    std::vector<float> inp;
    int ne_in[4] = {0, 0, 0, 0};
    if (!read_dump(in_path, inp, ne_in)) { std::fprintf(stderr, "cannot read %s\n", in_path.c_str()); return 1; }
    if (ne_in[0] != N_EMBD || ne_in[1] != HC || ne_in[2] < TOKENS) {
        std::fprintf(stderr, "unexpected hc_init shape [%d,%d,%d,%d]\n", ne_in[0], ne_in[1], ne_in[2], ne_in[3]);
        return 1;
    }
    std::printf("input: hc_init ne=[%d,%d,%d,%d], %zu floats\n", ne_in[0], ne_in[1], ne_in[2], ne_in[3], inp.size());

    // ---- run: the TOKEN loop is outer, because the layers are stateful and the state carries across it
    StageCount sc;
    std::vector<float> l_out((size_t) HC * N_EMBD, 0.0f);
    for (int t = 0; t < TOKENS; ++t) {
        const float* x = inp.data() + (size_t) t * N_EMBD * HC;
        if (!C::glm::glm_trunk_forward(x, N_LAYERS, provider, &P, P.kda_g, P.mla_g, EPS, st, l_out.data(), nullptr, err,
                                       0, &stage_cb, &sc)) {
            std::fprintf(stderr, "trunk failed at token %d: %s\n", t, err.c_str());
            return 1;
        }
        double ss = 0.0;
        for (float v : l_out) ss += (double) v * v;
        std::printf("  token %d: |l_out| %.6f, MLA cache depths", t);
        for (int b = 0; b < N_LAYERS; ++b) {
            if (mla_index[b] >= 0) std::printf(" %d", mla_len[mla_index[b]]);
        }
        std::printf("\n");
        std::fflush(stdout);
    }
    std::printf("stage reports: %ld (", sc.calls);
    for (const std::pair<const std::string, long>& kv : sc.by_name) std::printf("%s %ld ", kv.first.c_str(), kv.second);
    std::printf(")\n");

    // ---- the head: the mean over the four streams, then output_norm, then the tied projection
    std::vector<float> onorm;
    int ne_n[4] = {0, 0, 0, 0};
    if (!read_dump(wnorm_path, onorm, ne_n)) { std::fprintf(stderr, "cannot read %s\n", wnorm_path.c_str()); return 1; }
    std::vector<float> hidden((size_t) N_EMBD, 0.0f);
    if (!C::glm::glm_stage_head_mean_norm(l_out.data(), HC, N_EMBD, onorm.data(), hidden.data(), err)) {
        std::fprintf(stderr, "head: %s\n", err.c_str());
        return 1;
    }
    std::printf("hidden rms %.6f\n", [&] {
        double s = 0.0;
        for (float v : hidden) s += (double) v * v;
        return std::sqrt(s / hidden.size());
    }());

    // the projection: 154,880 rows of 4,096, ne0 fastest
    std::ifstream wf(wout_path, std::ios::binary);
    if (!wf) { std::fprintf(stderr, "cannot read %s\n", wout_path.c_str()); return 1; }
    wf.seekg(0, std::ios::end);
    const long wn = (long) wf.tellg();
    wf.seekg(0);
    const long vocab = wn / 4 / N_EMBD;
    std::printf("output.weight: %ld rows of %d\n", vocab, N_EMBD);
    std::vector<float> row((size_t) N_EMBD);
    int best = -1;
    double best_s = -1e300;
    for (long v = 0; v < vocab; ++v) {
        wf.read((char*) row.data(), (std::streamsize) (N_EMBD * 4));
        if (!wf) break;
        double s = 0.0;
        for (int j = 0; j < N_EMBD; ++j) s += (double) row[(size_t) j] * (double) hidden[(size_t) j];
        if (s > best_s) { best_s = s; best = (int) v; }
    }
    std::printf("\nargmax = %d   (logit %.6f)   expected 12089\n", best, best_s);
    std::printf("ENGINE TOKEN: %s\n", best == 12089 ? "PASS" : "MISMATCH");
    return best == 12089 ? 0 : 1;
}
