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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/expert_source.hpp"
#include "strata/core/glm_bind.hpp"
#include "strata/core/expert_row_cache.hpp"
#include "strata/core/glm_expert_device.hpp"
#include "strata/core/glm_layer.hpp"
#include "strata/core/glm_layer_weights.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/core/glm_trunk.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace C = strata::core;
namespace K = strata::kernels::glm;
namespace KCPU = strata::kernels::cpu;
namespace KN = strata::kernels;

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
    int token = 0;
    std::map<std::string, long> by_name;
};

/// WALL-CLOCK PER LAYER, AT FILE SCOPE because the stage counter is local to the token lambda while a profile is a
/// property of the whole run.  The callback fires at known points inside each block, so the interval between two
/// consecutive calls is the cost of what ran between them, and attributing it to the layer that was live gives a
/// per-layer profile with no profiler attached - which is what decides where the CUDA port should start.
struct StageProfile {
    int last_layer = -1;
    const char* last_name = "";
    std::chrono::steady_clock::time_point last_time;
    std::map<int, double> ms;
    std::map<std::string, double> by_stage;   ///< keyed by the stage that was LIVE during the interval
    /// The interval between two consecutive callbacks is the cost of whatever ran between them, and the stage that ran
    /// in the gap is identified by the callback that CLOSED it - so the delta is charged to the name of the callback
    /// BEFORE it, which is what the trunk reports at the end of the work it just did.
    void note(int layer, const char* name) {
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        const double d = (last_layer >= 0) ? std::chrono::duration<double, std::milli>(now - last_time).count() : 0.0;
        if (last_layer >= 0) {
            ms[last_layer] += d;
            by_stage[last_name] += d;
        }
        last_layer = layer;
        last_name = name;
        last_time = now;
    }
};
static StageProfile g_prof;

/// HOW MANY BYTES THE STAGING MATERIALISES, which is the measurement that says what a device MMVQ path is worth.  The
/// provider dequantizes a layer's quantized tensors to host floats and frees them when the next layer is asked for, so
/// with the token loop outer EVERY LAYER IS RE-STAGED ON EVERY TOKEN: a device dequant plus a device-to-host copy per
/// tensor per token.  A device MMVQ path reads the artifact's own blocks in place - no dequantization, no transfer - so
/// this number is what it eliminates, and it is evidence rather than an argument.
static unsigned long long g_staged_bytes = 0;

static void stage_cb(void* ctx, int layer, const char* name, const float* data, int n) {
    StageCount* sc = (StageCount*) ctx;
    ++sc->calls;
    sc->by_name[name] += 1;
    // ---- WALL-CLOCK PER LAYER.  The callback fires at known points inside each block, so the interval between two
    // consecutive callbacks is the cost of what ran between them, and attributing that to the layer that was live gives
    // a per-layer profile with no profiler attached.  This is what decides which stage the CUDA port should take first,
    // and it is measured rather than guessed, like everything else in this port.
    g_prof.note(layer, name);
    // ---- AND REPORT THE MAGNITUDE, because counting stages proves the loop ran and says nothing about what it produced.
    // The oracle's own per-layer dumps for these same four families are on disk, and the runner's input IS the oracle's
    // hc_init - so the two runs are comparable stage by stage from block 0, which is the bisection that found block 0's
    // double-norm, block 1's chaining and block 3's four verdicts.  Token 0 only: that is where the lifetime and state
    // questions do not arise, so a disagreement there is a stage defect rather than a state one.
    const int dump_token = std::getenv("STRATA_STAGE_TOKEN") ? std::atoi(std::getenv("STRATA_STAGE_TOKEN")) : 0;
    if (sc->token == dump_token && data != nullptr && n > 0) {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double) data[i] * (double) data[i];
        std::printf("STAGE %d %s %.9g %d\n", layer, name, std::sqrt(ss / (double) n), n);
        // Optional raw stage capture for an exact, elementwise comparison with the oracle.  Keeping this behind an
        // environment variable means the normal runner remains read-only and avoids hundreds of diagnostic files.
        if (const char* dir = std::getenv("STRATA_STAGE_DUMP")) {
            const std::string path = std::string(dir) + "/" + std::to_string(layer) + "-" + name + ".raw";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(data, sizeof(float), (size_t) n, f);
                std::fclose(f);
            }
        }
    }
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
    std::vector<std::vector<float*> > staged_by_layer;
    ~Provider() {
        for (std::vector<float*>& layer : staged_by_layer)
            for (float* f : layer) std::free(f);
    }
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

static void free_expert_row(void*, const C::ExpertRowKey&, const C::ExpertRowEntry& entry) {
    if (entry.dev != nullptr) (void) cudaFree(entry.dev);
}

static C::ExpertRowCacheConfig expert_cache_config(size_t budget) {
    C::ExpertRowCacheConfig cfg;
    cfg.budget_bytes = budget;
    cfg.on_evict = &free_expert_row;
    return cfg;
}

struct GlmExpertDeviceRuntime {
    C::ExpertRowCache cache;
    C::GlmExpertDeviceScratch scratch;

    explicit GlmExpertDeviceRuntime(size_t budget) : cache(expert_cache_config(budget)) {}
    ~GlmExpertDeviceRuntime() {
        cache.clear();
        scratch.release();
    }
};

static bool run_cached_device_expert(void* raw, int layer, int expert, const uint8_t* blob,
                                     const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (layout.bytes == 0 || layout.bytes != fmt.bytes) {
        err = "device expert layout disagrees with NativeFmt byte count";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;

    const C::ExpertRowKey key{layer, expert};
    const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
    void* device_row = nullptr;
    bool temporary = false;
    if (state == C::ExpertRowState::resident) {
        const C::ExpertRowEntry* entry = runtime.cache.find(key);
        if (entry == nullptr || entry->dev == nullptr || entry->bytes != layout.bytes) {
            err = "expert cache reported resident without a matching device row";
            return false;
        }
        device_row = entry->dev;
    } else {
        if (cudaMalloc(&device_row, layout.bytes) != cudaSuccess) {
            err = std::string("cudaMalloc expert row: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (cudaMemcpy(device_row, blob, layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = std::string("upload expert row: ") + cudaGetErrorString(cudaGetLastError());
            cudaFree(device_row);
            return false;
        }
        if (state == C::ExpertRowState::needs_upload) {
            runtime.cache.insert(key, C::ExpertRowEntry{device_row, layout.bytes});
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (entry == nullptr) {
                err = "expert cache did not retain a row it accepted";
                return false;
            }
            device_row = entry->dev;
        } else {
            temporary = true;  // correctness-preserving path for a row larger than the configured cache budget
        }
    }

    const bool ok = C::glm_expert_ffn_device_resident((const uint8_t*) device_row, layout, fmt.gu_type, fmt.d_type,
                                                       fmt.n_embd, fmt.n_ff, x, out, runtime.scratch, err);
    if (temporary) cudaFree(device_row);
    if (!ok && !err.empty())
        err = "layer " + std::to_string(layer) + " expert " + std::to_string(expert) + ": " + err;
    return ok;
}

static bool run_cached_device_moe(void* raw, int layer, int n_experts, const int32_t* experts,
                                 const uint8_t* const* blobs, const float* weights,
                                 const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (layout.bytes == 0 || layout.bytes != fmt.bytes || n_experts <= 0 || n_experts > 64) {
        err = "device MoE layout/count disagrees with NativeFmt";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;
    std::vector<const uint8_t*> rows((size_t) n_experts);
    std::vector<void*> temporary;
    for (int i = 0; i < n_experts; ++i) {
        if (!blobs[i]) { err = "device MoE has a null host blob"; return false; }
        const C::ExpertRowKey key{layer, (int) experts[i]};
        const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
        void* device_row = nullptr;
        if (state == C::ExpertRowState::resident) {
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (!entry || !entry->dev || entry->bytes != layout.bytes) {
                err = "expert cache reported a resident row without a matching device allocation";
                return false;
            }
            device_row = entry->dev;
        } else {
            if (cudaMalloc(&device_row, layout.bytes) != cudaSuccess) {
                err = std::string("cudaMalloc expert row: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (cudaMemcpy(device_row, blobs[i], layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = std::string("upload expert row: ") + cudaGetErrorString(cudaGetLastError());
                cudaFree(device_row);
                return false;
            }
            if (state == C::ExpertRowState::needs_upload) {
                runtime.cache.insert(key, C::ExpertRowEntry{device_row, layout.bytes});
                const C::ExpertRowEntry* entry = runtime.cache.find(key);
                if (!entry) { err = "expert cache did not retain an accepted row"; return false; }
                device_row = entry->dev;
            } else {
                temporary.push_back(device_row); // oversized rows still run correctly outside the cache
            }
        }
        rows[(size_t) i] = (const uint8_t*) device_row;
    }
    const bool ok = C::glm_expert_moe_device_resident(rows.data(), weights, n_experts, layout,
                                                       fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff,
                                                       x, out, runtime.scratch, err);
    for (void* row : temporary) cudaFree(row);
    if (!ok && !err.empty()) err = "layer " + std::to_string(layer) + " device MoE: " + err;
    return ok;
}

static bool native_kda_project(int count, const void* const* weights, const int* types,
                               const float* x, int n_in, int n_out, float* const* out) {
    struct Workspace {
        float* x = nullptr;
        float* y = nullptr;
        void* q = nullptr;
        cudaStream_t stream = nullptr;
        ~Workspace() { if (x) cudaFree(x); if (y) cudaFree(y); if (q) cudaFree(q); if (stream) cudaStreamDestroy(stream); }
    };
    static Workspace ws;
    if (!ws.stream) {
        if (cudaStreamCreateWithFlags(&ws.stream, cudaStreamNonBlocking) != cudaSuccess ||
            cudaMalloc(&ws.x, (size_t) 16384 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&ws.y, (size_t) 3 * 16384 * sizeof(float)) != cudaSuccess ||
            cudaMalloc(&ws.q, strata::kernels::native_q8_1_bytes(16384)) != cudaSuccess) return false;
    }
    if (count < 1 || count > 3 || n_in > 16384 || n_out > 16384) return false;
    if (cudaMemcpyAsync(ws.x, x, (size_t) n_in * sizeof(float), cudaMemcpyHostToDevice, ws.stream) != cudaSuccess)
        return false;
    strata::kernels::native_quantize_q8_1(ws.x, ws.q, n_in, 1, (void*) ws.stream);
    for (int i = 0; i < count; ++i) {
        if (!strata::kernels::native_mmvq_supported(types[i])) return false;
        float* dy = ws.y + (size_t) i * 16384;
        strata::kernels::native_mmvq(types[i], weights[i], ws.q, dy, n_in, n_out, 1, (void*) ws.stream);
        if (cudaMemcpyAsync(out[i], dy, (size_t) n_out * sizeof(float), cudaMemcpyDeviceToHost, ws.stream) != cudaSuccess)
            return false;
    }
    return cudaStreamSynchronize(ws.stream) == cudaSuccess;
}

static bool native_glm_ffn(const void* const* weights, const int* types, const K::MoeGeometry& g,
                           const float* x, float* out, float clamp_limit) {
    std::vector<float> gate((size_t) g.ff), up((size_t) g.ff), h((size_t) g.ff);
    float* gu_out[2] = {gate.data(), up.data()};
    if (!native_kda_project(2, weights, types, x, g.n_embd, g.ff, gu_out)) return false;
    const bool clamp = clamp_limit > 1e-6f;
    for (int i = 0; i < g.ff; ++i) {
        float a = gate[(size_t) i], u = up[(size_t) i];
        if (clamp) { a = std::min(a, clamp_limit); u = std::max(-clamp_limit, std::min(u, clamp_limit)); }
        h[(size_t) i] = (a / (1.0f + std::exp(-a))) * u;
    }
    const void* down[1] = {weights[2]};
    const int down_type[1] = {types[2]};
    float* down_out[1] = {out};
    return native_kda_project(1, down, down_type, h.data(), g.ff, g.n_embd, down_out);
}

static bool provider(void* raw, int layer, C::glm::GlmTrunkLayerWeights& out, std::string& err) {
    Provider* p = (Provider*) raw;
    if (layer < 0 || layer >= N_LAYERS) { err = "layer out of range"; return false; }
    // Retain CPU-consumed dequantized tensors per layer across tokens/requests. The current sweep materializes
    // about 1.55 GiB; freeing it each layer forced that same work to repeat for every prompt and decode token.
    {
        C::GlmBoundBlock& B = p->bound[(size_t) layer];
        const bool dense = (layer < 3);
        const bool kda_layer = C::glm::glm_attention_is_mla(layer) == 0;
        const bool routed_layer = !dense;
        for (size_t i = 0; i < B.tensors.size(); ++i) {
            C::GlmBoundBlock::Tensor& t = B.tensors[i];
            if (!t.quantized || t.ptr == nullptr) continue;
            const bool native_projection = kda_layer &&
                (t.name == "attn_q.weight" || t.name == "attn_k.weight" ||
                 t.name == "attn_v.weight" || t.name == "attn_output.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_shared = routed_layer &&
                (t.name == "ffn_gate_shexp.weight" || t.name == "ffn_up_shexp.weight" ||
                 t.name == "ffn_down_shexp.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_dense_ffn = dense &&
                (t.name == "ffn_gate.weight" || t.name == "ffn_up.weight" || t.name == "ffn_down.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_mla = !kda_layer && !dense &&
                (t.name == "attn_q_a.weight" || t.name == "attn_q_b.weight" ||
                 t.name == "attn_kv_a_mqa.weight" || t.name == "attn_output.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            if (native_projection || native_shared || native_mla || native_dense_ffn) continue;
            // NO EXCEPTIONS, and the measurement is why: I first skipped the dense FFN's three tensors on blocks
            // 0..2, on the assumption that a GPU path consumed them.  It does not - glm_stage_ffn calls the CPU
            // expert_ffn, so the crash simply moved from kda_forward to expert_ffn when the rest were staged.  Every
            // quantized tensor is staged, and the per-layer lifetime is what keeps it affordable.
            (void) dense;
            float* h = dequant_to_host(t, err);
            if (h == nullptr) return false;
            p->staged_by_layer[(size_t) layer].push_back(h);
            g_staged_bytes += (unsigned long long) t.ne0 * (unsigned long long) (t.ne1 > 0 ? t.ne1 : 1) * 4ull;
            t.ptr = h;
            t.quantized = false;      // cached host representation remains valid for the provider lifetime
        }
        // re-map: glm_fill_layer_weights is pure assignment, so doing it per layer costs nothing
        if (!C::glm::glm_fill_layer_weights(B, layer, p->kda_g, p->mla_g, p->kda[(size_t) layer],
                                           p->mla[(size_t) layer], p->shexp_store[(size_t) layer].data(),
                                           p->w[(size_t) layer], err)) {
            err = "provider: re-mapping layer " + std::to_string(layer) + ": " + err;
            return false;
        }
        auto type_of = [&](const char* name) {
            for (const auto& t : B.tensors) if (t.name == name) return t.quantized ? t.native_type : 0;
            return 0;
        };
        if (kda_layer) {
            K::KdaWeights& kw = p->kda[(size_t) layer];
            kw.wq_type = type_of("attn_q.weight"); kw.wk_type = type_of("attn_k.weight");
            kw.wv_type = type_of("attn_v.weight"); kw.wo_type = type_of("attn_output.weight");
        } else {
            // THE MLA'S FOUR PROJECTIONS, typed the same way and for the same reason: a non-zero type tells mla_forward
            // that the pointer is the artifact's quantized blocks on the device, and the shared native projection hook
            // reads them in place.  These are the largest tensors in the model - q_b at 25,165,824 elements and wo at
            // 67,108,864 - so they are also the ones that stop being copied to host floats on every token.
            K::MlaWeights& mw = p->mla[(size_t) layer];
            mw.wq_a_type = type_of("attn_q_a.weight");
            mw.wq_b_type = type_of("attn_q_b.weight");
            mw.kv_a_type = type_of("attn_kv_a_mqa.weight");
            mw.wo_type = type_of("attn_output.weight");
        }
        if (dense) {
            C::glm::GlmTrunkLayerWeights& tw = p->w[(size_t) layer];
            tw.ffn_types[0] = type_of("ffn_gate.weight");
            tw.ffn_types[1] = type_of("ffn_up.weight");
            tw.ffn_types[2] = type_of("ffn_down.weight");
        }
        if (routed_layer) {
            C::glm::GlmTrunkLayerWeights& tw = p->w[(size_t) layer];
            tw.shexp_types[0] = type_of("ffn_gate_shexp.weight");
            tw.shexp_types[1] = type_of("ffn_up_shexp.weight");
            tw.shexp_types[2] = type_of("ffn_down_shexp.weight");
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
        std::fprintf(stderr, "usage: strata-glm <pack-dir> <gguf-shard1> [hc-init.bin | --serve] [w-output.bin] [w-norm.bin]\n");
        return 2;
    }
    const std::string pack = argv[1];
    const std::string shard1 = argv[2];
    const bool serve = argc > 3 && std::string(argv[3]) == "--serve";
    const std::string in_path = !serve && argc > 3 ? argv[3] : "/home/peb/moredata/glm5-oracle-full/hc_init.bin";
    const std::string wout_path = argc > (serve ? 4 : 4) ? argv[4] : "/home/peb/moredata/glm5-head-gate/w_output.bin";
    const std::string wnorm_path = argc > (serve ? 5 : 5) ? argv[5] : "/home/peb/moredata/glm5-head-gate/w_output_norm.bin";
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
    C::NativeEmbed native_embed;
    if (serve && (shards.size() < 2 || !native_embed.load(shards, N_EMBD, 154880, err))) {
        std::fprintf(stderr, "embedding: %s\n", err.c_str());
        return 1;
    }
    // THE CUDA HEAD IS LOADED EVEN IN THE DIAGNOSTIC RUN.  It is not gated on serve any more, because the head is the
    // largest single item in the profile - 3.1 of 4.8 seconds per token for one 154,880 x 4,096 projection - and its
    // device path already exists here; gating it on serve meant the acceptance run measured the CPU projection and
    // reported it as the trunk's cost.  Unavailable is a fallback rather than a failure, so the gate still runs on a
    // machine or a shard set where the native head cannot load.
    C::NativeHead native_head;
    if (shards.size() < 2 || !native_head.load(shards, N_EMBD, 154880, err)) {
        std::fprintf(stderr, "output head: CUDA head unavailable (%s) - falling back to the CPU projection\n",
                     err.c_str());
    }
    const bool head_cuda = native_head.loaded();
    std::printf("head: %s\n", head_cuda ? "native (CUDA) over 154880 x 4096" : "CPU projection (fallback)");
    std::printf("arena %.2f GB, %zu tensors served from the GGUF\n", bytes / 1073741824.0, served.size());

    // ---- geometry.  The clamps are set EXPLICITLY: their defaults are 0.0f, which DISABLES them, and the header
    // says so in as many words.  A routed pre-activation past 10 would then be passed through unclamped, silently.
    Provider P;
    P.staged_by_layer.resize(N_LAYERS);
    K::kda_set_native_project(&native_kda_project);
    // The same function serves the MLA: the signature is already generic (weights, types, shapes), so there is no reason
    // for a second implementation to exist and drift from this one.
    K::mla_set_native_project(&native_kda_project);
    C::glm::glm_set_native_ffn(&native_glm_ffn);
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

    // Keep a bounded set of routed expert rows resident. The default leaves ample room for the model's other CUDA
    // allocations while fitting the measured hot set; misses upload once and then reuse the device pointer.
    GlmExpertDeviceRuntime expert_device_runtime((size_t) 18 * 1024 * 1024 * 1024);
    C::glm::glm_set_device_expert_ffn(&run_cached_device_expert, &expert_device_runtime);
    C::glm::glm_set_device_moe_ffn(&run_cached_device_moe, &expert_device_runtime);

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
    const int cache_cells = serve ? 8192 : TOKENS;
    std::vector<std::vector<float> > kda_state(N_LAYERS);
    std::vector<std::vector<float> > kda_conv(N_LAYERS);
    std::vector<std::vector<float> > mla_cache(N_LAYERS);
    float* kda_ptrs[N_LAYERS];
    float* kda_conv_ptrs[N_LAYERS];
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
            mla_cache[(size_t) b].assign((size_t) cache_cells * KV_LORA, 0.0f);
            mla_index[b] = n_mla;
            mla_ptrs[n_mla] = mla_cache[(size_t) b].data();
            ++n_mla;
        } else {
            kda_state[(size_t) b].assign((size_t) NH * HD * HD, 0.0f);
            kda_conv[(size_t) b].assign((size_t) 3 * (P.kda_g.d_conv - 1) * P.kda_g.d_inner(), 0.0f);
            kda_index[b] = n_kda;
            kda_ptrs[n_kda] = kda_state[(size_t) b].data();
            kda_conv_ptrs[n_kda] = kda_conv[(size_t) b].data();
            ++n_kda;
        }
    }
    std::printf("state: %d KDA slots (%.0f MB), %d MLA caches (%d cells each)\n", n_kda,
                n_kda * (double) NH * HD * HD * 4.0 / 1048576.0, n_mla, cache_cells);

    C::glm::GlmTrunkState st;
    st.kda_state = kda_ptrs;
    st.kda_conv = kda_conv_ptrs;
    st.kda_index = kda_index;
    st.mla_cache = mla_ptrs;
    st.mla_len = mla_len;
    st.mla_index = mla_index;

        // ---- the head: load its small norm and large projection once.  Serve mode keeps both resident across requests.
    std::vector<float> onorm;
    int ne_n[4] = {0, 0, 0, 0};
    // ---- output_norm: read it, and REFUSE rather than pass an empty vector on.
    //
    // This is the whole of the head's null.  onorm.data() on an EMPTY std::vector is guaranteed to be nullptr - not
    // garbage, nullptr - so when read_dump parsed this file and produced no elements, the null the head reported was
    // created right here, one line earlier, and the head was telling the truth about an argument this code handed it.
    // The lesson is the same one this port keeps relearning: an argument that is null is not always a binding failure.
    bool have_norm = false;
    if (serve) {
        const C::WeightRef* nr = table.find("output_norm.weight");
        if (nr && nr->data && nr->bytes == (uint64_t) N_EMBD * sizeof(float)) {
            onorm.assign((const float*) nr->data, (const float*) nr->data + N_EMBD);
            have_norm = true;
        }
    } else {
        have_norm = read_dump(wnorm_path, onorm, ne_n) && (int) onorm.size() == N_EMBD;
    }
    if (!have_norm) {
        onorm.assign((size_t) N_EMBD, 0.0f);
        std::FILE* f = std::fopen(wnorm_path.c_str(), "rb");
        if (f != nullptr) {
            const size_t got = std::fread(onorm.data(), sizeof(float), (size_t) N_EMBD, f);
            std::fclose(f);
            have_norm = (got == (size_t) N_EMBD);
        }
    }
    if (!have_norm) {
        std::fprintf(stderr, "output_norm: %s is neither a dump with %d elements nor %d raw floats\n",
                     wnorm_path.c_str(), N_EMBD, N_EMBD);
        return 1;
    }
    {
        double s2 = 0.0;
        for (float v : onorm) s2 += (double) v * v;
        std::printf("output_norm: %zu floats, rms %.6g, first 3: %.6g %.6g %.6g\n", onorm.size(),
                    std::sqrt(s2 / (double) onorm.size()), onorm[0], onorm[1], onorm[2]);
    }
    const long vocab = 154880;
    std::vector<float> output_w;
    if (!serve) {
        std::ifstream wf(wout_path, std::ios::binary);
        if (!wf) { std::fprintf(stderr, "cannot read %s\n", wout_path.c_str()); return 1; }
        output_w.resize((size_t) vocab * N_EMBD);
        wf.read((char*) output_w.data(), (std::streamsize) (output_w.size() * sizeof(float)));
        if (!wf) { std::fprintf(stderr, "short read from %s\n", wout_path.c_str()); return 1; }
    }

    std::vector<float> l_out((size_t) HC * N_EMBD), hidden((size_t) N_EMBD), x((size_t) HC * N_EMBD);
    float *d_embed = nullptr, *d_hidden = nullptr, *d_logits = nullptr;
    cudaStream_t head_stream = nullptr;
    std::vector<float> logits(head_cuda ? (size_t) vocab : 0);
    if (head_cuda && (cudaMalloc(&d_embed, (size_t) N_EMBD * sizeof(float)) != cudaSuccess ||
                  cudaMalloc(&d_hidden, (size_t) N_EMBD * sizeof(float)) != cudaSuccess ||
                  cudaMalloc(&d_logits, (size_t) vocab * sizeof(float)) != cudaSuccess ||
                  cudaStreamCreateWithFlags(&head_stream, cudaStreamNonBlocking) != cudaSuccess)) {
        std::fprintf(stderr, "embedding/head scratch allocation failed\n"); return 1;
    }
    auto reset_state = [&]() {
        for (auto& v : kda_state) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : kda_conv) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : mla_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (int i = 0; i < N_LAYERS; ++i) mla_len[i] = 0;
    };
    auto run_x = [&](const float* in, int pos, bool diagnostic, int& best, float& best_logit) -> bool {
        StageCount sc; sc.token = pos;
        if (!C::glm::glm_trunk_forward(in, N_LAYERS, provider, &P, P.kda_g, P.mla_g, EPS, st, l_out.data(), nullptr,
                                       err, 0, diagnostic ? &stage_cb : nullptr, diagnostic ? &sc : nullptr)) return false;
        if (!C::glm::glm_stage_head_mean_norm(l_out.data(), HC, N_EMBD, onorm.data(), hidden.data(), err)) return false;
        if (!head_cuda)
            return C::glm::glm_stage_head_project(output_w.data(), (int) vocab, N_EMBD, hidden.data(), best, best_logit,
                                                  err);
        if (cudaMemcpyAsync(d_hidden, hidden.data(), (size_t) N_EMBD * sizeof(float), cudaMemcpyHostToDevice,
                            head_stream) != cudaSuccess ||
            !native_head.run(d_hidden, d_logits, (void*) head_stream, err) ||
            cudaMemcpyAsync(logits.data(), d_logits, (size_t) vocab * sizeof(float), cudaMemcpyDeviceToHost,
                            head_stream) != cudaSuccess || cudaStreamSynchronize(head_stream) != cudaSuccess) {
            if (err.empty()) err = "native output head failed";
            return false;
        }
        best = (int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        best_logit = logits[(size_t) best];
        return true;
    };
    auto embed = [&](int64_t tok) -> bool {
        if (tok < 0 || tok >= vocab) { err = "token outside vocabulary"; return false; }
        native_embed.gather_one(tok, d_embed, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(x.data(), d_embed, (size_t) N_EMBD * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "token embedding gather failed"; return false;
        }
        for (int h = 1; h < HC; ++h)
            std::memcpy(x.data() + (size_t) h * N_EMBD, x.data(), (size_t) N_EMBD * sizeof(float));
        return true;
    };

    if (serve) {
        std::printf("READY glm-5.3-flash %ld 8192\n", vocab); std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected GEN <max_new> <id,id,...>\n"); std::fflush(stdout); continue; }
            char* ep = nullptr; long max_new = std::strtol(line.c_str() + 4, &ep, 10);
            while (ep && *ep == ' ') ++ep;
            std::vector<int64_t> ids; std::string item; std::stringstream ss(ep ? ep : "");
            while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::strtoll(item.c_str(), nullptr, 10));
            if (max_new < 1 || ids.empty() || ids.size() + (size_t) max_new > (size_t) cache_cells) {
                std::printf("ERR invalid request or context too long\n"); std::fflush(stdout); continue;
            }
            reset_state(); int best = -1; float logit = 0.0f; bool ok = true; int pos = 0;
            for (int64_t tok : ids) {
                if (!embed(tok) || !run_x(x.data(), pos++, false, best, logit)) { ok = false; break; }
            }
            long produced = 0;
            while (ok && produced < max_new) {
                std::printf("T %d\n", best); std::fflush(stdout); ++produced;
                if (best == 154820 || best == 154827 || produced == max_new) break;
                if (!embed(best) || !run_x(x.data(), pos++, false, best, logit)) ok = false;
            }
            if (ok) std::printf("DONE %ld %zu\n", produced, ids.size());
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
        }
        cudaFree(d_embed);
        cudaFree(d_hidden);
        cudaFree(d_logits);
        cudaStreamDestroy(head_stream);
        std::fprintf(stderr, "%s\n", expert_device_runtime.cache.report().c_str());
        return 0;
    }

    // Diagnostic acceptance run: preserve the oracle fixture gate.
    std::vector<float> inp; int ne_in[4] = {0,0,0,0};
    if (!read_dump(in_path, inp, ne_in) || ne_in[0] != N_EMBD || ne_in[1] != HC || ne_in[2] < TOKENS) {
        std::fprintf(stderr, "cannot read compatible hc_init from %s\n", in_path.c_str()); return 1;
    }
    reset_state(); int best = -1; float best_logit = 0.0f;
    for (int t = 0; t < TOKENS; ++t)
        if (!run_x(inp.data() + (size_t) t * N_EMBD * HC, t, true, best, best_logit)) {
            std::fprintf(stderr, "trunk failed at token %d: %s\n", t, err.c_str()); return 1;
        }
    {
        double total = 0.0, stem = 0.0, kda_ms = 0.0, mla_ms = 0.0;
        for (const std::pair<const int, double>& kv : g_prof.ms) {
            total += kv.second;
            if (kv.first < 3) stem += kv.second;
            else if (C::glm::glm_attention_is_mla(kv.first) == 1) mla_ms += kv.second;
            else kda_ms += kv.second;
        }
        if (total > 0.0) {
            std::pair<int, double> worst(-1, -1.0);
            for (const std::pair<const int, double>& kv : g_prof.ms) if (kv.second > worst.second) worst = kv;
            std::fprintf(stderr, "PROFILE %.1f ms for %d tokens = %.1f ms/token (%.2f tok/s)\n",
                         total, TOKENS, total / TOKENS, 1000.0 * TOKENS / total);
            std::fprintf(stderr, "  stem 0-2 : %8.1f ms %5.1f%%\n", stem, 100.0 * stem / total);
            std::fprintf(stderr, "  34 KDA   : %8.1f ms %5.1f%%\n", kda_ms, 100.0 * kda_ms / total);
            std::fprintf(stderr, "  11 MLA   : %8.1f ms %5.1f%%\n", mla_ms, 100.0 * mla_ms / total);
            std::fprintf(stderr, "  heaviest block: layer %d at %.1f ms\n", worst.first, worst.second);
            std::fprintf(stderr, "  host staging: %.2f GB cached across layers and materialised once\n",
                         (double) g_staged_bytes / 1073741824.0);
                    {
            std::vector<std::pair<std::string, double> > v(g_prof.by_stage.begin(), g_prof.by_stage.end());
            std::sort(v.begin(), v.end(), [](const std::pair<std::string, double>& a,
                                             const std::pair<std::string, double>& b) { return a.second > b.second; });
            std::fprintf(stderr, "  by stage (interval closed by the named callback):\n");
            for (size_t i = 0; i < v.size() && i < 10; ++i)
                std::fprintf(stderr, "    %-16s %8.1f ms %5.1f%%\n", v[i].first.c_str(), v[i].second,
                             100.0 * v[i].second / total);
        }
        std::fprintf(stderr, "  per layer ms:");
            for (const std::pair<const int, double>& kv : g_prof.ms) std::fprintf(stderr, " %d:%.1f", kv.first, kv.second);
            std::fprintf(stderr, "\n");
        }
    }
    // MAJOR VS MINOR FAULTS, because it is the number that says whether the routed-expert stage's 861.7 ms per token is
    // really disk I/O: a major fault is a page that had to come from the device, a minor fault is one that was already
    // in the page cache.  0.44 GB per token at 4 KB pages is about 115,000 major faults per token if the expert rows are
    // genuinely cold every time; far fewer means they are cache hits and the stage is CPU dequantisation after all,
    // which would send the work back to a device MMVQ kernel rather than to a residency plan.
    {
        struct rusage ru;
        std::memset(&ru, 0, sizeof ru);
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            std::fprintf(stderr, "FAULTS minflt %ld majflt %ld inblock %ld oublock %ld\n", ru.ru_minflt, ru.ru_majflt,
                         ru.ru_inblock, ru.ru_oublock);
            std::fprintf(stderr, "       majflt x 4096 = %.2f GB of pages faulted in from the device\n",
                         (double) ru.ru_majflt * 4096.0 / 1073741824.0);
        }
    }
    K::kda_print_profile();
    std::printf("\nargmax = %d   (logit %.6f)   expected 12089\n", best, best_logit);
    std::printf("ENGINE TOKEN: %s\n", best == 12089 ? "PASS" : "MISMATCH");
    std::fprintf(stderr, "%s\n", expert_device_runtime.cache.report().c_str());
    return best == 12089 ? 0 : 1;
}
