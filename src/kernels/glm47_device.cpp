// src/kernels/glm47_device.cpp - GLM-4.7-Flash on the GPU: the native-projection hook.
//
// The runner reads the artifact directly and keeps the artifact's OWN quantized blocks resident on the
// device; this module is the device side of that.  It installs the MLA native-projection adapter
// (mla_set_native_project) - the same over-native_mmvq shape the KDA already uses, so one implementation
// serves both - which turns mla_forward_rope's wq_a/wq_b/kv_a/wo sites from the float host GEMM into the
// engine's pinned-to-llama.cpp quantized device GEMVs, and it exposes the upload the runner binds with.
//
// The host path is untouched and still the fallback: if no type is set, the hook is absent, or it fails,
// mla_forward_rope falls back to its float GEMM (see glm_mla.cpp).  So a missing block degrades to slow,
// never to wrong.
#include "strata/core/glm_moe_native.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <chrono>
#include <cuda_runtime.h>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::kernels::glm {
namespace {

// One non-blocking stream for the whole projection path: the calls are ordered on it, so a batch of
// independent GEMVs sharing one activation serialises without extra synchronisation.
cudaStream_t project_stream() {
    static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        return st;
    }();
    return s;
}

// Per-shape device scratch, grown in place.  x is the uploaded activation, xq its Q8_1 image, y the GEMV
// output.  All are reused across every projection of every layer, so the per-token transfer count stays
// at most two H2D/D2H pairs per site regardless of model depth.
struct Scratch { float* x = nullptr; void* xq = nullptr; float* y = nullptr;
                 size_t xc = 0, qc = 0, yc = 0; };
Scratch& scratch() { static Scratch s; return s; }

// The GLU FFN's own scratch: the activation, its Q8_1 image, the two hidden projections (gate/up), the
// SwiGLU Q8_1 image and the output.
struct FfnScratch { float* x = nullptr; void* xq = nullptr; float* gt = nullptr; float* up = nullptr;
                    void* hq = nullptr; float* y = nullptr;
                    size_t xc = 0, xqc = 0, gtc = 0, upc = 0, hqc = 0, yc = 0; };
FfnScratch& ffn_scratch() { static FfnScratch s; return s; }

// Cumulative wall time (ms) inside the hooks, so a runner can weigh the device round-trips against the
// host stages.
double g_proj_ms = 0.0, g_ffn_ms = 0.0;
double now_ms() {
    using clk = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(clk::now().time_since_epoch()).count();
}

bool grow(void*& p, size_t& cap, size_t need) {
    if (need <= cap) return true;
    if (p) { cudaFree(p); p = nullptr; cap = 0; }
    if (cudaMalloc(&p, need) != cudaSuccess) return false;
    cap = need;
    return true;
}
bool grow_f32(float*& p, size_t& cap, size_t need) {
    void* v = p; size_t c = cap;
    if (!grow(v, c, need)) return false;
    p = static_cast<float*>(v); cap = c;
    return true;
}

}  // namespace

// H2D the activation, one Q8_1 quantise, then one native_mmvq per weight, D2H each result.  `count` weight
// matrices share the single input x[n_in]; weight i is [n_out][n_in] and produces out[i][n_out].  Returns
// false (and the caller keeps its float result) on any unsupported type or CUDA fault.
static bool project_impl(int count, const void* const* weights, const int* types,
                         const float* x, int n_in, int n_out, float* const* out) {
    if (count < 1 || !weights || !types || !x || !out || n_in <= 0 || n_out <= 0) return false;
    for (int i = 0; i < count; ++i)
        if (types[i] == 0 || !weights[i] || !out[i] || !native_mmvq_supported(types[i])) return false;
    cudaStream_t st = project_stream();
    if (!st) return false;
    Scratch& s = scratch();
    const size_t xb = (size_t) n_in * sizeof(float);
    const size_t qb = native_q8_1_bytes(n_in, 1);
    const size_t yb = (size_t) n_out * sizeof(float);
    if (!grow_f32(s.x, s.xc, xb) || !grow(s.xq, s.qc, qb) || !grow_f32(s.y, s.yc, yb)) return false;
    if (cudaMemcpyAsync(s.x, x, xb, cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    native_quantize_q8_1(s.x, s.xq, n_in, 1, st);
    for (int i = 0; i < count; ++i) {
        native_mmvq(types[i], weights[i], s.xq, s.y, n_in, n_out, 1, st);
        if (cudaMemcpyAsync(out[i], s.y, yb, cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    }
    return cudaStreamSynchronize(st) == cudaSuccess;
}

// The installed hook (mla_set_native_project): project_impl with its wall time accumulated.
bool glm47_native_project(int count, const void* const* weights, const int* types,
                          const float* x, int n_in, int n_out, float* const* out) {
    const double t0 = now_ms();
    const bool ok = project_impl(count, weights, types, x, n_in, n_out, out);
    g_proj_ms += now_ms() - t0;
    return ok;
}

double glm47_project_ms() { return g_proj_ms; }
double glm47_ffn_ms() { return g_ffn_ms; }

// The native shared-expert GLU (installed with glm_set_native_ffn): gate = mmvq(gate,x); up = mmvq(up,x);
// h = silu(gate)*up quantised to Q8_1 (native_swiglu_quantize_q8_1); out = mmvq(down,h).  weights are the
// artifact's own quantized blocks on the device.  GLM-4.7 clamps nowhere, so a non-zero limit declines and
// the caller keeps its float expert.
static bool ffn_impl(const void* const* weights, const int* types, const MoeGeometry& g,
                     const float* x, float* out) {
    if (!weights || !types || !x || !out || g.n_embd <= 0 || g.ff <= 0) return false;
    for (int i = 0; i < 3; ++i)
        if (!weights[i] || !types[i] || !native_mmvq_supported(types[i])) return false;
    cudaStream_t st = project_stream();
    if (!st) return false;
    FfnScratch& s = ffn_scratch();
    const int ne = g.n_embd, nf = g.ff;
    if (!grow_f32(s.x, s.xc, (size_t) ne * sizeof(float)) ||
        !grow(s.xq, s.xqc, native_q8_1_bytes(ne, 1)) ||
        !grow_f32(s.gt, s.gtc, (size_t) nf * sizeof(float)) ||
        !grow_f32(s.up, s.upc, (size_t) nf * sizeof(float)) ||
        !grow(s.hq, s.hqc, native_q8_1_bytes(nf, 1)) ||
        !grow_f32(s.y, s.yc, (size_t) ne * sizeof(float))) return false;
    if (cudaMemcpyAsync(s.x, x, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    native_quantize_q8_1(s.x, s.xq, ne, 1, st);
    native_mmvq(types[0], weights[0], s.xq, s.gt, ne, nf, 1, st);
    native_mmvq(types[1], weights[1], s.xq, s.up, ne, nf, 1, st);
    native_swiglu_quantize_q8_1(s.gt, s.up, s.hq, nf, 1, st);
    native_mmvq(types[2], weights[2], s.hq, s.y, nf, ne, 1, st);
    if (cudaMemcpyAsync(out, s.y, (size_t) ne * sizeof(float), cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    return cudaStreamSynchronize(st) == cudaSuccess;
}

bool glm47_native_ffn(const void* const* weights, const int* types, const MoeGeometry& g,
                      const float* x, float* out, float clamp_limit) {
    if (clamp_limit != 0.0f) return false;   // no clamped native kernel; leave the float expert in charge
    const double t0 = now_ms();
    const bool ok = ffn_impl(weights, types, g, x, out);
    g_ffn_ms += now_ms() - t0;
    return ok;
}

// ---- the routed experts on the device, BATCHED. ----
//
// The engine's device expert path is iq_mmvq, whose dispatch covers the IQ types and Q2_K but NOT the
// k-quants; this model's experts are Q4_K (gate/up) and Q6_K (down), so glm_expert_layer_supported is false
// and the stage falls back to the CPU.  native_mmvq - the kernel the MLA projections, the shared GLU and the
// dense stem all use - DOES cover the k-quants, and a routed expert is the same GLU shape as the shared one:
// gate = mmvq(gu, row); up = mmvq(gu, row + up_off); h = swiglu*quantise; out = mmvq(down, row + down_off).
//
// ONE expert at a time is a loss (measured: experts 3663 vs 1065 ms): every expert paid its own activation
// upload, Q8_1 quantise and stream sync.  Batching the layer's n_used experts amortises all three - the
// shape the V100's grouped MoE fix had (35e627f: '1,200 mmvq launches and a host round trip per layer, which
// is what the grouped device MoE is for').  Rows are uploaded once per (layer, expert) and reused, FIFO-
// evicted past the budget, so a warm row costs no PCIe traffic.
namespace {
struct ExpertRowDev {
    struct Slot { void* dev = nullptr; size_t bytes = 0; };
    std::unordered_map<int64_t, Slot> rows;
    std::vector<int64_t> fifo;
    size_t bytes = 0, budget = 0;
    uint64_t hits = 0, misses = 0, evicted = 0;
};
ExpertRowDev& experts() { static ExpertRowDev c; return c; }

const uint8_t* ensure_row(ExpertRowDev* c, int layer, int expert, const uint8_t* blob, size_t nb,
                          const char** err) {
    const int64_t key = ((int64_t) layer << 20) | (int64_t) expert;
    auto it = c->rows.find(key);
    if (it != c->rows.end()) { ++c->hits; return (const uint8_t*) it->second.dev; }
    while (c->budget != 0 && c->bytes + nb > c->budget && !c->fifo.empty()) {
        const int64_t ev = c->fifo.front();
        c->fifo.erase(c->fifo.begin());
        auto e = c->rows.find(ev);
        if (e == c->rows.end()) continue;
        if (e->second.dev) cudaFree(e->second.dev);
        c->bytes -= e->second.bytes;
        c->rows.erase(e);
        ++c->evicted;
    }
    void* dev = nullptr;
    if (cudaMalloc(&dev, nb) != cudaSuccess) { *err = "expert row cudaMalloc failed"; return nullptr; }
    if (cudaMemcpy(dev, blob, nb, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dev); *err = "expert row upload failed"; return nullptr;
    }
    c->rows.emplace(key, ExpertRowDev::Slot{dev, nb});
    c->fifo.push_back(key);
    c->bytes += nb;
    ++c->misses;
    return (const uint8_t*) dev;
}

// One activation, one Q8_1, then per-expert gate/up/SwiGLU-Q8_1/down on the stream; one D2H and one sync.
struct MoeScratch {
    float* x = nullptr; void* xq = nullptr; size_t xc = 0, xqc = 0;
    float* gate = nullptr; size_t gtc = 0;
    float* up = nullptr; size_t upc = 0;
    void* hq = nullptr; size_t hqc = 0;
    float* outp = nullptr; size_t outc = 0;
    std::vector<float> hostout;   // the combine's staging, grown once
};
MoeScratch& moe_scratch() { static MoeScratch s; return s; }

bool moe_batch_impl(const uint8_t* const* rows, const int* types, int64_t n_embd, int64_t n_ff, size_t up_off,
                    size_t down_off, const float* weights, int n_used, const float* x, float* out) {
    if (n_used < 1 || n_used > 64 || !rows || !types || !weights || !x || !out) return false;
    const int gu = types[0], dt = types[2];
    if (!native_mmvq_supported(gu) || !native_mmvq_supported(dt)) return false;
    cudaStream_t st = project_stream();
    if (!st) return false;
    const int ne = (int) n_embd, nf = (int) n_ff;
    const size_t hqb = native_q8_1_bytes(nf, 1);
    MoeScratch& s = moe_scratch();
    if (!grow_f32(s.x, s.xc, (size_t) ne * sizeof(float)) ||
        !grow(s.xq, s.xqc, native_q8_1_bytes(ne, 1)) ||
        !grow_f32(s.gate, s.gtc, (size_t) n_used * nf * sizeof(float)) ||
        !grow_f32(s.up, s.upc, (size_t) n_used * nf * sizeof(float)) ||
        !grow(s.hq, s.hqc, (size_t) n_used * hqb) ||
        !grow_f32(s.outp, s.outc, (size_t) n_used * ne * sizeof(float))) return false;
    if (cudaMemcpyAsync(s.x, x, (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess) return false;
    native_quantize_q8_1(s.x, s.xq, ne, 1, st);
    for (int i = 0; i < n_used; ++i) {
        float* gi = s.gate + (size_t) i * nf;
        float* ui = s.up + (size_t) i * nf;
        float* oi = s.outp + (size_t) i * ne;
        void* hi = (uint8_t*) s.hq + (size_t) i * hqb;
        native_mmvq(gu, rows[i], s.xq, gi, ne, nf, 1, st);
        native_mmvq(gu, rows[i] + up_off, s.xq, ui, ne, nf, 1, st);
        native_swiglu_quantize_q8_1(gi, ui, hi, nf, 1, st);
        native_mmvq(dt, rows[i] + down_off, hi, oi, nf, ne, 1, st);
    }
    s.hostout.resize((size_t) n_used * ne);
    if (cudaMemcpyAsync(s.hostout.data(), s.outp, (size_t) n_used * ne * sizeof(float),
                        cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;
    // combine in the SAME order the CPU path does: out[j] = sum_i weights[i] * expert_i[j]
    for (int j = 0; j < ne; ++j) {
        float acc = 0.0f;
        for (int i = 0; i < n_used; ++i) acc += s.hostout[(size_t) i * ne + j] * weights[i];
        out[j] = acc;
    }
    return true;
}

bool moe_hook(void* ctx, int layer, int n_experts, const int32_t* experts_, const uint8_t* const* blobs,
              const float* weights, const strata::kernels::cpu::NativeFmt& f, const float* x, float* out,
              std::string& err) {
    ExpertRowDev* c = (ExpertRowDev*) ctx;
    if (c == nullptr || !experts_ || !blobs || !weights || !x || !out) { err = "moe hook: null argument"; return false; }
    const uint8_t* rows[64];
    for (int i = 0; i < n_experts; ++i) {
        const char* e = nullptr;
        rows[i] = ensure_row(c, layer, (int) experts_[i], blobs[i], f.bytes, &e);
        if (rows[i] == nullptr) { err = e ? e : "expert row"; return false; }
    }
    const int t[3] = { f.gu_type, f.gu_type, f.d_type };
    const double t0 = now_ms();
    const bool ok = moe_batch_impl(rows, t, f.n_embd, f.n_ff, f.up_off, f.down_off, weights, n_experts, x, out);
    g_ffn_ms += now_ms() - t0;
    return ok;
}
}  // namespace

void glm47_install_device_experts(size_t budget_bytes) {
    ExpertRowDev& c = experts();
    c.budget = budget_bytes;
    strata::core::glm::glm_set_device_moe_ffn(moe_hook, &c);
    strata::core::glm::glm_set_device_expert_native(true);
}
void glm47_device_expert_stats(uint64_t* hits, uint64_t* misses, uint64_t* evicted, size_t* bytes) {
    const ExpertRowDev& c = experts();
    if (hits) *hits = c.hits;
    if (misses) *misses = c.misses;
    if (evicted) *evicted = c.evicted;
    if (bytes) *bytes = c.bytes;
}

// A one-shot quantized GEMV for the sites that are not MLA projections (the head, the dense stem, the
// shared expert).  `w_dev` is the artifact's quantized block on the device; x and y are host.  Returns
// false if the type has no native kernel, leaving the caller's float path in charge.
bool glm47_device_gemv(int type, const void* w_dev, const float* x, int n_in, int n_out, float* y) {
    const void* w[1] = {w_dev};
    const int   t[1] = {type};
    float* o[1] = {y};
    return glm47_native_project(1, w, t, x, n_in, n_out, o);
}

}  // namespace strata::kernels::glm
