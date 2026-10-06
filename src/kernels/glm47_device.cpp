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
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/glm_norm.hpp"
#include "strata/core/glm47_trunk.hpp"

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
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

// The stream the device trunk chains a decoded token on.  Deliberately NOT project_stream(): the shared-expert
// host hook (glm47_native_ffn -> ffn_impl) synchronises project_stream() internally, and if the trunk shared
// it, every shared-expert call would drain the whole chained layer.  The two streams only ever meet through
// host memory that the host hook has already finished writing (the routing activation and the shared-expert
// output), so there is no cross-stream hazard.
cudaStream_t trunk_stream() {
    static cudaStream_t s = [] {
        cudaStream_t st = nullptr;
        cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
        return st;
    }();
    return s;
}

// The resident copy of a per-layer RMSNorm gain, keyed on the caller's pointer so the same layer's norm is
// uploaded once for the life of the process.  (A) of the device trunk: a token no longer re-uploads 47x2
// width-2048 vectors per layer, and no longer round-trips them through the host norm.
std::unordered_map<const float*, float*>& norm_dev_table() {
    static std::unordered_map<const float*, float*> m;
    return m;
}
float* norm_dev(const float* w, int64_t n, const char** err) {
    std::unordered_map<const float*, float*>& m = norm_dev_table();
    auto it = m.find(w);
    if (it != m.end()) return it->second;
    float* d = nullptr;
    if (cudaMalloc(&d, (size_t) n * sizeof(float)) != cudaSuccess) {
        *err = "trunk-dev: norm cudaMalloc failed";
        return nullptr;
    }
    if (cudaMemcpy(d, w, (size_t) n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        *err = "trunk-dev: norm upload failed";
        return nullptr;
    }
    m.emplace(w, d);
    return d;
}

// The device trunk's per-token scratch: the hidden state and the layer's intermediates, all [n_embd].  Grown
// once and reused for every layer and every token, so a decoded token allocates nothing per layer.
struct TrunkScratch {
    float* cur = nullptr; size_t curc = 0;          // the running hidden state (device)
    float* xb = nullptr; size_t xbc = 0;            // attn-normed input to the MLA block
    float* attn = nullptr; size_t attnc = 0;        // the MLA output
    float* ffv = nullptr; size_t ffvc = 0;          // ffn-normed input to the FFN
    float* ffnout = nullptr; size_t ffnoutc = 0;    // the FFN output (routed combine / dense)
    float* shr = nullptr; size_t shrc = 0;          // the shared expert's device result
    std::vector<float> host_ffv, host_shr, host_ffnout;   // host staging (the route + the two host hooks)
};
TrunkScratch& trunk_scratch() { static TrunkScratch s; return s; }

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
    float* wd = nullptr; size_t wdc = 0;          // the combine weights, on the device (the device trunk)
    std::vector<float> hostout;   // the combine's staging, grown once
    std::vector<float> comb;      // the host-combine result (exact mode)
};
MoeScratch& moe_scratch() { static MoeScratch s; return s; }

// When set, the routed combine uses the host loop's arithmetic (a separate rounding per weight*output term) instead
// of the device weighted_routes kernel (an FMA under fast-math).  The two differ by ~1 ulp, which is enough to
// cross a Q8_1 quantisation boundary a few layers deeper and flip a near-tie token; exact mode removes that seed.
static bool g_host_combine = false;

// The device-resident form of the batch MoE: the activation is ALREADY on the device and the combined result is
// LEFT there (no upload, no download), so a device trunk can feed it the previous stage's output and consume its
// result without the token's FFN ever crossing the bus.  Same kernels, same combine order (the rank-ordered
// weighted sum, `weighted_routes` - bit-identical to the host loop).
bool moe_batch_impl_dev(const uint8_t* const* rows, const int* types, int64_t n_embd, int64_t n_ff, size_t up_off,
                        size_t down_off, const float* weights, int n_used, const float* d_x, float* d_out,
                        cudaStream_t st) {
    if (n_used < 1 || n_used > 64 || !rows || !types || !weights || !d_x || !d_out || !st) return false;
    const int gu = types[0], dt = types[2];
    if (!native_mmvq_supported(gu) || !native_mmvq_supported(dt)) return false;
    const int ne = (int) n_embd, nf = (int) n_ff;
    const size_t hqb = native_q8_1_bytes(nf, 1);
    MoeScratch& s = moe_scratch();
    if (!grow_f32(s.x, s.xc, (size_t) ne * sizeof(float)) ||
        !grow(s.xq, s.xqc, native_q8_1_bytes(ne, 1)) ||
        !grow_f32(s.gate, s.gtc, (size_t) n_used * nf * sizeof(float)) ||
        !grow_f32(s.up, s.upc, (size_t) n_used * nf * sizeof(float)) ||
        !grow(s.hq, s.hqc, (size_t) n_used * hqb) ||
        !grow_f32(s.outp, s.outc, (size_t) n_used * ne * sizeof(float)) ||
        !grow_f32(s.wd, s.wdc, (size_t) n_used * sizeof(float))) return false;
    native_quantize_q8_1(d_x, s.xq, ne, 1, st);
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
    if (cudaMemcpyAsync(s.wd, weights, (size_t) n_used * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return false;
    if (g_host_combine) {
        // the host loop's exact combine: one rounding per weight*output term, accumulated on the host
        s.hostout.resize((size_t) n_used * ne);
        if (cudaMemcpyAsync(s.hostout.data(), s.outp, (size_t) n_used * ne * sizeof(float),
                            cudaMemcpyDeviceToHost, st) != cudaSuccess) return false;
        if (cudaStreamSynchronize(st) != cudaSuccess) return false;
        s.comb.resize((size_t) ne);
        for (int j = 0; j < ne; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < n_used; ++i) acc += s.hostout[(size_t) i * ne + j] * weights[i];
            s.comb[(size_t) j] = acc;
        }
        if (cudaMemcpyAsync(d_out, s.comb.data(), (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
            return false;
        return true;
    }
    weighted_routes(s.outp, s.wd, d_out, ne, 1, n_used, st);   // combine on the device
    return true;   // async: the caller's next synchronisation lands it
}

// The shared expert (the +1 GLU every MoE layer adds UNWEIGHTED) on the device: the exact chain ffn_impl runs
// for glm_try_native_ffn - quantise x, gate, up, SwiGLU->Q8_1, down - but fed the token's own device activation
// and leaving its result there, so the trunk pays no H2D/D2H and no extra synchronisation per layer.  Same
// kernels, same order, so it is bit-identical to the host hook.
bool shared_expert_dev(const float* const* w, const int* types, const MoeGeometry& g, const float* d_x,
                       float* d_y, cudaStream_t st) {
    if (!w || !types || !d_x || !d_y || !st) return false;
    if (!types[0] || !types[1] || !types[2]) return false;
    if (!native_mmvq_supported(types[0]) || !native_mmvq_supported(types[1]) || !native_mmvq_supported(types[2]))
        return false;
    const int ne = g.n_embd, nf = g.ff;
    if (ne <= 0 || nf <= 0) return false;
    MoeScratch& s = moe_scratch();
    const size_t hqb = native_q8_1_bytes(nf, 1);
    if (!grow(s.xq, s.xqc, native_q8_1_bytes(ne, 1)) || !grow_f32(s.gate, s.gtc, (size_t) nf * sizeof(float)) ||
        !grow_f32(s.up, s.upc, (size_t) nf * sizeof(float)) || !grow(s.hq, s.hqc, hqb)) return false;
    native_quantize_q8_1(d_x, s.xq, ne, 1, st);
    native_mmvq(types[0], w[0], s.xq, s.gate, ne, nf, 1, st);
    native_mmvq(types[1], w[1], s.xq, s.up, ne, nf, 1, st);
    native_swiglu_quantize_q8_1(s.gate, s.up, s.hq, nf, 1, st);
    native_mmvq(types[2], w[2], s.hq, d_y, nf, ne, 1, st);
    return true;
}

// The RMSNorm done exactly as the host loop does (double accumulation, 1/sqrt, (x*inv)*w) - the device
// rms_norm_weighted uses a float reduction and the fast rsqrtf, which is ~1 ulp off and, over 47 layers, seeds a
// Q8_1 boundary crossing.  Exact mode routes each norm through the host so the trunk is bit-faithful to the loop.
bool norm_host(const float* w, int ne, const float* d_x, float* d_y, float eps, cudaStream_t st) {
    static std::vector<float> hx, hy;
    hx.resize((size_t) ne);
    hy.resize((size_t) ne);
    // the input was written on the non-blocking trunk stream: land it BEFORE the blocking D2H, or the copy races it
    if (cudaStreamSynchronize(st) != cudaSuccess) return false;
    if (cudaMemcpy(hx.data(), d_x, (size_t) ne * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
    strata::kernels::glm::rms_norm_gain(w, ne, hx.data(), hy.data(), eps);
    if (cudaMemcpyAsync(d_y, hy.data(), (size_t) ne * sizeof(float), cudaMemcpyHostToDevice, st) != cudaSuccess)
        return false;
    return true;
}

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
    const auto _m0 = std::chrono::steady_clock::now();
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
    const auto _m1 = std::chrono::steady_clock::now();
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
    {
        const auto _m2 = std::chrono::steady_clock::now();
        static double _l = 0, _s = 0; static long _n = 0;
        _l += std::chrono::duration<double, std::milli>(_m1 - _m0).count();
        _s += std::chrono::duration<double, std::milli>(_m2 - _m1).count();
        if (++_n % 250 == 0)
            std::fprintf(stderr, "[moe-split] launch-loop %.1f ms | sync+D2H+combine %.1f ms over %ld calls\n", _l, _s, _n);
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

// ---- the device-resident trunk: one decoded token through every block, chained on ONE stream ----
//
// This mirrors glm47_trunk_forward's per-layer graph exactly (attn_norm -> MLA -> residual -> ffn_norm -> FFN
// -> residual) but keeps `cur` - the token's hidden state - on the device for the whole depth.  The host loop
// re-uploads the activation into every MLA projection, the routed MoE and the shared GLU and downloads every
// result; here the MLA block takes the attn-normed input on the device and leaves its output there, the routed
// experts consume that same device activation and leave their combine on the device, and only the ROUTER's
// activation crosses back (the router is still a host matvec in v1).  So the token's hidden state is uploaded
// once and downloaded once per token, plus one activation D2H per MoE layer for the route.
//
// Reused verbatim: glm::mla_block_launch_cuda (the bit-exact-at-depth-1-2 device MLA block), the elementwise
// rms_norm_weighted/add_inplace kernels, moe_batch_impl_dev (the routed MMVQ chain with the combine on the
// device), glm::moe_route, and glm_try_native_ffn for the dense stem and the shared expert.
bool glm47_trunk_forward_device(const strata::core::glm::Glm47TrunkLayer* layers, int n_layer,
                                const strata::kernels::glm::MlaGeometry& g, const float* x, int pos, float eps,
                                std::vector<std::vector<float>>* caches, void* expert_ctx, float* out,
                                std::vector<std::vector<int32_t>>* ids_out, char* err, size_t err_cap) {
    (void) expert_ctx;   // the resident expert-row cache is this module's own (ensure_row/experts())
    auto seterr = [&](const std::string& m) { if (err && err_cap) std::snprintf(err, err_cap, "%s", m.c_str()); };
    const float kTrunkClamp = 0.0f;   // GLM-4.7-Flash has no swiglu_clamp key (mirrors glm47_trunk.cpp)
    // The trunk is host-faithful (bit-for-bit glm47_trunk_forward) by default: the two RMSNorms and the routed
    // combine go through the host loop's exact arithmetic.  STRATA_GLM_TRUNK_DEV_FAST=1 selects the pure device
    // kernels instead (rms_norm_weighted + the weighted_routes combine) - marginally faster, but their float
    // reduction / FMA rounding is ~1 ulp off the loop and seeds a Q8_1 boundary crossing a few layers deep.
    const char* fast_env = std::getenv("STRATA_GLM_TRUNK_DEV_FAST");
    const bool exact = !(fast_env && fast_env[0] == '1');
    g_host_combine = exact;

    if (!layers || n_layer <= 0 || !x || !caches || !out) { seterr("trunk-dev: null argument"); return false; }
    const int ne = g.n_embd, kv_lora = g.kv_lora, n_rot = g.n_rot;
    const int kv_dim = kv_lora + n_rot;
    if (ne <= 0 || kv_lora <= 0 || n_rot <= 0 || pos < 0) { seterr("trunk-dev: bad geometry"); return false; }
    cudaStream_t st = trunk_stream();
    if (!st) { seterr("trunk-dev: no stream"); return false; }

    // In verify mode the host reference needs the caches as they were BEFORE this token (rows 0..pos-1);
    // snapshot them now, then run glm47_trunk_forward on the copy once the device token has landed.
    const bool verify = std::getenv("STRATA_GLM_TRUNK_DEV_VERIFY") != nullptr;
    const int  verify_mode = verify ? std::atoi(std::getenv("STRATA_GLM_TRUNK_DEV_VERIFY")) : 0;
    std::vector<std::vector<float>> vcopy;
    std::vector<std::vector<int32_t>> dev_ids;
    std::vector<std::vector<float>> dev_per;
    if (verify) {
        dev_ids.assign((size_t) n_layer, {});
        dev_per.assign((size_t) n_layer, {});
        if (verify_mode != 3) {   // mode 3: run the reference on the SAME caches (shares the block's device cache)
            vcopy.resize(caches->size());
            for (size_t i = 0; i < vcopy.size(); ++i) {
                const std::vector<float>& src = (*caches)[i];
                vcopy[i].reserve(src.size() + (size_t) 16384);   // headroom: the host reference must not reallocate
                vcopy[i].assign(src.begin(), src.end());
            }
        }
    }
    std::vector<std::vector<float>>& href_caches = (verify_mode == 3) ? *caches : vcopy;

    TrunkScratch& s = trunk_scratch();
    const size_t nb = (size_t) ne * sizeof(float);
    if (!grow_f32(s.cur, s.curc, nb) || !grow_f32(s.xb, s.xbc, nb) || !grow_f32(s.attn, s.attnc, nb) ||
        !grow_f32(s.ffv, s.ffvc, nb) || !grow_f32(s.ffnout, s.ffnoutc, nb) || !grow_f32(s.shr, s.shrc, nb)) {
        seterr("trunk-dev: scratch allocation failed");
        return false;
    }
    if (cudaMemcpyAsync(s.cur, x, nb, cudaMemcpyHostToDevice, st) != cudaSuccess) {
        seterr("trunk-dev: hidden-state upload failed");
        return false;
    }

    for (int l = 0; l < n_layer; ++l) {
        const strata::core::glm::Glm47TrunkLayer& ly = layers[(size_t) l];
        if (!ly.attn_norm || !ly.ffn_norm || !ly.mla) {
            seterr("trunk-dev: layer " + std::to_string(l) + " lacks a norm or MLA weights");
            return false;
        }
        const char* nerr = nullptr;
        float* an = nullptr;
        float* fn = nullptr;
        if (!exact) {   // exact mode computes both norms on the host from the layer's float weights
            an = norm_dev(ly.attn_norm, ne, &nerr);
            fn = norm_dev(ly.ffn_norm, ne, &nerr);
            if (!an || !fn) { seterr(nerr ? nerr : "trunk-dev: norm upload failed"); return false; }
        }

        // ---- attn site: xb = rms_norm(cur, attn_norm); MLA block; cur += attn ----
        if (exact) {
            if (!norm_host(ly.attn_norm, ne, s.cur, s.xb, eps, st)) {
                seterr("trunk-dev: attn norm failed");
                return false;
            }
        } else {
            if (cudaMemcpyAsync(s.xb, s.cur, nb, cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
                seterr("trunk-dev: attn input copy failed");
                return false;
            }
            strata::kernels::rms_norm_weighted(s.xb, an, 1, ne, eps, st);
        }
        std::vector<float>& cache = (*caches)[(size_t) l];
        cache.resize((size_t) (pos + 1) * kv_dim);   // appends the token's own latent+k_pe row, as the host loop does
        {
            const int r = strata::kernels::glm::mla_block_launch_cuda(*ly.mla, g, s.xb, pos + 1, cache.data(),
                                                                     cache.data() + (size_t) pos * kv_dim, s.attn,
                                                                     st, err, err_cap);
            if (r != 1) {
                seterr(r == 0 ? (err && err_cap && err[0] ? std::string(err) : std::string("trunk-dev: MLA block declined"))
                              : (err && err_cap && err[0] ? std::string(err) : std::string("trunk-dev: MLA block failed")));
                return false;
            }
        }
        strata::kernels::add_inplace(s.cur, s.attn, ne, st);

        // ---- ffn site: ffv = rms_norm(cur, ffn_norm) ----
        if (exact) {
            if (!norm_host(ly.ffn_norm, ne, s.cur, s.ffv, eps, st)) {
                seterr("trunk-dev: ffn norm failed");
                return false;
            }
        } else {
            if (cudaMemcpyAsync(s.ffv, s.cur, nb, cudaMemcpyDeviceToDevice, st) != cudaSuccess) {
                seterr("trunk-dev: ffn input copy failed");
                return false;
            }
            strata::kernels::rms_norm_weighted(s.ffv, fn, 1, ne, eps, st);
        }

        if (ly.kind == 0) {
            // the dense stem (block 0): the host hook - glm_try_native_ffn uploads/downloads internally - or the
            // float host path, exactly as glm47_trunk_forward picks between them.
            if (cudaStreamSynchronize(st) != cudaSuccess) { seterr("trunk-dev: stream sync failed (dense)"); return false; }
            s.host_ffv.resize((size_t) ne);
            if (cudaMemcpy(s.host_ffv.data(), s.ffv, nb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                seterr("trunk-dev: dense activation download failed");
                return false;
            }
            s.host_ffnout.resize((size_t) ne);
            const bool dense_native = ly.dense_types != nullptr && ly.dense_dev[0] && ly.dense_dev[1] && ly.dense_dev[2];
            if (!dense_native && (!ly.ffn_gate || !ly.ffn_up || !ly.ffn_down)) {
                seterr("trunk-dev: dense layer " + std::to_string(l) + " lacks its FFN weights");
                return false;
            }
            if (dense_native) {
                const void* nw[3] = {ly.dense_dev[0], ly.dense_dev[1], ly.dense_dev[2]};
                if (!strata::core::glm::glm_try_native_ffn(nw, ly.dense_types, ly.dense_g, s.host_ffv.data(),
                                                           s.host_ffnout.data(), kTrunkClamp)) {
                    seterr("trunk-dev: dense device FFN declined");
                    return false;
                }
            } else {
                strata::kernels::glm::expert_ffn(ly.ffn_gate, ly.ffn_up, ly.ffn_down, ly.dense_g,
                                                 s.host_ffv.data(), s.host_ffnout.data(), kTrunkClamp);
            }
            if (cudaMemcpyAsync(s.ffnout, s.host_ffnout.data(), nb, cudaMemcpyHostToDevice, st) != cudaSuccess) {
                seterr("trunk-dev: dense result upload failed");
                return false;
            }
        } else {
            // the routed MoE: the ONE unavoidable D2H (the router is a host matvec in v1), then the whole expert
            // FFN - MMVQ chain and combine - on the device.
            if (!ly.moe_router || !ly.moe_probs_b || !ly.moe_g || !ly.moe_native_fmt || !ly.moe_native_blob || !ly.shexp) {
                seterr("trunk-dev: MoE layer " + std::to_string(l) + " lacks router/native wiring");
                return false;
            }
            s.host_ffv.resize((size_t) ne);
            if (cudaMemcpyAsync(s.host_ffv.data(), s.ffv, nb, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
                seterr("trunk-dev: route activation download failed");
                return false;
            }
            if (cudaStreamSynchronize(st) != cudaSuccess) { seterr("trunk-dev: stream sync failed (route)"); return false; }
            const int k = ly.moe_g->n_used;
            if (k < 1 || k > 64) { seterr("trunk-dev: n_used out of range"); return false; }
            std::vector<int32_t> rids((size_t) k);
            std::vector<float> rw((size_t) k);
            strata::kernels::glm::moe_route(ly.moe_router, ly.moe_probs_b, *ly.moe_g, s.host_ffv.data(),
                                            rids.data(), rw.data());
            if (ids_out) {
                (*ids_out)[(size_t) l].clear();
                for (int i = 0; i < k; ++i) (*ids_out)[(size_t) l].push_back(rids[(size_t) i]);
            }
            if (verify) dev_ids[(size_t) l].assign(rids.begin(), rids.end());
            const strata::kernels::cpu::NativeFmt& f = *ly.moe_native_fmt;
            const uint8_t* rows[64];
            for (int i = 0; i < k; ++i) {
                const uint8_t* blob = ly.moe_native_blob(ly.moe_native_ctx, l, (int) rids[(size_t) i]);
                if (!blob) { seterr("trunk-dev: no blob for expert " + std::to_string((int) rids[(size_t) i])); return false; }
                const char* erow = nullptr;
                rows[i] = ensure_row(&experts(), l, (int) rids[(size_t) i], blob, f.bytes, &erow);
                if (!rows[i]) { seterr(erow ? erow : "trunk-dev: expert row upload failed"); return false; }
            }
            const int etypes[3] = { f.gu_type, f.gu_type, f.d_type };
            if (!moe_batch_impl_dev(rows, etypes, f.n_embd, f.n_ff, f.up_off, f.down_off, rw.data(), k,
                                    s.ffv, s.ffnout, st)) {
                seterr("trunk-dev: device routed MoE failed");
                return false;
            }
            // the shared expert, added UNWEIGHTED on top of the routed combine, exactly as glm_stage_moe_native does.
            // With native types it runs on the device straight from the token's own activation - no H2D/D2H, no extra
            // synchronisation, which is the whole point of the chained trunk; a wiring without native types falls back
            // to the host float GLU.
            if (ly.shexp_types && ly.shexp_types[0] && ly.shexp_types[1] && ly.shexp_types[2]) {
                if (!shared_expert_dev(ly.shexp, ly.shexp_types, *ly.moe_g, s.ffv, s.shr, st)) {
                    seterr("trunk-dev: device shared expert failed");
                    return false;
                }
            } else {
                s.host_shr.resize((size_t) ne);
                strata::kernels::glm::expert_ffn(ly.shexp[0], ly.shexp[1], ly.shexp[2], *ly.moe_g, s.host_ffv.data(),
                                                 s.host_shr.data(), kTrunkClamp);
                if (cudaMemcpyAsync(s.shr, s.host_shr.data(), nb, cudaMemcpyHostToDevice, st) != cudaSuccess) {
                    seterr("trunk-dev: shared-expert result upload failed");
                    return false;
                }
            }
            strata::kernels::add_inplace(s.ffnout, s.shr, ne, st);
        }

        // ---- residual: cur += ffnout ----
        strata::kernels::add_inplace(s.cur, s.ffnout, ne, st);
        if (verify) {
            dev_per[(size_t) l].resize((size_t) ne);
            if (cudaStreamSynchronize(st) != cudaSuccess) { seterr("trunk-dev: verify sync failed"); return false; }
            if (cudaMemcpy(dev_per[(size_t) l].data(), s.cur, nb, cudaMemcpyDeviceToHost) != cudaSuccess) {
                seterr("trunk-dev: verify per-layer download failed");
                return false;
            }
        }
    }

    if (cudaMemcpyAsync(out, s.cur, nb, cudaMemcpyDeviceToHost, st) != cudaSuccess) {
        seterr("trunk-dev: hidden-state download failed");
        return false;
    }
    if (cudaStreamSynchronize(st) != cudaSuccess) { seterr("trunk-dev: final stream sync failed"); return false; }

    if (verify) {
        // The host reference is glm47_trunk_forward on a COPY of the same caches/pos.  Run it with whatever MLA
        // mode the caller has enabled: with the device block on (the trunk's own requirement) this isolates the
        // trunk CHAINING from the block's own (pre-existing) residual - the number the <= 2e-4 tolerance covers.
        std::vector<float> href((size_t) ne, 0.0f);
        std::vector<std::vector<int32_t>> host_ids;
        std::vector<std::vector<float>> hper;
        std::string herr;
        const bool hok = strata::core::glm::glm47_trunk_forward(layers, n_layer, g, x, pos, eps, &href_caches, nullptr,
                                                                nullptr, nullptr, href.data(), &host_ids, &hper, herr);
        if (!hok) {
            std::fprintf(stderr, "[trunk-dev] host reference failed: %s\n", herr.c_str());
        } else {
            double md = 0.0;
            for (int i = 0; i < ne; ++i)
                md = std::max(md, (double) std::fabs((double) out[(size_t) i] - (double) href[(size_t) i]));
            int rflip = 0, first_flip = -1;
            for (size_t l = 0; l < dev_ids.size() && l < host_ids.size(); ++l)
                if (dev_ids[l] != host_ids[l]) { ++rflip; if (first_flip < 0) first_flip = (int) l; }
            std::fprintf(stderr, "[trunk-dev] max|dev-host| = %.3e over %d (pos %d), route flips %d (first L%d)\n",
                         md, ne, pos, rflip, first_flip);
            // the first layer whose output already differs, so a mismatch can be attributed to a site
            int first_bad = -1; double first_bad_d = 0.0, worst_layer_d = 0.0; int worst_layer = -1;
            for (size_t l = 0; l < dev_per.size() && l < hper.size(); ++l) {
                if (hper[l].size() != (size_t) ne || dev_per[l].size() != (size_t) ne) continue;
                double d = 0.0;
                for (int i = 0; i < ne; ++i)
                    d = std::max(d, (double) std::fabs((double) dev_per[l][(size_t) i] - (double) hper[l][(size_t) i]));
                if (d > 1e-5 && first_bad < 0) { first_bad = (int) l; first_bad_d = d; }
                if (d > worst_layer_d) { worst_layer_d = d; worst_layer = (int) l; }
            }
            if (first_bad >= 0) {
                std::fprintf(stderr, "[trunk-dev]   first layer over 1e-5: L%d (%.3e); worst L%d (%.3e)\n",
                             first_bad, first_bad_d, worst_layer, worst_layer_d);
                if (std::getenv("STRATA_GLM_TRUNK_DEV_VERIFY")[0] == '2') {
                    std::fprintf(stderr, "[trunk-dev]   per-layer:");
                    for (size_t l = 0; l < dev_per.size() && l < hper.size(); ++l) {
                        if (hper[l].size() != (size_t) ne) { std::fprintf(stderr, " -"); continue; }
                        double d = 0.0;
                        for (int i = 0; i < ne; ++i)
                            d = std::max(d, (double) std::fabs((double) dev_per[l][(size_t) i] - (double) hper[l][(size_t) i]));
                        std::fprintf(stderr, " %.1e", d);
                    }
                    std::fprintf(stderr, "\n");
                }
            }
            if (rflip > 0) {
                const int L = first_flip;
                std::fprintf(stderr, "[trunk-dev]   L%d dev:", L);
                for (int32_t v : dev_ids[(size_t) L]) std::fprintf(stderr, " %d", (int) v);
                std::fprintf(stderr, "  host:");
                for (int32_t v : host_ids[(size_t) L]) std::fprintf(stderr, " %d", (int) v);
                std::fprintf(stderr, "\n");
            }
        }
    }
    return true;
}

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

// Free VRAM on the current device, so a caller can size its expert cache to the whole card (the GPU-first
// tier of a GPU -> RAM -> disk layout).  0 when the query fails.
size_t glm47_device_free_vram() {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) return 0;
    return free_b;
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
