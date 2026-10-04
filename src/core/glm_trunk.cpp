/// The trunk loop: 45 blocks, each dispatching to the stage its kind needs.
///
/// It calls only stages that already exist and are independently gated, and holds no arithmetic of its own beyond the
/// residual routing that is inherent in the two hyper-connection sites per block.  The order per block is the one the
/// dump's own per-block stage names confirm:
///
///     hc_pre -> attn_norm -> attention -> hc_post -> hc_pre -> ffn_norm -> FFN -> hc_post
///
/// and the shape alternates: the attention and FFN outputs are one vector per token ([n_embd]) while everything
/// between the sites is HC rows ([HC][n_embd]).  Both are float*, which is why the buffer names below say which is
/// which rather than leaving it to the reader.
#include "strata/core/glm_trunk.hpp"

#include <algorithm>
#include <cmath>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "strata/core/glm_layer.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/kernels/glm_decode_trunk.hpp"
#include "strata/kernels/native_mmvq.hpp"

namespace strata::core::glm {
namespace {

constexpr int HC_STREAMS = 4;

/// Copies `n` floats.  Used only for the final hand-back, so the caller's buffer is written once rather than the loop
/// needing to own it.
void copy_floats(float* dst, const float* src, size_t n) {
    for (size_t i = 0; i < n; ++i) dst[i] = src[i];
}

GlmParallelForFn g_parallel_for = nullptr;

/// Runs f(t, err_t) for every token, across the installed worker pool when there is one.  The per-token hyper-connection
/// stages are host code with no shared state, so a 1024-token chunk is 1024 independent jobs; run serially they were
/// ~35 s of the chunk.  Returns false with the first failing token's message.
template <class F>
bool for_tokens(int n, std::string& err, F&& f) {
    std::vector<char> ok((size_t) n, 1);
    std::vector<std::string> errs((size_t) n);
    const std::function<void(int)> job = [&](int t) { if (!f(t, errs[(size_t) t])) ok[(size_t) t] = 0; };
    if (g_parallel_for != nullptr && n > 1) g_parallel_for(n, job);
    else for (int t = 0; t < n; ++t) job(t);
    for (int t = 0; t < n; ++t)
        if (!ok[(size_t) t]) { err = errs[(size_t) t]; return false; }
    return true;
}

}  // namespace

void glm_set_parallel_for(GlmParallelForFn fn) { g_parallel_for = fn; }

void glm_parallel_for(int n, const std::function<void(int)>& job) {
    if (g_parallel_for != nullptr && n > 1) g_parallel_for(n, job);
    else for (int i = 0; i < n; ++i) job(i);
}

namespace {
GlmPrefetchFn g_layer_prefetch = nullptr;
void* g_layer_prefetch_ctx = nullptr;
}  // namespace
void glm_set_layer_prefetch(GlmPrefetchFn fn, void* ctx) { g_layer_prefetch = fn; g_layer_prefetch_ctx = ctx; }

namespace {
bool g_device_trunk = false;

/// Whether this token can run on the device trunk.  A token cannot switch paths half-way (the attention blocks mutate the
/// recurrent state), so everything it needs is established up front: every layer's weights are native (checked once - they
/// never change), the KDA/MLA device blocks are enabled, the state arrays exist, and no MLA cache is about to pass the
/// 8192-position limit of the device attention.
bool device_trunk_eligible(GlmTrunkProvider provider, void* provider_ctx, int first_layer, int layers,
                           const GlmTrunkState& state) {
    if (!kernels::glm::kda_device_block_enabled() || !kernels::glm::mla_device_block_enabled()) return false;
    if (!state.kda_state || !state.kda_conv || !state.kda_index || !state.mla_cache || !state.mla_len || !state.mla_index)
        return false;
    static int weights_ok = -1;   // -1 unknown, 0 no, 1 yes
    if (weights_ok < 0) {
        weights_ok = 1;
        for (int i = 0; i < layers && weights_ok; ++i) {
            const int layer = first_layer + i;
            GlmTrunkLayerWeights w;
            std::string e;
            if (!provider(provider_ctx, layer, w, e)) { weights_ok = 0; break; }
            if (!w.hc_attn_fn || !w.hc_attn_base || !w.hc_attn_scale || !w.hc_ffn_fn || !w.hc_ffn_base || !w.hc_ffn_scale ||
                !w.ffn_norm) { weights_ok = 0; break; }
            using kernels::native_mmvq_supported;
            if (glm_attention_is_mla(layer) == 1) {
                if (!w.mla || !w.attn_norm || !w.mla->wq_a_type || !w.mla->wq_b_type || !w.mla->kv_a_type || !w.mla->wo_type ||
                    !native_mmvq_supported(w.mla->wq_a_type) || !native_mmvq_supported(w.mla->wq_b_type) ||
                    !native_mmvq_supported(w.mla->kv_a_type) || !native_mmvq_supported(w.mla->wo_type)) weights_ok = 0;
            } else {
                if (!w.kda || !w.kda->wq_type || !w.kda->wk_type || !w.kda->wv_type || !w.kda->wo_type ||
                    !native_mmvq_supported(w.kda->wq_type) || !native_mmvq_supported(w.kda->wk_type) ||
                    !native_mmvq_supported(w.kda->wv_type) || !native_mmvq_supported(w.kda->wo_type)) weights_ok = 0;
            }
        }
    }
    if (weights_ok != 1) return false;
    for (int i = 0; i < layers; ++i) {
        const int layer = first_layer + i;
        if (glm_attention_is_mla(layer) != 1) continue;
        const int slot = state.mla_index[layer];
        if (slot < 0 || state.mla_len[slot] + 1 > 8192) return false;
    }
    return true;
}

bool dtrunk_fail(std::string& err, const char* what, const char* detail) {
    err = std::string("glm_trunk_forward (device trunk): ") + what + ": " + detail;
    return false;
}

/// The device-trunk token loop: the same layer sequence as the host loop in glm_trunk_forward, with the streams, the
/// hyper-connection sites and the attention blocks on the device.
bool glm_trunk_forward_device(const float* x, int layers, GlmTrunkProvider provider, void* provider_ctx,
                              const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                              float hc_rms_eps, GlmTrunkState& state, float* l_out, std::string& err, int first_layer) {
    using namespace kernels::glm;
    using Clock = std::chrono::steady_clock;
    const bool timing = std::getenv("STRATA_GLM_TIMING") != nullptr;
    const int ne = kda_g.n_embd;
    char cerr[256] = {};
    std::vector<float> ffn_in((size_t) ne), ffn_out((size_t) ne);
    double tm_provider = 0, tm_launch = 0, tm_wait = 0, tm_ffn = 0;
    auto elapsed = [](Clock::time_point a) { return std::chrono::duration<double, std::milli>(Clock::now() - a).count(); };
    const auto t_start = Clock::now();

    if (!dtrunk_begin(x, ne, cerr, sizeof cerr)) return dtrunk_fail(err, "begin", cerr);
    void* stream = dtrunk_stream();

    // STRATA_GLM_DT_VERIFY: after each device hyper-connection op, fetch its buffers and compare against the host glm_stage_hc_*
    // on the same data (stateless, so it cannot disturb the run).  Prints the max |difference| per op for early layers, and the
    // worst over the token.
    const bool verify = std::getenv("STRATA_GLM_DT_VERIFY") != nullptr;
    std::vector<float> v_cur, v_ao, v_mid, v_li, v_ref, v_nxt;
    HcMix v_mx_a, v_mx_f;
    double v_worst[4] = {0, 0, 0, 0}, v_cur_diff[4] = {0, 0, 0, 0}, v_cur_ref[4] = {0, 0, 0, 0};
    auto vdiff = [](const float* a, const float* b, size_t n, double& dmax, double& rmax) {
        dmax = 0; rmax = 0;
        for (size_t k = 0; k < n; ++k) { dmax = std::max(dmax, (double) std::fabs(a[k] - b[k])); rmax = std::max(rmax, (double) std::fabs(b[k])); }
    };
    auto vrecord = [&](int op, const float* dev, const float* ref, size_t n) {
        double d, r;
        vdiff(dev, ref, n, d, r);
        v_cur_diff[op] = d; v_cur_ref[op] = r;
        v_worst[op] = std::max(v_worst[op], r > 0 ? d / r : d);
    };

    for (int i = 0; i < layers; ++i) {
        const int layer = first_layer + i;
        const bool is_mla = glm_attention_is_mla(layer) == 1;
        const bool is_dense = glm_ffn_is_dense(layer);
        GlmTrunkLayerWeights w;
        auto tick = Clock::now();
        if (!provider(provider_ctx, layer, w, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + ": " + err;
            return false;
        }
        tm_provider += elapsed(tick);
        const bool ffn_ok = is_dense ? (w.ffn_gate && w.ffn_up && w.ffn_down && w.moe_g)
                                     : (w.moe_router && w.moe_g && w.moe_fmt && w.blob_fn);
        if (!ffn_ok) return dtrunk_fail(err, "a required FFN weight is null at layer", std::to_string(layer).c_str());

        // ---- attention site, entirely on the device ----
        tick = Clock::now();
        if (verify) { v_cur.resize((size_t) HC_STREAMS * ne); dtrunk_debug_fetch(0, v_cur.data(), ne); }
        if (!dtrunk_hc_pre(0, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, is_mla ? w.attn_norm : nullptr, hc_rms_eps, ne,
                           cerr, sizeof cerr)) return dtrunk_fail(err, "attention hc_pre", cerr);
        if (verify) {
            v_li.resize((size_t) ne); v_ref.resize((size_t) ne);
            dtrunk_debug_fetch(2, v_li.data(), ne);
            std::string e2;
            glm_stage_hc_norm(v_cur.data(), ne, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, is_mla ? w.attn_norm : nullptr,
                              v_ref.data(), &v_mx_a, hc_rms_eps, nullptr, e2);
            vrecord(0, v_li.data(), v_ref.data(), (size_t) ne);
        }
        int r = 0;
        if (is_mla) {
            const int slot = state.mla_index[layer];
            float* cache = state.mla_cache[slot];
            const int cells = state.mla_len[slot];
            float* kv_row = cache + (size_t) cells * (size_t) mla_g.kv_lora;
            r = mla_block_launch_cuda(*w.mla, mla_g, dtrunk_layer_in(), cells + 1, cache, kv_row, dtrunk_attn_out(), stream, cerr,
                                      sizeof cerr);
            if (r == 1) state.mla_len[slot] = cells + 1;
        } else {
            const int slot = state.kda_index[layer];
            r = kda_block_launch_cuda(*w.kda, kda_g, dtrunk_layer_in(), dtrunk_attn_out(), state.kda_state[slot],
                                      state.kda_conv[slot], stream, cerr, sizeof cerr);
        }
        if (r != 1) return dtrunk_fail(err, is_mla ? "MLA block" : "KDA block", cerr);
        if (!dtrunk_hc_post(0, ne, cerr, sizeof cerr)) return dtrunk_fail(err, "attention hc_post", cerr);
        if (verify) {
            v_ao.resize((size_t) ne); v_mid.resize((size_t) HC_STREAMS * ne); v_nxt.resize((size_t) HC_STREAMS * ne);
            dtrunk_debug_fetch(3, v_ao.data(), ne);
            dtrunk_debug_fetch(1, v_mid.data(), ne);
            std::string e2;
            glm_stage_hc_post(v_ao.data(), v_cur.data(), v_mx_a, ne, v_nxt.data(), e2);
            vrecord(1, v_mid.data(), v_nxt.data(), (size_t) HC_STREAMS * ne);
        }
        if (!dtrunk_hc_pre(1, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, hc_rms_eps, ne, cerr, sizeof cerr))
            return dtrunk_fail(err, "FFN hc_pre", cerr);
        if (verify) {
            dtrunk_debug_fetch(2, v_li.data(), ne);
            std::string e2;
            glm_stage_hc_norm(v_mid.data(), ne, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, v_ref.data(), &v_mx_f,
                              hc_rms_eps, nullptr, e2);
            vrecord(2, v_li.data(), v_ref.data(), (size_t) ne);
        }
        tm_launch += elapsed(tick);

        // ---- FFN site: the only per-layer host exchange ----
        tick = Clock::now();
        if (!dtrunk_fetch_ffn_in(ffn_in.data(), ne, cerr, sizeof cerr)) return dtrunk_fail(err, "fetch ffn_in", cerr);
        tm_wait += elapsed(tick);
        tick = Clock::now();
        if (is_dense) {
            if (!glm_stage_ffn(ffn_in.data(), w.ffn_gate, w.ffn_up, w.ffn_down, *w.moe_g, ffn_out.data(), w.clamp_limit, err,
                               w.ffn_types)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " dense FFN: " + err;
                return false;
            }
        } else {
            if (!glm_stage_moe_native(ffn_in.data(), w.moe_router, w.moe_probs_b, *w.moe_g, layer, *w.moe_fmt, w.blob_fn,
                                      w.blob_ctx, w.shexp_g, w.shexp, w.shexp_types, w.shexp_clamp, ffn_out.data(), err,
                                      nullptr)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " routed FFN: " + err;
                return false;
            }
        }
        tm_ffn += elapsed(tick);
        tick = Clock::now();
        if (!dtrunk_put_ffn_out(ffn_out.data(), ne, cerr, sizeof cerr)) return dtrunk_fail(err, "put ffn_out", cerr);
        if (!dtrunk_hc_post(1, ne, cerr, sizeof cerr)) return dtrunk_fail(err, "FFN hc_post", cerr);
        if (verify) {
            std::vector<float> dev_next((size_t) HC_STREAMS * ne);
            dtrunk_debug_fetch(0, dev_next.data(), ne);
            std::string e2;
            glm_stage_hc_post(ffn_out.data(), v_mid.data(), v_mx_f, ne, v_nxt.data(), e2);
            vrecord(3, dev_next.data(), v_nxt.data(), (size_t) HC_STREAMS * ne);
            if (layer < 3 || layer == 44)
                std::fprintf(stderr,
                             "DT_VERIFY layer %d: attn hc_pre %.2e/%.2e  hc_post %.2e/%.2e | ffn hc_pre %.2e/%.2e  hc_post %.2e/%.2e"
                             " (max|diff| / max|ref|)\n", layer, v_cur_diff[0], v_cur_ref[0], v_cur_diff[1], v_cur_ref[1],
                             v_cur_diff[2], v_cur_ref[2], v_cur_diff[3], v_cur_ref[3]);
        }
        tm_launch += elapsed(tick);
    }
    if (verify)
        std::fprintf(stderr, "DT_VERIFY token worst relative diff: attn hc_pre %.2e  attn hc_post %.2e  ffn hc_pre %.2e  ffn hc_post %.2e\n",
                     v_worst[0], v_worst[1], v_worst[2], v_worst[3]);
    if (!dtrunk_fetch_streams(l_out, ne, cerr, sizeof cerr)) return dtrunk_fail(err, "fetch streams", cerr);
    if (timing)
        std::fprintf(stderr,
                     "GLM_TIMING provider=%.3f hca=%.3f attn=%.3f hcap=0.000 hcf=0.000 ffn=%.3f hcfp=0.000 total=%.3f ms"
                     " (device trunk: launch=%.1f wait-for-gpu=%.1f)\n",
                     tm_provider, tm_launch, tm_wait, tm_ffn, elapsed(t_start), tm_launch, tm_wait);
    return true;
}
}  // namespace

void glm_set_device_trunk(bool enabled) { g_device_trunk = enabled; }

bool glm_trunk_forward(const float* x, int layers, GlmTrunkProvider provider, void* provider_ctx,
                       const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                       float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err,
                       int first_layer, GlmStageFn stage_fn, void* stage_ctx) {
    if (!x || !provider || !l_out) {
        err = "glm_trunk_forward: null argument";
        return false;
    }
    if (layers <= 0 || layers > GLM_TRUNK_BLOCKS) {
        err = "glm_trunk_forward: layers must be within the trunk (0.." + std::to_string(GLM_TRUNK_BLOCKS) + ")";
        return false;
    }
    const int ne = kda_g.n_embd;
    if (ne <= 0 || mla_g.n_embd != ne) {
        err = "glm_trunk_forward: the two attention geometries disagree on n_embd";
        return false;
    }
    // The device trunk keeps the streams and the hyper-connection sites on the GPU (no stage callback can observe them there).
    if (stage_fn == nullptr && g_device_trunk && device_trunk_eligible(provider, provider_ctx, first_layer, layers, state))
        return glm_trunk_forward_device(x, layers, provider, provider_ctx, kda_g, mla_g, hc_rms_eps, state, l_out, err,
                                        first_layer);

    const size_t hc = (size_t) HC_STREAMS;

    // one token at a time: `cur` is this layer's input streams, `nxt` its output, and they swap each layer
    std::vector<float> buf_a((size_t) hc * ne), buf_b((size_t) hc * ne);
    std::vector<float> xn((size_t) ne), attn_out((size_t) ne), mid((size_t) hc * ne), ffn_in((size_t) ne);
    std::vector<float> ffn_out((size_t) ne);
    std::vector<kernels::glm::HcMix> mix_a(1), mix_f(1);

    copy_floats(buf_a.data(), x, (size_t) hc * ne);
    float* cur = buf_a.data();
    float* nxt = buf_b.data();
    using Clock = std::chrono::steady_clock;
    const bool timing = std::getenv("STRATA_GLM_TIMING") != nullptr;
    double tm_provider = 0, tm_hca = 0, tm_attn = 0, tm_hcap = 0;
    double tm_hcf = 0, tm_ffn = 0, tm_hcfp = 0;
    auto elapsed = [](Clock::time_point a) {
        return std::chrono::duration<double, std::milli>(Clock::now() - a).count();
    };

    // STRATA_GLM_LOOKAHEAD_STATS: how well can layer L+1's routed experts be predicted from the state before its attention?
    // Predictor A applies L+1's FFN hyper-connection + norm + router to the streams after layer L; predictor B to the streams
    // after layer L's attention (earlier, so a longer window to act on the prediction).  Diagnostics only: no result changes.
    static const bool lookahead_stats = std::getenv("STRATA_GLM_LOOKAHEAD_STATS") != nullptr;
    static double la_hit[4] = {0, 0, 0, 0};   // A@8, A@12, B@8, B@12: predicted experts that the layer really used
    static double la_total = 0;
    static long la_pairs = 0;
    int pend_layer = -1, pred_a[12], pred_b[12];
    auto predict_ids = [&](const float* streams, const GlmTrunkLayerWeights& w2, int* out12) {
        std::vector<float> in((size_t) ne);
        kernels::glm::HcMix mx;
        std::string e3;
        glm_stage_hc_norm(streams, ne, w2.hc_ffn_fn, w2.hc_ffn_base, w2.hc_ffn_scale, w2.ffn_norm, in.data(), &mx,
                          hc_rms_eps, stream, e3);
        kernels::glm::MoeGeometry g12 = *w2.moe_g;
        g12.n_used = 12;
        int32_t ids[64];
        float wt[64];
        kernels::glm::moe_route(w2.moe_router, w2.moe_probs_b, g12, in.data(), ids, wt);
        for (int k = 0; k < 12; ++k) out12[k] = ids[k];
    };

    for (int i = 0; i < layers; ++i) {
        // the artifact's layer number, not the loop's counter: every dispatch and lookup below is keyed by it
        const int layer = first_layer + i;
        const bool is_mla = glm_attention_is_mla(layer) == 1;
        const bool is_dense = glm_ffn_is_dense(layer);

        GlmTrunkLayerWeights w;
        auto tick = Clock::now();
        if (!provider(provider_ctx, layer, w, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + ": " + err;
            return false;
        }
        tm_provider += elapsed(tick);
        // A missing weight for the kind this layer IS, is a wiring bug rather than a fallback: an MLA layer driven
        // down the KDA path would compute plausible numbers from the wrong weights.
        const bool common_ok = w.hc_attn_fn && w.hc_attn_base && w.hc_attn_scale && w.attn_norm &&
                               w.hc_ffn_fn && w.hc_ffn_base && w.hc_ffn_scale && w.ffn_norm;
        const bool attn_ok = is_mla ? (w.mla != nullptr) : (w.kda != nullptr);
        const bool ffn_ok = is_dense ? (w.ffn_gate && w.ffn_up && w.ffn_down && w.moe_g)
                                     : (w.moe_router && w.moe_g && w.moe_fmt && w.blob_fn);
        if (!common_ok || !attn_ok || !ffn_ok) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + ": the provider left a required weight null"
                  " (kind: " + (is_mla ? "MLA" : "KDA") + " attention, " + (is_dense ? "dense" : "routed") + " FFN)";
            return false;
        }

        // ---- the attention site ----
        //
        // THE NORM IS APPLIED HERE ONLY FOR MLA.  The KDA kernel norms its own input internally with w.attn_norm
        // (src/kernels/glm_kda.cpp:58 does `xn = x * inv * attn_norm[i]`), and the oracle's runner hands kda_block the
        // PRE-norm residual for exactly that reason - its own line reads `cur_raw = cur  # pre-norm residual` and then
        // `kda_block(kw, cur_raw.T, ...)`, while `cur = rms_norm(cur, attn_norm)` is used only for MLA.  Passing the
        // normed value here as well normalised the KDA's input TWICE, which is invisible to the KDA's own gate - that
        // fixture feeds it a raw input, as the oracle does - and to every other stage gate, because each of them is
        // correct in isolation.  It is exactly the class of mistake only an assembly test can find.
        tick = Clock::now();
        if (!glm_stage_hc_norm(cur, ne, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, is_mla ? w.attn_norm : nullptr,
                               xn.data(), &mix_a[0], hc_rms_eps, stream, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " attention site: " + err;
            return false;
        }
        tm_hca += elapsed(tick);
        // Report each stage as it is produced.  Defaulted to nullptr, so a caller that does not ask for the
        // stream is unaffected - the loop's behaviour must not change because a test is watching.
        auto stage = [&](const char* nm, const float* data, int n) {
            if (stage_fn != nullptr) stage_fn(stage_ctx, layer, nm, data, n);
        };
        stage("hc_attn_pre", cur, (int) (hc * (size_t) ne));
        stage("attn_input", xn.data(), ne);
        if (is_mla) stage("attn_norm", xn.data(), ne);
        tick = Clock::now();
        if (is_mla) {
            const int slot = state.mla_index ? state.mla_index[layer] : -1;
            if (slot < 0 || !state.mla_cache || !state.mla_len) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " is MLA but has no cache slot";
                return false;
            }
            float* cache = state.mla_cache[slot];
            const int cells = state.mla_len[slot];
            // the reference appends this position's latent BEFORE attending, so the new latent is written into the
            // next slot as part of the call rather than afterwards
            kernels::glm::MlaIntermediates want;
            want.kv = cache + (size_t) cells * (size_t) mla_g.kv_lora;
            if (!glm_stage_mla(*w.mla, mla_g, xn.data(), cells + 1, cache, attn_out.data(), err, &want)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " MLA: " + err;
                return false;
            }
            state.mla_len[slot] = cells + 1;
        } else {
            const int slot = state.kda_index ? state.kda_index[layer] : -1;
            if (slot < 0 || !state.kda_state) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " is KDA but has no state slot";
                return false;
            }
            float* conv = state.kda_conv ? state.kda_conv[slot] : nullptr;
            if (!glm_stage_kda(xn.data(), *w.kda, kda_g, 1, attn_out.data(), state.kda_state[slot], err, conv)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " KDA: " + err;
                return false;
            }
        }
        tm_attn += elapsed(tick);
        stage("attn_output", attn_out.data(), ne);
        tick = Clock::now();
        if (!glm_stage_hc_post(attn_out.data(), cur, mix_a[0], ne, mid.data(), err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " attention hc_post: " + err;
            return false;
        }
        tm_hcap += elapsed(tick);

        stage("hc_attn_post", mid.data(), (int) (hc * (size_t) ne));

        // ---- the FFN site ----
        tick = Clock::now();
        if (!glm_stage_hc_norm(mid.data(), ne, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, ffn_in.data(),
                               &mix_f[0], hc_rms_eps, stream, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " FFN site: " + err;
            return false;
        }
        tm_hcf += elapsed(tick);
        stage("hc_ffn_pre", mid.data(), (int) (hc * (size_t) ne));
        stage("ffn_norm", ffn_in.data(), ne);
        tick = Clock::now();
        if (is_dense) {
            if (!glm_stage_ffn(ffn_in.data(), w.ffn_gate, w.ffn_up, w.ffn_down, *w.moe_g, ffn_out.data(),
                               w.clamp_limit, err, w.ffn_types)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " dense FFN: " + err;
                return false;
            }
        } else {
            int moe_ids[64];
            for (int k = 0; k < 64; ++k) moe_ids[k] = -1;
            if (!glm_stage_moe_native(ffn_in.data(), w.moe_router, w.moe_probs_b, *w.moe_g, layer, *w.moe_fmt,
                                      w.blob_fn, w.blob_ctx, w.shexp_g, w.shexp, w.shexp_types,
                                      w.shexp_clamp, ffn_out.data(), err, moe_ids)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " routed FFN: " + err;
                return false;
            }
            if (lookahead_stats && pend_layer == layer) {
                const int n8 = w.moe_g->n_used < 8 ? w.moe_g->n_used : 8;
                for (int k = 0; k < n8; ++k)
                    for (int j = 0; j < 12; ++j) {
                        if (pred_a[j] == moe_ids[k]) { if (j < 8) la_hit[0] += 1; la_hit[1] += 1; }
                        if (pred_b[j] == moe_ids[k]) { if (j < 8) la_hit[2] += 1; la_hit[3] += 1; }
                    }
                la_total += n8;
                if (++la_pairs % 400 == 0)
                    std::fprintf(stderr, "LOOKAHEAD after %ld layer pairs - share of the layer's 8 experts predicted: "
                                         "A(top8) %.1f%% A(top12) %.1f%%  B(top8) %.1f%% B(top12) %.1f%%\n", la_pairs,
                                 100 * la_hit[0] / la_total, 100 * la_hit[1] / la_total, 100 * la_hit[2] / la_total,
                                 100 * la_hit[3] / la_total);
            }
            // the ids as floats, so the one stage stream carries them: the callback is float-based because it carries
            // activations, and these are small integers that survive the round trip exactly
            const int n_report = w.moe_g->n_used < 64 ? w.moe_g->n_used : 64;
            float ids_f[64];
            for (int k = 0; k < n_report; ++k) ids_f[k] = (float) moe_ids[k];
            stage("moe_ids", ids_f, n_report);
        }
        tm_ffn += elapsed(tick);
        stage("ffn_out", ffn_out.data(), ne);
        tick = Clock::now();
        if (!glm_stage_hc_post(ffn_out.data(), mid.data(), mix_f[0], ne, nxt, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " FFN hc_post: " + err;
            return false;
        }
        tm_hcfp += elapsed(tick);
        stage("l_out", nxt, (int) (hc * (size_t) ne));
        if (lookahead_stats && i + 1 < layers && !glm_ffn_is_dense(layer + 1)) {
            GlmTrunkLayerWeights w2;
            std::string e2;
            if (provider(provider_ctx, layer + 1, w2, e2) && w2.moe_router && w2.moe_g && w2.hc_ffn_fn && w2.ffn_norm) {
                predict_ids(nxt, w2, pred_a);       // streams after this layer
                predict_ids(mid.data(), w2, pred_b); // streams after this layer's attention (before its FFN)
                pend_layer = layer + 1;
            }
        }

        float* swap = cur; cur = nxt; nxt = swap;   // this layer's output is the next layer's input
    }

    copy_floats(l_out, cur, (size_t) hc * ne);
    if (timing)
        std::fprintf(stderr, "GLM_TIMING provider=%.3f hca=%.3f attn=%.3f hcap=%.3f hcf=%.3f ffn=%.3f hcfp=%.3f total=%.3f ms\n",
                     tm_provider, tm_hca, tm_attn, tm_hcap, tm_hcf, tm_ffn, tm_hcfp,
                     tm_provider + tm_hca + tm_attn + tm_hcap + tm_hcf + tm_ffn + tm_hcfp);
    return true;
}

bool glm_trunk_forward_batch(const float* x, int tokens, int layers, GlmTrunkProvider provider, void* provider_ctx,
                             const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                             float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err,
                             int first_layer) {
    if (!x || !provider || !l_out) { err = "glm_trunk_forward_batch: null argument"; return false; }
    if (tokens <= 0 || tokens > 4096 || layers <= 0 || layers > GLM_TRUNK_BLOCKS) {
        err = "glm_trunk_forward_batch: tokens must be 1..4096 and layers within the trunk";
        return false;
    }
    const int ne = kda_g.n_embd;
    if (ne <= 0 || mla_g.n_embd != ne) {
        err = "glm_trunk_forward_batch: the two attention geometries disagree on n_embd";
        return false;
    }
    const size_t hc = HC_STREAMS, row = hc * (size_t) ne, n = (size_t) tokens;
    std::vector<float> buf_a(n * row), buf_b(n * row), xn(n * ne), attn_out(n * ne), mid(n * row);
    std::vector<float> ffn_in(n * ne), ffn_out(n * ne);
    std::vector<kernels::glm::HcMix> mix_a(n), mix_f(n);
    copy_floats(buf_a.data(), x, n * row);
    float* cur = buf_a.data();
    float* nxt = buf_b.data();
    double phase_ms[4] = {0.0, 0.0, 0.0, 0.0};   // hyper-connection stages, attention, routed/dense FFN, (MLA share of attention)
    auto lap_t = std::chrono::steady_clock::now();
    auto lap = [&]() {
        const auto now = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(now - lap_t).count();
        lap_t = now;
        return ms;
    };

    for (int i = 0; i < layers; ++i) {
        const int layer = first_layer + i;
        const bool is_mla = glm_attention_is_mla(layer) == 1;
        const bool is_dense = glm_ffn_is_dense(layer);
        GlmTrunkLayerWeights w;
        if (!provider(provider_ctx, layer, w, err)) {
            err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + ": " + err;
            return false;
        }
        const bool common_ok = w.hc_attn_fn && w.hc_attn_base && w.hc_attn_scale && w.attn_norm &&
                               w.hc_ffn_fn && w.hc_ffn_base && w.hc_ffn_scale && w.ffn_norm;
        const bool attn_ok = is_mla ? (w.mla != nullptr) : (w.kda != nullptr);
        const bool ffn_ok = is_dense ? (w.ffn_gate && w.ffn_up && w.ffn_down && w.moe_g)
                                     : (w.moe_router && w.moe_g && w.moe_fmt && w.blob_fn);
        if (!common_ok || !attn_ok || !ffn_ok) {
            err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + ": required weights are missing";
            return false;
        }
        // Tell the expert source which layer is starting (every layer before it is finished), so it can keep reading the
        // layers ahead while this one computes.
        if (g_layer_prefetch != nullptr) g_layer_prefetch(g_layer_prefetch_ctx, layer);

        lap();
        if (!for_tokens(tokens, err, [&](int t, std::string& e) {
                return glm_stage_hc_norm(cur + (size_t) t * row, ne, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale,
                                         is_mla ? w.attn_norm : nullptr, xn.data() + (size_t) t * ne,
                                         &mix_a[(size_t) t], hc_rms_eps, stream, e);
            })) {
            err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " attention site: " + err;
            return false;
        }
        phase_ms[0] += lap();
        if (is_mla) {
            const int slot = state.mla_index ? state.mla_index[layer] : -1;
            if (slot < 0 || !state.mla_cache || !state.mla_len) {
                err = "glm_trunk_forward_batch: MLA layer has no cache slot";
                return false;
            }
            float* cache = state.mla_cache[slot];
            // The whole chunk at once (batched projections, one causal attention launch, resident device cache); when that
            // path is unavailable it declines and the chunk runs token by token exactly as before.
            const int cells0 = state.mla_len[slot];
            if (kernels::glm::mla_forward_batch(*w.mla, mla_g, xn.data(), tokens, cells0, cache, attn_out.data())) {
                state.mla_len[slot] += tokens;
                if (std::getenv("STRATA_GLM_MLA_VERIFY")) {
                    // Debug: run the per-token reference over the same chunk and report how far the batched result is.
                    const std::vector<float> batched(attn_out.begin(), attn_out.begin() + (size_t) tokens * ne);
                    std::vector<float> ref_cache((size_t) (cells0 + tokens) * mla_g.kv_lora);
                    for (size_t i = 0; i < (size_t) cells0 * mla_g.kv_lora; ++i) ref_cache[i] = cache[i];
                    std::vector<float> ref((size_t) tokens * ne);
                    for (int t = 0; t < tokens; ++t) {
                        kernels::glm::MlaIntermediates want;
                        want.kv = ref_cache.data() + (size_t) (cells0 + t) * mla_g.kv_lora;
                        std::string e2;
                        if (!glm_stage_mla(*w.mla, mla_g, xn.data() + (size_t) t * ne, cells0 + t + 1, ref_cache.data(),
                                           ref.data() + (size_t) t * ne, e2, &want)) break;
                    }
                    double max_abs = 0.0, max_ref = 0.0, kv_diff = 0.0;
                    std::vector<double> tok_diff((size_t) tokens, 0.0);
                    for (size_t i = 0; i < ref.size(); ++i) {
                        const double d = (double) std::fabs(ref[i] - batched[i]);
                        max_abs = std::max(max_abs, d);
                        max_ref = std::max(max_ref, (double) std::fabs(ref[i]));
                        tok_diff[i / (size_t) ne] = std::max(tok_diff[i / (size_t) ne], d);
                    }
                    int first_bad = -1;
                    for (int t = 0; t < tokens; ++t) if (tok_diff[(size_t) t] > 1e-3) { first_bad = t; break; }
                    std::fprintf(stderr, "MLA_VERIFY layer %d: first token with diff > 1e-3: %d; per-token diff @0 %.2e @1 %.2e @2 %.2e @10 %.2e @100 %.2e @254 %.2e @255 %.2e @256 %.2e @last %.2e\n",
                                 layer, first_bad, tok_diff[0], tokens > 1 ? tok_diff[1] : 0.0, tokens > 2 ? tok_diff[2] : 0.0,
                                 tokens > 10 ? tok_diff[10] : 0.0, tokens > 100 ? tok_diff[100] : 0.0,
                                 tokens > 254 ? tok_diff[254] : 0.0, tokens > 255 ? tok_diff[255] : 0.0,
                                 tokens > 256 ? tok_diff[256] : 0.0, tok_diff[(size_t) tokens - 1]);
                    for (size_t i = (size_t) cells0 * mla_g.kv_lora; i < ref_cache.size(); ++i)
                        kv_diff = std::max(kv_diff, (double) std::fabs(ref_cache[i] - cache[i]));
                    std::fprintf(stderr, "MLA_VERIFY layer %d tokens %d: max|batched-ref| %.3e (max|ref| %.3e), kv rows diff %.3e\n",
                                 layer, tokens, max_abs, max_ref, kv_diff);
                    // later tokens (and layers) must keep seeing the reference's cache rows and outputs
                    for (size_t i = (size_t) cells0 * mla_g.kv_lora; i < ref_cache.size(); ++i) cache[i] = ref_cache[i];
                    for (size_t i = 0; i < ref.size(); ++i) attn_out[i] = ref[i];
                }
            } else {
                for (int t = 0; t < tokens; ++t) {
                    const int cells = state.mla_len[slot];
                    kernels::glm::MlaIntermediates want;
                    want.kv = cache + (size_t) cells * (size_t) mla_g.kv_lora;
                    if (!glm_stage_mla(*w.mla, mla_g, xn.data() + (size_t) t * ne, cells + 1, cache,
                                       attn_out.data() + (size_t) t * ne, err, &want)) {
                        err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " MLA: " + err;
                        return false;
                    }
                    state.mla_len[slot] = cells + 1;
                }
            }
        } else {
            const int slot = state.kda_index ? state.kda_index[layer] : -1;
            if (slot < 0 || !state.kda_state) {
                err = "glm_trunk_forward_batch: KDA layer has no state slot";
                return false;
            }
            float* conv = state.kda_conv ? state.kda_conv[slot] : nullptr;
            if (!glm_stage_kda(xn.data(), *w.kda, kda_g, tokens, attn_out.data(), state.kda_state[slot], err, conv)) {
                err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " KDA: " + err;
                return false;
            }
        }
        {
            const double a = lap();
            phase_ms[1] += a;
            if (is_mla) phase_ms[3] += a;
        }
        if (!for_tokens(tokens, err, [&](int t, std::string& e) {
                return glm_stage_hc_post(attn_out.data() + (size_t) t * ne, cur + (size_t) t * row, mix_a[(size_t) t], ne,
                                         mid.data() + (size_t) t * row, e) &&
                       glm_stage_hc_norm(mid.data() + (size_t) t * row, ne, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale,
                                         w.ffn_norm, ffn_in.data() + (size_t) t * ne, &mix_f[(size_t) t], hc_rms_eps,
                                         stream, e);
            })) {
            err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " attention hc_post / FFN site: " + err;
            return false;
        }
        phase_ms[0] += lap();
        if (is_dense) {
            const void* weights[3] = {w.ffn_gate, w.ffn_up, w.ffn_down};
            bool used_batch = glm_try_native_ffn_batch(weights, w.ffn_types, *w.moe_g, tokens,
                                                        ffn_in.data(), ffn_out.data(), w.clamp_limit);
            if (!used_batch) {
                for (int t = 0; t < tokens; ++t) {
                    if (!glm_stage_ffn(ffn_in.data() + (size_t) t * ne, w.ffn_gate, w.ffn_up, w.ffn_down, *w.moe_g,
                                       ffn_out.data() + (size_t) t * ne, w.clamp_limit, err, w.ffn_types)) {
                        err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " dense FFN: " + err;
                        return false;
                    }
                }
            }
        } else {
            if (!glm_stage_moe_native_batch(ffn_in.data(), tokens, w.moe_router, w.moe_probs_b, *w.moe_g,
                                            layer, *w.moe_fmt, w.blob_fn, w.blob_ctx, w.shexp_g, w.shexp,
                                            w.shexp_types, w.shexp_clamp, ffn_out.data(), err)) {
                err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " routed FFN: " + err;
                return false;
            }
        }
        phase_ms[2] += lap();
        if (!for_tokens(tokens, err, [&](int t, std::string& e) {
                return glm_stage_hc_post(ffn_out.data() + (size_t) t * ne, mid.data() + (size_t) t * row,
                                         mix_f[(size_t) t], ne, nxt + (size_t) t * row, e);
            })) {
            err = "glm_trunk_forward_batch: layer " + std::to_string(layer) + " FFN hc_post: " + err;
            return false;
        }
        phase_ms[0] += lap();
        std::swap(cur, nxt);
    }
    copy_floats(l_out, cur, n * row);
    if (std::getenv("STRATA_GLM_TIMING"))
        std::fprintf(stderr, "BATCH_TIMING tokens=%d hc=%.0f attn=%.0f (MLA %.0f, KDA %.0f) ffn=%.0f ms\n", tokens,
                     phase_ms[0], phase_ms[1], phase_ms[3], phase_ms[1] - phase_ms[3], phase_ms[2]);
    return true;
}

}  // namespace strata::core::glm
