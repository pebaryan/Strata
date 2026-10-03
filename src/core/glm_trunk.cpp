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

namespace {
GlmPrefetchFn g_layer_prefetch = nullptr;
void* g_layer_prefetch_ctx = nullptr;
}  // namespace
void glm_set_layer_prefetch(GlmPrefetchFn fn, void* ctx) { g_layer_prefetch = fn; g_layer_prefetch_ctx = ctx; }

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
