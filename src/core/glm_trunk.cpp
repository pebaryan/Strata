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

#include <vector>

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

}  // namespace

bool glm_trunk_forward(const float* x, int layers, GlmTrunkProvider provider, void* provider_ctx,
                       const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                       float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err) {
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

    for (int layer = 0; layer < layers; ++layer) {
        const bool is_mla = glm_attention_is_mla(layer) == 1;
        const bool is_dense = glm_ffn_is_dense(layer);

        GlmTrunkLayerWeights w;
        if (!provider(provider_ctx, layer, w, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + ": " + err;
            return false;
        }
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
        if (!glm_stage_hc_norm(cur, ne, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale, is_mla ? w.attn_norm : nullptr,
                               xn.data(), &mix_a[0], hc_rms_eps, stream, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " attention site: " + err;
            return false;
        }
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
            if (!glm_stage_mla(*w.mla, mla_g, xn.data(), cells + 1, cache, attn_out.data(), err)) {
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
            if (!glm_stage_kda(xn.data(), *w.kda, kda_g, 1, attn_out.data(), state.kda_state[slot], err)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " KDA: " + err;
                return false;
            }
        }
        if (!glm_stage_hc_post(attn_out.data(), cur, mix_a[0], ne, mid.data(), err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " attention hc_post: " + err;
            return false;
        }

        // ---- the FFN site ----
        if (!glm_stage_hc_norm(mid.data(), ne, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale, w.ffn_norm, ffn_in.data(),
                               &mix_f[0], hc_rms_eps, stream, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " FFN site: " + err;
            return false;
        }
        if (is_dense) {
            if (!glm_stage_ffn(ffn_in.data(), w.ffn_gate, w.ffn_up, w.ffn_down, *w.moe_g, ffn_out.data(),
                               w.clamp_limit, err)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " dense FFN: " + err;
                return false;
            }
        } else {
            if (!glm_stage_moe_native(ffn_in.data(), w.moe_router, w.moe_probs_b, *w.moe_g, layer, *w.moe_fmt,
                                      w.blob_fn, w.blob_ctx, w.shexp_g, w.shexp, w.shexp_clamp, ffn_out.data(), err)) {
                err = "glm_trunk_forward: layer " + std::to_string(layer) + " routed FFN: " + err;
                return false;
            }
        }
        if (!glm_stage_hc_post(ffn_out.data(), mid.data(), mix_f[0], ne, nxt, err)) {
            err = "glm_trunk_forward: layer " + std::to_string(layer) + " FFN hc_post: " + err;
            return false;
        }

        float* swap = cur; cur = nxt; nxt = swap;   // this layer's output is the next layer's input
    }

    copy_floats(l_out, cur, (size_t) hc * ne);
    return true;
}

}  // namespace strata::core::glm
