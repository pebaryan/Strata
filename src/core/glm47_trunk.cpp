#include "strata/core/glm47_trunk.hpp"

#include <chrono>
#include <cstring>

#include "strata/core/glm_moe_native.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm_norm.hpp"

namespace strata::core::glm {
namespace glm = strata::kernels::glm;

namespace {
// GLM-4.7-Flash has no swiglu_clamp key, so both expert and dense SiLU are unclamped.  A nonzero limit
// here would be a silent behaviour change (invisible until a pre-activation exceeds it).
constexpr float kClamp = 0.0f;
// Cumulative stage timings (milliseconds), read back by glm47_trunk_mla_ms / glm47_trunk_ffn_ms.
double g_t_mla = 0.0, g_t_ffn = 0.0;
}  // namespace

double glm47_trunk_mla_ms() { return g_t_mla; }
double glm47_trunk_ffn_ms() { return g_t_ffn; }

bool glm47_trunk_forward(const Glm47TrunkLayer* layers, int n_layer,
                         const kernels::glm::MlaGeometry& mla_g, const float* x, int pos, float eps,
                         std::vector<std::vector<float>>* caches, const int* cache_index,
                         Glm47ExpertFn expert_fn, void* expert_ctx, float* out,
                         std::vector<std::vector<int32_t>>* ids_out,
                         std::vector<std::vector<float>>* per_layer_out, std::string& err) {
    if (!layers || n_layer <= 0 || !x || !caches || !out) {
        err = "glm47_trunk_forward: null argument";
        return false;
    }
    const int ne = mla_g.n_embd, q_lora = mla_g.q_lora, kv_lora = mla_g.kv_lora, n_rot = mla_g.n_rot;
    const int n_head = mla_g.n_head, head_dim = mla_g.head_dim;
    const int nope = head_dim - n_rot, kv_dim = kv_lora + n_rot;
    if (ne <= 0 || kv_lora <= 0 || n_rot <= 0 || n_head <= 0 || head_dim <= 0 || pos < 0) {
        err = "glm47_trunk_forward: bad geometry";
        return false;
    }
    if (ids_out) ids_out->assign((size_t) n_layer, {});
    if (per_layer_out) per_layer_out->assign((size_t) n_layer, {});

    std::vector<float> cur(x, x + ne);
    for (int l = 0; l < n_layer; ++l) {
        const Glm47TrunkLayer& ly = layers[l];
        if (!ly.attn_norm || !ly.ffn_norm || !ly.mla) {
            err = "glm47_trunk_forward: layer " + std::to_string(l) + " lacks a norm or MLA weights";
            return false;
        }
        std::vector<float>& cache = (*caches)[cache_index ? cache_index[l] : l];
        const auto t_attn0 = std::chrono::steady_clock::now();

        // site 1: attn_norm -> MLA -> residual.  The current token's kv / k_pe are computed once and
        // appended to this layer's cache, then attention runs over history + this token.
        std::vector<float> xb((size_t) ne);
        glm::rms_norm_gain(ly.attn_norm, ne, cur.data(), xb.data(), eps);
        cache.resize((size_t) (pos + 1) * kv_dim);
        std::vector<float> g_qr((size_t) q_lora), g_qn((size_t) n_head * nope), g_qp((size_t) n_head * n_rot);
        std::vector<float> g_kv((size_t) kv_lora), g_kp((size_t) n_rot), g_qc((size_t) n_head * kv_lora);
        std::vector<float> g_at((size_t) n_head * kv_lora), g_v((size_t) n_head * head_dim);
        glm::MlaIntermediates want;
        want.qr = g_qr.data(); want.q_nope = g_qn.data(); want.q_pe = g_qp.data(); want.kv = g_kv.data();
        want.k_pe = g_kp.data(); want.qcur = g_qc.data(); want.attn = g_at.data(); want.v = g_v.data();
        std::vector<float> attn((size_t) ne);
        // pass 1: harvest the new latent row only (kv_only - skips the absorption, attention and wo), append
        // it, then pass 2 attends over history + this token.
        want.kv_only = true;
        glm::mla_forward(*ly.mla, mla_g, xb.data(), pos + 1, cache.data(), attn.data(), want, pos);
        want.kv_only = false;
        std::memcpy(cache.data() + (size_t) pos * kv_dim, g_kv.data(), (size_t) kv_lora * sizeof(float));
        std::memcpy(cache.data() + (size_t) pos * kv_dim + kv_lora, g_kp.data(), (size_t) n_rot * sizeof(float));
        glm::mla_forward(*ly.mla, mla_g, xb.data(), pos + 1, cache.data(), attn.data(), want, pos);

        std::vector<float> x2((size_t) ne);
        for (int i = 0; i < ne; ++i) x2[(size_t) i] = cur[(size_t) i] + attn[(size_t) i];
        const auto t_ffn0 = std::chrono::steady_clock::now();

        // site 2: ffn_norm -> FFN -> residual
        std::vector<float> ffv((size_t) ne), ffnout((size_t) ne);
        glm::rms_norm_gain(ly.ffn_norm, ne, x2.data(), ffv.data(), eps);
        if (ly.kind == 0) {
            if (!ly.ffn_gate || !ly.ffn_up || !ly.ffn_down) {
                err = "glm47_trunk_forward: dense layer " + std::to_string(l) + " lacks its FFN weights";
                return false;
            }
            glm::expert_ffn(ly.ffn_gate, ly.ffn_up, ly.ffn_down, ly.dense_g, ffv.data(), ffnout.data(), kClamp);
        } else if (ly.moe_native_fmt && ly.moe_native_blob) {
            // the native (pack-quantized) path: glm_stage_moe_native routes and computes BOTH the routed
            // experts and the shared expert straight from the pack's blobs, so no expert is materialised.
            // Router weights stay float (routing is float on every path).
            if (!ly.moe_router || !ly.moe_probs_b || !ly.moe_g) {
                err = "glm47_trunk_forward: native MoE layer " + std::to_string(l) + " lacks router weights";
                return false;
            }
            const int k = ly.moe_g->n_used;
            std::vector<int> nids((size_t) k, -1);
            if (!glm_stage_moe_native(ffv.data(), ly.moe_router, ly.moe_probs_b, *ly.moe_g, l,
                                      *ly.moe_native_fmt, ly.moe_native_blob, ly.moe_native_ctx,
                                      ly.moe_g, ly.shexp, ly.shexp_types, kClamp, ffnout.data(), err,
                                      nids.data())) {
                return false;
            }
            if (ids_out) for (int i = 0; i < k; ++i) (*ids_out)[(size_t) l].push_back(nids[(size_t) i]);
        } else {
            if (!ly.moe_router || !ly.moe_probs_b || !ly.moe_g || !ly.shexp || !expert_fn) {
                err = "glm47_trunk_forward: MoE layer " + std::to_string(l) + " lacks " +
                      (expert_fn ? "weights" : "an expert fn");
                return false;
            }
            const int k = ly.moe_g->n_used;
            std::vector<int32_t> rids((size_t) k);
            std::vector<float> rw((size_t) k);
            glm::moe_route(ly.moe_router, ly.moe_probs_b, *ly.moe_g, ffv.data(), rids.data(), rw.data());
            std::vector<const float*> eptrs((size_t) k * 3);
            std::vector<const float* const*> experts((size_t) k);
            for (int i = 0; i < k; ++i) {
                const float* const* t = expert_fn(expert_ctx, l, (int) rids[(size_t) i]);
                if (!t) {
                    err = "glm47_trunk_forward: no weights for layer " + std::to_string(l) +
                          " expert " + std::to_string((int) rids[(size_t) i]);
                    return false;
                }
                eptrs[(size_t) i * 3 + 0] = t[0];
                eptrs[(size_t) i * 3 + 1] = t[1];
                eptrs[(size_t) i * 3 + 2] = t[2];
                experts[(size_t) i] = &eptrs[(size_t) i * 3];
                if (ids_out) (*ids_out)[(size_t) l].push_back(rids[(size_t) i]);
            }
            // moe_forward re-routes, but deterministically, and uses experts[] positionally by rank.
            std::vector<int32_t> gids((size_t) k);
            std::vector<float> gw((size_t) k);
            glm::moe_forward(ly.moe_router, ly.moe_probs_b, *ly.moe_g, ffv.data(), experts.data(),
                             ly.shexp, ffnout.data(), gids.data(), gw.data());
        }

        for (int i = 0; i < ne; ++i) cur[(size_t) i] = x2[(size_t) i] + ffnout[(size_t) i];
        g_t_mla += std::chrono::duration<double, std::milli>(t_ffn0 - t_attn0).count();
        g_t_ffn += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_ffn0).count();
        if (per_layer_out) (*per_layer_out)[(size_t) l] = cur;
    }
    std::memcpy(out, cur.data(), (size_t) ne * sizeof(float));
    return true;
}

}  // namespace strata::core::glm
