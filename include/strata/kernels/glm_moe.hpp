// include/strata/kernels/glm_moe.hpp - GLM-5.3-Flash's MoE feed-forward and its router.
//
// Transcribed from the shared build_moe_ffn that GLM's graph calls, plus the GLM graph's own wiring:
//
//     logits   = ffn_gate_inp @ x                       [n_expert]
//     probs    = sigmoid(logits)                        gating func 2 = SIGMOID, not softmax
//     sel      = probs + exp_probs_b                    the bias drives SELECTION ONLY
//     ids      = top_k(sel, 8)                          no expert groups (n_expert_groups = 1)
//     weights  = probs[ids]                             gathered from the UNBIASED probs
//     weights /= max(sum(weights), 6.103515625e-05)     the clamp is the f16 minimum normal
//     weights *= expert_weights_scale                   2.5, applied AFTER the normalization
//     moe      = sum_i weights[i] * down_i(silu(gate_i(x)) * up_i(x))
//     shexp    = down_s(silu(gate_s(x)) * up_s(x))      a plain parallel-SiLU FFN, added unweighted
//     out      = moe + shexp
//
// Two orderings here are silent if wrong: the bias biases the choice while the weights come from the
// unbiased probability, and the scale is applied to the normalised weights rather than before.
#pragma once

#include <cstdint>

namespace strata::kernels::glm {

struct MoeGeometry {
    int n_embd = 4096;
    int n_expert = 288;
    int n_used = 8;      ///< expert_used_count
    int ff = 2048;       ///< expert_feed_forward_length
    float w_scale = 2.5f;   ///< expert_weights_scale
    bool norm_w = true;     ///< expert_weights_norm

    /// ggml_swiglu_clamp's limit, from the artifact: swiglu_clamp_exp for the routed experts and
    /// swiglu_clamp_shexp for the shared expert AND the leading dense FFN (the reference reaches the
    /// dense site through build_ffn, which reads the shexp array).  10.0 for every layer of this model.
    /// 0 disables it, and that is a real behaviour difference, not a neutral default: with a limit of
    /// 10 it is invisible until a pre-activation exceeds 10, so a caller that forgets it will still
    /// look right on tame inputs.  Callers must set both from the artifact.
    float clamp_exp = 0.0f;
    float clamp_shexp = 0.0f;
};

/// The weights one MoE site needs.  The expert tensors are per-expert C rows: gate/up are [ff][n_embd],
/// down is [n_embd][ff] (the engine's expert blobs are the same layout, streamed rather than resident).
struct MoeWeights {
    const float* router = nullptr;    ///< [n_expert][n_embd]
    const float* probs_b = nullptr;   ///< [n_expert], ffn_exp_probs_b (may be null)
    const float(*expert_gate)[1] = nullptr;  ///< unused placeholder, see moe_forward's expert arrays
};

/// expert_weights_norm's clamp: the smallest normal f16, which is what the reference clamps the sum to.
inline constexpr float MOE_F16_MIN = 6.103515625e-05f;

/// The router.  `ids_out` gets `g.n_used` expert indices, best first with ties by ascending index (ggml's
/// argsort is index-ordered); `weights_out` gets their weights, normalised then scaled.
void moe_route(const float* router, const float* probs_b, const MoeGeometry& g, const float* x,
               int32_t* ids_out, float* weights_out, float* probs_out = nullptr);

/// One expert: down(silu(gate @ x) * (up @ x)).  gate/up are [ff][n_embd], down is [n_embd][ff].
/// `clamp_limit` is ggml_swiglu_clamp's limit (MoeGeometry::clamp_exp for a routed expert,
/// clamp_shexp for the shared expert / dense FFN); 0 disables it.
void expert_ffn(const float* wg, const float* wu, const float* wd, const MoeGeometry& g, const float* x,
                float* out, float clamp_limit);

/// The assembled site: route, then sum the weighted experts, then add the shared expert.
/// `experts[i]` is the i-th selected expert's {gate, up, down}, in the SAME order the router returns;
/// `shared` is one {gate, up, down} triple.  Both are arrays of pointers, so a caller holding its expert
/// weights as separate buffers can pass them directly - no cast, and no second level of indirection to
/// get wrong.
void moe_forward(const float* router, const float* probs_b, const MoeGeometry& g, const float* x,
                 const float* const* const* experts, const float* const* shared, float* out,
                 int32_t* ids_out = nullptr, float* weights_out = nullptr, float* moe_out = nullptr,
                 float* shexp_out = nullptr);

}  // namespace strata::kernels::glm
