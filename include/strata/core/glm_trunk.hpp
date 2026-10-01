/// The trunk: blocks 0..44, dispatching on the kind the artifact declares.
///
/// This is the assembly, not the arithmetic.  Every stage it calls is already written and independently gated:
/// glm_stage_hc_norm (both hyper-connection sites), glm_stage_kda, glm_stage_mla, glm_stage_hc_post, glm_stage_ffn for
/// the leading dense blocks and glm_stage_moe_native for the routed ones.  What was missing was the thing that walks
/// 45 blocks in order, threads the two hyper-connection sites per block, and keeps each attention kind's state.
///
/// WEIGHTS ARE SUPPLIED, NOT OWNED.  The loop asks a provider for one layer's weights and never learns where they
/// came from, so it can be driven from a bound artifact or from fixture files with no change - the distinction this
/// port has relied on throughout, and the reason every gate here can run without a pack.
#pragma once

#include <cstdint>
#include <string>

#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/core/glm_block_kinds.hpp"

namespace strata::kernels::cpu { struct NativeFmt; }

namespace strata::core::glm {

/// One layer's weights, whichever kind it is.  The fields a layer does not need are left null, and the loop asserts
/// that the ones its kind DOES need are present - an MLA layer given no MLA weights is a wiring bug, not a fallback.
struct GlmTrunkLayerWeights {
    // the two hyper-connection sites and the two norm weights, common to every trunk layer
    const float* hc_attn_fn = nullptr;
    const float* hc_attn_base = nullptr;
    const float* hc_attn_scale = nullptr;
    const float* attn_norm = nullptr;
    const float* hc_ffn_fn = nullptr;
    const float* hc_ffn_base = nullptr;
    const float* hc_ffn_scale = nullptr;
    const float* ffn_norm = nullptr;

    // the attention: exactly one of these, per glm_attention_is_mla(layer)
    const kernels::glm::KdaWeights* kda = nullptr;
    const kernels::glm::MlaWeights* mla = nullptr;

    // the feed-forward: dense uses the three matrices and its own geometry, routed uses the router and the blobs.
    // `moe_g` is the routed geometry (ff = expert_feed_forward_length); `shexp_g` is the shared expert's, which is a
    // separate width and therefore a separate geometry rather than an assumption.
    const float* ffn_gate = nullptr;
    const float* ffn_up = nullptr;
    const float* ffn_down = nullptr;
    const float* moe_router = nullptr;
    const float* moe_probs_b = nullptr;
    const kernels::glm::MoeGeometry* moe_g = nullptr;
    const kernels::glm::MoeGeometry* shexp_g = nullptr;
    const float* const* shexp = nullptr;              ///< {gate, up, down} of the shared expert, or null
    const kernels::cpu::NativeFmt* moe_fmt = nullptr;
    const uint8_t* (*blob_fn)(void*, int, int) = nullptr;
    void* blob_ctx = nullptr;
    float shexp_clamp = 10.0f;
    float clamp_limit = 10.0f;                        ///< swiglu_clamp_shexp for a leading dense block
};

/// Supplies one layer's weights.  Returning false with a reason stops the trunk - it does not skip the layer, because
/// a skipped layer in a 45-layer chain produces a plausible answer rather than an error.
using GlmTrunkProvider = bool (*)(void* ctx, int layer, GlmTrunkLayerWeights& out, std::string& err);

/// The per-layer state the trunk carries between tokens, one entry per layer of each kind.
///
/// KDA is recurrent: one S matrix per KDA layer, zeroed for a fresh sequence.  MLA is a growing cache of latents:
/// the reference appends the current position's latent BEFORE attending, so the loop writes the latent mla_forward
/// produces into the next slot and the caller's live count grows by one each token.
struct GlmTrunkState {
    float* const* kda_state = nullptr;      ///< one nh*hd*hd buffer per KDA layer, caller-owned, zeroed to start
    float* const* mla_cache = nullptr;      ///< one [max_cells][kv_lora] buffer per MLA layer, caller-owned
    int* mla_len = nullptr;                 ///< the live cell count per MLA layer (in: cells before this token)
    int* kda_index = nullptr;               ///< layer -> its slot in kda_state, or -1 (caller-supplied map)
    int* mla_index = nullptr;               ///< layer -> its slot in mla_cache, or -1
};

/// One token through every trunk layer.  `x` is [HC][n_embd] (the embedding's streams for block 0, and the previous
/// block's output thereafter); `l_out` is the same shape and becomes the next token's input.
///
/// Returns false on the first layer that fails, with the layer number in `err`, so a failure names a block rather
/// than the trunk.
bool glm_trunk_forward(const float* x, int layers, GlmTrunkProvider provider, void* provider_ctx,
                       const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                       float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err);

}  // namespace strata::core::glm
