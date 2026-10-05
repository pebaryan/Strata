/// The GLM-4.7-Flash (deepseek2) trunk loop, as engine code rather than a gate's inline loop.
///
/// One token through blocks 0..n_layer-1:
///
///     for l:  xb = rms_norm(x, attn_norm_l) ;  x = x + MLA_l(xb, cache_l) ;  x = x + FFN_l(rms_norm(x, ffn_norm_l))
///
/// where FFN is the dense stem for block 0 and the MoE (routed experts + the +1 shared expert) for the
/// rest.  Every operator it calls is independently gated (glm47_mla/moe/block/trunk_parity); this is the
/// assembly a runner drives.
///
/// WEIGHTS ARE SUPPLIED, NOT OWNED: the loop asks for one layer at a time and, for a routed expert, asks
/// an `expert fn` for the expert the router CHOSE.  So it can be driven from a bound pack, a fixture, or
/// in-memory weights with no change, and it never learns where a weight came from.
#pragma once

#include <string>
#include <vector>

#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"

namespace strata::kernels::cpu { struct NativeFmt; }   // the pack's expert descriptor (glm_moe_native)

namespace strata::core::glm {

/// One block's weights.  The fields a block's kind does not need are left null and the loop rejects the
/// wiring if the ones it does need are missing (an MLA block with no MLA weights is a wiring bug, not a
/// fallback).  All pointers are borrowed for the duration of the call.
struct Glm47TrunkLayer {
    int kind = 1;                                 ///< 0 = the dense stem (block 0); 1 = MoE
    const float* attn_norm = nullptr;             ///< [n_embd]
    const float* ffn_norm = nullptr;              ///< [n_embd]
    const kernels::glm::MlaWeights* mla = nullptr;

    // the dense stem (block 0)
    const float* ffn_gate = nullptr;              ///< [dense_ff][n_embd]
    const float* ffn_up = nullptr;                ///< [dense_ff][n_embd]
    const float* ffn_down = nullptr;              ///< [n_embd][dense_ff]
    kernels::glm::MoeGeometry dense_g{};          ///< n_embd and ff for the dense FFN
    const void* dense_dev[3] = {nullptr, nullptr, nullptr};  ///< the dense FFN's device (quantized) blocks
    const int* dense_types = nullptr;             ///< their {gate,up,down} GGML types, or null (host float)

    // a routed block
    const float* moe_router = nullptr;            ///< [n_expert][n_embd]
    const float* moe_probs_b = nullptr;           ///< [n_expert]
    const kernels::glm::MoeGeometry* moe_g = nullptr;   ///< ff is the shared expert's width too
    const float* const* shexp = nullptr;          ///< {gate, up, down} of the +1 shared expert

    // OPTIONAL native (pack-quantized) MoE.  When moe_native_fmt and moe_native_blob are set, the routed
    // experts AND the shared expert are computed by the engine's native stage (glm_stage_moe_native)
    // straight from the pack's blobs - the path the model actually runs, and the only way full depth is
    // affordable (dequantising 644 experts per token is not).  moe_router/moe_probs_b stay float (routing
    // is float on every path).  A gate leaves moe_native_fmt null and takes the float moe_forward branch.
    const kernels::cpu::NativeFmt* moe_native_fmt = nullptr;
    const uint8_t* (*moe_native_blob)(void*, int, int) = nullptr;
    void* moe_native_ctx = nullptr;
    const int* shexp_types = nullptr;             ///< {gate,up,down} GGML types of the shared expert, or null (float fallback)
};

/// `(ctx, layer, expert) -> {gate, up, down}` for an expert, or null if it has none.  The loop routes
/// FIRST and then asks for the chosen expert, so a provider never has to predict the routing.
using Glm47ExpertFn = const float* const* (*)(void* ctx, int layer, int expert);

/// Runs one token (already a hidden state - the embedding is the caller's) through blocks 0..n_layer-1.
/// `caches` holds one interleaved [n][kv_lora + n_rot] MLA buffer per layer; this grows the layer's buffer
/// to `pos + 1` rows and appends the token's own row.  `cache_index`, if given, maps layer -> cache slot
/// (identity when null) so a caller can alias buffers (used by the gate's teeth run).
/// `ids_out`, if given, is resized to n_layer and, for each MoE block, holds the selected expert ids in
/// the router's rank order (dense blocks get an empty vector).  `per_layer_out`, if given, is resized to
/// n_layer and holds each block's output hidden state (so a caller can localise a mismatch to a block).
/// `out` receives the final hidden state.  Returns false and sets `err` on bad wiring.
bool glm47_trunk_forward(const Glm47TrunkLayer* layers, int n_layer,
                         const kernels::glm::MlaGeometry& mla_g, const float* x, int pos, float eps,
                         std::vector<std::vector<float>>* caches, const int* cache_index,
                         Glm47ExpertFn expert_fn, void* expert_ctx, float* out,
                         std::vector<std::vector<int32_t>>* ids_out,
                         std::vector<std::vector<float>>* per_layer_out, std::string& err);

/// Diagnostics: cumulative milliseconds spent in the MLA (attention) vs the FFN (MoE/dense) stages, summed
/// over every glm47_trunk_forward call.  A runner prints them to localise the per-token cost.
double glm47_trunk_mla_ms();
double glm47_trunk_ffn_ms();
double glm47_trunk_dense_ms();

}  // namespace strata::core::glm
