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
#include <functional>
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
    int shexp_types[3] = {0, 0, 0};                   ///< native GGML types, all zero for host floats
    /// The DENSE feed-forward's three types, for blocks 0..2.  The shared expert had types and the dense FFN did not,
    /// which is the whole reason the stem was still being dequantised to host floats every token: ffn_gate/up/down at
    /// 12,288 x 4,096 are about 201 MB each, and three of them across three layers is 1.8 GB per token of materialisation
    /// that a device MMVQ read removes entirely.
    int ffn_types[3] = {0, 0, 0};
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
/// KDA is recurrent: one S matrix and three convolution histories per KDA layer, zeroed for a fresh sequence.
/// MLA is a growing cache of latents:
/// the reference appends the current position's latent BEFORE attending, so the loop writes the latent mla_forward
/// produces into the next slot and the caller's live count grows by one each token.
struct GlmTrunkState {
    float* const* kda_state = nullptr;      ///< one nh*hd*hd buffer per KDA layer, caller-owned, zeroed to start
    float* const* kda_conv = nullptr;       ///< one 3*(d_conv-1)*d_inner history per KDA layer, zeroed to start
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
/// `first_layer` is the index the loop STARTS at, and it is not cosmetic: the dispatch
/// (glm_attention_is_mla / glm_ffn_is_dense), the weight provider and the state maps are all keyed by the artifact's
/// layer number, so a run that starts at layer 3 must say so or it would be dispatched as layer 0 - KDA instead of
/// MLA, dense instead of routed.  Defaults to 0 for a run over the whole trunk.
/// Reports one stage of one layer: the name, the buffer, and its length.  This exists so a block whose l_out
/// disagrees can be SPLIT - compare the attention site's output to decide whether the error is before the FFN, and
/// the FFN's output to decide whether it is inside it - instead of arguing about which half is at fault.  The names
/// are the ones the reference drivers use ("hc_attn_pre", "attn_norm", "attn_output", "hc_attn_post", "hc_ffn_pre",
/// "ffn_norm", "ffn_out", "l_out"), so a dump and this stream are directly comparable.  Lengths: the hc_* stages are n_embd
/// times the stream count, the rest are n_embd.
typedef void (*GlmStageFn)(void* ctx, int layer, const char* name, const float* data, int n);

bool glm_trunk_forward(const float* x, int layers, GlmTrunkProvider provider, void* provider_ctx,
                       const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                       float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err,
                       int first_layer = 0, GlmStageFn stage_fn = nullptr, void* stage_ctx = nullptr);

/// Optional worker pool for the prompt path's per-token host stages (hyper-connection pre/norm/post), which are
/// independent across tokens.  `fn(n, job)` must run job(0..n-1), in any order and on any threads, and return when all are
/// done.  Unset, those loops run serially.
using GlmParallelForFn = void (*)(int n, const std::function<void(int)>& job);
void glm_set_parallel_for(GlmParallelForFn fn);
/// Runs job(0..n-1) on that pool, or serially when none is installed.  For other core stages with token-independent host work.
void glm_parallel_for(int n, const std::function<void(int)>& job);

/// Optional notification from the prompt path: called at the top of every layer with that layer's index, so every earlier
/// layer is finished and the layers after it will be wanted next.  A streaming expert source uses it to keep reading ahead
/// while this layer computes (a long chunk uses nearly every expert of every layer, so whole layers are worth fetching).
using GlmPrefetchFn = void (*)(void* ctx, int layer);
void glm_set_layer_prefetch(GlmPrefetchFn fn, void* ctx);

/// Decode with the hyper-connection streams resident on the device (see kernels/glm_decode_trunk.hpp): per layer, hc_pre, the
/// attention block and hc_post chain on one stream, and only the FFN site exchanges data with the host.  Used by
/// glm_trunk_forward for single-token calls when every layer is eligible (native projection types, device KDA/MLA blocks on,
/// context within 8192) and no stage callback is attached; otherwise the host loop runs.  Off by default; the runner enables it.
void glm_set_device_trunk(bool enabled);

/// Layer-major prompt path for a causal token chunk. Input/output rows are [tokens][HC][n_embd].
/// KDA recurrence and MLA cache updates advance in token order within each layer; only independent
/// projections are batched. State is left at the chunk's final position for the next chunk/decode call.
bool glm_trunk_forward_batch(const float* x, int tokens, int layers, GlmTrunkProvider provider, void* provider_ctx,
                             const kernels::glm::KdaGeometry& kda_g, const kernels::glm::MlaGeometry& mla_g,
                             float hc_rms_eps, GlmTrunkState& state, float* l_out, void* stream, std::string& err,
                             int first_layer = 0);

}  // namespace strata::core::glm
