// GLM-5.3's block chain, stage by stage, written against the interface recorded in
// tools/glm5_binding_contract.txt.
//
// Stage 1 is the hyper-connection front end, which every block performs before its attention or ffn site:
//
//     hc_pre(x, hc_fn, hc_base, hc_scale) -> layer_in
//     rms_norm_weighted(layer_in, attn_norm_w, eps = 1e-5)
//
// `x` is the block's [HC][n_embd] stream state (row-major), and `layer_in` is the single [n_embd] vector the
// attention (or ffn) site consumes.  The mix produced by hc_pre is what hc_post needs afterwards, so it is an
// output of this stage rather than an internal detail.
#pragma once

#include <string>

#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/glm_mla.hpp"

#include "strata/kernels/glm_hc.hpp"

#include "strata/core/glm_block_kinds.hpp"   // glm_attention_is_mla / glm_ffn_is_dense, compile-time asserted

namespace strata::core::glm {

/// One block's hyper-connection front end.
///
/// \param x          [HC][n_embd] stream state, row-major.  GLM calls this hc_init / hc_attn_post / hc_ffn_post.
/// \param hc_fn      the site's [(2+hc)*hc][hc*n_embd] matrix, as the binding serves it (blk.N.hc_attn_fn.weight).
/// \param hc_base    the site's [(2+hc)*hc] bias           (blk.N.hc_attn_base.weight).
/// \param hc_scale   the site's [hc] scale factors         (blk.N.hc_attn_scale.weight).
/// \param norm_w     the site's norm weight, [n_embd]      (blk.N.attn_norm.weight or blk.N.ffn_norm.weight).
/// \param layer_in   [n_embd] out: the normed vector the attention/ffn kernel takes.
/// \param mix        out: passed to hc_post once the site has produced its output.
/// \param eps        the norm's epsilon.  This is `attention.layer_norm_rms_epsilon` (1e-5) - NOT
///                   `hyper_connection.epsilon` (1e-6), which glm_hc.hpp:30 warns about by name.
/// \param stream     a CUDA stream, or nullptr for synchronous work.
///
/// The reference's tensor for `layer_in` is hc_attn_pre-N (pre-norm) and attn_norm-N (post-norm); both are in
/// the phase-8 dumps, which is what makes this stage gateable on its own rather than only end to end.
bool glm_stage_hc_norm(const float* x, int n_embd, const float* hc_fn, const float* hc_base,
                       const float* hc_scale, const float* norm_w, float* layer_in,
                       kernels::glm::HcMix* mix, float eps, void* stream, std::string& err);

/// Stage 2: the attention call.  For a KDA block, stage 1's output is the kernel's input directly, so this adds no
/// arithmetic - it is the wiring that lets the chain call the kernel with a block's own bound weights.
/// `state` must be zeroed for a fresh sequence (nh*hd*hd floats).
bool glm_stage_kda(const float* xn, const kernels::glm::KdaWeights& w, const kernels::glm::KdaGeometry& g,
                   int tokens, float* out, float* state, std::string& err);

/// Stage 3: the site's hc_post - the hyper-connection combination, which produces HC rows for the next site's
/// hc_pre rather than a single vector.  `residual` is the block input `x` ([HC][n_embd]).
bool glm_stage_hc_post(const float* site_out, const float* residual, const kernels::glm::HcMix& mix,
                       int n_embd, float* out, std::string& err);

/// Stage 4: the FFN site's feed-forward - one expert_ffn call, serving both a leading dense block and a routed
/// expert.  `clamp_limit` is the artifact's swiglu clamp for this site (10.0 for this model).
bool glm_stage_ffn(const float* xn, const float* wg, const float* wu, const float* wd,
                   const kernels::glm::MoeGeometry& g, float* out, float clamp_limit, std::string& err);

/// One block's weights, as plain pointers.  Deliberately NOT the binding layer's structures: every gate in this port
/// drives code with plain arrays loaded from fixture files, and a loop that takes plain pointers can be driven by
/// exactly those arrays - so the chain is testable stage by stage against the dump without a bound artifact, and
/// glm_layer.cpp stays independent of the loader.
struct GlmBlockWeights {
    // the attention site's hyper-connection front end
    const float* hc_attn_fn    = nullptr;
    const float* hc_attn_base  = nullptr;
    const float* hc_attn_scale = nullptr;
    const float* attn_norm     = nullptr;
    // the attention itself (a KDA block; the sixteen weight arrays are already grouped)
    const kernels::glm::KdaWeights*  kda      = nullptr;
    const kernels::glm::KdaGeometry* kda_geom = nullptr;
    // the FFN site's front end
    const float* hc_ffn_fn    = nullptr;
    const float* hc_ffn_base  = nullptr;
    const float* hc_ffn_scale = nullptr;
    const float* ffn_norm     = nullptr;
    // the feed-forward, plus its geometry (ff = 12288 dense vs 2048 routed) and clamp
    const float* ffn_gate = nullptr;
    const float* ffn_up   = nullptr;
    const float* ffn_down = nullptr;
    const kernels::glm::MoeGeometry* ffn_geom = nullptr;
    float clamp_limit = 10.0f;      ///< swiglu_clamp_shexp for a leading dense block
};

/// One whole block: hc_pre+norm -> attention -> hc_post -> hc_pre+norm -> FFN -> hc_post.
///
/// `x` is the block input as HC streams ([HC][n_embd]) and `l_out` the output in the same shape, which is the next
/// block's input.  `state` is the attention's recurrent S matrix and must be zeroed for a fresh sequence.  Buffers
/// are allocated internally; nothing is carried between calls except what the caller passes.
bool glm_block_forward(const float* x, int tokens, const GlmBlockWeights& w, float hc_rms_eps,
                       float* l_out, float* state, void* stream, std::string& err);

/// Stage 3b: the MLA site, for blocks whose attention is latent rather than KDA (the artifact declares the kind
/// per layer in glm5next.attention.head_count_kv).  `cache` is [n_cache][kv_lora] of latents, the current
/// position included, since the reference appends before attending.
/// `want` is how the caller gets this token's LATENT out.  mla_forward computes it to attend with, but without a
/// place to put it the caller cannot append it to the cache - and the cache is how MLA carries state across tokens,
/// so a caller that drops it gives every token a cache of the same value it started with.  Optional because a
/// single-token parity check does not need it; a sequence does.
bool glm_stage_mla(const kernels::glm::MlaWeights& w, const kernels::glm::MlaGeometry& g, const float* x,
                   int n_cache, const float* cache, float* out, std::string& err,
                   kernels::glm::MlaIntermediates* want = nullptr);

/// Stage 4b: the MoE site - route, weighted expert sum, and the unweighted shared expert, all inside moe_forward.
/// Takes pointers to the expert weights rather than owning them, because the experts are streamed from the pack.
/// `experts[i]` is the i-th selected expert's {gate, up, down} in the router's order; `shared` is one triple.
bool glm_stage_moe(const float* xn, const float* router, const float* probs_b,
                   const kernels::glm::MoeGeometry& g, const float* const* const* experts,
                   const float* const* shared, float* out, std::string& err);


/// Stage 5 (first half): the head's mean-over-streams and output_norm.  The projection follows outside this file.
bool glm_stage_head_mean_norm(const float* hc, int hc_streams, int n_embd, const float* output_norm, float* hidden,
                              std::string& err);


/// Stage 5 (second half): the tied output projection over an already-dequantized [vocab][n_embd] matrix, and argmax.
bool glm_stage_head_project(const float* W, int vocab, int n_embd, const float* hidden, int& argmax, float& best,
                            std::string& err);

}  // namespace strata::core::glm
