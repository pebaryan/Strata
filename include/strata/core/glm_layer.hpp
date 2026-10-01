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

#include "strata/kernels/glm_hc.hpp"

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

}  // namespace strata::core::glm
