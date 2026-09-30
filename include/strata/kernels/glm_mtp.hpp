// include/strata/kernels/glm_mtp.hpp - GLM-5.3-Flash's NextN/MTP draft head's own arithmetic.
//
// The head is mostly REUSE of operators this port already verifies: the same MLA (phase 4a) and the same
// sigmoid-gated MoE with a shared expert (phase 6).  What is new, and therefore what this file covers:
//
//     e_norm = rms_norm(tok_embd(token), nextn.enorm)
//     h_norm = rms_norm(h,              nextn.hnorm)
//     concat = concat(e_norm, h_norm)          e_norm FIRST, along dim 0 -> [2*n_embd]
//     cur    = nextn.eh_proj @ concat
//     h_nextn = rms_norm(cur, nextn.shared_head_norm)     feeds the LM head
//
// plus the block's RESIDUAL SHAPE: blk.45 has no hc_* tensors, so where the trunk runs the mHC this
// block runs two plain adds (attn_out + eh_proj_out, then moe_out + cur).
//
// The concat order is the silent detail: swapping the halves still produces a full-size, plausible
// vector, and the head would then be conditioned on the embedding where it expects the hidden state.
// glm5_mtp_reference.py writes a fixture whose head end is checked against exactly that.
#pragma once

#include <cstdint>

namespace strata::kernels::glm {

/// attention.layer_norm_rms_epsilon, the model's f_norm_rms_eps - used by both norms here.
inline constexpr float MTP_RMS_EPS = 1e-5f;

/// The head's prologue: enorm/hnorm, the concat, and eh_proj.
/// `e` is the token embedding and `h` the previous post-norm hidden state, both [n_embd].
/// `concat_out` receives [e_norm | h_norm] ([2*n_embd]) if non-null; `cur_out` receives eh_proj @ concat.
void mtp_input(const float* enorm, const float* hnorm, const float* eh_proj, int n_embd, const float* e,
               const float* h, float* cur_out, float* concat_out = nullptr, float* e_norm_out = nullptr,
               float* h_norm_out = nullptr);

/// rms_norm(x, nextn.shared_head_norm) - what the LM head consumes (the reference's "h_nextn").
void mtp_head_norm(const float* head_norm_w, int n_embd, const float* x, float* out);

}  // namespace strata::kernels::glm
