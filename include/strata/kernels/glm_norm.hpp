// include/strata/kernels/glm_norm.hpp - the GLM decoder block's RMSNorm (branch glm47-port).
//
//     out = x / sqrt(mean(x^2) + eps) * w
//
// i.e. ggml_rms_norm followed by a plain multiply.  The gain is NOT folded as (1 + w): some norms in
// this engine do fold, and folding both sides of a gate hides the difference, so the block gate reads
// this exactly and the reference is written the same way.  `eps` is the artifact's
// attention.layer_norm_rms_epsilon (1e-5 for GLM-4.7-Flash); it is a parameter, not a constant, because
// getting it wrong (e.g. using hyper_connection.epsilon) is invisible when both sides pick the same one.
#pragma once

namespace strata::kernels::glm {

/// `x`, `out` and `w` are length `n`, row-major, and `out` may alias `x`.  The sum of squares is
/// accumulated in double - ggml's kernel uses float, but the reference here is float64 and the
/// difference is ~1e-7, far under the gate's tolerance.
void rms_norm_gain(const float* w, int n, const float* x, float* out, float eps);

}  // namespace strata::kernels::glm
