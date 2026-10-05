// src/kernels/glm_norm.cpp - the GLM decoder block's RMSNorm.  Host code, float, no CUDA.
#include "strata/kernels/glm_norm.hpp"

#include <cmath>

namespace strata::kernels::glm {

void rms_norm_gain(const float* w, int n, const float* x, float* out, float eps) {
    // Match llama.cpp's rms_norm_f32 accumulation exactly: FLOAT sum-of-squares (not double) and rsqrtf,
    // then scale*x*w. The port previously summed in double with a double sqrt; a bit-match needs float.
    float ss = 0.0f;
    for (int i = 0; i < n; ++i) ss += x[i] * x[i];
    const float inv = 1.0f / std::sqrt(ss / (float) n + eps);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

}  // namespace strata::kernels::glm
