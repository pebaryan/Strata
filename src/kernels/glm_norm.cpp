// src/kernels/glm_norm.cpp - the GLM decoder block's RMSNorm.  Host code, float, no CUDA.
#include "strata/kernels/glm_norm.hpp"

#include <cmath>

namespace strata::kernels::glm {

void rms_norm_gain(const float* w, int n, const float* x, float* out, float eps) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * (double) x[i];
    const float inv = (float) (1.0 / std::sqrt(ss / (double) n + (double) eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

}  // namespace strata::kernels::glm
