// GLM-5.3 block chain, stage 1: the hyper-connection front end.  See glm_layer.hpp for the contract.

#include "strata/core/glm_layer.hpp"

#include <cmath>

#include "strata/kernels/elementwise.hpp"

namespace strata::core::glm {

bool glm_stage_hc_norm(const float* x, int n_embd, const float* hc_fn, const float* hc_base,
                       const float* hc_scale, const float* norm_w, float* layer_in,
                       kernels::glm::HcMix* mix, float eps, void* stream, std::string& err) {
    if (!x || !hc_fn || !hc_base || !hc_scale || !layer_in || !mix) {
        err = "glm_stage_hc_norm: null argument";
        return false;
    }
    if (n_embd <= 0) {
        err = "glm_stage_hc_norm: n_embd must be positive";
        return false;
    }

    // The site's input comes out of the streams first: hc_pre mixes [HC][n_embd] down to one [n_embd] vector
    // and records the mix, which hc_post needs once the site has produced its output.
    kernels::glm::hc_pre(x, hc_fn, hc_base, hc_scale, n_embd, layer_in, mix);

    // Then the site's own norm, ALSO on the host.  This was the last bug in the stage: the engine's
    // rms_norm_weighted (elementwise.cu:309) is a CUDA kernel, so handing it the host `layer_in` produced
    // "an illegal memory access was encountered".  hc_pre is host code and the reference normalizes on the
    // host, so the stage stays on one side of the boundary and does the norm here.
    //
    // The reference is ggml_rms_norm followed by a PLAIN multiply - y = rms_norm(x, eps) * w, with no (1 + w)
    // folding, which elementwise.hpp warns about explicitly because some norms in this family do fold.
    if (norm_w) {
        double ss = 0.0;
        for (int i = 0; i < n_embd; ++i) ss += (double) layer_in[i] * (double) layer_in[i];
        const float rms = 1.0f / std::sqrt((float) (ss / (double) n_embd) + eps);
        for (int i = 0; i < n_embd; ++i) layer_in[i] = layer_in[i] * rms * norm_w[i];
    }

    return true;
}

}  // namespace strata::core::glm
