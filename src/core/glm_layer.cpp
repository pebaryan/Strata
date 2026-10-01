// GLM-5.3 block chain, stage 1: the hyper-connection front end.  See glm_layer.hpp for the contract.

#include "strata/core/glm_layer.hpp"

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

    // Then the site's own norm.  eps is 1e-5 (attention.layer_norm_rms_epsilon), and the reference is
    // ggml_rms_norm followed by a PLAIN multiply - there is no (1 + w) folding here, which elementwise.hpp
    // states explicitly because some norms in this family do fold.
    kernels::rms_norm_weighted(layer_in, norm_w, /*rows=*/1, /*cols=*/n_embd, eps, stream);

    return true;
}

}  // namespace strata::core::glm
