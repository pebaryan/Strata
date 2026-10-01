// GLM-5.3 block chain, stage 1: the hyper-connection front end.  See glm_layer.hpp for the contract.

#include "strata/core/glm_layer.hpp"

#include <cmath>

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm_kda.hpp"

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

/// Stage 2: the attention call.  Stage 1 has already produced the site's input, and for a KDA block that input is
/// exactly what kda_forward consumes - the dump's attn_norm-N is the same quantity the gate passes as x.  So this
/// stage adds no arithmetic: it calls the kernel with the block's bound weights and geometry.
///
/// `state` is the recurrent S matrix, nh*hd*hd floats, and MUST be zeroed for a fresh sequence.  The gate fixtures
/// are all T=1 or a fresh T=5, so nothing here carries state across calls yet; a real sequential decode would.
bool glm_stage_kda(const float* xn, const kernels::glm::KdaWeights& w, const kernels::glm::KdaGeometry& g,
                   int tokens, float* out, float* state, std::string& err) {
    if (!xn || !out || !state) {
        err = "glm_stage_kda: null argument";
        return false;
    }
    if (tokens <= 0 || g.n_embd <= 0) {
        err = "glm_stage_kda: tokens and n_embd must be positive";
        return false;
    }
    kernels::glm::kda_forward(w, g, xn, tokens, out, state, nullptr);
    return true;
}

}  // namespace strata::core::glm
