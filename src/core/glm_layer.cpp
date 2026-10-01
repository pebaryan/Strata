// GLM-5.3 block chain, stage 1: the hyper-connection front end.  See glm_layer.hpp for the contract.

#include "strata/core/glm_layer.hpp"

#include <cmath>
#include <vector>

#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_moe.hpp"

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

/// Stage 3: the site's hc_post.  The hyper-connection block is not a plain residual: the site's output and the
/// incoming streams are combined through the mix that hc_pre recorded, and the result has HC rows again - which is
/// what the NEXT site's hc_pre consumes (`x` is [HC][n_embd]).  So the residual here is the block's own input, and
/// the output is the next site's input rather than a single vector.
///
/// No arithmetic: hc_post is verified (2.551e-07 against the oracle, the tightest agreement in the port).
bool glm_stage_hc_post(const float* site_out, const float* residual, const kernels::glm::HcMix& mix,
                       int n_embd, float* out, std::string& err) {
    if (!site_out || !residual || !out) {
        err = "glm_stage_hc_post: null argument";
        return false;
    }
    if (n_embd <= 0) {
        err = "glm_stage_hc_post: n_embd must be positive";
        return false;
    }
    kernels::glm::hc_post(site_out, residual, mix, n_embd, out);
    return true;
}

/// Stage 4: the FFN site's feed-forward.  One call to expert_ffn, which serves both a routed expert and a leading
/// dense block - the difference being the geometry's ff (12288 for a dense block, 2048 for a routed expert) and the
/// clamp limit, which is an explicit argument rather than a field because the engine's own header warns that with a
/// limit of 10 the clamp is invisible until a pre-activation exceeds 10, so a caller who forgets it passes most
/// tests.  The leading dense FFN uses swiglu_clamp_shexp, not the routed experts' clamp_exp; both are 10.0 here.
///
/// No arithmetic: expert_ffn is verified on block 0 at the fp32 floor with its clamp proven by a demonstrated
/// failure, and matches on block 1 at the same floor.
bool glm_stage_ffn(const float* xn, const float* wg, const float* wu, const float* wd,
                   const kernels::glm::MoeGeometry& g, float* out, float clamp_limit, std::string& err) {
    if (!xn || !wg || !wu || !wd || !out) {
        err = "glm_stage_ffn: null argument";
        return false;
    }
    if (g.n_embd <= 0 || g.ff <= 0) {
        err = "glm_stage_ffn: geometry must have positive n_embd and ff";
        return false;
    }
    kernels::glm::expert_ffn(wg, wu, wd, g, xn, out, clamp_limit);
    return true;
}

/// One whole block, in the order the reference uses and the dump's per-block stage names confirm:
///
///     hc_pre -> attn_norm -> attention -> hc_post -> hc_pre -> ffn_norm -> FFN -> hc_post
///
/// The shape alternates, and that is the part worth being careful about because both are float*: the attention and
/// FFN outputs are one vector per token ([n_embd]), while everything between the sites is HC rows ([HC][n_embd]).
/// hc_post is what converts back, and hc_pre is what consumes it.
bool glm_block_forward(const float* x, int tokens, const GlmBlockWeights& w, float hc_rms_eps,
                       float* l_out, float* state, void* stream, std::string& err) {
    if (!x || !l_out || !state || !w.hc_attn_fn || !w.hc_attn_base || !w.hc_attn_scale || !w.attn_norm ||
        !w.kda || !w.kda_geom || !w.hc_ffn_fn || !w.hc_ffn_base || !w.hc_ffn_scale || !w.ffn_norm ||
        !w.ffn_gate || !w.ffn_up || !w.ffn_down || !w.ffn_geom) {
        err = "glm_block_forward: null argument (every weight pointer must be set)";
        return false;
    }
    const int ne = w.kda_geom->n_embd;
    if (tokens <= 0 || ne <= 0) {
        err = "glm_block_forward: tokens and n_embd must be positive";
        return false;
    }
    const size_t hc = (size_t) kernels::glm::HC;

    // 1-3: the attention site.  hc_pre mixes the streams down and records its mix; the norm is applied in place, so
    // what comes back is attn_norm-N and the attention takes it directly; hc_post recombines using the same mix.
    std::vector<float> layer_in((size_t) tokens * ne), attn_out((size_t) tokens * ne);
    std::vector<float> mid((size_t) tokens * hc * ne);
    std::vector<kernels::glm::HcMix> mix_attn((size_t) tokens);
    for (int t = 0; t < tokens; ++t) {
        if (!glm_stage_hc_norm(x + (size_t) t * hc * ne, ne, w.hc_attn_fn, w.hc_attn_base, w.hc_attn_scale,
                               w.attn_norm, layer_in.data() + (size_t) t * ne, &mix_attn[(size_t) t],
                               hc_rms_eps, stream, err)) {
            err = "glm_block_forward: attention site: " + err;
            return false;
        }
        if (!glm_stage_kda(layer_in.data() + (size_t) t * ne, *w.kda, *w.kda_geom, 1,
                           attn_out.data() + (size_t) t * ne, state, err)) {
            err = "glm_block_forward: attention call: " + err;
            return false;
        }
        if (!glm_stage_hc_post(attn_out.data() + (size_t) t * ne, x + (size_t) t * hc * ne,
                               mix_attn[(size_t) t], ne, mid.data() + (size_t) t * hc * ne, err)) {
            err = "glm_block_forward: attention site hc_post: " + err;
            return false;
        }
    }

    // 4-6: the FFN site, the same shape of sequence against the FFN's own weights.
    std::vector<float> ffn_in((size_t) tokens * ne), ffn_out((size_t) tokens * ne);
    std::vector<kernels::glm::HcMix> mix_ffn((size_t) tokens);
    kernels::glm::MoeGeometry fg = *w.ffn_geom;
    fg.n_embd = ne;
    for (int t = 0; t < tokens; ++t) {
        if (!glm_stage_hc_norm(mid.data() + (size_t) t * hc * ne, ne, w.hc_ffn_fn, w.hc_ffn_base, w.hc_ffn_scale,
                               w.ffn_norm, ffn_in.data() + (size_t) t * ne, &mix_ffn[(size_t) t],
                               hc_rms_eps, stream, err)) {
            err = "glm_block_forward: FFN site: " + err;
            return false;
        }
        if (!glm_stage_ffn(ffn_in.data() + (size_t) t * ne, w.ffn_gate, w.ffn_up, w.ffn_down, fg,
                           ffn_out.data() + (size_t) t * ne, w.clamp_limit, err)) {
            err = "glm_block_forward: FFN call: " + err;
            return false;
        }
        if (!glm_stage_hc_post(ffn_out.data() + (size_t) t * ne, mid.data() + (size_t) t * hc * ne,
                               mix_ffn[(size_t) t], ne, l_out + (size_t) t * hc * ne, err)) {
            err = "glm_block_forward: FFN site hc_post: " + err;
            return false;
        }
    }
    return true;
}

}  // namespace strata::core::glm
