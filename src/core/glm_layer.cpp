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

/// Stage 3b: the MLA site, for the blocks whose attention is latent rather than KDA (the artifact declares the
/// kind per layer in glm5next.attention.head_count_kv: 1 = MLA, 0 = KDA).  `cache` is [n_cache][kv_lora] of
/// latents with the current position already in it, because the reference appends before attending.
///
/// The indexer's selection is all-tokens below 8192 positions, which is where this model's prompts sit; past that
/// a caller must apply the indexer's selection before calling.  Stated rather than assumed, since the reference
/// makes the same statement.
bool glm_stage_mla(const kernels::glm::MlaWeights& w, const kernels::glm::MlaGeometry& g, const float* x,
                   int n_cache, const float* cache, float* out, std::string& err) {
    if (!x || !cache || !out) {
        err = "glm_stage_mla: null argument";
        return false;
    }
    if (g.n_embd <= 0 || g.n_head <= 0 || n_cache <= 0) {
        err = "glm_stage_mla: geometry and n_cache must be positive";
        return false;
    }
    kernels::glm::mla_forward(w, g, x, n_cache, cache, out);
    return true;
}

/// Stage 4b: the MoE site, for the blocks whose FFN is routed (43 of this model's 46).  moe_forward does the
/// whole site - route, weighted sum, then the UNWEIGHTED shared expert - so this stage adds no arithmetic.
///
/// It takes POINTERS to the expert weights rather than owning storage, deliberately: the experts are streamed
/// from the 86 GB pack through native_experts.txt rather than held resident, so residency is the caller's
/// business and this stage stays a call.  `experts[i]` is the i-th selected expert's {gate, up, down} in the
/// order the router returns; `shared` is one {gate, up, down} triple.
bool glm_stage_moe(const float* xn, const float* router, const float* probs_b,
                   const kernels::glm::MoeGeometry& g, const float* const* const* experts,
                   const float* const* shared, float* out, std::string& err) {
    if (!xn || !router || !experts || !shared || !out) {
        err = "glm_stage_moe: null argument";
        return false;
    }
    if (g.n_embd <= 0 || g.ff <= 0 || g.n_expert <= 0) {
        err = "glm_stage_moe: geometry must be positive";
        return false;
    }
    kernels::glm::moe_forward(router, probs_b, g, xn, experts, shared, out);
    return true;
}


/// Stage 5: the head's first half.  The trunk's output is HC rows, so the head begins by AVERAGING them: the model
/// reads four hyper-connection streams and the reference consumes their mean, per its own line
/// `cur = HC.hc_mean(np.transpose(inpL, (1, 0, 2)))`, followed by `rms_norm_ne0(cur, output_norm.weight)`.
///
/// The averaging lives HERE rather than with the caller because it is the one step between the trunk and the
/// projection that is easy to omit - it is a single line in the reference, it produces no shape change, and a head
/// that skipped it would project one stream instead of their mean and emit plausible logits from the wrong vector.
/// The projection itself is deliberately NOT here: it is a 154,880 x 4,096 matvec against a dequantized tensor, and
/// keeping it out means this stage can be verified against the oracle's saved hidden state without it.
///
/// `hc` is the trunk output in the engine's layout, [hc_streams][n_embd], stream-major within the token.
bool glm_stage_head_mean_norm(const float* hc, int hc_streams, int n_embd, const float* output_norm, float* hidden,
                              std::string& err) {
    if (!hc || !hidden || !output_norm) {
        err = "glm_stage_head_mean_norm: null argument";
        return false;
    }
    if (hc_streams <= 0 || n_embd <= 0) {
        err = "glm_stage_head_mean_norm: hc_streams and n_embd must be positive";
        return false;
    }
    const float inv = 1.0f / (float) hc_streams;
    for (int e = 0; e < n_embd; ++e) {
        float acc = 0.0f;
        for (int s = 0; s < hc_streams; ++s) acc += hc[(size_t) s * n_embd + e];
        hidden[e] = acc * inv;
    }
    // the same eps the rest of this port uses for the layer norms (attention.layer_norm_rms_epsilon)
    double ss = 0.0;
    for (int e = 0; e < n_embd; ++e) ss += (double) hidden[e] * hidden[e];
    const float r = 1.0f / std::sqrt((float) (ss / (double) n_embd) + 1e-5f);
    for (int e = 0; e < n_embd; ++e) hidden[e] = hidden[e] * r * output_norm[e];
    return true;
}


/// Stage 5 (second half): the tied output projection and the greedy token.
///
/// logits[v] = dot(W row v, hidden), then argmax - the reference's own sequence (glm5_full_run.py:168-180) chunked
/// over rows.  `W` is [vocab][n_embd] row-major and ALREADY DEQUANTIZED, which is a deliberate exception: this is the
/// last operation in the model, the artifact's output.weight is 154,880 x 4,096 = 634,388,480 values (2.54 GB in
/// fp32), and streaming it through the quantized-blob path would buy nothing at inference time - the whole matrix is
/// touched exactly once per token either way.
///
/// The accumulation is in DOUBLE, matching the reference, because the argmax is a comparison over 154,880 sums of
/// 4,096 terms: a float accumulator is enough to move the winner when two logits are close, and the token is the
/// entire observable output of the model.
bool glm_stage_head_project(const float* W, int vocab, int n_embd, const float* hidden, int& argmax, float& best,
                            std::string& err) {
    if (!W || !hidden) {
        err = "glm_stage_head_project: null argument";
        return false;
    }
    if (vocab <= 0 || n_embd <= 0) {
        err = "glm_stage_head_project: vocab and n_embd must be positive";
        return false;
    }
    argmax = -1;
    best = 0.0f;
    double best_s = -1e300;
    bool any = false;
    for (int v = 0; v < vocab; ++v) {
        const float* row = W + (size_t) v * (size_t) n_embd;
        double s = 0.0;
        for (int e = 0; e < n_embd; ++e) s += (double) row[e] * (double) hidden[e];
        if (!any || s > best_s) {
            best_s = s;
            argmax = v;
            any = true;
        }
    }
    if (!any) {
        err = "glm_stage_head_project: no rows";
        return false;
    }
    best = (float) best_s;
    return true;
}

}  // namespace strata::core::glm
