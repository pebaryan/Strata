/// Maps one bound block's tensors into the weight structs the trunk loop wants.
///
/// It takes an ALREADY-BOUND block rather than the artifact: bind_glm_block already walks every row of a block
/// generically, and GlmBoundBlock::find(name) already looks any of them up.  Doing the mapping over that means this
/// function needs no pack, no LayerView and no shards - a gate can hand it a block assembled from fixture arrays and
/// check the assignment, which is the property every other gate in this port has and the reason they can run at all.
///
/// It is deliberately PURE ASSIGNMENT: no arithmetic, no dequantization, no layout fixups.  Those belong to the
/// binder (which already repacks ssm_conv1d and dequantizes what needs it), and duplicating any of them here would
/// create a second place for the two to disagree.
///
/// What it does decide is WHICH TENSOR IS WHICH FIELD, and it refuses by NAME when one is absent - because a null
/// weight reaching the trunk is a wiring bug, and the trunk's own check can only say "a required weight is null"
/// rather than which one.
#include "strata/core/glm_layer_weights.hpp"

#include "strata/core/glm_bind.hpp"

namespace strata::core::glm {
namespace {

/// One report per process: the indexer's presence is a per-layer fact with a single explanation.
static bool g_indexer_reported = false;

/// The tensor names, in one place, spelled exactly as the generated table spells them.  Two of them do not follow the
/// `name.weight` pattern and are the ones that have cost this port time before: `ssm_a` carries no suffix, and
/// `ssm_dt.bias` is a bias.
struct Names {
    // shared by every trunk layer
    static constexpr const char* hc_attn_fn = "hc_attn_fn.weight";
    static constexpr const char* hc_attn_base = "hc_attn_base.weight";
    static constexpr const char* hc_attn_scale = "hc_attn_scale.weight";
    static constexpr const char* attn_norm = "attn_norm.weight";
    static constexpr const char* hc_ffn_fn = "hc_ffn_fn.weight";
    static constexpr const char* hc_ffn_base = "hc_ffn_base.weight";
    static constexpr const char* hc_ffn_scale = "hc_ffn_scale.weight";
    static constexpr const char* ffn_norm = "ffn_norm.weight";
    static constexpr const char* attn_output = "attn_output.weight";
    // KDA
    static constexpr const char* kda_q = "attn_q.weight";
    static constexpr const char* kda_k = "attn_k.weight";
    static constexpr const char* kda_v = "attn_v.weight";
    static constexpr const char* conv_q = "ssm_conv1d_q.weight";
    static constexpr const char* conv_k = "ssm_conv1d_k.weight";
    static constexpr const char* conv_v = "ssm_conv1d_v.weight";
    static constexpr const char* ssm_a = "ssm_a";
    static constexpr const char* dt_bias = "ssm_dt.bias";
    static constexpr const char* f_a = "ssm_f_a.weight";
    static constexpr const char* f_b = "ssm_f_b.weight";
    static constexpr const char* beta = "ssm_beta.weight";
    static constexpr const char* g_a = "ssm_g_a.weight";
    static constexpr const char* g_b = "ssm_g_b.weight";
    static constexpr const char* o_norm = "ssm_norm.weight";
    // MLA
    static constexpr const char* q_a = "attn_q_a.weight";
    static constexpr const char* q_a_norm = "attn_q_a_norm.weight";
    static constexpr const char* q_b = "attn_q_b.weight";
    static constexpr const char* kv_a = "attn_kv_a_mqa.weight";
    static constexpr const char* kv_a_norm = "attn_kv_a_norm.weight";
    static constexpr const char* k_b = "attn_k_b.weight";
    static constexpr const char* v_b = "attn_v_b.weight";
    // FFN
    static constexpr const char* dense_gate = "ffn_gate.weight";
    static constexpr const char* dense_up = "ffn_up.weight";
    static constexpr const char* dense_down = "ffn_down.weight";
    static constexpr const char* router = "ffn_gate_inp.weight";
    static constexpr const char* probs_b = "exp_probs_b.bias";
    static constexpr const char* shexp_gate = "ffn_gate_shexp.weight";
    static constexpr const char* shexp_up = "ffn_up_shexp.weight";
    static constexpr const char* shexp_down = "ffn_down_shexp.weight";
};

/// One lookup that reports the NAME it could not find, so a failure says which tensor rather than "a weight is null".
bool want(const GlmBoundBlock& b, const char* name, const float*& out, std::string& err) {
    const float* p = b.find(name);
    if (!p) {
        err = std::string("no tensor ") + name + " in the bound block";
        return false;
    }
    out = p;
    return true;
}

const float* opt(const GlmBoundBlock& b, const char* name) { return b.find(name); }

}  // namespace

bool glm_fill_layer_weights(const GlmBoundBlock& b, int layer, const kernels::glm::KdaGeometry& kda_g,
                            const kernels::glm::MlaGeometry& mla_g, kernels::glm::KdaWeights& kda_storage,
                            kernels::glm::MlaWeights& mla_storage, const float** shexp_storage,
                            GlmTrunkLayerWeights& out, std::string& err) {
    (void) kda_g;
    (void) mla_g;
    const int is_mla = glm_attention_is_mla(layer);
    if (is_mla < 0) {
        err = "glm_fill_layer_weights: layer " + std::to_string(layer) + " is not a trunk layer";
        return false;
    }
    const bool dense = glm_ffn_is_dense(layer);

    // ---- the eight tensors every trunk layer has ----
    if (!want(b, Names::hc_attn_fn, out.hc_attn_fn, err) ||
        !want(b, Names::hc_attn_base, out.hc_attn_base, err) ||
        !want(b, Names::hc_attn_scale, out.hc_attn_scale, err) ||
        !want(b, Names::attn_norm, out.attn_norm, err) ||
        !want(b, Names::hc_ffn_fn, out.hc_ffn_fn, err) ||
        !want(b, Names::hc_ffn_base, out.hc_ffn_base, err) ||
        !want(b, Names::hc_ffn_scale, out.hc_ffn_scale, err) ||
        !want(b, Names::ffn_norm, out.ffn_norm, err)) {
        err = "glm_fill_layer_weights: layer " + std::to_string(layer) + ": " + err;
        return false;
    }

    // ---- the attention: exactly one kind, and the struct is filled or left null accordingly ----
    if (is_mla == 1) {
        kernels::glm::MlaWeights& m = mla_storage;
        if (!want(b, Names::q_a, m.wq_a, err) || !want(b, Names::q_a_norm, m.q_a_norm, err) ||
            !want(b, Names::q_b, m.wq_b, err) || !want(b, Names::kv_a, m.kv_a, err) ||
            !want(b, Names::kv_a_norm, m.kv_a_norm, err) || !want(b, Names::k_b, m.wk_b, err) ||
            !want(b, Names::v_b, m.wv_b, err) || !want(b, Names::attn_output, m.wo, err)) {
            err = "glm_fill_layer_weights: layer " + std::to_string(layer) + " (MLA): " + err;
            return false;
        }
        out.mla = &mla_storage;
        out.kda = nullptr;
        // The indexer's tensors (indexer.proj, indexer.k_norm.*, indexer_compressor_*, indexer.attn_{q_b,k}) are optional
        // here - a fixture may omit them - and bound when the pack has them.  The model reads only the best 512 four-cell
        // pools plus the tail once a context passes ~2051 positions (top_k 2048 CELLS, kpool 4: confirmed from the GGUF
        // metadata and the reference's build_dsa_top_k); the device MLA blocks apply it.  Anything that cannot (the host
        // fallback path) is dense beyond that point and says so (mla_forward).  The loader's old claim that selection is
        // all-tokens "below 8192 positions" was wrong: it holds only below ~2051.
        mla_storage.idx_attn_k = opt(b, "indexer.attn_k.weight");
        mla_storage.idx_attn_q_b = opt(b, "indexer.attn_q_b.weight");
        mla_storage.idx_c_gate = opt(b, "indexer_compressor_gate.weight");
        mla_storage.idx_k_norm_w = opt(b, "indexer.k_norm.weight");
        mla_storage.idx_k_norm_b = opt(b, "indexer.k_norm.bias");
        mla_storage.idx_proj = opt(b, "indexer.proj.weight");
        mla_storage.idx_ape = opt(b, "indexer_compressor_ape.weight");
        (void) g_indexer_reported;
    } else {
        kernels::glm::KdaWeights& k = kda_storage;
        if (!want(b, Names::kda_q, k.wq, err) || !want(b, Names::kda_k, k.wk, err) ||
            !want(b, Names::kda_v, k.wv, err) || !want(b, Names::conv_q, k.conv_q, err) ||
            !want(b, Names::conv_k, k.conv_k, err) || !want(b, Names::conv_v, k.conv_v, err) ||
            !want(b, Names::ssm_a, k.ssm_a, err) || !want(b, Names::dt_bias, k.dt_bias, err) ||
            !want(b, Names::f_a, k.ssm_f_a, err) || !want(b, Names::f_b, k.ssm_f_b, err) ||
            !want(b, Names::beta, k.ssm_beta, err) || !want(b, Names::g_a, k.ssm_g_a, err) ||
            !want(b, Names::g_b, k.ssm_g_b, err) || !want(b, Names::o_norm, k.o_norm, err) ||
            !want(b, Names::attn_output, k.wo, err)) {
            err = "glm_fill_layer_weights: layer " + std::to_string(layer) + " (KDA): " + err;
            return false;
        }
        k.attn_norm = out.attn_norm;   // the KDA kernel norms its own input with the layer's attn_norm
        out.kda = &kda_storage;
        out.mla = nullptr;
    }

    // ---- the feed-forward ----
    if (dense) {
        if (!want(b, Names::dense_gate, out.ffn_gate, err) || !want(b, Names::dense_up, out.ffn_up, err) ||
            !want(b, Names::dense_down, out.ffn_down, err)) {
            err = "glm_fill_layer_weights: layer " + std::to_string(layer) + " (dense FFN): " + err;
            return false;
        }
        out.moe_router = nullptr;
        out.shexp = nullptr;
    } else {
        if (!want(b, Names::router, out.moe_router, err) || !want(b, Names::probs_b, out.moe_probs_b, err)) {
            err = "glm_fill_layer_weights: layer " + std::to_string(layer) + " (routed FFN): " + err;
            return false;
        }
        if (shexp_storage) {
            const float* g = opt(b, Names::shexp_gate);
            const float* u = opt(b, Names::shexp_up);
            const float* d = opt(b, Names::shexp_down);
            if (g && u && d) {
                shexp_storage[0] = g;
                shexp_storage[1] = u;
                shexp_storage[2] = d;
                out.shexp = shexp_storage;
            }
        }
        // moe_g / shexp_g / moe_fmt / blob_fn are the caller's: the expert width, the native descriptor and the blob
        // source all come from the pack rather than from the bound block, so this function leaves them untouched.
    }
    return true;
}

}  // namespace strata::core::glm
