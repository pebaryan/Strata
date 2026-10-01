// block3_gate - run ONE MLA + routed block through glm_trunk_forward and compare against the oracle.
//
// This is the first execution of the two branches of the trunk loop that have never run: `is_mla` (which decides that
// MLA gets the site norm applied while KDA does not - the double-norm fix) and the ROUTED branch, which streams
// quantized expert blobs through ggml-cpu.  Every other block this port has executed was KDA + dense.
//
// Block 3 is the first MLA layer and it is routed, so both conditions differ from block 0's at once, and the oracle's
// full 45-block run supplies this block's input (l_out-2) and its reference (l_out-3) - so this gate needs no new
// oracle work and its comparison is apples-to-apples by construction.
//
// Everything it uses was validated beforehand rather than discovered here:
//   * the 21 dense tensors come from the artifact by name, and their sizes match the MLA geometry
//     (q_lora 1536, kv_lora 512, n_head 64, head_dim 256) that glm_mla_parity prints;
//   * the expert descriptor is built by native_fmt from the pack's OWN types for this layer, read out of
//     native_experts.txt rather than hardcoded - and native_fmt's bytes matches the pack's declared blob_bytes exactly
//     (7,536,640 for layer 3), so the blobs are read at the right stride;
//   * ExpertSource::blob(layer, expert) returns the blob, which is what glm_stage_moe_native wants.
//
// usage: block3_gate <block-dir> <pack-dir> <oracle-dump-dir>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "strata/core/glm_layer_weights.hpp"
#include "strata/core/glm_trunk.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

namespace {

const int N_EMBD = 4096, HC = 4, LAYER = 3, N_EXPERT = 288, N_USED = 8, FF = 2048;

/// The pack's per-layer expert types, read from native_experts.txt rather than hardcoded: the file is the authority on
/// what format this layer's experts are in, and a hardcoded type would silently misread every blob if the pack changed.
bool pack_types(const std::string& pack_dir, int layer, int& gu_type, int& d_type, int64_t& blob_bytes,
                std::string& err) {
    std::ifstream f(pack_dir + "/native_experts.txt");
    if (!f) { err = "cannot open " + pack_dir + "/native_experts.txt"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        int64_t l = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0;
        int gt = 0, dt = 0;
        std::string shard;
        if (!(is >> l >> gt >> dt >> off >> nb >> go >> uo >> dob)) continue;
        if (l == layer) { gu_type = gt; d_type = dt; blob_bytes = nb; return true; }
    }
    err = "layer " + std::to_string(layer) + " is not in native_experts.txt";
    return false;
}

bool read_floats(const std::string& p, std::vector<float>& out, const char* what) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::printf("  MISSING %s (%s)\n", p.c_str(), what); return false; }
    f.seekg(0, std::ios::end);
    const long n = (long) f.tellg();
    f.seekg(0);
    out.assign((size_t) n / 4, 0.0f);
    f.read((char*) out.data(), n);
    if (!f) { std::printf("  SHORT %s\n", p.c_str()); return false; }
    return true;
}

/// The dump container: five u32 [0, ne0..ne3], then ne0-fastest floats.
bool read_dump(const std::string& p, std::vector<float>& out, int ne[4], const char* what) {
    std::ifstream f(p, std::ios::binary);
    if (!f) { std::printf("  MISSING %s (%s)\n", p.c_str(), what); return false; }
    unsigned hdr[5] = {0, 0, 0, 0, 0};
    f.read((char*) hdr, 20);
    for (int i = 0; i < 4; ++i) ne[i] = (int) hdr[i + 1];
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) (ne[i] > 0 ? ne[i] : 1);
    out.assign(n, 0.0f);
    f.read((char*) out.data(), (std::streamsize) (n * 4));
    if (!f) { std::printf("  SHORT %s\n", p.c_str()); return false; }
    std::printf("  %s ne=[%d,%d,%d,%d] %zu floats\n", what, ne[0], ne[1], ne[2], ne[3], out.size());
    return true;
}

struct Fixture {
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale, attn_norm;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale, ffn_norm;
    std::vector<float> q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b, wo;
    std::vector<float> router, probs_b, shexp_gate, shexp_up, shexp_down;
};

void add(strata::core::GlmBoundBlock& b, const char* name, std::vector<float>& v) {
    strata::core::GlmBoundBlock::Tensor t;
    t.name = name;
    t.ptr = v.data();
    t.ne0 = (int) v.size();
    t.ne1 = 1;
    b.tensors.push_back(t);
}

/// Collects the loop's stage reports for the LAST token run.  Each call with the same name overwrites, so after the
/// token sequence the map holds exactly the stages of the token the comparison is about.
struct Stages {
    std::map<std::string, std::vector<float> > by_name;
};

void stage_cb(void* ctx, int layer, const char* name, const float* data, int n) {
    Stages* st = (Stages*) ctx;
    if (layer != LAYER) return;
    std::vector<float>& v = st->by_name[name];
    v.assign(data, data + (size_t) n);
}

struct Ctx {
    const strata::core::GlmBoundBlock* block = nullptr;
    strata::kernels::glm::KdaGeometry kda_g;
    strata::kernels::glm::MlaGeometry mla_g;
    strata::kernels::glm::MoeGeometry moe_g, shexp_g;
    strata::kernels::glm::KdaWeights kda;
    strata::kernels::glm::MlaWeights mla;
    const float* shexp[3] = {nullptr, nullptr, nullptr};
    strata::kernels::cpu::NativeFmt fmt;
    strata::core::ExpertSource* src = nullptr;
};

const uint8_t* blob_adapter(void* ctx, int layer, int expert) {
    return ((strata::core::ExpertSource*) ctx)->blob(layer, expert);
}

bool provider(void* raw, int layer, strata::core::glm::GlmTrunkLayerWeights& out, std::string& err) {
    Ctx* c = (Ctx*) raw;
    if (!strata::core::glm::glm_fill_layer_weights(*c->block, layer, c->kda_g, c->mla_g, c->kda, c->mla, c->shexp,
                                                   out, err)) {
        return false;
    }
    // the routed half: the geometry, the descriptor, the blobs and the shared expert, none of which the binder supplies
    out.moe_g = &c->moe_g;
    out.shexp_g = &c->shexp_g;
    out.shexp = c->shexp;
    out.moe_fmt = &c->fmt;
    out.blob_fn = &blob_adapter;
    out.blob_ctx = c->src;
    out.shexp_clamp = 10.0f;
    out.clamp_limit = 10.0f;
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: block3_gate <block-dir> <pack-dir> <oracle-dump-dir>\n");
        return 2;
    }
    const std::string bd = argv[1], pack = argv[2], dd = argv[3];

    Fixture fx;
    bool all = true;
    const char* keys[] = {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "attn_norm", "hc_ffn_fn", "hc_ffn_base",
                          "hc_ffn_scale", "ffn_norm", "q_a", "q_a_norm", "q_b", "kv_a", "kv_a_norm", "k_b", "v_b",
                          "wo", "router", "probs_b", "shexp_gate", "shexp_up", "shexp_down"};
    std::vector<float>* slots[] = {&fx.hc_attn_fn, &fx.hc_attn_base, &fx.hc_attn_scale, &fx.attn_norm, &fx.hc_ffn_fn,
                                   &fx.hc_ffn_base, &fx.hc_ffn_scale, &fx.ffn_norm, &fx.q_a,  &fx.q_a_norm, &fx.q_b,
                                   &fx.kv_a, &fx.kv_a_norm, &fx.k_b, &fx.v_b, &fx.wo, &fx.router, &fx.probs_b,
                                   &fx.shexp_gate, &fx.shexp_up, &fx.shexp_down};
    for (size_t i = 0; i < sizeof(keys) / sizeof(keys[0]); ++i) {
        all &= read_floats(bd + "/w_" + keys[i] + ".bin", *slots[i], keys[i]);
    }
    if (!all) { std::fprintf(stderr, "block3_gate: a fixture is missing - refusing to run\n"); return 2; }

    strata::core::GlmBoundBlock block;
    add(block, "hc_attn_fn.weight", fx.hc_attn_fn);
    add(block, "hc_attn_base.weight", fx.hc_attn_base);
    add(block, "hc_attn_scale.weight", fx.hc_attn_scale);
    add(block, "attn_norm.weight", fx.attn_norm);
    add(block, "hc_ffn_fn.weight", fx.hc_ffn_fn);
    add(block, "hc_ffn_base.weight", fx.hc_ffn_base);
    add(block, "hc_ffn_scale.weight", fx.hc_ffn_scale);
    add(block, "ffn_norm.weight", fx.ffn_norm);
    add(block, "attn_q_a.weight", fx.q_a);
    add(block, "attn_q_a_norm.weight", fx.q_a_norm);
    add(block, "attn_q_b.weight", fx.q_b);
    add(block, "attn_kv_a_mqa.weight", fx.kv_a);
    add(block, "attn_kv_a_norm.weight", fx.kv_a_norm);
    add(block, "attn_k_b.weight", fx.k_b);
    add(block, "attn_v_b.weight", fx.v_b);
    add(block, "attn_output.weight", fx.wo);
    add(block, "ffn_gate_inp.weight", fx.router);
    add(block, "exp_probs_b.bias", fx.probs_b);
    add(block, "ffn_gate_shexp.weight", fx.shexp_gate);
    add(block, "ffn_up_shexp.weight", fx.shexp_up);
    add(block, "ffn_down_shexp.weight", fx.shexp_down);

    Ctx ctx;
    ctx.block = &block;
    ctx.kda_g.n_embd = N_EMBD; ctx.kda_g.nh = 64; ctx.kda_g.hd = 128; ctx.kda_g.d_conv = 4;
    // MlaGeometry's defaults are the artifact's own MLA geometry, and the fetched tensor sizes agree with them
    ctx.mla_g.n_embd = N_EMBD;
    ctx.moe_g.n_embd = N_EMBD; ctx.moe_g.n_expert = N_EXPERT; ctx.moe_g.n_used = N_USED; ctx.moe_g.ff = FF;
    // THE CLAMPS MUST COME FROM THE ARTIFACT, and this gate had left them at their defaults - which are 0, which
    // DISABLES them.  The header warns about precisely this: "0 disables it, and that is a real behaviour difference,
    // not a neutral default: with a limit of 10 it is invisible until a pre-activation exceeds 10, so a caller that
    // forgets it will still look right on tame inputs.  Callers must set both from the artifact."  It was not tame
    // here: the oracle's own four-block run counts exp_up 1 and shexp_up 1, so the clamp fires in this block - which
    // is exactly why ffn_out had the right rms and elements off by 4.6%, the signature of a few values moving.
    // swiglu_clamp_exp / swiglu_clamp_shexp are 10.0 for every layer of this model.
    const float CLAMP_EXP = 10.0f;
    ctx.moe_g.clamp_exp = CLAMP_EXP;
    ctx.moe_g.clamp_shexp = CLAMP_EXP;
    ctx.shexp_g.clamp_exp = CLAMP_EXP;
    ctx.shexp_g.clamp_shexp = CLAMP_EXP;
    ctx.shexp_g.n_embd = N_EMBD; ctx.shexp_g.ff = FF;
    ctx.shexp[0] = fx.shexp_gate.data(); ctx.shexp[1] = fx.shexp_up.data(); ctx.shexp[2] = fx.shexp_down.data();

    int gu_type = 0, d_type = 0;
    int64_t blob_bytes = 0;
    std::string err;
    if (!pack_types(pack, LAYER, gu_type, d_type, blob_bytes, err)) {
        std::printf("BLOCK3 GATE: FAIL - %s\n", err.c_str());
        return 1;
    }
    if (!strata::kernels::cpu::native_fmt(gu_type, d_type, N_EMBD, FF, ctx.fmt, err)) {
        std::printf("BLOCK3 GATE: FAIL - native_fmt(%d,%d): %s\n", gu_type, d_type, err.c_str());
        return 1;
    }
    std::printf("  layer %d experts: gu_type %d, d_type %d, pack blob_bytes %lld, descriptor bytes %zu  %s\n", LAYER,
                gu_type, d_type, (long long) blob_bytes, (size_t) ctx.fmt.bytes,
                (int64_t) ctx.fmt.bytes == blob_bytes ? "(MATCH)" : "(MISMATCH)");

    strata::core::FileExpertSource src;
    // THE LAYOUT MUST BE LOADED BEFORE THE SOURCE IS OPENED.  open() validates against the process-wide
    // expert_layout(), which is the canonical Q2_0 default until expert_layout_load reads the pack's
    // native_experts.txt - so calling open() first compares against a default and fails with "the requested geometry
    // does not match the loaded expert layout", which is exactly what happened here.  The load takes (hidden,
    // expert_ffn) = (4096, 2048) for GLM-5.3-Flash; with the qwen4exp defaults (2560/640) the block-size check refuses
    // the layer rather than decoding it wrongly, which the header calls "a refusal to get right, not a guess".
    // THE PACK'S LAYER NUMBERING AND THE LAYOUT LOADER DISAGREE, AND THIS IS THE OPEN QUESTION, not a parameter to
    // guess.  The pack's native_experts.txt carries rows for layers 3..45 with ABSOLUTE layer numbers (GLM's first
    // three blocks are dense and have no experts), while the loader wants a dense 0..n_layers-1 set:
    //   * with n_layers = 46 it parses, then refuses with "the native expert layout is invalid at layer 0" - layers
    //     0, 1, 2 legitimately have no entry;
    //   * with n_layers = 43 it calls the row for layer 43 "malformed", because that row's layer number is outside the
    //     range a 43-layer layout declares.
    // Both refusals are correct given their premise, so this is a decision about how the pack represents a layer with
    // NO experts, not a number to try until something runs.  Passing 43 would be the tempting shortcut and it is the
    // dangerous one: the layout would index row order rather than layer number, so layer 3 would be served layer 6's
    // experts - plausible numbers, silently wrong, and exactly the failure this port has spent the session learning to
    // refuse.  46 is used below because it is the value that reaches the real blocker rather than a parse error.
    if (!strata::kernels::cpu::expert_layout_load(pack, 46, N_EXPERT, err, N_EMBD, FF)) {
        std::printf("BLOCK3 GATE: FAIL - expert_layout_load: %s\n", err.c_str());
        return 1;
    }
    const auto& lay = strata::kernels::cpu::expert_layout();
    std::printf("  loaded expert layout: n_layers %lld, n_expert %lld\n", (long long) lay.n_layers,
                (long long) lay.n_expert);
    if (!src.open(pack, lay.n_layers, lay.n_expert, err)) {
        std::printf("BLOCK3 GATE: FAIL - FileExpertSource::open: %s\n", err.c_str());
        return 1;
    }
    ctx.src = &src;

    // block 3's input and the oracle's expectation for it, both from the full 45-block run
    std::vector<float> x_all, want_all;
    int ne_x[4] = {0, 0, 0, 0}, ne_w[4] = {0, 0, 0, 0};
    if (!read_dump(dd + "/l_out-2.bin", x_all, ne_x, "l_out-2 (block 3's input)")) return 2;
    if (!read_dump(dd + "/l_out-3.bin", want_all, ne_w, "l_out-3 (the oracle's block 3)")) return 2;
    const int tokens = (int) (x_all.size() / ((size_t) N_EMBD * HC));
    if (tokens < 1 || want_all.size() < (size_t) N_EMBD * HC) {
        std::printf("BLOCK3 GATE: FAIL - the dump is too small\n");
        return 2;
    }
    const float* want = want_all.data() + (size_t) (tokens - 1) * N_EMBD * HC;
    std::printf("  %d token(s) in the dump; running ALL of them in order through block %d (MLA + routed),\n", tokens,
                LAYER);
    std::printf("  threading the MLA cache, and comparing the LAST token - because that is the sequence the oracle\n");
    std::printf("  ran: its token %d attends to the latents of tokens 0..%d.  Feeding only the last token to a fresh\n",
                tokens - 1, tokens - 1);
    std::printf("  cache is what made the first attempt fail with a 0.998-of-rms error at a 0.978 rms ratio.\n");

    // one MLA cache with room for this token; the loop writes the latent into slot `cells` then attends with cells+1
    std::vector<float> cache((size_t) 8 * ctx.mla_g.kv_lora, 0.0f);
    float* cache_ptrs[1] = {cache.data()};
    int mla_len[16] = {0};
    int mla_index[16];
    for (int i = 0; i < 16; ++i) mla_index[i] = (i == LAYER) ? 0 : -1;

    strata::core::glm::GlmTrunkState st;
    st.kda_state = nullptr;
    st.mla_cache = cache_ptrs;
    st.mla_len = mla_len;
    st.kda_index = mla_index;   // never consulted for an MLA layer
    st.mla_index = mla_index;

    Stages stages;
    std::vector<float> out((size_t) HC * N_EMBD, 0.0f);
    for (int t = 0; t < tokens; ++t) {
        const float* x = x_all.data() + (size_t) t * N_EMBD * HC;
        if (!strata::core::glm::glm_trunk_forward(x, 1, provider, &ctx, ctx.kda_g, ctx.mla_g, 1e-5f, st, out.data(),
                                                  nullptr, err, LAYER, &stage_cb, &stages)) {
            std::printf("BLOCK3 GATE: FAIL - the loop returned false at token %d: %s\n", t, err.c_str());
            return 1;
        }
        std::printf("    token %d done; MLA cache now holds %d latent(s)\n", t, mla_len[0]);
    }

    // ---- the bisection: which half of the block is wrong?
    //
    // The oracle's stages come from the SAME runner that produces the token gate, so comparing against them is not
    // circular.  Each file holds every token; the token axis is ne[2] when it is not 1 (the hc stages, one block of
    // streams per token) and ne[1] otherwise (the attention and FFN stages, one row per token) - the same container
    // rule the dump format uses, applied to pick THIS token out of it.
    if (argc >= 5 && !stages.by_name.empty()) {
        const std::string sd = argv[4];
        const char* names[] = {"attn_output", "hc_attn_post", "ffn_norm", "ffn_out"};
        std::printf("  --- stage-by-stage, token %d ---\n", tokens - 1);
        for (size_t k = 0; k < sizeof(names) / sizeof(names[0]); ++k) {
            std::map<std::string, std::vector<float> >::const_iterator it = stages.by_name.find(names[k]);
            if (it == stages.by_name.end()) {
                std::printf("    %-13s the loop reported no such stage\n", names[k]);
                continue;
            }
            const std::vector<float>& got = it->second;
            std::vector<float> file;
            int ne[4] = {0, 0, 0, 0};
            const std::string path = sd + "/" + names[k] + "-" + std::to_string(LAYER) + ".bin";
            if (!read_dump(path, file, ne, names[k])) {
                std::printf("    %-13s no reference\n", names[k]);
                continue;
            }
            size_t off = 0, len = 0;
            if (ne[2] == 1) {
                off = (size_t) ne[0] * (size_t) (tokens - 1);
                len = (size_t) ne[0];
            } else {
                off = (size_t) ne[0] * (size_t) ne[1] * (size_t) (tokens - 1);
                len = (size_t) ne[0] * (size_t) ne[1];
            }
            if (len != got.size() || off + len > file.size()) {
                std::printf("    %-13s extent mismatch: engine %zu, reference slice %zu (file %zu)\n", names[k],
                            got.size(), len, file.size());
                continue;
            }
            double worst_s = 0.0, ss = 0.0, ws = 0.0;
            for (size_t j = 0; j < len; ++j) {
                worst_s = std::max(worst_s, std::fabs((double) got[j] - (double) file[off + j]));
                ss += (double) got[j] * got[j];
                ws += (double) file[off + j] * file[off + j];
            }
            const double gr = std::sqrt(ss / (double) len), wr2 = std::sqrt(ws / (double) len);
            std::printf("    %-13s engine rms %.6g  oracle rms %.6g  worst %.6g  = %.4g of rms\n", names[k], gr,
                        wr2, worst_s, worst_s / (wr2 > 0.0 ? wr2 : 1.0));
        }
    }

    // ---- is the engine routing to the same experts?
    //
    // The ids the stage ACTUALLY used come out on the stream; the top-8 is then computed HERE, independently, from the
    // same input (ffn_norm, itself verified against the oracle) with the same fixture weights.  Two separate
    // implementations of the reference's rule agreeing is the point - and if they disagree, the mixture has the right
    // magnitude from the wrong experts, which is exactly what ffn_out's matching-rms-with-4.6%-error looks like.
    if (argc >= 5) {
        std::map<std::string, std::vector<float> >::const_iterator it = stages.by_name.find("moe_ids");
        std::map<std::string, std::vector<float> >::const_iterator ix = stages.by_name.find("ffn_norm");
        if (it != stages.by_name.end() && ix != stages.by_name.end() && ix->second.size() == (size_t) N_EMBD) {
            std::printf("    engine ids   :");
            for (size_t k = 0; k < it->second.size(); ++k) std::printf(" %d", (int) (it->second[k] + 0.5f));
            std::printf("\n");
            std::vector<float> sel((size_t) N_EXPERT, 0.0f);
            for (int e = 0; e < N_EXPERT; ++e) {
                double s = 0.0;
                for (int j = 0; j < N_EMBD; ++j) {
                    s += (double) fx.router[(size_t) e * N_EMBD + (size_t) j] * (double) ix->second[(size_t) j];
                }
                sel[(size_t) e] = (float) (1.0 / (1.0 + std::exp(-s)) + (double) fx.probs_b[(size_t) e]);
            }
            std::printf("    top-8 by hand:");
            for (int k = 0; k < N_USED; ++k) {
                int best = -1;
                float bv = -1e30f;
                for (int e = 0; e < N_EXPERT; ++e) {
                    if (sel[(size_t) e] > bv) { bv = sel[(size_t) e]; best = e; }
                }
                std::printf(" %d", best);
                sel[(size_t) best] = -1e30f;
            }
            std::printf("\n");
        }
    }

    double worst = 0.0, ssum = 0.0, wsum = 0.0;
    bool finite = true;
    for (size_t i = 0; i < out.size(); ++i) {
        if (!std::isfinite(out[i])) finite = false;
        worst = std::max(worst, std::fabs((double) out[i] - (double) want[i]));
        ssum += (double) out[i] * out[i];
        wsum += (double) want[i] * want[i];
    }
    const double er = std::sqrt(ssum / out.size()), wr = std::sqrt(wsum / out.size());
    std::printf("  engine l_out: rms %.9g\n", er);
    std::printf("  oracle l_out: rms %.9g\n", wr);
    std::printf("  worst absolute difference %.6g   relative to rms %.6g   ratio %.9g\n", worst,
                worst / (wr ? wr : 1.0), er / (wr ? wr : 1.0));
    if (!finite) { std::printf("BLOCK3 GATE: FAIL - not finite\n"); return 1; }
    // a KDA block agrees to ~1e-5 of rms; this is a 2048-wide SwiGLU over 8 streamed experts plus a shared expert, so
    // the bar is set at 1e-3 of rms and anything at the fp32 floor passes comfortably
    const bool pass = worst < 1e-3 * (wr ? wr : 1.0);
    std::printf("BLOCK3 GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
