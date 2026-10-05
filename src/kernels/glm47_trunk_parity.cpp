// src/kernels/glm47_trunk_parity.cpp - does our WHOLE GLM-4.7-Flash trunk match the reference?
//
// The oracle is tools/glm47_trunk_reference.py.  The loop itself is engine code now
// (strata/core/glm47_trunk.cpp + glm47_trunk.hpp); this gate drives THAT function, so it tests the shipped
// trunk rather than a copy of it:
//
//     x = hidden state
//     for l in 0..L-1:  x = block_l(x, cache_l)     # block 0 dense, the rest MoE; EACH LAYER ITS OWN CACHE
//     hn = rms_norm(x, output_norm, 1e-5)
//     logits = W_out @ hn ;  token = argmax(logits)
//
// It compares, per token and per layer, the hidden state (so a mismatch names the block that made it), the
// top-4 expert ids of every MoE block (EXACT), the head's normed hidden state, the logits and the argmax.
// The failure this exists to catch is the cache: a loop that reuses one cache for every layer hands layer l
// the keys layer l-1 wrote.  The teeth run drives exactly that through the engine (cache_index all-zero) and
// requires the hidden state to move.
//
//   python tools/glm47_trunk_reference.py --raw-fixture D:/tmp/glm47_trunk.bin [--layers 2] [--tokens 2]
//   build/glm47_trunk_parity.exe D:/tmp/glm47_trunk.bin
#include "strata/core/glm47_trunk.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/glm_norm.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace glm = strata::kernels::glm;
namespace cglm = strata::core::glm;

namespace {

constexpr uint32_t MAGIC = 0x47543437u;   // 'GT47'

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

void read_into(std::FILE* f, void* dst, size_t bytes) {
    if (bytes && std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
}

int32_t read_i32(std::FILE* f) { int32_t v = 0; read_into(f, &v, sizeof v); return v; }

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = (double) a[i] - (double) b[i];
        num += d * d;
        den += (double) b[i] * (double) b[i];
    }
    return std::sqrt(num) / std::max(std::sqrt(den), 1e-12);
}

struct Layer {
    int kind = 0;                                   // 0 dense (block 0), 1 MoE
    std::vector<float> mw[8];                       // owned MLA weights, in MLA_KEYS order
    glm::MlaWeights mla;
    std::vector<float> attn_norm, ffn_norm;
    // dense
    int dff = 0;
    std::vector<float> wg, wu, wd;
    glm::MoeGeometry dg;
    // moe
    glm::MoeGeometry gg;
    std::vector<float> router, probs_b, s_gate, s_up, s_down;
    const float* shared[3] = {nullptr, nullptr, nullptr};
    std::map<int, std::array<std::vector<float>, 3>> kept;         // expert id -> owned {gate,up,down}
    std::map<int, std::array<const float*, 3>> kept_ptrs;          // expert id -> pointers into kept
};

struct Ctx {
    const std::vector<Layer>* layers = nullptr;
    bool tolerate = false;
    std::vector<float> zero;
    std::array<const float*, 3> zero_triple{};
};

const float* const* gate_expert(void* p, int layer, int expert) {
    Ctx* c = (Ctx*) p;
    const std::vector<Layer>& ls = *c->layers;
    auto it = ls[(size_t) layer].kept_ptrs.find(expert);
    if (it != ls[(size_t) layer].kept_ptrs.end()) return it->second.data();
    if (c->tolerate) return c->zero_triple.data();   // the teeth run is expected to diverge
    return nullptr;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm47_trunk_parity <fixture.bin>\n"
                    "  build the fixture with: python tools/glm47_trunk_reference.py --raw-fixture <fixture.bin>\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    const uint32_t magic = (uint32_t) read_i32(f), version = (uint32_t) read_i32(f);
    if (magic != MAGIC) { std::fprintf(stderr, "bad fixture magic 0x%08x (want 0x%08x)\n", (unsigned) magic, MAGIC); return 1; }
    if (version != 1) { std::fprintf(stderr, "unsupported fixture version %u\n", (unsigned) version); return 1; }

    int32_t hdr[9];
    read_into(f, hdr, sizeof hdr);
    const int ne = hdr[0], L = hdr[1], V = hdr[2], n_head = hdr[3], head_dim = hdr[4];
    const int kv_lora = hdr[5], n_rot = hdr[6], q_lora = hdr[7], T = hdr[8];
    float freq_base = 0.0f, eps = 0.0f;
    read_into(f, &freq_base, sizeof freq_base);
    read_into(f, &eps, sizeof eps);
    if (ne <= 0 || L <= 0 || V <= 0 || n_head <= 0 || head_dim <= 0 || kv_lora <= 0 || n_rot <= 0 ||
        q_lora <= 0 || n_rot >= head_dim || T <= 0) {
        std::fprintf(stderr, "bad header geometry\n");
        return 1;
    }
    const int nope = head_dim - n_rot;
    const int q_dim = n_head * head_dim, kv_dim = kv_lora + n_rot;

    std::vector<Layer> layers((size_t) L);
    int max_ff = 0;
    for (int l = 0; l < L; ++l) {
        Layer& ly = layers[(size_t) l];
        ly.kind = read_i32(f);
        const size_t sz[8] = {(size_t) q_lora * ne, (size_t) q_lora, (size_t) q_dim * q_lora,
                              (size_t) n_head * kv_lora * nope, (size_t) kv_dim * ne, (size_t) kv_lora,
                              (size_t) n_head * head_dim * kv_lora, (size_t) ne * q_dim};
        for (int i = 0; i < 8; ++i) ly.mw[i] = read_floats(f, sz[i]);
        ly.mla.wq_a = ly.mw[0].data(); ly.mla.q_a_norm = ly.mw[1].data(); ly.mla.wq_b = ly.mw[2].data();
        ly.mla.wk_b = ly.mw[3].data(); ly.mla.kv_a = ly.mw[4].data(); ly.mla.kv_a_norm = ly.mw[5].data();
        ly.mla.wv_b = ly.mw[6].data(); ly.mla.wo = ly.mw[7].data(); ly.mla.rope_freq_base = freq_base;
        ly.attn_norm = read_floats(f, (size_t) ne);
        ly.ffn_norm = read_floats(f, (size_t) ne);
        if (ly.kind == 0) {
            ly.dff = read_i32(f);
            ly.wg = read_floats(f, (size_t) ly.dff * ne);
            ly.wu = read_floats(f, (size_t) ly.dff * ne);
            ly.wd = read_floats(f, (size_t) ne * ly.dff);
            ly.dg.n_embd = ne; ly.dg.n_expert = 0; ly.dg.n_used = 0; ly.dg.ff = ly.dff;
            ly.dg.w_scale = 1.0f; ly.dg.norm_w = false; ly.dg.clamp_exp = 0.0f; ly.dg.clamp_shexp = 0.0f;
        } else {
            const int E = read_i32(f), k = read_i32(f), ff = read_i32(f), sf = read_i32(f);
            if (E <= 0 || k <= 0 || k > E || ff <= 0 || sf <= 0) { std::fprintf(stderr, "bad MoE header\n"); return 1; }
            max_ff = std::max(max_ff, ff);
            ly.gg.n_embd = ne; ly.gg.n_expert = E; ly.gg.n_used = k; ly.gg.ff = ff;
            ly.gg.w_scale = 1.8f; ly.gg.norm_w = true; ly.gg.clamp_exp = 0.0f; ly.gg.clamp_shexp = 0.0f;
            ly.router = read_floats(f, (size_t) E * ne);
            ly.probs_b = read_floats(f, (size_t) E);
            ly.s_gate = read_floats(f, (size_t) sf * ne);
            ly.s_up = read_floats(f, (size_t) sf * ne);
            ly.s_down = read_floats(f, (size_t) ne * sf);
            ly.shared[0] = ly.s_gate.data(); ly.shared[1] = ly.s_up.data(); ly.shared[2] = ly.s_down.data();
            const int nk = read_i32(f);
            for (int j = 0; j < nk; ++j) {
                const int id = read_i32(f);
                std::array<std::vector<float>, 3> t;
                t[0] = read_floats(f, (size_t) ff * ne);
                t[1] = read_floats(f, (size_t) ff * ne);
                t[2] = read_floats(f, (size_t) ne * ff);
                ly.kept[id] = std::move(t);
            }
        }
    }
    for (Layer& ly : layers) {
        for (auto& kv : ly.kept)
            ly.kept_ptrs[kv.first] = {kv.second[0].data(), kv.second[1].data(), kv.second[2].data()};
    }

    // the MLA geometry is shared by every block
    glm::MlaGeometry mg;
    mg.n_embd = ne; mg.n_head = n_head; mg.head_dim = head_dim; mg.kv_lora = kv_lora;
    mg.q_lora = q_lora; mg.n_rot = n_rot;

    const std::vector<float> output_norm = read_floats(f, (size_t) ne);
    const std::vector<float> Wout = read_floats(f, (size_t) V * (size_t) ne);
    std::vector<std::vector<float>> x0((size_t) T);
    for (int t = 0; t < T; ++t) x0[(size_t) t] = read_floats(f, (size_t) ne);
    std::vector<std::vector<std::vector<float>>> eh((size_t) T, std::vector<std::vector<float>>((size_t) L));
    for (int t = 0; t < T; ++t) for (int l = 0; l < L; ++l) eh[(size_t) t][(size_t) l] = read_floats(f, (size_t) ne);
    std::map<std::pair<int, int>, std::vector<int32_t>> e_ids;
    for (int t = 0; t < T; ++t) for (int l = 0; l < L; ++l) {
        if (layers[(size_t) l].kind == 1) {
            std::vector<int32_t> ids((size_t) layers[(size_t) l].gg.n_used);
            read_into(f, ids.data(), ids.size() * sizeof(int32_t));
            e_ids[{t, l}] = std::move(ids);
        }
    }
    const std::vector<float> e_hn = read_floats(f, (size_t) ne);
    const std::vector<float> e_logits = read_floats(f, (size_t) V);
    const int32_t e_argmax = read_i32(f);
    std::fclose(f);

    // the engine's view of the same weights
    std::vector<cglm::Glm47TrunkLayer> arr((size_t) L);
    for (int l = 0; l < L; ++l) {
        Layer& ly = layers[(size_t) l];
        cglm::Glm47TrunkLayer& a = arr[(size_t) l];
        a.kind = ly.kind;
        a.attn_norm = ly.attn_norm.data();
        a.ffn_norm = ly.ffn_norm.data();
        a.mla = &ly.mla;
        if (ly.kind == 0) {
            a.ffn_gate = ly.wg.data(); a.ffn_up = ly.wu.data(); a.ffn_down = ly.wd.data(); a.dense_g = ly.dg;
        } else {
            a.moe_router = ly.router.data(); a.moe_probs_b = ly.probs_b.data();
            a.moe_g = &ly.gg; a.shexp = ly.shared;
        }
    }

    Ctx ctx;
    ctx.layers = &layers;
    ctx.zero.assign((size_t) std::max(max_ff, 1) * ne, 0.0f);
    ctx.zero_triple = {ctx.zero.data(), ctx.zero.data(), ctx.zero.data()};

    std::printf("glm47 trunk parity vs tools/glm47_trunk_reference.py\n");
    std::printf("  %d layers (0 dense, rest MoE), %d tokens, vocab %d, n_embd %d, eps %.0e\n",
                L, T, V, ne, (double) eps);

    std::vector<float> final_hidden;
    // `caches` holds one interleaved buffer per layer; `cache_index` maps layer -> slot (null == identity).
    // The teeth run aliases every layer to slot 0, which is the bug the gate must reject.
    auto run = [&](std::vector<std::vector<float>>& caches, const int* cache_index, bool verbose,
                   bool tolerate, int& ids_exact_out, int& ids_total_out) -> double {
        ctx.tolerate = tolerate;
        double worst = 0.0;
        for (int t = 0; t < T; ++t) {
            std::vector<std::vector<int32_t>> ids;
            std::vector<std::vector<float>> per;
            std::vector<float> hidden((size_t) ne);
            std::string err;
            if (!cglm::glm47_trunk_forward(arr.data(), L, mg, x0[(size_t) t].data(), t, eps,
                                           &caches, cache_index, gate_expert, &ctx, hidden.data(),
                                           &ids, &per, err)) {
                std::fprintf(stderr, "trunk: %s\n", err.c_str());
                std::exit(1);
            }
            if (t == T - 1) final_hidden = hidden;
            std::vector<double> hrow((size_t) L);
            for (int l = 0; l < L; ++l) {
                hrow[(size_t) l] = rel_l2(per[(size_t) l], eh[(size_t) t][(size_t) l]);
                worst = std::max(worst, hrow[(size_t) l]);
            }
            for (int l = 0; l < L; ++l) {
                if (layers[(size_t) l].kind != 1) continue;
                const std::vector<int32_t>& want_ids = e_ids[{t, l}];
                int mism = 0;
                for (size_t i = 0; i < ids[(size_t) l].size(); ++i)
                    mism += (ids[(size_t) l][i] != want_ids[i]);
                ++ids_total_out;
                ids_exact_out += (mism == 0);
                if (verbose && (t == 0 || mism)) {
                    std::printf("  t%d l%d MoE ids", t, l);
                    for (int32_t id : ids[(size_t) l]) std::printf(" %d", (int) id);
                    std::printf("  %s\n", mism ? "FAIL" : "PASS");
                }
            }
            if (verbose) {
                std::printf("  t%d hidden rel err per layer:", t);
                for (int l = 0; l < L; ++l) std::printf(" %.2e", hrow[(size_t) l]);
                std::printf("  %s\n", hrow[(size_t) L - 1] < 1e-4 ? "PASS" : "FAIL");
            }
        }
        return worst;
    };

    // teeth run FIRST, so the correct run below leaves `final_hidden` holding the true last hidden state.
    std::vector<std::vector<float>> one_cache(1);
    std::vector<int> one_index((size_t) L, 0);
    int teeth_ids = 0, teeth_tot = 0;
    const double teeth_h = run(one_cache, one_index.data(), false, true, teeth_ids, teeth_tot);

    bool ok = true;
    int ids_exact = 0, ids_total = 0;
    std::vector<std::vector<float>> caches((size_t) L);
    const double worst_h = run(caches, nullptr, true, false, ids_exact, ids_total);
    ok = ok && (ids_total > 0) && (ids_exact == ids_total);
    // teeth = the bug moves the hidden state FAR ABOVE the parity noise.  Not an absolute threshold: the
    // artifact's own weights put a small share of the signal in attention (the residuals dominate), so at
    // only a couple of tokens the cache bug perturbs the hidden by ~1e-3 - still ~1e5 x the ~1e-8 noise.
    const bool has_teeth = teeth_h > 100.0 * std::max(worst_h, 1e-7);

    // The head: (hc_streams == 1, so no mean) an rms_norm through output_norm, then the projection and
    // argmax.  The norm is the engine's rms_norm_gain; the projection stays inline so its vector is checked.
    std::vector<float> hn((size_t) ne);
    glm::rms_norm_gain(output_norm.data(), ne, final_hidden.data(), hn.data(), eps);
    const double d_hn = rel_l2(hn, e_hn);
    std::vector<float> logits((size_t) V);
    int argmax = 0;
    for (int v = 0; v < V; ++v) {
        double s = 0.0;
        for (int e = 0; e < ne; ++e) s += (double) Wout[(size_t) v * ne + e] * (double) hn[(size_t) e];
        logits[(size_t) v] = (float) s;
        if (v == 0 || s > (double) logits[(size_t) argmax]) argmax = v;
    }
    const double d_log = rel_l2(logits, e_logits);

    std::printf("  head    hn rel %.2e   logits rel %.2e   argmax %d (want %d)  %s\n",
                d_hn, d_log, argmax, (int) e_argmax, argmax == e_argmax ? "PASS" : "FAIL");

    const bool h_ok = worst_h < 1e-4, hn_ok = d_hn < 1e-4, log_ok = d_log < 1e-4;
    const bool ids_ok = (ids_total > 0) && (ids_exact == ids_total);
    const bool am_ok = (argmax == e_argmax);
    ok = ok && h_ok && hn_ok && log_ok && ids_ok && am_ok && has_teeth;

    std::printf("  hidden  worst rel %.3e   %s\n", worst_h, h_ok ? "PASS" : "FAIL");
    std::printf("  hn      rel %.3e   %s\n", d_hn, hn_ok ? "PASS" : "FAIL");
    std::printf("  logits  rel %.3e   %s\n", d_log, log_ok ? "PASS" : "FAIL");
    std::printf("  ids     %d/%d exact   %s\n", ids_exact, ids_total, ids_ok ? "PASS" : "FAIL");
    std::printf("  argmax  %s\n", am_ok ? "PASS" : "FAIL");
    std::printf("  teeth   one shared cache moves the hidden state by rel %.3e   %s\n",
                teeth_h, has_teeth ? "PASS" : "FAIL");
    std::printf("glm47_trunk_parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
