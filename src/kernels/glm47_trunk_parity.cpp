// src/kernels/glm47_trunk_parity.cpp - does our WHOLE GLM-4.7-Flash trunk match the reference?
//
// The oracle is tools/glm47_trunk_reference.py: the 47-block loop, composed from the block kinds the
// phase-4 gates already pinned, plus the head.  Every per-block operator is independently gated
// (glm47_mla_parity, glm47_moe_parity, glm47_block_parity); this gate pins what the LOOP adds:
//
//     x = hidden state
//     for l in 0..L-1:  x = block_l(x, cache_l)     # block 0 dense, the rest MoE; EACH LAYER OWNS ITS CACHE
//     hn = rms_norm(x, output_norm, 1e-5)
//     logits = W_out @ hn ;  token = argmax(logits)
//
// It compares, per token and per layer, the hidden state (so a mismatch names the block that made it),
// the top-4 expert ids of every MoE block (EXACT), the head's normed hidden state, the logits and the
// argmax.  The failure this exists to catch is the cache: a loop that reuses one cache for every layer
// hands layer l the keys layer l-1 wrote - a finite, plausible, wrong hidden state that no per-block
// gate can see.
//
//   python tools/glm47_trunk_reference.py --raw-fixture D:/tmp/glm47_trunk.bin [--layers 2] [--tokens 2]
//   build/glm47_trunk_parity.exe D:/tmp/glm47_trunk.bin
//
// PASSES when every hidden state, hn and logits is within 1e-4 RELATIVE, the expert ids are exact, and
// the argmax is exact.
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
    glm::MlaGeometry mg;
    std::vector<float> attn_norm, ffn_norm;
    // dense
    int dff = 0;
    std::vector<float> wg, wu, wd;
    glm::MoeGeometry dg;
    // moe
    glm::MoeGeometry gg;
    std::vector<float> router, probs_b, s_gate, s_up, s_down;
    const float* shared[3] = {nullptr, nullptr, nullptr};
    std::map<int, std::array<std::vector<float>, 3>> kept;   // expert id -> {gate, up, down}
};

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
        ly.mg.n_embd = ne; ly.mg.n_head = n_head; ly.mg.head_dim = head_dim; ly.mg.kv_lora = kv_lora;
        ly.mg.q_lora = q_lora; ly.mg.n_rot = n_rot;
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

    const std::vector<float> output_norm = read_floats(f, (size_t) ne);
    const std::vector<float> Wout = read_floats(f, (size_t) V * (size_t) ne);
    std::vector<std::vector<float>> xs((size_t) T);
    for (int t = 0; t < T; ++t) xs[(size_t) t] = read_floats(f, (size_t) ne);
    // expected: hidden[T][L][ne], then per (t,moe-layer) the k ids, then hn, logits, argmax
    std::vector<std::vector<std::vector<float>>> eh((size_t) T,
        std::vector<std::vector<float>>((size_t) L));
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

    std::printf("glm47 trunk parity vs tools/glm47_trunk_reference.py\n");
    std::printf("  %d layers (0 dense, rest MoE), %d tokens, vocab %d, n_embd %d, eps %.0e\n",
                L, T, V, ne, (double) eps);
    std::printf("  per-layer MLA, block 0 ff %d dense, MoE top-%d/%d\n",
                layers[0].dff, layers[0].kind == 1 ? layers[0].gg.n_used : 0,
                layers[0].kind == 1 ? layers[0].gg.n_expert : 0);

    // one interleaved [rows][kv_dim] cache PER LAYER - the whole point of the loop
    std::vector<std::vector<float>> caches((size_t) L);

    bool ok = true;
    int ids_exact = 0, ids_total = 0;
    const std::vector<std::vector<float>> x0 = xs;   // fixture inputs; xs is consumed in place by the loop

    // The loop, parameterised by the cache set: one cache per layer for the correct run, and a single
    // shared cache for the teeth run (the classic "reuse one cache for every layer" bug).
    auto run_trunk = [&](std::vector<std::vector<float>>& caches, bool verbose, bool tolerate,
                         int& ids_exact_out, int& ids_total_out) -> double {
        xs = x0;
        double worst = 0.0;
    for (int t = 0; t < T; ++t) {
        std::vector<double> hrow((size_t) L);
        for (int l = 0; l < L; ++l) {
            Layer& ly = layers[(size_t) l];
            std::vector<float>& cache = caches[caches.size() == 1 ? 0 : (size_t) l];
            const int rows = t;                     // rows already stored (one per previous token)
            std::vector<float> xb((size_t) ne);
            glm::rms_norm_gain(ly.attn_norm.data(), ne, xs[(size_t) t].data(), xb.data(), eps);

            // pass 1: compute THIS token's kv / k_pe and append them to this layer's cache
            cache.resize((size_t) (rows + 1) * kv_dim);
            std::vector<float> g_qr((size_t) q_lora), g_qn((size_t) n_head * nope), g_qp((size_t) n_head * n_rot);
            std::vector<float> g_kv((size_t) kv_lora), g_kp((size_t) n_rot), g_qc((size_t) n_head * kv_lora);
            std::vector<float> g_at((size_t) n_head * kv_lora), g_v((size_t) n_head * head_dim);
            glm::MlaIntermediates want;
            want.qr = g_qr.data(); want.q_nope = g_qn.data(); want.q_pe = g_qp.data(); want.kv = g_kv.data();
            want.k_pe = g_kp.data(); want.qcur = g_qc.data(); want.attn = g_at.data(); want.v = g_v.data();
            std::vector<float> attn((size_t) ne);
            glm::mla_forward(ly.mla, ly.mg, xb.data(), rows + 1, cache.data(), attn.data(), want, t);
            std::memcpy(cache.data() + (size_t) rows * kv_dim, g_kv.data(), (size_t) kv_lora * sizeof(float));
            std::memcpy(cache.data() + (size_t) rows * kv_dim + kv_lora, g_kp.data(), (size_t) n_rot * sizeof(float));
            // pass 2: attend over history + this token
            glm::mla_forward(ly.mla, ly.mg, xb.data(), rows + 1, cache.data(), attn.data(), want, t);

            std::vector<float> x2((size_t) ne), ffv((size_t) ne), ffnout((size_t) ne);
            for (int i = 0; i < ne; ++i) x2[(size_t) i] = xs[(size_t) t][(size_t) i] + attn[(size_t) i];
            glm::rms_norm_gain(ly.ffn_norm.data(), ne, x2.data(), ffv.data(), eps);

            if (ly.kind == 0) {
                glm::expert_ffn(ly.wg.data(), ly.wu.data(), ly.wd.data(), ly.dg, ffv.data(), ffnout.data(), 0.0f);
            } else {
                const int k = ly.gg.n_used;
                std::vector<int32_t> rids((size_t) k);
                std::vector<float> rw((size_t) k);
                glm::moe_route(ly.router.data(), ly.probs_b.data(), ly.gg, ffv.data(), rids.data(), rw.data());
                std::vector<const float*> expert_ptrs((size_t) k * 3);
                std::vector<const float* const*> experts((size_t) k);
                const std::vector<float> zero_w((size_t) ly.gg.ff * ne, 0.0f);
                for (int i = 0; i < k; ++i) {
                    auto it = ly.kept.find(rids[(size_t) i]);
                    if (it == ly.kept.end()) {
                        if (!tolerate) {
                            std::fprintf(stderr, "token %d layer %d: engine chose expert %d, not in the fixture\n",
                                         t, l, (int) rids[(size_t) i]);
                            std::exit(1);
                        }
                        // the teeth run is expected to diverge; an absent expert stands in as zero
                        expert_ptrs[(size_t) i * 3 + 0] = zero_w.data();
                        expert_ptrs[(size_t) i * 3 + 1] = zero_w.data();
                        expert_ptrs[(size_t) i * 3 + 2] = zero_w.data();
                        experts[(size_t) i] = &expert_ptrs[(size_t) i * 3];
                        continue;
                    }
                    expert_ptrs[(size_t) i * 3 + 0] = it->second[0].data();
                    expert_ptrs[(size_t) i * 3 + 1] = it->second[1].data();
                    expert_ptrs[(size_t) i * 3 + 2] = it->second[2].data();
                    experts[(size_t) i] = &expert_ptrs[(size_t) i * 3];
                }
                std::vector<int32_t> got_ids((size_t) k, -1);
                std::vector<float> got_w((size_t) k);
                glm::moe_forward(ly.router.data(), ly.probs_b.data(), ly.gg, ffv.data(), experts.data(),
                                 ly.shared, ffnout.data(), got_ids.data(), got_w.data());
                const std::vector<int32_t>& want_ids = e_ids[{t, l}];
                int mism = 0;
                for (int i = 0; i < k; ++i) mism += (got_ids[(size_t) i] != want_ids[(size_t) i]);
                ++ids_total_out;
                ids_exact_out += (mism == 0);
                if (verbose && (t == 0 || mism)) {
                    std::printf("  t%d l%d MoE ids", t, l);
                    for (int i = 0; i < k; ++i) std::printf(" %d", (int) got_ids[(size_t) i]);
                    std::printf("  %s\n", mism ? "FAIL" : "PASS");
                }
            }

            std::vector<float> out((size_t) ne);
            for (int i = 0; i < ne; ++i) out[(size_t) i] = x2[(size_t) i] + ffnout[(size_t) i];
            xs[(size_t) t] = out;                    // feed the NEXT block
            hrow[(size_t) l] = rel_l2(out, eh[(size_t) t][(size_t) l]);
            worst = std::max(worst, hrow[(size_t) l]);
        }
        if (verbose) {
            std::printf("  t%d hidden rel err per layer:", t);
            for (int l = 0; l < L; ++l) std::printf(" %.2e", hrow[(size_t) l]);
            std::printf("  %s\n", hrow[(size_t) L - 1] < 1e-4 ? "PASS" : "FAIL");
        }
    }
        return worst;
    };

    // teeth run FIRST, so the correct run below leaves `xs` holding the true final hidden state for the head.
    std::vector<std::vector<float>> one_cache(1);
    int teeth_ids = 0, teeth_tot = 0;
    const double teeth_h = run_trunk(one_cache, false, true, teeth_ids, teeth_tot);
    const bool has_teeth = teeth_h > 1e-2;

    const double worst_h = run_trunk(caches, true, false, ids_exact, ids_total);
    ok = ok && (ids_total > 0) && (ids_exact == ids_total);

    // The head: (hc_streams == 1, so no mean) an rms_norm through output_norm, then the projection
    // and argmax.  The norm is the engine's rms_norm_gain (the same op the head stage wraps); the
    // projection and argmax stay inline so the head's matmul can be checked as a vector here.
    std::vector<float> hn((size_t) ne);
    glm::rms_norm_gain(output_norm.data(), ne, xs[(size_t) T - 1].data(), hn.data(), eps);
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
