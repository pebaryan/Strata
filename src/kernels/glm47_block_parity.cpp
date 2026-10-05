// src/kernels/glm47_block_parity.cpp - does our WHOLE decoder block match the reference?
//
// The oracle is tools/glm47_block_reference.py, which composes the already-gated MLA (phase 3) and MoE
// (phase 4) operators and pins the wiring the composition adds:
//
//     xb   = rms_norm(x,  attn_norm_w, 1e-5)      ggml_rms_norm then *w, NOT a folded (1 + w)
//     attn = MLA(xb, cache)
//     x2   = x + attn
//     ff   = rms_norm(x2, ffn_norm_w, 1e-5)
//     ffn  = MoE(ff)                              moe + shared expert
//     out  = x2 + ffn
//
//   python tools/glm47_block_reference.py --gguf <model.gguf> --layer 1 --raw-fixture D:/tmp/glm47_block.bin
//   build/glm47_block_parity.exe D:/tmp/glm47_block.bin
//
// Every stage is compared (xb, attn, x2, ff, moe, shexp, out) so a mismatch names the site that made it,
// and the top-4 expert ids are EXACT.  The gate PASSES when every stage's max abs error is < 1e-4.
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/glm_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace glm = strata::kernels::glm;

namespace {

constexpr uint32_t MAGIC = 0x47423437u;   // 'GB47'

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

double max_abs(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::fabs((double) a[i] - (double) b[i]));
    return m;
}

struct Stage { const char* name; double worst = 0.0; };

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm47_block_parity <fixture.bin>\n"
                    "  build the fixture with: python tools/glm47_block_reference.py --raw-fixture <fixture.bin>\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    const uint32_t magic = (uint32_t) read_i32(f), version = (uint32_t) read_i32(f);
    if (magic != MAGIC) { std::fprintf(stderr, "bad fixture magic 0x%08x\n", (unsigned) magic); return 1; }
    if (version != 1) { std::fprintf(stderr, "unsupported fixture version %u\n", (unsigned) version); return 1; }

    int32_t hdr[10];
    read_into(f, hdr, sizeof hdr);
    const int ne = hdr[0], n_head = hdr[1], head_dim = hdr[2], kv_lora = hdr[3], q_lora = hdr[4];
    const int n_rot = hdr[5], E = hdr[6], k = hdr[7], ff = hdr[8], n_cases = hdr[9];
    float freq_base = 0.0f, eps = 0.0f;
    read_into(f, &freq_base, sizeof freq_base);
    read_into(f, &eps, sizeof eps);
    if (ne <= 0 || n_head <= 0 || head_dim <= 0 || kv_lora <= 0 || q_lora <= 0 || n_rot <= 0 ||
        ff <= 0 || n_rot >= head_dim || E < 0 || k < 0 || (E > 0 && (k <= 0 || k > E))) {
        std::fprintf(stderr, "bad header geometry\n");
        return 1;
    }
    const int nope = head_dim - n_rot;
    const int q_dim = n_head * head_dim, kv_dim = kv_lora + n_rot;

    glm::MlaWeights mw;
    std::vector<float> wq_a = read_floats(f, (size_t) q_lora * ne);
    std::vector<float> q_a_norm = read_floats(f, (size_t) q_lora);
    std::vector<float> wq_b = read_floats(f, (size_t) q_dim * q_lora);
    std::vector<float> wk_b = read_floats(f, (size_t) n_head * kv_lora * nope);
    std::vector<float> kv_a = read_floats(f, (size_t) kv_dim * ne);
    std::vector<float> kv_a_norm = read_floats(f, (size_t) kv_lora);
    std::vector<float> wv_b = read_floats(f, (size_t) n_head * head_dim * kv_lora);
    std::vector<float> wo = read_floats(f, (size_t) ne * q_dim);
    mw.wq_a = wq_a.data(); mw.q_a_norm = q_a_norm.data(); mw.wq_b = wq_b.data(); mw.wk_b = wk_b.data();
    mw.kv_a = kv_a.data(); mw.kv_a_norm = kv_a_norm.data(); mw.wv_b = wv_b.data(); mw.wo = wo.data();
    mw.rope_freq_base = freq_base;

    std::vector<float> attn_norm_w = read_floats(f, (size_t) ne);
    std::vector<float> ffn_norm_w = read_floats(f, (size_t) ne);

    if (E == 0) {
        // block 0: the leading dense stem.  Same shell, but a plain parallel-SiLU FFN (ff 10240), no
        // router and no shared expert - the one block the MoE path below does not describe.
        std::vector<float> wg = read_floats(f, (size_t) ff * ne);
        std::vector<float> wu = read_floats(f, (size_t) ff * ne);
        std::vector<float> wd = read_floats(f, (size_t) ne * ff);
        glm::MoeGeometry dg;
        dg.n_embd = ne; dg.n_expert = 0; dg.n_used = 0; dg.ff = ff; dg.w_scale = 1.0f; dg.norm_w = false;
        glm::MlaGeometry dmg;
        dmg.n_embd = ne; dmg.n_head = n_head; dmg.head_dim = head_dim; dmg.kv_lora = kv_lora;
        dmg.q_lora = q_lora; dmg.n_rot = n_rot;
        std::printf("glm47 dense-stem parity (block 0) vs tools/glm47_block_reference.py --dense\n");
        std::printf("  dense FFN ff %d, eps %.0e, clamp off (no swiglu key)\n", ff, (double) eps);
        std::printf("  %d case(s)\n", (int) n_cases);
        const char* dn[6] = {"xb", "attn", "x2", "ff", "ffn", "out"};
        Stage ds[6];
        for (int i = 0; i < 6; ++i) ds[i].name = dn[i];
        bool dok = true;
        for (int c = 0; c < n_cases; ++c) {
            const int32_t pos = read_i32(f), n_cache = read_i32(f);
            if (n_cache <= 0) { std::fprintf(stderr, "bad n_cache\n"); return 1; }
            const std::vector<float> x = read_floats(f, (size_t) ne);
            const std::vector<float> cache_latent = read_floats(f, (size_t) n_cache * kv_lora);
            const std::vector<float> cache_kpe = read_floats(f, (size_t) n_cache * n_rot);
            const std::vector<float> e_xb = read_floats(f, (size_t) ne);
            const std::vector<float> e_attn = read_floats(f, (size_t) ne);
            const std::vector<float> e_x2 = read_floats(f, (size_t) ne);
            const std::vector<float> e_ff = read_floats(f, (size_t) ne);
            const std::vector<float> e_ffn = read_floats(f, (size_t) ne);
            const std::vector<float> e_out = read_floats(f, (size_t) ne);

            std::vector<float> cache((size_t) n_cache * kv_dim);
            for (int t = 0; t < n_cache; ++t) {
                std::memcpy(cache.data() + (size_t) t * kv_dim, cache_latent.data() + (size_t) t * kv_lora,
                            (size_t) kv_lora * sizeof(float));
                std::memcpy(cache.data() + (size_t) t * kv_dim + kv_lora, cache_kpe.data() + (size_t) t * n_rot,
                            (size_t) n_rot * sizeof(float));
            }
            std::vector<float> xb((size_t) ne), attn((size_t) ne), x2((size_t) ne), ffv((size_t) ne);
            std::vector<float> ffnv((size_t) ne);
            glm::rms_norm_gain(attn_norm_w.data(), ne, x.data(), xb.data(), eps);
            std::vector<float> g_qr((size_t) q_lora), g_qn((size_t) n_head * nope), g_qp((size_t) n_head * n_rot);
            std::vector<float> g_kv((size_t) kv_lora), g_kp((size_t) n_rot), g_qc((size_t) n_head * kv_lora);
            std::vector<float> g_at((size_t) n_head * kv_lora), g_v((size_t) n_head * head_dim);
            glm::MlaIntermediates want;
            want.qr = g_qr.data(); want.q_nope = g_qn.data(); want.q_pe = g_qp.data(); want.kv = g_kv.data();
            want.k_pe = g_kp.data(); want.qcur = g_qc.data(); want.attn = g_at.data(); want.v = g_v.data();
            glm::mla_forward(mw, dmg, xb.data(), n_cache, cache.data(), attn.data(), want, pos);
            for (int i = 0; i < ne; ++i) x2[(size_t) i] = x[(size_t) i] + attn[(size_t) i];
            glm::rms_norm_gain(ffn_norm_w.data(), ne, x2.data(), ffv.data(), eps);
            glm::expert_ffn(wg.data(), wu.data(), wd.data(), dg, ffv.data(), ffnv.data(), 0.0f);
            std::vector<float> out((size_t) ne);
            for (int i = 0; i < ne; ++i) out[(size_t) i] = x2[(size_t) i] + ffnv[(size_t) i];

            const double d[6] = {max_abs(xb, e_xb), max_abs(attn, e_attn), max_abs(x2, e_x2),
                                 max_abs(ffv, e_ff), max_abs(ffnv, e_ffn), max_abs(out, e_out)};
            std::printf("  case %d: pos %d, n_cache %d  |", c, pos, n_cache);
            for (int i = 0; i < 6; ++i) { ds[i].worst = std::max(ds[i].worst, d[i]); std::printf(" %s %.1e", dn[i], d[i]); }
            std::printf("\n");
        }
        std::fclose(f);
        std::printf("  stage    max abs error (worst over cases)\n");
        double dw = 0.0;
        for (int i = 0; i < 6; ++i) {
            const bool good = ds[i].worst < 1e-4;
            std::printf("  %-8s %.3e   %s\n", ds[i].name, ds[i].worst, good ? "PASS" : "FAIL");
            dok = dok && good;
            dw = std::max(dw, ds[i].worst);
        }
        std::printf("glm47_block_parity (dense stem): %s (worst stage max abs %.3e)\n", dok ? "PASS" : "FAIL", dw);
        return dok ? 0 : 1;
    }

    std::vector<float> router = read_floats(f, (size_t) E * ne);
    std::vector<float> probs_b = read_floats(f, (size_t) E);
    std::vector<float> s_gate = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_up = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_down = read_floats(f, (size_t) ne * ff);
    const float* shared[3] = {s_gate.data(), s_up.data(), s_down.data()};

    glm::MlaGeometry mg;
    mg.n_embd = ne; mg.n_head = n_head; mg.head_dim = head_dim; mg.kv_lora = kv_lora;
    mg.q_lora = q_lora; mg.n_rot = n_rot;

    glm::MoeGeometry gg;
    gg.n_embd = ne; gg.n_expert = E; gg.n_used = k; gg.ff = ff;
    gg.w_scale = 1.8f; gg.norm_w = true;   // read below from the fixture, kept in sync by the writer

    std::printf("glm47 block parity vs tools/glm47_block_reference.py\n");
    std::printf("  block: attn_norm(eps %.0e) -> MLA(%d head x %d, rope %d) -> +x -> ffn_norm -> "
                "MoE(%d experts, top-%d, ff %d) -> +x\n", (double) eps, n_head, head_dim, n_rot, E, k, ff);
    std::printf("  %d case(s)\n", (int) n_cases);

    const char* names[7] = {"xb", "attn", "x2", "ff", "moe", "shexp", "out"};
    Stage stages[7];
    for (int i = 0; i < 7; ++i) stages[i].name = names[i];
    double ids_ok = 0;
    bool ok = true;

    for (int c = 0; c < n_cases; ++c) {
        const int32_t pos = read_i32(f), n_cache = read_i32(f);
        if (n_cache <= 0) { std::fprintf(stderr, "bad n_cache\n"); return 1; }
        const std::vector<float> x = read_floats(f, (size_t) ne);
        const std::vector<float> cache_latent = read_floats(f, (size_t) n_cache * kv_lora);
        const std::vector<float> cache_kpe = read_floats(f, (size_t) n_cache * n_rot);
        std::vector<std::vector<float>> ex_gate((size_t) k), ex_up((size_t) k), ex_down((size_t) k);
        for (int i = 0; i < k; ++i) {
            ex_gate[(size_t) i] = read_floats(f, (size_t) ff * ne);
            ex_up[(size_t) i] = read_floats(f, (size_t) ff * ne);
            ex_down[(size_t) i] = read_floats(f, (size_t) ne * ff);
        }
        const std::vector<float> e_xb = read_floats(f, (size_t) ne);
        const std::vector<float> e_attn = read_floats(f, (size_t) ne);
        const std::vector<float> e_x2 = read_floats(f, (size_t) ne);
        const std::vector<float> e_ff = read_floats(f, (size_t) ne);
        std::vector<int32_t> e_ids((size_t) k, -1);
        read_into(f, e_ids.data(), (size_t) k * sizeof(int32_t));
        const std::vector<float> e_weights = read_floats(f, (size_t) k);
        const std::vector<float> e_moe = read_floats(f, (size_t) ne);
        const std::vector<float> e_shexp = read_floats(f, (size_t) ne);
        const std::vector<float> e_out = read_floats(f, (size_t) ne);

        // the kernel reads one cache as [n_cache][kv_lora + n_rot]: interleave the fixture's two banks
        std::vector<float> cache((size_t) n_cache * kv_dim);
        for (int t = 0; t < n_cache; ++t) {
            std::memcpy(cache.data() + (size_t) t * kv_dim, cache_latent.data() + (size_t) t * kv_lora,
                        (size_t) kv_lora * sizeof(float));
            std::memcpy(cache.data() + (size_t) t * kv_dim + kv_lora, cache_kpe.data() + (size_t) t * n_rot,
                        (size_t) n_rot * sizeof(float));
        }

        // the block, through the engine's own kernels
        std::vector<float> xb((size_t) ne), attn((size_t) ne), x2((size_t) ne), ffv((size_t) ne);
        glm::rms_norm_gain(attn_norm_w.data(), ne, x.data(), xb.data(), eps);
        std::vector<float> got_qr((size_t) q_lora), got_q_nope((size_t) n_head * nope);
        std::vector<float> got_q_pe((size_t) n_head * n_rot), got_kv((size_t) kv_lora);
        std::vector<float> got_k_pe((size_t) n_rot), got_qcur((size_t) n_head * kv_lora);
        std::vector<float> got_attnh((size_t) n_head * kv_lora), got_v((size_t) n_head * head_dim);
        glm::MlaIntermediates want;
        want.qr = got_qr.data(); want.q_nope = got_q_nope.data(); want.q_pe = got_q_pe.data();
        want.kv = got_kv.data(); want.k_pe = got_k_pe.data(); want.qcur = got_qcur.data();
        want.attn = got_attnh.data(); want.v = got_v.data();
        glm::mla_forward(mw, mg, xb.data(), n_cache, cache.data(), attn.data(), want, pos);
        for (int i = 0; i < ne; ++i) x2[(size_t) i] = x[(size_t) i] + attn[(size_t) i];
        glm::rms_norm_gain(ffn_norm_w.data(), ne, x2.data(), ffv.data(), eps);

        std::vector<const float*> expert_ptrs((size_t) k * 3);
        std::vector<const float* const*> experts((size_t) k);
        for (int i = 0; i < k; ++i) {
            expert_ptrs[(size_t) i * 3 + 0] = ex_gate[(size_t) i].data();
            expert_ptrs[(size_t) i * 3 + 1] = ex_up[(size_t) i].data();
            expert_ptrs[(size_t) i * 3 + 2] = ex_down[(size_t) i].data();
            experts[(size_t) i] = &expert_ptrs[(size_t) i * 3];
        }
        std::vector<int32_t> got_ids((size_t) k, -1);
        std::vector<float> got_weights((size_t) k);
        std::vector<float> moe((size_t) ne), shexp((size_t) ne), ffn_out((size_t) ne);
        glm::moe_forward(router.data(), probs_b.data(), gg, ffv.data(), experts.data(), shared,
                         ffn_out.data(), got_ids.data(), got_weights.data(), moe.data(), shexp.data());
        std::vector<float> out((size_t) ne);
        for (int i = 0; i < ne; ++i) out[(size_t) i] = x2[(size_t) i] + ffn_out[(size_t) i];

        int mism = 0;
        for (int i = 0; i < k; ++i) mism += (got_ids[(size_t) i] != e_ids[(size_t) i]);
        const double d[7] = {max_abs(xb, e_xb), max_abs(attn, e_attn), max_abs(x2, e_x2), max_abs(ffv, e_ff),
                             max_abs(moe, e_moe), max_abs(shexp, e_shexp), max_abs(out, e_out)};
        std::printf("  case %d: pos %d, n_cache %d, ids", c, pos, n_cache);
        for (int i = 0; i < k; ++i) std::printf(" %d", (int) got_ids[(size_t) i]);
        std::printf("  %s  |", mism ? "FAIL" : "PASS");
        for (int i = 0; i < 7; ++i) { stages[i].worst = std::max(stages[i].worst, d[i]); std::printf(" %s %.1e", names[i], d[i]); }
        std::printf("\n");
        ids_ok += (mism == 0);
        ok = ok && (mism == 0);
    }
    std::fclose(f);

    std::printf("  stage    max abs error (worst over cases)\n");
    double worst = 0.0;
    for (int i = 0; i < 7; ++i) {
        const bool good = stages[i].worst < 1e-4;
        std::printf("  %-8s %.3e   %s\n", stages[i].name, stages[i].worst, good ? "PASS" : "FAIL");
        ok = ok && good;
        worst = std::max(worst, stages[i].worst);
    }
    std::printf("  ids      %d/%d cases exact   %s\n", (int) ids_ok, (int) n_cases,
                ids_ok == n_cases ? "PASS" : "FAIL");
    std::printf("glm47_block_parity: %s (worst stage max abs %.3e)\n", ok ? "PASS" : "FAIL", worst);
    return ok ? 0 : 1;
}
