// src/kernels/glm47_mla_parity.cpp - does our decoupled-RoPE MLA (deepseek2) match the reference?
//
// The oracle is tools/glm47_mla_reference.py, transcribed from llama.cpp/Kolibri's deepseek2 build_mla,
// and the fixture carries random weights of the REAL GLM-4.7-Flash shapes plus a set of cases at
// different cache depths and positions:
//
//   python tools/glm47_mla_reference.py --raw-fixture D:/tmp/glm47_mla.bin
//   build/glm47_mla_parity.exe D:/tmp/glm47_mla.bin
//
// Every stage is compared (qr, q_nope, q_pe-after-rope, kv, k_pe-after-rope, Qcur, attn, v, out), so a
// mismatch names the stage that made it rather than only the output.  The gate PASSES when every stage's
// max abs error is < 1e-4 (target < 1e-5).
#include "strata/kernels/glm_mla.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace glm = strata::kernels::glm;

namespace {

constexpr uint32_t MAGIC = 0x47344D4Cu;   // 'GLM4'

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

int32_t read_i32(std::FILE* f) {
    int32_t v = 0;
    if (std::fread(&v, sizeof(int32_t), 1, f) != 1) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

/// Max abs error against the fixture's own float32 copy of the float64 oracle.
double max_abs(const std::vector<float>& got, const std::vector<float>& want) {
    double m = 0.0;
    for (size_t i = 0; i < got.size(); ++i) m = std::max(m, std::fabs((double) got[i] - (double) want[i]));
    return m;
}

struct Stage {
    const char* name;
    double worst = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm47_mla_parity <fixture.bin>\n"
                    "  build the fixture with: python tools/glm47_mla_reference.py --raw-fixture <fixture.bin>\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    const int32_t magic = read_i32(f), version = read_i32(f);
    glm::MlaGeometry g;
    g.n_embd = read_i32(f);
    g.n_head = read_i32(f);
    g.head_dim = read_i32(f);
    g.kv_lora = read_i32(f);
    g.q_lora = read_i32(f);
    g.n_rot = read_i32(f);
    const int n_cases = read_i32(f);
    float freq_base = 0.0f;
    if (std::fread(&freq_base, sizeof(float), 1, f) != 1) { std::fprintf(stderr, "bad header\n"); return 1; }
    if (magic != MAGIC) { std::fprintf(stderr, "bad fixture magic 0x%08x\n", (unsigned) magic); return 1; }
    if (version != 1) { std::fprintf(stderr, "unsupported fixture version %d\n", version); return 1; }
    if (g.n_rot <= 0 || g.n_rot >= g.head_dim) { std::fprintf(stderr, "bad n_rot %d\n", g.n_rot); return 1; }

    const int nope = g.head_dim - g.n_rot;
    const int q_dim = g.n_head * g.head_dim;
    const int kv_dim = g.kv_lora + g.n_rot;

    std::vector<float> wq_a = read_floats(f, (size_t) g.q_lora * g.n_embd);
    std::vector<float> q_a_norm = read_floats(f, (size_t) g.q_lora);
    std::vector<float> wq_b = read_floats(f, (size_t) q_dim * g.q_lora);
    std::vector<float> wk_b = read_floats(f, (size_t) g.n_head * g.kv_lora * nope);
    std::vector<float> kv_a = read_floats(f, (size_t) kv_dim * g.n_embd);
    std::vector<float> kv_a_norm = read_floats(f, (size_t) g.kv_lora);
    std::vector<float> wv_b = read_floats(f, (size_t) g.n_head * g.head_dim * g.kv_lora);
    std::vector<float> wo = read_floats(f, (size_t) g.n_embd * q_dim);

    glm::MlaWeights w;
    w.wq_a = wq_a.data();
    w.q_a_norm = q_a_norm.data();
    w.wq_b = wq_b.data();
    w.wk_b = wk_b.data();
    w.kv_a = kv_a.data();
    w.kv_a_norm = kv_a_norm.data();
    w.wv_b = wv_b.data();
    w.wo = wo.data();
    w.rope_freq_base = freq_base;

    std::printf("glm47 MLA parity vs tools/glm47_mla_reference.py\n");
    std::printf("  geometry: n_embd %d, %d head(s) x %d (nope %d + rope %d), kv_lora %d, q_lora %d, freq_base %.0f\n",
                g.n_embd, g.n_head, g.head_dim, nope, g.n_rot, g.kv_lora, g.q_lora, (double) freq_base);
    std::printf("  %d case(s)\n", n_cases);

    const char* names[9] = {"qr", "q_nope", "q_pe", "kv", "k_pe", "qcur", "attn", "v", "out"};
    Stage stages[9];
    for (int i = 0; i < 9; ++i) stages[i].name = names[i];

    for (int c = 0; c < n_cases; ++c) {
        const int32_t pos = read_i32(f);
        const int32_t n_cache = read_i32(f);
        if (n_cache <= 0) { std::fprintf(stderr, "bad n_cache %d in case %d\n", n_cache, c); return 1; }
        const std::vector<float> x = read_floats(f, (size_t) g.n_embd);
        const std::vector<float> cache_latent = read_floats(f, (size_t) n_cache * g.kv_lora);
        const std::vector<float> cache_kpe = read_floats(f, (size_t) n_cache * g.n_rot);
        const std::vector<float> e_qr = read_floats(f, (size_t) g.q_lora);
        const std::vector<float> e_q_nope = read_floats(f, (size_t) g.n_head * nope);
        const std::vector<float> e_q_pe = read_floats(f, (size_t) g.n_head * g.n_rot);
        const std::vector<float> e_kv = read_floats(f, (size_t) g.kv_lora);
        const std::vector<float> e_k_pe = read_floats(f, (size_t) g.n_rot);
        const std::vector<float> e_qcur = read_floats(f, (size_t) g.n_head * g.kv_lora);
        const std::vector<float> e_attn = read_floats(f, (size_t) g.n_head * g.kv_lora);
        const std::vector<float> e_v = read_floats(f, (size_t) g.n_head * g.head_dim);
        const std::vector<float> e_out = read_floats(f, (size_t) g.n_embd);

        // The kernel reads one cache as [n_cache][kv_lora + n_rot]: interleave the fixture's two banks.
        std::vector<float> cache((size_t) n_cache * kv_dim);
        for (int t = 0; t < n_cache; ++t) {
            std::memcpy(cache.data() + (size_t) t * kv_dim, cache_latent.data() + (size_t) t * g.kv_lora,
                        (size_t) g.kv_lora * sizeof(float));
            std::memcpy(cache.data() + (size_t) t * kv_dim + g.kv_lora, cache_kpe.data() + (size_t) t * g.n_rot,
                        (size_t) g.n_rot * sizeof(float));
        }

        std::vector<float> got_qr((size_t) g.q_lora), got_q_nope((size_t) g.n_head * nope);
        std::vector<float> got_q_pe((size_t) g.n_head * g.n_rot), got_kv((size_t) g.kv_lora);
        std::vector<float> got_k_pe((size_t) g.n_rot), got_qcur((size_t) g.n_head * g.kv_lora);
        std::vector<float> got_attn((size_t) g.n_head * g.kv_lora), got_v((size_t) g.n_head * g.head_dim);
        std::vector<float> got_out((size_t) g.n_embd);
        glm::MlaIntermediates want;
        want.qr = got_qr.data();
        want.q_nope = got_q_nope.data();
        want.q_pe = got_q_pe.data();
        want.kv = got_kv.data();
        want.k_pe = got_k_pe.data();
        want.qcur = got_qcur.data();
        want.attn = got_attn.data();
        want.v = got_v.data();
        glm::mla_forward(w, g, x.data(), n_cache, cache.data(), got_out.data(), want, pos);

        const double d[9] = {
            max_abs(got_qr, e_qr),       max_abs(got_q_nope, e_q_nope), max_abs(got_q_pe, e_q_pe),
            max_abs(got_kv, e_kv),       max_abs(got_k_pe, e_k_pe),     max_abs(got_qcur, e_qcur),
            max_abs(got_attn, e_attn),   max_abs(got_v, e_v),           max_abs(got_out, e_out),
        };
        std::printf("  case %d: n_cache %d, pos %d  |", c, n_cache, pos);
        for (int i = 0; i < 9; ++i) {
            stages[i].worst = std::max(stages[i].worst, d[i]);
            std::printf(" %s %.1e", names[i], d[i]);
        }
        std::printf("\n");
    }
    std::fclose(f);

    std::printf("  stage        max abs error (worst over cases)\n");
    bool ok = true;
    double worst = 0.0;
    for (int i = 0; i < 9; ++i) {
        const bool good = stages[i].worst < 1e-4;
        std::printf("  %-8s     %.3e   %s\n", stages[i].name, stages[i].worst, good ? "PASS" : "FAIL");
        ok = ok && good;
        worst = std::max(worst, stages[i].worst);
    }
    std::printf("glm47_mla_parity: %s (worst stage max abs %.3e)\n", ok ? "PASS" : "FAIL", worst);
    return ok ? 0 : 1;
}
