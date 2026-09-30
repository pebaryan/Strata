// src/kernels/glm_moe_parity.cpp - does our MoE site and router match the reference?
//
// The oracle is tools/glm5_moe_reference.py, transcribed from the shared build_moe_ffn:
//
//   python tools/glm5_moe_reference.py --gguf <shard1> --layer 4 --bias-matters --raw-fixture /tmp/moe.bin
//   build-volta/glm_moe_parity /tmp/moe.bin
//
// --bias-matters matters: it searches for an input whose selection CHANGES when ffn_exp_probs_b is
// applied.  With an arbitrary input the biased and unbiased choices often agree, and then a port that
// dropped the bias entirely would still match the fixture exactly.
#include "strata/kernels/glm_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace glm = strata::kernels::glm;

namespace {

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, bool& ok,
             double scale_tol = 1e-4) {
    double scale = 1e-30, ma = 0.0, mr = 0.0;
    for (float v : want) scale = std::max(scale, std::fabs((double) v));
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        if (std::fabs((double) want[i]) >= 0.1 * scale) mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const bool good = ma / scale < scale_tol && mr < 1e-3;
    std::printf("  %-8s max abs %.3e (= %.2e of scale)   worst element rel %.3e   %s\n", what, ma, ma / scale,
                mr, good ? "PASS" : "FAIL");
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_moe_parity <fixture.bin>   (tools/glm5_moe_reference.py --raw-fixture)\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int32_t hdr[5] = {0};
    if (std::fread(hdr, sizeof(int32_t), 5, f) != 5) { std::fprintf(stderr, "bad header\n"); return 1; }

    glm::MoeGeometry g;
    g.n_embd = hdr[0];
    g.n_expert = hdr[1];
    g.n_used = hdr[2];
    g.ff = hdr[3];
    const int ne = g.n_embd, E = g.n_expert, k = g.n_used, ff = g.ff;

    std::vector<float> x = read_floats(f, (size_t) ne);
    std::vector<float> router = read_floats(f, (size_t) E * ne);
    std::vector<float> probs_b = read_floats(f, (size_t) E);

    // the experts arrive in the router's order, which is the order the expected ids were gathered in
    std::vector<std::vector<float>> ex_gate((size_t) k, std::vector<float>((size_t) ff * ne));
    std::vector<std::vector<float>> ex_up((size_t) k, std::vector<float>((size_t) ff * ne));
    std::vector<std::vector<float>> ex_down((size_t) k, std::vector<float>((size_t) ne * ff));
    std::vector<const float*> expert_ptrs((size_t) k * 3);
    for (int i = 0; i < k; ++i) {
        ex_gate[(size_t) i] = read_floats(f, (size_t) ff * ne);
        ex_up[(size_t) i] = read_floats(f, (size_t) ff * ne);
        ex_down[(size_t) i] = read_floats(f, (size_t) ne * ff);
    }
    std::vector<float> s_gate = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_up = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_down = read_floats(f, (size_t) ne * ff);

    std::vector<int32_t> e_ids((size_t) k, -1);
    if (std::fread(e_ids.data(), sizeof(int32_t), (size_t) k, f) != (size_t) k) {
        std::fprintf(stderr, "fixture is truncated (ids)\n");
        return 1;
    }
    const std::vector<float> e_weights = read_floats(f, (size_t) k);
    const std::vector<float> e_moe = read_floats(f, (size_t) ne);
    const std::vector<float> e_shexp = read_floats(f, (size_t) ne);
    const std::vector<float> e_out = read_floats(f, (size_t) ne);
    std::fclose(f);

    std::printf("MoE parity vs tools/glm5_moe_reference.py: layer %d, %d experts, %d used, ff %d, "
                "scale %.1f\n", hdr[4], E, k, ff, (double) g.w_scale);

    std::vector<const float* const*> experts((size_t) k);
    for (int i = 0; i < k; ++i) {
        expert_ptrs[(size_t) i * 3 + 0] = ex_gate[(size_t) i].data();
        expert_ptrs[(size_t) i * 3 + 1] = ex_up[(size_t) i].data();
        expert_ptrs[(size_t) i * 3 + 2] = ex_down[(size_t) i].data();
        experts[(size_t) i] = &expert_ptrs[(size_t) i * 3];
    }
    const float* shared[3] = {s_gate.data(), s_up.data(), s_down.data()};

    std::vector<int32_t> got_ids((size_t) k, -1);
    std::vector<float> got_weights((size_t) k), got_moe((size_t) ne), got_shexp((size_t) ne);
    std::vector<float> got_out((size_t) ne);
    glm::moe_forward(router.data(), probs_b.data(), g, x.data(), experts.data(), shared, got_out.data(),
                     got_ids.data(), got_weights.data(), got_moe.data(), got_shexp.data());

    bool ok = true;
    int mismatched = 0;
    for (int i = 0; i < k; ++i) mismatched += (got_ids[(size_t) i] != e_ids[(size_t) i]);
    std::printf("  %-8s %s   got %d,%d,%d,%d,%d,%d,%d,%d   want %d,%d,%d,%d,%d,%d,%d,%d\n", "ids",
                mismatched ? "FAIL" : "PASS", got_ids[0], got_ids[1], got_ids[2], got_ids[3], got_ids[4],
                got_ids[5], got_ids[6], got_ids[7], e_ids[0], e_ids[1], e_ids[2], e_ids[3], e_ids[4],
                e_ids[5], e_ids[6], e_ids[7]);
    ok = ok && mismatched == 0;
    compare("weights", got_weights, e_weights, ok);
    compare("moe", got_moe, e_moe, ok);
    compare("shexp", got_shexp, e_shexp, ok);
    compare("out", got_out, e_out, ok);

    // invariants that do not depend on the oracle agreeing: the weights are normalised then scaled, so
    // their sum is exactly w_scale (scaling before normalising would leave a sum of 1), all are positive,
    // and the ids are distinct and in range.
    {
        double sum = 0.0;
        bool pos = true;
        for (int i = 0; i < k; ++i) { sum += got_weights[(size_t) i]; pos = pos && got_weights[(size_t) i] > 0.0f; }
        std::vector<int32_t> sorted = got_ids;
        std::sort(sorted.begin(), sorted.end());
        const bool distinct = std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end();
        const bool in_range = sorted.front() >= 0 && sorted.back() < E;
        const bool sum_ok = std::fabs(sum - (double) g.w_scale) < 1e-4;
        std::printf("  invariant  sum(weights) %.6f (== w_scale %.1f, i.e. normalised THEN scaled) %s   "
                    "all positive %s   ids distinct %s in range %s\n", sum, (double) g.w_scale,
                    sum_ok ? "PASS" : "FAIL", pos ? "PASS" : "FAIL", distinct ? "PASS" : "FAIL",
                    in_range ? "PASS" : "FAIL");
        ok = ok && sum_ok && pos && distinct && in_range;
    }

    std::printf("glm_moe_parity: %s\n", ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
