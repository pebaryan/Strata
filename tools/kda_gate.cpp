// A gate for kda_forward against the reference implementation's own numbers.
//
// The weights, the input and the expected values come from the fixture built by the phase-6-validated oracle
// (tools/glm5_kda_reference.py), so this validates the KERNEL against the artifact's real weights - not the
// binding.  Binding these same sixteen tensors is a separate check, and this gate must not be reported as
// covering it.
//
// kda_forward is host code (src/kernels/glm_kda.cpp), so everything here is a plain std::vector: no CUDA, no
// device copies, no stream.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/kernels/glm_kda.hpp"

namespace {

std::vector<float> read_bin(const std::string& path, size_t n) {
    std::vector<float> v(n);
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); std::exit(1); }
    const size_t got = std::fread(v.data(), sizeof(float), n, f);
    std::fclose(f);
    if (got != n) { std::fprintf(stderr, "%s: short read (%zu of %zu floats)\n", path.c_str(), got, n); std::exit(1); }
    return v;
}

// Prints the stage's stats and, when the oracle's own file for that stage exists, a PASS/FAIL against it.
// The tolerance is relative to the reference's own scale, so a stage whose values are small is not judged by an
// absolute epsilon.
int report(const std::string& name, const std::vector<float>& got, const std::string& dir, const std::string& ref) {
    double mn = got[0], mx = got[0], sm = 0.0;
    for (float v : got) { if (v < mn) mn = v; if (v > mx) mx = v; sm += v; }
    std::printf("  %-8s n=%-6zu min %+.9g max %+.9g sum %+.9g", name.c_str(), got.size(), mn, mx, sm);

    const std::string rp = dir + "/" + ref;
    FILE* f = std::fopen(rp.c_str(), "rb");
    if (!f) { std::printf("   (no %s)\n", ref.c_str()); return 0; }
    std::vector<float> want(got.size());
    const size_t got_n = std::fread(want.data(), sizeof(float), want.size(), f);
    std::fclose(f);
    if (got_n != want.size()) {
        std::printf("   (%s is %zu floats, not %zu)\n", ref.c_str(), got_n, got.size());
        return 0;
    }
    double worst = 0.0, scale = 1e-30;
    for (float v : want) scale = std::max(scale, (double) std::fabs(v));
    for (size_t i = 0; i < got.size(); ++i) worst = std::max(worst, std::fabs((double) got[i] - (double) want[i]));
    const double rel = worst / scale;
    const bool pass = rel < 1e-4;
    std::printf("   vs %-22s rel %.3e  %s\n", ref.c_str(), rel, pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/peb/moredata/glm5-kda-l0-t1";
    const int tokens = argc > 2 ? std::atoi(argv[2]) : 1;
    const int n_embd = 4096, nh = 64, hd = 128, d_conv = 4, d_inner = nh * hd;

    strata::kernels::glm::KdaGeometry g;
    g.n_embd = n_embd; g.nh = nh; g.hd = hd; g.d_conv = d_conv;

    // KdaWeights holds only pointers, so every vector must outlive the call - they live here, not in the struct.
    struct Store {
        std::vector<float> attn_norm, wq, wk, wv, conv_q, conv_k, conv_v, ssm_a, dt_bias,
                           ssm_f_a, ssm_f_b, ssm_beta, ssm_g_a, ssm_g_b, o_norm, wo;
    } s;

    s.attn_norm = read_bin(dir + "/w_attn_norm.bin", (size_t) n_embd);
    s.wq        = read_bin(dir + "/w_wq.bin",        (size_t) d_inner * n_embd);
    s.wk        = read_bin(dir + "/w_wk.bin",        (size_t) d_inner * n_embd);
    s.wv        = read_bin(dir + "/w_wv.bin",        (size_t) d_inner * n_embd);
    s.conv_q    = read_bin(dir + "/w_conv_q.bin",    (size_t) d_conv * d_inner);
    s.conv_k    = read_bin(dir + "/w_conv_k.bin",    (size_t) d_conv * d_inner);
    s.conv_v    = read_bin(dir + "/w_conv_v.bin",    (size_t) d_conv * d_inner);
    s.ssm_a     = read_bin(dir + "/w_ssm_a.bin",     (size_t) nh);
    s.dt_bias   = read_bin(dir + "/w_dt_bias.bin",   (size_t) d_inner);
    s.ssm_f_a   = read_bin(dir + "/w_ssm_f_a.bin",   (size_t) hd * n_embd);
    s.ssm_f_b   = read_bin(dir + "/w_ssm_f_b.bin",   (size_t) d_inner * hd);
    s.ssm_beta  = read_bin(dir + "/w_ssm_beta.bin",  (size_t) nh * n_embd);
    s.ssm_g_a   = read_bin(dir + "/w_ssm_g_a.bin",   (size_t) hd * n_embd);
    s.ssm_g_b   = read_bin(dir + "/w_ssm_g_b.bin",   (size_t) d_inner * hd);
    s.o_norm    = read_bin(dir + "/w_o_norm.bin",    (size_t) hd);
    s.wo        = read_bin(dir + "/w_wo.bin",        (size_t) n_embd * d_inner);

    strata::kernels::glm::KdaWeights w;
    w.attn_norm = s.attn_norm.data(); w.wq = s.wq.data(); w.wk = s.wk.data(); w.wv = s.wv.data();
    w.conv_q = s.conv_q.data(); w.conv_k = s.conv_k.data(); w.conv_v = s.conv_v.data();
    w.ssm_a = s.ssm_a.data(); w.dt_bias = s.dt_bias.data();
    w.ssm_f_a = s.ssm_f_a.data(); w.ssm_f_b = s.ssm_f_b.data();
    w.ssm_beta = s.ssm_beta.data();
    w.ssm_g_a = s.ssm_g_a.data(); w.ssm_g_b = s.ssm_g_b.data();
    w.o_norm = s.o_norm.data(); w.wo = s.wo.data();

    const size_t tok_nh_hd = (size_t) tokens * nh * hd;
    const std::vector<float> x = read_bin(dir + "/input.bin", (size_t) tokens * n_embd);
    std::vector<float> out((size_t) tokens * n_embd, 0.f);
    std::vector<float> state((size_t) nh * hd * hd, 0.f);                         // fresh state: the S matrix

    std::vector<float> mg(tok_nh_hd), mbeta((size_t) tokens * nh), mq(tok_nh_hd), mk(tok_nh_hd),
                       mv(tok_nh_hd), mattn(tok_nh_hd), mo((size_t) tokens * d_inner),
                       mxn((size_t) tokens * n_embd), mqc(tok_nh_hd), mkc(tok_nh_hd), mvc(tok_nh_hd);
    strata::kernels::glm::KdaIntermediates mid;
    mid.g = mg.data(); mid.beta = mbeta.data(); mid.q = mq.data(); mid.k = mk.data();
    mid.v = mv.data(); mid.attn = mattn.data(); mid.o = mo.data();
    mid.xn = mxn.data(); mid.qc = mqc.data(); mid.kc = mkc.data(); mid.vc = mvc.data();

    strata::kernels::glm::kda_forward(w, g, x.data(), tokens, out.data(), state.data(), &mid);

    std::printf("kda gate: %s, %d token(s), weights and input from the oracle's fixture\n", dir.c_str(), tokens);
    int failures = 0;
    if (tokens > 1) {
        std::vector<float> split_out((size_t) tokens * n_embd, 0.0f);
        std::vector<float> split_state((size_t) nh * hd * hd, 0.0f);
        std::vector<float> split_conv((size_t) 3 * (d_conv - 1) * d_inner, 0.0f);
        for (int t = 0; t < tokens; ++t) {
            strata::kernels::glm::kda_forward(w, g, x.data() + (size_t) t * n_embd, 1,
                                               split_out.data() + (size_t) t * n_embd,
                                               split_state.data(), nullptr, split_conv.data());
        }
        double worst = 0.0, scale = 1e-30;
        for (size_t i = 0; i < out.size(); ++i) {
            worst = std::max(worst, std::fabs((double) split_out[i] - (double) out[i]));
            scale = std::max(scale, std::fabs((double) out[i]));
        }
        const double rel = worst / scale;
        const bool pass = rel < 1e-6;
        std::printf("  split    %d one-token calls vs one %d-token call rel %.3e  %s\n",
                    tokens, tokens, rel, pass ? "PASS" : "FAIL");
        failures += pass ? 0 : 1;
    }
    failures += report("result", out,   dir, "result.bin");
    failures += report("xn",     mxn,   dir, "inter_xn.bin");
    failures += report("qc",     mqc,   dir, "inter_qc.bin");
    failures += report("kc",     mkc,   dir, "inter_kc.bin");
    failures += report("vc",     mvc,   dir, "inter_vc.bin");
    failures += report("g",      mg,    dir, "inter_g.bin");
    failures += report("beta",   mbeta, dir, "inter_beta.bin");
    failures += report("q",      mq,    dir, "inter_q.bin");
    failures += report("k",      mk,    dir, "inter_k.bin");
    failures += report("v",      mv,    dir, "inter_v.bin");
    failures += report("attn",   mattn, dir, "inter_attn.bin");
    failures += report("o",      mo,    dir, "inter_o.bin");
    std::printf("KDA GATE: %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
