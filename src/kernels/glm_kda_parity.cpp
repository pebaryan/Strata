// src/kernels/glm_kda_parity.cpp - does our KDA block match the reference?
//
// The oracle is tools/glm5_kda_reference.py, transcribed from build_kda_layer, glm5next_causal_conv1d and
// the fused op's CPU reference:
//
//   python tools/glm5_kda_reference.py --gguf <shard1> --layer 4 --tokens 8 --raw-fixture /tmp/kda.bin
//   build-volta/glm_kda_parity /tmp/kda.bin
//
// Every stage is compared, so a mismatch localizes: the input norm, the three convs (+SiLU), the gate,
// beta, the L2-normalised q and k, the recurrence's output, the gated norm, and the projection.
#include "strata/kernels/glm_kda.hpp"

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

/// Scale-relative: float32 round-off is proportional to the array's magnitude, and this block mixes
/// values that differ by orders of magnitude (a normalised q against an unnormalised projection).
void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, bool& ok,
             double scale_tol = 1e-4, double elem_tol = 1e-3) {
    double scale = 1e-30, ma = 0.0, mr = 0.0;
    for (float x : want) scale = std::max(scale, std::fabs((double) x));
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        if (std::fabs((double) want[i]) >= 0.1 * scale) mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const bool good = ma / scale < scale_tol && mr < elem_tol;
    std::printf("  %-8s max abs %.3e (= %.2e of scale)   worst element rel %.3e   %s\n", what, ma, ma / scale,
                mr, good ? "PASS" : "FAIL");
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_kda_parity <fixture.bin>   (tools/glm5_kda_reference.py --raw-fixture)\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int32_t hdr[6] = {0};
    if (std::fread(hdr, sizeof(int32_t), 6, f) != 6) { std::fprintf(stderr, "bad header\n"); return 1; }

    glm::KdaGeometry g;
    g.n_embd = hdr[0];
    g.nh = hdr[1];
    g.hd = hdr[2];
    g.d_conv = hdr[3];
    const int tokens = hdr[4];
    const int ne = g.n_embd, nh = g.nh, hd = g.hd, di = g.d_inner();

    std::vector<float> attn_norm = read_floats(f, (size_t) ne);
    std::vector<float> wq = read_floats(f, (size_t) di * ne);
    std::vector<float> wk = read_floats(f, (size_t) di * ne);
    std::vector<float> wv = read_floats(f, (size_t) di * ne);
    std::vector<float> conv_q = read_floats(f, (size_t) g.d_conv * di);
    std::vector<float> conv_k = read_floats(f, (size_t) g.d_conv * di);
    std::vector<float> conv_v = read_floats(f, (size_t) g.d_conv * di);
    std::vector<float> ssm_a = read_floats(f, (size_t) nh);
    std::vector<float> dt_bias = read_floats(f, (size_t) di);
    std::vector<float> ssm_f_a = read_floats(f, (size_t) hd * ne);
    std::vector<float> ssm_f_b = read_floats(f, (size_t) di * hd);
    std::vector<float> ssm_beta = read_floats(f, (size_t) nh * ne);
    std::vector<float> ssm_g_a = read_floats(f, (size_t) hd * ne);
    std::vector<float> ssm_g_b = read_floats(f, (size_t) di * hd);
    std::vector<float> o_norm = read_floats(f, (size_t) hd);
    std::vector<float> wo = read_floats(f, (size_t) ne * di);
    std::vector<float> x = read_floats(f, (size_t) tokens * ne);

    const std::vector<float> e_qc = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_kc = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_vc = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_g = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_beta = read_floats(f, (size_t) tokens * nh);
    const std::vector<float> e_q = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_k = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_v = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_attn = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_o = read_floats(f, (size_t) tokens * di);
    const std::vector<float> e_result = read_floats(f, (size_t) tokens * ne);
    std::fclose(f);

    glm::KdaWeights w;
    w.attn_norm = attn_norm.data();
    w.wq = wq.data();
    w.wk = wk.data();
    w.wv = wv.data();
    w.conv_q = conv_q.data();
    w.conv_k = conv_k.data();
    w.conv_v = conv_v.data();
    w.ssm_a = ssm_a.data();
    w.dt_bias = dt_bias.data();
    w.ssm_f_a = ssm_f_a.data();
    w.ssm_f_b = ssm_f_b.data();
    w.ssm_beta = ssm_beta.data();
    w.ssm_g_a = ssm_g_a.data();
    w.ssm_g_b = ssm_g_b.data();
    w.o_norm = o_norm.data();
    w.wo = wo.data();

    std::printf("KDA parity vs tools/glm5_kda_reference.py: layer %d, %d tokens, %d heads x %d, conv %d\n",
                hdr[5], tokens, nh, hd, g.d_conv);

    std::vector<float> got_xn((size_t) tokens * ne), got_qc((size_t) tokens * di), got_kc((size_t) tokens * di);
    std::vector<float> got_vc((size_t) tokens * di), got_g((size_t) tokens * di);
    std::vector<float> got_beta((size_t) tokens * nh), got_q((size_t) tokens * di), got_k((size_t) tokens * di);
    std::vector<float> got_v((size_t) tokens * di), got_attn((size_t) tokens * di), got_o((size_t) tokens * di);
    std::vector<float> got_result((size_t) tokens * ne), state((size_t) nh * hd * hd, 0.0f);
    glm::KdaIntermediates mid;
    mid.xn = got_xn.data();
    mid.qc = got_qc.data();
    mid.kc = got_kc.data();
    mid.vc = got_vc.data();
    mid.g = got_g.data();
    mid.beta = got_beta.data();
    mid.q = got_q.data();
    mid.k = got_k.data();
    mid.v = got_v.data();
    mid.attn = got_attn.data();
    mid.o = got_o.data();
    glm::kda_forward(w, g, x.data(), tokens, got_result.data(), state.data(), &mid);

    bool ok = true;
    compare("conv q", got_qc, e_qc, ok);
    compare("conv k", got_kc, e_kc, ok);
    compare("conv v", got_vc, e_vc, ok);
    compare("gate", got_g, e_g, ok);
    compare("beta", got_beta, e_beta, ok);
    compare("q norm", got_q, e_q, ok);
    compare("k norm", got_k, e_k, ok);
    compare("v", got_v, e_v, ok);
    compare("attn", got_attn, e_attn, ok);
    compare("gated", got_o, e_o, ok);
    compare("result", got_result, e_result, ok);

    std::vector<float> device_attn((size_t)tokens*di), device_state((size_t)nh*hd*hd,0.0f);
    char device_error[256] = {};
    const bool device_ok = glm::kda_recurrence_cuda(got_q.data(),got_k.data(),got_v.data(),got_g.data(),got_beta.data(),
                                                     tokens,nh,hd,device_state.data(),device_attn.data(),
                                                     device_error,sizeof(device_error));
    if (device_ok) {
        compare("CUDA attn",device_attn,e_attn,ok,3e-4,2e-3);
        compare("CUDA/CPU S",device_state,state,ok,3e-4,2e-3);
    } else {
        std::printf("  CUDA recurrence unavailable: %s\n",device_error);
        ok=false;
    }

    // structural invariants that do not depend on the weights agreeing with the oracle
    {
        double gmin = 1e30, gmax = -1e30, qn_mean = 0.0;
        for (float v : got_g) { gmin = std::min(gmin, (double) v); gmax = std::max(gmax, (double) v); }
        for (int t = 0; t < tokens; ++t)
            for (int h = 0; h < nh; ++h) {
                double s = 0.0;
                for (int i = 0; i < hd; ++i) s += (double) got_q[((size_t) t * nh + h) * hd + i] *
                                                  (double) got_q[((size_t) t * nh + h) * hd + i];
                qn_mean += std::sqrt(s);
            }
        qn_mean /= (double) tokens * nh;
        double state_norm = 0.0;
        for (float v : state) state_norm += (double) v * v;
        const bool gok = gmin >= (double) glm::KDA_GATE_LOWER - 1e-6 && gmax <= 0.0;
        const bool nok = std::fabs(qn_mean - 1.0) < 1e-4;
        std::printf("  invariant  gate in [%.1f, 0]: [%.4f, %.4f] %s   |q| per head %.6f %s   "
                    "|state| %.4f\n",
                    (double) glm::KDA_GATE_LOWER, gmin, gmax, gok ? "PASS" : "FAIL", qn_mean,
                    nok ? "PASS" : "FAIL", std::sqrt(state_norm));
        ok = ok && gok && nok;
    }

    std::printf("glm_kda_parity: %s\n", ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
