// src/kernels/glm_kda.cpp - the KDA block operator (branch glm5next-port).  Host code, float, no CUDA:
// correctness against the reference first, like the mHC and MLA.  The math and the traps are quoted in
// the header; the oracle is tools/glm5_kda_reference.py and src/kernels/glm_kda_parity.cpp compares.
#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_mla.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels::glm {

namespace {

KdaNativeProjectFn g_native_project = nullptr;
bool g_device_recurrence = false;
bool g_device_gates = false;

/// WHERE THE KDA STAGE'S HOST TIME GOES, phase by phase.  The blocks are 69.9% of the wall time and their projections are
/// already native, so the remainder is the non-GEMM work - and there are four distinct candidates in here (the input
/// norm, the conv, the gates, the delta-rule recurrence with its fp64 state copies).  Timing them costs a handful of
/// chrono calls and says which one to write a kernel for, which is cheaper than reasoning about it and has been right
/// far more often in this port.
double g_kda_ms[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
long g_kda_layers = 0;

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

KdaParallelFor g_parallel_for = nullptr;

/// Runs f(0..n-1) on the installed worker pool, or serially when none is set (or there is a single job).
void pfor(int n, const std::function<void(int)>& f) {
    if (g_parallel_for != nullptr && n > 1) g_parallel_for(n, f);
    else for (int i = 0; i < n; ++i) f(i);
}

/// A matrix-times-vector with the weight's rows contiguous, which is how every projection in the block
/// is shaped (ggml's mul_mat reduces over the weight's ne0, i.e. over the row's elements).
inline void matvec(const float* w, const float* x, float* out, int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        const float* row = w + (size_t) r * cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; ++c) acc += row[c] * x[c];
        out[r] = acc;
    }
}

/// Depthwise causal conv over the sequence, kernel `d_conv`, applied in FORWARD order over
/// [state(d_conv-1) | tokens]: out[t][ch] = sum_k w[k][ch] * x_conv[t+k][ch].  The state is the previous
/// d_conv-1 inputs per channel and is zero for a fresh sequence (the reference concatenates it as
/// ggml_concat(conv_state, transpose(x_proj)) and then runs ggml_ssm_conv).
void conv1d_silu(const float* conv_w, const float* proj /* [tokens][d_inner] */, float* out, int tokens,
                 int d_inner, int d_conv, float* history) {
    // Every output row reads only the (unchanged) projection and the history snapshot, so token blocks are independent;
    // the history update below runs after all of them.
    constexpr int kBlock = 16;
    pfor((tokens + kBlock - 1) / kBlock, [&](int b) {
        const int t_end = std::min(tokens, (b + 1) * kBlock);
        for (int t = b * kBlock; t < t_end; ++t) {
            for (int ch = 0; ch < d_inner; ++ch) {
                float acc = 0.0f;
                for (int k = 0; k < d_conv; ++k) {
                    const int src = t + k - (d_conv - 1);
                    const float v = src >= 0 ? proj[(size_t) src * d_inner + ch]
                                             : (history ? history[(size_t) (d_conv - 1 + src) * d_inner + ch] : 0.0f);
                    if (src < tokens) acc += conv_w[(size_t) k * d_inner + ch] * v;
                }
                out[(size_t) t * d_inner + ch] = acc / (1.0f + std::exp(-acc));
            }
        }
    });
    if (history) {
        for (int h = 0; h < d_conv - 1; ++h) {
            const int src = tokens - (d_conv - 1) + h;
            if (src >= 0) {
                std::memcpy(history + (size_t) h * d_inner, proj + (size_t) src * d_inner,
                            (size_t) d_inner * sizeof(float));
            } else {
                std::memmove(history + (size_t) h * d_inner,
                             history + (size_t) (d_conv - 1 + src) * d_inner,
                             (size_t) d_inner * sizeof(float));
            }
        }
    }
}

}  // namespace

void kda_set_native_project(KdaNativeProjectFn fn) { g_native_project = fn; }
void kda_set_device_recurrence(bool enabled) { g_device_recurrence = enabled; }
void kda_set_device_gates(bool enabled) { g_device_gates = enabled; }
void kda_set_parallel_for(KdaParallelFor fn) { g_parallel_for = fn; }

#if !defined(STRATA_ENABLE_CUDA)
bool kda_recurrence_cuda(const float*, const float*, const float*, const float*, const float*, int, int, int,
                         float*, float*, char* error, size_t error_capacity) {
    if (error && error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
void kda_set_lazy_state(bool) {}
void kda_invalidate_state() {}
bool kda_rows_cuda(const float*, const float*, int, int, int, float*, char* error, size_t error_capacity) {
    if (error && error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
bool kda_gates_cuda(const float*, const float*, const float*, const float*, const float*, const float*, int, int, int, int,
                    float*, float*, char* error, size_t error_capacity) {
    if (error && error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
#endif

void kda_forward(const KdaWeights& w, const KdaGeometry& g, const float* x, int tokens, float* out,
                 float* state, const KdaIntermediates* mid, float* conv_state) {
    using Clock = std::chrono::steady_clock;
    auto mark = Clock::now();
    auto lap = [&]() { auto n = Clock::now(); double ms = std::chrono::duration<double, std::milli>(n-mark).count(); mark=n; return ms; };
    const int ne = g.n_embd, nh = g.nh, hd = g.hd, di = g.d_inner();
    const float scale = 1.0f / std::sqrt((float) hd);

    std::vector<float> xn((size_t) tokens * ne);
    {
        auto tick = [] { return std::chrono::steady_clock::now(); };
        auto since = [](const std::chrono::steady_clock::time_point& a) {
            return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        };
        (void) tick; (void) since;
    }
    ++g_kda_layers;
    const auto kd_t = [] { return std::chrono::steady_clock::now(); };
    const auto kd_since = [](const std::chrono::steady_clock::time_point& a) {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
    };
    auto kd_t0 = kd_t();
    for (int t = 0; t < tokens; ++t) {
        const float* xt = x + (size_t) t * ne;
        double ss = 0.0;
        for (int i = 0; i < ne; ++i) ss += (double) xt[i] * (double) xt[i];
        const float inv = 1.0f / std::sqrt((float) (ss / ne) + KDA_RMS_EPS);
        for (int i = 0; i < ne; ++i) xn[(size_t) t * ne + i] = xt[i] * inv * w.attn_norm[i];
    }
    if (mid && mid->xn) std::memcpy(mid->xn, xn.data(), xn.size() * sizeof(float));
    const double tm_norm = lap();

    g_kda_ms[0] += kd_since(kd_t0);            // the input norm
    auto kd_t1 = kd_t();
    // q, k, v: project, then a separate causal conv each, then SiLU.  The conv reads earlier rows, so the
    // raw projection and the convolved output must be different buffers (run this in place and iteration
    // t+1 would read rows that iteration t already overwrote).
    std::vector<float> qc((size_t) tokens * di), kc((size_t) tokens * di), vc((size_t) tokens * di);
    std::vector<float> raw_q((size_t) tokens * di), raw_k((size_t) tokens * di), raw_v((size_t) tokens * di);
    const float* conv_w[3] = {w.conv_q, w.conv_k, w.conv_v};
    const float* proj_w[3] = {w.wq, w.wk, w.wv};
    const int proj_type[3] = {w.wq_type, w.wk_type, w.wv_type};
    float* conv_dst[3] = {qc.data(), kc.data(), vc.data()};
    float* raw_dst[3] = {raw_q.data(), raw_k.data(), raw_v.data()};
    if (g_native_project && proj_type[0] && proj_type[1] && proj_type[2]) {
        const void* native_w[3] = {proj_w[0], proj_w[1], proj_w[2]};
        if (!g_native_project(3, native_w, proj_type, tokens, xn.data(), ne, di, raw_dst)) {
            std::fprintf(stderr, "GLM KDA native QKV projection failed\n");
            std::exit(1);
        }
    } else {
        for (int which = 0; which < 3; ++which)
            for (int t = 0; t < tokens; ++t)
                matvec(proj_w[which], xn.data() + (size_t) t * ne,
                       raw_dst[which] + (size_t) t * di, di, ne);
    }
    for (int which = 0; which < 3; ++which) {
        float* history = conv_state ? conv_state + (size_t) which * (g.d_conv - 1) * di : nullptr;
        conv1d_silu(conv_w[which], raw_dst[which], conv_dst[which], tokens, di, g.d_conv, history);
    }
    if (mid && mid->qc) std::memcpy(mid->qc, qc.data(), qc.size() * sizeof(float));
    if (mid && mid->kc) std::memcpy(mid->kc, kc.data(), kc.size() * sizeof(float));
    if (mid && mid->vc) std::memcpy(mid->vc, vc.data(), vc.size() * sizeof(float));
    const double tm_qkv = lap();

    g_kda_ms[1] += kd_since(kd_t1);            // projections, three convs, three SiLUs
    auto kd_t2 = kd_t();
    // the forget gate: per (head, channel), in [KDA_GATE_LOWER, 0].  ssm_a holds -exp(A_log), so the
    // reference's `sigmoid(-(ssm_a * (...)))` is sigmoid(exp(A_log) * (...)), and the magnitude of the
    // exponent then multiplies the gate's lower bound.
    std::vector<float> gv((size_t) tokens * nh * hd), beta((size_t) tokens * nh);
    bool gates_done = false;
    if (g_device_gates) {
        char cuda_err[256] = {};
        gates_done = kda_gates_cuda(xn.data(),w.ssm_f_a,w.ssm_f_b,w.ssm_beta,w.ssm_a,w.dt_bias,
                                    tokens,ne,nh,hd,gv.data(),beta.data(),cuda_err,sizeof(cuda_err));
        if (!gates_done) std::fprintf(stderr,"GLM KDA CUDA gates unavailable: %s; using host gates\n",cuda_err);
    }
    if (!gates_done) {
        std::vector<float> fa((size_t) hd), fb((size_t) di);
        for (int t = 0; t < tokens; ++t) {
            const float* xt = xn.data() + (size_t) t * ne;
            matvec(w.ssm_f_a, xt, fa.data(), hd, ne);
            matvec(w.ssm_f_b, fa.data(), fb.data(), di, hd);
            for (int h = 0; h < nh; ++h) {
                for (int i = 0; i < hd; ++i) {
                    const float v = fb[(size_t) h * hd + i] + w.dt_bias[(size_t) h * hd + i];
                    gv[((size_t) t * nh + h) * hd + i] = KDA_GATE_LOWER * sigmoid(-(w.ssm_a[h] * v));
                }
                const float* b_row = w.ssm_beta + (size_t) h * ne;
                float acc = 0.0f;
                for (int c = 0; c < ne; ++c) acc += b_row[c] * xt[c];
                beta[(size_t) t * nh + h] = sigmoid(acc);
            }
        }
    }
    if (mid && mid->g) std::memcpy(mid->g, gv.data(), gv.size() * sizeof(float));
    if (mid && mid->beta) std::memcpy(mid->beta, beta.data(), beta.size() * sizeof(float));
    const double tm_gates = lap();

    g_kda_ms[2] += kd_since(kd_t2);            // the gates (beta, f, g)
    auto kd_t3 = kd_t();
    // the recurrence, per head.  Decay first (S[i][:] *= exp(g[i]) - the gate index is S's FIRST index),
    // then the delta-rule correction, then the read, which happens AFTER the update.
    std::vector<double> S;
    std::vector<double> dec((size_t) hd), delta((size_t) hd), qn((size_t) hd), kn((size_t) hd);
    std::vector<float> attn((size_t) tokens * nh * hd);
    std::vector<float> qnorm((size_t) tokens * di), knorm((size_t) tokens * di);
    bool device_done = false;
    if (g_device_recurrence) {
        for (int t = 0; t < tokens; ++t) for (int h = 0; h < nh; ++h) {
            const float* qh = qc.data() + (size_t)t * di + (size_t)h * hd;
            const float* kh = kc.data() + (size_t)t * di + (size_t)h * hd;
            double nq=0.0, nk=0.0;
            for (int i=0;i<hd;++i) { nq+=(double)qh[i]*qh[i]; nk+=(double)kh[i]*kh[i]; }
            const double iq=1.0/std::max(std::sqrt(nq),(double)KDA_L2_EPS);
            const double ik=1.0/std::max(std::sqrt(nk),(double)KDA_L2_EPS);
            for (int i=0;i<hd;++i) {
                qnorm[(size_t)t*di+(size_t)h*hd+i]=(float)((double)qh[i]*iq);
                knorm[(size_t)t*di+(size_t)h*hd+i]=(float)((double)kh[i]*ik);
            }
        }
        char cuda_err[256] = {};
        device_done = kda_recurrence_cuda(qnorm.data(), knorm.data(), vc.data(), gv.data(), beta.data(),
                                           tokens, nh, hd, state, attn.data(), cuda_err, sizeof(cuda_err));
        if (!device_done) std::fprintf(stderr, "GLM KDA CUDA recurrence unavailable: %s; using host recurrence\n", cuda_err);
    }
    // The device path owns the recurrence state and returns it to the host for reset/checkpoint semantics. Avoid
    // constructing and converting the full double-precision host state on every decode step when CUDA succeeded.
    if (!device_done) {
        S.assign((size_t) nh * hd * hd, 0.0);
        if (state)
            for (size_t i = 0; i < S.size(); ++i) S[i] = (double) state[i];
    }
    for (int t = 0; t < tokens; ++t) {
        for (int h = 0; h < nh; ++h) {
            const float* gth = gv.data() + ((size_t) t * nh + h) * hd;
            const float* qh = qc.data() + (size_t) t * di + (size_t) h * hd;
            const float* kh = kc.data() + (size_t) t * di + (size_t) h * hd;
            const float* vh = vc.data() + (size_t) t * di + (size_t) h * hd;
            const float bt = beta[(size_t) t * nh + h];

            double nq = 0.0, nk = 0.0;
            for (int i = 0; i < hd; ++i) {
                nq += (double) qh[i] * (double) qh[i];
                nk += (double) kh[i] * (double) kh[i];
            }
            const double iq = 1.0 / std::max(std::sqrt(nq), (double) KDA_L2_EPS);
            const double ik = 1.0 / std::max(std::sqrt(nk), (double) KDA_L2_EPS);
            for (int i = 0; i < hd; ++i) {
                qn[(size_t) i] = (double) qh[i] * iq;
                kn[(size_t) i] = (double) kh[i] * ik;
                dec[(size_t) i] = std::exp((double) gth[i]);
            }
            if (device_done) {
                if (mid && mid->q) for (int i=0;i<hd;++i) mid->q[((size_t)t*nh+h)*hd+i]=qnorm[(size_t)t*di+(size_t)h*hd+i];
                if (mid && mid->k) for (int i=0;i<hd;++i) mid->k[((size_t)t*nh+h)*hd+i]=knorm[(size_t)t*di+(size_t)h*hd+i];
                if (mid && mid->v) std::memcpy(mid->v+((size_t)t*nh+h)*hd,vh,(size_t)hd*sizeof(float));
                continue;
            }
            if (mid && mid->q) for (int i = 0; i < hd; ++i) mid->q[((size_t) t * nh + h) * hd + i] = (float) qn[(size_t) i];
            if (mid && mid->k) for (int i = 0; i < hd; ++i) mid->k[((size_t) t * nh + h) * hd + i] = (float) kn[(size_t) i];
            if (mid && mid->v) std::memcpy(mid->v + ((size_t) t * nh + h) * hd, vh, (size_t) hd * sizeof(float));

            double* S_h = S.data() + (size_t) h * hd * hd;
            // S[i][j] *= exp(g[i]) for every j
            for (int i = 0; i < hd; ++i) {
                const double d = dec[(size_t) i];
                for (int j = 0; j < hd; ++j) S_h[(size_t) i * hd + j] *= d;
            }
            // delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta, then S[i][j] += k[i]*delta[j], then the read.
            // The state's FIRST index is the KEY axis - it pairs with k in the prediction and with q in the read
            // - so these loops sum over i and are indexed by j.  Getting this backwards agrees at the first
            // token (S = 0, so both conventions collapse to delta*(k.q)) and diverges from the second on, which
            // is exactly how the parity gate passed on 1 token while failing on 3 and 5.
            for (int j = 0; j < hd; ++j) {
                double acc = 0.0;
                for (int i = 0; i < hd; ++i) acc += S_h[(size_t) i * hd + j] * kn[(size_t) i];
                delta[(size_t) j] = ((double) vh[j] - acc) * (double) bt;
            }
            // S[i][j] += k[i] * delta[j]
            for (int i = 0; i < hd; ++i) {
                const double kv = kn[(size_t) i];
                for (int j = 0; j < hd; ++j) S_h[(size_t) i * hd + j] += kv * delta[(size_t) j];
            }
            // attn[j] = (sum_i S[i][j]*q[i]) * scale, read AFTER the update
            for (int j = 0; j < hd; ++j) {
                double acc = 0.0;
                for (int i = 0; i < hd; ++i) acc += S_h[(size_t) i * hd + j] * qn[(size_t) i];
                attn[((size_t) t * nh + h) * hd + j] = (float) (acc * scale);
            }
        }
    }
    if (state && !device_done)
        for (size_t i = 0; i < S.size(); ++i) state[i] = (float) S[i];
    if (mid && mid->attn) std::memcpy(mid->attn, attn.data(), attn.size() * sizeof(float));
    const double tm_rec = lap();

    // the gated norm: rms over ne0 = head_dim, per head, then a SIGMOID gate (not SiLU), then wo
    std::vector<float> o((size_t) tokens * di);
    // The two gate projections for ALL tokens in one resident-weight GPU call each (the per-token version was two
    // synchronous launches per token per layer); the host path below is the fallback.
    std::vector<float> ga_all, gb_all;
    bool gate_dev = false;
    if (g_device_gates) {
        char cuda_err[256] = {};
        ga_all.resize((size_t) tokens * hd);
        gb_all.resize((size_t) tokens * di);
        gate_dev = kda_rows_cuda(w.ssm_g_a, xn.data(), tokens, hd, ne, ga_all.data(), cuda_err, sizeof(cuda_err)) &&
                   kda_rows_cuda(w.ssm_g_b, ga_all.data(), tokens, di, hd, gb_all.data(), cuda_err, sizeof(cuda_err));
    }
    constexpr int kGateBlock = 16;
    pfor((tokens + kGateBlock - 1) / kGateBlock, [&](int b) {
        std::vector<float> ga((size_t) hd), gb_host((size_t) di);
        const int t_end = std::min(tokens, (b + 1) * kGateBlock);
        for (int t = b * kGateBlock; t < t_end; ++t) {
            const float* gb = nullptr;
            if (gate_dev) {
                gb = gb_all.data() + (size_t) t * di;
            } else {
                const float* xt = xn.data() + (size_t) t * ne;
                matvec(w.ssm_g_a, xt, ga.data(), hd, ne);
                matvec(w.ssm_g_b, ga.data(), gb_host.data(), di, hd);
                gb = gb_host.data();
            }
            float* og = o.data() + (size_t) t * di;
            for (int h = 0; h < nh; ++h) {
                const float* at = attn.data() + ((size_t) t * nh + h) * hd;
                double ss = 0.0;
                for (int i = 0; i < hd; ++i) ss += (double) at[i] * (double) at[i];
                const float inv = 1.0f / std::sqrt((float) (ss / hd) + KDA_RMS_EPS);
                for (int i = 0; i < hd; ++i) {
                    const int c = h * hd + i;
                    og[c] = at[i] * inv * w.o_norm[i] * sigmoid(gb[c]);
                }
            }
        }
    });
    if (mid && mid->o) std::memcpy(mid->o, o.data(), o.size() * sizeof(float));
    const double tm_gateout = lap();

    g_kda_ms[3] += kd_since(kd_t3);            // the delta-rule recurrence and its state traffic
    auto kd_t4 = kd_t();
    if (g_native_project && w.wo_type) {
        const void* native_w[1] = {w.wo};
        const int native_type[1] = {w.wo_type};
        float* native_out[1] = {out};
        if (!g_native_project(1, native_w, native_type, tokens, o.data(), di, ne, native_out)) {
            std::fprintf(stderr, "GLM KDA native output projection failed\n");
            std::exit(1);
        }
    } else {
        for (int t = 0; t < tokens; ++t)
            matvec(w.wo, o.data() + (size_t) t * di, out + (size_t) t * ne, ne, di);
    }
    const double tm_wo = lap();
    if (std::getenv("STRATA_GLM_TIMING"))
        std::fprintf(stderr, "KDA_TIMING norm=%.3f qkv=%.3f gates=%.3f rec=%.3f outgate=%.3f wo=%.3f\n",
                     tm_norm, tm_qkv, tm_gates, tm_rec, tm_gateout, tm_wo);
    g_kda_ms[4] += kd_since(kd_t4);            // the output norm and wo
}

void kda_print_profile() {
    const double total = g_kda_ms[0] + g_kda_ms[1] + g_kda_ms[2] + g_kda_ms[3] + g_kda_ms[4];
    if (total <= 0.0) return;
    const char* nm[5] = {"attn_norm", "proj+conv+silu", "gates", "recurrence", "out_norm+wo"};
    std::fprintf(stderr, "KDA PROFILE over %ld layer calls, %.1f ms in these phases\n", g_kda_layers, total);
    for (int i = 0; i < 5; ++i)
        std::fprintf(stderr, "  %-16s %9.1f ms  %5.1f%%\n", nm[i], g_kda_ms[i], 100.0 * g_kda_ms[i] / total);
}

}  // namespace strata::kernels::glm
