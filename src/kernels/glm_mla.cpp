// src/kernels/glm_mla.cpp - the MLA block (branch glm5next-port).  Host code, float, no CUDA, for the same
// reason as the mHC: the architecture has to be exactly right against a reference before any of it moves
// onto the device.  The graph is quoted in the header; the oracle lives in tools/glm5_mla_reference.py.
#include "strata/kernels/glm_mla.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels::glm {
namespace {

MlaNativeProjectFn g_native_project = nullptr;
bool g_device_attention = false;
// The V100 launch/copy overhead exceeds the host loop at short cache depths. Keep the measured default, while
// allowing a controlled A/B threshold without rebuilding the persistent serving process.
int mla_cuda_min_cache() {
    static const int value = [] {
        const char* env = std::getenv("STRATA_GLM_MLA_MIN_CACHE");
        if (!env || !*env) return 256;
        char* end = nullptr;
        const long parsed = std::strtol(env, &end, 10);
        if (end == env || *end != '\0') return 256;
        return (int) std::max(1L, std::min(8192L, parsed));
    }();
    return value;
}

/// x -> rms_norm(x) * weight, over the whole row (the reference normalizes ne0 and scales by the weight).
void rms_norm_inplace(float* x, int n, const float* weight) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * (double) x[i];
    const float inv = 1.0f / std::sqrt((float) (ss / n) + MLA_RMS_EPS);
    for (int i = 0; i < n; ++i) x[i] = x[i] * inv * (weight ? weight[i] : 1.0f);
}

}  // namespace

MlaNativeProjectBatchFn g_native_project_batch = nullptr;
MlaParallelFor g_parallel_for = nullptr;

void mla_set_native_project(MlaNativeProjectFn fn) { g_native_project = fn; }
void mla_set_device_attention(bool enabled) { g_device_attention = enabled; }
void mla_set_native_project_batch(MlaNativeProjectBatchFn fn) { g_native_project_batch = fn; }
void mla_set_parallel_for(MlaParallelFor fn) { g_parallel_for = fn; }

#if !defined(STRATA_ENABLE_CUDA)
bool mla_attention_cuda(const float*,const float*,int,int,int,int,float*,char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
bool mla_head_matvec_cuda(const float*,const float*,int,int,int,float*,char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
bool mla_attend_batch_cuda(const float*,const float*,const float*,const float*,int,int,int,int,int,float*,
                           char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
#endif

bool mla_forward_batch(const MlaWeights& w, const MlaGeometry& g, const float* x, int T, int c0, float* cache, float* out) {
    if (!g_native_project_batch || !g_device_attention || T < 1 || c0 < 0 || c0 + T > 8192) return false;
    if (!w.wq_a_type || !w.wq_b_type || !w.kv_a_type || !w.wo_type) return false;
    const int n_head = g.n_head, head_dim = g.head_dim, kv_lora = g.kv_lora, q_lora = g.q_lora, n_embd = g.n_embd;
    const int q_dim = n_head * head_dim;
    // Sub-blocks bound the device scratch (q, qcur, attn, v are all [S][n_head * ...]); the cache grows block by block, so
    // the causal order across blocks is preserved.
    constexpr int kBlock = 512;
    for (int t0 = 0; t0 < T; t0 += kBlock) {
        const int S = std::min(kBlock, T - t0);
        const float* xs = x + (size_t) t0 * n_embd;
        float* kv_rows = cache + (size_t) (c0 + t0) * kv_lora;      // the new latents are computed straight into the cache
        std::vector<float> qr((size_t) S * q_lora), q((size_t) S * q_dim), v((size_t) S * q_dim);
        {
            const void* nw[1] = {w.wq_a};
            const int nt[1] = {w.wq_a_type};
            float* no[1] = {qr.data()};
            if (!g_native_project_batch(1, nw, nt, S, xs, n_embd, q_lora, no)) return false;
        }
        {
            const void* nw[1] = {w.kv_a};
            const int nt[1] = {w.kv_a_type};
            float* no[1] = {kv_rows};
            if (!g_native_project_batch(1, nw, nt, S, xs, n_embd, kv_lora, no)) return false;
        }
        const std::function<void(int)> norms = [&](int t) {
            rms_norm_inplace(qr.data() + (size_t) t * q_lora, q_lora, w.q_a_norm);
            rms_norm_inplace(kv_rows + (size_t) t * kv_lora, kv_lora, w.kv_a_norm);
        };
        if (g_parallel_for != nullptr && S > 1) g_parallel_for(S, norms);
        else for (int t = 0; t < S; ++t) norms(t);
        {
            const void* nw[1] = {w.wq_b};
            const int nt[1] = {w.wq_b_type};
            float* no[1] = {q.data()};
            if (!g_native_project_batch(1, nw, nt, S, qr.data(), q_lora, q_dim, no)) return false;
        }
        char cuda_err[256] = {};
        if (!mla_attend_batch_cuda(w.wk_b, w.wv_b, q.data(), cache, c0 + t0, S, n_head, head_dim, kv_lora, v.data(),
                                   cuda_err, sizeof(cuda_err))) {
            std::fprintf(stderr, "GLM MLA batched attention unavailable: %s; falling back to per-token\n", cuda_err);
            return false;
        }
        if (std::getenv("STRATA_GLM_MLA_DEBUG")) {
            // Debug: which stage disagrees?  (a) batched projections vs single-row projections of the same inputs;
            // (b) the device absorb/attention/un-absorb chain vs a host computation of it from the same q.
            for (const int tt : {0, S - 1}) {
                std::vector<float> qr1((size_t) q_lora), q1((size_t) q_dim);
                {
                    const void* nw[1] = {w.wq_a};
                    const int nt[1] = {w.wq_a_type};
                    float* no[1] = {qr1.data()};
                    g_native_project_batch(1, nw, nt, 1, xs + (size_t) tt * n_embd, n_embd, q_lora, no);
                    rms_norm_inplace(qr1.data(), q_lora, w.q_a_norm);
                }
                {
                    const void* nw[1] = {w.wq_b};
                    const int nt[1] = {w.wq_b_type};
                    float* no[1] = {q1.data()};
                    g_native_project_batch(1, nw, nt, 1, qr.data() + (size_t) tt * q_lora, q_lora, q_dim, no);
                }
                double d_qr = 0, d_q = 0, r_q = 0;
                for (int i = 0; i < q_lora; ++i) d_qr = std::max(d_qr, (double) std::fabs(qr1[(size_t) i] - qr[(size_t) tt * q_lora + i]));
                for (int i = 0; i < q_dim; ++i) {
                    d_q = std::max(d_q, (double) std::fabs(q1[(size_t) i] - q[(size_t) tt * q_dim + i]));
                    r_q = std::max(r_q, (double) std::fabs(q1[(size_t) i]));
                }
                // host chain from the batched q row
                const int n = c0 + t0 + tt + 1;
                const float scale = 1.0f / std::sqrt((float) head_dim);
                std::vector<float> vref((size_t) q_dim, 0.0f), scores((size_t) n), attn_h((size_t) kv_lora);
                for (int h = 0; h < n_head; ++h) {
                    const float* qh = q.data() + (size_t) tt * q_dim + (size_t) h * head_dim;
                    std::vector<float> qc((size_t) kv_lora);
                    for (int i = 0; i < kv_lora; ++i) {
                        const float* row = w.wk_b + ((size_t) h * kv_lora + i) * head_dim;
                        float acc = 0.0f;
                        for (int j = 0; j < head_dim; ++j) acc += row[j] * qh[j];
                        qc[(size_t) i] = acc;
                    }
                    float best = -INFINITY;
                    for (int p = 0; p < n; ++p) {
                        const float* kt = cache + (size_t) p * kv_lora;
                        float acc = 0.0f;
                        for (int i = 0; i < kv_lora; ++i) acc += qc[(size_t) i] * kt[i];
                        scores[(size_t) p] = acc * scale;
                        best = std::max(best, scores[(size_t) p]);
                    }
                    double sum = 0.0;
                    for (int p = 0; p < n; ++p) { scores[(size_t) p] = std::exp(scores[(size_t) p] - best); sum += scores[(size_t) p]; }
                    std::fill(attn_h.begin(), attn_h.end(), 0.0f);
                    for (int p = 0; p < n; ++p) {
                        const float pr = (float) (scores[(size_t) p] / sum);
                        const float* kt = cache + (size_t) p * kv_lora;
                        for (int i = 0; i < kv_lora; ++i) attn_h[(size_t) i] += pr * kt[i];
                    }
                    for (int i = 0; i < head_dim; ++i) {
                        const float* row = w.wv_b + ((size_t) h * head_dim + i) * kv_lora;
                        float acc = 0.0f;
                        for (int j = 0; j < kv_lora; ++j) acc += row[j] * attn_h[(size_t) j];
                        vref[(size_t) h * head_dim + i] = acc;
                    }
                }
                double d_v = 0, r_v = 0;
                for (int i = 0; i < q_dim; ++i) {
                    d_v = std::max(d_v, (double) std::fabs(vref[(size_t) i] - v[(size_t) tt * q_dim + i]));
                    r_v = std::max(r_v, (double) std::fabs(vref[(size_t) i]));
                }
                std::fprintf(stderr,
                             "MLA_DEBUG t0 %d tok %d: qr diff %.2e | q diff %.2e (max %.2e) | device-chain vs host-chain v diff %.2e (max %.2e)\n",
                             t0, tt, d_qr, d_q, r_q, d_v, r_v);
            }
        }
        {
            const void* nw[1] = {w.wo};
            const int nt[1] = {w.wo_type};
            float* no[1] = {out + (size_t) t0 * n_embd};
            if (!g_native_project_batch(1, nw, nt, S, v.data(), q_dim, n_embd, no)) return false;
            if (std::getenv("STRATA_GLM_MLA_DEBUG")) {
                for (const int tt : {0, S - 1}) {
                    std::vector<float> o1((size_t) n_embd);
                    float* no1[1] = {o1.data()};
                    g_native_project_batch(1, nw, nt, 1, v.data() + (size_t) tt * q_dim, q_dim, n_embd, no1);
                    double d = 0, r = 0;
                    for (int i = 0; i < n_embd; ++i) {
                        d = std::max(d, (double) std::fabs(o1[(size_t) i] - out[(size_t) (t0 + tt) * n_embd + i]));
                        r = std::max(r, (double) std::fabs(o1[(size_t) i]));
                    }
                    std::fprintf(stderr, "MLA_DEBUG wo tok %d: batched vs single-row diff %.2e (max %.2e)\n", tt, d, r);
                }
            }
        }
    }
    return true;
}

void mla_forward(const MlaWeights& w, const MlaGeometry& g, const float* x, int n_cache, const float* cache,
                 float* out, const MlaIntermediates& want) {
    const int n_head = g.n_head, head_dim = g.head_dim, kv_lora = g.kv_lora, q_lora = g.q_lora;
    const int q_dim = n_head * head_dim;

    // qr = rms_norm(wq_a @ x)
    std::vector<float> qr((size_t) q_lora, 0.0f);
    // NO CACHE CONDITION HERE, unlike the KDA's `tokens == 1`: these projections are per-token whatever the cache depth
    // is, so gating them on n_cache would put the device path on the first token only and leave the rest on the host -
    // which is exactly the kind of thing that looks like a working port and is not one.
    bool qr_done = false;
    if (g_native_project != nullptr && w.wq_a_type != 0) {
        const void* nw[1] = {w.wq_a};
        const int nt[1] = {w.wq_a_type};
        float* no[1] = {qr.data()};
        qr_done = g_native_project(1, nw, nt, x, g.n_embd, q_lora, no);
    }
    for (int j = 0; !qr_done && j < q_lora; ++j) {
        const float* row = w.wq_a + (size_t) j * g.n_embd;
        float acc = 0.0f;
        for (int i = 0; i < g.n_embd; ++i) acc += row[i] * x[i];
        qr[(size_t) j] = acc;
    }
    rms_norm_inplace(qr.data(), q_lora, w.q_a_norm);
    if (want.qr) std::copy(qr.begin(), qr.end(), want.qr);

    // q = wq_b @ qr, laid out head-major with the per-head dim innermost (ggml's reshape of [16384, nt])
    std::vector<float> q((size_t) q_dim, 0.0f);
    bool q_done = false;
    if (g_native_project != nullptr && w.wq_b_type != 0) {
        const void* nw[1] = {w.wq_b};
        const int nt[1] = {w.wq_b_type};
        float* no[1] = {q.data()};
        q_done = g_native_project(1, nw, nt, qr.data(), q_lora, q_dim, no);
    }
    for (int i = 0; !q_done && i < q_dim; ++i) {
        const float* row = w.wq_b + (size_t) i * q_lora;
        float acc = 0.0f;
        for (int j = 0; j < q_lora; ++j) acc += row[j] * qr[(size_t) j];
        q[(size_t) i] = acc;
    }

    // Qcur[h] = wk_b[h]^T (kv_lora x head_dim) @ q[h]
    std::vector<float> qcur((size_t) n_head * kv_lora, 0.0f);
    bool absorb_done = false;
    if (g_device_attention) {
        char cuda_err[256] = {};
        absorb_done = mla_head_matvec_cuda(w.wk_b, q.data(), n_head, kv_lora, head_dim, qcur.data(), cuda_err, sizeof(cuda_err));
        if (!absorb_done) std::fprintf(stderr, "GLM MLA CUDA K absorption unavailable: %s; using host\n", cuda_err);
    }
    for (int h = 0; !absorb_done && h < n_head; ++h) {
        const float* m = w.wk_b + (size_t) h * kv_lora * head_dim;
        const float* qh = q.data() + (size_t) h * head_dim;
        float* Qh = qcur.data() + (size_t) h * kv_lora;
        for (int i = 0; i < kv_lora; ++i) {
            const float* row = m + (size_t) i * head_dim;
            float acc = 0.0f;
            for (int j = 0; j < head_dim; ++j) acc += row[j] * qh[j];
            Qh[i] = acc;
        }
    }
    if (want.qcur)
        std::copy(qcur.begin(), qcur.end(), want.qcur);

    // kv = rms_norm(wkv_a_mqa @ x): the latent, ONE head
    std::vector<float> kv((size_t) kv_lora, 0.0f);
    bool kv_done = false;
    if (g_native_project != nullptr && w.kv_a_type != 0) {
        const void* nw[1] = {w.kv_a};
        const int nt[1] = {w.kv_a_type};
        float* no[1] = {kv.data()};
        kv_done = g_native_project(1, nw, nt, x, g.n_embd, kv_lora, no);
    }
    for (int i = 0; !kv_done && i < kv_lora; ++i) {
        const float* row = w.kv_a + (size_t) i * g.n_embd;
        float acc = 0.0f;
        for (int e = 0; e < g.n_embd; ++e) acc += row[e] * x[e];
        kv[(size_t) i] = acc;
    }
    rms_norm_inplace(kv.data(), kv_lora, w.kv_a_norm);
    if (want.kv) std::copy(kv.begin(), kv.end(), want.kv);

    // attention over the cache, in the latent space, then the un-absorption back to head_dim
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> attn((size_t) n_head * kv_lora, 0.0f);
    std::vector<float> v((size_t) q_dim, 0.0f);
    bool attention_done = false;
    if (g_device_attention && n_cache >= mla_cuda_min_cache()) {
        char cuda_err[256] = {};
        attention_done = mla_attention_cuda(qcur.data(),cache,n_cache,n_head,head_dim,kv_lora,
                                            attn.data(),cuda_err,sizeof(cuda_err));
        if (!attention_done) std::fprintf(stderr,"GLM MLA CUDA attention unavailable: %s; using host attention\n",cuda_err);
    }
    if (!attention_done) {
        std::vector<float> scores((size_t) (n_cache > 0 ? n_cache : 1), 0.0f);
        for (int h = 0; h < n_head; ++h) {
            const float* Qh = qcur.data() + (size_t) h * kv_lora;
            float best = -INFINITY;
            for (int t = 0; t < n_cache; ++t) {
                const float* kt = cache + (size_t) t * kv_lora;
                float acc = 0.0f;
                for (int i = 0; i < kv_lora; ++i) acc += Qh[i] * kt[i];
                scores[(size_t) t] = acc * kq_scale;
                if (scores[(size_t) t] > best) best = scores[(size_t) t];
            }
            double sum = 0.0;
            for (int t = 0; t < n_cache; ++t) {
                scores[(size_t) t] = std::exp(scores[(size_t) t] - best);
                sum += scores[(size_t) t];
            }
            float* Ah = attn.data() + (size_t) h * kv_lora;
            for (int t = 0; t < n_cache; ++t) {
                const float p = (float) (scores[(size_t) t] / sum);
                const float* kt = cache + (size_t) t * kv_lora;
                for (int i = 0; i < kv_lora; ++i) Ah[i] += p * kt[i];
            }
        }
    }
    bool unabsorb_done = false;
    if (g_device_attention) {
        char cuda_err[256] = {};
        unabsorb_done = mla_head_matvec_cuda(w.wv_b, attn.data(), n_head, head_dim, kv_lora, v.data(), cuda_err, sizeof(cuda_err));
        if (!unabsorb_done) std::fprintf(stderr, "GLM MLA CUDA V un-absorption unavailable: %s; using host\n", cuda_err);
    }
    for (int h = 0; !unabsorb_done && h < n_head; ++h) {
        const float* Ah = attn.data() + (size_t) h * kv_lora;
        // v[h] = wv_b[h] (head_dim x kv_lora) @ attn[h]
        const float* m = w.wv_b + (size_t) h * head_dim * kv_lora;
        float* vh = v.data() + (size_t) h * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            const float* row = m + (size_t) i * kv_lora;
            float acc = 0.0f;
            for (int j = 0; j < kv_lora; ++j) acc += row[j] * Ah[j];
            vh[i] = acc;
        }
    }
    if (want.attn)
        std::copy(attn.begin(), attn.end(), want.attn);

    // out = wo @ concat_heads(v)
    bool wo_done = false;
    if (g_native_project != nullptr && w.wo_type != 0) {
        const void* nw[1] = {w.wo};
        const int nt[1] = {w.wo_type};
        float* no[1] = {out};
        wo_done = g_native_project(1, nw, nt, v.data(), q_dim, g.n_embd, no);
    }
    for (int e = 0; !wo_done && e < g.n_embd; ++e) {
        const float* row = w.wo + (size_t) e * q_dim;
        float acc = 0.0f;
        for (int i = 0; i < q_dim; ++i) acc += row[i] * v[(size_t) i];
        out[e] = acc;
    }
}

}  // namespace strata::kernels::glm
