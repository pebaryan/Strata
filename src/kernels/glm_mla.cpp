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

void mla_set_native_project(MlaNativeProjectFn fn) { g_native_project = fn; }
void mla_set_device_attention(bool enabled) { g_device_attention = enabled; }

#if !defined(STRATA_ENABLE_CUDA)
bool mla_attention_cuda(const float*,const float*,int,int,int,int,float*,char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
#endif

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
    for (int h = 0; h < n_head; ++h) {
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
    for (int h = 0; h < n_head; ++h) {
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
