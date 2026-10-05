// src/kernels/glm_mla.cpp - the MLA block (branch glm5next-port).  Host code, float, no CUDA, for the same
// reason as the mHC: the architecture has to be exactly right against a reference before any of it moves
// onto the device.  The graph is quoted in the header; the oracle lives in tools/glm5_mla_reference.py.
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_indexer_device.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels::glm {
namespace {

MlaNativeProjectFn g_native_project = nullptr;
bool g_device_attention = false;
// The decoupled-RoPE graph's K-absorb and V-un-absorb are plain head-matvecs (wk_b: [n_head][kv_lora][nope],
// wv_b: [n_head][head_dim][kv_lora]); when set, mla_forward_rope runs them on the device kernel and falls
// back to the host loop if the kernel declines.  This is the GLM-4.7 path's missing device work.
bool g_rope_head_cuda = false;
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

/// ggml GGML_ROPE_TYPE_NEOX over `n_rot` dims: rotate the pair (i, i + n_rot/2) by theta_i = pos * freq_base^(-2i/n_rot).
/// The deepseek2 rope carries no YaRN (freq_scale 1, ext_factor 0), so n_ctx_orig/attn_factor do not enter; GLM-4.7-Flash
/// sets rope.dimension_count 64 and deepseek2.rope.freq_base 1e6.  The angle is evaluated in DOUBLE: at these bases and
/// positions theta reaches thousands of radians, where a float pow() angle is off by ~1e-4 rad - enough to move the rope
/// half by ~1e-4, i.e. right at the gate.  Only the coefficients, not the activations, go through double.
void rope_neox_inplace(float* v, int n_rot, int pos, float freq_base) {
    const int half = n_rot / 2;
    for (int i = 0; i < half; ++i) {
        const double theta = (double) pos * std::pow((double) freq_base, -2.0 * (double) i / (double) n_rot);
        const float c = (float) std::cos(theta), s = (float) std::sin(theta);
        const float x0 = v[i], x1 = v[i + half];
        v[i]        = x0 * c - x1 * s;
        v[i + half] = x0 * s + x1 * c;
    }
}

/// GLM-4.7-Flash (deepseek2) MLA: decoupled RoPE.  q is split per head into nope (absorbed by wk_b) and a roped
/// n_rot-half that is carried alongside; K is the normed latent concatenated with the roped k_pe; V is the latent.
void mla_forward_rope(const MlaWeights& w, const MlaGeometry& g, const float* x, int n_cache, const float* cache,
                      float* out, const MlaIntermediates& want, int pos) {
    const int n_head = g.n_head, head_dim = g.head_dim, kv_lora = g.kv_lora, q_lora = g.q_lora;
    const int n_rot = g.n_rot, nope = head_dim - n_rot;
    const int q_dim = n_head * head_dim;
    const int kv_dim = kv_lora + n_rot;
    const int stride = kv_lora + n_rot;   // one cache row: latent then roped k_pe
    if (pos < 0) pos = n_cache - 1;

    std::vector<float> qr((size_t) q_lora, 0.0f);
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

    // q = wq_b @ qr -> [n_head][head_dim], split into nope / rope halves per head
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
    std::vector<float> q_nope((size_t) n_head * nope, 0.0f), q_pe((size_t) n_head * n_rot, 0.0f);
    for (int h = 0; h < n_head; ++h) {
        const float* qh = q.data() + (size_t) h * head_dim;
        std::copy(qh, qh + nope, q_nope.data() + (size_t) h * nope);
        std::copy(qh + nope, qh + head_dim, q_pe.data() + (size_t) h * n_rot);
        rope_neox_inplace(q_pe.data() + (size_t) h * n_rot, n_rot, pos, w.rope_freq_base);
    }
    if (want.q_nope) std::copy(q_nope.begin(), q_nope.end(), want.q_nope);
    if (want.q_pe) std::copy(q_pe.begin(), q_pe.end(), want.q_pe);

    // kv_mqa = wkv_a_mqa @ x -> [kv_lora + n_rot]; kv = rms_norm(first kv_lora); k_pe roped from the tail rows
    std::vector<float> kv_mqa((size_t) kv_dim, 0.0f);
    bool kv_done = false;
    if (g_native_project != nullptr && w.kv_a_type != 0) {
        const void* nw[1] = {w.kv_a};
        const int nt[1] = {w.kv_a_type};
        float* no[1] = {kv_mqa.data()};
        kv_done = g_native_project(1, nw, nt, x, g.n_embd, kv_dim, no);
    }
    for (int i = 0; !kv_done && i < kv_dim; ++i) {
        const float* row = w.kv_a + (size_t) i * g.n_embd;
        float acc = 0.0f;
        for (int e = 0; e < g.n_embd; ++e) acc += row[e] * x[e];
        kv_mqa[(size_t) i] = acc;
    }
    std::vector<float> kv(kv_mqa.begin(), kv_mqa.begin() + kv_lora);
    rms_norm_inplace(kv.data(), kv_lora, w.kv_a_norm);
    std::vector<float> k_pe(kv_mqa.begin() + kv_lora, kv_mqa.end());
    rope_neox_inplace(k_pe.data(), n_rot, pos, w.rope_freq_base);
    if (want.kv) std::copy(kv.begin(), kv.end(), want.kv);
    if (want.k_pe) std::copy(k_pe.begin(), k_pe.end(), want.k_pe);
    if (want.kv_only) return;   // the caller only wanted the new cache row

    // Qcur[h] = wk_b[h] (kv_lora x nope) @ q_nope[h].  wk_b is [n_head][kv_lora][nope] and q_nope is
    // [n_head][nope], which is exactly the head-matvec kernel's [n_head][rows][cols] @ [n_head][cols].
    std::vector<float> qcur((size_t) n_head * kv_lora, 0.0f);
    bool absorb_done = false;
    if (g_rope_head_cuda) {
        char cuda_err[256] = {};
        absorb_done = mla_head_matvec_cuda(w.wk_b, q_nope.data(), n_head, kv_lora, nope, qcur.data(),
                                           cuda_err, sizeof(cuda_err));
        if (!absorb_done) std::fprintf(stderr, "GLM MLA rope K-absorb unavailable: %s; using host\n", cuda_err);
    }
    for (int h = 0; !absorb_done && h < n_head; ++h) {
        const float* m = w.wk_b + (size_t) h * kv_lora * nope;
        const float* qh = q_nope.data() + (size_t) h * nope;
        float* Qh = qcur.data() + (size_t) h * kv_lora;
        for (int i = 0; i < kv_lora; ++i) {
            const float* row = m + (size_t) i * nope;
            float acc = 0.0f;
            for (int j = 0; j < nope; ++j) acc += row[j] * qh[j];
            Qh[i] = acc;
        }
    }
    if (want.qcur) std::copy(qcur.begin(), qcur.end(), want.qcur);

    // decoupled attention: score[h][t] = (Qcur[h] . latent_t + q_pe[h] . kpe_t) * kq_scale, V = latent
    const float kq_scale = 1.0f / std::sqrt((float) head_dim);
    std::vector<float> attn((size_t) n_head * kv_lora, 0.0f);
    std::vector<float> scores((size_t) (n_cache > 0 ? n_cache : 1), 0.0f);
    for (int h = 0; h < n_head; ++h) {
        const float* Qh = qcur.data() + (size_t) h * kv_lora;
        const float* pe = q_pe.data() + (size_t) h * n_rot;
        float best = -INFINITY;
        for (int t = 0; t < n_cache; ++t) {
            const float* kt = cache + (size_t) t * stride;
            const float* pt = kt + kv_lora;
            float acc = 0.0f;
            for (int i = 0; i < kv_lora; ++i) acc += Qh[i] * kt[i];
            for (int i = 0; i < n_rot; ++i) acc += pe[i] * pt[i];
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
            const float* kt = cache + (size_t) t * stride;
            for (int i = 0; i < kv_lora; ++i) Ah[i] += p * kt[i];
        }
    }
    if (want.attn) std::copy(attn.begin(), attn.end(), want.attn);

    // v[h] = wv_b[h] @ attn[h]; out = wo @ concat(v).  wv_b is [n_head][head_dim][kv_lora], attn is [n_head][kv_lora].
    std::vector<float> v((size_t) q_dim, 0.0f);
    bool unabsorb_done = false;
    if (g_rope_head_cuda) {
        char cuda_err[256] = {};
        unabsorb_done = mla_head_matvec_cuda(w.wv_b, attn.data(), n_head, head_dim, kv_lora, v.data(),
                                             cuda_err, sizeof(cuda_err));
        if (!unabsorb_done) std::fprintf(stderr, "GLM MLA rope V-un-absorb unavailable: %s; using host\n", cuda_err);
    }
    for (int h = 0; !unabsorb_done && h < n_head; ++h) {
        const float* Ah = attn.data() + (size_t) h * kv_lora;
        const float* m = w.wv_b + (size_t) h * head_dim * kv_lora;
        float* vh = v.data() + (size_t) h * head_dim;
        for (int i = 0; i < head_dim; ++i) {
            const float* row = m + (size_t) i * kv_lora;
            float acc = 0.0f;
            for (int j = 0; j < kv_lora; ++j) acc += row[j] * Ah[j];
            vh[i] = acc;
        }
    }
    if (want.v) std::copy(v.begin(), v.end(), want.v);
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

}  // namespace

MlaNativeProjectBatchFn g_native_project_batch = nullptr;
MlaParallelFor g_parallel_for = nullptr;
bool g_device_block = false;

void mla_set_device_block(bool enabled) { g_device_block = enabled; }
bool mla_device_block_enabled() { return g_device_block && g_device_attention; }
void mla_set_native_project(MlaNativeProjectFn fn) { g_native_project = fn; }
void mla_set_device_attention(bool enabled) { g_device_attention = enabled; }
void mla_set_native_project_batch(MlaNativeProjectBatchFn fn) { g_native_project_batch = fn; }
void mla_set_rope_head_cuda(bool enabled) { g_rope_head_cuda = enabled; }
void mla_set_parallel_for(MlaParallelFor fn) { g_parallel_for = fn; }

#if !defined(STRATA_ENABLE_CUDA)
bool idx_available(const MlaWeights&) { return false; }
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
bool mla_attend_batch_idx_cuda(const MlaWeights&,const MlaGeometry&,const float*,const float*,const float*,const float*,int,int,
                               float*,char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return false;
}
int mla_block_decode_cuda(const MlaWeights&,const MlaGeometry&,const float*,int,const float*,float*,float*,
                          char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return 0;
}
int mla_block_launch_cuda(const MlaWeights&,const MlaGeometry&,const float*,int,const float*,float*,float*,void*,
                          char* error,size_t error_capacity) {
    if(error&&error_capacity) std::snprintf(error,error_capacity,"CUDA support was not compiled");
    return 0;
}
#endif

bool mla_forward_batch(const MlaWeights& w, const MlaGeometry& g, const float* x, int T, int c0, float* cache, float* out) {
    if (g.n_rot != 0) return false;   // the device batch path has no decoupled RoPE; fall back to mla_forward per token
    if (!g_native_project_batch || !g_device_attention || T < 1 || c0 < 0) return false;
    if (c0 + T > 8192 && !kernels::glm::idx_available(w)) return false;   // dense device attention stops at 8192; the indexer lifts it
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
        if (!mla_attend_batch_idx_cuda(w, g, xs, qr.data(), q.data(), cache, c0 + t0, S, v.data(), cuda_err, sizeof(cuda_err))) {
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
                 float* out, const MlaIntermediates& want, int pos) {
    const int n_head = g.n_head, head_dim = g.head_dim, kv_lora = g.kv_lora, q_lora = g.q_lora;
    const int q_dim = n_head * head_dim;

    // GLM-4.7-Flash (deepseek2): the decoupled-RoPE graph, host-only for now.  Kept as its own function so the
    // nope-only path below is byte-identical to the GLM-5.3 port (n_rot defaults to 0 and never reaches here).
    if (g.n_rot > 0) {
        if (g_device_attention || g_device_block)
            std::fprintf(stderr, "[mla] decoupled-RoPE MLA has no device path yet; using the host kernel\n");
        mla_forward_rope(w, g, x, n_cache, cache, out, want, pos);
        return;
    }

    // A decoded token on the device-resident block: it receives the new latent through want.kv (the cache row this call
    // appends) and declines, touching nothing, whenever a precondition is not met - then the host path below runs.
    if (g_device_block && g_device_attention && want.kv && !want.qr && !want.qcur && !want.attn && n_cache >= 1 &&
        want.kv == cache + (size_t) (n_cache - 1) * (size_t) kv_lora) {
        char block_err[256] = {};
        const int r = mla_block_decode_cuda(w, g, x, n_cache, cache, want.kv, out, block_err, sizeof(block_err));
        if (r == 1) {
            if (std::getenv("STRATA_GLM_MLA_BLOCK_VERIFY")) {
                // Debug: run the host path on the same inputs, report how far the block is, and keep the host result so
                // the run proceeds exactly as the baseline would.
                const std::vector<float> blk(out, out + g.n_embd), kv_blk(want.kv, want.kv + kv_lora);
                std::vector<float> ref((size_t) g.n_embd);
                g_device_block = false;
                mla_forward(w, g, x, n_cache, cache, ref.data(), want);
                g_device_block = true;
                double d = 0, m = 0, dkv = 0;
                for (int i = 0; i < g.n_embd; ++i) {
                    d = std::max(d, (double) std::fabs(ref[(size_t) i] - blk[(size_t) i]));
                    m = std::max(m, (double) std::fabs(ref[(size_t) i]));
                }
                for (int i = 0; i < kv_lora; ++i) dkv = std::max(dkv, (double) std::fabs(want.kv[i] - kv_blk[(size_t) i]));
                std::fprintf(stderr, "MLA_BLOCK_VERIFY n_cache %d: out diff %.3e (max|ref| %.3e), latent row diff %.3e\n", n_cache, d,
                             m, dkv);
                std::copy(ref.begin(), ref.end(), out);
            }
            return;
        }
        if (r < 0) {
            std::fprintf(stderr, "GLM MLA device block failed: %s\n", block_err);
            std::exit(1);
        }
    }

    if (n_cache >= 2052) {   // IDX_SPARSE_FROM: the host path has no indexer, so it attends every cell
        static bool warned = false;
        if (!warned) {
            warned = true;
            std::fprintf(stderr, "[mla] host attention at %d positions attends densely; the model's sparse indexer applies only on the device blocks\n", n_cache);
        }
    }
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
    if (want.v)
        std::copy(v.begin(), v.end(), want.v);

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
