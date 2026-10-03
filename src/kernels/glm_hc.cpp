// src/kernels/glm_hc.cpp - the mHC implementation (branch glm5next-port).  Host code, float, no CUDA:
// the point of this stage is to get the ARCHITECTURE exactly right against a reference before the layout
// work of putting it on the device.  The graph this mirrors is quoted in the header.
#include "strata/kernels/glm_hc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace strata::kernels::glm {
namespace {

inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }

HcParallelFor g_parallel_for = nullptr;

/// Sinkhorn, exactly as the reference does it: softmax over dst, +eps, one column normalization, then
/// (iters-1) x (row, column) - every divisor has eps added.  comb is comb[dst][src], which is the ggml
/// layout of a [hc, hc] tensor with ne0 = dst, so "sum_rows" in the reference sums over dst.
void sinkhorn(float (*comb)[HC]) {
    // softmax over dst (ne0) - per src column
    for (int src = 0; src < HC; ++src) {
        float m = comb[0][src];
        for (int dst = 1; dst < HC; ++dst) m = std::max(m, comb[dst][src]);
        float sum = 0.0f;
        for (int dst = 0; dst < HC; ++dst) {
            comb[dst][src] = std::exp(comb[dst][src] - m);
            sum += comb[dst][src];
        }
        for (int dst = 0; dst < HC; ++dst) comb[dst][src] = comb[dst][src] / sum + HC_EPS;
    }
    // The names follow the reference's ggml calls, where `sum_rows` sums over ne0 - which for a
    // [hc, hc] tensor with ne0 = dst is the dst axis:
    //   norm_cols (the permuted sum_rows) divides by the sum over src  -> one divisor per dst
    //   norm_rows divides by the sum over dst                          -> one divisor per src
    auto norm_cols = [&]() {                       // divide by the sum over src
        for (int dst = 0; dst < HC; ++dst) {
            float sum = HC_EPS;
            for (int src = 0; src < HC; ++src) sum += comb[dst][src];
            for (int src = 0; src < HC; ++src) comb[dst][src] /= sum;
        }
    };
    auto norm_rows = [&]() {                       // divide by the sum over dst
        for (int src = 0; src < HC; ++src) {
            float sum = HC_EPS;
            for (int dst = 0; dst < HC; ++dst) sum += comb[dst][src];
            for (int dst = 0; dst < HC; ++dst) comb[dst][src] /= sum;
        }
    };
    norm_cols();
    for (int i = 1; i < SINKHORN_ITERS; ++i) {
        norm_rows();
        norm_cols();
    }
}

}  // namespace

void hc_set_parallel_for(HcParallelFor fn) { g_parallel_for = fn; }

void hc_pre(const float* x, const float* fn, const float* base, const float* scale, int n_embd,
            float* layer_in, HcMix* mix, int n_threads) {
    (void) n_threads;   // pool use is via hc_set_parallel_for
    const int dim = HC * n_embd;

    // rms_norm over the flattened streams (the reference normalizes the whole hc*n_embd block at once)
    double ss = 0.0;
    for (int i = 0; i < dim; ++i) ss += (double) x[i] * (double) x[i];
    const float rms = 1.0f / std::sqrt((float) (ss / dim) + HC_RMS_EPS);

    // mixes[m] = sum_i fn[m][i] * flat_norm[i]  (ggml_mul_mat reduces over the weight's ne0).  The normalised input is
    // formed once (the same x[i] * rms the rows used to recompute 24 times) and the rows - independent, each a single
    // serial reduction in the same order as before - run on the pool when one is installed.
    std::vector<float> flat_norm((size_t) dim);
    for (int i = 0; i < dim; ++i) flat_norm[(size_t) i] = x[i] * rms;
    std::vector<float> mixes((size_t) HC_MIX_DIM, 0.0f);
    const std::function<void(int)> mix_row = [&](int m) {
        const float* row = fn + (size_t) m * (size_t) dim;
        float acc = 0.0f;
        for (int i = 0; i < dim; ++i) acc += row[i] * flat_norm[(size_t) i];
        mixes[(size_t) m] = acc;
    };
    if (g_parallel_for != nullptr) g_parallel_for(HC_MIX_DIM, mix_row);
    else for (int m = 0; m < HC_MIX_DIM; ++m) mix_row(m);

    for (int h = 0; h < HC; ++h) {
        mix->pre[h] = sigmoid(mixes[(size_t) h] * scale[0] + base[h]) + HC_EPS;
        mix->post[h] = 2.0f * sigmoid(mixes[(size_t) (HC + h)] * scale[1] + base[HC + h]);
    }
    // comb[dst][src] = mixes[2*hc + dst + src*hc]  (a [hc, hc] tensor flattened with ne0 = dst fastest)
    float raw[HC][HC];
    for (int src = 0; src < HC; ++src)
        for (int dst = 0; dst < HC; ++dst) {
            const int k = 2 * HC + dst + src * HC;
            raw[dst][src] = mixes[(size_t) k] * scale[2] + base[k];
        }
    sinkhorn(raw);
    if (std::getenv("STRATA_HC_DEBUG")) {
        std::fprintf(stderr, "hc_debug raw comb (dst x src):");
        for (int dst = 0; dst < HC; ++dst)
            for (int src = 0; src < HC; ++src)
                std::fprintf(stderr, " %+.6f", (mixes[(size_t) (2 * HC + dst + src * HC)] * scale[2] +
                                                base[2 * HC + dst + src * HC]));
        std::fprintf(stderr, "\nhc_debug final comb:");
        for (int dst = 0; dst < HC; ++dst)
            for (int src = 0; src < HC; ++src) std::fprintf(stderr, " %+.8e", raw[dst][src]);
        std::fprintf(stderr, "\nhc_debug mixes:");
        for (int m = 0; m < HC_MIX_DIM; ++m) std::fprintf(stderr, " %+.6f", mixes[(size_t) m]);
        std::fprintf(stderr, "\n");
    }
    for (int dst = 0; dst < HC; ++dst)
        for (int src = 0; src < HC; ++src) mix->comb[dst][src] = raw[dst][src];

    for (int e = 0; e < n_embd; ++e) {
        float acc = 0.0f;
        for (int h = 0; h < HC; ++h) acc += mix->pre[h] * x[(size_t) h * n_embd + e];
        layer_in[e] = acc;
    }
}

void hc_post(const float* site_out, const float* residual, const HcMix& mix, int n_embd, float* out,
             int n_threads) {
    (void) n_threads;
    for (int dst = 0; dst < HC; ++dst) {
        float* dst_row = out + (size_t) dst * n_embd;
        for (int e = 0; e < n_embd; ++e) dst_row[e] = mix.post[dst] * site_out[e];
        for (int src = 0; src < HC; ++src) {
            const float c = mix.comb[dst][src];
            const float* src_row = residual + (size_t) src * n_embd;
            for (int e = 0; e < n_embd; ++e) dst_row[e] += c * src_row[e];
        }
    }
}

void hc_mean(const float* x, int n_embd, float* out) {
    for (int e = 0; e < n_embd; ++e) {
        float acc = 0.0f;
        for (int h = 0; h < HC; ++h) acc += x[(size_t) h * n_embd + e];
        out[e] = acc / (float) HC;
    }
}

}  // namespace strata::kernels::glm
