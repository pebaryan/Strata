// src/kernels/glm_indexer.cpp - the sparse indexer (branch glm5next-port).  Host code, float, no CUDA:
// as with the mHC and MLA, correctness against the reference first.  The math is quoted in the header.
#include "strata/kernels/glm_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace strata::kernels::glm {

void idx_cache_row(const IdxWeights& w, const IdxGeometry& g, const float* x, float* row_out) {
    // key = LayerNorm(attn_k @ x, weight, bias)  -- a LAYER norm, not rms, and with a bias
    std::vector<float> key((size_t) g.d, 0.0f);
    for (int i = 0; i < g.d; ++i) {
        const float* row = w.attn_k + (size_t) i * g.n_embd;
        float acc = 0.0f;
        for (int e = 0; e < g.n_embd; ++e) acc += row[e] * x[e];
        key[(size_t) i] = acc;
    }
    double mean = 0.0, sq = 0.0;
    for (int i = 0; i < g.d; ++i) mean += key[(size_t) i];
    mean /= g.d;
    for (int i = 0; i < g.d; ++i) {
        const double c = (double) key[(size_t) i] - mean;
        sq += c * c;
    }
    const float inv = 1.0f / std::sqrt((float) (sq / g.d) + IDX_NORM_EPS);
    for (int i = 0; i < g.d; ++i)
        row_out[i] = (float) ((double) key[(size_t) i] - mean) * inv * (w.k_norm_w ? w.k_norm_w[i] : 1.0f) +
                     (w.k_norm_b ? w.k_norm_b[i] : 0.0f);

    // gate = compressor_gate @ x
    for (int i = 0; i < g.d; ++i) {
        const float* row = w.c_gate + (size_t) i * g.n_embd;
        float acc = 0.0f;
        for (int e = 0; e < g.n_embd; ++e) acc += row[e] * x[e];
        row_out[g.d + i] = acc;
    }
}

IdxResult idx_select(const IdxWeights& w, const IdxGeometry& g, const float* cur, const float* qr,
                     int n_positions, const float* cache, int q_pos, int32_t* cells_out, int cells_capacity,
                     float* pooled_out, float* score_out, float* bias_out) {
    IdxResult res;
    const int r = g.kpool, d = g.d;
    const int n_pool = n_positions / r;
    if (n_pool <= 0) {
        // no complete pool: an incomplete pool has no pool key, so nothing is scored and only the tail
        // cells are visible to the query (the reference's index_kpool_always_select_tail)
        for (int p = 0; p < n_positions && res.n_cells < cells_capacity; ++p)
            cells_out[res.n_cells++] = p;
        return res;
    }

    // ---- pooled keys: a channel-wise gate-weighted average over each pool's r members
    std::vector<float> pooled((size_t) n_pool * d, 0.0f);
    for (int b = 0; b < n_pool; ++b) {
        for (int ch = 0; ch < d; ++ch) {
            float mx = -INFINITY;
            for (int m = 0; m < r; ++m) {
                const float* row = cache + (size_t) (b * r + m) * 2 * d;
                const float logit = row[d + ch] + (w.ape ? w.ape[(size_t) m * d + ch] : 0.0f);
                if (logit > mx) mx = logit;
            }
            double sum = 0.0;
            float num = 0.0f;
            for (int m = 0; m < r; ++m) {
                const float* row = cache + (size_t) (b * r + m) * 2 * d;
                const float e = std::exp(row[d + ch] + (w.ape ? w.ape[(size_t) m * d + ch] : 0.0f) - mx);
                sum += e;
                num += e * row[ch];
            }
            pooled[(size_t) b * d + ch] = (float) ((double) num / sum);
        }
    }
    if (pooled_out)
        std::copy(pooled.begin(), pooled.end(), pooled_out);

    // ---- the query, the per-head gate, and the score
    std::vector<float> q((size_t) g.nh * d, 0.0f);
    for (int h = 0; h < g.nh * d; ++h) {
        const float* row = w.attn_q_b + (size_t) h * g.q_lora;
        float acc = 0.0f;
        for (int j = 0; j < g.q_lora; ++j) acc += row[j] * qr[j];
        q[(size_t) h] = acc;
    }
    const float scale = 1.0f / std::sqrt((float) d * (float) g.nh);
    std::vector<float> wts((size_t) g.nh, 0.0f);
    for (int h = 0; h < g.nh; ++h) {
        const float* row = w.proj + (size_t) h * g.n_embd;
        float acc = 0.0f;
        for (int e = 0; e < g.n_embd; ++e) acc += row[e] * cur[e];
        wts[(size_t) h] = acc * scale;
    }
    std::vector<double> score((size_t) n_pool, 0.0);
    for (int b = 0; b < n_pool; ++b) {
        const float* pb = pooled.data() + (size_t) b * d;
        double acc = 0.0;
        for (int h = 0; h < g.nh; ++h) {
            const float* qh = q.data() + (size_t) h * d;
            float dot = 0.0f;
            for (int i = 0; i < d; ++i) dot += qh[i] * pb[i];
            acc += wts[(size_t) h] * std::max(dot, 0.0f);     // relu, then the weighted head sum
        }
        score[(size_t) b] = acc;
    }

    // ---- the visibility bias: a query may pick a pool only if it is complete and its last member is
    //      at or before the query.  Incomplete pools are masked out here because they have no key.
    std::vector<double> bias((size_t) n_pool, 0.0);
    for (int b = 0; b < n_pool; ++b) {
        const bool complete = (b + 1) * r <= n_positions;
        const bool visible = (b + 1) * r - 1 <= q_pos;
        bias[(size_t) b] = (complete && visible) ? 0.0 : -std::numeric_limits<double>::infinity();
    }
    if (score_out)
        for (int b = 0; b < n_pool; ++b) score_out[b] = (float) score[(size_t) b];
    if (bias_out)
        for (int b = 0; b < n_pool; ++b) bias_out[b] = (float) bias[(size_t) b];

    // ---- select whole pools, best first; ties by index (deterministic, unlike ggml_top_k)
    const int n_sel = std::min(n_pool, g.top_k / r);
    std::vector<int> order((size_t) n_pool);
    for (int b = 0; b < n_pool; ++b) order[(size_t) b] = b;
    std::stable_sort(order.begin(), order.end(), [&](int a, int bb) {
        const double sa = score[(size_t) a] + bias[(size_t) a];
        const double sb = score[(size_t) bb] + bias[(size_t) bb];
        return sa > sb;
    });
    std::vector<int> chosen(order.begin(), order.begin() + n_sel);
    std::sort(chosen.begin(), chosen.end());
    for (int b : chosen)
        for (int m = 0; m < r && res.n_cells < cells_capacity; ++m) cells_out[res.n_cells++] = b * r + m;
    res.n_sel = n_sel;

    // ---- the trailing incomplete pool is ALWAYS selected: no pool key, so it can never win on score
    for (int p = n_pool * r; p < n_positions && res.n_cells < cells_capacity; ++p)
        cells_out[res.n_cells++] = p;
    std::sort(cells_out, cells_out + res.n_cells);
    return res;
}

}  // namespace strata::kernels::glm
