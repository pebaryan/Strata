// src/kernels/glm_moe.cpp - the MoE site and its router (branch glm5next-port).  Host code, float, no
// CUDA: correctness against the reference first, like the other GLM kernels.  The math, and the two
// orderings that are silent if wrong, are quoted in the header.
#include "strata/kernels/glm_moe.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace strata::kernels::glm {

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }

MoeParallelFor g_parallel_for = nullptr;

}  // namespace

void moe_set_parallel_for(MoeParallelFor fn) { g_parallel_for = fn; }

void moe_route(const float* router, const float* probs_b, const MoeGeometry& g, const float* x,
               int32_t* ids_out, float* weights_out, float* probs_out) {
    const int E = g.n_expert, ne = g.n_embd;
    std::vector<float> probs((size_t) E);
    // The expert rows are independent dot products (each reduced serially in the same order as ever), so blocks of them
    // run on the pool when one is installed - a single token's routing is ~1.6 ms of dependent float adds otherwise.
    constexpr int kBlock = 16;
    const std::function<void(int)> rows = [&](int b) {
        const int e_end = std::min(E, (b + 1) * kBlock);
        for (int e = b * kBlock; e < e_end; ++e) {
            const float* row = router + (size_t) e * ne;
            float acc = 0.0f;
            for (int i = 0; i < ne; ++i) acc += row[i] * x[i];
            probs[(size_t) e] = sigmoidf(acc);
        }
    };
    const int n_blocks = (E + kBlock - 1) / kBlock;
    if (g_parallel_for != nullptr) g_parallel_for(n_blocks, rows);
    else for (int b = 0; b < n_blocks; ++b) rows(b);
    if (probs_out)
        for (int e = 0; e < E; ++e) probs_out[e] = probs[(size_t) e];

    // selection is on the biased probs, weights come from the unbiased ones
    std::vector<int> order((size_t) E);
    for (int e = 0; e < E; ++e) order[(size_t) e] = e;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = probs[(size_t) a] + (probs_b ? probs_b[a] : 0.0f);
        const float sb = probs[(size_t) b] + (probs_b ? probs_b[b] : 0.0f);
        return sa > sb;
    });

    const int k = std::min(g.n_used, E);
    double sum = 0.0;
    for (int i = 0; i < k; ++i) {
        ids_out[i] = order[(size_t) i];
        weights_out[i] = probs[(size_t) order[(size_t) i]];
        sum += (double) weights_out[i];
    }
    if (g.norm_w) {
        const double s = std::max(sum, (double) MOE_F16_MIN);   // the reference clamps the sum
        for (int i = 0; i < k; ++i) weights_out[i] = (float) ((double) weights_out[i] / s);
    }
    if (g.w_scale != 0.0f && g.w_scale != 1.0f)
        for (int i = 0; i < k; ++i) weights_out[i] *= g.w_scale;   // scaled AFTER normalising
}

void expert_ffn(const float* wg, const float* wu, const float* wd, const MoeGeometry& g, const float* x,
                float* out, float clamp_limit) {
    const int ne = g.n_embd, ff = g.ff;
    const bool clamp = clamp_limit > 1e-6f;
    std::vector<float> h((size_t) ff);
    for (int j = 0; j < ff; ++j) {
        const float* rg = wg + (size_t) j * ne;
        const float* ru = wu + (size_t) j * ne;
        float ag = 0.0f, au = 0.0f;
        for (int i = 0; i < ne; ++i) {
            ag += rg[i] * x[i];
            au += ru[i] * x[i];
        }
        if (clamp) {
            // ggml_swiglu_clamp, exactly as ggml's CPU kernel does it (ops.cpp,
            // ggml_compute_forward_swiglu_clamp_f32): the GATE clamp is one-sided - only from above -
            // and the up clamp is two-sided, both applied BEFORE the SiLU.
            ag = std::min(ag, clamp_limit);
            au = std::max(-clamp_limit, std::min(au, clamp_limit));
        }
        h[(size_t) j] = (ag / (1.0f + std::exp(-ag))) * au;         // silu(gate) * up
    }
    for (int i = 0; i < ne; ++i) {
        const float* rd = wd + (size_t) i * ff;
        float acc = 0.0f;
        for (int j = 0; j < ff; ++j) acc += rd[j] * h[(size_t) j];
        out[i] = acc;
    }
}

void moe_forward(const float* router, const float* probs_b, const MoeGeometry& g, const float* x,
                 const float* const* const* experts, const float* const* shared, float* out,
                 int32_t* ids_out, float* weights_out, float* moe_out, float* shexp_out) {
    const int k = std::min(g.n_used, g.n_expert), ne = g.n_embd;
    std::vector<int32_t> ids((size_t) k);
    std::vector<float> weights((size_t) k);
    moe_route(router, probs_b, g, x, ids.data(), weights.data());
    if (ids_out)
        for (int i = 0; i < k; ++i) ids_out[i] = ids[(size_t) i];
    if (weights_out)
        for (int i = 0; i < k; ++i) weights_out[i] = weights[(size_t) i];

    std::vector<float> acc((size_t) ne, 0.0f), tmp((size_t) ne);
    for (int i = 0; i < k; ++i) {
        expert_ffn(experts[i][0], experts[i][1], experts[i][2], g, x, tmp.data(), g.clamp_exp);
        for (int j = 0; j < ne; ++j) acc[(size_t) j] += weights[(size_t) i] * tmp[(size_t) j];
    }
    if (moe_out) std::memcpy(moe_out, acc.data(), (size_t) ne * sizeof(float));

    // the shared expert uses swiglu_clamp_SHEXP, not the experts' limit: the reference builds it with
    // build_ffn, which reads the shexp array
    expert_ffn(shared[0], shared[1], shared[2], g, x, tmp.data(), g.clamp_shexp);
    if (shexp_out) std::memcpy(shexp_out, tmp.data(), (size_t) ne * sizeof(float));

    for (int j = 0; j < ne; ++j) out[j] = acc[(size_t) j] + tmp[(size_t) j];
}

}  // namespace strata::kernels::glm
