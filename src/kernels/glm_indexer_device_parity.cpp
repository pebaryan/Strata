// Device sparse indexer vs the host reference (glm_indexer.cpp), random weights, real geometry.
// Weights are Q8_0 on both sides (the host reads the dequantized values); activations are float on the host and
// q8_1 on the device, so rows/scores agree to ~1e-2 and the selections are compared as sets (a few boundary pools may differ).
#include <cuda_runtime.h>
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "strata/kernels/glm_indexer.hpp"
#include "strata/kernels/glm_indexer_device.hpp"

using namespace strata::kernels::glm;

static void quant_q8_0(const std::vector<float>& w, std::vector<uint8_t>& blocks, std::vector<float>& deq) {
    const size_t nb = w.size() / 32;
    blocks.assign(nb * 34, 0);
    deq.resize(w.size());
    for (size_t b = 0; b < nb; ++b) {
        float amax = 0;
        for (int i = 0; i < 32; ++i) amax = std::max(amax, std::fabs(w[b * 32 + i]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        const uint16_t dh = _cvtss_sh(d, 0);
        std::memcpy(&blocks[b * 34], &dh, 2);
        const float dd = _cvtsh_ss(dh);
        for (int i = 0; i < 32; ++i) {
            const int q = (int) std::lround(w[b * 32 + i] * id);
            blocks[b * 34 + 2 + i] = (uint8_t) (int8_t) q;
            deq[b * 32 + i] = q * dd;
        }
    }
}
static void* up(const void* h, size_t n) { void* d = nullptr; cudaMalloc(&d, n); cudaMemcpy(d, h, n, cudaMemcpyHostToDevice); return d; }

int main(int argc, char** argv) {
    const int N = argc > 1 ? std::atoi(argv[1]) : 2700;
    const IdxGeometry hg;
    MlaGeometry g;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto rnd = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };
    const float ws = 1.0f / std::sqrt((float) hg.n_embd);
    auto k = rnd((size_t) hg.d * hg.n_embd, ws), gt = rnd((size_t) hg.d * hg.n_embd, ws), qb = rnd((size_t) hg.nh * hg.d * hg.q_lora, 1.0f / std::sqrt((float) hg.q_lora));
    auto proj = rnd((size_t) hg.nh * hg.n_embd, ws), ape = rnd((size_t) hg.kpool * hg.d, 0.5f);
    auto nw = rnd(hg.d, 0.2f), nb = rnd(hg.d, 0.2f);
    for (auto& v : nw) v += 1.0f;
    std::vector<uint8_t> kq, gq, qq;
    std::vector<float> kd, gd, qd;
    quant_q8_0(k, kq, kd); quant_q8_0(gt, gq, gd); quant_q8_0(qb, qq, qd);

    IdxWeights hw;
    hw.attn_k = kd.data(); hw.c_gate = gd.data(); hw.attn_q_b = qd.data(); hw.proj = proj.data(); hw.ape = ape.data();
    hw.k_norm_w = nw.data(); hw.k_norm_b = nb.data();

    MlaWeights dw;
    dw.idx_attn_k = (const float*) up(kq.data(), kq.size()); dw.idx_attn_k_type = 8;
    dw.idx_c_gate = (const float*) up(gq.data(), gq.size()); dw.idx_c_gate_type = 8;
    dw.idx_attn_q_b = (const float*) up(qq.data(), qq.size()); dw.idx_attn_q_b_type = 8;
    dw.idx_proj = proj.data(); dw.idx_ape = ape.data(); dw.idx_k_norm_w = nw.data(); dw.idx_k_norm_b = nb.data();
    if (!idx_available(dw)) { std::printf("FAIL: idx_available\n"); return 1; }
    idx_set_capacity(N + 64);

    auto x = rnd((size_t) N * hg.n_embd, 1.0f), qr = rnd((size_t) N * hg.q_lora, 1.0f);
    float* d_x = (float*) up(x.data(), x.size() * 4);
    float* d_qr = (float*) up(qr.data(), qr.size() * 4);
    char err[256] = "";
    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    const int key = 0; const void* lk = &key;
    // write: a mix of chunk sizes, including ones that start mid-pool
    for (int pos = 0, step = 1; pos < N;) {
        const int T = std::min(N - pos, step);
        if (idx_write_device(dw, g, lk, d_x + (size_t) pos * hg.n_embd, T, pos, (void*) stream, err, sizeof err) != 1) { std::printf("FAIL write @%d: %s\n", pos, err); return 1; }
        pos += T; step = step == 1 ? 3 : (step == 3 ? 64 : (step == 64 ? 1 : 1));
    }
    std::vector<float> hrows((size_t) N * 2 * hg.d), drows((size_t) N * 2 * hg.d);
    for (int p = 0; p < N; ++p) idx_cache_row(hw, hg, &x[(size_t) p * hg.n_embd], &hrows[(size_t) p * 2 * hg.d]);
    if (!idx_debug_fetch_rows(lk, N, drows.data())) { std::printf("FAIL fetch rows\n"); return 1; }
    double rmax = 0;
    for (size_t i = 0; i < hrows.size(); ++i) rmax = std::max(rmax, (double) std::fabs(hrows[i] - drows[i]) / (1.0 + std::fabs(hrows[i])));
    std::printf("rows: max rel diff %.3g\n", rmax);

    int bad = 0;
    double worst_overlap = 1.0;
    std::vector<int32_t> hc(IDX_CELL_STRIDE + 8);
    auto check = [&](int pos0, int T) {
        int32_t *dc = nullptr, *dn = nullptr;
        const int r = idx_cells_device(dw, g, lk, d_x + (size_t) pos0 * hg.n_embd, d_qr + (size_t) pos0 * hg.q_lora, T, pos0, &dc, &dn, (void*) stream, err, sizeof err);
        if (r != 1) { std::printf("FAIL cells @%d: %s\n", pos0, err); ++bad; return; }
        std::vector<int32_t> cells((size_t) T * IDX_CELL_STRIDE), nc(T);
        cudaMemcpy(cells.data(), dc, cells.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(nc.data(), dn, T * 4, cudaMemcpyDeviceToHost);
        for (int t = 0; t < T; ++t) {
            const int q = pos0 + t;
            const IdxResult hr = idx_select(hw, hg, &x[(size_t) q * hg.n_embd], &qr[(size_t) q * hg.q_lora], q + 1, hrows.data(), q, hc.data(), (int) hc.size());
            std::vector<int32_t> a(hc.begin(), hc.begin() + hr.n_cells), b(cells.begin() + (size_t) t * IDX_CELL_STRIDE, cells.begin() + (size_t) t * IDX_CELL_STRIDE + nc[t]);
            std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
            std::vector<int32_t> inter;
            std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(inter));
            const double ov = a.empty() ? 1.0 : (double) inter.size() / a.size();
            worst_overlap = std::min(worst_overlap, ov);
            if ((int) a.size() != (int) b.size() || ov < 0.975) { ++bad; std::printf("MISMATCH q=%d host %zu dev %zu overlap %.4f\n", q, a.size(), b.size(), ov); }
        }
    };
    const int T0 = idx_max_tokens();
    std::printf("idx_max_tokens %d\n", T0);
    check(N - 1, 1);
    check(std::min(N - 1, 2051), 1);
    check(std::min(N - 1, 2052), 1);
    check(std::min(N - 5, 2300), 1);
    check(N - std::min(N, 40), std::min(N, 40));
    check(N - std::min(N, T0), std::min(N, T0));
    std::printf("worst overlap %.4f, %d bad\n", worst_overlap, bad);
    std::printf(bad ? "FAIL\n" : "PASS\n");
    return bad ? 1 : 0;
}
