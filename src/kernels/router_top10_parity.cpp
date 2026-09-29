// src/kernels/router_top10_parity.cpp - P2.S2's parity test for the MoE router.
//
// The reference here is a HOST implementation of `ref/moe.py::router` written from the same specification the
// kernel was written from - so this checks the kernel against the spec independently of the kernel, which is
// the strongest thing available without running llama.cpp.  It is NOT a check against llama.cpp itself, and
// that gap is stated rather than implied.
//
// WHAT THE TEST CAN AND CANNOT SEE.  It compares expert IDs exactly and weights to a tolerance, on several
// logit distributions including ties (where the "smallest index wins" rule is the only thing that decides) and
// near-ties (where a float difference decides).  It also asserts that the 2**-14 CLAMP CANNOT TRIGGER for this
// model's geometry - see below - so the absence of a clamp test is a proven fact rather than an omission.
//
// LOCAL PRUNED-MODEL SUPPORT (peb, 2026-09-29): this test used to fix NE = 512 and exercise only the generic
// kernel, which is exactly the geometry that could not see a real bug.  The fork's pruned pack has 256 experts,
// and the engine reaches the router through THREE entry points (the generic kernel, and the native router's
// single-row and multi-row launches).  A row stride pinned at 512 in the native MULTI-ROW launch returned wrong
// experts for every row past the first on a 256-expert pack, while the single-row launch - all the engine used
// for text generation at the time - stayed correct.  So this test now:
//   * sweeps both expert counts (512 and 256) across all three entries,
//   * asserts the native multi-row launch is BIT-IDENTICAL to the per-row single launches (its documented
//     contract: "each row exactly as the single call"), not merely within tolerance,
//   * asserts the multi-row launch is bit-identical to a repeat of itself,
//   * uses a per-row dominant expert, so a row read from the wrong offset cannot accidentally look right,
//   * requires an expert count with no layout to be REFUSED rather than served by the 512-expert kernel.
#include "strata/kernels/router_top10.hpp"
#include "strata/kernels/native_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr double RENORM_CLAMP = 6.103515625e-05;      // 2**-14

// The kernel's own top-k weight sum for one row; a wrong set of experts shows up as a wildly wrong sum.
double w_sum(const std::vector<float>& w, size_t off, int k) {
    double s = 0;
    for (int i = 0; i < k; ++i) s += (double) w[off + (size_t) i];
    return s;
}

// The host reference: softmax over all experts, stable descending argsort, gather, clamp, renormalise.
void reference(const float* logits, int n_expert, int k, int* ids, float* w) {
    double mx = logits[0];
    for (int e = 1; e < n_expert; ++e) mx = std::max(mx, (double) logits[e]);
    double sum = 0;
    std::vector<double> p((size_t) n_expert);
    for (int e = 0; e < n_expert; ++e) {
        p[(size_t) e] = std::exp((double) logits[e] - mx);
        sum += p[(size_t) e];
    }
    for (int e = 0; e < n_expert; ++e) p[(size_t) e] /= sum;

    // stable descending: std::stable_sort with a strict greater-than keeps equal elements in index order
    std::vector<int> idx((size_t) n_expert);
    std::iota(idx.begin(), idx.end(), 0);
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return p[(size_t) a] > p[(size_t) b]; });
    double s = 0;
    for (int i = 0; i < k; ++i) {
        ids[i] = idx[(size_t) i];
        w[i] = (float) p[(size_t) idx[(size_t) i]];
        s += p[(size_t) idx[(size_t) i]];
    }
    s = std::max(s, RENORM_CLAMP);
    for (int i = 0; i < k; ++i) w[i] = (float) ((double) w[i] / s);
}

int run_case(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k,
             double tol) {
    std::vector<int> h_ids((size_t) n_tokens * k);
    std::vector<float> h_w((size_t) n_tokens * k);
    std::vector<int> r_ids((size_t) n_tokens * k);
    std::vector<float> r_w((size_t) n_tokens * k);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t) t * n_expert], n_expert, k, &r_ids[(size_t) t * k], &r_w[(size_t) t * k]);
    }

    float* d_l = nullptr;
    int* d_ids = nullptr;
    float* d_w = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_ids, h_ids.size() * sizeof(int)), "malloc ids");
    check(cudaMalloc(&d_w, h_w.size() * sizeof(float)), "malloc w");
    check(cudaMemcpy(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice), "copy");
    strata::kernels::router_top10(d_l, n_tokens, n_expert, k, d_ids, d_w, nullptr);
    check(cudaMemcpy(h_ids.data(), d_ids, h_ids.size() * sizeof(int), cudaMemcpyDeviceToHost), "back ids");
    check(cudaMemcpy(h_w.data(), d_w, h_w.size() * sizeof(float), cudaMemcpyDeviceToHost), "back w");

    long long id_bad = 0, w_bad = 0;
    double worst = 0;
    for (size_t i = 0; i < h_ids.size(); ++i) {
        if (h_ids[i] != r_ids[i]) ++id_bad;
        const double rel = std::fabs((double) h_w[i] - (double) r_w[i]) /
                           (std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (!(rel <= tol)) ++w_bad;
    }
    // every token's weights must sum to 1 unless the clamp replaced the sum
    double worst_sum = 0;
    for (int t = 0; t < n_tokens; ++t) {
        double s = 0;
        for (int i = 0; i < k; ++i) s += (double) h_w[(size_t) t * k + i];
        // both sides divide by the same clamped scale, so the ratios agree; check the kernel's own sum
        worst_sum = std::max(worst_sum, std::fabs(s - 1.0));
    }
    std::printf("  %-26s ids %s (%lld bad)   weights worst rel %.3e (%lld over tol)   |sum-1| %.1e\n", name,
                id_bad ? "*** WRONG ***" : "exact", id_bad, worst, w_bad, worst_sum);
    cudaFree(d_l);
    cudaFree(d_ids);
    cudaFree(d_w);
    return (int) (id_bad + w_bad);
}

// ---- the NATIVE router (src/kernels/cuda/native_router.cu) ----------------------------------------------
// Two entries - one row, and a multi-row launch - with a documented promise that the multi-row call yields
// "each row exactly as the single call".  That promise is what a pinned row stride broke: on a 256-expert pack
// the multi-row launch read logits 2x too far, so every row past the first routed to the wrong experts while
// the single-row launch stayed correct.  A single-row, 512-expert check could not have seen it, so this checks
// at BOTH expert counts: agreement with the host reference, the multi-vs-single bit-identity, and repeatability.
int run_native_case(const char* name, const std::vector<float>& logits, int n_tokens, int n_expert, int k,
                    double tol, cudaStream_t cs) {
    const size_t n_rows = (size_t) n_tokens * (size_t) k;
    std::vector<int> r_ids(n_rows), m_ids(n_rows), s_ids(n_rows), rep_ids(n_rows);
    std::vector<float> r_w(n_rows), m_w(n_rows), s_w(n_rows), rep_w(n_rows);
    for (int t = 0; t < n_tokens; ++t) {
        reference(&logits[(size_t) t * n_expert], n_expert, k, &r_ids[(size_t) t * k], &r_w[(size_t) t * k]);
    }

    float* d_l = nullptr;
    int *d_ids_m = nullptr, *d_ids_s = nullptr, *d_ids_r = nullptr;
    float *d_w_m = nullptr, *d_w_s = nullptr, *d_w_r = nullptr;
    check(cudaMalloc(&d_l, logits.size() * sizeof(float)), "native malloc logits");
    check(cudaMalloc(&d_ids_m, n_rows * sizeof(int)), "native malloc ids");
    check(cudaMalloc(&d_w_m, n_rows * sizeof(float)), "native malloc w");
    check(cudaMalloc(&d_ids_s, n_rows * sizeof(int)), "native malloc ids s");
    check(cudaMalloc(&d_w_s, n_rows * sizeof(float)), "native malloc w s");
    check(cudaMalloc(&d_ids_r, n_rows * sizeof(int)), "native malloc ids r");
    check(cudaMalloc(&d_w_r, n_rows * sizeof(float)), "native malloc w r");
    // The engine's stream is non-blocking (cudaStreamCreateWithFlags(..., cudaStreamNonBlocking)), which does
    // NOT synchronize with the legacy default stream - so the H2D copy must go on THIS stream.  A plain
    // cudaMemcpy here races the first launch and the tail of the buffer reads as garbage.
    check(cudaMemcpyAsync(d_l, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice, cs),
          "native copy");

    strata::kernels::native_router_top10_multi(d_l, d_ids_m, d_w_m, n_tokens, n_expert, (void*) cs);
    strata::kernels::native_router_top10_multi(d_l, d_ids_r, d_w_r, n_tokens, n_expert, (void*) cs);
    for (int t = 0; t < n_tokens; ++t) {   // row by row through the SINGLE entry
        strata::kernels::native_router_top10(d_l + (size_t) t * n_expert, d_ids_s + (size_t) t * k,
                                             d_w_s + (size_t) t * k, n_expert, (void*) cs);
    }
    check(cudaStreamSynchronize(cs), "native sync");
    check(cudaMemcpy(m_ids.data(), d_ids_m, n_rows * sizeof(int), cudaMemcpyDeviceToHost), "native back m ids");
    check(cudaMemcpy(m_w.data(), d_w_m, n_rows * sizeof(float), cudaMemcpyDeviceToHost), "native back m w");
    check(cudaMemcpy(s_ids.data(), d_ids_s, n_rows * sizeof(int), cudaMemcpyDeviceToHost), "native back s ids");
    check(cudaMemcpy(s_w.data(), d_w_s, n_rows * sizeof(float), cudaMemcpyDeviceToHost), "native back s w");
    check(cudaMemcpy(rep_ids.data(), d_ids_r, n_rows * sizeof(int), cudaMemcpyDeviceToHost), "native back r ids");
    check(cudaMemcpy(rep_w.data(), d_w_r, n_rows * sizeof(float), cudaMemcpyDeviceToHost), "native back r w");

    long long id_bad = 0, w_bad = 0, row_diff = 0, rep_diff = 0, s_id_bad = 0;
    double worst = 0;
    for (size_t i = 0; i < n_rows; ++i) {
        if (m_ids[i] != r_ids[i]) ++id_bad;
        if (s_ids[i] != r_ids[i]) ++s_id_bad;
        const double rel = std::fabs((double) m_w[i] - (double) r_w[i]) /
                           (std::fabs((double) r_w[i]) > 1e-30 ? std::fabs((double) r_w[i]) : 1e-30);
        worst = std::max(worst, rel);
        if (!(rel <= tol)) ++w_bad;
        // the documented contract: a multi-row row IS the single-row result, bit for bit
        if (m_ids[i] != s_ids[i] || std::memcmp(&m_w[i], &s_w[i], sizeof(float)) != 0) ++row_diff;
        if (m_ids[i] != rep_ids[i] || std::memcmp(&m_w[i], &rep_w[i], sizeof(float)) != 0) ++rep_diff;
    }
    // On any mismatch, show WHERE - a row-stride bug and a selection-order bug look nothing alike.
    if (row_diff || id_bad) {
        std::vector<int> bad_rows;
        for (int t = 0; t < n_tokens; ++t) {
            for (int r = 0; r < k; ++r) {
                const size_t i = (size_t) t * k + r;
                if (m_ids[i] != r_ids[i] || m_ids[i] != s_ids[i]) { bad_rows.push_back(t); break; }
            }
        }
        std::printf("      failing rows: %zu of %d (", bad_rows.size(), n_tokens);
        for (size_t j = 0; j < bad_rows.size() && j < 12; ++j) std::printf("%d ", bad_rows[j]);
        std::printf("%s)\n", bad_rows.size() > 12 ? "..." : "");
        for (size_t j = 0; j < bad_rows.size() && j < 3; ++j) {
            const int t = bad_rows[j];
            std::printf("      row %2d  ref ", t);
            for (int r = 0; r < k; ++r) std::printf("%d ", r_ids[(size_t) t * k + r]);
            std::printf(" | single ");
            for (int r = 0; r < k; ++r) std::printf("%d ", s_ids[(size_t) t * k + r]);
            std::printf(" | multi ");
            for (int r = 0; r < k; ++r) std::printf("%d ", m_ids[(size_t) t * k + r]);
            std::printf("   (multi sum %.6f)\n", (double) w_sum(m_w, (size_t) t * k, k));
        }
    }

    const bool ok = (id_bad == 0 && w_bad == 0 && row_diff == 0 && rep_diff == 0 && s_id_bad == 0);
    std::printf("  %-26s multi ids %s (%lld bad)  single ids %s (%lld bad)  multi-vs-single %s (%lld)  "
                "repeat %s (%lld)  w rel %.3e\n",
                name, id_bad ? "*** WRONG ***" : "exact", id_bad, s_id_bad ? "*** WRONG ***" : "exact",
                s_id_bad, row_diff ? "*** DIFFERS ***" : "bit-identical", row_diff,
                rep_diff ? "*** DIFFERS ***" : "bit-identical", rep_diff, worst);
    cudaFree(d_l);
    cudaFree(d_ids_m); cudaFree(d_w_m);
    cudaFree(d_ids_s); cudaFree(d_w_s);
    cudaFree(d_ids_r); cudaFree(d_w_r);
    return ok ? 0 : 1;
}

// An expert count the router has no layout for must be REFUSED, not silently served by the 512-expert kernel:
// the kernel is a public entry point with two documented layouts (512 -> 16 values per lane, 256 -> 8), and
// callers do validate - but a future call site passing, say, a 384-expert draft model would otherwise read past
// its own logits and return plausible junk without any error.
int run_guard_case(int n_expert, cudaStream_t cs) {
    float* d_l = nullptr;
    int* d_ids = nullptr;
    float* d_w = nullptr;
    check(cudaMalloc(&d_l, (size_t) n_expert * sizeof(float) + 4096), "guard malloc logits");
    check(cudaMalloc(&d_ids, 10 * sizeof(int)), "guard malloc ids");
    check(cudaMalloc(&d_w, 10 * sizeof(float)), "guard malloc w");
    bool threw_single = false, threw_multi = false;
    try {
        strata::kernels::native_router_top10(d_l, d_ids, d_w, n_expert, (void*) cs);
    } catch (const std::invalid_argument&) {
        threw_single = true;
    }
    try {
        strata::kernels::native_router_top10_multi(d_l, d_ids, d_w, 4, n_expert, (void*) cs);
    } catch (const std::invalid_argument&) {
        threw_multi = true;
    }
    std::printf("  %-26s (n_expert %3d) single %s   multi %s\n", "unsupported count refused", n_expert,
                threw_single ? "refused" : "*** SILENTLY SERVED ***",
                threw_multi ? "refused" : "*** SILENTLY SERVED ***");
    cudaFree(d_l);
    cudaFree(d_ids);
    cudaFree(d_w);
    return (threw_single && threw_multi) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    bool selftest = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--selftest") selftest = true;
        else { std::fprintf(stderr, "usage: router_top10_parity [--selftest]\n"); return 2; }
    }

    const int K = 10;            // the real top-k
    const int NT = 64;           // rows per multi-row launch
    cudaStream_t cs = nullptr;
    check(cudaStreamCreateWithFlags(&cs, cudaStreamNonBlocking), "stream");
    strata::kernels::native_router_set_enabled(true);
    int bad = 0;

    // Both expert counts the artifact can have.  256 is the pruned pack this fork serves.
    for (const int NE : {512, 256}) {
        std::printf("\n---- %d experts, %d tokens, top-%d ----\n", NE, NT, K);
        std::mt19937 rng(2024);
        std::normal_distribution<float> gauss(0.0f, 1.0f);

        // ---- random logits
        std::vector<float> a((size_t) NT * NE);
        for (auto& v : a) v = gauss(rng);
        bad += run_case("random normal", a, NT, NE, K, 1e-5);
        bad += run_native_case("random normal (native)", a, NT, NE, K, 1e-5, cs);

        // ---- ALL-EQUAL logits: every expert ties, so the "smallest index wins" rule is the ONLY thing deciding
        // the ids.  An unstable sort or a >= comparison in the maximum scan fails this and passes everything else.
        std::vector<float> b((size_t) NT * NE, 0.5f);
        bad += run_case("all equal (ties)", b, NT, NE, K, 1e-5);
        bad += run_native_case("all equal (ties, native)", b, NT, NE, K, 1e-5, cs);

        // ---- a few exact ties among otherwise distinct values
        std::vector<float> c((size_t) NT * NE);
        for (int t = 0; t < NT; ++t) {
            for (int e = 0; e < NE; ++e) c[(size_t) t * NE + e] = (e < 12) ? 2.0f : gauss(rng);
        }
        bad += run_case("12-way exact tie", c, NT, NE, K, 1e-5);
        bad += run_native_case("12-way exact tie (native)", c, NT, NE, K, 1e-5, cs);

        // ---- one dominant expert per row: the top-1 must be that expert and its weight must dominate.
        // The dominant index WALKS with the row, so a row whose logits are read from the wrong offset cannot
        // accidentally look correct - this is the distribution that exposes a bad row stride.
        std::vector<float> d((size_t) NT * NE, 0.0f);
        for (int t = 0; t < NT; ++t) d[(size_t) t * NE + (t % NE)] = 20.0f;
        bad += run_case("dominant expert (per row)", d, NT, NE, K, 1e-5);
        bad += run_native_case("dominant expert (native)", d, NT, NE, K, 1e-5, cs);

        // ---- THE CLAMP CANNOT TRIGGER, and that is provable rather than untested.
        // The sum of the top k of a probability vector over n outcomes is at least k/n (the minimum is the
        // uniform distribution).  Here k/n = 10/512 = 0.0195 (10/256 = 0.0391), hundreds of times the
        // 2**-14 = 6.1035e-05 clamp, so no input whatsoever can make this geometry's renormalisation clamp -
        // not an extreme one, not an adversarial one.  Asserting the bound is worth more than a test that
        // cannot reach it.
        const double min_possible = (double) K / (double) NE;
        std::printf("  %-26s top-%d sum >= k/n = %.5f, clamp is %.3e -> margin %.0fx  %s\n", "clamp reachability",
                    K, min_possible, RENORM_CLAMP, min_possible / RENORM_CLAMP,
                    min_possible > RENORM_CLAMP ? "(CLAMP IS UNREACHABLE, asserted)" : "*** REACHABLE ***");
        if (min_possible <= RENORM_CLAMP) {
            std::fprintf(stderr, "the clamp CAN trigger at %d experts, so it needs a test\n", NE);
            ++bad;
        }
    }

    std::printf("\n---- router entry points with an expert count they have no layout for ----\n");
    bad += run_guard_case(384, cs);
    bad += run_guard_case(128, cs);

    std::printf("\nrouter_top10: %d failures over 2 expert counts x 4 distributions x %d tokens (top-%d): "
                "generic + native single + native multi, plus the unsupported-count guards\n", bad, NT, K);
    if (bad) return 1;
    if (selftest) std::printf("router_top10_parity OK\n");
    return 0;
}
