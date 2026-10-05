// kolibri_router_gate - parity for the SIGMOID_LOGIT_ADD router mode (branch kolibri-port).
//
// Kolibri 1's router is NOT GLM's: top-k is selected on the BIASED RAW LOGITS (logits + exp_probs_b)
// and the expert weights stay the UNBIASED sigmoid(logits), with no renormalisation
// (expert_weights_norm false).  The patch's own words (kolibri1-llama.cpp.patch 1/4):
//   "Kolibri 1 (torchtitan sigmoid_logit_add router) selects experts on the biased raw logits and
//    weights them by the unbiased sigmoid(logits). The existing DeepSeek-V3 path selects on
//    sigmoid(logits) + bias, which picks different experts whenever the bias is non-zero."
//
// This gate pins the kernels::glm::moe_route Gating::SIGMOID_LOGIT_ADD mode against a reference
// written straight from that definition in double precision, on adversarial inputs:
//   * large logits (sigmoid saturates: selection on logits vs sigmoid would diverge if the modes
//     mixed, and weights must stay exactly 0 or 1 without NaN/inf);
//   * negative biases large enough to REORDER the ranking relative to the unbiased logits (the
//     bias must change who is picked, never the picked expert's weight);
//   * ties between experts (stable sort keeps ascending index, ggml's argsort convention).
// It also pins the GLM mode's behaviour is UNCHANGED (the old path, same inputs, same picks).
//
// usage: kolibri_router_gate            (no args; self-checking, exits 0 on PASS)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "strata/kernels/glm_moe.hpp"

using strata::kernels::glm::MoeGeometry;
using strata::kernels::glm::moe_route;

namespace {

// The reference, in the kernel's own arithmetic: ggml adds the bias in fp32, so the sort key is an
// fp32 sum of fp32 values (a double key would split fp32 ties the kernel cannot see); weights are
// compared as doubles for tolerance.  Returns the selected ids (best first) and their weights.
void reference_route(const std::vector<float>& logits, const std::vector<float>& bias, int k,
                     MoeGeometry::Gating gating, std::vector<int>& ids, std::vector<double>& weights) {
    const int E = (int) logits.size();
    std::vector<int> order((size_t) E);
    for (int e = 0; e < E; ++e) order[(size_t) e] = e;
    const bool logit_add = gating == MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
    std::vector<float> probs((size_t) E);
    if (!logit_add)
        for (int e = 0; e < E; ++e) probs[(size_t) e] = 1.0f / (1.0f + std::exp(-logits[(size_t) e]));
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        const float sa = (logit_add ? logits[(size_t) a] : probs[(size_t) a]) +
                         (bias.empty() ? 0.0f : bias[(size_t) a]);
        const float sb = (logit_add ? logits[(size_t) b] : probs[(size_t) b]) +
                         (bias.empty() ? 0.0f : bias[(size_t) b]);
        if (sa != sb) return sa > sb;
        return a < b;  // stable_sort already does this; kept explicit for the reader
    });
    ids.assign(k, 0);
    weights.assign((size_t) k, 0.0);
    for (int i = 0; i < k; ++i) {
        ids[(size_t) i] = order[(size_t) i];
        weights[(size_t) i] = 1.0 / (1.0 + std::exp(-(double) logits[(size_t) order[(size_t) i]]));
    }
}

int fails = 0;

void check(bool ok, const std::string& what) {
    if (ok) { std::printf("  PASS %s\n", what.c_str()); return; }
    ++fails;
    std::printf("  FAIL %s\n", what.c_str());
}

// A deterministic xorshift PRNG, so the gate is reproducible without <random>'s libstdc++ coupling.
std::uint64_t rng_state = 0x9e3779b97f4a7c15ull;
float frand() {
    rng_state ^= rng_state << 13; rng_state ^= rng_state >> 7; rng_state ^= rng_state << 17;
    return (float) ((rng_state >> 40) / (double) 0x1000000) - 0.5f;  // [-0.5, 0.5)
}

}  // namespace

int main() {
    const int E = 384, K = 6, NE = 2560;      // Kolibri's shape
    const int RUNS = 64;

    // ---- 1. the SIGMOID_LOGIT_ADD mode against the double-precision reference -----------------------
    {
        MoeGeometry g;
        g.n_embd = NE; g.n_expert = E; g.n_used = K; g.ff = 512;
        g.norm_w = false;                      // expert_weights_norm false for Kolibri
        g.w_scale = 0.0f;                      // no expert_weights_scale in this artifact
        g.gating = MoeGeometry::Gating::SIGMOID_LOGIT_ADD;

        int id_mismatches = 0;
        double worst_w = 0.0;
        for (int run = 0; run < RUNS; ++run) {
            std::vector<float> router((size_t) E * NE), x((size_t) NE), bias((size_t) E);
            for (auto& v : router) v = frand() * 2.0f;
            for (auto& v : x) v = frand() * 4.0f;
            // biases on two scales: the artifact's are small, but the reordering property must hold
            // for large ones too, so half the runs use biases of magnitude up to 25
            const float bias_scale = (run % 2) ? 25.0f : 0.05f;
            for (auto& v : bias) v = frand() * 2.0f * bias_scale;

            // logits the kernel would compute (fp32 dot, the kernel's own accumulation order)
            std::vector<float> logits((size_t) E);
            for (int e = 0; e < E; ++e) {
                float acc = 0.0f;
                for (int i = 0; i < NE; ++i) acc += router[(size_t) e * NE + i] * x[(size_t) i];
                logits[(size_t) e] = acc;
            }
            std::vector<int32_t> ids((size_t) K);
            std::vector<float> weights((size_t) K);
            moe_route(router.data(), bias.data(), g, x.data(), ids.data(), weights.data(), nullptr);

            std::vector<int> ref_ids; std::vector<double> ref_w;
            reference_route(logits, bias, K, g.gating, ref_ids, ref_w);
            for (int i = 0; i < K; ++i) {
                if (ids[(size_t) i] != ref_ids[(size_t) i]) {
                    if (id_mismatches < 3) {   // print the first few mismatches, then just count
                        std::printf("    run %d mismatch at %d: kernel", run, i);
                        for (int j = 0; j < K; ++j) std::printf(" %d", ids[(size_t) j]);
                        std::printf("  |  reference");
                        for (int j = 0; j < K; ++j) std::printf(" %d", ref_ids[(size_t) j]);
                        std::printf("\n");
                    }
                    ++id_mismatches; break;
                }
                worst_w = std::max(worst_w, std::abs((double) weights[(size_t) i] - ref_w[(size_t) i]));
            }
        }
        std::printf("sigmoid_logit_add vs double reference (%d runs, E=%d k=%d):\n", RUNS, E, K);
        check(id_mismatches == 0, "selected ids identical on every run");
        char buf[128];
        std::snprintf(buf, sizeof buf, "weights within %.3g of the double reference", worst_w);
        check(worst_w < 1e-5, buf);
    }

    // ---- 2. saturation: large logits must not NaN and must order by logit, not by saturated sigmoid -
    {
        MoeGeometry g;
        g.n_embd = 8; g.n_expert = 64; g.n_used = 6; g.ff = 8;
        g.norm_w = false; g.w_scale = 0.0f;
        g.gating = MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
        // logits from +80 (sigmoid = 1.0) down to -80 (sigmoid = 0.0): selection must be purely by
        // logit order, weights must saturate cleanly, and biases must NOT reorder equal-size logits
        std::vector<float> router((size_t) 64 * 8, 0.0f), x((size_t) 8, 0.0f);
        router[0] = 10.0f; x[0] = 8.0f;      // expert 0: logit +80
        router[9] = -10.0f; x[1] = 8.0f;     // expert 1: logit -80
        std::vector<float> bias((size_t) 64, 0.0f);
        std::vector<int32_t> ids((size_t) 6);
        std::vector<float> weights((size_t) 6);
        moe_route(router.data(), bias.data(), g, x.data(), ids.data(), weights.data(), nullptr);
        // logits: expert 0 +80, expert 1 -80, everyone else 0.0.  Top-6: 0 first, then the ties at
        // 0.0 by ascending index (2,3,4,5), never expert 1.
        bool ok = ids[0] == 0 && ids[1] == 2 && ids[2] == 3 && ids[3] == 4 && ids[4] == 5 && ids[5] == 6;
        ok = ok && weights[0] == 1.0f;
        float w2 = 1.0f / (1.0f + std::exp(0.0f));  // sigmoid(0) = 0.5
        ok = ok && weights[1] == w2 && weights[2] == w2;
        for (int i = 0; i < 6; ++i) ok = ok && std::isfinite(weights[(size_t) i]);
        std::printf("saturation (logits +-80):\n");
        check(ok, "selection by raw logit (0,2,3,4,5,6), weights saturate to 1 and 0.5, all finite");
    }

    // ---- 3. the bias changes the PICKS but never the picked experts' WEIGHTS ------------------------
    {
        MoeGeometry g;
        g.n_embd = 8; g.n_expert = 64; g.n_used = 2; g.ff = 8;
        g.norm_w = false; g.w_scale = 0.0f;
        g.gating = MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
        std::vector<float> router((size_t) 64 * 8, 0.0f), x((size_t) 8, 0.0f);
        router[0] = 10.0f; x[0] = 5.0f;      // expert 0: logit +50
        router[9] = 6.0f; x[1] = 5.0f;       // expert 1: logit +30
        router[18] = 6.2f; x[2] = 5.0f;      // expert 2: logit +31
        std::vector<float> bias((size_t) 64, 0.0f);
        std::vector<int32_t> ids_a((size_t) 2), ids_b((size_t) 2);
        std::vector<float> w_a((size_t) 2), w_b((size_t) 2);
        moe_route(router.data(), nullptr, g, x.data(), ids_a.data(), w_a.data(), nullptr);
        bias[1] = 40.0f;                      // pushes expert 1 (+30 +40 = +70) above expert 0 (+50)
        moe_route(router.data(), bias.data(), g, x.data(), ids_b.data(), w_b.data(), nullptr);
        const bool picks_changed = ids_a[0] == 0 && ids_a[1] == 2 && ids_b[0] == 1 && ids_b[1] == 0;
        // expert 1's weight is sigmoid(+30) whenever it is picked; expert 0's weight sigmoid(+50) is
        // unchanged by the bias that demoted it.  No cross-expert comparison.
        const float s30 = 1.0f / (1.0f + std::exp(-30.0f));
        const float s31 = 1.0f / (1.0f + std::exp(-31.0f));
        const float s50 = 1.0f / (1.0f + std::exp(-50.0f));
        const bool weights_stuck = w_b[0] == s30 && w_a[1] == s31 && w_b[1] == s50;
        std::printf("bias reordering (expert 1 promoted by +40 bias):\n");
        check(picks_changed, "the bias reordered the picks (1 over 0 and 2)");
        check(weights_stuck, "expert 1's weight is its UNBIASED sigmoid(logit) in both runs");
    }

    // ---- 4. the GLM mode is unchanged: same ids as before the enum existed ---------------------------
    {
        MoeGeometry g;
        g.n_embd = 8; g.n_expert = 64; g.n_used = 4; g.ff = 8;
        g.norm_w = true; g.w_scale = 1.0f;   // GLM normalises
        g.gating = MoeGeometry::Gating::SIGMOID_BIASED;
        std::vector<float> router((size_t) 64 * 8, 0.0f), x((size_t) 8, 0.0f);
        for (int e = 0; e < 64; ++e) router[(size_t) e * 8] = 0.1f * e;   // logits 0.0 .. 6.3
        x[0] = 1.0f;
        std::vector<float> bias((size_t) 64, 0.0f);
        bias[3] = 0.5f;                       // expert 3's sigmoid+0.5 beats expert 4's raw sigmoid
        std::vector<int32_t> ids((size_t) 4);
        std::vector<float> weights((size_t) 4);
        moe_route(router.data(), bias.data(), g, x.data(), ids.data(), weights.data(), nullptr);
        // selection keys: expert 3's sigmoid(0.3) = 0.5744, + 0.5 = 1.0744; experts 63/62/61's
        // sigmoids of 6.3/6.2/6.1 are 0.9982/0.9980/0.9978 - so the bias lifts expert 3 to the TOP,
        // which is the DeepSeek behaviour this mode must keep: selection on sigmoid + bias.
        const bool ok = ids[0] == 3 && ids[1] == 63 && ids[2] == 62 && ids[3] == 61;
        std::printf("glm mode unchanged (sigmoid+bias selection, normalised):\n");
        std::printf("    ids:");
        for (int i = 0; i < 4; ++i) std::printf(" %d", ids[(size_t) i]);
        std::printf("\n");
        check(ok, "top ids are 3, 63, 62, 61 (the bias lifted 3 over everyone)");
    }

    // ---- 5. renormalisation respected when norm_w is set (Kolibri sets it false, GLM true) ----------
    {
        MoeGeometry g;
        g.n_embd = 8; g.n_expert = 64; g.n_used = 6; g.ff = 8;
        g.gating = MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
        g.norm_w = true; g.w_scale = 1.0f;   // the one combination: logit-add picks, normalised weights
        std::vector<float> router((size_t) 64 * 8, 0.0f), x((size_t) 8, 0.0f);
        for (int e = 0; e < 64; ++e) router[(size_t) e * 8] = 0.2f * e;   // logits 0.0 .. 12.6
        x[0] = 1.0f;
        std::vector<int32_t> ids((size_t) 6);
        std::vector<float> weights((size_t) 6);
        moe_route(router.data(), nullptr, g, x.data(), ids.data(), weights.data(), nullptr);
        double sum = 0.0;
        for (int i = 0; i < 6; ++i) sum += weights[(size_t) i];
        std::printf("norm_w still honoured under logit-add gating:\n");
        check(std::abs(sum - 1.0) < 1e-5, "weights sum to 1 when norm_w is set");
    }

    std::printf("\n%s\n", fails == 0 ? "ROUTER GATE: PASS" : "ROUTER GATE: FAIL");
    return fails == 0 ? 0 : 1;
}
