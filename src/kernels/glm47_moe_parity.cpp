// src/kernels/glm47_moe_parity.cpp - does our router/MoE site (deepseek2 SIGMOID gating) match the reference?
//
// The oracle is tools/glm47_moe_reference.py, transcribed from the SIGMOID branch of llama.cpp/Kolibri's
// shared build_moe_ffn as src/models/deepseek2.cpp calls it for GLM-4.7-Flash (47 layers, vocab 154880,
// no expert_gating_func key -> SIGMOID):
//
//   python tools/glm47_moe_reference.py --gguf <model.gguf> --layer 1 --raw-fixture D:/tmp/glm47_moe.bin
//   build/glm47_moe_parity.exe D:/tmp/glm47_moe.bin
//
// The fixture carries the model's OWN router, exp_probs_b, shared expert and the selected routed experts
// (float32), plus random hidden states and the float64 oracle's expected ids/weights/probs/moe/shexp/out.
//
// THE GATE: the top-4 expert IDs are EXACT integers against the reference on every case, the routing
// weights match to < 1e-5, and the MoE / shared-expert / combine outputs match to < 1e-4.
//
// The gate also proves it has TEETH: a fixture whose selection is unchanged without the bias, or under
// Kolibri's bias-then-sigmoid (SIGMOID_LOGIT_ADD) order, could not catch a port that dropped or swapped
// the bias.  So the gate recomputes both alternatives itself and requires the fixture to tell them apart.
#include "strata/kernels/glm_moe.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace glm = strata::kernels::glm;
namespace cglm = strata::core::glm;

/// The pack reader as the native stage wants it: a plain blob_fn (ExpertSource::blob in production).
const uint8_t* moe_blob_adapter(void* ctx, int layer, int expert) {
    return ((strata::core::ExpertSource*) ctx)->blob(layer, expert);
}

namespace {

constexpr uint32_t MAGIC = 0x474D3437u;   // 'GM47'
constexpr uint32_t VERSION = 1;

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

void read_into(std::FILE* f, void* dst, size_t bytes) {
    if (bytes && std::fread(dst, 1, bytes, f) != bytes) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
}

int32_t read_i32(std::FILE* f) {
    int32_t v = 0;
    read_into(f, &v, sizeof v);
    return v;
}

double max_abs(const std::vector<float>& got, const std::vector<float>& want) {
    double m = 0.0;
    for (size_t i = 0; i < got.size(); ++i) m = std::max(m, std::fabs((double) got[i] - (double) want[i]));
    return m;
}

double sigmoid64(double x) { return 1.0 / (1.0 + std::exp(-x)); }

/// Descending by score, ties by ascending index - ggml's argsort_top_k order.
std::vector<int> top_k(const std::vector<double>& score, int k) {
    std::vector<int> idx(score.size());
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](int a, int b) { return score[(size_t) a] > score[(size_t) b]; });
    idx.resize((size_t) k);
    return idx;
}

struct Stage {
    const char* name;
    double worst = 0.0;
    double tol;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm47_moe_parity <fixture.bin>\n"
                    "  build the fixture with: python tools/glm47_moe_reference.py"
                    " [--gguf <model.gguf> --layer 1] --raw-fixture <fixture.bin>\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    const uint32_t magic = (uint32_t) read_i32(f), version = (uint32_t) read_i32(f);
    if (magic != MAGIC) { std::fprintf(stderr, "bad fixture magic 0x%08x\n", (unsigned) magic); return 1; }
    if (version != VERSION) { std::fprintf(stderr, "unsupported fixture version %u\n", (unsigned) version); return 1; }

    glm::MoeGeometry g;
    g.n_embd = read_i32(f);
    g.n_expert = read_i32(f);
    g.n_used = read_i32(f);
    const int32_t n_shared = read_i32(f);
    g.ff = read_i32(f);
    float w_scale = 0.0f;
    read_into(f, &w_scale, sizeof w_scale);
    g.w_scale = w_scale;
    g.norm_w = read_i32(f) != 0;
    const int32_t n_cases = read_i32(f);
    if (g.n_embd <= 0 || g.n_expert <= 0 || g.n_used <= 0 || g.ff <= 0 || n_shared < 0 || n_cases < 0) {
        std::fprintf(stderr, "bad header geometry\n");
        return 1;
    }
    const int ne = g.n_embd, E = g.n_expert, k = g.n_used, ff = g.ff;

    std::vector<float> router = read_floats(f, (size_t) E * ne);
    std::vector<float> probs_b = read_floats(f, (size_t) E);
    std::vector<float> s_gate = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_up = read_floats(f, (size_t) ff * ne);
    std::vector<float> s_down = read_floats(f, (size_t) ne * ff);
    const float* shared[3] = {s_gate.data(), s_up.data(), s_down.data()};

    std::printf("glm47 router/MoE parity vs tools/glm47_moe_reference.py\n");
    std::printf("  geometry: n_embd %d, %d experts, top-%d, ff %d, +%d shared, w_scale %.1f, norm_w %s\n",
                ne, E, k, ff, (int) n_shared, (double) g.w_scale, g.norm_w ? "true" : "false");
    std::printf("  gating: SIGMOID - probs = sigmoid(logits), selection on probs + exp_probs_b, "
                "weights from the UNBIASED probs\n");
    std::printf("  %d case(s)\n", (int) n_cases);

    Stage stages[4] = {{"weights", 0.0, 1e-5}, {"moe", 0.0, 1e-4}, {"shexp", 0.0, 1e-4}, {"out", 0.0, 1e-4}};
    double probs_worst = 0.0, ids_ok_all = 0;
    double sum_worst = 0.0;
    bool pos_ok = true, distinct_ok = true, range_ok = true;
    bool ok = true;
    int bias_teeth = 0, mode_teeth = 0;
    // case 0's data, kept for the native (pack) path after the loop
    std::vector<float> nat_x, nat_got, nat_want;
    std::vector<int32_t> nat_ids;

    for (int c = 0; c < n_cases; ++c) {
        const std::vector<float> x = read_floats(f, (size_t) ne);
        std::vector<std::vector<float>> ex_gate((size_t) k), ex_up((size_t) k), ex_down((size_t) k);
        for (int i = 0; i < k; ++i) {
            ex_gate[(size_t) i] = read_floats(f, (size_t) ff * ne);
            ex_up[(size_t) i] = read_floats(f, (size_t) ff * ne);
            ex_down[(size_t) i] = read_floats(f, (size_t) ne * ff);
        }
        std::vector<int32_t> e_ids((size_t) k, -1);
        read_into(f, e_ids.data(), (size_t) k * sizeof(int32_t));
        const std::vector<float> e_weights = read_floats(f, (size_t) k);
        const std::vector<float> e_probs = read_floats(f, (size_t) E);
        const std::vector<float> e_moe = read_floats(f, (size_t) ne);
        const std::vector<float> e_shexp = read_floats(f, (size_t) ne);
        const std::vector<float> e_out = read_floats(f, (size_t) ne);

        std::vector<const float*> expert_ptrs((size_t) k * 3);
        std::vector<const float* const*> experts((size_t) k);
        for (int i = 0; i < k; ++i) {
            expert_ptrs[(size_t) i * 3 + 0] = ex_gate[(size_t) i].data();
            expert_ptrs[(size_t) i * 3 + 1] = ex_up[(size_t) i].data();
            expert_ptrs[(size_t) i * 3 + 2] = ex_down[(size_t) i].data();
            experts[(size_t) i] = &expert_ptrs[(size_t) i * 3];
        }

        std::vector<int32_t> got_ids((size_t) k, -1), nobias_ids((size_t) k, -1);
        std::vector<float> got_weights((size_t) k), nobias_w((size_t) k), got_probs((size_t) E);
        std::vector<float> got_moe((size_t) ne), got_shexp((size_t) ne), got_out((size_t) ne);
        glm::moe_forward(router.data(), probs_b.data(), g, x.data(), experts.data(), shared, got_out.data(),
                         got_ids.data(), got_weights.data(), got_moe.data(), got_shexp.data());
        glm::moe_route(router.data(), probs_b.data(), g, x.data(), got_ids.data(), got_weights.data(),
                       got_probs.data());
        // the same engine router with the bias DROPPED: an input that selects the same experts cannot
        // catch a port that ignores exp_probs_b
        glm::moe_route(router.data(), nullptr, g, x.data(), nobias_ids.data(), nobias_w.data(), nullptr);

        // Kolibri's SIGMOID_LOGIT_ADD order (bias-then-sigmoid), in float64: an input where this agrees
        // with the SIGMOID order cannot catch a port that swapped the two
        std::vector<double> logits((size_t) E), alt_sel((size_t) E);
        for (int e = 0; e < E; ++e) {
            double acc = 0.0;
            const float* row = router.data() + (size_t) e * ne;
            for (int i = 0; i < ne; ++i) acc += (double) row[i] * (double) x[(size_t) i];
            logits[(size_t) e] = acc;
            alt_sel[(size_t) e] = sigmoid64(acc + (double) probs_b[(size_t) e]);
        }
        const std::vector<int> alt_ids = top_k(alt_sel, k);

        int mism = 0, noflip = 0;
        for (int i = 0; i < k; ++i) {
            mism += (got_ids[(size_t) i] != e_ids[(size_t) i]);
            noflip += (got_ids[(size_t) i] != nobias_ids[(size_t) i]);
        }
        bool mode_diff = false;
        for (int i = 0; i < k; ++i) mode_diff = mode_diff || (got_ids[(size_t) i] != alt_ids[(size_t) i]);
        bias_teeth += (noflip > 0);
        mode_teeth += mode_diff;

        const double d_w = max_abs(got_weights, e_weights);
        const double d_p = max_abs(got_probs, e_probs);
        const double d_moe = max_abs(got_moe, e_moe);
        const double d_shexp = max_abs(got_shexp, e_shexp);
        const double d_out = max_abs(got_out, e_out);
        stages[0].worst = std::max(stages[0].worst, d_w);
        stages[1].worst = std::max(stages[1].worst, d_moe);
        stages[2].worst = std::max(stages[2].worst, d_shexp);
        stages[3].worst = std::max(stages[3].worst, d_out);
        probs_worst = std::max(probs_worst, d_p);
        ids_ok_all += (mism == 0);
        ok = ok && (mism == 0);

        // invariants that do not depend on the oracle agreeing: normalised then scaled, so the sum is
        // exactly w_scale (scaling before normalising would leave a sum of 1); all positive; ids
        // distinct and in range.
        {
            double sum = 0.0;
            for (int i = 0; i < k; ++i) { sum += (double) got_weights[(size_t) i]; pos_ok = pos_ok && got_weights[(size_t) i] > 0.0f; }
            sum_worst = std::max(sum_worst, std::fabs(sum - (double) g.w_scale));
            std::vector<int32_t> sorted(got_ids.begin(), got_ids.end());
            std::sort(sorted.begin(), sorted.end());
            distinct_ok = distinct_ok && (std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
            range_ok = range_ok && sorted.front() >= 0 && sorted.back() < E;
        }

        if (c == 0) { nat_x = x; nat_ids = e_ids; nat_got = got_out; nat_want = e_out; }

        std::printf("  case %d: ids", c);
        for (int i = 0; i < k; ++i) std::printf(" %d", (int) got_ids[(size_t) i]);
        std::printf("  %s (want", mism ? "FAIL" : "PASS");
        for (int i = 0; i < k; ++i) std::printf(" %d", (int) e_ids[(size_t) i]);
        std::printf(")  |  bias flips ids %s  bias-then-sigmoid flips ids %s  |  "
                    "w %.1e p %.1e moe %.1e shexp %.1e out %.1e\n",
                    noflip ? "yes" : "no", mode_diff ? "yes" : "no",
                    d_w, d_p, d_moe, d_shexp, d_out);
    }
    std::fclose(f);

    std::printf("  stage    max abs error (worst over cases)   tolerance\n");
    for (const Stage& s : stages) {
        const bool good = s.worst < s.tol;
        std::printf("  %-8s %.3e                    < %.0e   %s\n", s.name, s.worst, s.tol,
                    good ? "PASS" : "FAIL");
        ok = ok && good;
    }
    const bool probs_good = probs_worst < 1e-5;
    std::printf("  %-8s %.3e                    < 1e-05   %s   (unbiased sigmoid probs)\n", "probs",
                probs_worst, probs_good ? "PASS" : "FAIL");
    ok = ok && probs_good;
    std::printf("  ids      %d/%d cases exact                                %s\n", (int) ids_ok_all,
                (int) n_cases, ids_ok_all == n_cases ? "PASS" : "FAIL");

    // Invariants independent of the oracle: normalised THEN scaled, so the weights sum to w_scale
    // (scaling before normalising would leave a sum of 1) - the second silent ordering.
    {
        const bool inv = sum_worst < 1e-4 && pos_ok && distinct_ok && range_ok;
        std::printf("  invariant sum(weights) == w_scale (normalised THEN scaled) worst |delta| %.2e   "
                    "positive %s   ids distinct %s   in range %s   %s\n", sum_worst,
                    pos_ok ? "PASS" : "FAIL", distinct_ok ? "PASS" : "FAIL", range_ok ? "PASS" : "FAIL",
                    inv ? "PASS" : "FAIL");
        ok = ok && inv;
    }
    // Teeth: a fixture whose selection is unchanged without the bias, or that cannot tell the two bias
    // orders apart, would pass a port that dropped or swapped the bias.
    {
        const bool teeth_ok = (bias_teeth > 0) && (mode_teeth > 0);
        std::printf("  teeth    %d/%d case(s) change ids when exp_probs_b is dropped; %d/%d change ids "
                    "under bias-then-sigmoid (SIGMOID_LOGIT_ADD)   %s\n",
                    bias_teeth, (int) n_cases, mode_teeth, (int) n_cases, teeth_ok ? "PASS" : "FAIL");
        ok = ok && teeth_ok;
    }

    // ============ THE PRODUCTION PATH: the same input and ids, QUANTIZED BLOBS FROM THE PACK ============
    //
    // Everything above compares glm::moe_forward - FLOAT experts - against the oracle.  That is not the path
    // the model runs: the trunk hands glm_stage_moe_native QUANTIZED BYTES FROM A PACK.  So drive the native
    // stage with the pack's OWN blobs for the ids case 0 selected, on case 0's input, and report the result
    // against the float64 oracle and against the float engine path.  A gap far beyond what the pack's expert
    // format can explain would be a real defect; the ids must be identical (same router).
    if (argc < 3) {
        std::printf("  native   no pack named (arg 2) - the pack path is skipped; pass a pack dir to exercise it\n");
    } else {
        const char* pack = argv[2];
        const int layer = (argc >= 4) ? std::atoi(argv[3]) : 1;
        int gu_type = -1, d_type = -1;
        {
            std::ifstream pf(std::string(pack) + "/native_experts.txt");
            std::string line;
            while (std::getline(pf, line)) {
                if (line.empty() || line[0] == '#') continue;
                std::istringstream is(line);
                int64_t l = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0;
                int gt = 0, dt = 0;
                if (!(is >> l >> gt >> dt >> off >> nb >> go >> uo >> dob)) continue;
                if (l == (int64_t) layer) { gu_type = gt; d_type = dt; break; }
            }
        }
        std::string nerr;
        strata::kernels::cpu::NativeFmt fmt;
        strata::core::FileExpertSource src;
        // this pack keeps no experts.bin: they are read IN PLACE from the model GGUF, so name the --native shard
        const char* gguf = (argc >= 5) ? argv[4] : "D:/aimodels/Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf";
        src.set_gguf(gguf);
        if (gu_type < 0) {
            std::printf("  native   layer %d is not in %s/native_experts.txt (skipped)\n", layer, pack);
        } else if (!strata::kernels::cpu::expert_layout_load(pack, 47, E, nerr, ne, ff) ||
                   !src.open(pack, 47, E, nerr)) {
            std::printf("  native   the pack will not open: %s (skipped)\n", nerr.c_str());
        } else if (!strata::kernels::cpu::native_fmt(gu_type, d_type, ne, ff, fmt, nerr)) {
            std::printf("  native   native_fmt(%d,%d) refused: %s (skipped)\n", gu_type, d_type, nerr.c_str());
        } else {
            glm::MoeGeometry sg;
            sg.n_embd = ne; sg.ff = ff; sg.n_expert = E; sg.n_used = k;
            sg.w_scale = g.w_scale; sg.norm_w = g.norm_w;
            sg.clamp_exp = g.clamp_exp; sg.clamp_shexp = g.clamp_shexp;
            std::vector<float> nat_out((size_t) ne, 0.0f);
            std::vector<int32_t> nids((size_t) k, -1);
            // shared_types = nullptr -> the +1 shared expert runs through the float expert_ffn fallback
            const bool nat_ok = cglm::glm_stage_moe_native(
                nat_x.data(), router.data(), probs_b.data(), g, layer, fmt, &moe_blob_adapter, &src,
                &sg, shared, nullptr, g.clamp_shexp, nat_out.data(), nerr, nids.data());
            if (!nat_ok) {
                std::printf("  native   the stage refused: %s\n", nerr.c_str());
            } else {
                bool ids_same = true;
                for (int i = 0; i < k; ++i) ids_same = ids_same && (nids[(size_t) i] == nat_ids[(size_t) i]);
                double dn = 0.0, df = 0.0, rr = 0.0;
                for (int i = 0; i < ne; ++i) {
                    dn = std::max(dn, (double) std::fabs((double) nat_out[(size_t) i] - (double) nat_want[(size_t) i]));
                    df = std::max(df, (double) std::fabs((double) nat_out[(size_t) i] - (double) nat_got[(size_t) i]));
                    rr += (double) nat_want[(size_t) i] * (double) nat_want[(size_t) i];
                }
                const double rms = std::sqrt(rr / (double) ne);
                std::printf("  native   pack blobs (Q4_K/Q6_K via ggml-cpu), layer %d: ids %s\n",
                            layer, ids_same ? "IDENTICAL to the oracle's" : "DIFFERENT");
                std::printf("  native   vs oracle (float64) worst %.4g = %.3g of rms   "
                            "vs float engine worst %.4g = %.3g of rms\n",
                            dn, dn / (rms > 0 ? rms : 1.0), df, df / (rms > 0 ? rms : 1.0));
                std::printf("  native   %s   (a gap of a few %% of rms vs the float path is the expert quantization;\n"
                            "           far more would be a defect in the stage)\n", ids_same ? "PASS" : "FAIL");
                ok = ok && ids_same;
            }
        }
    }

    std::printf("glm47_moe_parity: %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
