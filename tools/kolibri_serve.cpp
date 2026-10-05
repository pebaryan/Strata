// strata-kolibri - Kolibri-1 serving entrypoint for the Strata engine.
//
// This is the validated Kolibri trunk used by kolibri_multitoken_gate, made stateful and connected
// to Strata's line protocol (READY / GEN / T / DONE).  It deliberately keeps the graph in one place:
// changes to attention, routing or the sandwich norms can be parity-gated before they reach serving.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "strata/artifact/gguf_reader.hpp"
#include "strata/artifact/dequant.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/kolibri_swa.hpp"

using strata::GgufModel;
using strata::TensorInfo;

namespace {

constexpr int N_EMBD = 2560, N_HEAD = 48, N_KV = 4, HEAD_DIM = 128;
constexpr int FF = 512, N_EXPERT = 384, N_USED = 6;

void dequant_row(const GgufModel& m, const std::string& name, int64_t row, int n, float* out) {
    size_t shard = 0;
    const TensorInfo* t = m.find(name, &shard);
    if (!t) { std::fprintf(stderr, "missing tensor %s\n", name.c_str()); std::exit(1); }
    const uint8_t* base = m.shard(shard).tensor_data(*t);
    int be = 0, bb = 0;
    if (!strata::block_geometry(t->type, be, bb)) { std::fprintf(stderr, "bad type %d for %s\n", t->type, name.c_str()); std::exit(1); }
    int64_t total = 1;
    for (uint64_t d : t->shape) total *= (int64_t) d;
    const int64_t row_elems = (int64_t) t->shape[0];
    if (n != row_elems && !(row == 0 && n == total)) {
        std::fprintf(stderr, "row length %d != ne0 %lld (total %lld) for %s\n", n,
                     (long long) row_elems, (long long) total, name.c_str());
        std::exit(1);
    }
    const uint8_t* row_base = base + (uint64_t) row * ((uint64_t) n / be) * bb;
    for (int b = 0; b < n / be; ++b) {
        float* o = out + (size_t) b * be;
        switch (t->type) {
            case 0:  std::memcpy(o, row_base + (size_t) b * bb, (size_t) be * 4); break;
            case 8:  strata::dequantize_q8_0(row_base + (size_t) b * bb, o); break;
            case 12: strata::dequantize_q4_K(row_base + (size_t) b * bb, o); break;
            case 14: strata::dequantize_q6_K(row_base + (size_t) b * bb, o); break;
            default: std::fprintf(stderr, "unhandled type %d for %s\n", t->type, name.c_str()); std::exit(1);
        }
    }
}

void gemv(const std::vector<float>& w, int out_n, int in_n, const float* x, float* y) {
    for (int o = 0; o < out_n; ++o) {
        float acc = 0.0f;
        const float* row = w.data() + (size_t) o * in_n;
        for (int i = 0; i < in_n; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

// one layer, one token, at `pos`; k/v are APPENDED to the layer's cache by the caller
void layer_forward(const GgufModel& m, const strata::Kolibri1Geometry& geo, int b, int pos,
                   std::vector<float>& x, std::vector<float>& k_cache, std::vector<float>& v_cache,
                   std::vector<float>& w_q, std::vector<float>& w_k, std::vector<float>& w_v,
                   std::vector<float>& w_o, std::vector<float>& w_gate, std::vector<float>& w_up,
                   std::vector<float>& router, std::vector<float>& e_gate, std::vector<float>& e_up,
                   std::vector<float>& e_h, std::vector<float>& e_down) {
    const std::string p = "blk." + std::to_string(b) + ".";
    const bool sliding = b < (int) geo.swa_pattern.size() && geo.swa_pattern[(size_t) b] != 0;
    std::vector<float> nrm((size_t) N_EMBD), po((size_t) N_EMBD), wn((size_t) N_EMBD);

    // attn sandwich
    strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, geo.rms_eps, nrm.data());
    dequant_row(m, p + "attn_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) nrm[(size_t) i] *= wn[(size_t) i];

    dequant_row(m, p + "attn_q.weight", 0, N_HEAD * HEAD_DIM * N_EMBD, w_q.data());
    dequant_row(m, p + "attn_k.weight", 0, N_KV * HEAD_DIM * N_EMBD, w_k.data());
    dequant_row(m, p + "attn_v.weight", 0, N_KV * HEAD_DIM * N_EMBD, w_v.data());
    std::vector<float> q((size_t) N_HEAD * HEAD_DIM), k((size_t) N_KV * HEAD_DIM), v((size_t) N_KV * HEAD_DIM);
    gemv(w_q, N_HEAD * HEAD_DIM, N_EMBD, nrm.data(), q.data());
    gemv(w_k, N_KV * HEAD_DIM, N_EMBD, nrm.data(), k.data());
    gemv(w_v, N_KV * HEAD_DIM, N_EMBD, nrm.data(), v.data());

    std::vector<float> qn((size_t) N_HEAD * HEAD_DIM), kn((size_t) N_KV * HEAD_DIM);
    {
        std::vector<float> qw((size_t) HEAD_DIM), kw((size_t) HEAD_DIM);
        dequant_row(m, p + "attn_q_norm.weight", 0, HEAD_DIM, qw.data());
        dequant_row(m, p + "attn_k_norm.weight", 0, HEAD_DIM, kw.data());
        strata::kernels::kolibri::qk_norm_per_head(q.data(), qw.data(), N_HEAD, HEAD_DIM, geo.rms_eps, qn.data());
        strata::kernels::kolibri::qk_norm_per_head(k.data(), kw.data(), N_KV, HEAD_DIM, geo.rms_eps, kn.data());
    }
    if (sliding) {   // NoPE on full layers
        strata::kernels::kolibri::rope_neox(qn.data(), pos, N_HEAD, HEAD_DIM, geo.rope_freq_base, HEAD_DIM);
        strata::kernels::kolibri::rope_neox(kn.data(), pos, N_KV, HEAD_DIM, geo.rope_freq_base, HEAD_DIM);
    }
    std::memcpy(k_cache.data() + (size_t) pos * N_KV * HEAD_DIM, kn.data(), (size_t) N_KV * HEAD_DIM * 4);
    std::memcpy(v_cache.data() + (size_t) pos * N_KV * HEAD_DIM, v.data(), (size_t) N_KV * HEAD_DIM * 4);

    // causal attention.  Sliding layers retain exactly the last `swa_window` cells; full layers
    // are unbounded and NoPE, matching the artifact's SSSS-F pattern.
    const int first = sliding ? std::max(0, pos - (int) geo.swa_window + 1) : 0;
    std::vector<float> att_out((size_t) N_HEAD * HEAD_DIM, 0.0f);
    for (int h = 0; h < N_HEAD; ++h) {
        const int hk = h / (N_HEAD / N_KV);
        double maxi = -1e300;
        std::vector<double> sc((size_t) (pos - first + 1));
        for (int t = first; t <= pos; ++t) {
            double d = 0.0;
            const float* kc = k_cache.data() + (size_t) t * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; ++i) d += (double) qn[(size_t) h * HEAD_DIM + i] * kc[i];
            sc[(size_t) (t - first)] = d / std::sqrt((double) HEAD_DIM);
            maxi = std::max(maxi, sc[(size_t) (t - first)]);
        }
        double sum = 0.0;
        for (double& score : sc) { score = std::exp(score - maxi); sum += score; }
        for (int i = 0; i < HEAD_DIM; ++i) {
            double acc = 0.0;
            for (int t = first; t <= pos; ++t)
                acc += sc[(size_t) (t - first)] * (double) v_cache[(size_t) t * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
            att_out[(size_t) h * HEAD_DIM + i] = (float) (acc / sum);
        }
    }
    dequant_row(m, p + "attn_output.weight", 0, N_EMBD * N_HEAD * HEAD_DIM, w_o.data());
    std::vector<float> wo_out((size_t) N_EMBD);
    gemv(w_o, N_EMBD, N_HEAD * HEAD_DIM, att_out.data(), wo_out.data());
    strata::kernels::kolibri::rms_norm(wo_out.data(), nullptr, N_EMBD, geo.rms_eps, po.data());
    dequant_row(m, p + "post_attention_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= wn[(size_t) i];
    for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];

    // ffn sandwich
    strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, geo.rms_eps, nrm.data());
    dequant_row(m, p + "ffn_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) nrm[(size_t) i] *= wn[(size_t) i];

    {
        size_t s = 0;
        const TensorInfo* t = m.find(p + "ffn_gate_inp.weight", &s);
        std::memcpy(router.data(), m.shard(s).tensor_data(*t), (size_t) N_EXPERT * N_EMBD * 4);
        t = m.find(p + "exp_probs_b.bias", &s);
        std::memcpy(wn.data(), m.shard(s).tensor_data(*t), (size_t) N_EXPERT * 4);   // reuse wn for the bias
    }
    strata::kernels::glm::MoeGeometry mg{};
    mg.n_expert = N_EXPERT; mg.n_used = N_USED; mg.n_embd = N_EMBD; mg.ff = FF;
    mg.norm_w = false;
    mg.gating = strata::kernels::glm::MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
    std::vector<int32_t> order((size_t) N_USED);
    std::vector<float> wts((size_t) N_USED), probs((size_t) N_EXPERT);
    strata::kernels::glm::moe_route(router.data(), wn.data(), mg, nrm.data(), order.data(), wts.data(), probs.data());

    std::vector<float> moe_sum((size_t) N_EMBD, 0.0f), e_out((size_t) N_EMBD);
    for (int e = 0; e < N_USED; ++e) {
        const int ei = order[(size_t) e];
        for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_gate_exps.weight", ei * FF + r, N_EMBD, w_gate.data() + (size_t) r * N_EMBD);
        for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_up_exps.weight", ei * FF + r, N_EMBD, w_up.data() + (size_t) r * N_EMBD);
        gemv(w_gate, FF, N_EMBD, nrm.data(), e_gate.data());
        gemv(w_up, FF, N_EMBD, nrm.data(), e_up.data());
        for (int r = 0; r < FF; ++r) e_h[(size_t) r] = e_gate[(size_t) r] / (1.0f + std::exp(-e_gate[(size_t) r])) * e_up[(size_t) r];
        for (int r = 0; r < N_EMBD; ++r) dequant_row(m, p + "ffn_down_exps.weight", ei * N_EMBD + r, FF, e_down.data() + (size_t) r * FF);
        gemv(e_down, N_EMBD, FF, e_h.data(), e_out.data());
        for (int i = 0; i < N_EMBD; ++i) moe_sum[(size_t) i] += wts[(size_t) e] * e_out[(size_t) i];
    }
    for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_gate_shexp.weight", r, N_EMBD, w_gate.data() + (size_t) r * N_EMBD);
    for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_up_shexp.weight", r, N_EMBD, w_up.data() + (size_t) r * N_EMBD);
    gemv(w_gate, FF, N_EMBD, nrm.data(), e_gate.data());
    gemv(w_up, FF, N_EMBD, nrm.data(), e_up.data());
    for (int r = 0; r < FF; ++r) e_h[(size_t) r] = e_gate[(size_t) r] / (1.0f + std::exp(-e_gate[(size_t) r])) * e_up[(size_t) r];
    for (int r = 0; r < N_EMBD; ++r) dequant_row(m, p + "ffn_down_shexp.weight", r, FF, e_down.data() + (size_t) r * FF);
    gemv(e_down, N_EMBD, FF, e_h.data(), e_out.data());
    for (int i = 0; i < N_EMBD; ++i) moe_sum[(size_t) i] += e_out[(size_t) i];

    strata::kernels::kolibri::rms_norm(moe_sum.data(), nullptr, N_EMBD, geo.rms_eps, po.data());
    dequant_row(m, p + "post_ffw_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= wn[(size_t) i];
    for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];
}

}  // namespace

int main(int argc, char** argv) {
    bool serve = false;
    std::string model_path;
    int max_context = 4096;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--serve") serve = true;
        else if ((a == "--native" || a == "--model") && i + 1 < argc) model_path = argv[++i];
        else if (a == "--max-context" && i + 1 < argc) max_context = std::atoi(argv[++i]);
        else if (!a.empty() && a[0] != '-' && model_path.empty()) model_path = a;
    }
    if (!serve || model_path.empty() || max_context < 1) {
        std::fprintf(stderr, "usage: strata-kolibri --serve --native MODEL.gguf [--max-context N] [server options]\n");
        return 2;
    }
    GgufModel m = GgufModel::open(model_path);
    strata::Kolibri1Geometry geo;
    const std::string guard_err = strata::check_kolibri1_architecture(m.meta(), geo, nullptr);
    if (!guard_err.empty()) { std::fprintf(stderr, "guard: %s\n", guard_err.c_str()); return 1; }
    const TensorInfo* output = m.find("output.weight");
    if (!output || output->shape.size() != 2) { std::fprintf(stderr, "guard: output.weight is missing\n"); return 1; }
    const int n_vocab = (int) output->shape[1];
    int eos = -1;
    if (const strata::MetaValue* v = m.meta().get("tokenizer.ggml.eos_token_id")) eos = (int) v->u;
    max_context = std::min<int64_t>(max_context, geo.context_length);

    // per-layer caches
    std::vector<std::vector<float>> kc((size_t) geo.block_count), vc((size_t) geo.block_count);
    for (int b = 0; b < (int) geo.block_count; ++b) {
        kc[(size_t) b].assign((size_t) max_context * N_KV * HEAD_DIM, 0.0f);
        vc[(size_t) b].assign((size_t) max_context * N_KV * HEAD_DIM, 0.0f);
    }
    std::vector<float> x((size_t) N_EMBD), w_q((size_t) N_HEAD * HEAD_DIM * N_EMBD),
        w_k((size_t) N_KV * HEAD_DIM * N_EMBD), w_v((size_t) N_KV * HEAD_DIM * N_EMBD),
        w_o((size_t) N_EMBD * N_HEAD * HEAD_DIM), w_gate((size_t) FF * N_EMBD),
        w_up((size_t) FF * N_EMBD), router((size_t) N_EXPERT * N_EMBD),
        e_gate((size_t) FF), e_up((size_t) FF), e_h((size_t) FF), e_down((size_t) N_EMBD * FF);

    std::vector<float> wn_head((size_t) N_EMBD), nx_head((size_t) N_EMBD), logits((size_t) n_vocab);
    dequant_row(m, "output_norm.weight", 0, N_EMBD, wn_head.data());

    auto run_token = [&](int token, int pos, int& best) -> bool {
        if (token < 0 || token >= n_vocab || pos < 0 || pos >= max_context) return false;
        dequant_row(m, "token_embd.weight", token, N_EMBD, x.data());
        for (int b = 0; b < (int) geo.block_count; ++b)
            layer_forward(m, geo, b, pos, x, kc[(size_t) b], vc[(size_t) b],
                          w_q, w_k, w_v, w_o, w_gate, w_up, router, e_gate, e_up, e_h, e_down);
        strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, geo.rms_eps, nx_head.data());
        for (int i = 0; i < N_EMBD; ++i) nx_head[(size_t) i] *= wn_head[(size_t) i];
        best = 0;
        for (int o = 0; o < n_vocab; ++o) {
            dequant_row(m, "output.weight", o, N_EMBD, w_q.data());
            float acc = 0.0f;
            for (int i = 0; i < N_EMBD; ++i) acc += w_q[(size_t) i] * nx_head[(size_t) i];
            logits[(size_t) o] = acc;
            if (o && acc > logits[(size_t) best]) best = o;
        }
        return true;
    };

    std::printf("INFO architecture=kolibri1 vocab=%d layers=%lld expert_cache=cpu_mmap\n",
                n_vocab, (long long) geo.block_count);
    std::printf("READY %d\n", max_context);
    std::fflush(stdout);
    std::mt19937 rng(std::random_device{}());
    std::string line;
    while (std::getline(std::cin, line)) {
        if (std::getenv("STRATA_KOLIBRI_TRACE")) std::fprintf(stderr, "request: %s\n", line.c_str());
        if (line == "QUIT") break;
        if (line.rfind("GEN ", 0) != 0) {
            std::printf("ERR expected GEN <max_new> [sampling...] <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        std::istringstream request(line.substr(4));
        int max_new = 0;
        request >> max_new;
        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
        float penalty_repeat = 1.0f, penalty_freq = 0.0f, penalty_present = 0.0f;
        int top_k = 64, penalty_last_n = 64;
        std::string word, ids_text;
        while (request >> word) {
            const size_t eq = word.find('=');
            if (eq == std::string::npos) { ids_text = word; continue; }
            const std::string key = word.substr(0, eq), value = word.substr(eq + 1);
            if (key == "temperature") temperature = std::strtof(value.c_str(), nullptr);
            else if (key == "top_p") top_p = std::strtof(value.c_str(), nullptr);
            else if (key == "top_k") top_k = std::atoi(value.c_str());
            else if (key == "min_p") min_p = std::strtof(value.c_str(), nullptr);
            else if (key == "penalty_repeat") penalty_repeat = std::strtof(value.c_str(), nullptr);
            else if (key == "penalty_freq") penalty_freq = std::strtof(value.c_str(), nullptr);
            else if (key == "penalty_present") penalty_present = std::strtof(value.c_str(), nullptr);
            else if (key == "penalty_last_n") penalty_last_n = std::atoi(value.c_str());
            else if (key == "seed" && std::strtoul(value.c_str(), nullptr, 10) > 0)
                rng.seed((uint32_t) std::strtoul(value.c_str(), nullptr, 10));
        }
        std::vector<int> ids;
        std::stringstream csv(ids_text);
        while (std::getline(csv, word, ',')) if (!word.empty()) ids.push_back(std::atoi(word.c_str()));
        if (max_new < 1 || ids.empty() || ids.size() + (size_t) max_new > (size_t) max_context) {
            std::printf("ERR invalid request or context too long\n"); std::fflush(stdout); continue;
        }
        bool ok = true;
        int best = -1, pos = 0;
        std::vector<int> history = ids;
        auto sample = [&]() -> int {
            if (temperature <= 0.0f) return best;
            std::vector<int> counts((size_t) n_vocab, 0);
            const size_t begin = history.size() > (size_t) std::max(0, penalty_last_n)
                               ? history.size() - (size_t) std::max(0, penalty_last_n) : 0;
            for (size_t i = begin; i < history.size(); ++i)
                if (history[i] >= 0 && history[i] < n_vocab) ++counts[(size_t) history[i]];
            struct Candidate { int id; double p; };
            std::vector<Candidate> candidates;
            candidates.reserve((size_t) n_vocab);
            double high = -std::numeric_limits<double>::infinity();
            for (int i = 0; i < n_vocab; ++i) {
                double value = logits[(size_t) i];
                if (counts[(size_t) i]) {
                    if (penalty_repeat != 1.0f && penalty_repeat > 0.0f)
                        value = value >= 0.0 ? value / penalty_repeat : value * penalty_repeat;
                    value -= penalty_freq * counts[(size_t) i] + penalty_present;
                }
                value /= temperature;
                candidates.push_back({i, value});
                high = std::max(high, value);
            }
            for (Candidate& c : candidates) c.p = std::exp(c.p - high);
            std::sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
                return a.p > b.p || (a.p == b.p && a.id < b.id);
            });
            if (top_k > 0 && top_k < (int) candidates.size()) candidates.resize((size_t) top_k);
            if (min_p > 0.0f && !candidates.empty()) {
                const double floor = candidates.front().p * min_p;
                candidates.erase(std::remove_if(candidates.begin(), candidates.end(),
                    [&](const Candidate& c) { return c.p < floor; }), candidates.end());
            }
            double sum = 0.0;
            for (const Candidate& c : candidates) sum += c.p;
            if (top_p > 0.0f && top_p < 1.0f && sum > 0.0) {
                double kept = 0.0;
                size_t n = 0;
                do { kept += candidates[n++].p; } while (n < candidates.size() && kept / sum < top_p);
                candidates.resize(n);
                sum = kept;
            }
            if (candidates.empty() || sum <= 0.0) return best;
            std::uniform_real_distribution<double> pick(0.0, sum);
            double at = pick(rng);
            for (const Candidate& c : candidates) { if ((at -= c.p) <= 0.0) return c.id; }
            return candidates.back().id;
        };
        const double begun = (double) std::clock() / CLOCKS_PER_SEC;
        for (int tok : ids) {
            if (!run_token(tok, pos++, best)) { ok = false; break; }
            std::printf("PP %d %zu\n", pos, ids.size()); std::fflush(stdout);
        }
        if (ok) best = sample();
        const double prompt_done = (double) std::clock() / CLOCKS_PER_SEC;
        int produced = 0;
        while (ok && produced < max_new) {
            std::printf("T %d\n", best); std::fflush(stdout);
            ++produced;
            if (best == eos || produced == max_new) break;
            history.push_back(best);
            if (!run_token(best, pos++, best)) ok = false;
            else best = sample();
        }
        const double ended = (double) std::clock() / CLOCKS_PER_SEC;
        if (ok) std::printf("DONE %d %zu %.3f %.3f %s\n", produced, ids.size(),
                            1000.0 * (prompt_done - begun), 1000.0 * (ended - prompt_done),
                            (best == eos ? "eos" : "length"));
        else std::printf("ERR token outside vocabulary or context\n");
        std::fflush(stdout);
    }
    return 0;
}
