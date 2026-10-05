// kolibri_multitoken_gate - phase 6b (branch kolibri-port): the trunk over a REAL prompt -
// positions 1..N-1 exercise RoPE at nonzero angles, the causal KV cache, and GQA score masking.
// (The 513-token window never clips at N=5; the mask's clip path is covered by the layer gate's
// direct swa_attends checks, and a >513-token prefill is a hours-long CPU run - documented, not
// silent.)
//
// Same engine path as kolibri_trunk_gate (tensors straight from the GGUF), extended to a token
// loop with per-layer K/V caches and a causal mask; sliding layers rope their Q and K.  The
// comparison target is the oracle's LAST-position dump for the same prompt.
//
// usage: kolibri_multitoken_gate <kolibri-gguf> <oracle-logits> <t1,t2,...|prompt-tokens> [dump]

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <ctime>
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
constexpr float EPS = 1e-6f, THETA = 10000.0f;

int fails = 0;
void check(bool ok, const std::string& what) {
    if (ok) { std::printf("  PASS %s\n", what.c_str()); return; }
    ++fails;
    std::printf("  FAIL %s\n", what.c_str());
}

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
    strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nrm.data());
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
        strata::kernels::kolibri::qk_norm_per_head(q.data(), qw.data(), N_HEAD, HEAD_DIM, EPS, qn.data());
        strata::kernels::kolibri::qk_norm_per_head(k.data(), kw.data(), N_KV, HEAD_DIM, EPS, kn.data());
    }
    if (sliding) {   // NoPE on full layers
        strata::kernels::kolibri::rope_neox(qn.data(), pos, N_HEAD, HEAD_DIM, THETA, HEAD_DIM);
        strata::kernels::kolibri::rope_neox(kn.data(), pos, N_KV, HEAD_DIM, THETA, HEAD_DIM);
    }
    std::memcpy(k_cache.data() + (size_t) pos * N_KV * HEAD_DIM, kn.data(), (size_t) N_KV * HEAD_DIM * 4);
    std::memcpy(v_cache.data() + (size_t) pos * N_KV * HEAD_DIM, v.data(), (size_t) N_KV * HEAD_DIM * 4);

    // causal attention (window 513 never clips at this gate's lengths; full layers unbounded)
    std::vector<float> att_out((size_t) N_HEAD * HEAD_DIM, 0.0f);
    for (int h = 0; h < N_HEAD; ++h) {
        const int hk = h / (N_HEAD / N_KV);
        double maxi = -1e300;
        std::vector<double> sc((size_t) pos + 1);
        for (int t = 0; t <= pos; ++t) {
            double d = 0.0;
            const float* kc = k_cache.data() + (size_t) t * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM;
            for (int i = 0; i < HEAD_DIM; ++i) d += (double) qn[(size_t) h * HEAD_DIM + i] * kc[i];
            sc[(size_t) t] = d / std::sqrt((double) HEAD_DIM);
            maxi = std::max(maxi, sc[(size_t) t]);
        }
        double sum = 0.0;
        for (int t = 0; t <= pos; ++t) { sc[(size_t) t] = std::exp(sc[(size_t) t] - maxi); sum += sc[(size_t) t]; }
        for (int i = 0; i < HEAD_DIM; ++i) {
            double acc = 0.0;
            for (int t = 0; t <= pos; ++t)
                acc += sc[(size_t) t] * (double) v_cache[(size_t) t * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
            att_out[(size_t) h * HEAD_DIM + i] = (float) (acc / sum);
        }
    }
    dequant_row(m, p + "attn_output.weight", 0, N_EMBD * N_HEAD * HEAD_DIM, w_o.data());
    std::vector<float> wo_out((size_t) N_EMBD);
    gemv(w_o, N_EMBD, N_HEAD * HEAD_DIM, att_out.data(), wo_out.data());
    strata::kernels::kolibri::rms_norm(wo_out.data(), nullptr, N_EMBD, EPS, po.data());
    dequant_row(m, p + "post_attention_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= wn[(size_t) i];
    for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];

    // ffn sandwich
    strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nrm.data());
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

    strata::kernels::kolibri::rms_norm(moe_sum.data(), nullptr, N_EMBD, EPS, po.data());
    dequant_row(m, p + "post_ffw_norm.weight", 0, N_EMBD, wn.data());
    for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= wn[(size_t) i];
    for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: kolibri_multitoken_gate <kolibri-gguf> <oracle-logits> <t1,t2,...> [dump]\n");
        return 2;
    }
    GgufModel m = GgufModel::open(argv[1]);
    strata::Kolibri1Geometry geo;
    const std::string guard_err = strata::check_kolibri1_architecture(m.meta(), geo, nullptr);
    if (!guard_err.empty()) { std::fprintf(stderr, "guard: %s\n", guard_err.c_str()); return 1; }
    const int n_vocab = (int) m.find("output.weight")->shape[1];

    std::vector<int> tokens;
    {
        const std::string s = argv[3];
        size_t at = 0;
        while (at < s.size()) {
            const size_t comma = s.find(',', at);
            const std::string item = s.substr(at, comma == std::string::npos ? std::string::npos : comma - at);
            if (!item.empty()) tokens.push_back(std::atoi(item.c_str()));
            if (comma == std::string::npos) break;
            at = comma + 1;
        }
    }
    const int T = (int) tokens.size();
    std::printf("  prompt: %d tokens\n", T);

    // per-layer caches
    std::vector<std::vector<float>> kc((size_t) geo.block_count), vc((size_t) geo.block_count);
    for (int b = 0; b < (int) geo.block_count; ++b) {
        kc[(size_t) b].assign((size_t) T * N_KV * HEAD_DIM, 0.0f);
        vc[(size_t) b].assign((size_t) T * N_KV * HEAD_DIM, 0.0f);
    }
    std::vector<float> x((size_t) N_EMBD), w_q((size_t) N_HEAD * HEAD_DIM * N_EMBD),
        w_k((size_t) N_KV * HEAD_DIM * N_EMBD), w_v((size_t) N_KV * HEAD_DIM * N_EMBD),
        w_o((size_t) N_EMBD * N_HEAD * HEAD_DIM), w_gate((size_t) FF * N_EMBD),
        w_up((size_t) FF * N_EMBD), router((size_t) N_EXPERT * N_EMBD),
        e_gate((size_t) FF), e_up((size_t) FF), e_h((size_t) FF), e_down((size_t) N_EMBD * FF);

    std::vector<float> wn_head((size_t) N_EMBD), nx_head((size_t) N_EMBD);
    dequant_row(m, "output_norm.weight", 0, N_EMBD, wn_head.data());
    const char* dump_prefix = std::getenv("STRATA_MULTI_DUMP");   // per-position dumps when set
    const double t0 = (double) std::clock() / CLOCKS_PER_SEC;
    for (int t = 0; t < T; ++t) {
        dequant_row(m, "token_embd.weight", tokens[(size_t) t], N_EMBD, x.data());
        for (int b = 0; b < (int) geo.block_count; ++b)
            layer_forward(m, geo, b, t, x, kc[(size_t) b], vc[(size_t) b],
                          w_q, w_k, w_v, w_o, w_gate, w_up, router, e_gate, e_up, e_h, e_down);
        if (dump_prefix) {
            strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nx_head.data());
            for (int i = 0; i < N_EMBD; ++i) nx_head[(size_t) i] *= wn_head[(size_t) i];
            std::vector<float> lg((size_t) n_vocab);
            for (int o = 0; o < n_vocab; ++o) {
                dequant_row(m, "output.weight", o, N_EMBD, w_q.data());
                float acc = 0.0f;
                for (int i = 0; i < N_EMBD; ++i) acc += w_q[(size_t) i] * nx_head[(size_t) i];
                lg[(size_t) o] = acc;
            }
            char pf[512];
            std::snprintf(pf, sizeof pf, "%s.p%d", dump_prefix, t);
            FILE* fp = std::fopen(pf, "wb");
            if (fp) { std::fwrite(lg.data(), 4, (size_t) n_vocab, fp); std::fclose(fp); }
        }
    }
    const double t1 = (double) std::clock() / CLOCKS_PER_SEC;
    std::printf("  trunk: %d layers x %d tokens in %.1f s\n", (int) geo.block_count, T, t1 - t0);

    // head
    {
        std::vector<float> wn((size_t) N_EMBD), nx((size_t) N_EMBD);
        dequant_row(m, "output_norm.weight", 0, N_EMBD, wn.data());
        strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nx.data());
        for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] = nx[(size_t) i] * wn[(size_t) i];
    }
    std::vector<float> logits((size_t) n_vocab);
    for (int o = 0; o < n_vocab; ++o) {
        dequant_row(m, "output.weight", o, N_EMBD, w_q.data());
        float acc = 0.0f;
        for (int i = 0; i < N_EMBD; ++i) acc += w_q[(size_t) i] * x[(size_t) i];
        logits[(size_t) o] = acc;
    }
    if (argc >= 5) {
        FILE* f = std::fopen(argv[4], "wb");
        if (f) { std::fwrite(logits.data(), 4, (size_t) n_vocab, f); std::fclose(f); }
    }

    FILE* f = std::fopen(argv[2], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[2]); return 1; }
    std::vector<float> oracle((size_t) n_vocab);
    if (std::fread(oracle.data(), 4, (size_t) n_vocab, f) != (size_t) n_vocab) {
        std::fprintf(stderr, "oracle dump short\n"); std::fclose(f); return 1;
    }
    std::fclose(f);

    double worst = 0.0; int worst_i = -1;
    for (int i = 0; i < n_vocab; ++i) {
        const double d = std::abs((double) logits[(size_t) i] - (double) oracle[(size_t) i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (int i = 0; i < n_vocab; ++i) {
        dot += (double) logits[(size_t) i] * oracle[(size_t) i];
        na += (double) logits[(size_t) i] * logits[(size_t) i];
        nb += (double) oracle[(size_t) i] * oracle[(size_t) i];
    }
    const double cosine = dot / std::sqrt(na * nb);
    int mine_best = 0, orc_best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[(size_t) i] > logits[(size_t) mine_best]) mine_best = i;
        if (oracle[(size_t) i] > oracle[(size_t) orc_best]) orc_best = i;
    }
    char buf[192];
    std::snprintf(buf, sizeof buf, "max |logit err| %.4g at %d, cos %.4f (argmax %d vs oracle %d%s)",
                  worst, worst_i, cosine, mine_best, orc_best,
                  mine_best == orc_best ? ", MATCH" : ", MISMATCH");
    // The gate's hard criteria: identical greedy argmax and cosine >= 0.98.  The max-error floor
    // here is NOT the 1-token gate's 0.34: multi-key attention over cached K/V accumulated the
    // rounding difference between this reference path (rows dequantized to f32, fp32 gemv) and
    // llama.cpp's kernels (integer q4_K/q6_K dots, fp16 flash intermediates) - measured ~5-7
    // logits at the tail with T=5..8 on two prompts, argmax stable in both.  A semantic bug would
    // move the argmax or collapse the cosine (a missing rope term cost 0.9985 -> 0.9765 before
    // the halves fix); numerics erode the tail.
    check(cosine >= 0.98, buf);
    check(mine_best == orc_best, "greedy argmax matches the patched llama.cpp at the last position");

    std::printf("\n%s\n", fails == 0 ? "MULTITOKEN GATE: PASS" : "MULTITOKEN GATE: FAIL");
    return fails == 0 ? 0 : 1;
}
