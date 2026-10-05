// kolibri_trunk_gate - phase 6 of KOLIBRI-PORT.md (branch kolibri-port): the engine's own trunk,
// one token, end to end, against the patched llama.cpp's dumped logits.
//
// The engine path here is Strata's semantics with Strata's kernels, driven straight from the GGUF
// (the pack's index semantics belong to the serving phase, not to this parity gate):
//   token_embd (Q8_0) -> 50 layers of { rms(attn_norm) -> QKV (Q8_0/Q4_K per the artifact) ->
//   QK-norm -> RoPE (sliding layers only; at pos 0 it is the identity, which is why the 1-token
//   gate is exact for every layer) -> single-key attention (softmax over one key) -> wo ->
//   x += rms(post_attention_norm) -> rms(ffn_norm) -> moe_route(LOGIT_ADD) -> top-6 experts
//   (rows dequantized from Q4_K/Q6_K, SwiGLU) + shared expert -> x += rms(post_ffw_norm) }
//   -> rms(output_norm) -> output head -> logits, compared with the oracle's dump.
//
// usage: kolibri_trunk_gate <kolibri-gguf> <oracle-logits-dump> [token-id]
//   (token-id defaults to 15351 = " The"; the oracle dump must come from the SAME single token)

#include <cmath>
#include <ctime>
#include <cstdio>
#include <cstring>
#include <cstdlib>
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

// dequantize one row (the row's first `n` elements) of a quantized tensor into out
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
            case 8:  strata::dequantize_q8_0(row_base + (size_t) b * bb, o); break;
            case 12: strata::dequantize_q4_K(row_base + (size_t) b * bb, o); break;
            case 14: strata::dequantize_q6_K(row_base + (size_t) b * bb, o); break;
            case 0:  std::memcpy(o, row_base + (size_t) b * bb, (size_t) be * 4); break;
            case 1:  for (int i = 0; i < be; ++i) { uint16_t h; std::memcpy(&h, row_base + (size_t) b * bb + 2 * i, 2);
                         o[i] = strata::fp16_to_fp32(h); } break;
            default: std::fprintf(stderr, "unhandled type %d for %s\n", t->type, name.c_str()); std::exit(1);
        }
    }
}

void gemv(const std::vector<float>& w_rows_deq, int out_n, int in_n, const float* x, float* y) {
    for (int o = 0; o < out_n; ++o) {
        float acc = 0.0f;
        const float* row = w_rows_deq.data() + (size_t) o * in_n;
        for (int i = 0; i < in_n; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: kolibri_trunk_gate <kolibri-gguf> <oracle-logits> [token-id]\n");
        return 2;
    }
    const std::string gguf_path = argv[1], oracle_path = argv[2];
    const int token = argc >= 4 ? std::atoi(argv[3]) : 15351;

    GgufModel m = GgufModel::open(gguf_path);
    strata::Kolibri1Geometry geo;
    const std::string guard_err = strata::check_kolibri1_architecture(m.meta(), geo, nullptr);
    if (!guard_err.empty()) { std::fprintf(stderr, "guard: %s\n", guard_err.c_str()); return 1; }

    const int n_vocab = (int) m.find("output.weight")->shape[1];

    // ---- the engine path ----------------------------------------------------------------------------------------------------------------
    std::vector<float> x((size_t) N_EMBD);
    dequant_row(m, "token_embd.weight", token, N_EMBD, x.data());

    // per-layer scratch, sized once
    std::vector<float> nrm((size_t) N_EMBD), q((size_t) N_HEAD * HEAD_DIM), k((size_t) N_KV * HEAD_DIM),
        v((size_t) N_KV * HEAD_DIM), qn((size_t) N_HEAD * HEAD_DIM), kn((size_t) N_KV * HEAD_DIM),
        att_out((size_t) N_HEAD * HEAD_DIM), wo_out((size_t) N_EMBD), po((size_t) N_EMBD),
        w_q((size_t) N_HEAD * HEAD_DIM * N_EMBD), w_k((size_t) N_KV * HEAD_DIM * N_EMBD),
        w_v((size_t) N_KV * HEAD_DIM * N_EMBD), w_o((size_t) N_EMBD * N_HEAD * HEAD_DIM),
        w_gate((size_t) FF * N_EMBD), w_up((size_t) FF * N_EMBD), w_down((size_t) N_EMBD * FF),
        moe_sum((size_t) N_EMBD), e_gate((size_t) FF), e_up((size_t) FF), e_h((size_t) FF), e_down((size_t) N_EMBD * FF);
    std::vector<float> q_norm_w((size_t) HEAD_DIM), k_norm_w((size_t) HEAD_DIM);

    const double t0 = (double) std::clock() / CLOCKS_PER_SEC;
    for (int b = 0; b < (int) geo.block_count; ++b) {
        const std::string p = "blk." + std::to_string(b) + ".";
        // 1. attn sandwich: rms(attn_norm) -> QKV
        strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nrm.data());
        {
            std::vector<float> w_norm((size_t) N_EMBD);
            dequant_row(m, p + "attn_norm.weight", 0, N_EMBD, w_norm.data());
            for (int i = 0; i < N_EMBD; ++i) nrm[(size_t) i] *= w_norm[(size_t) i];
        }
        dequant_row(m, p + "attn_q.weight", 0, N_HEAD * HEAD_DIM * N_EMBD, w_q.data());
        dequant_row(m, p + "attn_k.weight", 0, N_KV * HEAD_DIM * N_EMBD, w_k.data());
        dequant_row(m, p + "attn_v.weight", 0, N_KV * HEAD_DIM * N_EMBD, w_v.data());
        gemv(w_q, N_HEAD * HEAD_DIM, N_EMBD, nrm.data(), q.data());
        gemv(w_k, N_KV * HEAD_DIM, N_EMBD, nrm.data(), k.data());
        gemv(w_v, N_KV * HEAD_DIM, N_EMBD, nrm.data(), v.data());
        // 2. QK-norm (before RoPE; at pos 0 rope is the identity so this single-token gate is exact)
        dequant_row(m, p + "attn_q_norm.weight", 0, HEAD_DIM, q_norm_w.data());
        dequant_row(m, p + "attn_k_norm.weight", 0, HEAD_DIM, k_norm_w.data());
        strata::kernels::kolibri::qk_norm_per_head(q.data(), q_norm_w.data(), N_HEAD, HEAD_DIM, EPS, qn.data());
        strata::kernels::kolibri::qk_norm_per_head(k.data(), k_norm_w.data(), N_KV, HEAD_DIM, EPS, kn.data());
        // 3. single-key attention: softmax over one key = 1, so out = v per GQA group
        for (int h = 0; h < N_HEAD; ++h) {
            const int hk = h / (N_HEAD / N_KV);
            std::memcpy(att_out.data() + (size_t) h * HEAD_DIM, v.data() + (size_t) hk * HEAD_DIM,
                        (size_t) HEAD_DIM * 4);
        }
        dequant_row(m, p + "attn_output.weight", 0, N_EMBD * N_HEAD * HEAD_DIM, w_o.data());
        gemv(w_o, N_EMBD, N_HEAD * HEAD_DIM, att_out.data(), wo_out.data());
        // 4. post-norm INSIDE the residual
        strata::kernels::kolibri::rms_norm(wo_out.data(), nullptr, N_EMBD, EPS, po.data());
        {
            std::vector<float> w_norm((size_t) N_EMBD);
            dequant_row(m, p + "post_attention_norm.weight", 0, N_EMBD, w_norm.data());
            for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= w_norm[(size_t) i];
        }
        for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];

        // 5. ffn sandwich: rms(ffn_norm) -> router -> top-6 experts + shared
        strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nrm.data());
        {
            std::vector<float> w_norm((size_t) N_EMBD);
            dequant_row(m, p + "ffn_norm.weight", 0, N_EMBD, w_norm.data());
            for (int i = 0; i < N_EMBD; ++i) nrm[(size_t) i] *= w_norm[(size_t) i];
        }
        // router: ffn_gate_inp is the [E][n_embd] matrix (F32 in the GGUF); moe_route does the dots
        static std::vector<float> router((size_t) N_EXPERT * N_EMBD);
        std::vector<float> probs((size_t) N_EXPERT), bias((size_t) N_EXPERT);
        {
            size_t s = 0;
            const TensorInfo* t = m.find(p + "ffn_gate_inp.weight", &s);
            const uint8_t* base = m.shard(s).tensor_data(*t);
            std::memcpy(router.data(), base, (size_t) N_EXPERT * N_EMBD * 4);
            t = m.find(p + "exp_probs_b.bias", &s);
            base = m.shard(s).tensor_data(*t);
            std::memcpy(bias.data(), base, (size_t) N_EXPERT * 4);
        }
        strata::kernels::glm::MoeGeometry mg{};
        mg.n_expert = N_EXPERT; mg.n_used = N_USED; mg.n_embd = N_EMBD; mg.ff = FF;
        mg.norm_w = false;
        mg.gating = strata::kernels::glm::MoeGeometry::Gating::SIGMOID_LOGIT_ADD;
        std::vector<int32_t> order((size_t) N_USED);
        std::vector<float> wts((size_t) N_USED);
        strata::kernels::glm::moe_route(router.data(), bias.data(), mg, nrm.data(),
                                        order.data(), wts.data(), probs.data());
        // experts: dequant the selected experts' three slabs, SwiGLU, accumulate
        std::fill(moe_sum.begin(), moe_sum.end(), 0.0f);
        for (int e = 0; e < N_USED; ++e) {
            const int ei = order[(size_t) e];
            // ffn_*_exps are [n_ff, n_embd, n_expert]: expert ei's slab starts at row ei*FF
            // gate: rows ei*FF .. ei*FF+FF-1 of ffn_gate_exps (row = n_embd)
            for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_gate_exps.weight", ei * FF + r, N_EMBD, w_gate.data() + (size_t) r * N_EMBD);
            for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_up_exps.weight", ei * FF + r, N_EMBD, w_up.data() + (size_t) r * N_EMBD);
            gemv(w_gate, FF, N_EMBD, nrm.data(), e_gate.data());
            gemv(w_up, FF, N_EMBD, nrm.data(), e_up.data());
            for (int r = 0; r < FF; ++r) e_h[(size_t) r] = e_gate[(size_t) r] * (1.0f / (1.0f + std::exp(-e_gate[(size_t) r]))) * e_up[(size_t) r];
            // down is [n_ff, n_embd, n_expert] in gguf ne order (ne0 = n_ff contiguous), so expert
            // ei's rows are ei*n_embd + r, each of length n_ff
            for (int r = 0; r < N_EMBD; ++r) dequant_row(m, p + "ffn_down_exps.weight", ei * N_EMBD + r, FF, e_down.data() + (size_t) r * FF);
            // y = down @ h, scaled by the router weight
            for (int i = 0; i < N_EMBD; ++i) {
                float acc = 0.0f;
                const float* row = e_down.data() + (size_t) i * FF;
                for (int r = 0; r < FF; ++r) acc += row[r] * e_h[(size_t) r];
                moe_sum[(size_t) i] += wts[(size_t) e] * acc;
            }
        }
        // shared expert, unweighted
        for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_gate_shexp.weight", r, N_EMBD, w_gate.data() + (size_t) r * N_EMBD);
        for (int r = 0; r < FF; ++r) dequant_row(m, p + "ffn_up_shexp.weight", r, N_EMBD, w_up.data() + (size_t) r * N_EMBD);
        gemv(w_gate, FF, N_EMBD, nrm.data(), e_gate.data());
        gemv(w_up, FF, N_EMBD, nrm.data(), e_up.data());
        for (int r = 0; r < FF; ++r) e_h[(size_t) r] = e_gate[(size_t) r] * (1.0f / (1.0f + std::exp(-e_gate[(size_t) r]))) * e_up[(size_t) r];
        for (int r = 0; r < N_EMBD; ++r) dequant_row(m, p + "ffn_down_shexp.weight", r, FF, e_down.data() + (size_t) r * FF);
        for (int i = 0; i < N_EMBD; ++i) {
            float acc = 0.0f;
            const float* row = e_down.data() + (size_t) i * FF;
            for (int r = 0; r < FF; ++r) acc += row[r] * e_h[(size_t) r];
            moe_sum[(size_t) i] += acc;
        }
        // 6. post-norm INSIDE the residual
        strata::kernels::kolibri::rms_norm(moe_sum.data(), nullptr, N_EMBD, EPS, po.data());
        {
            std::vector<float> w_norm((size_t) N_EMBD);
            dequant_row(m, p + "post_ffw_norm.weight", 0, N_EMBD, w_norm.data());
            for (int i = 0; i < N_EMBD; ++i) po[(size_t) i] *= w_norm[(size_t) i];
        }
        for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] += po[(size_t) i];
    }
    const double t1 = (double) std::clock() / CLOCKS_PER_SEC;
    std::printf("  trunk: 50 layers in %.1f s\n", t1 - t0);

    {
        double ss = 0.0;
        for (int i = 0; i < N_EMBD; ++i) ss += (double) x[(size_t) i] * x[(size_t) i];
        std::fprintf(stderr, "pre-head |x| rms = %.4f\n", std::sqrt(ss / N_EMBD));
    }
    // ---- head: output_norm then output.weight --------------------------------------------------------------------------------------------
    {
        std::vector<float> w_norm((size_t) N_EMBD), nx((size_t) N_EMBD);
        dequant_row(m, "output_norm.weight", 0, N_EMBD, w_norm.data());
        strata::kernels::kolibri::rms_norm(x.data(), nullptr, N_EMBD, EPS, nx.data());
        for (int i = 0; i < N_EMBD; ++i) x[(size_t) i] = nx[(size_t) i] * w_norm[(size_t) i];
    }
    {
        double ss = 0.0;
        for (int i = 0; i < N_EMBD; ++i) ss += (double) x[(size_t) i] * x[(size_t) i];
        std::fprintf(stderr, "post-output_norm |x| rms = %.4f\n", std::sqrt(ss / N_EMBD));
    }
    std::vector<float> logits((size_t) n_vocab);
    for (int o = 0; o < n_vocab; ++o) {
        dequant_row(m, "output.weight", o, N_EMBD, w_q.data());   // reuse scratch (N_EMBD floats)
        float acc = 0.0f;
        for (int i = 0; i < N_EMBD; ++i) acc += w_q[(size_t) i] * x[(size_t) i];
        logits[(size_t) o] = acc;
    }

    // ---- compare with the oracle ----------------------------------------------------------------------------------------------------------
    FILE* f = std::fopen(oracle_path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", oracle_path.c_str()); return 1; }
    std::vector<float> oracle((size_t) n_vocab);
    const size_t got = std::fread(oracle.data(), 4, (size_t) n_vocab, f);
    std::fclose(f);
    if (got != (size_t) n_vocab) { std::fprintf(stderr, "oracle dump has %zu floats, expected %d\n", got, n_vocab); return 1; }

    double worst = 0.0; int worst_i = -1;
    for (int i = 0; i < n_vocab; ++i) {
        const double d = std::abs((double) logits[(size_t) i] - (double) oracle[(size_t) i]);
        if (d > worst) { worst = d; worst_i = i; }
    }
    int mine_best = 0, orc_best = 0;
    for (int i = 1; i < n_vocab; ++i) {
        if (logits[(size_t) i] > logits[(size_t) mine_best]) mine_best = i;
        if (oracle[(size_t) i] > oracle[(size_t) orc_best]) orc_best = i;
    }
    char buf[192];
    std::snprintf(buf, sizeof buf, "max |logit err| %.4g at token %d (argmax %d vs oracle %d%s)",
                  worst, worst_i, mine_best, orc_best, mine_best == orc_best ? ", MATCH" : ", MISMATCH");
    check(worst < 0.35, buf);
    check(mine_best == orc_best, "greedy argmax matches the patched llama.cpp");

    {
        const char* dp = std::getenv("STRATA_TRUNK_DUMP");
        if (dp) { FILE* f = std::fopen(dp, "wb"); if (f) { std::fwrite(logits.data(), 4, (size_t) n_vocab, f); std::fclose(f); } }
    }
    std::printf("\n%s\n", fails == 0 ? "TRUNK GATE: PASS" : "TRUNK GATE: FAIL");
    return fails == 0 ? 0 : 1;
}
