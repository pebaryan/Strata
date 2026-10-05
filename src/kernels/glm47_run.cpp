// src/kernels/glm47_run.cpp - run GLM-4.7-Flash END TO END on the Strata engine.
//
// This is the phase-5 runner: it reads the ARTIFACT directly (no Python-written fixtures), binds every
// weight site of all 47 blocks, embeds the prompt, runs the trunk, and takes the head's argmax.  Experts
// come from the pack's quantized blobs (glm_stage_moe_native), never materialised.
//
//   glm47_run --gguf <gguf> --pack <pack> --tokens "1,2,3" [--layers N] [--bind-only]
//
// Tokens are the raw ids (the oracle's /tokenize gives them); the prediction is printed as the argmax id
// plus the top-5, so a caller can diff it against the oracle's first generated token.
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm47_trunk.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm47_read.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/glm_norm.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace glm = strata::kernels::glm;
namespace cglm = strata::core::glm;

namespace {

// GLM-4.7-Flash geometry (the manifest's own numbers; the artifact is fixed).
constexpr int N_EMBD = 2048, N_HEAD = 20, HEAD_DIM = 256, KV_LORA = 512, Q_LORA = 768, N_ROT = 64;
constexpr int N_LAYER = 47, N_EXPERT = 64, N_USED = 4, EXP_FF = 1536, DENSE_FF = 10240;
constexpr float W_SCALE = 1.8f, ROPE_BASE = 1000000.0f, EPS = 1e-5f;
constexpr int NOPE = HEAD_DIM - N_ROT;            // 192
constexpr int Q_DIM = N_HEAD * HEAD_DIM;          // 5120
constexpr int KV_DIM = KV_LORA + N_ROT;           // 576

bool die(const std::string& m) { std::fprintf(stderr, "glm47_run: %s\n", m.c_str()); return false; }

// One block: the owned vectors + the engine's view of them.
struct Block {
    int kind = 1;
    std::vector<float> attn_norm, ffn_norm;
    std::vector<float> mw[8];
    glm::MlaWeights mla;
    // dense (block 0)
    std::vector<float> wg, wu, wd;
    glm::MoeGeometry dg;
    // moe
    glm::MoeGeometry gg;
    std::vector<float> router, probs_b, s_gate, s_up, s_down;
    const float* shared[3] = {nullptr, nullptr, nullptr};
    strata::kernels::cpu::NativeFmt fmt;
    bool has_fmt = false;
    cglm::Glm47TrunkLayer tl;
};

const uint8_t* blob_adapter(void* ctx, int layer, int expert) {
    return ((strata::core::ExpertSource*) ctx)->blob(layer, expert);
}

bool load(strata::GgufFile& g, const char* name, std::vector<float>& out, const char* what) {
    std::string err;
    if (!glm::load_tensor_f32(g, name, out, err)) return die(std::string(what) + ": " + err);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const char* gguf = "D:/aimodels/Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf";
    const char* pack = "D:/aimodels/strata-pack-glm47";
    std::string tokens;
    int n_layer = N_LAYER, verbosity = 1, gen = 1;
    bool bind_only = false;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gguf") && i + 1 < argc) gguf = argv[++i];
        else if (!std::strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = argv[++i];
        else if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) n_layer = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--bind-only")) bind_only = true;
        else if (!std::strcmp(argv[i], "--gen") && i + 1 < argc) gen = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--quiet")) verbosity = 0;
    }

    std::printf("glm47_run: %d blocks (block 0 dense), n_embd %d, experts %d/%d, ff %d\n",
                n_layer, N_EMBD, N_USED, N_EXPERT, EXP_FF);

    // ---- the pack's expert layout + the artifact ----
    std::string err;
    strata::core::FileExpertSource src;
    src.set_gguf(gguf);
    if (!strata::kernels::cpu::expert_layout_load(pack, N_LAYER, N_EXPERT, err, N_EMBD, EXP_FF) ||
        !src.open(pack, N_LAYER, N_EXPERT, err))
        return die("pack: " + err) ? 0 : 1;

    std::map<int, int> gu_of, d_of;
    { std::ifstream pf(std::string(pack) + "/native_experts.txt"); std::string line;
      while (std::getline(pf, line)) {
          if (line.empty() || line[0] == '#') continue;
          std::istringstream is(line);
          int64_t blk = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0; int gt = 0, dt = 0;
          if (!(is >> blk >> gt >> dt >> off >> nb >> go >> uo >> dob)) continue;
          gu_of[(int) blk] = gt; d_of[(int) blk] = dt;
      } }

    strata::GgufFile g(gguf);

    // ---- bind every site ----
    std::vector<Block> blk((size_t) n_layer);
    const char* nm[8] = {"attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_k_b.weight",
                         "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_v_b.weight", "attn_output.weight"};
    for (int l = 0; l < n_layer; ++l) {
        Block& b = blk[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        b.kind = (l == 0) ? 0 : 1;
        if (!load(g, (p + "attn_norm.weight").c_str(), b.attn_norm, "attn_norm")) return 1;
        if (!load(g, (p + "ffn_norm.weight").c_str(), b.ffn_norm, "ffn_norm")) return 1;
        for (int m = 0; m < 8; ++m)
            if (!load(g, (p + nm[m]).c_str(), b.mw[m], nm[m])) return 1;
        b.mla.wq_a = b.mw[0].data(); b.mla.q_a_norm = b.mw[1].data(); b.mla.wq_b = b.mw[2].data();
        b.mla.wk_b = b.mw[3].data(); b.mla.kv_a = b.mw[4].data(); b.mla.kv_a_norm = b.mw[5].data();
        b.mla.wv_b = b.mw[6].data(); b.mla.wo = b.mw[7].data(); b.mla.rope_freq_base = ROPE_BASE;
        if (b.kind == 0) {
            if (!load(g, (p + "ffn_gate.weight").c_str(), b.wg, "dense gate")) return 1;
            if (!load(g, (p + "ffn_up.weight").c_str(), b.wu, "dense up")) return 1;
            if (!load(g, (p + "ffn_down.weight").c_str(), b.wd, "dense down")) return 1;
            b.dg.n_embd = N_EMBD; b.dg.n_expert = 0; b.dg.n_used = 0; b.dg.ff = DENSE_FF;
            b.dg.w_scale = 1.0f; b.dg.norm_w = false; b.dg.clamp_exp = 0.0f; b.dg.clamp_shexp = 0.0f;
        } else {
            if (!load(g, (p + "ffn_gate_inp.weight").c_str(), b.router, "router")) return 1;
            if (!load(g, (p + "exp_probs_b.bias").c_str(), b.probs_b, "probs_b")) return 1;
            if (!load(g, (p + "ffn_gate_shexp.weight").c_str(), b.s_gate, "shexp gate")) return 1;
            if (!load(g, (p + "ffn_up_shexp.weight").c_str(), b.s_up, "shexp up")) return 1;
            if (!load(g, (p + "ffn_down_shexp.weight").c_str(), b.s_down, "shexp down")) return 1;
            b.shared[0] = b.s_gate.data(); b.shared[1] = b.s_up.data(); b.shared[2] = b.s_down.data();
            b.gg.n_embd = N_EMBD; b.gg.n_expert = N_EXPERT; b.gg.n_used = N_USED; b.gg.ff = EXP_FF;
            b.gg.w_scale = W_SCALE; b.gg.norm_w = true; b.gg.clamp_exp = 0.0f; b.gg.clamp_shexp = 0.0f;
            auto gt = gu_of.find(l), dt = d_of.find(l);
            std::string ferr;
            if (gt == gu_of.end() || dt == d_of.end() ||
                !strata::kernels::cpu::native_fmt(gt->second, dt->second, N_EMBD, EXP_FF, b.fmt, ferr))
                return die("native_fmt layer " + std::to_string(l) + ": " + ferr) ? 0 : 1;
            b.has_fmt = true;
        }
    }

    std::vector<float> output_norm, Wout, tok_emb;
    if (!load(g, "output_norm.weight", output_norm, "output_norm")) return 1;
    if (verbosity) std::printf("  binding the head + embedding (two 154880-row tensors, this is the slow part)\n");
    if (!load(g, "output.weight", Wout, "output")) return 1;
    if (!load(g, "token_embd.weight", tok_emb, "token_embd")) return 1;

    // the engine's per-layer views
    std::vector<cglm::Glm47TrunkLayer> arr((size_t) n_layer);
    for (int l = 0; l < n_layer; ++l) {
        Block& b = blk[(size_t) l];
        cglm::Glm47TrunkLayer& a = b.tl;
        a.kind = b.kind; a.attn_norm = b.attn_norm.data(); a.ffn_norm = b.ffn_norm.data(); a.mla = &b.mla;
        if (b.kind == 0) { a.ffn_gate = b.wg.data(); a.ffn_up = b.wu.data(); a.ffn_down = b.wd.data(); a.dense_g = b.dg; }
        else {
            a.moe_router = b.router.data(); a.moe_probs_b = b.probs_b.data(); a.moe_g = &b.gg; a.shexp = b.shared;
            a.shexp_types = nullptr;                       // float shared expert (native shared needs the uninstalled hook)
            a.moe_native_fmt = &b.fmt; a.moe_native_blob = &blob_adapter; a.moe_native_ctx = &src;
        }
        arr[(size_t) l] = a;
    }

    glm::MlaGeometry mg; mg.n_embd = N_EMBD; mg.n_head = N_HEAD; mg.head_dim = HEAD_DIM;
    mg.kv_lora = KV_LORA; mg.q_lora = Q_LORA; mg.n_rot = N_ROT;

    // ---- the prompt tokens -> hidden states ----
    std::vector<int> ids;
    { std::stringstream ss(tokens); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) ids.push_back(std::atoi(t.c_str())); }
    if (ids.empty() && !bind_only) return die("no --tokens given") ? 0 : 1;

    if (bind_only) {
        std::printf("  bind-only: %d tokens, embed rows %d x %d, head %d x %d\n",
                    (int) ids.size(), (int)(tok_emb.size() / (N_EMBD > 0 ? N_EMBD : 1)), N_EMBD,
                    (int)(Wout.size() / (N_EMBD > 0 ? N_EMBD : 1)), N_EMBD);
        std::printf("glm47_run: BIND OK\n");
        return 0;
    }

    std::vector<std::vector<float>> caches((size_t) n_layer);
    std::vector<float> hidden((size_t) N_EMBD);
    auto feed = [&](int tok, int pos) -> bool {
        if (tok < 0 || (size_t) tok * N_EMBD + N_EMBD > tok_emb.size()) return false;
        std::vector<float> x(tok_emb.begin() + (size_t) tok * N_EMBD,
                             tok_emb.begin() + (size_t) (tok + 1) * N_EMBD);
        std::string terr;
        return cglm::glm47_trunk_forward(arr.data(), n_layer, mg, x.data(), pos, EPS, &caches, nullptr,
                                         nullptr, nullptr, hidden.data(), nullptr, nullptr, terr);
    };
    const int V = (int) (Wout.size() / N_EMBD);
    std::vector<float> hn((size_t) N_EMBD), logits((size_t) V);
    std::vector<int> order((size_t) V);
    auto predict = [&]() -> int {
        glm::rms_norm_gain(output_norm.data(), N_EMBD, hidden.data(), hn.data(), EPS);
        for (int v = 0; v < V; ++v) {
            double s = 0.0;
            for (int e = 0; e < N_EMBD; ++e) s += (double) Wout[(size_t) v * N_EMBD + e] * (double) hn[(size_t) e];
            logits[(size_t) v] = (float) s;
        }
        for (int v = 0; v < V; ++v) order[(size_t) v] = v;
        std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                          [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        return order[0];
    };

    for (size_t t = 0; t < ids.size(); ++t)
        if (!feed(ids[t], (int) t)) return die("trunk (prompt)") ? 0 : 1;

    std::printf("  %zu prompt tokens -> generated:\n", ids.size());
    for (int g = 0; g < gen; ++g) {
        const int t1 = predict();
        std::printf("    step %d: %d   (margin %.4f over %d)\n", g, t1,
                    (double) (logits[(size_t) order[0]] - logits[(size_t) order[1]]), order[1]);
        if (g + 1 < gen && !feed(t1, (int) ids.size() + g)) return die("trunk (gen)") ? 0 : 1;
    }
    std::printf("  top-5 (last):");
    for (int i = 0; i < 5; ++i) std::printf(" %d(%.3f)", order[(size_t) i], (double) logits[(size_t) order[(size_t) i]]);
    std::printf("\n");
    std::printf("glm47_run: OK\n");
    return 0;
}
