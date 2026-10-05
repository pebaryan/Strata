// src/kernels/glm47_run.cpp - run GLM-4.7-Flash END TO END on the Strata engine.
//
// The phase-5 runner reads the ARTIFACT directly (no Python-written fixtures), binds every weight site of
// all 47 blocks, embeds the prompt, runs the trunk, and takes the head's argmax.  Experts come from the
// pack's quantized blobs (glm_stage_moe_native), never materialised.
//
// THE GPU PATH.  By default the four MLA projections (wq_a/wq_b/kv_a/wo) and the head are bound as the
// artifact's OWN quantized blocks, resident on the device, and driven by the engine's native device GEMV
// (native_mmvq, pinned to llama.cpp).  mla_forward_rope dispatches to them through the installed
// mla_set_native_project hook; the head through glm47_device_gemv.  --cpu drops both and uses the float
// host GEMMs, so the two can be A/B'd on the same binary.  Model maths (`glm47_device.cpp`).
//
//   glm47_run --gguf <gguf> --pack <pack> --tokens "1,2,3" [--gen N] [--layers N] [--cpu] [--quiet]
#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/expert_source.hpp"
#include "strata/core/glm47_trunk.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/glm47_read.hpp"
#include "strata/kernels/glm_mla.hpp"
#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/glm_norm.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <unordered_map>

#include <algorithm>
#include <chrono>
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

// The GPU side: src/kernels/glm47_device.cpp.
namespace strata::kernels::glm {
bool glm47_native_project(int count, const void* const* weights, const int* types,
                          const float* x, int n_in, int n_out, float* const* out);
bool glm47_device_gemv(int type, const void* w_dev, const float* x, int n_in, int n_out, float* y);
bool glm47_native_ffn(const void* const* weights, const int* types, const MoeGeometry& g,
                      const float* x, float* out, float clamp_limit);
double glm47_project_ms();
double glm47_ffn_ms();
void glm47_install_device_experts(size_t budget_bytes);
void glm47_device_expert_stats(uint64_t* hits, uint64_t* misses, uint64_t* evicted, size_t* bytes);
}  // namespace strata::kernels::glm

namespace {

// GLM-4.7-Flash geometry (the manifest's own numbers; the artifact is fixed).
constexpr int N_EMBD = 2048, N_HEAD = 20, HEAD_DIM = 256, KV_LORA = 512, Q_LORA = 768, N_ROT = 64;
constexpr int N_LAYER = 47, N_EXPERT = 64, N_USED = 4, EXP_FF = 1536, DENSE_FF = 10240;
constexpr float W_SCALE = 1.8f, ROPE_BASE = 1000000.0f, EPS = 1e-5f;
constexpr int NOPE = HEAD_DIM - N_ROT;            // 192
constexpr int Q_DIM = N_HEAD * HEAD_DIM;          // 5120
constexpr int KV_DIM = KV_LORA + N_ROT;           // 576

bool die(const std::string& m) { std::fprintf(stderr, "glm47_run: %s\n", m.c_str()); return false; }

// A weight resident on the device as the artifact's own quantized blocks.  n_in is the GGUF contiguous
// (reduction) dim = shape[0]; n_out the row count = shape[1]; both in the native_mmvq order.
struct DevBlocks { void* dev = nullptr; int type = 0, n_in = 0, n_out = 0; };

// Upload `name`'s raw GGUF blocks to the device.  Returns false (without dying) when the tensor is missing
// or its type has no native kernel, so the caller can keep its float fallback.
bool upload_blocks(const strata::GgufFile& g, const std::string& name, DevBlocks& out) {
    const strata::TensorInfo* t = g.find(name);
    if (t == nullptr || t->shape.size() != 2) return false;
    if (!strata::kernels::native_mmvq_supported((int) t->type)) return false;
    const uint64_t nb = strata::tensor_payload_bytes(*t);
    if (nb == 0) return false;
    void* d = nullptr;
    if (cudaMalloc(&d, (size_t) nb) != cudaSuccess) return false;
    if (cudaMemcpy(d, g.tensor_data(*t), (size_t) nb, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(d);
        return false;
    }
    out.dev = d; out.type = (int) t->type;
    out.n_in = (int) t->shape[0]; out.n_out = (int) t->shape[1];
    return true;
}

// One block: the owned vectors + the engine's view of them.  The four MLA projections and, at layer level,
// the head may instead live on the device (DevBlocks) with the float vector left empty.
struct Block {
    int kind = 1;
    std::vector<float> attn_norm, ffn_norm;
    std::vector<float> mw[8];                 // host-float MLA weights (only 1,3,5,6 used when device)
    glm::MlaWeights mla;
    DevBlocks dwq_a, dwq_b, dkv_a, dwo;        // the four native MLA projections (device)
    // dense (block 0)
    std::vector<float> wg, wu, wd;
    glm::MoeGeometry dg;
    // moe
    glm::MoeGeometry gg;
    std::vector<float> router, probs_b, s_gate, s_up, s_down;
    DevBlocks dw_sg, dw_su, dw_sd;               // the shared expert as device blocks
    int s_types[3] = {0, 0, 0};                  // its GGML types (0 = host float, the fallback)
    const float* shared[3] = {nullptr, nullptr, nullptr};
    strata::kernels::cpu::NativeFmt fmt;
    bool has_fmt = false;
    cglm::Glm47TrunkLayer tl;
};

// The routed-expert row cache.  In gguf mode (a pack with no experts.bin) FileExpertSource has no
// contiguous blob in any file, so its staged path re-assembles the [gate|up|down] row out of the GGUF
// on EVERY blob() call - hit or miss - round-tripping ~5.8 MB per expert off disk each token.  A row
// is byte-identical every token, so assemble each (layer, expert) once and hand back the copy: on a hit
// no host bytes move.  This is the same fix the V100 kolibri port landed (35e627f) for its row cache.
struct RowCache {
    strata::core::ExpertSource* src = nullptr;
    std::vector<uint64_t> row_bytes;                    // per layer, from native_experts.txt
    std::unordered_map<int64_t, std::unique_ptr<uint8_t[]>> rows;
    uint64_t hits = 0, misses = 0, bytes = 0;
};
const uint8_t* blob_adapter(void* ctx, int layer, int expert) {
    RowCache* c = (RowCache*) ctx;
    const int64_t key = ((int64_t) layer << 20) | (int64_t) expert;
    auto it = c->rows.find(key);
    if (it != c->rows.end()) { ++c->hits; return it->second.get(); }
    const uint64_t nb = c->row_bytes[(size_t) layer];
    if (nb == 0) return nullptr;
    // copy_blob assembles the [gate|up|down] row STRAIGHT into our buffer - no FileExpertSource stage copy,
    // so a miss is one read off the GGUF, not two.  make_unique (not vector) so we do not zero 5.8 MB first.
    auto row = std::make_unique<uint8_t[]>(nb);
    if (!c->src->copy_blob(layer, expert, row.get())) return nullptr;
    uint8_t* p = row.get();
    c->rows.emplace(key, std::move(row));
    ++c->misses; c->bytes += nb;
    return p;
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
    bool bind_only = false, use_device = true, warm = false, dev_experts = false;
    size_t exp_budget = 6ull << 30;   // device bytes for resident expert rows (sized to the free VRAM)
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--gguf") && i + 1 < argc) gguf = argv[++i];
        else if (!std::strcmp(argv[i], "--pack") && i + 1 < argc) pack = argv[++i];
        else if (!std::strcmp(argv[i], "--tokens") && i + 1 < argc) tokens = argv[++i];
        else if (!std::strcmp(argv[i], "--layers") && i + 1 < argc) n_layer = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--bind-only")) bind_only = true;
        else if (!std::strcmp(argv[i], "--gen") && i + 1 < argc) gen = std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i], "--quiet")) verbosity = 0;
        else if (!std::strcmp(argv[i], "--cpu")) use_device = false;
        else if (!std::strcmp(argv[i], "--warm")) warm = true;
        else if (!std::strcmp(argv[i], "--dev-experts")) dev_experts = true;
        else if (!std::strcmp(argv[i], "--exp-budget-mb") && i + 1 < argc) exp_budget = (size_t) std::atoi(argv[++i]) << 20;
    }

    std::printf("glm47_run: %d blocks (block 0 dense), n_embd %d, experts %d/%d, ff %d, %s\n",
                n_layer, N_EMBD, N_USED, N_EXPERT, EXP_FF, use_device ? "GPU" : "cpu");

    // ---- the pack's expert layout + the artifact ----
    std::string err;
    strata::core::FileExpertSource src;
    src.set_gguf(gguf);
    if (!strata::kernels::cpu::expert_layout_load(pack, N_LAYER, N_EXPERT, err, N_EMBD, EXP_FF) ||
        !src.open(pack, N_LAYER, N_EXPERT, err))
        return die("pack: " + err) ? 0 : 1;

    std::map<int, int> gu_of, d_of;
    RowCache rc;
    rc.row_bytes.assign((size_t) N_LAYER, 0);
    { std::ifstream pf(std::string(pack) + "/native_experts.txt"); std::string line;
      while (std::getline(pf, line)) {
          if (line.empty() || line[0] == '#') continue;
          std::istringstream is(line);
          int64_t blk = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0; int gt = 0, dt = 0;
          if (!(is >> blk >> gt >> dt >> off >> nb >> go >> uo >> dob)) continue;
          gu_of[(int) blk] = gt; d_of[(int) blk] = dt;
          if (blk >= 0 && blk < N_LAYER) rc.row_bytes[(size_t) blk] = (uint64_t) nb;
      } }
    rc.src = &src;

    strata::GgufFile g(gguf);
    const auto t_bind0 = std::chrono::steady_clock::now();

    // ---- bind every site ----
    std::vector<Block> blk((size_t) n_layer);
    const char* nm[8] = {"attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight", "attn_k_b.weight",
                         "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_v_b.weight", "attn_output.weight"};
    int n_dev_proj = 0, n_dev_shexp = 0;
    for (int l = 0; l < n_layer; ++l) {
        Block& b = blk[(size_t) l];
        const std::string p = "blk." + std::to_string(l) + ".";
        b.kind = (l == 0) ? 0 : 1;
        if (!load(g, (p + "attn_norm.weight").c_str(), b.attn_norm, "attn_norm")) return 1;
        if (!load(g, (p + "ffn_norm.weight").c_str(), b.ffn_norm, "ffn_norm")) return 1;
        // MLA weights that stay host float: q_a_norm, wk_b, kv_a_norm, wv_b
        if (!load(g, (p + nm[1]).c_str(), b.mw[1], nm[1])) return 1;
        if (!load(g, (p + nm[3]).c_str(), b.mw[3], nm[3])) return 1;
        if (!load(g, (p + nm[5]).c_str(), b.mw[5], nm[5])) return 1;
        if (!load(g, (p + nm[6]).c_str(), b.mw[6], nm[6])) return 1;
        b.mla.q_a_norm = b.mw[1].data(); b.mla.wk_b = b.mw[3].data();
        b.mla.kv_a_norm = b.mw[5].data(); b.mla.wv_b = b.mw[6].data();
        b.mla.rope_freq_base = ROPE_BASE;
        // the four native projections: the artifact's own quantized blocks on the device, else host float
        const bool d0 = use_device && upload_blocks(g, p + nm[0], b.dwq_a);
        const bool d2 = use_device && upload_blocks(g, p + nm[2], b.dwq_b);
        const bool d4 = use_device && upload_blocks(g, p + nm[4], b.dkv_a);
        const bool d7 = use_device && upload_blocks(g, p + nm[7], b.dwo);
        if (d0) { b.mla.wq_a = (const float*) b.dwq_a.dev; b.mla.wq_a_type = b.dwq_a.type; }
        else if (!load(g, (p + nm[0]).c_str(), b.mw[0], nm[0])) return 1;
        else b.mla.wq_a = b.mw[0].data();
        if (d2) { b.mla.wq_b = (const float*) b.dwq_b.dev; b.mla.wq_b_type = b.dwq_b.type; }
        else if (!load(g, (p + nm[2]).c_str(), b.mw[2], nm[2])) return 1;
        else b.mla.wq_b = b.mw[2].data();
        if (d4) { b.mla.kv_a = (const float*) b.dkv_a.dev; b.mla.kv_a_type = b.dkv_a.type; }
        else if (!load(g, (p + nm[4]).c_str(), b.mw[4], nm[4])) return 1;
        else b.mla.kv_a = b.mw[4].data();
        if (d7) { b.mla.wo = (const float*) b.dwo.dev; b.mla.wo_type = b.dwo.type; }
        else if (!load(g, (p + nm[7]).c_str(), b.mw[7], nm[7])) return 1;
        else b.mla.wo = b.mw[7].data();
        n_dev_proj += (int) d0 + (int) d2 + (int) d4 + (int) d7;
        if (b.kind == 0) {
            if (!load(g, (p + "ffn_gate.weight").c_str(), b.wg, "dense gate")) return 1;
            if (!load(g, (p + "ffn_up.weight").c_str(), b.wu, "dense up")) return 1;
            if (!load(g, (p + "ffn_down.weight").c_str(), b.wd, "dense down")) return 1;
            b.dg.n_embd = N_EMBD; b.dg.n_expert = 0; b.dg.n_used = 0; b.dg.ff = DENSE_FF;
            b.dg.w_scale = 1.0f; b.dg.norm_w = false; b.dg.clamp_exp = 0.0f; b.dg.clamp_shexp = 0.0f;
        } else {
            if (!load(g, (p + "ffn_gate_inp.weight").c_str(), b.router, "router")) return 1;
            if (!load(g, (p + "exp_probs_b.bias").c_str(), b.probs_b, "probs_b")) return 1;
            // the shared expert: the artifact's own quantized blocks on the device (native GLU), else host float
            const bool sg = use_device && upload_blocks(g, p + "ffn_gate_shexp.weight", b.dw_sg);
            const bool su = use_device && upload_blocks(g, p + "ffn_up_shexp.weight", b.dw_su);
            const bool sd = use_device && upload_blocks(g, p + "ffn_down_shexp.weight", b.dw_sd);
            if (sg && su && sd) {
                b.shared[0] = (const float*) b.dw_sg.dev; b.shared[1] = (const float*) b.dw_su.dev;
                b.shared[2] = (const float*) b.dw_sd.dev;
                b.s_types[0] = b.dw_sg.type; b.s_types[1] = b.dw_su.type; b.s_types[2] = b.dw_sd.type;
                n_dev_shexp += 3;
            } else {
                if (!load(g, (p + "ffn_gate_shexp.weight").c_str(), b.s_gate, "shexp gate")) return 1;
                if (!load(g, (p + "ffn_up_shexp.weight").c_str(), b.s_up, "shexp up")) return 1;
                if (!load(g, (p + "ffn_down_shexp.weight").c_str(), b.s_down, "shexp down")) return 1;
                b.shared[0] = b.s_gate.data(); b.shared[1] = b.s_up.data(); b.shared[2] = b.s_down.data();
            }
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
    // the head: artifact blocks on the device when they have a native kernel, else host float
    DevBlocks dhead;
    const bool head_dev = use_device && upload_blocks(g, "output.weight", dhead);
    int V = 0;
    if (head_dev) {
        V = dhead.n_out;
        if (verbosity) std::printf("  head on the device: [%d x %d] type %d\n", dhead.n_in, dhead.n_out, dhead.type);
    } else {
        if (verbosity) std::printf("  head on the host (no native kernel / --cpu)\n");
        if (!load(g, "output.weight", Wout, "output")) return 1;
        V = (int) (Wout.size() / N_EMBD);
    }
    if (verbosity) std::printf("  binding the embedding (154880 rows, this is the slow part)\n");
    if (!load(g, "token_embd.weight", tok_emb, "token_embd")) return 1;

    // install the MLA native-projection hook: the four projections now run on the device
    if (use_device && n_dev_proj > 0) glm::mla_set_native_project(glm::glm47_native_project);
    // and the decoupled-RoPE attention's K-absorb / V-un-absorb head-matvecs on the device
    if (use_device && n_dev_proj > 0) glm::mla_set_rope_head_cuda(true);
    // install the native GLU: the shared expert runs on the device too
    if (use_device && n_dev_shexp > 0) cglm::glm_set_native_ffn(glm::glm47_native_ffn);
    // install the routed-expert device path (native_mmvq over resident rows - the engine's own expert guard
    // only admits the IQ types, which this model's k-quant experts are not)
    if (use_device && dev_experts) glm::glm47_install_device_experts(exp_budget);

    // the engine's per-layer views
    std::vector<cglm::Glm47TrunkLayer> arr((size_t) n_layer);
    for (int l = 0; l < n_layer; ++l) {
        Block& b = blk[(size_t) l];
        cglm::Glm47TrunkLayer& a = b.tl;
        a.kind = b.kind; a.attn_norm = b.attn_norm.data(); a.ffn_norm = b.ffn_norm.data(); a.mla = &b.mla;
        if (b.kind == 0) { a.ffn_gate = b.wg.data(); a.ffn_up = b.wu.data(); a.ffn_down = b.wd.data(); a.dense_g = b.dg; }
        else {
            a.moe_router = b.router.data(); a.moe_probs_b = b.probs_b.data(); a.moe_g = &b.gg; a.shexp = b.shared;
            a.shexp_types = (b.s_types[0] && b.s_types[1] && b.s_types[2]) ? b.s_types : nullptr;  // native GLU or float
            a.moe_native_fmt = &b.fmt; a.moe_native_blob = &blob_adapter; a.moe_native_ctx = &rc;
        }
        arr[(size_t) l] = a;
    }

    if (verbosity) std::printf("  bind+upload: %.0f ms\n",
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_bind0).count());

    // --warm: assemble every routed expert's row once up front, so decode runs against host-resident rows
    // (the V100's prefill-then-gen warm cache).  Layer 0 is dense and has no experts.
    if (warm) {
        const auto w0 = std::chrono::steady_clock::now();
        for (int l = 1; l < n_layer; ++l)
            for (int e = 0; e < N_EXPERT; ++e) blob_adapter(&rc, l, e);
        const double wms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
        if (verbosity) std::printf("  warmed the expert-row cache: %zu rows, %.2f GiB, %.1f s\n",
                                   rc.rows.size(), (double) rc.bytes / 1073741824.0, wms / 1000.0);
        rc.hits = rc.misses = 0;
    }

    glm::MlaGeometry mg; mg.n_embd = N_EMBD; mg.n_head = N_HEAD; mg.head_dim = HEAD_DIM;
    mg.kv_lora = KV_LORA; mg.q_lora = Q_LORA; mg.n_rot = N_ROT;

    // ---- the prompt tokens -> hidden states ----
    std::vector<int> ids;
    { std::stringstream ss(tokens); std::string t; while (std::getline(ss, t, ',')) if (!t.empty()) ids.push_back(std::atoi(t.c_str())); }
    if (ids.empty() && !bind_only) return die("no --tokens given") ? 0 : 1;

    if (bind_only) {
        std::printf("  bind-only: %d tokens, embed rows %d x %d, head %d x %d, device projections %d/%d\n",
                    (int) ids.size(), (int)(tok_emb.size() / (N_EMBD > 0 ? N_EMBD : 1)), N_EMBD,
                    V, N_EMBD, n_dev_proj, n_layer * 4);
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
    std::vector<float> hn((size_t) N_EMBD), logits((size_t) V);
    std::vector<int> order((size_t) V);
    auto predict = [&]() -> int {
        glm::rms_norm_gain(output_norm.data(), N_EMBD, hidden.data(), hn.data(), EPS);
        bool dev_ok = head_dev && glm::glm47_device_gemv(dhead.type, dhead.dev, hn.data(), N_EMBD, V, logits.data());
        if (!dev_ok) {
            for (int v = 0; v < V; ++v) {
                double s = 0.0;
                for (int e = 0; e < N_EMBD; ++e) s += (double) Wout[(size_t) v * N_EMBD + e] * (double) hn[(size_t) e];
                logits[(size_t) v] = (float) s;
            }
        }
        for (int v = 0; v < V; ++v) order[(size_t) v] = v;
        std::partial_sort(order.begin(), order.begin() + 5, order.end(),
                          [&](int a, int b) { return logits[(size_t) a] > logits[(size_t) b]; });
        return order[0];
    };

    double feed_ms = 0.0;
    int feed_n = 0;
    auto tfeed = [&](int tok, int pos) -> bool {
        const auto a = std::chrono::steady_clock::now();
        const bool ok = feed(tok, pos);
        feed_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count();
        ++feed_n;
        return ok;
    };

    for (size_t t = 0; t < ids.size(); ++t)
        if (!tfeed(ids[t], (int) t)) return die("trunk (prompt)") ? 0 : 1;

    std::printf("  %zu prompt tokens -> generated:\n", ids.size());
    for (int g = 0; g < gen; ++g) {
        const int t1 = predict();
        std::printf("    step %d: %d   (margin %.4f over %d)\n", g, t1,
                    (double) (logits[(size_t) order[0]] - logits[(size_t) order[1]]), order[1]);
        if (g + 1 < gen && !tfeed(t1, (int) ids.size() + g)) return die("trunk (gen)") ? 0 : 1;
    }
    if (verbosity && feed_n > 0)
        std::printf("  trunk: %.0f ms over %d forward(s) = %.0f ms/token  [attn %.0f | ffn %.0f | proj %.0f | ffn-hook %.0f]\n",
                    feed_ms, feed_n, feed_ms / feed_n, cglm::glm47_trunk_mla_ms(), cglm::glm47_trunk_ffn_ms(),
                    glm::glm47_project_ms(), glm::glm47_ffn_ms());
    if (verbosity) {
        double moe_ms[4] = {0, 0, 0, 0};
        long moe_calls = 0;
        cglm::glm_moe_single_token_timing(moe_ms, &moe_calls);
        std::printf("  moe (%ld calls): route %.0f | blob %.0f | experts %.0f | shared %.0f ms\n",
                    moe_calls, moe_ms[0], moe_ms[1], moe_ms[2], moe_ms[3]);
        std::printf("  row cache: %llu hits / %llu misses, %.2f GiB assembled once\n",
                    (unsigned long long) rc.hits, (unsigned long long) rc.misses,
                    (double) rc.bytes / 1073741824.0);
        uint64_t eh = 0, em = 0, ee = 0; size_t eb = 0;
        glm::glm47_device_expert_stats(&eh, &em, &ee, &eb);
        if (eh || em) std::printf("  expert rows on the device: %llu hits / %llu misses (%llu evicted), %.2f GiB\n",
                                  (unsigned long long) eh, (unsigned long long) em, (unsigned long long) ee,
                                  (double) eb / 1073741824.0);
    }
    std::printf("  top-5 (last):");
    for (int i = 0; i < 5; ++i) std::printf(" %d(%.3f)", order[(size_t) i], (double) logits[(size_t) order[(size_t) i]]);
    std::printf("\n");
    std::printf("glm47_run: OK\n");
    return 0;
}
