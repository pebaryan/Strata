#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t kHidden = 2880;
constexpr int64_t kTokens = 20;
constexpr int64_t kQWidth = 64 * 64;
constexpr int64_t kKvWidth = 8 * 64;
constexpr size_t kHeaderWords = 11;
constexpr uint64_t kMagic = 0x31544253;

struct Capture {
    std::vector<uint8_t> bytes;
    uint64_t type = 0;
    uint64_t ne[4] = {};
    uint64_t nb[4] = {};
    const uint8_t* data() const { return bytes.data() + kHeaderWords * sizeof(uint64_t); }
};

Capture read_capture(const std::string& path) {
    Capture c;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open capture: " + path);
    const auto length = in.tellg();
    if (length < static_cast<std::streamoff>(kHeaderWords * sizeof(uint64_t))) throw std::runtime_error("truncated capture: " + path);
    c.bytes.resize(static_cast<size_t>(length));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(c.bytes.data()), length);
    if (!in) throw std::runtime_error("cannot read capture: " + path);
    uint64_t h[kHeaderWords];
    std::memcpy(h, c.bytes.data(), sizeof(h));
    if (h[0] != kMagic || h[2] + sizeof(h) != c.bytes.size()) throw std::runtime_error("invalid capture header: " + path);
    c.type = h[1];
    for (size_t i = 0; i < 4; ++i) { c.ne[i] = h[3+i]; c.nb[i] = h[7+i]; }
    return c;
}

std::vector<float> dense_f32(const Capture& c) {
    if (c.type != GGML_TYPE_F32) throw std::runtime_error("expected F32 checkpoint");
    const size_t count = static_cast<size_t>(c.ne[0] * c.ne[1] * c.ne[2]);
    std::vector<float> out(count);
    size_t at = 0;
    for (uint64_t i2 = 0; i2 < c.ne[2]; ++i2)
        for (uint64_t i1 = 0; i1 < c.ne[1]; ++i1)
            for (uint64_t i0 = 0; i0 < c.ne[0]; ++i0, ++at)
                std::memcpy(&out[at], c.data() + i0*c.nb[0] + i1*c.nb[1] + i2*c.nb[2], sizeof(float));
    return out;
}

ggml_tensor* make_weight(ggml_context* ctx, const strata::GgufFile& model,
                         const std::string& name, int64_t ne0, int64_t ne1) {
    const auto* info = model.find(name);
    if (!info || info->shape.size() != 2 || info->shape[0] != static_cast<uint64_t>(ne0) || info->shape[1] != static_cast<uint64_t>(ne1))
        throw std::runtime_error("unexpected attention matrix geometry: " + name);
    return ggml_new_tensor_2d(ctx, static_cast<ggml_type>(info->type), ne0, ne1);
}

ggml_tensor* make_bias(ggml_context* ctx, const strata::GgufFile& model, const std::string& name, int64_t length) {
    const auto* info = model.find(name);
    if (!info) throw std::runtime_error("missing attention vector: " + name);
    if (info->type != GGML_TYPE_F32 || info->shape.size() != 1 || info->shape[0] != static_cast<uint64_t>(length))
        throw std::runtime_error("unexpected attention vector geometry: " + name + " type=" + info->type_name() +
            " dims=" + std::to_string(info->shape.size()) + " first=" + std::to_string(info->shape[0]));
    return ggml_new_tensor_1d(ctx, GGML_TYPE_F32, length);
}

void upload_model_tensor(ggml_tensor* dst, const strata::GgufFile& model, const std::string& name) {
    const auto* info = model.find(name);
    ggml_backend_tensor_set(dst, model.tensor_data(*info), 0, static_cast<size_t>(strata::tensor_payload_bytes(*info)));
}

void write_capture(const std::string& path, ggml_tensor* tensor, ggml_backend_t backend) {
    const size_t bytes = ggml_nbytes(tensor);
    std::vector<uint8_t> payload(bytes);
    ggml_backend_tensor_get(tensor, payload.data(), 0, bytes);
    const uint64_t header[] = {kMagic, static_cast<uint64_t>(tensor->type), static_cast<uint64_t>(bytes),
        static_cast<uint64_t>(tensor->ne[0]), static_cast<uint64_t>(tensor->ne[1]),
        static_cast<uint64_t>(tensor->ne[2]), static_cast<uint64_t>(tensor->ne[3]),
        static_cast<uint64_t>(tensor->nb[0]), static_cast<uint64_t>(tensor->nb[1]),
        static_cast<uint64_t>(tensor->nb[2]), static_cast<uint64_t>(tensor->nb[3])};
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(header), sizeof(header));
    out.write(reinterpret_cast<const char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
    if (!out) throw std::runtime_error("failed writing candidate activation: " + path);
    (void) backend;
}

void compare_capture(const char* name, const std::vector<float>& actual, const Capture& expected,
                    double& worst_rel_l2, double& worst_abs) {
    if (actual.size() != expected.ne[0] * expected.ne[1] * expected.ne[2]) throw std::runtime_error(std::string(name) + " capture shape mismatch");
    const auto reference = dense_f32(expected);
    double e2 = 0.0, r2 = 0.0, max_abs = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double d = static_cast<double>(actual[i]) - reference[i];
        e2 += d*d; r2 += static_cast<double>(reference[i])*reference[i]; max_abs = (std::max)(max_abs, std::abs(d));
    }
    const double rel = std::sqrt(e2 / (std::max)(r2, 1.0e-30));
    worst_rel_l2 = (std::max)(worst_rel_l2, rel);
    worst_abs = (std::max)(worst_abs, max_abs);
    std::cout << name << " max_abs=" << max_abs << " relative_l2=" << rel << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 3) { std::cerr << "Usage: " << argv[0] << " MODEL.gguf CAPTURE_PREFIX\n"; return 2; }
    try {
        const std::string prefix = argv[2];
        const auto input_ref = read_capture(prefix + ".attn_norm-0.bin");
        const auto q_pre_ref = read_capture(prefix + ".Qcur_pre_rope-0.bin");
        const auto q_rope_ref = read_capture(prefix + ".Qcur_rope-0.bin");
        const auto k_pre_ref = read_capture(prefix + ".Kcur_pre_rope-0.bin");
        const auto k_rope_ref = read_capture(prefix + ".Kcur_rope-0.bin");
        const auto v_ref = read_capture(prefix + ".Vcur-0.bin");
        const auto ffn_inp_ref = read_capture(prefix + ".ffn_inp-0.bin");
        const auto post_norm_ref = read_capture(prefix + ".attn_post_norm-0.bin");
        std::ifstream token_file(prefix + ".tokens.txt");
        std::vector<int32_t> token_ids;
        int32_t token_id = 0;
        while (token_file >> token_id) token_ids.push_back(token_id);
        if (!token_file.eof() || token_ids.size() != kTokens) throw std::runtime_error("prompt token capture must contain 20 token IDs");

        strata::GgufFile model(argv[1]);
        std::string contract = strata::check_gpt_oss_120b_architecture(model);
        if (contract.empty()) contract = strata::check_gpt_oss_120b_tensors(model);
        if (!contract.empty()) throw std::runtime_error("model contract failed: " + contract);

        ggml_backend_load_all();
        ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_t cpu = cpu_device ? ggml_backend_dev_init(cpu_device, nullptr) : nullptr;
        if (!cpu) throw std::runtime_error("CPU ggml backend unavailable");
        ggml_init_params params{}; params.mem_size = 32 * 1024 * 1024; params.no_alloc = true;
        ggml_context* ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");

        auto* q_w = make_weight(ctx, model, "blk.0.attn_q.weight", kHidden, kQWidth);
        auto* k_w = make_weight(ctx, model, "blk.0.attn_k.weight", kHidden, kKvWidth);
        auto* v_w = make_weight(ctx, model, "blk.0.attn_v.weight", kHidden, kKvWidth);
        auto* q_b = make_bias(ctx, model, "blk.0.attn_q.bias", kQWidth);
        auto* k_b = make_bias(ctx, model, "blk.0.attn_k.bias", kKvWidth);
        auto* v_b = make_bias(ctx, model, "blk.0.attn_v.bias", kKvWidth);
        auto* wo = make_weight(ctx, model, "blk.0.attn_output.weight", kQWidth, kHidden);
        auto* wo_b = make_bias(ctx, model, "blk.0.attn_output.bias", kHidden);
        auto* sinks = make_bias(ctx, model, "blk.0.attn_sinks.weight", 64);
        const auto* emb_info = model.find("token_embd.weight");
        if (!emb_info || emb_info->shape.size() != 2 || emb_info->shape[0] != kHidden)
            throw std::runtime_error("unexpected token embedding matrix geometry");
        auto* embeddings = make_weight(ctx, model, "token_embd.weight", kHidden, static_cast<int64_t>(emb_info->shape[1]));
        auto* token_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kTokens);
        auto* attn_norm_w = make_bias(ctx, model, "blk.0.attn_norm.weight", kHidden);
        auto* post_norm_w = make_bias(ctx, model, "blk.0.post_attention_norm.weight", kHidden);
        auto* positions = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, kTokens);
        auto* mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kTokens, kTokens, 64);
        auto* sink_scores = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, kTokens, 64);
        auto* sink_values = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, 64, 1, 64);
        auto* layer_input = ggml_get_rows(ctx, embeddings, token_tensor);
        auto* attn_norm = ggml_mul(ctx, ggml_rms_norm(ctx, layer_input, 1.0e-5f), attn_norm_w);
        auto* q2 = ggml_add(ctx, ggml_mul_mat(ctx, q_w, attn_norm), q_b);
        auto* k2 = ggml_add(ctx, ggml_mul_mat(ctx, k_w, attn_norm), k_b);
        auto* v2 = ggml_add(ctx, ggml_mul_mat(ctx, v_w, attn_norm), v_b);
        auto* q3 = ggml_reshape_3d(ctx, q2, 64, 64, kTokens);
        auto* k3 = ggml_reshape_3d(ctx, k2, 64, 8, kTokens);
        auto* v3 = ggml_reshape_3d(ctx, v2, 64, 8, kTokens);
        // llama.cpp's default YaRN attention factor is get_mscale(32, 1),
        // then canceled by 1 / (1 + 0.1 * log(32)); the product is 1.0.
        auto* q_rope = ggml_rope_ext(ctx, q3, positions, nullptr, 64, GGML_ROPE_TYPE_NEOX, 4096,
                                     150000.0f, 1.0f/32.0f, 1.0f, 1.0f, 32.0f, 1.0f);
        auto* k_rope = ggml_rope_ext(ctx, k3, positions, nullptr, 64, GGML_ROPE_TYPE_NEOX, 4096,
                                     150000.0f, 1.0f/32.0f, 1.0f, 1.0f, 32.0f, 1.0f);
        auto* qh = ggml_permute(ctx, q_rope, 0, 2, 1, 3);
        auto* kh = ggml_cast(ctx, ggml_permute(ctx, k_rope, 0, 2, 1, 3), GGML_TYPE_F16);
        auto* vh = ggml_cast(ctx, ggml_permute(ctx, v3, 0, 2, 1, 3), GGML_TYPE_F16);
        // Repeat each KV head across its contiguous group of eight query heads.
        kh = ggml_reshape_4d(ctx, kh, 64, kTokens, 1, 8);
        kh = ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, kh, 64, kTokens, 8, 8), 64, kTokens, 64);
        vh = ggml_reshape_4d(ctx, vh, 64, kTokens, 1, 8);
        vh = ggml_reshape_3d(ctx, ggml_repeat_4d(ctx, vh, 64, kTokens, 8, 8), 64, kTokens, 64);
        auto* kq = ggml_mul_mat(ctx, kh, qh);
        ggml_prec_set_acc(kq, GGML_PREC_F32);
        kq = ggml_scale(ctx, kq, 1.0f / 8.0f);
        kq = ggml_add(ctx, kq, mask);
        kq = ggml_concat(ctx, kq, sink_scores, 0);
        auto* probs = ggml_soft_max(ctx, kq);
        auto* v_with_sink = ggml_concat(ctx, vh, sink_values, 1);
        auto* v_transposed = ggml_cont(ctx, ggml_permute(ctx, v_with_sink, 1, 0, 2, 3));
        auto* context_heads = ggml_mul_mat(ctx, v_transposed, probs);
        auto* context_merged = ggml_cont(ctx, ggml_permute(ctx, context_heads, 0, 2, 1, 3));
        auto* context_2d = ggml_reshape_2d(ctx, context_merged, kQWidth, kTokens);
        auto* kqv_out = ggml_cont(ctx, context_2d);
        auto* attn_out = ggml_add(ctx, ggml_mul_mat(ctx, wo, kqv_out), wo_b);
        auto* ffn_inp = ggml_add(ctx, attn_out, layer_input);
        auto* post_norm = ggml_mul(ctx, ggml_rms_norm(ctx, ffn_inp, 1.0e-5f), post_norm_w);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, layer_input); ggml_build_forward_expand(graph, attn_norm);
        ggml_build_forward_expand(graph, q3); ggml_build_forward_expand(graph, k3); ggml_build_forward_expand(graph, v3);
        ggml_build_forward_expand(graph, q_rope); ggml_build_forward_expand(graph, k_rope);
        ggml_build_forward_expand(graph, kqv_out); ggml_build_forward_expand(graph, attn_out);
        ggml_build_forward_expand(graph, ffn_inp); ggml_build_forward_expand(graph, post_norm);
        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        if (!buffer) throw std::runtime_error("CPU tensor allocation failed");
        upload_model_tensor(q_w, model, "blk.0.attn_q.weight"); upload_model_tensor(k_w, model, "blk.0.attn_k.weight");
        upload_model_tensor(v_w, model, "blk.0.attn_v.weight"); upload_model_tensor(q_b, model, "blk.0.attn_q.bias");
        upload_model_tensor(k_b, model, "blk.0.attn_k.bias"); upload_model_tensor(v_b, model, "blk.0.attn_v.bias");
        upload_model_tensor(wo, model, "blk.0.attn_output.weight"); upload_model_tensor(wo_b, model, "blk.0.attn_output.bias");
        upload_model_tensor(sinks, model, "blk.0.attn_sinks.weight");
        upload_model_tensor(embeddings, model, "token_embd.weight");
        upload_model_tensor(attn_norm_w, model, "blk.0.attn_norm.weight");
        upload_model_tensor(post_norm_w, model, "blk.0.post_attention_norm.weight");
        ggml_backend_tensor_set(token_tensor, token_ids.data(), 0, token_ids.size()*sizeof(int32_t));
        std::vector<int32_t> pos(kTokens); for (int32_t i=0;i<kTokens;++i) pos[i]=i;
        ggml_backend_tensor_set(positions, pos.data(), 0, pos.size()*sizeof(int32_t));
        std::vector<float> mask_values(static_cast<size_t>(kTokens*kTokens*64));
        for (int64_t h=0;h<64;++h) for (int64_t q=0;q<kTokens;++q) for (int64_t key=0;key<kTokens;++key)
            mask_values[static_cast<size_t>(key + kTokens*(q + kTokens*h))] = key > q ? -INFINITY : 0.0f;
        ggml_backend_tensor_set(mask, mask_values.data(), 0, mask_values.size()*sizeof(float));
        std::vector<float> sink_input(static_cast<size_t>(64*kTokens));
        std::vector<float> sink_per_head(64);
        ggml_backend_tensor_get(sinks, sink_per_head.data(), 0, sink_per_head.size()*sizeof(float));
        for (int64_t h=0;h<64;++h) for (int64_t q=0;q<kTokens;++q)
            sink_input[static_cast<size_t>(q + kTokens*h)] = sink_per_head[static_cast<size_t>(h)];
        ggml_backend_tensor_set(sink_scores, sink_input.data(), 0, sink_input.size()*sizeof(float));
        std::vector<uint16_t> zero_values(static_cast<size_t>(64*64), 0);
        ggml_backend_tensor_set(sink_values, zero_values.data(), 0, zero_values.size()*sizeof(uint16_t));
        const auto status = ggml_backend_graph_compute(cpu, graph);
        ggml_backend_synchronize(cpu);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("CPU attention projection graph failed: " + std::to_string(status));

        auto get = [&](ggml_tensor* tensor) {
            std::vector<float> values(static_cast<size_t>(ggml_nelements(tensor)));
            ggml_backend_tensor_get(tensor, values.data(), 0, values.size()*sizeof(float));
            return values;
        };
        double worst_rel = 0.0, worst_abs = 0.0;
        compare_capture("attn_norm", get(attn_norm), input_ref, worst_rel, worst_abs);
        compare_capture("Q_pre_rope", get(q3), q_pre_ref, worst_rel, worst_abs);
        compare_capture("Q_rope", get(q_rope), q_rope_ref, worst_rel, worst_abs);
        compare_capture("K_pre_rope", get(k3), k_pre_ref, worst_rel, worst_abs);
        compare_capture("K_rope", get(k_rope), k_rope_ref, worst_rel, worst_abs);
        compare_capture("V", get(v3), v_ref, worst_rel, worst_abs);
        compare_capture("attention_output", get(attn_out), read_capture(prefix + ".attn_out-0.bin"), worst_rel, worst_abs);
        compare_capture("ffn_inp", get(ffn_inp), ffn_inp_ref, worst_rel, worst_abs);
        compare_capture("attn_post_norm", get(post_norm), post_norm_ref, worst_rel, worst_abs);
        write_capture(prefix + ".candidate.ffn_inp-0.bin", ffn_inp, cpu);
        write_capture(prefix + ".candidate.attn_post_norm-0.bin", post_norm, cpu);
        std::cout << "worst_max_abs=" << worst_abs << " worst_relative_l2=" << worst_rel
                  << " tolerance_max_abs=0.05 tolerance_relative_l2=0.003\n";
        ggml_backend_buffer_free(buffer); ggml_free(ctx); ggml_backend_free(cpu);
        const bool passed = worst_abs <= 5.0e-2 && worst_rel <= 3.0e-3;
        std::cout << (passed ? "ggml attention/block-input parity PASS\n" : "ggml attention/block-input parity FAIL\n");
        return passed ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "attention parity failed: " << e.what() << '\n'; return 1;
    }
}
