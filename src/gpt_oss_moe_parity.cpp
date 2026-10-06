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
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int64_t kWidth = 2880;
constexpr int64_t kExpertCount = 128;
constexpr int64_t kTopK = 4;
constexpr size_t kCaptureHeaderWords = 11;
constexpr uint64_t kCaptureMagic = 0x31544253;

struct Capture {
    std::vector<uint8_t> bytes;
    uint64_t type = 0;
    uint64_t ne[4] = {};
    uint64_t nb[4] = {};
    const uint8_t* data() const { return bytes.data() + kCaptureHeaderWords * sizeof(uint64_t); }
};

Capture read_capture(const std::string& path) {
    Capture capture;
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open reference capture: " + path);
    const auto length = in.tellg();
    if (length < static_cast<std::streamoff>(kCaptureHeaderWords * sizeof(uint64_t)))
        throw std::runtime_error("capture header is truncated: " + path);
    capture.bytes.resize(static_cast<size_t>(length));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(capture.bytes.data()), length);
    if (!in) throw std::runtime_error("cannot read reference capture: " + path);
    uint64_t header[kCaptureHeaderWords];
    std::memcpy(header, capture.bytes.data(), sizeof(header));
    if (header[0] != kCaptureMagic || header[2] + sizeof(header) != capture.bytes.size())
        throw std::runtime_error("invalid capture header or payload length: " + path);
    capture.type = header[1];
    for (size_t i = 0; i < 4; ++i) {
        capture.ne[i] = header[3 + i];
        capture.nb[i] = header[7 + i];
    }
    return capture;
}

std::vector<float> read_f32_column(const Capture& capture, uint64_t column) {
    if (capture.type != GGML_TYPE_F32 || capture.ne[0] != kWidth || column >= capture.ne[1])
        throw std::runtime_error("capture is not an expected F32 hidden-state column");
    std::vector<float> values(static_cast<size_t>(kWidth));
    for (int64_t row = 0; row < kWidth; ++row) {
        std::memcpy(&values[static_cast<size_t>(row)], capture.data() + row * capture.nb[0] + column * capture.nb[1], sizeof(float));
    }
    return values;
}

void load_capture_f32_vector(ggml_tensor* tensor, const Capture& capture, uint64_t column) {
    const auto values = read_f32_column(capture, column);
    ggml_backend_tensor_set(tensor, values.data(), 0, values.size() * sizeof(float));
}

void load_expert_weight(ggml_tensor* tensor, const strata::GgufFile& model, const std::string& name) {
    const auto* info = model.find(name);
    if (!info || info->type != 39 || info->shape.size() != 3 ||
        info->shape[0] != kWidth || info->shape[1] != kWidth || info->shape[2] != kExpertCount)
        throw std::runtime_error("unexpected GPT-OSS expert tensor: " + name);
    const size_t bytes = static_cast<size_t>(strata::tensor_payload_bytes(*info));
    ggml_backend_tensor_set(tensor, model.tensor_data(*info), 0, bytes);
}

void load_expert_bias(ggml_tensor* tensor, const strata::GgufFile& model, const std::string& name) {
    const auto* info = model.find(name);
    if (!info || info->type != GGML_TYPE_F32 || info->shape.size() != 2 ||
        info->shape[0] != kWidth || info->shape[1] != kExpertCount)
        throw std::runtime_error("unexpected GPT-OSS expert bias tensor: " + name);
    const size_t bytes = static_cast<size_t>(strata::tensor_payload_bytes(*info));
    ggml_backend_tensor_set(tensor, model.tensor_data(*info), 0, bytes);
}

std::vector<int32_t> read_expert_ids(const Capture& capture, uint64_t token) {
    if (capture.type != GGML_TYPE_I32 || capture.ne[0] != kTopK || token >= capture.ne[1])
        throw std::runtime_error("capture is not an expected top-k expert ID tensor");
    std::vector<int32_t> ids(kTopK);
    for (int64_t i = 0; i < kTopK; ++i)
        std::memcpy(&ids[static_cast<size_t>(i)], capture.data() + i * capture.nb[0] + token * capture.nb[1], sizeof(int32_t));
    for (const int32_t id : ids) if (id < 0 || id >= kExpertCount) throw std::runtime_error("captured expert ID is out of range");
    return ids;
}

std::vector<float> read_expert_weights(const Capture& capture, uint64_t token) {
    if (capture.type != GGML_TYPE_F32 || capture.ne[1] != kTopK || token >= capture.ne[2])
        throw std::runtime_error("capture is not an expected normalized top-k weight tensor");
    std::vector<float> weights(kTopK);
    for (int64_t i = 0; i < kTopK; ++i)
        std::memcpy(&weights[static_cast<size_t>(i)], capture.data() + i * capture.nb[1] + token * capture.nb[2], sizeof(float));
    return weights;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 4 && argc != 5) {
        std::cerr << "Usage: " << argv[0] << " MODEL.gguf CAPTURE_PREFIX TOKEN_INDEX [ACTIVATION_CAPTURE]\n";
        return 2;
    }
    try {
        const std::string prefix = argv[2];
        const uint64_t token = std::stoull(argv[3]);
        const Capture input_ref = read_capture(argc == 5 ? argv[4] : prefix + ".attn_post_norm-0.bin");
        const Capture ids_ref = read_capture(prefix + ".ffn_moe_topk-0.bin");
        const Capture weights_ref = read_capture(prefix + ".ffn_moe_weights_softmax-0.bin");
        const Capture output_ref = read_capture(prefix + ".ffn_moe_out-0.bin");
        const Capture layer_output_ref = read_capture(prefix + ".l_out-0.bin");
        const auto input = read_f32_column(input_ref, token);
        const auto ids = read_expert_ids(ids_ref, token);
        const auto weights = read_expert_weights(weights_ref, token);
        const auto expected = read_f32_column(output_ref, token);

        strata::GgufFile model(argv[1]);
        std::string error = strata::check_gpt_oss_120b_architecture(model);
        if (error.empty()) error = strata::check_gpt_oss_120b_tensors(model);
        if (!error.empty()) throw std::runtime_error("model contract failed: " + error);

        ggml_backend_load_all();
        ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_t cpu = cpu_device ? ggml_backend_dev_init(cpu_device, nullptr) : nullptr;
        if (!cpu) throw std::runtime_error("CPU ggml backend is unavailable");

        ggml_init_params params{};
        params.mem_size = 64 * 1024 * 1024;
        params.no_alloc = true;
        ggml_context* ctx = ggml_init(params);
        if (!ctx) throw std::runtime_error("ggml_init failed");

        auto* gate_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, kWidth, kWidth, kExpertCount);
        auto* up_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, kWidth, kWidth, kExpertCount);
        auto* down_w = ggml_new_tensor_3d(ctx, GGML_TYPE_MXFP4, kWidth, kWidth, kExpertCount);
        auto* gate_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, kExpertCount);
        auto* up_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, kExpertCount);
        auto* down_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, kExpertCount);
        auto* x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, kWidth, kTopK, 1);
        const auto* router_info = model.find("blk.0.ffn_gate_inp.weight");
        const auto* router_bias_info = model.find("blk.0.ffn_gate_inp.bias");
        if (!router_info || router_info->shape.size() != 2 || router_info->shape[0] != kWidth || router_info->shape[1] != kExpertCount ||
            !router_bias_info || router_bias_info->type != GGML_TYPE_F32 || router_bias_info->shape.size() != 1 || router_bias_info->shape[0] != kExpertCount)
            throw std::runtime_error("unexpected GPT-OSS router tensor geometry");
        auto* router_w = ggml_new_tensor_2d(ctx, static_cast<ggml_type>(router_info->type), kWidth, kExpertCount);
        auto* router_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, kExpertCount);
        auto* router_x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, 1);
        ggml_tensor* router_logits = ggml_add(ctx, ggml_mul_mat(ctx, router_w, router_x), router_b);
        ggml_tensor* routed_ids = ggml_argsort_top_k(ctx, router_logits, kTopK);
        ggml_tensor* routed_weights = ggml_reshape_3d(ctx,
            ggml_soft_max(ctx, ggml_reshape_2d(ctx,
                ggml_get_rows(ctx, ggml_reshape_3d(ctx, router_logits, 1, kExpertCount, 1), routed_ids), kTopK, 1)),
            1, kTopK, 1);
        ggml_tensor* gate = ggml_mul_mat_id(ctx, gate_w, x, routed_ids);
        gate = ggml_add_id(ctx, gate, gate_b, routed_ids);
        ggml_tensor* up = ggml_mul_mat_id(ctx, up_w, x, routed_ids);
        up = ggml_add_id(ctx, up, up_b, routed_ids);
        ggml_tensor* hidden = ggml_swiglu_oai(ctx, gate, up, 1.702f, 7.0f);
        ggml_tensor* down = ggml_mul_mat_id(ctx, down_w, hidden, routed_ids);
        down = ggml_add_id(ctx, down, down_b, routed_ids);
        ggml_tensor* weighted = ggml_mul(ctx, down, routed_weights);
        ggml_tensor* reduced = ggml_view_2d(ctx, weighted, kWidth, 1, weighted->nb[2], 0);
        for (int64_t i = 1; i < kTopK; ++i) {
            auto* expert = ggml_view_2d(ctx, weighted, kWidth, 1, weighted->nb[2], static_cast<size_t>(i) * weighted->nb[1]);
            reduced = ggml_add(ctx, reduced, expert);
        }
        reduced = ggml_cont(ctx, reduced);
        ggml_cgraph* graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, reduced);
        ggml_build_forward_expand(graph, routed_ids);
        ggml_build_forward_expand(graph, routed_weights);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, cpu);
        if (!buffer) throw std::runtime_error("CPU tensor allocation failed");
        load_expert_weight(gate_w, model, "blk.0.ffn_gate_exps.weight");
        load_expert_weight(up_w, model, "blk.0.ffn_up_exps.weight");
        load_expert_weight(down_w, model, "blk.0.ffn_down_exps.weight");
        load_expert_bias(gate_b, model, "blk.0.ffn_gate_exps.bias");
        load_expert_bias(up_b, model, "blk.0.ffn_up_exps.bias");
        load_expert_bias(down_b, model, "blk.0.ffn_down_exps.bias");
        ggml_backend_tensor_set(router_w, model.tensor_data(*router_info), 0,
            static_cast<size_t>(strata::tensor_payload_bytes(*router_info)));
        ggml_backend_tensor_set(router_b, model.tensor_data(*router_bias_info), 0,
            static_cast<size_t>(strata::tensor_payload_bytes(*router_bias_info)));
        std::vector<float> tiled_input(static_cast<size_t>(kWidth * kTopK));
        for (int64_t expert = 0; expert < kTopK; ++expert)
            std::copy(input.begin(), input.end(), tiled_input.begin() + expert * kWidth);
        ggml_backend_tensor_set(x, tiled_input.data(), 0, tiled_input.size() * sizeof(float));
        ggml_backend_tensor_set(router_x, input.data(), 0, input.size() * sizeof(float));
        const ggml_status status = ggml_backend_graph_compute(cpu, graph);
        ggml_backend_synchronize(cpu);
        if (status != GGML_STATUS_SUCCESS) throw std::runtime_error("ggml CPU MoE graph failed: " + std::to_string(status));
        std::vector<float> actual(static_cast<size_t>(kWidth));
        ggml_backend_tensor_get(reduced, actual.data(), 0, actual.size() * sizeof(float));
        std::vector<int32_t> actual_ids(kTopK);
        std::vector<float> actual_weights(kTopK);
        ggml_backend_tensor_get(routed_ids, actual_ids.data(), 0, actual_ids.size() * sizeof(int32_t));
        ggml_backend_tensor_get(routed_weights, actual_weights.data(), 0, actual_weights.size() * sizeof(float));

        double squared_error = 0.0, squared_reference = 0.0, max_abs = 0.0;
        for (size_t i = 0; i < actual.size(); ++i) {
            const double delta = static_cast<double>(actual[i]) - expected[i];
            squared_error += delta * delta;
            squared_reference += static_cast<double>(expected[i]) * expected[i];
            max_abs = (std::max)(max_abs, std::abs(delta));
        }
        const double relative_l2 = std::sqrt(squared_error / (std::max)(squared_reference, 1.0e-30));
        double block_relative_l2 = 0.0, block_max_abs = 0.0;
        if (argc == 5) {
            const Capture candidate_ffn_inp = read_capture(prefix + ".candidate.ffn_inp-0.bin");
            const auto residual = read_f32_column(candidate_ffn_inp, token);
            const auto reference_layer_output = read_f32_column(layer_output_ref, token);
            double e2 = 0.0, r2 = 0.0;
            for (size_t i = 0; i < actual.size(); ++i) {
                const double candidate = static_cast<double>(actual[i]) + residual[i];
                const double delta = candidate - reference_layer_output[i];
                e2 += delta*delta;
                r2 += static_cast<double>(reference_layer_output[i])*reference_layer_output[i];
                block_max_abs = (std::max)(block_max_abs, std::abs(delta));
            }
            block_relative_l2 = std::sqrt(e2 / (std::max)(r2, 1.0e-30));
        }
        std::cout << "token=" << token << " selected_experts=";
        for (size_t i = 0; i < ids.size(); ++i) std::cout << (i ? "," : "") << ids[i];
        std::cout << " weights=";
        for (size_t i = 0; i < weights.size(); ++i) std::cout << (i ? "," : "") << weights[i];
        bool routing_match = actual_ids == ids;
        double max_weight_error = 0.0;
        for (size_t i = 0; i < weights.size(); ++i)
            max_weight_error = (std::max)(max_weight_error, std::abs(static_cast<double>(actual_weights[i]) - weights[i]));
        std::cout << "\nrecomputed_experts=";
        for (size_t i = 0; i < actual_ids.size(); ++i) std::cout << (i ? "," : "") << actual_ids[i];
        std::cout << " max_router_weight_abs=" << max_weight_error;
        std::cout << "\nmoe_max_abs=" << max_abs << " relative_l2=" << relative_l2 << '\n';
        if (argc == 5) std::cout << "block_out_max_abs=" << block_max_abs << " relative_l2=" << block_relative_l2 << '\n';
        ggml_backend_buffer_free(buffer);
        ggml_free(ctx);
        ggml_backend_free(cpu);
        const double max_abs_gate = argc == 5 ? 1.5e-1 : 1.0e-3;
        const double relative_gate = argc == 5 ? 1.0e-2 : 1.0e-4;
        const double weight_gate = argc == 5 ? 1.0e-3 : 1.0e-5;
        const bool block_match = argc != 5 || (block_max_abs <= 1.5e-1 && block_relative_l2 <= 2.5e-3);
        const bool passed = routing_match && max_weight_error <= weight_gate && max_abs <= max_abs_gate && relative_l2 <= relative_gate && block_match;
        std::cout << (passed ? "ggml MoE/block parity PASS\n" : "ggml MoE/block parity FAIL\n");
        return passed ? 0 : 1;
    } catch (const std::exception& e) {
        std::cerr << "MoE parity failed: " << e.what() << '\n';
        return 1;
    }
}
