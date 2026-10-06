#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "strata/artifact/gguf_reader.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int64_t kWidth = 2880;
constexpr uint32_t kExperts = 128;
constexpr double kMaxAbsTolerance = 1.0e-4;
constexpr double kRelativeL2Tolerance = 1.0e-5;

struct BackendResult {
    std::vector<float> output;
    std::string error;
};

BackendResult run_matmul(ggml_backend_t backend, const uint8_t* weights, size_t weight_bytes,
                        const std::vector<float>& input) {
    BackendResult result;
    ggml_init_params params{};
    params.mem_size = 16 * 1024 * 1024;
    params.no_alloc = true;
    ggml_context* ctx = ggml_init(params);
    if (!ctx) {
        result.error = "ggml_init failed";
        return result;
    }

    ggml_tensor* w = ggml_new_tensor_2d(ctx, GGML_TYPE_MXFP4, kWidth, kWidth);
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, kWidth, 1);
    ggml_tensor* y = ggml_mul_mat(ctx, w, x);
    ggml_cgraph* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, y);

    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    if (!buffer) {
        result.error = "backend tensor allocation failed";
        ggml_free(ctx);
        return result;
    }

    ggml_backend_tensor_set(w, weights, 0, weight_bytes);
    ggml_backend_tensor_set(x, input.data(), 0, input.size() * sizeof(float));
    const ggml_status status = ggml_backend_graph_compute(backend, graph);
    ggml_backend_synchronize(backend);
    if (status != GGML_STATUS_SUCCESS) {
        result.error = "ggml backend graph compute failed with status " + std::to_string(status);
    } else {
        result.output.resize(static_cast<size_t>(kWidth));
        ggml_backend_tensor_get(y, result.output.data(), 0, result.output.size() * sizeof(float));
    }
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
    return result;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " MODEL.gguf [LAYER=0] [EXPERT=0]\n";
        return 2;
    }
    uint32_t layer = 0;
    uint32_t expert = 0;
    try {
        if (argc > 2) layer = static_cast<uint32_t>(std::stoul(argv[2]));
        if (argc > 3) expert = static_cast<uint32_t>(std::stoul(argv[3]));
    } catch (...) {
        std::cerr << "layer and expert must be unsigned integers\n";
        return 2;
    }
    if (layer >= 36 || expert >= kExperts) {
        std::cerr << "layer must be 0..35 and expert must be 0..127\n";
        return 2;
    }

    try {
        strata::GgufFile model(argv[1]);
        std::string error = strata::check_gpt_oss_120b_architecture(model);
        if (error.empty()) error = strata::check_gpt_oss_120b_tensors(model);
        if (!error.empty()) {
            std::cerr << "model contract failed: " << error << '\n';
            return 1;
        }

        const std::string tensor_name = "blk." + std::to_string(layer) + ".ffn_gate_exps.weight";
        const strata::TensorInfo* tensor = model.find(tensor_name);
        const size_t expert_bytes = static_cast<size_t>(strata::tensor_payload_bytes(*tensor) / kExperts);
        const uint8_t* expert_weights = model.tensor_data(*tensor) + expert_bytes * expert;
        std::vector<float> input(static_cast<size_t>(kWidth));
        for (size_t i = 0; i < input.size(); ++i) {
            input[i] = static_cast<float>(static_cast<int>(i % 31) - 15) / 17.0f;
        }

        ggml_backend_load_all();
        ggml_backend_dev_t cpu_device = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
        ggml_backend_dev_t cuda_device = ggml_backend_dev_by_name("CUDA0");
        if (!cpu_device || !cuda_device) {
            std::cerr << "CPU and CUDA0 ggml devices are required\n";
            return 1;
        }
        ggml_backend_t cpu = ggml_backend_dev_init(cpu_device, nullptr);
        ggml_backend_t cuda = ggml_backend_dev_init(cuda_device, nullptr);
        if (!cpu || !cuda) {
            std::cerr << "failed to initialize CPU/CUDA0 ggml backends\n";
            if (cpu) ggml_backend_free(cpu);
            if (cuda) ggml_backend_free(cuda);
            return 1;
        }

        BackendResult cpu_result = run_matmul(cpu, expert_weights, expert_bytes, input);
        BackendResult cuda_result = run_matmul(cuda, expert_weights, expert_bytes, input);
        ggml_backend_free(cuda);
        ggml_backend_free(cpu);
        if (!cpu_result.error.empty() || !cuda_result.error.empty()) {
            std::cerr << "CPU: " << cpu_result.error << " CUDA0: " << cuda_result.error << '\n';
            return 1;
        }

        double squared_error = 0.0;
        double squared_reference = 0.0;
        double max_abs = 0.0;
        for (size_t i = 0; i < cpu_result.output.size(); ++i) {
            const double reference = cpu_result.output[i];
            const double delta = static_cast<double>(cuda_result.output[i]) - reference;
            squared_error += delta * delta;
            squared_reference += reference * reference;
            max_abs = (std::max)(max_abs, std::abs(delta));
        }
        const double relative_l2 = std::sqrt(squared_error / (std::max)(squared_reference, 1.0e-30));
        std::cout << "tensor=" << tensor_name << " expert=" << expert
                  << " matrix=" << kWidth << 'x' << kWidth << " activation=F32[" << kWidth << "]\n"
                  << "cpu_cuda_max_abs=" << max_abs << " relative_l2=" << relative_l2 << '\n'
                  << "tolerance_max_abs=" << kMaxAbsTolerance
                  << " tolerance_relative_l2=" << kRelativeL2Tolerance << '\n';
        if (max_abs > kMaxAbsTolerance || relative_l2 > kRelativeL2Tolerance) {
            std::cerr << "MXFP4 CPU/CUDA parity FAILED\n";
            return 1;
        }
        std::cout << "MXFP4 CPU/CUDA parity PASS\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "parity harness failed: " << e.what() << '\n';
        return 1;
    }
}
