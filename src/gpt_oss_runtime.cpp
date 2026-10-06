#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

struct Options {
    std::string model;
    std::string prompt;
    int32_t predict = 32;
    int32_t gpu_layers = 12;
    uint32_t context = 512;
    uint32_t ubatch = 512;
    bool custom_tensor_split = false;
    float tensor_split[2] = {0.0f, 0.0f};
    int32_t dump_block = -1;
    std::string dump_prefix;
};

struct BlockDump {
    int32_t layer = -1;
    std::string prefix;
    bool written[13] = {};
    uint32_t q_rope_stage = 0;
    uint32_t k_rope_stage = 0;
};

struct CaptureTarget {
    const char * graph_name;
    const char * file_name;
    int64_t ne0;
    int64_t ne1;
    int64_t ne2;
};

constexpr CaptureTarget kBlockTargets[] = {
    {"attn_norm", "attn_norm", 2880, 20, 1},
    {"Qcur", "Qcur_pre_rope", 64, 64, 20},
    {"Qcur", "Qcur_rope", 64, 64, 20},
    {"Kcur", "Kcur_pre_rope", 64, 8, 20},
    {"Kcur", "Kcur_rope", 64, 8, 20},
    {"Vcur", "Vcur", 64, 8, 20},
    {"attn_out", "attn_out", 2880, 20, 1},
    {"ffn_inp", "ffn_inp", 2880, 20, 1},
    {"attn_post_norm", "attn_post_norm", 2880, 20, 1},
    {"ffn_moe_topk", "ffn_moe_topk", 4, 20, 1},
    {"ffn_moe_weights_softmax", "ffn_moe_weights_softmax", 1, 4, 20},
    {"ffn_moe_out", "ffn_moe_out", 2880, 20, 1},
    {"l_out", "l_out", 2880, 20, 1},
};

bool capture_block_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & dump = *static_cast<BlockDump *>(user_data);
    const std::string expected_suffix = "-" + std::to_string(dump.layer);
    const std::string name = tensor->name;
    for (size_t i = 0; i < std::size(kBlockTargets); ++i) {
        const auto & target = kBlockTargets[i];
        if (dump.written[i] || name != std::string(target.graph_name) + expected_suffix ||
            tensor->ne[0] != target.ne0 || tensor->ne[1] != target.ne1 || tensor->ne[2] != target.ne2) continue;
        if (ask) return true;

        size_t target_index = i;
        std::string file_name = target.file_name;
        if (i == 1 || i == 2) {
            const uint32_t stage = dump.q_rope_stage++;
            if (stage > 1) return true;
            target_index = 1 + stage;
        } else if (i == 3 || i == 4) {
            const uint32_t stage = dump.k_rope_stage++;
            if (stage > 1) return true;
            target_index = 3 + stage;
        }
        const std::string path = dump.prefix + "." + file_name + "-" + std::to_string(dump.layer) + ".bin";
        std::ofstream out(path, std::ios::binary | std::ios::trunc);
        if (!out) {
            std::cerr << "failed to write block checkpoint: " << path << '\n';
            return false;
        }
        const size_t bytes = ggml_nbytes(tensor);
        std::vector<uint8_t> data(bytes);
        ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
        // Store dimensions and strides because top-k results may be views into a
        // wider argsort result and are not necessarily tightly packed.
        const uint64_t header[] = {0x31544253, static_cast<uint64_t>(tensor->type),
            static_cast<uint64_t>(bytes),
            static_cast<uint64_t>(tensor->ne[0]), static_cast<uint64_t>(tensor->ne[1]),
            static_cast<uint64_t>(tensor->ne[2]), static_cast<uint64_t>(tensor->ne[3]),
            static_cast<uint64_t>(tensor->nb[0]), static_cast<uint64_t>(tensor->nb[1]),
            static_cast<uint64_t>(tensor->nb[2]), static_cast<uint64_t>(tensor->nb[3])};
        out.write(reinterpret_cast<const char *>(header), sizeof(header));
        out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(bytes));
        if (!out) {
            std::cerr << "failed while writing block checkpoint: " << path << '\n';
            return false;
        }
        dump.written[target_index] = true;
        std::cerr << "captured " << name << " type=" << ggml_type_name(tensor->type)
                  << " bytes=" << bytes << " -> " << path << '\n';
        return true;
    }
    return false;
}

bool parse_i32(std::string_view s, int32_t & value) {
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}

bool parse_u32(std::string_view s, uint32_t & value) {
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size();
}

bool parse_f32(std::string_view s, float & value) {
    const auto result = std::from_chars(s.data(), s.data() + s.size(), value);
    return result.ec == std::errc{} && result.ptr == s.data() + s.size() && std::isfinite(value);
}

void usage(const char * exe) {
    std::cerr << "Usage: " << exe
              << " --model FILE --prompt TEXT [--tokens N] [--gpu-layers N] [--context N]"
              << " [--ubatch N] [--tensor-split GPU0,GPU1] [--dump-block LAYER --dump-prefix PATH]\n";
}

bool parse_options(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (i + 1 >= argc) return false;
        const std::string_view value(argv[++i]);
        if (key == "--model") o.model.assign(value);
        else if (key == "--prompt") o.prompt.assign(value);
        else if (key == "--tokens") {
            if (!parse_i32(value, o.predict) || o.predict < 1 || o.predict > 4096) return false;
        } else if (key == "--gpu-layers") {
            if (!parse_i32(value, o.gpu_layers) || o.gpu_layers < 0) return false;
        } else if (key == "--context") {
            if (!parse_u32(value, o.context) || o.context < 16) return false;
        } else if (key == "--ubatch") {
            if (!parse_u32(value, o.ubatch) || o.ubatch < 1) return false;
        } else if (key == "--tensor-split") {
            const size_t comma = value.find(',');
            if (comma == std::string_view::npos ||
                !parse_f32(value.substr(0, comma), o.tensor_split[0]) ||
                !parse_f32(value.substr(comma + 1), o.tensor_split[1]) ||
                o.tensor_split[0] <= 0.0f || o.tensor_split[1] <= 0.0f) return false;
            o.custom_tensor_split = true;
        } else if (key == "--dump-block") {
            if (!parse_i32(value, o.dump_block) || o.dump_block < 0 || o.dump_block >= 36) return false;
        } else if (key == "--dump-prefix") {
            o.dump_prefix.assign(value);
            if (o.dump_prefix.empty()) return false;
        } else {
            return false;
        }
    }
    return !o.model.empty() && !o.prompt.empty() &&
        ((o.dump_block < 0 && o.dump_prefix.empty()) || (o.dump_block >= 0 && !o.dump_prefix.empty()));
}

} // namespace

int main(int argc, char ** argv) {
    Options options;
    if (!parse_options(argc, argv, options)) {
        usage(argv[0]);
        return 2;
    }

    // The graph, tokenizer, MXFP4 implementation, and backend dispatch all come from
    // the pinned llama.cpp runtime. This runner deliberately contains no model math.
    ggml_backend_load_all();
    llama_backend_init();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = options.gpu_layers;
    model_params.split_mode = LLAMA_SPLIT_MODE_LAYER;
    std::vector<float> tensor_split;
    if (options.custom_tensor_split) {
        if (llama_max_devices() < 2) {
            std::cerr << "--tensor-split requires at least two backend devices\n";
            llama_backend_free();
            return 2;
        }
        tensor_split.resize(llama_max_devices(), 0.0f);
        tensor_split[0] = options.tensor_split[0];
        tensor_split[1] = options.tensor_split[1];
        model_params.tensor_split = tensor_split.data();
    }
    llama_model * model = llama_model_load_from_file(options.model.c_str(), model_params);
    if (!model) {
        std::cerr << "failed to load GPT-OSS model: " << options.model << '\n';
        llama_backend_free();
        return 1;
    }

    const char * chat_template = llama_model_chat_template(model, nullptr);
    if (!chat_template) {
        std::cerr << "model does not provide a chat template\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    const llama_chat_message user_message{"user", options.prompt.c_str()};
    const int32_t formatted_size = llama_chat_apply_template(
        chat_template, &user_message, 1, true, nullptr, 0);
    if (formatted_size <= 0) {
        std::cerr << "llama.cpp could not apply the model chat template\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    std::vector<char> formatted_prompt(static_cast<size_t>(formatted_size) + 1);
    const int32_t formatted_bytes = llama_chat_apply_template(
        chat_template, &user_message, 1, true, formatted_prompt.data(),
        static_cast<int32_t>(formatted_prompt.size()));
    if (formatted_bytes != formatted_size) {
        std::cerr << "llama.cpp returned an inconsistent formatted prompt length\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t needed = -llama_tokenize(vocab, formatted_prompt.data(),
        formatted_bytes, nullptr, 0, true, true);
    if (needed <= 0) {
        std::cerr << "failed to size prompt tokenization\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(needed));
    const int32_t tokenized = llama_tokenize(vocab, formatted_prompt.data(),
        formatted_bytes, tokens.data(), needed, true, true);
    if (tokenized < 0) {
        std::cerr << "failed to tokenize prompt\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
    tokens.resize(static_cast<size_t>(tokenized));
    if (options.dump_block >= 0) {
        const std::string token_path = options.dump_prefix + ".tokens.txt";
        std::ofstream token_file(token_path, std::ios::trunc);
        if (!token_file) {
            std::cerr << "failed to write prompt token capture: " << token_path << '\n';
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        for (size_t i = 0; i < tokens.size(); ++i) token_file << (i ? " " : "") << tokens[i];
        token_file << '\n';
        if (!token_file) {
            std::cerr << "failed while writing prompt token capture: " << token_path << '\n';
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
    }

    const uint64_t required_context = static_cast<uint64_t>(tokens.size()) + options.predict;
    if (required_context > std::numeric_limits<uint32_t>::max()) {
        std::cerr << "prompt plus requested generation is too large\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_context_params context_params = llama_context_default_params();
    context_params.n_ctx = std::max(options.context, static_cast<uint32_t>(required_context));
    context_params.n_batch = std::min<uint32_t>(2048, context_params.n_ctx);
    context_params.n_ubatch = std::min(options.ubatch, context_params.n_batch);
    context_params.n_seq_max = 1;
    context_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    context_params.no_perf = false;
    BlockDump block_dump;
    if (options.dump_block >= 0) {
        block_dump.layer = options.dump_block;
        block_dump.prefix = options.dump_prefix;
        context_params.cb_eval = capture_block_tensor;
        context_params.cb_eval_user_data = &block_dump;
    }
    llama_context * context = llama_init_from_model(model, context_params);
    if (!context) {
        std::cerr << "failed to create llama context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    llama_sampler_chain_params sampler_params = llama_sampler_chain_default_params();
    sampler_params.no_perf = true;
    llama_sampler * sampler = llama_sampler_chain_init(sampler_params);
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());

    std::cerr << "prompt_token_ids=";
    for (size_t i = 0; i < tokens.size(); ++i) {
        if (i) std::cerr << ',';
        std::cerr << tokens[i];
    }
    std::cerr << '\n';

    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    llama_token next_token = LLAMA_TOKEN_NULL;
    int32_t generated = 0;
    double prompt_eval_ms = 0.0;
    double decode_eval_ms = 0.0;
    double sampling_ms = 0.0;
    for (; generated < options.predict; ++generated) {
        const auto decode_start = std::chrono::steady_clock::now();
        const int32_t rc = llama_decode(context, batch);
        // CUDA execution can be asynchronous; synchronize before stopping the
        // per-evaluation timer so the measurement includes device work.
        llama_synchronize(context);
        const auto decode_end = std::chrono::steady_clock::now();
        const double decode_ms = std::chrono::duration<double, std::milli>(decode_end - decode_start).count();
        if (generated == 0) prompt_eval_ms += decode_ms;
        else decode_eval_ms += decode_ms;
        if (rc != 0) {
            std::cerr << "llama_decode failed with status " << rc << '\n';
            break;
        }
        const auto sample_start = std::chrono::steady_clock::now();
        next_token = llama_sampler_sample(sampler, context, -1);
        const auto sample_end = std::chrono::steady_clock::now();
        sampling_ms += std::chrono::duration<double, std::milli>(sample_end - sample_start).count();
        if (llama_vocab_is_eog(vocab, next_token)) break;

        char piece[256];
        const int32_t bytes = llama_token_to_piece(vocab, next_token, piece, sizeof(piece), 0, true);
        if (bytes < 0) {
            std::cerr << "failed to convert generated token to text\n";
            break;
        }
        std::cout.write(piece, bytes);
        std::cout.flush();
        std::cerr << "generated_token_id=" << next_token << '\n';
        batch = llama_batch_get_one(&next_token, 1);
    }
    std::cout << '\n';
    std::cerr << "generated_count=" << generated << '\n';
    std::cerr << "prompt_eval_ms=" << prompt_eval_ms << '\n';
    std::cerr << "decode_eval_ms=" << decode_eval_ms << '\n';
    std::cerr << "sampling_ms=" << sampling_ms << '\n';
    // Keep llama.cpp's own counters alongside wall-clock measurements while
    // validating against its CLI, which reports these same context statistics.
    const llama_perf_context_data perf = llama_perf_context(context);
    std::cerr << "llama_perf_prompt_ms=" << perf.t_p_eval_ms
              << " llama_perf_prompt_tokens=" << perf.n_p_eval << '\n';
    std::cerr << "llama_perf_decode_ms=" << perf.t_eval_ms
              << " llama_perf_decode_runs=" << perf.n_eval << '\n';

    llama_sampler_free(sampler);
    llama_free(context);
    llama_model_free(model);
    llama_backend_free();
    if (options.dump_block >= 0 && !std::all_of(std::begin(block_dump.written), std::end(block_dump.written),
                                                [](bool written) { return written; })) {
        std::cerr << "block capture incomplete; one or more reference tensors were not observed\n";
        return 1;
    }
    return generated == 0 ? 1 : 0;
}
