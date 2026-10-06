#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
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
    bool dump_all_layers = false;
    std::string dump_prefix;
    std::string dump_logits_path;
    bool serve = false;
    bool tokenize_only = false;
};

struct BlockDump {
    int32_t layer = -1;
    bool all_layers = false;
    std::string prefix;
    bool written[16] = {};
    bool written_layers[36] = {};
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
    {"ffn_moe_logits", "ffn_moe_logits", 128, 20, 1},
    {"ffn_moe_logits_biased", "ffn_moe_logits_biased", 128, 20, 1},
    {"ffn_moe_probs_biased", "ffn_moe_probs_biased", 128, 20, 1},
    {"ffn_moe_topk", "ffn_moe_topk", 4, 20, 1},
    {"ffn_moe_weights_softmax", "ffn_moe_weights_softmax", 1, 4, 20},
    {"ffn_moe_out", "ffn_moe_out", 2880, 20, 1},
    {"l_out", "l_out", 2880, 20, 1},
};

bool capture_block_tensor(ggml_tensor * tensor, bool ask, void * user_data) {
    auto & dump = *static_cast<BlockDump *>(user_data);
    const std::string name = tensor->name;
    if (dump.all_layers && name.rfind("l_out-", 0) == 0 &&
        tensor->ne[0] == 2880 && tensor->ne[1] == 20 && tensor->ne[2] == 1) {
        const std::string_view suffix(name.data() + 6, name.size() - 6);
        int32_t layer = -1;
        const auto result = std::from_chars(suffix.data(), suffix.data() + suffix.size(), layer);
        if (result.ec == std::errc{} && result.ptr == suffix.data() + suffix.size() && layer >= 0 && layer < 36) {
            if (dump.written_layers[layer]) return false;
            if (ask) return true;
            const std::string path = dump.prefix + ".l_out-" + std::to_string(layer) + ".bin";
            std::ofstream out(path, std::ios::binary | std::ios::trunc);
            if (!out) {
                std::cerr << "failed to write layer checkpoint: " << path << '\n';
                return false;
            }
            const size_t bytes = ggml_nbytes(tensor);
            std::vector<uint8_t> data(bytes);
            ggml_backend_tensor_get(tensor, data.data(), 0, bytes);
            const uint64_t header[] = {0x31544253, static_cast<uint64_t>(tensor->type),
                static_cast<uint64_t>(bytes), static_cast<uint64_t>(tensor->ne[0]),
                static_cast<uint64_t>(tensor->ne[1]), static_cast<uint64_t>(tensor->ne[2]),
                static_cast<uint64_t>(tensor->ne[3]), static_cast<uint64_t>(tensor->nb[0]),
                static_cast<uint64_t>(tensor->nb[1]), static_cast<uint64_t>(tensor->nb[2]),
                static_cast<uint64_t>(tensor->nb[3])};
            out.write(reinterpret_cast<const char *>(header), sizeof(header));
            out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(bytes));
            if (!out) return false;
            dump.written_layers[layer] = true;
            std::cerr << "captured " << name << " bytes=" << bytes << " -> " << path << '\n';
            return true;
        }
    }
    const std::string expected_suffix = "-" + std::to_string(dump.layer);
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
              << " [--ubatch N] [--tensor-split GPU0,GPU1] [--dump-block LAYER --dump-prefix PATH]"
              << " [--dump-all-layers --dump-prefix PATH] [--dump-logits FILE]\n";
}

bool parse_options(int argc, char ** argv, Options & o) {
    for (int i = 1; i < argc; ++i) {
        const std::string_view key(argv[i]);
        if (key == "--serve") { o.serve = true; continue; }
        if (key == "--tokenize-only") { o.tokenize_only = true; continue; }
        if (key == "--dump-all-layers") { o.dump_all_layers = true; continue; }
        // The Python server adds its native engine's layer-split hint for multi-GPU
        // configs. This adapter uses llama.cpp tensor_split instead.
        if (key == "--layer-split") { if (i + 1 >= argc) return false; ++i; continue; }
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
        } else if (key == "--dump-logits") {
            o.dump_logits_path.assign(value);
            if (o.dump_logits_path.empty()) return false;
        } else {
            return false;
        }
    }
    const bool has_dump = o.dump_block >= 0 || o.dump_all_layers;
    return !o.model.empty() && (o.serve || !o.prompt.empty()) &&
        (o.dump_block < 0 || !o.dump_all_layers) && (has_dump == !o.dump_prefix.empty());
}

struct ServeCommandQueue {
    std::mutex mutex;
    std::condition_variable ready;
    std::queue<std::string> commands;
    std::atomic<bool> stop{false};
};

int serve_model(llama_model * model, const Options & options) {
    const uint32_t n_ctx = options.context;
    llama_context_params params = llama_context_default_params();
    params.n_ctx = n_ctx;
    params.n_batch = std::min<uint32_t>(2048, n_ctx);
    params.n_ubatch = std::min(options.ubatch, params.n_batch);
    params.n_seq_max = 1;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
    params.no_perf = false;
    llama_context * context = llama_init_from_model(model, params);
    if (!context) {
        std::cout << "ERR could not create GPT-OSS context\n" << std::flush;
        return 1;
    }

    ServeCommandQueue input;
    std::thread reader([&input] {
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "STOP") input.stop.store(true);
            {
                std::lock_guard lock(input.mutex);
                input.commands.push(std::move(line));
            }
            input.ready.notify_one();
        }
        {
            std::lock_guard lock(input.mutex);
            input.commands.push("QUIT");
        }
        input.ready.notify_one();
    });
    auto next_command = [&input] {
        std::unique_lock lock(input.mutex);
        input.ready.wait(lock, [&input] { return !input.commands.empty(); });
        std::string line = std::move(input.commands.front());
        input.commands.pop();
        return line;
    };

    std::cout << "READY " << n_ctx << " stop\n" << std::flush;
    bool quit = false;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    while (!quit) {
        const std::string line = next_command();
        if (line == "QUIT") break;
        if (line == "STOP") { input.stop.store(false); continue; }
        std::istringstream fields(line);
        std::string command;
        int32_t max_new = 0;
        fields >> command >> max_new;
        if (command != "GEN" || max_new < 1 || max_new > 4096) {
            std::cout << "ERR expected GEN <max_new> [sampling keys] <token ids>\n" << std::flush;
            continue;
        }

        float temperature = 0.0f, top_p = 1.0f, min_p = 0.0f;
        float repeat_penalty = 1.0f, frequency_penalty = 0.0f, presence_penalty = 0.0f;
        int32_t top_k = 0, penalty_last_n = 0;
        uint32_t seed = LLAMA_DEFAULT_SEED;
        std::string word, token_list;
        while (fields >> word) {
            const size_t eq = word.find('=');
            if (eq == std::string::npos) { token_list = word; break; }
            const std::string key = word.substr(0, eq);
            const std::string_view val(word.data() + eq + 1, word.size() - eq - 1);
            if (key == "temperature") parse_f32(val, temperature);
            else if (key == "top_p") parse_f32(val, top_p);
            else if (key == "min_p") parse_f32(val, min_p);
            else if (key == "penalty_repeat") parse_f32(val, repeat_penalty);
            else if (key == "penalty_freq") parse_f32(val, frequency_penalty);
            else if (key == "penalty_present") parse_f32(val, presence_penalty);
            else if (key == "top_k") parse_i32(val, top_k);
            else if (key == "penalty_last_n") parse_i32(val, penalty_last_n);
            else if (key == "seed") parse_u32(val, seed);
        }
        std::vector<llama_token> prompt;
        std::istringstream token_fields(token_list);
        while (std::getline(token_fields, word, ',')) {
            int32_t token = -1;
            if (!parse_i32(word, token) || token < 0 || token >= llama_vocab_n_tokens(vocab)) {
                prompt.clear();
                break;
            }
            prompt.push_back(token);
        }
        if (prompt.empty() || prompt.size() + static_cast<size_t>(max_new) > n_ctx) {
            std::cout << "ERR prompt is empty, invalid, or exceeds the configured context\n" << std::flush;
            continue;
        }

        input.stop.store(false);
        llama_memory_clear(llama_get_memory(context), true);
        llama_sampler_chain_params chain_params = llama_sampler_chain_default_params();
        chain_params.no_perf = true;
        llama_sampler * sampler = llama_sampler_chain_init(chain_params);
        if (penalty_last_n > 0 && (repeat_penalty != 1.0f || frequency_penalty != 0.0f || presence_penalty != 0.0f))
            llama_sampler_chain_add(sampler, llama_sampler_init_penalties(llama_vocab_n_tokens(vocab), penalty_last_n,
                repeat_penalty, frequency_penalty, presence_penalty));
        if (top_k > 0) llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k));
        if (top_p < 1.0f) llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(sampler, llama_sampler_init_min_p(min_p, 1));
        if (temperature > 0.0f) {
            llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature));
            llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
        } else {
            llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
        }

        const auto prompt_start = std::chrono::steady_clock::now();
        int32_t prompt_rc = 0;
        size_t prompt_offset = 0;
        while (prompt_offset < prompt.size()) {
            const size_t remaining = prompt.size() - prompt_offset;
            const int32_t chunk_size = static_cast<int32_t>(
                std::min<size_t>(params.n_batch, remaining));
            llama_batch batch = llama_batch_get_one(prompt.data() + prompt_offset, chunk_size);
            prompt_rc = llama_decode(context, batch);
            if (prompt_rc != 0) break;
            prompt_offset += static_cast<size_t>(chunk_size);
        }
        llama_synchronize(context);
        const double prompt_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - prompt_start).count();
        if (prompt_rc != 0) {
            llama_sampler_free(sampler);
            std::cout << "ERR llama.cpp prompt decode failed with status " << prompt_rc << "\n" << std::flush;
            continue;
        }
        std::cout << "PP " << prompt.size() << ' ' << prompt.size() << ' ' << prompt_ms << ' '
                  << (prompt_ms > 0 ? prompt.size() * 1000.0 / prompt_ms : 0.0) << "\n" << std::flush;

        llama_batch batch{};
        int32_t generated = 0;
        double decode_ms = 0.0;
        std::string finish = "length";
        for (; generated < max_new; ++generated) {
            if (input.stop.load()) { finish = "cancel"; break; }
            const llama_token token = llama_sampler_sample(sampler, context, -1);
            std::cout << "T " << token << "\n" << std::flush;
            if (llama_vocab_is_eog(vocab, token)) { ++generated; finish = "stop"; break; }
            if (generated + 1 >= max_new) { ++generated; break; }
            batch = llama_batch_get_one(const_cast<llama_token *>(&token), 1);
            const auto decode_start = std::chrono::steady_clock::now();
            const int32_t rc = llama_decode(context, batch);
            llama_synchronize(context);
            decode_ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - decode_start).count();
            if (rc != 0) { finish = "cancel"; break; }
        }
        llama_sampler_free(sampler);
        std::cout << "DONE " << generated << ' ' << prompt.size() << ' ' << prompt_ms << ' '
                  << decode_ms << ' ' << finish << " 0 0 0\n" << std::flush;
    }
    {
        std::lock_guard lock(input.mutex);
        input.commands.push("QUIT");
    }
    input.ready.notify_one();
    if (reader.joinable()) reader.join();
    llama_free(context);
    return 0;
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
    model_params.vocab_only = options.tokenize_only;
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

    if (options.tokenize_only) {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t needed = -llama_tokenize(vocab, options.prompt.data(),
            static_cast<int32_t>(options.prompt.size()), nullptr, 0, true, true);
        if (needed <= 0) {
            std::cerr << "failed to size prompt tokenization\n";
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        std::vector<llama_token> tokens(static_cast<size_t>(needed));
        const int32_t count = llama_tokenize(vocab, options.prompt.data(),
            static_cast<int32_t>(options.prompt.size()), tokens.data(), needed, true, true);
        if (count < 0) {
            std::cerr << "failed to tokenize prompt\n";
            llama_model_free(model);
            llama_backend_free();
            return 1;
        }
        for (int32_t i = 0; i < count; ++i) std::cout << (i ? " " : "") << tokens[i];
        std::cout << '\n';
        llama_model_free(model);
        llama_backend_free();
        return 0;
    }

    if (options.serve) {
        const int result = serve_model(model, options);
        llama_model_free(model);
        llama_backend_free();
        return result;
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
    if (options.dump_block >= 0 || options.dump_all_layers) {
        block_dump.layer = options.dump_block;
        block_dump.all_layers = options.dump_all_layers;
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

    llama_batch batch{};
    llama_token next_token = LLAMA_TOKEN_NULL;
    int32_t generated = 0;
    double prompt_eval_ms = 0.0;
    double decode_eval_ms = 0.0;
    double sampling_ms = 0.0;
    for (; generated < options.predict; ++generated) {
        const auto decode_start = std::chrono::steady_clock::now();
        int32_t rc = 0;
        if (generated == 0) {
            size_t prompt_offset = 0;
            while (prompt_offset < tokens.size()) {
                const size_t remaining = tokens.size() - prompt_offset;
                const int32_t chunk_size = static_cast<int32_t>(
                    std::min<size_t>(context_params.n_batch, remaining));
                batch = llama_batch_get_one(tokens.data() + prompt_offset, chunk_size);
                rc = llama_decode(context, batch);
                if (rc != 0) break;
                prompt_offset += static_cast<size_t>(chunk_size);
            }
        } else {
            batch = llama_batch_get_one(&next_token, 1);
            rc = llama_decode(context, batch);
        }
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
        if (!options.dump_logits_path.empty()) {
            const int32_t n_logits = llama_vocab_n_tokens(vocab);
            const int32_t logits_index = generated == 0 ? static_cast<int32_t>(tokens.size() - 1) : -1;
            const float * logits = llama_get_logits_ith(context, logits_index);
            const auto mode = std::ios::binary | (generated == 0 ? std::ios::trunc : std::ios::app);
            std::ofstream output(options.dump_logits_path, mode);
            if (!logits || !output) {
                std::cerr << "failed to open logits or output file: " << options.dump_logits_path << '\n';
                break;
            }
            output.write(reinterpret_cast<const char *>(logits),
                         static_cast<std::streamsize>(n_logits) * sizeof(float));
            if (!output) {
                std::cerr << "failed while writing logits: " << options.dump_logits_path << '\n';
                break;
            }
            std::cerr << "dumped_logits=" << n_logits << " position=" << generated
                      << " path=" << options.dump_logits_path << '\n';
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
