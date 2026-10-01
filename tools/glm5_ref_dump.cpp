// tools/glm5_ref_dump.cpp - runs the FORK's own GLM5-Next graph on a fixed prompt and writes the named
// tensors, so this port can be checked layer by layer instead of only by its final tokens.
//
// Why this exists: phase 8 wires 46 blocks together, and "the logits differ" localizes nothing.  The
// reference names every intermediate it builds (`cb(...)` -> "l_out-12", "hc_head", "h_nextn",
// "result_output", and per-site ones like "attn_out-4"/"kda_gate-4"/"ffn_moe_out-4"), so capturing them
// turns a single opaque mismatch into "the first layer whose hidden state diverges".
//
// Build (against the fork, which is the reference build):
//   g++ -O2 -std=c++17 -I/home/peb/llama.cpp-glm5/include -I/home/peb/llama.cpp-glm5/ggml/include \
//       tools/glm5_ref_dump.cpp -o build-volta/glm5_ref_dump \
//       -L/home/peb/llama.cpp-glm5/build-glm5/bin -lllama -lggml -lggml-base -Wl,-rpath,/home/peb/llama.cpp-glm5/build-glm5/bin
//
// Usage:
//   glm5_ref_dump <model.gguf> <outdir> "<prompt>" [n_predict] [n_cpu_moe] [--all]
//
// Writes, into outdir: dump.tsv (a manifest of name/shape/type/file) and one .bin per tensor (fp32,
// row-major, ne0 fastest - the same layout the port's kernels use), plus tokens.txt with the greedy ids.
#include "llama.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <list>
#include <set>
#include <string>
#include <vector>
#include <fstream>
#include <algorithm>

namespace {

struct Dumper {
    std::string dir;
    bool all = false;
    bool dumping = false;
    std::ofstream manifest;
    int dumped = 0;
    std::set<std::string> seen;              // names already written
    std::vector<std::string> dupes;          // names that appeared more than once

    bool wanted(const std::string & n) const {
        if (all) return true;
        static const char * keys[] = {"l_out", "hc_head", "h_nextn", "result_norm", "result_output",
                                      "attn_out", "kda_gate", "kda_beta", "ffn_out", "inp_embd", "hc_init",
                                      "hc_attn", "hc_ffn", "attn_norm", "ffn_norm", "ffn_inp",
                                      "Qcur", "kv_cmpr", "ffn_moe_out", "mtp_"};
        for (const char * k : keys)
            if (n.find(k) != std::string::npos) return true;
        return false;
    }
};

Dumper g_d;

/// Write one tensor as fp32 with a tiny header (ndim, ne[4], type), naively converted.
void dump_tensor(ggml_tensor * t) {
    const std::string name(t->name);
    std::string safe = name;
    std::replace(safe.begin(), safe.end(), '/', '_');
    const std::string path = g_d.dir + "/" + safe + ".bin";

    // Provenance, recorded for EVERY tensor including the ones skipped below: which backend buffer it
    // lives in, whether it is a view of something else, whether it is contiguous, and its strides.
    // Without this a "wrong values" dump cannot be told from a "wrong tensor" dump.
    char prov[256];
    const char * bufn = t->buffer ? ggml_backend_buffer_name(t->buffer) : "(none)";
    std::snprintf(prov, sizeof(prov), "view=%s contig=%s nb=[%zu,%zu,%zu,%zu] buf=%s",
                  t->view_src ? "Y" : "N", ggml_is_contiguous(t) ? "Y" : "N",
                  (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], bufn);

    // A non-contiguous tensor is a VIEW (a broadcast, a permute, a reshape of something else).  Reading
    // its buffer yields the underlying bytes, not the values the graph sees through it - which silently
    // produced a "hc_init" whose four broadcast streams differed from each other.  Skip, and say so.
    if (!ggml_is_contiguous(t)) {
        g_d.manifest << name << "\t" << t->ne[0] << "\t" << t->ne[1] << "\t" << t->ne[2] << "\t" << t->ne[3] << "\t" << prov
                     << "\t" << ggml_type_name(t->type) << "\tSKIPPED-NON-CONTIGUOUS-VIEW\n";
        return;
    }
    // Two tensors can share a name in this graph (the graph's inputs-embeds placeholder and the real
    // embedding lookup are both "inp_embd"), in which case one file overwrites the other.  Record which.
    if (g_d.seen.count(name)) {
        g_d.manifest << name << "\t" << t->ne[0] << "\t" << t->ne[1] << "\t" << t->ne[2] << "\t" << t->ne[3] << "\t" << prov
                     << "\t" << ggml_type_name(t->type) << "\tDUPLICATE-NAME (already written)\n";
        g_d.dupes.push_back(name);
        return;
    }
    g_d.seen.insert(name);

    const int64_t n = ggml_nelements(t);
    std::vector<float> f((size_t) n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, f.data(), 0, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> h((size_t) n);
        ggml_backend_tensor_get(t, h.data(), 0, (size_t) n * sizeof(ggml_fp16_t));
        for (int64_t i = 0; i < n; ++i) f[(size_t) i] = ggml_fp16_to_fp32(h[(size_t) i]);
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> v((size_t) n);
        ggml_backend_tensor_get(t, v.data(), 0, (size_t) n * sizeof(int32_t));
        for (int64_t i = 0; i < n; ++i) f[(size_t) i] = (float) v[(size_t) i];
    } else {
        return;   // quantized intermediates are not interesting here
    }

    std::ofstream o(path, std::ios::binary);
    const int32_t hdr[5] = {(int32_t) t->type, (int32_t) t->ne[0], (int32_t) t->ne[1], (int32_t) t->ne[2],
                            (int32_t) t->ne[3]};
    o.write((const char *) hdr, sizeof(hdr));
    o.write((const char *) f.data(), (std::streamsize) (f.size() * sizeof(float)));
    o.close();

    g_d.manifest << name << "\t" << t->ne[0] << "\t" << t->ne[1] << "\t" << t->ne[2] << "\t" << t->ne[3] << "\t" << prov
                 << "\t" << ggml_type_name(t->type) << "\t" << safe << ".bin\n";
    g_d.dumped++;
}

bool cb_eval(ggml_tensor * t, bool ask, void * /*ud*/) {
    if (ask) return true;                    // evaluate everything: skipping a tensor would break the graph
    if (!g_d.dumping || !t->name || !t->name[0]) return true;
    if (!g_d.wanted(std::string(t->name))) return true;
    dump_tensor(t);
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: glm5_ref_dump <model.gguf> <outdir> \"<prompt>\" [n_predict] "
                             "[n_cpu_moe] [--all]\n");
        return 2;
    }
    const char * model_path = argv[1];
    g_d.dir = argv[2];
    const char * prompt = argv[3];
    const int n_predict = argc > 4 ? std::atoi(argv[4]) : 4;
    const int n_cpu_moe = argc > 5 ? std::atoi(argv[5]) : 46;
    for (int i = 4; i < argc; ++i)
        if (std::strcmp(argv[i], "--all") == 0) g_d.all = true;

    g_d.manifest.open(g_d.dir + "/dump.tsv", std::ios::out | std::ios::trunc);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 99;
    mparams.load_mode = LLAMA_LOAD_MODE_MMAP;
    // Keep the first `n_cpu_moe` layers' expert weights on the CPU, exactly as the CLI's -ncmoe does:
    // it appends one tensor-buft override per layer (pattern "blk\.N\.ffn_(up|down|gate|gate_up)_(ch|)exps")
    // with the CPU buffer type, NULL-terminated.  86 GB of experts cannot live in 32 GB of VRAM.
    static std::list<std::string> buft_strings;
    std::vector<llama_model_tensor_buft_override> overrides;
    for (int i = 0; i < n_cpu_moe; ++i) {
        buft_strings.push_back("blk\\." + std::to_string(i) + "\\.ffn_(up|down|gate|gate_up)_(ch|)exps");
        overrides.push_back({buft_strings.back().c_str(), ggml_backend_cpu_buffer_type()});
    }
    overrides.push_back({nullptr, nullptr});
    mparams.tensor_buft_overrides = overrides.data();

    std::printf("loading %s (cpu moe layers %d)...\n", model_path, n_cpu_moe);
    std::fflush(stdout);
    llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) { std::fprintf(stderr, "model load failed\n"); return 1; }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 1024;
    cparams.n_batch = 1024;
    cparams.cb_eval = cb_eval;
    cparams.cb_eval_user_data = nullptr;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) { std::fprintf(stderr, "context create failed\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    std::vector<llama_token> tokens((size_t) std::strlen(prompt) + 8);
    const int n_tok = llama_tokenize(vocab, prompt, (int32_t) std::strlen(prompt), tokens.data(),
                                     (int32_t) tokens.size(), true, true);
    if (n_tok < 0) { std::fprintf(stderr, "tokenize needs %d\n", -n_tok); return 1; }
    tokens.resize((size_t) n_tok);
    std::printf("prompt: \"%s\" -> %zu tokens\n", prompt, tokens.size());
    for (llama_token t : tokens) std::printf("  %d '%s'\n", t, llama_vocab_get_text(vocab, t));

    llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    std::ofstream tokfile(g_d.dir + "/tokens.txt");
    tokfile << "prompt";
    for (llama_token t : tokens) tokfile << " " << t;
    tokfile << "\ngeneration";
    std::printf("generation:");

    // one batch carrying the prompt; the dump happens during this first decode
    g_d.dumping = true;
    const bool logits_last = true;
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    g_d.dumping = false;                       // the per-layer dump is for the prompt step
    std::printf(" dumped %d tensors", g_d.dumped);

    std::vector<llama_token> generated;
    for (int i = 0; i < n_predict; ++i) {
        llama_token id = llama_sampler_sample(smpl, ctx, -1);
        generated.push_back(id);
        tokfile << " " << id;
        std::printf(" %d", id);
        std::fflush(stdout);
        llama_batch b1 = llama_batch_get_one(&id, 1);
        if (llama_decode(ctx, b1) != 0) break;
        (void) logits_last;
    }
    tokfile << "\ndone " << generated.size() << " tokens\n";
    std::printf("\nwrote %s/dump.tsv (%d tensors) and %s/tokens.txt\n", g_d.dir.c_str(), g_d.dumped,
                g_d.dir.c_str());
    if (!g_d.dupes.empty()) {
        std::printf("note: %zu tensor name(s) appeared more than once; the first was kept:\n",
                    g_d.dupes.size());
        for (size_t i = 0; i < g_d.dupes.size() && i < 12; ++i) std::printf("  %s\n", g_d.dupes[i].c_str());
    }

    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
