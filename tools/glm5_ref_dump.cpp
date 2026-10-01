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
    bool embd_test = false;                  // feed a known embedding pattern instead of tokens
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

/// The host buffer every dumped tensor is copied into.  One context and one buffer, reused: the largest
/// tensor in this graph is result_output at 154880 fp32 = 620 KB.

static std::vector<uint8_t> g_host_out;
static ggml_backend_t       g_cpu = nullptr;

static void host_dump_init() {
    g_cpu = ggml_backend_cpu_init();
    g_host_out.reserve(8u << 20);
    if (!g_cpu)
        std::fprintf(stderr, "warning: no CPU backend; every dump will be SKIPPED-NO-HOST-COPY\n");
}

/// A HOST copy of `t`'s bytes, taken through the backend copy path.  Null if that is not possible.
///
/// The context's tensors are allocated WITH THE CPU BACKEND (ggml_backend_alloc_ctx_tensors), which is
/// ggml's own pattern - attaching a buffer by hand produced a tensor ggml_backend_tensor_copy could not
/// write to (it aborted in ggml_backend_tensor_set).
static const void * host_copy_of(ggml_tensor * t) {
    if (!g_cpu) return nullptr;
    const size_t nb = ggml_nbytes(t);
    if (nb > (8u << 20)) return nullptr;
    ggml_init_params p = { ggml_tensor_overhead() * 4 + nb + 65536, nullptr, true };
    ggml_context * c = ggml_init(p);
    if (!c) return nullptr;
    ggml_tensor * h = ggml_new_tensor(c, t->type, ggml_n_dims(t), t->ne);
    ggml_backend_buffer_t b = h ? ggml_backend_alloc_ctx_tensors(c, g_cpu) : nullptr;
    if (!h || !b) { if (b) ggml_backend_buffer_free(b); ggml_free(c); return nullptr; }
    ggml_backend_tensor_copy(t, h);
    g_host_out.assign((const uint8_t *) h->data, (const uint8_t *) h->data + nb);
    ggml_backend_buffer_free(b);
    ggml_free(c);
    return g_host_out.empty() ? nullptr : g_host_out.data();
}

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
    // The offset of the tensor inside its buffer, and the view offset, because a read that ignores one
    // of them returns a DIFFERENT tensor's data - plausible values, deterministic, and wrong.
    const char * base = t->buffer ? (const char *) ggml_backend_buffer_get_base(t->buffer) : nullptr;
    const long long off = (base && t->data) ? (long long) ((const char *) t->data - base) : -1;
    std::snprintf(prov, sizeof(prov), "view=%s contig=%s nb=[%zu,%zu,%zu,%zu] buf=%s off=%lld voff=%zu",
                  t->view_src ? "Y" : "N", ggml_is_contiguous(t) ? "Y" : "N",
                  (size_t) t->nb[0], (size_t) t->nb[1], (size_t) t->nb[2], (size_t) t->nb[3], bufn, off,
                  (size_t) t->view_offs);

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

    // READ THROUGH THE COPY PATH, NOT ggml_backend_tensor_get.  For a device tensor that sits at a
    // nonzero offset inside its buffer, tensor_get returns a DIFFERENT region of the same buffer -
    // plausible values, byte-identical run to run, and wrong.  Measured on this model: a host tensor and
    // a device tensor at offset 0 read correctly, while hc_init (offset 441088) did not, which is what
    // produced a "broadcast" whose four copies differed.  ggml_backend_tensor_copy is the path llama.cpp
    // itself uses to bring logits back, so it is the one to trust.
    const void * src = host_copy_of(t);
    if (!src) {
        g_d.manifest << name << "\t" << t->ne[0] << "\t" << t->ne[1] << "\t" << t->ne[2] << "\t" << t->ne[3] << "\t" << prov
                     << "\t" << ggml_type_name(t->type) << "\tSKIPPED-NO-HOST-COPY\n";
        return;
    }

    if (t->type == GGML_TYPE_F32) {
        std::memcpy(f.data(), src, (size_t) n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t * h = (const ggml_fp16_t *) src;
        for (int64_t i = 0; i < n; ++i) f[(size_t) i] = ggml_fp16_to_fp32(h[(size_t) i]);
    } else if (t->type == GGML_TYPE_I32) {
        const int32_t * v = (const int32_t *) src;
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
    for (int i = 4; i < argc; ++i) {
        if (std::strcmp(argv[i], "--all") == 0) g_d.all = true;
        if (std::strcmp(argv[i], "--embd-test") == 0) g_d.embd_test = true;
    }

    g_d.manifest.open(g_d.dir + "/dump.tsv", std::ios::out | std::ios::trunc);

    llama_backend_init();
    host_dump_init();          // the CPU buffer every dump is copied into; needs the backend registry up

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
    if (g_d.embd_test) {
        // THE READ-PATH TEST.  Feed the graph EMBEDDINGS instead of tokens: in that path the graph's
        // placeholder IS the input (build_inp_embd selects inps[1] == inp->embd), so its values are
        // known exactly, and every downstream copy - hc_init's four streams above all - can be
        // checked against them.  "Is the dump's read path right?" stops being an inference about ggml
        // and becomes a measurement.
        const int64_t ne_inp = llama_model_n_embd(llama_get_model(ctx));
        const int n_tok = (int) tokens.size();
        std::vector<float> pat((size_t) ne_inp * n_tok);
        // distinct per channel AND per token, so an axis mixup, a transposed read or a stale copy
        // cannot look right
        for (int t = 0; t < n_tok; ++t)
            for (int64_t e = 0; e < ne_inp; ++e)
                pat[(size_t) t * ne_inp + e] = 0.001f * (float) (e + 1) + 0.5f * (float) t;
        std::vector<llama_pos> pos(n_tok);
        std::vector<int32_t> nseq(n_tok, 1);
        std::vector<llama_seq_id *> seq(n_tok);
        std::vector<int8_t> logits(n_tok, 0);
        llama_seq_id s0 = 0;
        for (int t = 0; t < n_tok; ++t) { pos[t] = t; seq[t] = &s0; }
        llama_batch b = {};
        b.n_tokens = n_tok;
        b.embd = pat.data();
        b.pos = pos.data();
        b.n_seq_id = nseq.data();
        b.seq_id = seq.data();
        b.logits = logits.data();
        if (llama_decode(ctx, b) != 0) { std::fprintf(stderr, "embd decode failed\n"); return 1; }
        // write the pattern in the dump's own layout ([ne_inp][n_tok], ne0 fastest) for the compare
        std::FILE * pf = std::fopen((g_d.dir + "/EMBD-PATTERN.bin").c_str(), "wb");
        if (pf) {
            const int32_t hdr[5] = { 0, (int32_t) ne_inp, n_tok, 1, 1 };
            std::fwrite(hdr, sizeof(int32_t), 5, pf);
            for (int t = 0; t < n_tok; ++t)
                std::fwrite(&pat[(size_t) t * ne_inp], sizeof(float), (size_t) ne_inp, pf);
            std::fclose(pf);
            std::printf("embd-test: wrote a known pattern of %d x %d\n", (int) ne_inp, n_tok);
        }
    } else {
    llama_batch batch = llama_batch_get_one(tokens.data(), (int32_t) tokens.size());
    if (llama_decode(ctx, batch) != 0) { std::fprintf(stderr, "decode failed\n"); return 1; }
    }
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
