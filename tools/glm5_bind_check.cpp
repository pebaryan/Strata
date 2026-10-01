/// 9.1's pass check: bind layer 0 from the pack and print a checksum per bound tensor.
///
/// The binding is only proven right when its NUMBERS match the Python block-0 outputs, which are themselves
/// verified against llama.cpp.  Checksums are FNV-1a over the fp32 values, printed one per line, so
/// tools/glm5_bind_check.py can compute the same numbers from the verified oracle and diff them.
///
/// Usage: glm5_bind_check <pack_dir> [block]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <set>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/core/glm_bind.hpp"
#include "strata/core/glm_layer.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

static uint64_t fnv1a(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = (const uint8_t*) v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: glm5_bind_check <pack_dir> <gguf_shard1> [block]\n"); return 2; }
    const std::string pack = argv[1];
    const std::string shard = argc > 2 ? argv[2] : "";
    const int block = argc > 3 ? std::atoi(argv[3]) : 0;
    const char* refdump = argc > 4 ? argv[4] : nullptr;
    std::string err;

    strata::core::WeightTable table;
    // The 14 quantized tensors are served from the GGUF, not from dense.bin: this pack records a shape for
    // them and nothing else, so the native projection has to be registered with the table or the loader
    // refuses - which is what its "run with --native SHARD1" message means.  Register BEFORE loading, because
    // pool_bytes/load decide which rows they own from what the table already knows is served natively.
    strata::core::NativeDense nd;
    if (shard.empty()) { std::fprintf(stderr, "usage: glm5_bind_check <pack_dir> <gguf_shard1> [block]\n"); return 2; }
    // ALL shards, not just the first: this artifact's shard 1 is metadata-only, so a one-shard NativeDense
    // sees an empty tensor list and reports that there are no supported GDN/QSA matrices.
    std::vector<std::string> shards;
    {
        std::string fn = shard;
        const size_t slash = fn.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "" : fn.substr(0, slash + 1);
        const std::string base = slash == std::string::npos ? fn : fn.substr(slash + 1);
        const size_t pos = base.find("-00001-of-");
        if (pos != std::string::npos) {
            const std::string stem = base.substr(0, pos);
            const std::string tail = base.substr(base.find("-of-") + 4);
            const int n = std::atoi(tail.substr(0, tail.find('.')).c_str());
            for (int i = 1; i <= n; ++i) {
                char buf[4096];
                std::snprintf(buf, sizeof buf, "%s%s-%05d-of-%05d.gguf", dir.c_str(), stem.c_str(), i, n);
                shards.emplace_back(buf);
            }
        } else {
            shards.push_back(shard);
        }
    }
    // The two loaders depend on each other, and the engine's own mechanism is served_names: it names the
    // tensors NativeDense will serve from the GGUF, which the table then SKIPS instead of expecting in
    // dense.bin.  Compute it first, hand it to the table, and only then register the native projection -
    // otherwise the table has no row to mark native and reports "tensor absent from canonical table".
    std::set<std::string> served;
    if (!strata::core::NativeDense::served_names(shards, false, served, err)) {
        std::fprintf(stderr, "served_names: %s\n", err.c_str()); return 1;
    }
    std::fprintf(stderr, "natively served tensors: %zu\n", served.size());
    for (const char* probe : {"blk.3.attn_k_b.weight", "blk.3.attn_v_b.weight", "blk.0.attn_q.weight"}) {
        std::fprintf(stderr, "  probe %-26s in served set: %s\n", probe, served.count(probe) ? "yes" : "NO");
    }
    uint64_t bytes = 0;
    if (!table.pool_bytes(pack, bytes, err, &served)) { std::fprintf(stderr, "pool_bytes: %s\n", err.c_str()); return 1; }
    void* arena = nullptr;
    if (cudaMalloc(&arena, bytes) != cudaSuccess) { std::fprintf(stderr, "arena alloc of %llu failed\n", (unsigned long long) bytes); return 1; }
    if (!table.load(pack, arena, bytes, err, &served)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    if (!nd.load(shards, table, err)) { std::fprintf(stderr, "native: %s\n", err.c_str()); return 1; }

    strata::core::LayerView v(table, block);
    strata::core::GlmBoundBlock bound;
    cudaStream_t stream = nullptr;
    // GLM-5.3-Flash geometry, from the artifact's own metadata
    const int d_inner = 8192, d_conv = 4;
    if (!bind_glm_block(v, block, d_inner, d_conv, shards, bound, (void*) stream, err)) {
        std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1;
    }

    std::vector<float> host;
    for (const strata::core::GlmBoundBlock::Tensor& t : bound.tensors) {
        if (!t.ptr) { std::printf("%s\tNULL\n", t.name.c_str()); continue; }
        // ne1 == 0 means a 1-D tensor, not an empty one: ne0*ne1 silently gave n=0 for every norm vector.
        const size_t n = (size_t) t.ne0 * (size_t) (t.ne1 > 0 ? t.ne1 : 1);
        if (t.quantized) {
            std::printf("blk.%d.%s\tne=%d,%d\n=quantized (native GGUF blocks; \n=%zu)\n", block, t.name.c_str(),
                        t.ne0, t.ne1, n);
            continue;
        }
        std::fprintf(stderr, "  loopcopy %-22s t.ptr=%p n=%zu\n", t.name.c_str(), (const void*) t.ptr, n);
        if (t.name == "hc_attn_base.weight") {
            // Same scope, same vector type, same expressions as the stage lambda, immediately after the copy
            // that just succeeded.  If this fails, the lambda's body differs from its appearance; if it
            // succeeds, the difference between the loop and the stage is TEMPORAL.
            std::vector<float> tmp;
            tmp.resize(n);
            const cudaError_t st2 = cudaMemcpy(tmp.data(), t.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
            std::fprintf(stderr, "  inloop 2nd copy of hc_attn_base: %s\n", cudaGetErrorString(st2));
            fflush(stderr);
        }
        host.resize(n);
        const cudaError_t st = cudaMemcpy(host.data(), t.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
        if (st != cudaSuccess) {
            std::printf("blk.%d.%s\tCOPY FAILED: %s\n", block, t.name.c_str(), cudaGetErrorString(st));
            continue;
        }
        // stats, not just a hash: the engine's dequantizer and llama.cpp's need not agree bit for bit, and a
        // hash cannot tell a ULP apart from a wrong tensor.
        double sum = 0.0;
        float mn = host[0], mx = host[0];
        for (float f : host) { sum += f; if (f < mn) mn = f; if (f > mx) mx = f; }
        std::printf("blk.%d.%s\tne=%d,%d\tn=%zu\t%.9g\t%.9g\t%.9g\t%016llx\n", block, t.name.c_str(),
                    t.ne0, t.ne1, n, mn, mx, sum, (unsigned long long) fnv1a(host));
    }
    cudaFree(arena);
    
    // ---- chain stage 1: hc_pre -> rms_norm, on the REFERENCE's own hc_init -----------------------
    // Judged against hc_attn_pre-N / attn_norm-N rather than only end to end, so an error is localised
    // to a stage instead of to "the model".
    if (refdump) {
        auto tensor_by_name = [&](const char* n) -> const float* {
            for (const auto& t : bound.tensors) if (t.name == n) return t.ptr;
            return nullptr;
        };
        const std::string tsv = std::string(refdump) + "/dump.tsv";
        std::ifstream mf(tsv);
        if (!mf) { std::fprintf(stderr, "stage1: cannot read %s\n", tsv.c_str()); return 1; }
        std::string file, line;
        std::vector<float> hc_init;
        int ne0 = 0, ne1 = 0, ne2 = 0, ne3 = 0;
        while (std::getline(mf, line)) {
            std::vector<std::string> c; std::stringstream ss(line); std::string f;
            while (std::getline(ss, f, '\t')) c.push_back(f);
            if (c.size() < 8 || c[0] != "hc_init") continue;
            ne0 = std::atoi(c[1].c_str()); ne1 = std::atoi(c[2].c_str());
            ne2 = std::atoi(c[3].c_str()); ne3 = std::atoi(c[4].c_str());
            file = c[7];
        }
        if (file.empty()) { std::fprintf(stderr, "stage1: no hc_init in %s\n", tsv.c_str()); return 1; }
        std::ifstream bf(std::string(refdump) + "/" + file, std::ios::binary);
        hc_init.resize((size_t) ne0 * ne1 * ne2 * ne3);
        bf.read((char*) hc_init.data(), (std::streamsize) (hc_init.size() * sizeof(float)));
        std::fprintf(stderr, "stage1: hc_init ne=%d,%d,%d,%d\n", ne0, ne1, ne2, ne3);

        // ggml order is ne0 fastest.  Element (embd, hc, tok) of [ne0,ne1,ne2,ne3] = [embd,hc,tok] sits at
        // embd + ne0*hc + ne0*ne1*tok - the reshape lesson from phase 8 - so token 0's [HC][n_embd] streams
        // are the first ne0*ne1 floats.
        for (const char* n : {"hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
                              "attn_norm.weight"}) {
            const float* q = nullptr; bool quant = false; int d0 = 0, d1 = 0;
            for (const auto& t : bound.tensors)
                if (t.name == n) { q = t.ptr; quant = t.quantized; d0 = t.ne0; d1 = t.ne1; }
            std::fprintf(stderr, "  arg %-22s ptr=%p quantized=%d ne=%d,%d\n", n, (const void*) q, (int) quant,
                         d0, d1);
        }
        std::vector<float> layer_in(ne0);
        std::string serr;
        // hc_pre is HOST code - std::vector, std::sqrt, std::getenv, no CUDA anywhere in it - so every pointer
        // it receives must be host-accessible.  The bound weights live in the device arena (and the
        // dequantized hc_fn in a device buffer), which is exactly what the original segfault was: a DEVICE
        // pointer dereferenced as host memory.  Copy them across first.
        auto host_copy = [&](const char* n, size_t count, std::vector<float>& dst) -> bool {
            const float* q = nullptr; int d0 = 0, d1 = 0; bool quant = false;
            for (const auto& t : bound.tensors)
                if (t.name == n) { q = t.ptr; d0 = t.ne0; d1 = t.ne1; quant = t.quantized; }
            if (!q) { std::fprintf(stderr, "stage1: %s not bound\n", n); return false; }
            if (quant) {
                // Sizing a QUANTIZED tensor's copy as floats asks CUDA for more bytes than the allocation
                // holds and it refuses ("invalid argument") - which is what happened here for hc_attn_fn,
                // whose Q8_0 blocks are 417792 bytes but whose 393216 elements would be 1.5 MB.  Either
                // dequantize it first (STRATA_HC_DEQUANT=1 makes the binding do exactly that) or refuse
                // loudly; do not silently ask for the wrong number of bytes.
                std::fprintf(stderr, "stage1: %s is QUANTIZED - re-run with STRATA_HC_DEQUANT=1 so the binding "
                                     "dequantizes it, rather than copying %zu floats out of a block buffer\n",
                             n, (size_t) d0 * (size_t) (d1 > 0 ? d1 : 1));
                return false;
            }
            const size_t want = count ? count : (size_t) d0 * (size_t) (d1 > 0 ? d1 : 1);
            dst.resize(want);
            std::fprintf(stderr, "  hostcopy %-22s q=%p ne=%d,%d want=%zu dst=%p\n", n, (const void*) q, d0, d1,
                         want, (void*) dst.data());
            fflush(stderr);
            // The arena is cudaMalloc'd device memory (this driver allocates it that way), and the
            // dequantized hc_fn sits in a cudaMalloc buffer too, while the loader's own staging buffers are
            // cudaHostAlloc'd.  Rather than infer which is which - three earlier attempts did, and both
            // cudaMemcpyDeviceToHost and plain memcpy were wrong for some tensor - use cudaMemcpyDefault,
            // which resolves the direction under UVA.
            // Use exactly what the bind print loop uses, because that is PROVEN to work on these same
            // tensors in the same run: cudaMemcpyDeviceToHost.  cudaPointerGetAttributes does not recognise
            // this arena as device ("other"), and cudaMemcpyDefault needs those attributes to infer a
            // direction - so it faults.  The pointer is fine; only Default's inference is not.
            const cudaError_t cst = cudaMemcpy(dst.data(), q, want * sizeof(float), cudaMemcpyDeviceToHost);
            if (cst != cudaSuccess) {
                std::fprintf(stderr, "stage1: copy %s failed: %s\n", n, cudaGetErrorString(cst));
                return false;
            }
            std::fprintf(stderr, "  %-22s copied %zu floats\n", n, want);
            return true;
        };
        std::vector<float> h_fn, h_base, h_scale, h_norm;
        // ORDER TEST: hc_attn_base FIRST, so if it succeeds here the hc_attn_fn copy is what breaks what
        // follows - the print loop reads base before fn and never fails.
        if (!host_copy("hc_attn_base.weight", 0, h_base) || !host_copy("hc_attn_scale.weight", 0, h_scale) ||
            !host_copy("attn_norm.weight", 0, h_norm) || !host_copy("hc_attn_fn.weight", 0, h_fn))
            return 1;
        std::fprintf(stderr, "stage1: host copies fn=%zu base=%zu scale=%zu norm=%zu\n", h_fn.size(),
                     h_base.size(), h_scale.size(), h_norm.size());
        strata::kernels::glm::HcMix mix;
        if (!strata::core::glm::glm_stage_hc_norm(hc_init.data(), ne0, h_fn.data(), h_base.data(),
                                                 h_scale.data(), h_norm.data(), layer_in.data(), &mix, 1e-5f,
                                                 (void*) stream, serr)) {
            std::fprintf(stderr, "stage1: call FAILED: %s\n", serr.c_str()); return 1;
        }
        std::fprintf(stderr, "stage1: call returned\n");
        double mn = 1e30, mx = -1e30, sm = 0;
        for (float v : layer_in) { mn = std::min(mn, (double) v); mx = std::max(mx, (double) v); sm += v; }
        std::printf("blk.%d.attn_norm.weight\tne=%d,0\tn=%zu\t%.9g\t%.9g\t%.9g\t0\n", block, ne0,
                    layer_in.size(), mn, mx, sm);
    }

return 0;
}
