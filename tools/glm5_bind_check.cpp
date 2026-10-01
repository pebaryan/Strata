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
#include <algorithm>
#include <cstring>
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
    
    cudaFree(arena);
    
    // ---- chain stage 1, when a reference dump directory is given as argv[4] --------------------
    if (argc > 4) {
        const std::string dir = argv[4];
        std::ifstream mf(dir + "/dump.tsv");
        if (!mf) { std::fprintf(stderr, "stage1: cannot open %s/dump.tsv\n", dir.c_str()); return 1; }
        std::string line, file; int n0 = 0, n1 = 0, n2 = 0, n3 = 0;
        while (std::getline(mf, line)) {
            std::vector<std::string> c; std::stringstream ss(line); std::string f;
            while (std::getline(ss, f, '\t')) c.push_back(f);
            if (c.size() < 8 || c[0] != "hc_init") continue;
            n0 = std::atoi(c[1].c_str()); n1 = std::atoi(c[2].c_str());
            n2 = std::atoi(c[3].c_str()); n3 = std::atoi(c[4].c_str()); file = c[7];
        }
        if (file.empty()) { std::fprintf(stderr, "stage1: no hc_init in the dump\n"); return 1; }
        std::vector<float> hc_init((size_t) n0 * n1 * n2 * n3);
        { std::ifstream bf(dir + "/" + file, std::ios::binary);
          bf.read((char*) hc_init.data(), (std::streamsize) (hc_init.size() * sizeof(float))); }
        std::fprintf(stderr, "stage1: hc_init ne=%d,%d,%d,%d from %s\n", n0, n1, n2, n3, file.c_str());
        std::vector<float> fn, base, scale, norm;
        {
            // ORDER TEST in clean code: base first, so if it succeeds the hc_attn_fn copy is the poisoner.
            std::vector<std::string> want = {"hc_attn_base.weight", "hc_attn_scale.weight",
                                             "attn_norm.weight", "hc_attn_fn.weight"};
            std::vector<std::vector<float>*> dst = {&fn, &base, &scale, &norm};
            for (size_t k = 0; k < want.size(); ++k) {
                const float* q = nullptr; size_t cnt = 0; bool quant = false;
                for (const auto& t : bound.tensors)
                    if (t.name == want[k]) { q = t.ptr; quant = t.quantized;
                                             cnt = (size_t) t.ne0 * (size_t) (t.ne1 > 0 ? t.ne1 : 1); }
                if (!q || quant) { std::fprintf(stderr, "stage1: %s unavailable%s\n", want[k].c_str(),
                                                quant ? " (quantized: needs STRATA_HC_DEQUANT=1)" : ""); return 1; }
                dst[k]->resize(cnt);
                const cudaError_t st = cudaMemcpy(dst[k]->data(), q, cnt * sizeof(float),
                                                  cudaMemcpyDeviceToHost);
                std::fprintf(stderr, "stage1: %-22s %zu floats: %s\n", want[k].c_str(), cnt,
                             cudaGetErrorString(st));
                if (st != cudaSuccess) return 1;
            }
        }
        std::vector<float> layer_in(n0);
        strata::kernels::glm::HcMix mix;
        std::string serr;
        if (!strata::core::glm::glm_stage_hc_norm(hc_init.data(), n0, fn.data(), base.data(), scale.data(),
                                                 norm.data(), layer_in.data(), &mix, 1e-5f,
                                                 (void*) stream, serr)) {
            std::fprintf(stderr, "stage1: %s\n", serr.c_str()); return 1;
        }
        double mn = 1e30, mx = -1e30, sm = 0;
        for (float v : layer_in) { mn = std::min(mn, (double) v); mx = std::max(mx, (double) v); sm += v; }
        std::printf("stage1.attn_norm-0\tne=%d,0\tn=%zu\t%.9g\t%.9g\t%.9g\t0\n", n0, layer_in.size(), mn, mx, sm);
    }

return 0;
}
