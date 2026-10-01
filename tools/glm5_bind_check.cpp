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
#include <set>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/core/glm_bind.hpp"
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
    return 0;
}
