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
    if (!nd.load({shard}, table, err)) { std::fprintf(stderr, "native: %s\n", err.c_str()); return 1; }
    uint64_t bytes = 0;
    if (!table.pool_bytes(pack, bytes, err)) { std::fprintf(stderr, "pool_bytes: %s\n", err.c_str()); return 1; }
    void* arena = nullptr;
    if (cudaMalloc(&arena, bytes) != cudaSuccess) { std::fprintf(stderr, "arena alloc of %llu failed\n", (unsigned long long) bytes); return 1; }
    if (!table.load(pack, arena, bytes, err)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }

    strata::core::LayerView v(table, block);
    strata::core::GlmBoundBlock bound;
    cudaStream_t stream = nullptr;
    // GLM-5.3-Flash geometry, from the artifact's own metadata
    const int d_inner = 8192, d_conv = 4;
    if (!bind_glm_block(v, block, d_inner, d_conv, bound, (void*) stream, err)) {
        std::fprintf(stderr, "bind: %s\n", err.c_str()); return 1;
    }

    std::vector<float> host;
    for (const strata::core::GlmBoundBlock::Tensor& t : bound.tensors) {
        if (!t.ptr) { std::printf("%s\tNULL\n", t.name.c_str()); continue; }
        const size_t n = (size_t) t.ne0 * (size_t) t.ne1;
        host.resize(n);
        cudaMemcpy(host.data(), t.ptr, n * sizeof(float), cudaMemcpyDeviceToHost);
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
