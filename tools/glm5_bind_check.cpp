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
#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

static uint64_t fnv1a(const std::vector<float>& v) {
    uint64_t h = 1469598103934665603ull;
    const uint8_t* p = (const uint8_t*) v.data();
    for (size_t i = 0; i < v.size() * sizeof(float); ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: glm5_bind_check <pack_dir> [block]\n"); return 2; }
    const std::string pack = argv[1];
    const int block = argc > 2 ? std::atoi(argv[2]) : 0;
    std::string err;

    strata::core::WeightTable table;
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
        std::printf("blk.%d.%s\tne=%d,%d\t%016llx\n", block, t.name.c_str(), t.ne0, t.ne1,
                    (unsigned long long) fnv1a(host));
    }
    cudaFree(arena);
    return 0;
}
