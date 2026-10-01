// Parity gate for dequant_q5_K against an INDEPENDENT reference.
//
// The reference is the oracle's own dequantization of blk.0.attn_q.weight - not llama.cpp's C, from which this
// kernel was transcribed.  Comparing a transcription to its source only confirms that the copying was faithful;
// it says nothing about whether the copy is right.  The oracle has verified every other gate in this port, so it
// is the reference that can actually fail.
//
//   usage: q5k_parity [fixture-dir]        default /home/peb/moredata/glm5-q5k-parity
//
// The fixture holds raw_q5k_blocks.bin (the tensor's raw 176-byte blocks, straight from the GGUF) and
// oracle_dequantised.bin (33,554,432 floats).  This tool runs the kernel over the blocks and writes its own
// floats to engine_output.bin; comparing the two files is the gate, and tools/q5k_parity.py does that after
// checking the sizes agree - a length mismatch must be reported as a geometry failure, not as a value failure.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/kernels/quantize_act.hpp"

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/peb/moredata/glm5-q5k-parity";
    // Block geometry and input name are arguments: 176 bytes is Q5_K, 210 is Q6_K.  Generalised rather than
    // duplicated so that the second K-quant gets the same gate as the first, and so a future type needs an
    // argument rather than another copy of this file.
    const int block_bytes = argc > 2 ? std::atoi(argv[2]) : 176;
    const std::string in_name = argc > 3 ? argv[3] : "raw_q5k_blocks.bin";

    const std::string in_path = dir + "/" + in_name;
    FILE* f = std::fopen(in_path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", in_path.c_str()); return 1; }
    std::fseek(f, 0, SEEK_END);
    const long nbytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blocks((size_t) nbytes);
    if (std::fread(blocks.data(), 1, (size_t) nbytes, f) != (size_t) nbytes) {
        std::fprintf(stderr, "short read on %s\n", in_path.c_str());
        std::fclose(f);
        return 1;
    }
    std::fclose(f);

    const int64_t n_blocks = (int64_t) nbytes / block_bytes;
    const int64_t n = n_blocks * 256;
    std::printf("q5k parity: %lld raw blocks (%ld bytes) -> %lld elements\n",
                (long long) n_blocks, nbytes, (long long) n);
    if (n_blocks * block_bytes != (int64_t) nbytes) {
        std::fprintf(stderr, "the file size is not a multiple of %d: this is a geometry failure, not a value "
                             "failure\n", block_bytes);
        return 1;
    }

    uint8_t* d_blocks = nullptr;
    float* d_out = nullptr;
    cudaError_t e = cudaMalloc(&d_blocks, (size_t) nbytes);
    if (e == cudaSuccess) e = cudaMalloc(&d_out, (size_t) n * sizeof(float));
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc: %s\n", cudaGetErrorString(e));
        return 1;
    }
    e = cudaMemcpy(d_blocks, blocks.data(), (size_t) nbytes, cudaMemcpyHostToDevice);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy host->device: %s\n", cudaGetErrorString(e));
        return 1;
    }

    // The kernel reads device memory; a null stream means it synchronises internally.
    if (block_bytes == 176) {
        strata::kernels::dequant_q5_K(d_blocks, d_out, n, nullptr);
    } else if (block_bytes == 210) {
        strata::kernels::dequant_q6_K(d_blocks, d_out, n, nullptr);
    } else {
        std::fprintf(stderr, "no dequantizer for a %d-byte block\n", block_bytes);
        return 1;
    }
    e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_q5_K: %s\n", cudaGetErrorString(e));
        return 1;
    }

    std::vector<float> out((size_t) n);
    e = cudaMemcpy(out.data(), d_out, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
    if (e != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy device->host: %s\n", cudaGetErrorString(e));
        return 1;
    }

    double mn = out[0], mx = out[0], sum = 0.0;
    for (float v : out) { if (v < mn) mn = v; if (v > mx) mx = v; sum += v; }
    std::printf("  engine : min %+.9g  max %+.9g  sum %+.9g\n", mn, mx, sum);

    const std::string out_path = dir + "/engine_output.bin";
    FILE* g = std::fopen(out_path.c_str(), "wb");
    if (!g) { std::fprintf(stderr, "cannot write %s\n", out_path.c_str()); return 1; }
    std::fwrite(out.data(), sizeof(float), out.size(), g);
    std::fclose(g);
    std::printf("  wrote %s (%zu floats) - compare with tools/q5k_parity.py\n", out_path.c_str(), out.size());

    cudaFree(d_blocks);
    cudaFree(d_out);
    return 0;
}
