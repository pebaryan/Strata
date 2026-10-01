// FFN gate for block 0 of GLM-5.3 - the first artifact in this port that exercises a SECOND stage of a block
// rather than a second aspect of the first.
//
// The reference is tools/glm5_moe_reference.py's dense_forward (line 204), which is swiglu_ffn over the block's
// ffn_gate/ffn_up/ffn_down with the SHEXP clamp limit.  The engine reaches the same arithmetic through expert_ffn,
// the single function that serves both a routed expert and a leading dense block (the difference being ff = 12288
// instead of 2048, and the clamp limit passed explicitly).
//
//   usage: ffn_gate [fixture-dir]        default /home/peb/moredata/glm5-ffn-l0
//
// The fixture holds input.bin (4096 floats - the dump's attn_norm-0, a real activation), result.bin (the oracle's
// output) and the three tensors' RAW BLOCKS, which this tool dequantizes with the kernels that were each gated
// bit-exact against the oracle before being used here.  Writing engine_output.bin lets the comparison be
// elementwise rather than statistical: q5k_parity taught that min/max can be blind to systematically wrong scales,
// and this model's ffn_down has a symmetric range that would hide exactly that.
//
// THE CLAMP IS ASSERTED, NOT ASSUMED.  expert_ffn takes the clamp limit as an argument, and the engine's own
// header warns that with a limit of 10 the clamp is invisible until a pre-activation exceeds 10, so a caller who
// forgets it still passes most tests.  This tool therefore computes wg @ x, counts the entries above the limit and
// PRINTS the count, and refuses to report a verdict of PASS if that count is zero - because in that case the gate
// would be reporting on arithmetic it never touched.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/quantize_act.hpp"

static bool read_floats(const std::string& path, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n / sizeof(float));
    if (std::fread(out.data(), 1, (size_t) n, f) != (size_t) n) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

// Dequantize a raw block file with the kernel for its block size.  Both kernels are bit-exact against the oracle
// over their own tensors, so any discrepancy below is the FFN's, not the dequantizer's.
static bool dequant_file(const std::string& path, int block_bytes, int64_t n_elems, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long nbytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blocks((size_t) nbytes);
    if (std::fread(blocks.data(), 1, (size_t) nbytes, f) != (size_t) nbytes) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);

    if ((int64_t) nbytes / block_bytes * 256 != n_elems) {
        std::fprintf(stderr, "%s: %ld bytes at %d per block is not %lld elements - geometry failure\n",
                     path.c_str(), nbytes, block_bytes, (long long) n_elems);
        return false;
    }

    uint8_t* d_blocks = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_blocks, (size_t) nbytes) != cudaSuccess ||
        cudaMalloc(&d_out, (size_t) n_elems * sizeof(float)) != cudaSuccess) {
        std::fprintf(stderr, "cudaMalloc failed\n");
        return false;
    }
    if (cudaMemcpy(d_blocks, blocks.data(), (size_t) nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::fprintf(stderr, "cudaMemcpy failed\n");
        return false;
    }
    if (block_bytes == 176)      strata::kernels::dequant_q5_K(d_blocks, d_out, n_elems, nullptr);
    else if (block_bytes == 210) strata::kernels::dequant_q6_K(d_blocks, d_out, n_elems, nullptr);
    else { std::fprintf(stderr, "no dequantizer for a %d-byte block\n", block_bytes); return false; }

    out.resize((size_t) n_elems);
    const cudaError_t e = cudaMemcpy(out.data(), d_out, (size_t) n_elems * sizeof(float),
                                     cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    cudaFree(d_out);
    if (e != cudaSuccess) { std::fprintf(stderr, "copy back failed: %s\n", cudaGetErrorString(e)); return false; }
    return true;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/peb/moredata/glm5-ffn-l0";
    const int n_embd = 4096, ff = 12288;
    const float clamp_limit = 10.0f;

    std::vector<float> x, ref;
    if (!read_floats(dir + "/input.bin", x) || !read_floats(dir + "/result.bin", ref)) return 1;
    if ((int) x.size() != n_embd || (int) ref.size() != n_embd) {
        std::fprintf(stderr, "input/result are %zu/%zu, expected %d each\n", x.size(), ref.size(), n_embd);
        return 1;
    }

    std::vector<float> wg, wu, wd;
    if (!dequant_file(dir + "/raw_gate.bin", 176, (int64_t) ff * n_embd, wg)) return 1;
    if (!dequant_file(dir + "/raw_up.bin",   176, (int64_t) ff * n_embd, wu)) return 1;
    if (!dequant_file(dir + "/raw_down.bin", 210, (int64_t) ff * n_embd, wd)) return 1;

    // The clamp assertion: count the gate pre-activations above the limit.  With limit = 10 the clamp is a no-op
    // unless this is non-zero, so a zero count means the gate is not testing the clamp at all.
    int64_t n_clamped = 0;
    double pre_max = 0.0;
    for (int i = 0; i < ff; ++i) {
        const float* row = &wg[(size_t) i * n_embd];
        double acc = 0.0;
        for (int j = 0; j < n_embd; ++j) acc += (double) row[j] * (double) x[j];
        if (acc > pre_max) pre_max = acc;
        if (acc > (double) clamp_limit) ++n_clamped;
    }
    std::printf("  gate pre-activations: max %.6g   above the %.1f limit: %lld of %d\n",
                pre_max, (double) clamp_limit, (long long) n_clamped, ff);

    strata::kernels::glm::MoeGeometry g;
    g.n_embd = n_embd;
    g.ff = ff;                      // 12288 for a leading dense block, NOT the 2048 of a routed expert
    g.n_expert = 288;
    g.n_used = 8;
    g.w_scale = 2.5f;
    g.norm_w = true;

    std::vector<float> out((size_t) n_embd);
    strata::kernels::glm::expert_ffn(wg.data(), wu.data(), wd.data(), g, x.data(), out.data(), clamp_limit);

    double e2 = 0.0, r2 = 0.0, worst = 0.0;
    int64_t bad = 0;
    for (int i = 0; i < n_embd; ++i) {
        const double d = std::fabs((double) out[i] - (double) ref[i]);
        if (d > worst) worst = d;
        e2 += (double) out[i] * (double) out[i];
        r2 += (double) ref[i] * (double) ref[i];
    }
    const double scale = std::sqrt(r2 / n_embd);
    for (int i = 0; i < n_embd; ++i)
        if (std::fabs((double) out[i] - (double) ref[i]) > 1e-5 * scale) ++bad;
    std::printf("  engine rms %.6g   oracle rms %.6g\n", std::sqrt(e2 / n_embd), scale);
    std::printf("  worst absolute difference %.3e   relative to oracle rms %.3e   elements off: %lld\n",
                worst, worst / (scale > 0 ? scale : 1.0), (long long) bad);

    const std::string op = dir + "/engine_output.bin";
    FILE* f = std::fopen(op.c_str(), "wb");
    if (f) { std::fwrite(out.data(), sizeof(float), out.size(), f); std::fclose(f); }

    if (n_clamped == 0) {
        std::printf("  FFN GATE: INCONCLUSIVE - no pre-activation exceeds the clamp limit, so this input cannot "
                    "distinguish a correct clamp from no clamp at all\n");
        return 2;
    }
    const bool pass = (bad == 0);
    std::printf("  FFN GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
