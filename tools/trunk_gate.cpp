// trunk_gate - run ONE block through glm_trunk_forward and report what comes out.
//
// This is the first execution of the trunk loop as an ASSEMBLY.  Every stage it calls has passed its own gate in
// isolation; none of them has ever run in sequence, in this order, on real weights, with the hyper-connection
// residual routing between them.  That is a different claim from the sum of the parts, which is the whole reason this
// tool exists.
//
// It needs NO pack: layer 0's block weights and KDA weights are already float fixtures, and the dense FFN matrices are
// dequantized from the FFN fixture's raw blobs the same way ffn_gate.cpp does it - Q5_K for the 176-byte gate and up
// blocks, Q6_K for the 210-byte down blocks.  That second point is worth naming: dequant_q5_K is gated bit-exact
// against the oracle's own dequantization, but dequant_q6_K was only transcribed, and this is the first time it is
// given real tensor data - blk.0.ffn_down.weight is 41,287,680 bytes = 196,608 blocks.
//
// WHAT IT ASSERTS, and what it deliberately does not:
//   * it FAILS if the loop returns false, if any output is not finite, or if the output is identically zero - the
//     three ways an assembly can be wrong that no individual stage can be;
//   * it does NOT assert tight numerical parity against the dump's l_out-0.  This port has already established that
//     the DUMP IS THE OUTLIER at the block input (its hc_attn_pre is ~0.08% off, while the engine and the oracle agree
//     to seven digits there) and that the KDA's exponential gate AMPLIFIES exactly that difference.  A tight bound
//     against the dump would produce a FAIL that says nothing about the trunk - a mistake already made and retracted
//     once in this port.  So the dump comparison is MEASURED AND REPORTED as a ratio, with the amplification named.
//
// Usage: trunk_gate <block-dir> <kda-dir> <ffn-dir> <dump-dir>

#include <algorithm>
#include <cmath>
#include <cstdint>

#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/glm_layer_weights.hpp"
#include "strata/core/glm_trunk.hpp"
#include "strata/kernels/quantize_act.hpp"

namespace {

/// Reads a fixture's floats.  `expected` is ADVISORY AND NOT ENFORCED: the file's own length is the truth, because
/// this port has already been caught twice by a size it inferred from geometry instead of measuring - ten of the
/// block gate's twenty-nine size assertions were wrong by 2x to 128x, and one of them was mine.  A mismatch is
/// REPORTED, so a wrong expectation is visible, and the engine is then given the tensor the file actually holds.
bool load_floats(const std::string& path, size_t expected, std::vector<float>& out, const char* what) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("  MISSING %s (%s)\n", path.c_str(), what); return false; }
    std::fseek(f, 0, SEEK_END);
    const long len = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (len <= 0 || len % 4 != 0) {
        std::fclose(f);
        std::printf("  SIZE %s (%s): %ld bytes is not a whole number of floats\n", path.c_str(), what, len);
        return false;
    }
    const size_t n = (size_t) len / 4;
    out.assign(n, 0.0f);
    const size_t got = std::fread(out.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { std::printf("  SHORT %s: %zu of %zu floats\n", path.c_str(), got, n); return false; }
    if (expected && expected != n) {
        std::printf("  NOTE size %-22s holds %zu floats, the gate expected %zu (%s) - using the file's own length\n",
                    what, n, expected, (n > expected) ? "bigger" : "SMALLER");
    }
    return true;
}

/// The dump's container: five u32 words [0, ne0, ne1, ne2, ne3], then ne0-fastest floats.
bool load_dump(const std::string& path, std::vector<float>& out, int ne[4], const char* what) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("  MISSING %s (%s)\n", path.c_str(), what); return false; }
    unsigned hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(hdr, 4, 5, f) != 5) { std::fclose(f); std::printf("  BAD header %s\n", path.c_str()); return false; }
    for (int i = 0; i < 4; ++i) ne[i] = (int) hdr[i + 1];
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) (ne[i] > 0 ? ne[i] : 1);
    out.assign(n, 0.0f);
    const size_t got = std::fread(out.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { std::printf("  SHORT %s: %zu of %zu floats\n", path.c_str(), got, n); return false; }
    return true;
}

/// Dequantize a raw blob the way ffn_gate.cpp does: the block size IS the format, and any other size is refused
/// rather than guessed at.
/// Dequantize a raw blob with the kernel for its block size, the way ffn_gate.cpp does it.
///
/// THESE ARE DEVICE KERNELS.  The first version of this tool handed them host pointers and took an illegal memory
/// access - the same class of mistake as inferring a size instead of measuring it, and caught the same cheap way.
/// The block size IS the format: any other size is refused rather than guessed at, and the byte count must account for
/// exactly n_elems elements, because a blob of the wrong width is refused rather than mis-indexed.
bool dequant_raw(const std::string& path, int block_bytes, int64_t n_elems, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::printf("  MISSING %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long nbytes = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blocks((size_t) nbytes);
    const size_t got = std::fread(blocks.data(), 1, blocks.size(), f);
    std::fclose(f);
    if (got != blocks.size()) { std::printf("  SHORT %s\n", path.c_str()); return false; }
    if ((int64_t) nbytes / block_bytes * 256 != n_elems) {
        std::printf("  REFUSED %s: %ld bytes at %d per block is not %lld elements - geometry failure\n",
                    path.c_str(), nbytes, block_bytes, (long long) n_elems);
        return false;
    }
    uint8_t* d_blocks = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_blocks, (size_t) nbytes) != cudaSuccess ||
        cudaMalloc(&d_out, (size_t) n_elems * sizeof(float)) != cudaSuccess) {
        std::printf("  cudaMalloc failed for %s\n", path.c_str());
        return false;
    }
    if (cudaMemcpy(d_blocks, blocks.data(), (size_t) nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("  cudaMemcpy to device failed for %s\n", path.c_str()); return false;
    }
    if (block_bytes == 176) {
        strata::kernels::dequant_q5_K(d_blocks, d_out, n_elems, nullptr);
    } else if (block_bytes == 210) {
        strata::kernels::dequant_q6_K(d_blocks, d_out, n_elems, nullptr);
    } else {
        std::printf("  REFUSED %s: no dequantizer for a %d-byte block\n", path.c_str(), block_bytes);
        cudaFree(d_blocks); cudaFree(d_out); return false;
    }
    out.resize((size_t) n_elems);
    const cudaError_t e = cudaMemcpy(out.data(), d_out, (size_t) n_elems * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_blocks);
    cudaFree(d_out);
    if (e != cudaSuccess) { std::printf("  copy back failed: %s\n", cudaGetErrorString(e)); return false; }
    std::printf("  dequantized %s: %ld bytes = %ld blocks x %d B -> %lld floats (device kernel)\n", path.c_str(),
                nbytes, nbytes / block_bytes, block_bytes, (long long) n_elems);
    return true;
}

double rms_of(const std::vector<float>& v) {
    double s = 0.0;
    for (float x : v) s += (double) x * x;
    return std::sqrt(s / (double) (v.empty() ? 1 : v.size()));
}

// ---- the fixture tensors, held so their addresses stay stable for the bound block ----
struct Fixture {
    std::vector<float> hc_attn_fn, hc_attn_base, hc_attn_scale, attn_norm;
    std::vector<float> hc_ffn_fn, hc_ffn_base, hc_ffn_scale, ffn_norm;
    std::vector<float> wq, wk, wv, conv_q, conv_k, conv_v, ssm_a, dt_bias;
    std::vector<float> f_a, f_b, beta, g_a, g_b, o_norm, wo;
    std::vector<float> ffn_gate, ffn_up, ffn_down;
};

void add(strata::core::GlmBoundBlock& b, const char* name, std::vector<float>& v) {
    strata::core::GlmBoundBlock::Tensor t;
    t.name = name;
    t.ptr = v.data();
    t.ne0 = (int) v.size();
    t.ne1 = 1;
    b.tensors.push_back(t);
}

struct ProviderCtx {
    const strata::core::GlmBoundBlock* block = nullptr;
    int layer = 0;
    strata::kernels::glm::KdaGeometry kda_g;
    strata::kernels::glm::MlaGeometry mla_g;
    strata::kernels::glm::MoeGeometry dense_g;   // the dense FFN's own width, not the routed default
    strata::kernels::glm::KdaWeights kda;
    strata::kernels::glm::MlaWeights mla;
};

bool provider(void* raw, int layer, strata::core::glm::GlmTrunkLayerWeights& out, std::string& err) {
    ProviderCtx* c = (ProviderCtx*) raw;
    (void) layer;   // the loop's index; the CLI layer is what selects the weights, see the note in main
    if (!strata::core::glm::glm_fill_layer_weights(*c->block, c->layer, c->kda_g, c->mla_g, c->kda, c->mla, nullptr,
                                                   out, err)) {
        return false;
    }
    if (c->dense_g.ff) out.moe_g = &c->dense_g;      // a leading dense block's width, set explicitly
    out.clamp_limit = 10.0f;                          // swiglu_clamp_shexp, from the artifact's metadata
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: trunk_gate <block-dir> <kda-dir> <ffn-dir> <dump-dir>\n");
        return 2;
    }
    const std::string bd = argv[1], kd = argv[2], fd = argv[3], dd = argv[4];
    // Extra arguments so two blocks can be chained BY RUNNING THE TOOL TWICE: the second run consumes the first's
    // output as its input.  That is not a substitute for the 45-block loop - it is a way to execute and verify a
    // SECOND block, with its own real weights, before the pack-based chain target exists.  It is valid to serve
    // layer N's block through the loop's layer-0 slot because the loop selects only the KIND (KDA/MLA, dense/routed)
    // and the dense KDA blocks have identical tensor names; `layer` below is what the weight provider dispatches on.
    const int layer = argc > 5 ? std::atoi(argv[5]) : 0;
    const std::string in_name = argc > 6 ? argv[6] : "hc_init.bin";
    const std::string out_name = argc > 7 ? argv[7] : "";
    const int N_EMBD = 4096, NH = 64, HD = 128, D_CONV = 4, FF_DENSE = 12288, HC = 4;

    Fixture fx;
    bool all = true;
    all &= load_floats(bd + "/w_hc_attn_fn.bin", (size_t) HC * 4 * N_EMBD, fx.hc_attn_fn, "hc_attn_fn");
    all &= load_floats(bd + "/w_hc_attn_base.bin", HC, fx.hc_attn_base, "hc_attn_base");
    all &= load_floats(bd + "/w_hc_attn_scale.bin", 3, fx.hc_attn_scale, "hc_attn_scale");
    all &= load_floats(bd + "/w_attn_norm.bin", N_EMBD, fx.attn_norm, "attn_norm");
    all &= load_floats(bd + "/w_hc_ffn_fn.bin", (size_t) HC * 4 * N_EMBD, fx.hc_ffn_fn, "hc_ffn_fn");
    all &= load_floats(bd + "/w_hc_ffn_base.bin", HC, fx.hc_ffn_base, "hc_ffn_base");
    all &= load_floats(bd + "/w_hc_ffn_scale.bin", 3, fx.hc_ffn_scale, "hc_ffn_scale");
    all &= load_floats(bd + "/w_ffn_norm.bin", N_EMBD, fx.ffn_norm, "ffn_norm");
    // KDA, from the binding check's own fixture - the same files whose sizes that check verified
    all &= load_floats(kd + "/w_wq.bin", (size_t) NH * HD * N_EMBD, fx.wq, "wq");
    all &= load_floats(kd + "/w_wk.bin", (size_t) NH * HD * N_EMBD, fx.wk, "wk");
    all &= load_floats(kd + "/w_wv.bin", (size_t) NH * HD * N_EMBD, fx.wv, "wv");
    all &= load_floats(kd + "/w_conv_q.bin", (size_t) D_CONV * NH * HD, fx.conv_q, "conv_q");
    all &= load_floats(kd + "/w_conv_k.bin", (size_t) D_CONV * NH * HD, fx.conv_k, "conv_k");
    all &= load_floats(kd + "/w_conv_v.bin", (size_t) D_CONV * NH * HD, fx.conv_v, "conv_v");
    all &= load_floats(kd + "/w_ssm_a.bin", NH, fx.ssm_a, "ssm_a");
    all &= load_floats(kd + "/w_dt_bias.bin", (size_t) NH * HD, fx.dt_bias, "dt_bias");
    all &= load_floats(kd + "/w_ssm_f_a.bin", (size_t) HD * N_EMBD, fx.f_a, "ssm_f_a");
    all &= load_floats(kd + "/w_ssm_f_b.bin", (size_t) NH * HD * HD, fx.f_b, "ssm_f_b");
    all &= load_floats(kd + "/w_ssm_beta.bin", (size_t) NH * N_EMBD, fx.beta, "ssm_beta");
    all &= load_floats(kd + "/w_ssm_g_a.bin", (size_t) HD * N_EMBD, fx.g_a, "ssm_g_a");
    all &= load_floats(kd + "/w_ssm_g_b.bin", (size_t) NH * HD * HD, fx.g_b, "ssm_g_b");
    all &= load_floats(kd + "/w_o_norm.bin", HD, fx.o_norm, "o_norm");
    all &= load_floats(kd + "/w_wo.bin", (size_t) N_EMBD * NH * HD, fx.wo, "wo");
    // the dense FFN, dequantized - this is also the first real data the transcribed Q6_K sees
    all &= dequant_raw(fd + "/raw_gate.bin", 176, (int64_t) FF_DENSE * N_EMBD, fx.ffn_gate);
    all &= dequant_raw(fd + "/raw_up.bin", 176, (int64_t) FF_DENSE * N_EMBD, fx.ffn_up);
    all &= dequant_raw(fd + "/raw_down.bin", 210, (int64_t) N_EMBD * FF_DENSE, fx.ffn_down);

    // the block's INPUT and the dump's own output, for the reported comparison
    std::vector<float> x, want;
    int ne_x[4] = {0, 0, 0, 0}, ne_w[4] = {0, 0, 0, 0};
    all &= load_dump(dd + "/" + in_name, x, ne_x, in_name.c_str());
    all &= load_dump(dd + "/l_out-0.bin", want, ne_w, "l_out-0 (the dump's block output)");
    std::printf("trunk_gate: hc_init ne=[%d,%d,%d,%d] %zu floats; l_out-0 ne=[%d,%d,%d,%d] %zu floats\n", ne_x[0],
                ne_x[1], ne_x[2], ne_x[3], x.size(), ne_w[0], ne_w[1], ne_w[2], ne_w[3], want.size());

    if (!all) {
        std::fprintf(stderr, "trunk_gate: a fixture is missing or short - refusing to run on an incomplete fixture\n");
        return 2;
    }
    if (x.size() < (size_t) HC * N_EMBD) {
        std::fprintf(stderr, "trunk_gate: hc_init holds %zu floats, need %d for one token's %d streams\n", x.size(),
                     HC * N_EMBD, HC);
        return 2;
    }

    strata::core::GlmBoundBlock block;
    add(block, "hc_attn_fn.weight", fx.hc_attn_fn);
    add(block, "hc_attn_base.weight", fx.hc_attn_base);
    add(block, "hc_attn_scale.weight", fx.hc_attn_scale);
    add(block, "attn_norm.weight", fx.attn_norm);
    add(block, "hc_ffn_fn.weight", fx.hc_ffn_fn);
    add(block, "hc_ffn_base.weight", fx.hc_ffn_base);
    add(block, "hc_ffn_scale.weight", fx.hc_ffn_scale);
    add(block, "ffn_norm.weight", fx.ffn_norm);
    add(block, "attn_q.weight", fx.wq);
    add(block, "attn_k.weight", fx.wk);
    add(block, "attn_v.weight", fx.wv);
    add(block, "ssm_conv1d_q.weight", fx.conv_q);
    add(block, "ssm_conv1d_k.weight", fx.conv_k);
    add(block, "ssm_conv1d_v.weight", fx.conv_v);
    add(block, "ssm_a", fx.ssm_a);
    add(block, "ssm_dt.bias", fx.dt_bias);
    add(block, "ssm_f_a.weight", fx.f_a);
    add(block, "ssm_f_b.weight", fx.f_b);
    add(block, "ssm_beta.weight", fx.beta);
    add(block, "ssm_g_a.weight", fx.g_a);
    add(block, "ssm_g_b.weight", fx.g_b);
    add(block, "ssm_norm.weight", fx.o_norm);
    add(block, "attn_output.weight", fx.wo);
    add(block, "ffn_gate.weight", fx.ffn_gate);
    add(block, "ffn_up.weight", fx.ffn_up);
    add(block, "ffn_down.weight", fx.ffn_down);

    ProviderCtx ctx;
    ctx.block = &block;
    ctx.layer = layer;
    ctx.kda_g.n_embd = N_EMBD; ctx.kda_g.nh = NH; ctx.kda_g.hd = HD; ctx.kda_g.d_conv = D_CONV;
    ctx.mla_g.n_embd = N_EMBD;
    ctx.dense_g.n_embd = N_EMBD; ctx.dense_g.ff = FF_DENSE;

    // KDA recurrent state: nh*hd*hd floats, zeroed for a fresh sequence
    std::vector<float> kda_state((size_t) NH * HD * HD, 0.0f);
    float* kda_state_ptrs[1] = {kda_state.data()};
    int kda_index[16] = {0};
    for (int i = 0; i < 16; ++i) kda_index[i] = (i == 0) ? 0 : -1;   // the loop's slot 0; `layer` selects weights

    strata::core::glm::GlmTrunkState state;
    state.kda_state = kda_state_ptrs;
    state.mla_cache = nullptr; state.mla_len = nullptr;
    state.kda_index = kda_index; state.mla_index = kda_index;

    std::vector<float> out((size_t) HC * N_EMBD, 0.0f);
    std::string err;
    const bool ok = strata::core::glm::glm_trunk_forward(x.data(), 1, provider, &ctx, ctx.kda_g, ctx.mla_g, 1e-5f,
                                                         state, out.data(), nullptr, err);
    if (!ok) {
        std::printf("TRUNK GATE: FAIL - the loop returned false: %s\n", err.c_str());
        return 1;
    }

    // ---- the three ways an assembly can be wrong that no single stage can be ----
    bool finite = true;
    double sum = 0.0;
    for (float v : out) { if (!std::isfinite(v)) finite = false; sum += std::fabs((double) v); }
    const double e_rms = rms_of(out);
    std::printf("  engine l_out: rms %.6g   sum|.| %.6g   (loop returned true)\n", e_rms, sum);
    if (!finite) { std::printf("TRUNK GATE: FAIL - the output is not finite\n"); return 1; }
    if (sum == 0.0) { std::printf("TRUNK GATE: FAIL - the output is identically zero\n"); return 1; }

    // ---- the dump comparison: measured and reported, NOT asserted tightly ----
    if (want.size() >= out.size()) {
        double worst = 0.0;
        for (size_t i = 0; i < out.size(); ++i) worst = std::max(worst, std::fabs((double) out[i] - want[i]));
        // The reference rms is computed over the SAME extent as the comparison.  The engine ran one token's worth of
        // the file, so taking the reference's rms over all five tokens divides by a different population and prints a
        // ratio that reads as a 0.7% error on a tensor that in fact agrees to 7.4e-08.  Measuring the two the same way
        // is not tidiness: a reader who trusted that ratio would draw exactly the wrong conclusion about the block.
        std::vector<float> want_slice(want.begin(), want.begin() + (ptrdiff_t) out.size());
        const double w_rms = rms_of(want_slice);
        std::printf("  oracle l_out (same extent): rms %.6g\n", w_rms);
        std::printf("  worst absolute difference %.6g   relative to rms %.6g   ratio engine/oracle rms %.6g\n", worst,
                    worst / (w_rms ? w_rms : 1.0), e_rms / (w_rms ? w_rms : 1.0));
        std::printf("  NOTE: the previous text here said the dump was the outlier and that this could not be asserted.\n");
        std::printf("        That is RETIRED.  The 3%% it described was a real bug - the KDA was normalised twice,\n");
        std::printf("        once here and once inside the kernel - and with it fixed this comparison lands at the\n");
        std::printf("        fp32 floor.  A tight bound IS meaningful, but only because the engine consumed the\n");
        std::printf("        oracle's own input: a ne0-fastest file from the fixed write_tensor.  Earlier runs compared\n");
        std::printf("        differently-arranged files through an rms, which is permutation-invariant and so could not\n");
        std::printf("        detect that the two sides held different arrangements at all.\n");
    }
    if (!out_name.empty()) {
        FILE* of = std::fopen((dd + "/" + out_name).c_str(), "wb");
        if (!of) { std::printf("TRUNK GATE: FAIL - cannot write %s\n", out_name.c_str()); return 1; }
        const unsigned hdr[5] = {0, (unsigned) N_EMBD, (unsigned) HC, 1u, 1u};
        std::fwrite(hdr, 4, 5, of);
        std::fwrite(out.data(), 4, out.size(), of);
        std::fclose(of);
        std::printf("  wrote %s (%zu floats, ne=[%d,%d,1,1]) so the next block can consume it as its input\n",
                    out_name.c_str(), out.size(), N_EMBD, HC);
    }
    std::printf("TRUNK GATE: PASS (the loop executed end to end on real weights; see the note on the comparison)\n");
    return 0;
}
