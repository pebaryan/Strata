// tools/expert_parity_gate.cpp - does a per-expert DEVICE expert FFN agree with the CPU native kernels on real pack data?
//
// This is the gate for the design the grouped kernel could not serve: the grouped down switch covers only 20, 23 and 42,
// while the model's down projection is type 18 on 39 of 43 layers, so the device path has to be per-expert iq_mmvq calls
// against iq_mmvq's own dispatch (which does cover 16 and 18).  Before any of that is wired into the trunk, this asks the
// only question that matters: given the SAME blob, the SAME weights and the SAME activation codes, do the two paths
// produce the same numbers?
//
// WHY THE TOLERANCE CAN BE TIGHT: both sides quantise the activation to the same Q8_1 shape (4 B scale + 4 B sum + 32
// codes = 40 B per 32 elements - block_q8_1 on the device, the CPU fmt's own format), so the difference is ACCUMULATION
// ORDER and nothing else.  A large difference is therefore a bug and not arithmetic drift, which is what makes this gate
// worth having.
//
// THE CPU SIDE IS THE REFERENCE, NOT THE TRUTH: native_gu_rows already computes silu(gate.a) * (up.a) including the
// activation, so the CPU sequence is exactly native_quant_act -> native_gu_rows -> native_quant_h -> native_down_rows.
// The device side reproduces it with two iq_mmvq calls, an explicit SiLU and multiply, and one more iq_mmvq.
//
// Usage: expert_parity_gate <pack_dir> [layer] [expert]
#include <cuda_runtime.h>

#include "strata/core/expert_source.hpp"
#include "strata/core/glm_expert_types.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace C = strata::core;

int main(int argc, char** argv) {
    const std::string pack = (argc > 1) ? argv[1] : "/home/peb/moredata/strata-pack-glm5";
    const int layer = (argc > 2) ? std::atoi(argv[2]) : 3;
    const int expert = (argc > 3) ? std::atoi(argv[3]) : 0;
    const int64_t n_embd = 4096, n_ff = 2048;

    if (!K::cpu::native_experts_available()) {
        std::fprintf(stderr, "  this build has no ggml-cpu path, so there is no CPU reference to compare against\n");
        return 2;
    }

    std::string err;
    // THE LAYOUT MUST BE LOADED BEFORE THE SOURCE IS OPENED, and this is the step the first version of this gate skipped -
    // which produced "the requested geometry does not match the loaded expert layout" with a correct-looking (46, 288).
    // open() validates the caller's numbers against the LOADED layout, and nothing had loaded one.  The runner does this
    // at line 505 -> 517 and tools/block3_gate.cpp:255 shows the exact signature: (pack, 46, N_EXPERT, err, N_EMBD, FF).
    {
        K::cpu::ExpertLayout layout;
        if (!K::cpu::expert_layout_load(pack, 46, 288, err, 4096, 2048)) {
            std::fprintf(stderr, "  expert_layout_load: %s\n", err.c_str());
            return 1;
        }
        std::printf("  layout: n_layers %lld n_expert %lld native %d first_layer %lld\n", (long long) layout.n_layers,
                    (long long) layout.n_expert, layout.native ? 1 : 0, (long long) layout.first_layer);
    }

    C::FileExpertSource src;
    if (!src.open(pack, 46, 288, err)) { std::fprintf(stderr, "  open: %s\n", err.c_str()); return 1; }
    const uint8_t* blob = src.blob(layer, expert);
    if (blob == nullptr) { std::fprintf(stderr, "  blob(%d, %d) is null\n", layer, expert); return 1; }

    // the pack's own types for this layer, from the table the earlier gates validated
    int gu_type = 16, d_type = 18;
    {
        FILE* f = std::fopen((pack + "/native_experts.txt").c_str(), "r");
        if (f == nullptr) { std::fprintf(stderr, "  cannot read native_experts.txt\n"); return 1; }
        char line[512];
        while (std::fgets(line, sizeof line, f)) {
            if (line[0] == '#') continue;
            int l = 0, gu = 0, d = 0;
            if (std::sscanf(line, "%d %d %d", &l, &gu, &d) == 3 && l == layer) { gu_type = gu; d_type = d; break; }
        }
        std::fclose(f);
    }
    std::printf("  layer %d expert %d: gu_type %d d_type %d\n", layer, expert, gu_type, d_type);
    if (!C::glm_expert_layer_supported(gu_type, d_type)) {
        std::fprintf(stderr, "  REFUSING: iq_mmvq's dispatch does not cover (gu %d, down %d).  This is the guard the\n"
                             "            stage must use - calling iq_mmvq on an uncovered type calls std::exit(1).\n",
                     gu_type, d_type);
        return 2;
    }

    const K::NativeExpertLayout L = K::native_expert_layout(gu_type, d_type, n_embd, n_ff);
    std::printf("  layout: gu_row %zu d_row %zu up_off %zu down_off %zu bytes %zu\n", L.gu_row, L.d_row, L.up_off,
                L.down_off, L.bytes);

    // a deterministic activation, so a rerun is a rerun
    std::vector<float> x((size_t) n_embd);
    for (int i = 0; i < n_embd; ++i) x[(size_t) i] = std::sin(0.0137f * (float) (i + 1)) * 2.0f + std::cos(0.0031f * (float) i);

    // ---------------- the CPU reference ----------------
    K::cpu::NativeFmt fmt;
    if (!K::cpu::native_fmt(gu_type, d_type, n_embd, n_ff, fmt, err)) {
        std::fprintf(stderr, "  native_fmt: %s\n", err.c_str());
        return 1;
    }
    std::vector<uint8_t> act(fmt.act_bytes), hq(fmt.h_bytes);
    std::vector<float> ff((size_t) n_ff), out_cpu((size_t) n_embd);
    K::cpu::native_quant_act(fmt, x.data(), act.data());
    // the API takes `const void* const*` and `float* const*` - arrays of pointers - so the addresses must be named
    // lvalues; taking &vector::data() is an rvalue and does not compile.  One row here, so arrays of one.
    const void* act_row[1] = {act.data()};
    const void* hq_row[1] = {hq.data()};
    float* ff_row[1] = {ff.data()};
    float* out_row[1] = {out_cpu.data()};
    K::cpu::native_gu_rows(fmt, blob, act_row, 1, ff_row, 0, (int) n_ff);
    K::cpu::native_quant_h(fmt, ff.data(), hq.data());
    K::cpu::native_down_rows(fmt, blob, hq_row, 1, out_row, 0, (int) n_embd);

    // ---------------- the device path, per expert ----------------
    void* dev_row = nullptr;
    void* dev_xq = nullptr;
    void* dev_hq = nullptr;
    float* dev_gate = nullptr;
    float* dev_up = nullptr;
    float* dev_out = nullptr;
    cudaStream_t stream = nullptr;
    const size_t xq_bytes = (size_t) (n_embd / 32) * sizeof(K::cpu::NativeFmt) * 0 + (size_t) (n_embd / 32) * 40;
    const size_t hq_bytes = (size_t) (n_ff / 32) * 40;
    bool ok = cudaMalloc(&dev_row, L.bytes) == cudaSuccess && cudaMalloc(&dev_xq, xq_bytes) == cudaSuccess &&
              cudaMalloc(&dev_hq, hq_bytes) == cudaSuccess && cudaMalloc((void**) &dev_gate, (size_t) n_ff * 4) == cudaSuccess &&
              cudaMalloc((void**) &dev_up, (size_t) n_ff * 4) == cudaSuccess &&
              cudaMalloc((void**) &dev_out, (size_t) n_embd * 4) == cudaSuccess && cudaStreamCreate(&stream) == cudaSuccess;
    if (!ok) { std::fprintf(stderr, "  device allocation failed: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }

    // x and h must be ON THE DEVICE: quantize_q8_1_rows's first argument is a device pointer (it is a kernel wrapper),
    // and the illegal memory access came from handing it host memory - the destination was already device, which is what
    // made the mistake look like a stream problem.  Two small uploads, one per activation.
    float* dev_x = nullptr;
    float* dev_h = nullptr;
    if (cudaMalloc((void**) &dev_x, (size_t) n_embd * 4) != cudaSuccess ||
        cudaMalloc((void**) &dev_h, (size_t) n_ff * 4) != cudaSuccess) {
        std::fprintf(stderr, "  activation upload buffers failed\n");
        return 1;
    }
    cudaMemcpy(dev_row, blob, L.bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_x, x.data(), (size_t) n_embd * 4, cudaMemcpyHostToDevice);
    K::quantize_q8_1_rows((const float*) dev_x, 1, n_embd, dev_xq, stream);
    K::iq_mmvq(gu_type, dev_row, dev_xq, dev_gate, (int) n_embd, (int) n_ff, 1, stream);
    K::iq_mmvq(gu_type, (const uint8_t*) dev_row + L.up_off, dev_xq, dev_up, (int) n_embd, (int) n_ff, 1, stream);
    cudaStreamSynchronize(stream);

    // silu(gate) * up on the host, the same step native_gu_rows does internally and native_glm_ffn does on the host
    std::vector<float> g((size_t) n_ff), u((size_t) n_ff), h((size_t) n_ff);
    cudaMemcpy(g.data(), dev_gate, (size_t) n_ff * 4, cudaMemcpyDeviceToHost);
    cudaMemcpy(u.data(), dev_up, (size_t) n_ff * 4, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n_ff; ++i) {
        const float gi = g[(size_t) i];
        h[(size_t) i] = (gi / (1.0f + std::exp(-gi))) * u[(size_t) i];
    }
    cudaMemcpy(dev_h, h.data(), (size_t) n_ff * 4, cudaMemcpyHostToDevice);
    K::quantize_q8_1_rows((const float*) dev_h, 1, n_ff, dev_hq, stream);
    K::iq_mmvq(d_type, (const uint8_t*) dev_row + L.down_off, dev_hq, dev_out, (int) n_ff, (int) n_embd, 1, stream);
    cudaStreamSynchronize(stream);

    std::vector<float> out_dev((size_t) n_embd);
    cudaMemcpy(out_dev.data(), dev_out, (size_t) n_embd * 4, cudaMemcpyDeviceToHost);
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) { std::fprintf(stderr, "  cuda: %s\n", cudaGetErrorString(last)); return 1; }

    // ---------------- the comparison ----------------
    double max_abs = 0.0, max_rel = 0.0, norm_cpu = 0.0, norm_diff = 0.0;
    int worst = -1;
    for (int i = 0; i < n_embd; ++i) {
        const double a = out_cpu[(size_t) i], b = out_dev[(size_t) i];
        const double d = std::fabs(a - b);
        norm_cpu += a * a;
        norm_diff += d * d;
        if (d > max_abs) { max_abs = d; worst = i; }
        if (std::fabs(a) > 1e-6) max_rel = std::max(max_rel, d / std::fabs(a));
    }
    const double rms_cpu = std::sqrt(norm_cpu / (double) n_embd);
    const double rms_diff = std::sqrt(norm_diff / (double) n_embd);
    std::printf("  cpu   out[0..3] %.6f %.6f %.6f %.6f   rms %.6f\n", out_cpu[0], out_cpu[1], out_cpu[2], out_cpu[3], rms_cpu);
    std::printf("  dev   out[0..3] %.6f %.6f %.6f %.6f   rms %.6f\n", out_dev[0], out_dev[1], out_dev[2], out_dev[3], rms_cpu);
    std::printf("  max |cpu - dev| %.6e at index %d ; rms difference %.6e (%.3f%% of the CPU rms)\n", max_abs, worst,
                rms_diff, 100.0 * rms_diff / (rms_cpu > 0 ? rms_cpu : 1.0));
    std::printf("  max relative difference %.6e\n", max_rel);

    const double rel_rms = rms_diff / (rms_cpu > 0 ? rms_cpu : 1.0);
    const bool pass = (rel_rms < 0.02);
    std::printf("\n  %s - the per-expert device path reproduces the CPU native kernels to %.3f%% rms\n",
                pass ? "PASS" : "FAIL", 100.0 * rel_rms);
    cudaFree(dev_row); cudaFree(dev_xq); cudaFree(dev_hq); cudaFree(dev_gate); cudaFree(dev_up); cudaFree(dev_out);
    cudaStreamDestroy(stream);
    return pass ? 0 : 1;
}
