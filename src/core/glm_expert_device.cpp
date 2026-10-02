// src/core/glm_expert_device.cpp - the packaged per-expert device FFN.
//
// The sequence below is the one tools/expert_parity_gate.cpp measured at 1.705-1.886% rms against the CPU native kernels
// over eight layers of two weight classes.  It is deliberately unoptimised and deliberately explicit: two iq_mmvq calls
// for gate and up, a host SiLU and multiply, a q8_1 quantisation of the hidden in DEVICE memory, and one more iq_mmvq for
// down.  Everything that could be fused is left separate because this is the version whose numbers have been measured.
//
// THE DEVICE-POINTER CONVENTION THAT COST AN HOUR: quantize_q8_1_rows takes a `const float*` that must be on the DEVICE.
// Passing host memory raises an illegal-memory-access whose report looks like a stream problem, because the destination
// argument was already a device pointer.  Both quantisation calls here upload first.
#include "strata/core/glm_expert_device.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <vector>
#include "strata/kernels/elementwise.hpp"

namespace strata {
namespace core {

namespace {

const char* cuda_reason(const char* what) {
    static char buf[256];
    std::snprintf(buf, sizeof buf, "%s: %s", what, cudaGetErrorString(cudaGetLastError()));
    return buf;
}

}  // namespace

bool GlmExpertDeviceScratch::alloc(int64_t n_embd_in, int64_t n_ff_in, std::string& err) {
    if (n_embd_in <= 0 || n_ff_in <= 0 || (n_embd_in % 32) != 0 || (n_ff_in % 32) != 0) {
        err = "scratch geometry must be positive and a multiple of 32 in both dimensions";
        return false;
    }
    if (row != nullptr && (n_embd != n_embd_in || n_ff != n_ff_in)) {
        err = "scratch holds a different geometry; release() before reallocating";
        return false;
    }
    n_embd = n_embd_in;
    n_ff = n_ff_in;
    if (stream == nullptr && cudaStreamCreate((cudaStream_t*) &stream) != cudaSuccess) {
        err = cuda_reason("cudaStreamCreate");
        return false;
    }
    const size_t xq_bytes = (size_t) (n_embd / 32) * 40;      // block_q8_1: 4 scale + 4 sum + 32 codes
    const size_t hq_bytes = (size_t) (n_ff / 32) * 40;
    if (xq == nullptr && cudaMalloc(&xq, xq_bytes) != cudaSuccess) { err = cuda_reason("cudaMalloc xq"); return false; }
    if (hq == nullptr && cudaMalloc(&hq, hq_bytes) != cudaSuccess) { err = cuda_reason("cudaMalloc hq"); return false; }
    if (x == nullptr && cudaMalloc((void**) &x, (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc x"); return false; }
    if (h == nullptr && cudaMalloc((void**) &h, (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc h"); return false; }
    if (gate == nullptr && cudaMalloc((void**) &gate, (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc gate"); return false; }
    if (up == nullptr && cudaMalloc((void**) &up, (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc up"); return false; }
    if (out == nullptr && cudaMalloc((void**) &out, (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc out"); return false; }
    if (accum == nullptr && cudaMalloc((void**) &accum, (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc accum"); return false; }
    return true;
}

void GlmExpertDeviceScratch::release() {
    if (row != nullptr) { cudaFree(row); row = nullptr; }
    if (xq != nullptr) { cudaFree(xq); xq = nullptr; }
    if (hq != nullptr) { cudaFree(hq); hq = nullptr; }
    if (x != nullptr) { cudaFree(x); x = nullptr; }
    if (h != nullptr) { cudaFree(h); h = nullptr; }
    if (gate != nullptr) { cudaFree(gate); gate = nullptr; }
    if (up != nullptr) { cudaFree(up); up = nullptr; }
    if (out != nullptr) { cudaFree(out); out = nullptr; }
    if (accum != nullptr) { cudaFree(accum); accum = nullptr; }
    if (stream != nullptr) { cudaStreamDestroy((cudaStream_t) stream); stream = nullptr; }
    n_embd = n_ff = 0;
}

bool glm_expert_ffn_device_resident(const uint8_t* blob_device, const kernels::NativeExpertLayout& layout,
                                    int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                                    const float* x_host, float* out_host,
                                    GlmExpertDeviceScratch& s, std::string& err) {
    if (blob_device == nullptr) { err = "resident expert blob is null"; return false; }
    if (x_host == nullptr || out_host == nullptr) { err = "activation or output is null"; return false; }

    // THE GUARD, and it must be this one rather than iq_supported: type 11 is claimed supported, is sized correctly, and
    // has no case in any dispatch, whose default branch is std::exit(1).
    if (!glm_expert_layer_supported(gu_type, d_type)) {
        char buf[192];
        std::snprintf(buf, sizeof buf,
                      "iq_mmvq's dispatch does not cover (gu %d, down %d); refusing rather than calling std::exit(1)",
                      gu_type, d_type);
        err = buf;
        return false;
    }
    if (s.n_embd != n_embd || s.n_ff != n_ff) { err = "scratch geometry does not match the requested geometry"; return false; }
    if (cudaMemcpy(s.x, x_host, (size_t) n_embd * 4, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = cuda_reason("upload the activation"); return false;
    }

    kernels::quantize_q8_1_rows((const float*) s.x, 1, n_embd, s.xq, s.stream);
    kernels::iq_mmvq(gu_type, blob_device, s.xq, s.gate, (int) n_embd, (int) n_ff, 1, s.stream);
    kernels::iq_mmvq(gu_type, blob_device + layout.up_off, s.xq, s.up, (int) n_embd, (int) n_ff, 1, s.stream);
    // Keep the intermediate on-device; the prior host SiLU added gate/up downloads, hidden upload, and a sync per expert.
    kernels::silu_mul((const float*) s.gate, (const float*) s.up, (float*) s.h, n_ff, s.stream);


    kernels::quantize_q8_1_rows((const float*) s.h, 1, n_ff, s.hq, s.stream);
    kernels::iq_mmvq(d_type, blob_device + layout.down_off, s.hq, s.out, (int) n_ff, (int) n_embd, 1, s.stream);
    if (cudaStreamSynchronize((cudaStream_t) s.stream) != cudaSuccess) { err = cuda_reason("down"); return false; }
    if (cudaMemcpy(out_host, s.out, (size_t) n_embd * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = cuda_reason("download the output"); return false;
    }
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) { err = cuda_reason("after the expert FFN"); return false; }
    return true;
}


bool glm_expert_moe_device_resident(const uint8_t* const* rows_device, const float* weights, int n_experts,
                                   const kernels::NativeExpertLayout& layout, int gu_type, int d_type,
                                   int64_t n_embd, int64_t n_ff, const float* x_host, float* out_host,
                                   GlmExpertDeviceScratch& s, std::string& err) {
    if (!rows_device || !weights || n_experts <= 0 || n_experts > 64 || !x_host || !out_host) {
        err = "device MoE requires rows, weights, activation, output and 1..64 experts";
        return false;
    }
    if (!glm_expert_layer_supported(gu_type, d_type)) {
        char buf[192];
        std::snprintf(buf, sizeof buf, "iq_mmvq dispatch does not cover (gu %d, down %d); refusing device MoE", gu_type, d_type);
        err = buf;
        return false;
    }
    if (layout.bytes == 0 || layout.gu_row == 0 || layout.d_row == 0 ||
        layout.up_off >= layout.bytes || layout.down_off >= layout.bytes) {
        err = "device MoE received an invalid native expert layout";
        return false;
    }
    if (s.n_embd != n_embd || s.n_ff != n_ff || !s.x || !s.xq || !s.hq || !s.h ||
        !s.gate || !s.up || !s.out || !s.accum || !s.stream) {
        err = "device MoE scratch geometry or allocation is invalid";
        return false;
    }
    cudaStream_t stream = (cudaStream_t) s.stream;
    if (cudaMemcpyAsync(s.x, x_host, (size_t) n_embd * sizeof(float), cudaMemcpyHostToDevice, stream) != cudaSuccess ||
        cudaMemsetAsync(s.accum, 0, (size_t) n_embd * sizeof(float), stream) != cudaSuccess) {
        err = cuda_reason("upload activation or clear MoE accumulator");
        return false;
    }
    kernels::quantize_q8_1_rows((const float*) s.x, 1, n_embd, s.xq, stream);
    for (int i = 0; i < n_experts; ++i) {
        if (!rows_device[i]) { err = "device MoE contains a null resident expert row"; return false; }
        const uint8_t* row = rows_device[i];
        kernels::iq_mmvq(gu_type, row, s.xq, s.gate, (int) n_embd, (int) n_ff, 1, stream);
        kernels::iq_mmvq(gu_type, row + layout.up_off, s.xq, s.up, (int) n_embd, (int) n_ff, 1, stream);
        kernels::silu_mul((const float*) s.gate, (const float*) s.up, s.h, n_ff, stream);
        kernels::quantize_q8_1_rows((const float*) s.h, 1, n_ff, s.hq, stream);
        kernels::iq_mmvq(d_type, row + layout.down_off, s.hq, s.out, (int) n_ff, (int) n_embd, 1, stream);
        kernels::scaled_add_inplace(s.accum, s.out, n_embd, weights[i], stream);
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = cuda_reason("batched expert FFNs"); return false; }
    if (cudaMemcpy(out_host, s.accum, (size_t) n_embd * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = cuda_reason("download weighted MoE result");
        return false;
    }
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) { err = cuda_reason("after batched expert FFNs"); return false; }
    return true;
}

bool glm_expert_ffn_device(const uint8_t* blob_host, const kernels::NativeExpertLayout& layout, int gu_type, int d_type,
                           int64_t n_embd, int64_t n_ff, const float* x_host, float* out_host,
                           GlmExpertDeviceScratch& s, std::string& err) {
    if (blob_host == nullptr) { err = "expert blob is null"; return false; }
    if (s.row == nullptr && cudaMalloc(&s.row, layout.bytes) != cudaSuccess) {
        err = cuda_reason("cudaMalloc row"); return false;
    }
    if (cudaMemcpy(s.row, blob_host, layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
        err = cuda_reason("upload the expert blob"); return false;
    }
    return glm_expert_ffn_device_resident((const uint8_t*) s.row, layout, gu_type, d_type, n_embd, n_ff,
                                          x_host, out_host, s, err);
}

}  // namespace core
}  // namespace strata
