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

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
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
    const size_t bxq = 8 * (size_t) (n_embd / 32) * 40;
    const size_t bhq = 8 * (size_t) (n_ff / 32) * 40;
    if (batch_xq == nullptr && cudaMalloc(&batch_xq, bxq) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_xq"); return false; }
    if (batch_hq == nullptr && cudaMalloc(&batch_hq, bhq) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_hq"); return false; }
    if (batch_x == nullptr && cudaMalloc((void**) &batch_x, 8 * (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_x"); return false; }
    if (batch_h == nullptr && cudaMalloc((void**) &batch_h, 8 * (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_h"); return false; }
    if (batch_gate == nullptr && cudaMalloc((void**) &batch_gate, 8 * (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_gate"); return false; }
    if (batch_up == nullptr && cudaMalloc((void**) &batch_up, 8 * (size_t) n_ff * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_up"); return false; }
    if (batch_out == nullptr && cudaMalloc((void**) &batch_out, 8 * (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_out"); return false; }
    if (batch_accum == nullptr && cudaMalloc((void**) &batch_accum, 8 * (size_t) n_embd * 4) != cudaSuccess) { err = cuda_reason("cudaMalloc batch_accum"); return false; }
    if (batch_accum && batch_accum_bytes == 0) batch_accum_bytes = 8 * (size_t) n_embd * sizeof(float);
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
    if (batch_xq != nullptr) { cudaFree(batch_xq); batch_xq = nullptr; }
    if (batch_hq != nullptr) { cudaFree(batch_hq); batch_hq = nullptr; }
    if (batch_x != nullptr) { cudaFree(batch_x); batch_x = nullptr; }
    if (batch_h != nullptr) { cudaFree(batch_h); batch_h = nullptr; }
    if (batch_gate != nullptr) { cudaFree(batch_gate); batch_gate = nullptr; }
    if (batch_up != nullptr) { cudaFree(batch_up); batch_up = nullptr; }
    if (batch_out != nullptr) { cudaFree(batch_out); batch_out = nullptr; }
    if (batch_accum != nullptr) { cudaFree(batch_accum); batch_accum = nullptr; }
    if (batch_route_out != nullptr) { cudaFree(batch_route_out); batch_route_out = nullptr; }
    if (batch_weights != nullptr) { cudaFree(batch_weights); batch_weights = nullptr; }
    batch_accum_bytes = batch_route_out_bytes = batch_weights_bytes = 0;
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

bool glm_expert_moe_device_batch_resident(const uint8_t* const* rows_device, int n_unique,
                                          const int32_t* route_slot, const float* weights, int tokens, int n_used,
                                          const kernels::NativeExpertLayout& layout, int gu_type, int d_type,
                                          int64_t n_embd, int64_t n_ff, const float* x_host, float* out_host,
                                          GlmExpertDeviceScratch& s, std::string& err) {
    if (!rows_device || !route_slot || !weights || !x_host || !out_host || n_unique <= 0 || n_used <= 0 ||
        tokens <= 0 || tokens > 2048 || n_unique > tokens * n_used) {
        err = "batched device MoE requires valid rows/routes and 1..2048 tokens";
        return false;
    }
    if (!glm_expert_layer_supported(gu_type, d_type)) {
        char buf[192];
        std::snprintf(buf, sizeof buf, "iq_mmvq dispatch does not cover (gu %d, down %d); refusing batch MoE", gu_type, d_type);
        err = buf;
        return false;
    }
    if (layout.bytes == 0 || layout.gu_row == 0 || layout.d_row == 0 || layout.up_off >= layout.bytes ||
        layout.down_off >= layout.bytes || s.n_embd != n_embd || s.n_ff != n_ff || !s.batch_xq || !s.batch_hq ||
        !s.batch_x || !s.batch_h || !s.batch_gate || !s.batch_up || !s.batch_out || !s.batch_accum ||
        !s.batch_accum || !s.stream || n_used > 64) {
        err = "batched device MoE layout or scratch is invalid";
        return false;
    }
    const size_t accum_bytes = (size_t) tokens * n_embd * sizeof(float);
    const size_t route_bytes = (size_t) tokens * n_used * n_embd * sizeof(float);
    const size_t weights_bytes = (size_t) tokens * n_used * sizeof(float);
    if (accum_bytes > s.batch_accum_bytes) {
        cudaFree(s.batch_accum); s.batch_accum = nullptr; s.batch_accum_bytes = 0;
        if (cudaMalloc((void**) &s.batch_accum, accum_bytes) != cudaSuccess) {
            err = cuda_reason("grow batch accumulator"); return false;
        }
        s.batch_accum_bytes = accum_bytes;
    }
    if (route_bytes > s.batch_route_out_bytes) {
        cudaFree(s.batch_route_out); s.batch_route_out = nullptr; s.batch_route_out_bytes = 0;
        if (cudaMalloc((void**) &s.batch_route_out, route_bytes) != cudaSuccess) {
            err = cuda_reason("grow batch route outputs"); return false;
        }
        s.batch_route_out_bytes = route_bytes;
    }
    if (weights_bytes > s.batch_weights_bytes) {
        cudaFree(s.batch_weights); s.batch_weights = nullptr; s.batch_weights_bytes = 0;
        if (cudaMalloc((void**) &s.batch_weights, weights_bytes) != cudaSuccess) {
            err = cuda_reason("grow batch route weights"); return false;
        }
        s.batch_weights_bytes = weights_bytes;
    }
    cudaStream_t stream = (cudaStream_t) s.stream;
    if (cudaMemsetAsync(s.batch_accum, 0, accum_bytes, stream) != cudaSuccess) {
        err = cuda_reason("clear batched MoE accumulator");
        return false;
    }
    std::vector<int> selected((size_t) tokens), selected_route((size_t) tokens);
    std::vector<float> input((size_t) 8 * n_embd);
    for (int u = 0; u < n_unique; ++u) {
        if (!rows_device[u]) { err = "batched device MoE contains a null expert row"; return false; }
        int count = 0;
        for (int t = 0; t < tokens; ++t) {
            for (int k = 0; k < n_used; ++k) {
                const size_t p = (size_t) t * n_used + k;
                if (route_slot[p] == u) {
                    selected[(size_t) count] = t;
                    selected_route[(size_t) count++] = (int) p;
                } else if (route_slot[p] < 0 || route_slot[p] >= n_unique) {
                    err = "batched device MoE route index is outside the expert table";
                    return false;
                }
            }
        }
        if (count == 0) continue;
        for (int begin = 0; begin < count; begin += 8) {
            const int rows = std::min(8, count - begin);
            for (int r = 0; r < rows; ++r)
                std::memcpy(input.data() + (size_t) r * n_embd,
                            x_host + (size_t) selected[(size_t) (begin + r)] * n_embd,
                            (size_t) n_embd * sizeof(float));
            if (cudaMemcpyAsync(s.batch_x, input.data(), (size_t) rows * n_embd * sizeof(float),
                                cudaMemcpyHostToDevice, stream) != cudaSuccess) {
                err = cuda_reason("upload grouped MoE activations");
                return false;
            }
            kernels::quantize_q8_1_rows(s.batch_x, rows, n_embd, s.batch_xq, stream);
            const uint8_t* row = rows_device[u];
            kernels::iq_mmvq(gu_type, row, s.batch_xq, s.batch_gate, (int) n_embd, (int) n_ff, rows, stream);
            kernels::iq_mmvq(gu_type, row + layout.up_off, s.batch_xq, s.batch_up, (int) n_embd, (int) n_ff,
                            rows, stream);
            kernels::silu_mul(s.batch_gate, s.batch_up, s.batch_h, rows * n_ff, stream);
            kernels::quantize_q8_1_rows(s.batch_h, rows, n_ff, s.batch_hq, stream);
            kernels::iq_mmvq(d_type, row + layout.down_off, s.batch_hq, s.batch_out, (int) n_ff, (int) n_embd,
                            rows, stream);
            for (int r = 0; r < rows; ++r)
                if (cudaMemcpyAsync(s.batch_route_out + (size_t) selected_route[(size_t) (begin + r)] * n_embd,
                                    s.batch_out + (size_t) r * n_embd, (size_t) n_embd * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
                    err = cuda_reason("stage grouped expert result");
                    return false;
                }
        }
    }
    if (cudaMemcpyAsync(s.batch_weights, weights, weights_bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
        err = cuda_reason("upload batch route weights"); return false;
    }
    kernels::weighted_routes(s.batch_route_out, s.batch_weights, s.batch_accum, n_embd, tokens, n_used, stream);
    if (cudaStreamSynchronize(stream) != cudaSuccess) { err = cuda_reason("grouped expert kernels"); return false; }
    if (cudaMemcpy(out_host, s.batch_accum, accum_bytes, cudaMemcpyDeviceToHost) != cudaSuccess) {
        err = cuda_reason("download batched MoE result");
        return false;
    }
    const cudaError_t last = cudaGetLastError();
    if (last != cudaSuccess) { err = cuda_reason("after grouped expert kernels"); return false; }
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
