// include/strata/core/glm_expert_device.hpp - the per-expert DEVICE expert FFN, packaged.
//
// This is the sequence tools/expert_parity_gate.cpp validated on the real pack: 1.705-1.886% rms against the CPU native
// kernels across eight layers of both weight classes, with the difference measured to be the ACTIVATION FORMAT (the CPU
// side quantises to Q8_K super-blocks, the device to Q8_1 32-element blocks) rather than accumulation order.
//
// It is packaged as a function rather than left inline in the gate so that the gate tests THE FUNCTION and not a copy of
// it - a gate that tests a duplicate is a gate that can pass while the shipped code differs, which is this port's
// recurring failure mode wearing one more hat.
//
// NO CUDA HEADER IS INCLUDED HERE: the scratch holds void* and float*, so this header stays usable from translation units
// that have nothing to do with the device.
//
// THE GUARD IS INSIDE THE FUNCTION, not left to the caller.  iq_mmvq's dispatch ends in `default: std::exit(1)`, so an
// uncovered type pair is a process death rather than an error return, and glm_expert_layer_supported is the only safe
// check (not iq_supported, which blesses type 11, which has no case anywhere).
#pragma once

#include "strata/core/glm_expert_types.hpp"
#include "strata/kernels/iq_kernels.hpp"

#include <cstdint>
#include <string>

namespace strata {
namespace core {

/// Device buffers for one expert's FFN, sized once per (n_embd, n_ff) and reused across experts and layers.
///
/// The caller owns the lifetime.  n_embd and n_ff are recorded so that alloc() can refuse a geometry change rather than
/// silently reuse undersized buffers - the failure that looks like a correct run producing wrong numbers.
struct GlmExpertDeviceScratch {
    void* row = nullptr;        ///< L.bytes of the expert's blob, uploaded per expert
    void* xq = nullptr;         ///< (n_embd/32)*40   q8_1 of the activation
    void* hq = nullptr;         ///< (n_ff/32)*40     q8_1 of the hidden
    float* x = nullptr;         ///< n_embd floats    the activation, on the device
    float* h = nullptr;         ///< n_ff floats      the hidden, on the device
    float* gate = nullptr;      ///< n_ff floats
    float* up = nullptr;        ///< n_ff floats
    float* out = nullptr;       ///< n_embd floats
    void* stream = nullptr;     ///< cudaStream_t
    int64_t n_embd = 0, n_ff = 0;

    /// Allocates every buffer and the stream.  False with a reason on any failure, in which case release() is safe.
    bool alloc(int64_t n_embd, int64_t n_ff, std::string& err);
    /// Frees whatever was allocated.  Idempotent.
    void release();
};

/// One expert's full FFN on the device: silu(gate.x) * (up.x) then down, from a blob and an activation.
///
/// `blob_host` is the expert's bytes - exactly what ExpertSource::blob(layer, expert) returns - and `L` is what
/// native_expert_layout returned for the same type pair and geometry, so the in-blob offsets are the kernel's own rather
/// than a re-derivation.  `x_host` is n_embd floats, `out_host` receives n_embd floats.
///
/// Returns false with a reason on an uncovered type pair, on a CUDA failure, or on a geometry that disagrees with the
/// scratch. The router weights are deliberately NOT applied here: glm_stage_moe_native combines the unweighted expert
/// outputs with the router weights after this call returns.
bool glm_expert_ffn_device(const uint8_t* blob_host, const kernels::NativeExpertLayout& layout, int gu_type, int d_type,
                           int64_t n_embd, int64_t n_ff, const float* x_host, float* out_host,
                           GlmExpertDeviceScratch& scratch, std::string& err);

/// Same FFN when the expert blob is already resident on the device. The caller owns blob_device; this is the entry
/// point used by the routed-stage cache so cache hits do not copy the row over PCIe again.
bool glm_expert_ffn_device_resident(const uint8_t* blob_device, const kernels::NativeExpertLayout& layout,
                                   int gu_type, int d_type, int64_t n_embd, int64_t n_ff,
                                   const float* x_host, float* out_host,
                                   GlmExpertDeviceScratch& scratch, std::string& err);

}  // namespace core
}  // namespace strata
