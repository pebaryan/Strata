/// The streamed-expert MoE stage.  Separate from glm_layer.hpp so that including the chain does not pull the
/// CPU/ggml expert path, and separate from glm_moe.hpp because that one is the resident (float) path.
#pragma once

#include <cstdint>
#include <string>

#include "strata/kernels/glm_moe.hpp"

namespace strata::kernels::cpu { struct NativeFmt; }   // defined in native_expert.hpp, which this header does not need

namespace strata::core::glm {

using GlmNativeFfnFn = bool (*)(const void* const* weights, const int* types,
                                const kernels::glm::MoeGeometry& g, const float* x,
                                float* out, float clamp_limit);
void glm_set_native_ffn(GlmNativeFfnFn fn);

using GlmDeviceExpertFfnFn = bool (*)(void* ctx, int layer, int expert, const uint8_t* blob,
                                      const kernels::cpu::NativeFmt& fmt, const float* x, float* out,
                                      std::string& err);
void glm_set_device_expert_ffn(GlmDeviceExpertFfnFn fn, void* ctx);

/// Run the dense feed-forward through the native (device MMVQ) hook if one is installed and all three types are set.
/// Returns false when the caller should fall back to the host expert_ffn - a missing hook is not an error, it just means
/// this build has no device path, so the caller degrades rather than failing.
bool glm_try_native_ffn(const void* const* weights, const int* types, const kernels::glm::MoeGeometry& g,
                        const float* x, float* out, float clamp_limit);

/// Stage 4c: the MoE site for streamed, quantized experts - what the 86 GB pack actually takes.
///
/// Unlike glm_stage_moe (float pointers, resident weights), this computes from the quantized blob via ggml-cpu, which
/// is the only reason the experts can be streamed.  The router weight is applied EXACTLY ONCE, in the combine; the
/// activation is quantized once per token and never cached (a cache keyed on the activation would quantize layer 0
/// and reuse it for every remaining layer).
///
/// `blob_fn(ctx, layer, expert)` returns the expert's blob - ExpertSource::blob in production, a fixture buffer in a
/// gate - so the stage is testable without a pack.  The shared expert is added UNWEIGHTED with its own geometry and
/// clamp, both caller-supplied rather than assumed.
bool glm_stage_moe_native(const float* xn, const float* router, const float* probs_b,
                          const kernels::glm::MoeGeometry& g, int layer,
                          const kernels::cpu::NativeFmt& fmt,
                          const uint8_t* (*blob_fn)(void*, int, int), void* blob_ctx,
                          const kernels::glm::MoeGeometry* shexp_g, const float* const* shared,
                          const int* shared_types, float shexp_clamp,
                          float* out, std::string& err, int* ids_out = nullptr);

}  // namespace strata::core::glm
