/// The streamed (quantized) MoE expert path, deliberately in its own translation unit.
///
/// glm_layer.cpp holds the chain stages and stays free of the CPU/ggml dependency; this file is the one place that
/// reaches into ggml-cpu's dot products for the native expert blobs.  Keeping them apart is not tidiness: linking the
/// CPU library into a host-only binary drags ggml and its backends in, and every stage gate in this port is host-only
/// and should stay that way.  The eventual chain target links both.
#include "strata/core/glm_moe_native.hpp"

#include <string>
#include <vector>

#include "strata/kernels/glm_moe.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/core/glm_expert_types.hpp"

namespace strata::core::glm {

namespace {
GlmNativeFfnFn g_native_ffn = nullptr;
GlmDeviceExpertFfnFn g_device_expert_ffn = nullptr;
GlmDeviceMoeFfnFn g_device_moe_ffn = nullptr;
void* g_device_moe_ctx = nullptr;
void* g_device_expert_ctx = nullptr;
}
void glm_set_native_ffn(GlmNativeFfnFn fn) { g_native_ffn = fn; }
void glm_set_device_expert_ffn(GlmDeviceExpertFfnFn fn, void* ctx) {
    g_device_expert_ffn = fn;
    g_device_expert_ctx = ctx;
}
void glm_set_device_moe_ffn(GlmDeviceMoeFfnFn fn, void* ctx) {
    g_device_moe_ffn = fn;
    g_device_moe_ctx = ctx;
}

bool glm_try_native_ffn(const void* const* weights, const int* types, const kernels::glm::MoeGeometry& g,
                        const float* x, float* out, float clamp_limit) {
    if (g_native_ffn == nullptr || types == nullptr || weights == nullptr) return false;
    if (!types[0] || !types[1] || !types[2]) return false;
    return g_native_ffn(weights, types, g, x, out, clamp_limit);
}

/// Stage 4c: the MoE site for STREAMED (quantized) experts - the path the 86 GB pack actually takes.
///
/// Not interchangeable with glm_stage_moe.  That one takes float pointers to gate/up/down and is right for resident
/// weights (fixtures, small models, every per-kernel gate in this port); this one computes from the quantized blob
/// through ggml-cpu's own dot products, which is the only reason the experts can be streamed at all.  A chain that
/// called the other one for these blocks would try to materialise 288 experts per layer.
///
/// The router weight is applied EXACTLY ONCE, in the combine at the end - the engine's own moe_route/moe_finish split
/// documents the same rule, and a per-expert helper that also scales produces a model that one layer cannot show.
/// The activation is quantized ONCE per token and deliberately not cached: a cache keyed on the activation would
/// quantise layer 0 and reuse it for every remaining layer, which is a finite, plausible, completely wrong token.
///
/// `blob_fn(ctx, layer, expert)` returns the expert's blob (ExpertSource::blob in production, a fixture buffer in a
/// gate), so this stage needs neither the expert source nor a pack to be testable.  The per-expert buffers are
/// allocated here from the descriptor's own act_bytes / h_bytes, which is what keeps sizes from being guessed.
bool glm_stage_moe_native(const float* xn, const float* router, const float* probs_b,
                          const kernels::glm::MoeGeometry& g, int layer,
                          const kernels::cpu::NativeFmt& fmt,
                          const uint8_t* (*blob_fn)(void*, int, int), void* blob_ctx,
                          const kernels::glm::MoeGeometry* shexp_g, const float* const* shared,
                          const int* shared_types, float shexp_clamp,
                          float* out, std::string& err, int* ids_out) {
    if (!xn || !router || !out || !blob_fn) {
        err = "glm_stage_moe_native: null argument";
        return false;
    }
    if (g.n_embd <= 0 || g.ff <= 0 || g.n_expert <= 0 || g.n_used <= 0) {
        err = "glm_stage_moe_native: geometry must be positive";
        return false;
    }
    if ((int64_t) fmt.n_embd != g.n_embd || (int64_t) fmt.n_ff != g.ff) {
        err = "glm_stage_moe_native: the descriptor's widths disagree with the geometry";
        return false;
    }
    if (!kernels::cpu::native_experts_available()) {
        err = "glm_stage_moe_native: this build has no ggml-cpu expert path";
        return false;
    }
    if (g.n_used > 64) {
        err = "glm_stage_moe_native: n_used is implausibly large";
        return false;
    }

    int32_t ids[64];
    float weights[64];
    kernels::glm::moe_route(router, probs_b, g, xn, ids, weights);
    // Reported for the caller's benefit: which experts were chosen is otherwise invisible, and a mixture of
    // the right magnitude from the wrong experts looks exactly like a numerical error from the outside.
    if (ids_out != nullptr) {
        for (int i = 0; i < g.n_used && i < 64; ++i) ids_out[i] = (int) ids[i];
    }

    std::vector<float> parts((size_t) g.n_used * g.n_embd);
    std::vector<uint8_t> act(fmt.act_bytes), hq(fmt.h_bytes);
    std::vector<float> ff((size_t) g.ff);
    std::vector<const uint8_t*> blobs((size_t) g.n_used);
    for (int i = 0; i < g.n_used; ++i) {
        blobs[(size_t) i] = blob_fn(blob_ctx, layer, (int) ids[i]);
        if (!blobs[(size_t) i]) {
            err = "glm_stage_moe_native: no blob for selected expert " + std::to_string(ids[i]);
            return false;
        }
    }

    bool device_combined = false;
    if (g_device_moe_ffn != nullptr && glm_expert_layer_supported(fmt.gu_type, fmt.d_type)) {
        if (!g_device_moe_ffn(g_device_moe_ctx, layer, g.n_used, ids, blobs.data(), weights,
                              fmt, xn, out, err)) {
            if (err.empty()) err = "glm_stage_moe_native: device MoE FFN failed";
            return false;
        }
        device_combined = true;
    }
    if (!device_combined) {
        kernels::cpu::native_quant_act(fmt, xn, act.data());
        for (int i = 0; i < g.n_used; ++i) {
            const uint8_t* blob = blobs[(size_t) i];
            float* expert_out = parts.data() + (size_t) i * g.n_embd;
            // Only dispatch pairs proven by iq_mmvq enter the CUDA path. The remaining expert formats stay on the
            // validated CPU path until their kernels exist; a CUDA failure for a supported pair is an error.
            if (g_device_expert_ffn != nullptr && glm_expert_layer_supported(fmt.gu_type, fmt.d_type)) {
                if (!g_device_expert_ffn(g_device_expert_ctx, layer, (int) ids[i], blob, fmt, xn, expert_out, err)) {
                    if (err.empty()) err = "glm_stage_moe_native: device expert FFN failed";
                    return false;
                }
                continue;
            }
            const void* act_p[1] = { act.data() };
            float* ff_p[1] = { ff.data() };
            kernels::cpu::native_gu_rows(fmt, blob, act_p, 1, ff_p, 0, (int) g.ff);
            kernels::cpu::native_quant_h(fmt, ff.data(), hq.data());
            const void* hq_p[1] = { hq.data() };
            float* out_p[1] = { expert_out };
            kernels::cpu::native_down_rows(fmt, blob, hq_p, 1, out_p, 0, (int) g.n_embd);
        }
    }

    // the weight is applied here, once - the per-expert work above is deliberately UNWEIGHTED
    if (!device_combined) {
        for (int j = 0; j < g.n_embd; ++j) {
            float acc = 0.0f;
            for (int i = 0; i < g.n_used; ++i) acc += parts[(size_t) i * g.n_embd + j] * weights[i];
            out[j] = acc;
        }
    }

    // the shared expert, added UNWEIGHTED (the reference adds the parallel-SiLU shared expert with no router weight)
    if (shared && shared[0] && shared[1] && shared[2] && shexp_g) {
        std::vector<float> shr((size_t) g.n_embd);
        if (shared_types && shared_types[0] && shared_types[1] && shared_types[2] && g_native_ffn) {
            const void* nw[3] = {shared[0], shared[1], shared[2]};
            if (!g_native_ffn(nw, shared_types, *shexp_g, xn, shr.data(), shexp_clamp)) {
                err = "glm_stage_moe_native: native shared expert failed";
                return false;
            }
        } else {
            kernels::glm::expert_ffn(shared[0], shared[1], shared[2], *shexp_g, xn, shr.data(), shexp_clamp);
        }
        for (int j = 0; j < g.n_embd; ++j) out[j] += shr[j];
    }
    return true;
}

}  // namespace strata::core::glm
