// include/strata/kernels/glm_mla.hpp - the MLA block, nope-only, with the absorption folding.
//
// GLM-5.3-Flash's sparse-attention blocks (11 of the 46) keep attention in the LATENT space: the query is
// absorbed into it with wk_b, K and V are the SAME 512-dimensional vector with a single head, and wv_b
// un-absorbs per head on the way out.  Transcribed from llama.cpp's GLM5-Next `build_mla_layer`; the
// oracle is tools/glm5_mla_reference.py and src/kernels/glm_mla_parity.cpp compares this against it.
//
//   qr      = rms_norm(wq_a @ x)                        q_lora 1536
//   q       = wq_b @ qr             -> [64][256]
//   Qcur[h] = wk_b[h]^T @ q[h]      -> [64][512]        the absorbed query
//   kv      = rms_norm(wkv_a_mqa @ x) -> [512]          one head; no rope and no k_pe anywhere
//   attn[h] = softmax_j((Qcur[h] . kv_j) / sqrt(256)) . kv_j
//   v[h]    = wv_b[h] @ attn[h]     -> [64][256]
//   out     = wo @ concat_heads(v)  -> [4096]
//
// The scale is 1/sqrt(head_dim) = 1/16, i.e. the MLA head size rather than the latent rank.
#pragma once

#include <cstdint>

namespace strata::kernels::glm {

struct MlaGeometry {
    int n_embd = 4096;
    int n_head = 64;
    int head_dim = 256;   ///< attention.key_length_mla
    int kv_lora = 512;    ///< attention.kv_lora_rank: the latent, and K and V both live here
    int q_lora = 1536;    ///< attention.q_lora_rank
};

/// Every weight as the engine reads it: C rows, ggml index order, ne0 contiguous.
struct MlaWeights {
    const float* wq_a = nullptr;       ///< [q_lora][n_embd]
    const float* q_a_norm = nullptr;   ///< [q_lora]
    const float* wq_b = nullptr;       ///< [n_head*head_dim][q_lora]
    const float* wk_b = nullptr;       ///< [n_head][kv_lora][head_dim]
    const float* kv_a = nullptr;       ///< [kv_lora][n_embd]
    const float* kv_a_norm = nullptr;  ///< [kv_lora]
    const float* wv_b = nullptr;       ///< [n_head][head_dim][kv_lora]
    const float* wo = nullptr;         ///< [n_embd][n_head*head_dim]
    /// Native device-block GGML types, or zero when the pointer above is host floats.  The MLA carried NO types at all
    /// until now, which is precisely why it had no device path: the kernel had nothing to dispatch on, while the KDA
    /// beside it has had wq_type/wk_type/wv_type/wo_type for a while.  A non-zero type means the pointer is the
    /// artifact's own quantized blocks on the device and the native projection hook reads it in place.
    int wq_a_type = 0, wq_b_type = 0, kv_a_type = 0, wo_type = 0;
};

/// Optional intermediate outputs, for a parity test to localize a mismatch.  Any pointer may be null.
struct MlaIntermediates {
    float* qr = nullptr;     ///< [q_lora]
    float* qcur = nullptr;   ///< [n_head][kv_lora]
    float* kv = nullptr;     ///< [kv_lora]
    float* attn = nullptr;   ///< [n_head][kv_lora]
};

/// The model's rms norm epsilon (glm5next.attention.layer_norm_rms_epsilon = 1e-5).
inline constexpr float MLA_RMS_EPS = 1e-5f;

/// One token through one MLA block.  `cache` is [n_cache][kv_lora] of latents the token may read, the
/// current position included (the reference appends it before attending).  `out` is [n_embd].
/// The native (device MMVQ) projection hook, the same signature the KDA already uses so one implementation serves both:
/// H2D the activations, quantize to q8_1, one native_mmvq per weight, D2H the results.  When it is not installed, or a
/// type is zero, or it fails, the stage falls back to its host GEMM - so a device fault degrades to the CPU path rather
/// than to a wrong answer.
using MlaNativeProjectFn = bool (*)(int count, const void* const* weights, const int* types,
                                    const float* x, int n_in, int n_out, float* const* out);
void mla_set_native_project(MlaNativeProjectFn fn);

void mla_forward(const MlaWeights& w, const MlaGeometry& g, const float* x, int n_cache, const float* cache,
                 float* out, const MlaIntermediates& want = {});

}  // namespace strata::kernels::glm
