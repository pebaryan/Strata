// src/kernels/kolibri_swa.hpp - Kolibri 1's per-layer attention helpers (branch kolibri-port).
//
// From kolibri1-llama.cpp.patch's models/kolibri1.cpp (the reference graph):
//   * 50 layers, an SSSSF interleave (4 sliding, then 1 full), stated per layer by the GGUF array
//     `attention.sliding_window_pattern` (bools; true = sliding).  The converter writes the exact
//     pattern from config.layer_types; a 5-periodic pattern is the documented default.
//   * Sliding layers: RoPE (theta 10000) and a 513-token window.  Full layers: NO positional
//     encoding at all (NoPE / RNoPE), and an unbounded causal window.
//   * QK-norm: RMSNorm over head_dim (128) applied to Q and K BEFORE RoPE, per head, one weight
//     vector shared by the heads.
//   * Sandwich norms: attn_norm -> attention -> attn_post_norm (inside the residual add), and
//     ffn_norm -> FFN -> ffn_post_norm (inside the residual add).
//
// Pure CPU, float, header-only: the trunk will call these from both the reference gate and the
// engine path, so the semantics live in exactly one place.

#pragma once

#include <cmath>
#include <cstdint>
#include <vector>

namespace strata::kernels::kolibri {

/// The per-layer attention kind, from the GGUF's `attention.sliding_window_pattern` array
/// (true = sliding).  `pattern` has `n_layers` entries; a shorter array means "periodic, period
/// n_pattern" per llama.cpp's set_swa_pattern, which Kolibri's converter never writes (it writes
/// the full array), so this helper only takes the explicit form.
inline bool layer_is_swa(const std::vector<uint8_t>& pattern, int layer) {
    return pattern.at((size_t) layer) != 0;
}

/// SWA mask: token at query position q may attend key position k iff k <= q (causal) and
/// q - k < window (the 513-token window; ggml's build_attn masks the same set for
/// LLAMA_SWA_TYPE_STANDARD).  Full-attention layers pass window <= 0 for "unbounded causal".
inline bool swa_attends(int q, int k, int window) {
    return k <= q && (window <= 0 || q - k < window);
}

/// RoPE, theta base 10000, llama.cpp's ROPE_TYPE_NEOX as kolibri1 declares it
/// (llama_model_rope_type: LLM_ARCH_KOLIBRI1 -> LLAMA_ROPE_TYPE_NEOX): within each head, dim i
/// pairs with dim i + head_dim/2 (ggml's rotate_pairs(n_dims, n_dims/2) - the rotate-half
/// convention, NOT GPT-J's interleaved (2i, 2i+1)); pair i rotates by pos * base^(-2i/d).
/// Applied to Q and K AFTER QK-norm, per token, per head.
inline void rope_neox(float* vec, int64_t pos, int heads, int head_dim, float theta_base,
                      int64_t stride) {
    const int half = head_dim / 2;
    for (int h = 0; h < heads; ++h) {
        float* head = vec + (size_t) h * head_dim;
        for (int i = 0; i < half; ++i) {
            const float freq = std::pow(theta_base, -2.0f * (float) i / (float) head_dim);
            const float angle = (float) pos * freq;
            const float c = std::cos(angle), s = std::sin(angle);
            const float x0 = head[i], x1 = head[i + half];
            head[i] = x0 * c - x1 * s;
            head[i + half] = x0 * s + x1 * c;
        }
        (void) stride;
    }
}

/// RMSNorm over one vector: x / sqrt(mean(x^2) + eps) * w.  `w` may be null (no scale).
inline void rms_norm(const float* x, const float* w, int n, float eps, float* out) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * x[i];
    const float inv = (float) (1.0 / std::sqrt(ss / n + eps));
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * (w ? w[i] : 1.0f);
}

/// QK-norm: RMSNorm over head_dim, per head, one weight vector shared by all heads.
inline void qk_norm_per_head(const float* x, const float* w, int heads, int head_dim, float eps,
                             float* out) {
    for (int h = 0; h < heads; ++h) rms_norm(x + (size_t) h * head_dim, w, head_dim, eps,
                                             out + (size_t) h * head_dim);
}

}  // namespace strata::kernels::kolibri
