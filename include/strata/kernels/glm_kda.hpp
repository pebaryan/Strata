// include/strata/kernels/glm_kda.hpp - GLM-5.3-Flash's KDA (linear attention) block operator.
//
// 34 of the model's 46 blocks are linear attention.  The recurrence is the GDN that qwen4exp already
// uses - the reference encodes the ONLY difference as `const bool kda = (neg0 == S_v)`, i.e. the decay
// is per channel rather than per head:
//
//     decay:   KDA  S[i][j] *= exp(g[i])        GDN  S[i][j] *= exp(g[0])
//     delta:   delta[i] = (v[i] - sum_j S[i][j]*k[j]) * beta
//     update:  S[i][j] += delta[i] * k[j]
//     output:  attn[i]  = (sum_j S[i][j]*q[j]) * 1/sqrt(hd)     read AFTER the update
//
// and the front end (transcribed from build_kda_layer + glm5next_causal_conv1d):
//
//     x     = rms_norm(cur, attn_norm, 1e-5)
//     q,k,v = silu(conv1d(w{q,k,v} @ x, ssm_conv1d_{q,k,v}))   depthwise, kernel 4, causal, per q/k/v
//     g     = gate_lower_bound * sigmoid(-(ssm_a * (ssm_f_b(ssm_f_a(x)) + ssm_dt.bias)))
//     beta  = sigmoid(ssm_beta @ x)                            one scalar per head
//     q, k  = l2_norm(..., 1e-6)                               hard-coded, NOT the model's rms eps
//     out   = rms_norm(attn, ssm_norm, 1e-5) * sigmoid(ssm_g_b(ssm_g_a(x)))
//
// Traps, all of them silent: the decay's g index is S's FIRST index (the query/output channel); the
// output is read after the update; the conv kernel is applied in forward order (a correlation, not a
// flip); the l2 eps is 1e-6 while every rms norm in the block uses 1e-5; and the output gate is a
// SIGMOID where qwen4exp's equivalent path uses SiLU.
#pragma once

#include <cstdint>

namespace strata::kernels::glm {

struct KdaGeometry {
    int n_embd = 4096;
    int nh = 64;        ///< attention.head_count: the head count this block uses
    int hd = 128;       ///< kda.head_dim
    int d_conv = 4;     ///< ssm.conv_kernel
    int d_inner() const { return nh * hd; }
};

/// Weights as C rows in ggml index order (ne0 contiguous).
struct KdaWeights {
    const float* attn_norm = nullptr;  ///< [n_embd]
    const float* wq = nullptr;         ///< [d_inner][n_embd]
    const float* wk = nullptr;
    const float* wv = nullptr;
    const float* conv_q = nullptr;     ///< [d_conv][d_inner]
    const float* conv_k = nullptr;
    const float* conv_v = nullptr;
    const float* ssm_a = nullptr;      ///< [nh]: holds -exp(A_log)
    const float* dt_bias = nullptr;    ///< [d_inner]
    const float* ssm_f_a = nullptr;    ///< [hd][n_embd]
    const float* ssm_f_b = nullptr;    ///< [d_inner][hd]
    const float* ssm_beta = nullptr;   ///< [nh][n_embd]
    const float* ssm_g_a = nullptr;    ///< [hd][n_embd]
    const float* ssm_g_b = nullptr;    ///< [d_inner][hd]
    const float* o_norm = nullptr;     ///< [hd], the gain of ssm_norm
    const float* wo = nullptr;         ///< [n_embd][d_inner]
    int wq_type = 0, wk_type = 0, wv_type = 0, wo_type = 0; ///< native device-block GGML types, or zero
};

using KdaNativeProjectFn = bool (*)(int count, const void* const* weights, const int* types,
                                    const float* x, int n_in, int n_out, float* const* out);
void kda_set_native_project(KdaNativeProjectFn fn);

/// The model's constants, named by the metadata key each comes from so the two eps cannot be swapped.
inline constexpr float KDA_RMS_EPS = 1e-5f;   ///< attention.layer_norm_rms_epsilon
inline constexpr float KDA_L2_EPS = 1e-6f;    ///< hard-coded in the reference
inline constexpr float KDA_GATE_LOWER = -5.0f;  ///< kda.gate_lower_bound

/// Intermediates, for the parity test and for debugging.  Any pointer may be null.
struct KdaIntermediates {
    float* xn = nullptr;    ///< [tokens][n_embd]
    float* qc = nullptr;    ///< [tokens][d_inner]  the convolved+SiLU q
    float* kc = nullptr;
    float* vc = nullptr;
    float* g = nullptr;     ///< [tokens][nh][hd]
    float* beta = nullptr;  ///< [tokens][nh]
    float* q = nullptr;     ///< [tokens][nh][hd]  after the l2 norm
    float* k = nullptr;
    float* v = nullptr;
    float* attn = nullptr;  ///< [tokens][nh][hd]  the recurrence's output, pre-norm
    float* o = nullptr;     ///< [tokens][d_inner] after the gated norm
};

/// Runs the block's operator over `tokens` inputs.  `state` is [nh][hd][hd], S[head][i][j]. `conv_state`, when
/// supplied, is the three [d_conv-1][d_inner] projection histories. Both are updated for the next decode call.
void kda_forward(const KdaWeights& w, const KdaGeometry& g, const float* x, int tokens, float* out,
                 float* state, const KdaIntermediates* mid = nullptr, float* conv_state = nullptr);

}  // namespace strata::kernels::glm
