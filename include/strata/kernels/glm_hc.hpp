// include/strata/kernels/glm_hc.hpp - the manifold-constrained hyper-connection (mHC).
//
// GLM-5.3-Flash keeps FOUR residual streams per token instead of one, and every attention/FFN site mixes
// them with a small learned matrix that is normalized on the manifold (Sinkhorn) between uses.  This is
// the whole of it, from llama.cpp's GLM5-Next graph:
//
//   inpL = repeat_4d(embed, n_embd, 4)                      the streams start as four copies
//   residual = inpL
//   layer_in = hc_pre(inpL, fn, scale, base) -> post, comb  what the site actually consumes
//   layer_out = attention_or_ffn(rms_norm(layer_in))
//   inpL = hc_post(layer_out, residual, post, comb)         written back into the four streams
//   ... the head mean-collapses the streams before output_norm
//
// The oracle is tools/glm5_hc_reference.py, generated from the same reference, and
// src/kernels/glm_hc_parity.cpp compares this implementation against a fixture it produces.
#pragma once

#include <cstdint>
#include <functional>

namespace strata::kernels::glm {

inline constexpr int HC = 4;                      ///< hyper_connection.count (the reference asserts 4)
inline constexpr int HC_MIX_DIM = (2 + HC) * HC;  ///< (2 + hc) * hc = 24: pre | post | the mixing matrix
inline constexpr int SINKHORN_ITERS = 20;         ///< hyper_connection.sinkhorn_iterations

/// hyper_connection.epsilon: added to every Sinkhorn divisor and to `pre` after its sigmoid.
inline constexpr float HC_EPS = 1e-6f;

/// attention.layer_norm_rms_epsilon, i.e. the model's f_norm_rms_eps, which the reference passes to the
/// graph context as norm_rms_eps and feeds to the stream rms_norm.  It is NOT hyper_connection.epsilon:
/// the two differ by 10x and using either one for both is invisible to a parity gate, because the
/// oracle and the kernel then agree with each other while differing from the reference.
inline constexpr float HC_RMS_EPS = 1e-5f;

/// The per-token mix a site needs: `pre` selects the layer input out of the streams, `post` scales what
/// the site returns, `comb` is the Sinkhorn-normalized stream-to-stream matrix.
struct HcMix {
    float pre[HC] = {0, 0, 0, 0};
    float post[HC] = {0, 0, 0, 0};
    float comb[HC][HC] = {{0}};    ///< [dst][src]
};

/// Optional worker pool for the mix rows of hc_pre (24 independent dot products of 16384 elements, ~0.4 ms serially, which
/// is ~35 ms per decoded token over 90 calls).  `fn(n, job)` runs job(0..n-1) and returns when done; it must run a call
/// made from inside one of its own jobs serially.  Results are bit-identical with or without it.
using HcParallelFor = void (*)(int n, const std::function<void(int)>& job);
void hc_set_parallel_for(HcParallelFor fn);

/// One token's streams (`x` is [HC][n_embd], row-major) -> the layer input [n_embd] and the mix.
/// `fn` is [(2+hc)*hc][hc*n_embd] as the engine reads a weight of ggml shape [hc*n_embd, (2+hc)*hc].
void hc_pre(const float* x, const float* fn, const float* base, const float* scale, int n_embd,
            float* layer_in, HcMix* mix, int n_threads = 1);

/// Writes the site's output back into the streams: out[dst] = post[dst]*site_out + sum_src comb[dst][src]*residual[src].
void hc_post(const float* site_out, const float* residual, const HcMix& mix, int n_embd, float* out,
             int n_threads = 1);

/// The head's collapse: the mean over the streams (glm5next_hc_mean).
void hc_mean(const float* x, int n_embd, float* out);

}  // namespace strata::kernels::glm
