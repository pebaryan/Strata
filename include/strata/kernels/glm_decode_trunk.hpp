// include/strata/kernels/glm_decode_trunk.hpp - the decode trunk's device-resident state (branch glm5next-port).
//
// For ONE decoded token the four hyper-connection residual streams stay on the GPU, and each layer's hyper-connection
// pre-mix + norm, attention block and hyper-connection post-mix chain on a single stream with no host synchronisation.  The
// only per-layer host exchange is the FFN site, which needs its input on the host (routing, expert uploads, the shared
// expert) and sends its output back.  This removes the host hc_pre (~0.15 ms x 90 per token) and the attention blocks' own
// per-layer upload/download.
//
// Everything here is asynchronous on dtrunk_stream() except the two fetch calls, which synchronise.  The arithmetic mirrors
// glm_hc.cpp (hc_pre incl. the Sinkhorn normalisation, hc_post) and glm_stage_hc_norm (the optional norm after hc_pre).
#pragma once

#include <cstddef>

#include "strata/kernels/glm_hc.hpp"

namespace strata::kernels::glm {

/// Allocates the buffers for this width (idempotent) and uploads the token's initial streams [HC][n_embd] to the device.
bool dtrunk_begin(const float* streams_host, int n_embd, char* error, size_t error_capacity);
/// The one stream every decode-trunk kernel (and the attention blocks) is launched on.
void* dtrunk_stream();

/// hc_pre for `site` (0 = attention, 1 = FFN) on the current streams (site 0) or the post-attention streams (site 1):
/// layer_in = sum_h pre[h] * stream[h], then, when `norm_w` is non-null, layer_in = rms_norm(layer_in, eps) * norm_w (the
/// glm_stage_hc_norm sequence).  The mix (pre/post/comb) is kept on the device for the matching dtrunk_hc_post.
/// `fn`, `base`, `scale`, `norm_w` are host pointers whose contents never change: each is uploaded once and kept resident.
bool dtrunk_hc_pre(int site, const float* fn, const float* base, const float* scale, const float* norm_w, float eps,
                   int n_embd, char* error, size_t error_capacity);
/// Device buffers the attention block reads / writes.
const float* dtrunk_layer_in();
float* dtrunk_attn_out();

/// hc_post for `site`: site 0 combines (attn_out, current streams) into the post-attention streams; site 1 combines
/// (ffn_out, post-attention streams) into the next streams, which then become the current streams.
bool dtrunk_hc_post(int site, int n_embd, char* error, size_t error_capacity);

/// Copy the FFN site's input (layer_in after hc_pre + norm) to the host and wait for everything queued so far.
bool dtrunk_fetch_ffn_in(float* host, int n_embd, char* error, size_t error_capacity);
/// Copy the FFN's output to the device (asynchronous; ordered before the next hc_post on the stream).
bool dtrunk_put_ffn_out(const float* host, int n_embd, char* error, size_t error_capacity);
/// Copy the current streams to the host and wait.
bool dtrunk_fetch_streams(float* host, int n_embd, char* error, size_t error_capacity);
/// Debug: synchronous copy of one device buffer (0 current streams [HC][n], 1 post-attention streams [HC][n], 2 layer_in [n],
/// 3 attn_out [n], 4 ffn_out [n]); returns the element count, or 0 on failure.  Used by STRATA_GLM_DT_VERIFY.
int dtrunk_debug_fetch(int which, float* host, int n_embd);

}  // namespace strata::kernels::glm
