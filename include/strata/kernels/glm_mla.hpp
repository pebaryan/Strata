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
#include <cstddef>
#include <functional>

namespace strata::kernels::glm {

struct MlaGeometry {
    int n_embd = 4096;
    int n_head = 64;
    int head_dim = 256;   ///< attention.key_length_mla
    int kv_lora = 512;    ///< attention.kv_lora_rank: the latent, and K and V both live here
    int q_lora = 1536;    ///< attention.q_lora_rank
    /// GLM-4.7-Flash (deepseek2) decoupled RoPE: the head split is head_dim = (head_dim - n_rot) nope dims
    /// followed by n_rot rope dims, and the query/key rope halves carry their own projection.  DEFAULT 0 keeps
    /// the GLM-5.3 nope-only graph bit for bit (skip the whole rope branch).  When it is non-zero, `cache` rows
    /// are (kv_lora + n_rot) wide: kv_lora latent dims then n_rot roped k_pe dims.
    int n_rot = 0;        ///< attention.rope.dimension_count
};

/// Every weight as the engine reads it: C rows, ggml index order, ne0 contiguous.
struct MlaWeights {
    const float* wq_a = nullptr;       ///< [q_lora][n_embd]
    const float* q_a_norm = nullptr;   ///< [q_lora]
    const float* wq_b = nullptr;       ///< [n_head*head_dim][q_lora]
    const float* wk_b = nullptr;       ///< [n_head][kv_lora][head_dim]
    const float* kv_a = nullptr;       ///< [kv_lora + n_rot][n_embd]: the MQA projection; the tail n_rot rows are wkv_a_mqa's k_pe
    const float* kv_a_norm = nullptr;  ///< [kv_lora]
    const float* wv_b = nullptr;       ///< [n_head][head_dim][kv_lora]
    const float* wo = nullptr;         ///< [n_embd][n_head*head_dim]
    /// Decoupled-RoPE base (deepseek2.rope.freq_base = 1e6 for GLM-4.7-Flash).  Read only when g.n_rot != 0.
    float rope_freq_base = 1000000.0f;
    /// Native device-block GGML types, or zero when the pointer above is host floats.  The MLA carried NO types at all
    /// until now, which is precisely why it had no device path: the kernel had nothing to dispatch on, while the KDA
    /// beside it has had wq_type/wk_type/wv_type/wo_type for a while.  A non-zero type means the pointer is the
    /// artifact's own quantized blocks on the device and the native projection hook reads it in place.
    int wq_a_type = 0, wq_b_type = 0, kv_a_type = 0, wo_type = 0;

    /// The sparse indexer's tensors (null when the pack has none).  Beyond ~2051 positions the model reads only the best 512
    /// four-cell pools plus the tail, chosen by this indexer; see glm_indexer.hpp.  The three projections are native (Q8_0)
    /// device blocks like the weights above, the rest F32.
    const float* idx_attn_k = nullptr;     ///< [128][n_embd]
    const float* idx_attn_q_b = nullptr;   ///< [32*128][q_lora]
    const float* idx_c_gate = nullptr;     ///< [128][n_embd]  (indexer_compressor_gate)
    const float* idx_k_norm_w = nullptr;   ///< [128]
    const float* idx_k_norm_b = nullptr;   ///< [128]
    const float* idx_proj = nullptr;       ///< [32][n_embd]
    const float* idx_ape = nullptr;        ///< [4][128]  (indexer_compressor_ape)
    int idx_attn_k_type = 0, idx_attn_q_b_type = 0, idx_c_gate_type = 0;
};

/// Optional intermediate outputs, for a parity test to localize a mismatch.  Any pointer may be null.
struct MlaIntermediates {
    float* qr = nullptr;     ///< [q_lora]
    float* qcur = nullptr;   ///< [n_head][kv_lora]   the absorbed q_nope (n_rot>0) or the absorbed q (n_rot==0)
    float* kv = nullptr;     ///< [kv_lora]
    float* attn = nullptr;   ///< [n_head][kv_lora]
    float* v = nullptr;      ///< [n_head][head_dim]  the per-head un-absorbed values (before wo)
    /// Decoupled-RoPE intermediates, filled only when g.n_rot != 0.
    float* q_nope = nullptr; ///< [n_head][head_dim - n_rot]  q split before absorption
    float* q_pe = nullptr;   ///< [n_head][n_rot]             q rope half, AFTER RoPE
    float* k_pe = nullptr;   ///< [n_rot]                     k_pe, AFTER RoPE
    /// When set, mla_forward returns right after producing kv/k_pe - the caller only wants the new cache
    /// row (e.g. to append it before a second, full call).  Skips the absorption, attention and wo, so a
    /// "harvest the latent" pass costs a fraction of a full one.
    bool kv_only = false;
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
void mla_set_device_attention(bool enabled);
bool mla_attention_cuda(const float* qcur, const float* cache, int n_cache, int n_head, int head_dim, int kv_lora,
                        float* attn, char* error, size_t error_capacity);

bool mla_head_matvec_cuda(const float* weights, const float* x, int n_head, int rows, int cols, float* y,
                          char* error, size_t error_capacity);

/// Prompt-chunk MLA.  The batched projection hook has the KDA's signature (several weights, `tokens` rows of x, one
/// shared n_in/n_out); `parallel_for` runs the independent per-token host work (row norms) on a pool.
using MlaNativeProjectBatchFn = bool (*)(int count, const void* const* weights, const int* types, int tokens,
                                         const float* x, int n_in, int n_out, float* const* out);
using MlaParallelFor = void (*)(int n, const std::function<void(int)>& job);
void mla_set_native_project_batch(MlaNativeProjectBatchFn fn);
void mla_set_parallel_for(MlaParallelFor fn);

/// For a block of S tokens: absorb the K up-projection into q, causal attention over cache rows [0, c_base + t] for token t,
/// then un-absorb V.  `q` is [S][n_head*head_dim], `cache` the host latent cache whose rows [0, c_base + S) are filled, and
/// `v_out` is [S][n_head*head_dim].  The latent cache is kept resident on the device across calls.
bool mla_attend_batch_cuda(const float* wk_b, const float* wv_b, const float* q, const float* cache, int c_base, int S,
                           int n_head, int head_dim, int kv_lora, float* v_out, char* error, size_t error_capacity);

/// The same, plus the sparse indexer: `x` ([S][n_embd], the attn-normed input) and `qr` ([S][q_lora], the normed q_a output) feed it.
/// The tokens' indexer rows are written, and once the context reaches 2052 positions each token attends only the cells the indexer
/// selects.  `w` supplies wk_b/wv_b and the indexer weights; without indexer weights this is mla_attend_batch_cuda.
bool mla_attend_batch_idx_cuda(const MlaWeights& w, const MlaGeometry& g, const float* x, const float* qr, const float* q,
                               const float* cache, int c_base, int S, float* v_out, char* error, size_t error_capacity);

/// Runs T prompt tokens through one MLA block, causally, appending their latents to `cache` at rows [c0, c0 + T).
/// `x` is [T][n_embd] (already attn-normed), `out` is [T][n_embd].  Returns false, leaving the cache rows possibly
/// written but harmless, when the batched path is not available (no hooks, a type without a native projection, a context
/// beyond 8192 positions without the device indexer, a device failure) - the caller then runs mla_forward token by token.
bool mla_forward_batch(const MlaWeights& w, const MlaGeometry& g, const float* x, int T, int c0, float* cache, float* out);

/// Serve-mode decode: one token through the whole MLA layer on the device - q_a, norm, q_b, absorb, kv_a, norm, append to the
/// resident latent cache, causal attention, un-absorb, wo - with one upload and one download (plus the new latent row, so the
/// host cache stays complete).  Needs native projection types.  STRATA_GLM_MLA_BLOCK=0 turns it off in the runner.
void mla_set_device_block(bool enabled);
bool mla_device_block_enabled();
/// 1 = done; 0 = declined with nothing touched (the host path must run); -1 = failed after work began.
/// `x` is the attn-normed input, `kv_row_host` receives the new latent row (cache row n_cache-1).
int mla_block_decode_cuda(const MlaWeights& w, const MlaGeometry& g, const float* x, int n_cache, const float* cache,
                          float* kv_row_host, float* out, char* error, size_t error_capacity);
/// The same block with the (attn-normed) input already on the device and the result left there, launched on `stream` with no
/// copies of either and no synchronisation, so a caller can chain further device work (the decode trunk).  Same return codes.
int mla_block_launch_cuda(const MlaWeights& w, const MlaGeometry& g, const float* d_x, int n_cache, const float* cache,
                          float* kv_row_host, float* d_out, void* stream, char* error, size_t error_capacity);

/// `pos` is the current token's position (for the decoupled RoPE); a negative value means n_cache - 1, the row the
/// caller is expected to have appended.  It is ignored when g.n_rot == 0.
void mla_forward(const MlaWeights& w, const MlaGeometry& g, const float* x, int n_cache, const float* cache,
                 float* out, const MlaIntermediates& want = {}, int pos = -1);

}  // namespace strata::kernels::glm
