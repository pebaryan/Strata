// include/strata/kernels/glm_indexer_device.hpp - the sparse indexer on the device (branch glm5next-port).
//
// Beyond ~2051 positions GLM-5.3-Flash's MLA layers do not attend the whole cache: an indexer scores 4-cell pools of keys and
// the layer reads the cells of the best 512 pools (top_k 2048 CELLS / kpool 4) plus the trailing partial pool.  The semantics
// are glm_indexer.hpp's (transcribed from llama.cpp's build_dsa_top_k and set_input_kpool, parity-tested on the host); this
// is the same computation on the GPU, one MLA layer's state per `layer_key` (the layer's host latent-cache pointer):
//
//   write side  every token, every MLA layer:  row[pos] = [ LayerNorm(attn_k . x) | compressor_gate . x ]  (256 floats),
//               and each pool of 4 rows becomes one pooled key (a channel-wise softmax(gate + ape)-weighted mean) once it
//               is complete.  The rows cannot be rebuilt later (x is gone), so they are written even at short context.
//   read side   per query q:  n_vis = (q+1)/4 complete pools end at or before q;  score[b] = sum_h w[h] relu(Q[h] . pooled[b]);
//               the best min(n_vis, 512) pools (ties to the lower index) expand to cells, the tail cells
//               [n_vis*4 .. q] are appended.  With n_vis <= 512 that is every cell 0..q, i.e. dense attention.
#pragma once

#include <cstddef>
#include <cstdint>

#include "strata/kernels/glm_mla.hpp"

namespace strata::kernels::glm {

inline constexpr int IDX_D = 128;                          ///< attention.indexer.key_length
inline constexpr int IDX_NH = 32;                          ///< attention.indexer.head_count
inline constexpr int IDX_KPOOL = 4;                        ///< attention.indexer.kpool
inline constexpr int IDX_TOP_K = 2048;                     ///< attention.indexer.top_k, in CELLS
inline constexpr int IDX_N_SEL = IDX_TOP_K / IDX_KPOOL;    ///< pools selected per query
inline constexpr int IDX_CELL_STRIDE = IDX_TOP_K + IDX_KPOOL;   ///< per-token cell-list capacity: 2048 + <=3 tail, or a dense run <= 2051
/// First context length (positions) at which selection differs from dense attention: n_vis = n/4 > IDX_N_SEL.
inline constexpr int IDX_SPARSE_FROM = IDX_N_SEL * IDX_KPOOL + IDX_KPOOL;   // 2052

/// Whether `w` carries the indexer's tensors as device-usable weights (three native Q8_0-class projections plus the F32 arrays).
bool idx_available(const MlaWeights& w);

/// Positions the indexer caches may hold (set once from the context size; default 32768).
void idx_set_capacity(int positions);
/// Largest token count one idx_cells_device call accepts at that capacity (its score/sort scratch is bounded).
int idx_max_tokens();

/// Write side for the tokens at positions [pos0, pos0 + T).  `d_x` is the layer's attn-normed input [T][n_embd] on the device.
/// Asynchronous on `stream`.  Returns 1 done, 0 declined (nothing done), -1 failed.
int idx_write_device(const MlaWeights& w, const MlaGeometry& g, const void* layer_key, const float* d_x, int T, int pos0, void* stream, char* error,
                     size_t error_capacity);

/// Read side for the tokens at positions [pos0, pos0 + T), T <= idx_max_tokens(): `d_x` as above, `d_qr` the normed q_a output
/// [T][q_lora].  All rows for positions < pos0 + T must already have been written.  On success `*d_cells` is [T][IDX_CELL_STRIDE]
/// int32 cell indices and `*d_ncells` [T] the used counts (device memory owned by the module, valid until the next call).
int idx_cells_device(const MlaWeights& w, const MlaGeometry& g, const void* layer_key, const float* d_x, const float* d_qr, int T, int pos0,
                     int32_t** d_cells, int32_t** d_ncells, void* stream, char* error, size_t error_capacity);

/// Debug: copy `n` positions of a layer's [key | gate] rows back to the host (256 floats each).  Returns false if unknown.
bool idx_debug_fetch_rows(const void* layer_key, int n, float* host);

// ---- VRAM reclaim -------------------------------------------------------------------------------------------------
// The device caches below grow with the context while the expert cache is sized to fill the card.  When a cudaMalloc here
// fails, the owner of the expert cache is asked to give back at least `bytes` and the allocation is retried once.
using VramReclaimFn = bool (*)(size_t bytes);
void vram_set_reclaim(VramReclaimFn fn);
/// cudaMalloc with that one retry.  Returns true on success.
bool vram_malloc(void** p, size_t bytes);

}  // namespace strata::kernels::glm
