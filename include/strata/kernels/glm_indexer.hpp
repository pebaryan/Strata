// include/strata/kernels/glm_indexer.hpp - the sparse indexer: the 4:1 key pool and the cell selection.
//
// GLM-5.3-Flash's MLA blocks do not attend the whole cache: an indexer pools every 4 consecutive keys
// into one, scores the pools for the current token, and lets attention read the cells of the best
// min(n_pool, top_k/4) pools.  Transcribed from llama.cpp's GLM5-Next `build_dsa_top_k` plus the pool
// inputs filled in `llama_memory_hybrid_idx_context::set_input_kpool`; the oracle is
// tools/glm5_indexer_reference.py and src/kernels/glm_indexer_parity.cpp compares this against it.
//
//   write side   cache row = [ LayerNorm(attn_k @ x) (d) | compressor_gate @ x (d) ]
//   read side    pooled[b][ch] = sum_m softmax_over_m(gate + ape)[m][ch] * key[m][ch]
//                q[head]       = attn_q_b @ qr,  wts[head] = (proj @ cur)[head] / sqrt(d*nh)
//                score[b]      = sum_head wts[head] * relu(q[head] . pooled[b]) + bias[b]
//                bias[b]       = 0 if the pool is complete AND its last member is at/before the query,
//                                else -inf
//                select        = the best n_sel = min(n_pool, top_k/kpool) POOLS, expanded to their
//                                cells, plus the trailing incomplete pool's cells (always)
//
// Two behaviours the reference documents as silent-wrong if missed:
//   * the cut is on WHOLE POOLS, never on cells: relu sends many pools to exactly 0.0 and top_k is
//     unordered among equal keys, so cutting on cells splits pools apart (PR #27754);
//   * an incomplete pool has no pool key and can never win on score, so the tail cells are ALWAYS
//     appended rather than consuming pool budget.
#pragma once

#include <cstdint>

namespace strata::kernels::glm {

struct IdxGeometry {
    int n_embd = 4096;
    int q_lora = 1536;
    int d = 128;        ///< attention.indexer.key_length
    int nh = 32;        ///< attention.indexer.head_count
    int kpool = 4;      ///< attention.indexer.kpool: the pool size, called r in the reference
    int top_k = 2048;   ///< attention.indexer.top_k: CELLS, so n_sel = min(n_pool, top_k / kpool)
};

/// Weights as the engine reads them (C rows, ggml index order with ne0 contiguous).
struct IdxWeights {
    const float* attn_k = nullptr;    ///< [d][n_embd]
    const float* k_norm_w = nullptr;  ///< [d]
    const float* k_norm_b = nullptr;  ///< [d]
    const float* attn_q_b = nullptr;  ///< [nh*d][q_lora]
    const float* proj = nullptr;      ///< [nh][n_embd]
    const float* c_gate = nullptr;    ///< [d][n_embd]
    const float* ape = nullptr;       ///< [kpool][d]: the intra-pool position bias
};

/// The model's layer-norm epsilon, used by the indexer's key norm (LLM_NORM, weight AND bias).
inline constexpr float IDX_NORM_EPS = 1e-5f;

struct IdxResult {
    int n_sel = 0;    ///< whole pools selected
    int n_cells = 0;  ///< cells the token may read (the selected pools' members plus the tail)
};

/// Write side: one cache row [key(d) | gate(d)] for the token state x.
void idx_cache_row(const IdxWeights& w, const IdxGeometry& g, const float* x, float* row_out);

/// Read side.  `cache` is [n_positions][2*d] of rows written by idx_cache_row.  `q_pos` is the query's
/// position.  `cells_out` receives the selected cell indices, sorted, and must hold
/// min(n_positions, n_sel*kpool + kpool-1) entries.  The optional outputs exist for the parity test.
IdxResult idx_select(const IdxWeights& w, const IdxGeometry& g, const float* cur, const float* qr,
                     int n_positions, const float* cache, int q_pos, int32_t* cells_out, int cells_capacity,
                     float* pooled_out = nullptr, float* score_out = nullptr, float* bias_out = nullptr);

}  // namespace strata::kernels::glm
