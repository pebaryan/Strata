#pragma once
#include <cstdint>

namespace strata::kernels {
// Configure before session capture. Existing graphs retain their selected kernels.
// This experiment is off by default and does not modify the legacy router.
void native_router_set_enabled(bool enabled);
bool native_router_enabled();

// Pinned CUDA topk-moe contract for ONE token, 512 experts, top 10, softmax,
// no selection bias, lower normalization clamp 2^-14, and scale 1.
// Reads 512 finite F32 logits; writes 10 I32 IDs and 10 F32 weights. Equal
// computed probabilities select the lower expert index. All spans must be
// four-byte aligned and outputs disjoint from each other and the input.
// Requires a nonnull ordered CUDA stream. No allocation or synchronization.
/// The expert count is the caller's: upstream's kernel hardwires 512, this fork's warp layout follows the
/// artifact (512 -> 16 values per lane, 256 -> 8), which is what lets a pruned pack use the fused router.
void native_router_top10(const float* logits, int32_t* ids, float* weights, int n_expert, void* stream);
/// n_tok rows at once (logits [n, n_expert], ids/weights [n,10]); each row exactly as the single call.
void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, int n_expert,
                               void* stream);
}
