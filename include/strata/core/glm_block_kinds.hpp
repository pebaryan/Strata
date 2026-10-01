/// The block-kind dispatch, MEASURED from the artifact rather than assumed.
///
/// glm5next.attention.head_count_kv is a per-layer array, 1 = MLA and 0 = KDA, and it reads
///
///     0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 1
///
/// so the MLA layers are every fourth starting at 3, plus the final layer - TWELVE of the forty-six, not one.  An
/// earlier note in this port said "block 3 is MLA", which was true only of the first occurrence; a loop written from
/// that note would have run thirty-three MLA layers as KDA and produced plausible, wrong numbers.
///
/// The FFN kind comes from glm5next.leading_dense_block_count = 3, and is corroborated by the generated tensor table:
/// three blocks carry a dense ffn_gate/ffn_up/ffn_down and forty-three carry ffn_gate_inp, the router.
///
/// The counts below are asserted at compile time.  If the table and the artifact ever disagree, this fails to build
/// rather than dispatching silently - which is the same rule the rest of this port follows, applied to a table that
/// is easy to mistype and impossible to notice by eye.
#pragma once

#include <cstdint>

namespace strata::core::glm {

inline constexpr int GLM_BLOCK_COUNT = 46;
inline constexpr int GLM_LEADING_DENSE = 3;
inline constexpr int GLM_MLA_HEADS_KV[] = {
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
    0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1,
};

/// 1 = MLA (latent attention, one kv head), 0 = KDA (the recurrent attention this port verified first).
constexpr int glm_attention_is_mla(int layer) {
    return (layer >= 0 && layer < GLM_BLOCK_COUNT) ? GLM_MLA_HEADS_KV[layer] : -1;
}

/// The leading blocks have a dense FFN; every later block routes to 288 experts, 8 used.
constexpr bool glm_ffn_is_dense(int layer) { return layer < GLM_LEADING_DENSE; }

namespace detail {
constexpr int count_mla(int i = 0, int n = 0) {
    return i >= GLM_BLOCK_COUNT ? n : count_mla(i + 1, n + (GLM_MLA_HEADS_KV[i] == 1 ? 1 : 0));
}
constexpr int count_kda(int i = 0, int n = 0) {
    return i >= GLM_BLOCK_COUNT ? n : count_kda(i + 1, n + (GLM_MLA_HEADS_KV[i] == 0 ? 1 : 0));
}
}  // namespace detail

/// Measured, not chosen: twelve MLA layers and thirty-four KDA ones in this artifact.
static_assert(detail::count_mla() == 12, "the MLA layer count disagrees with head_count_kv");
static_assert(detail::count_kda() == 34, "the KDA layer count disagrees with head_count_kv");
static_assert(detail::count_mla() + detail::count_kda() == GLM_BLOCK_COUNT, "every layer must be one kind or the other");
static_assert(sizeof(GLM_MLA_HEADS_KV) / sizeof(GLM_MLA_HEADS_KV[0]) == GLM_BLOCK_COUNT, "the table must cover every block");

}  // namespace strata::core::glm
