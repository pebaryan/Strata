/// The block-kind dispatch, MEASURED from the artifact rather than assumed.
///
/// TRUNK vs MTP - the finding that matters most here.  glm5next.block_count is 46 and
/// glm5next.nextn_predict_layers is 1, so the last layer is the multi-token-prediction head and the trunk is
/// FORTY-FIVE blocks (0..44).  Three independent measurements agree:
///
///   * blk.45 carries nextn.eh_proj / nextn.enorm / nextn.hnorm / nextn.shared_head_norm, and no other block does;
///   * blk.45 is the ONLY block with no hc_* tensors - the hyper-connection sites exist on 45 blocks, and a
///     prediction head has no use for them;
///   * head_count_kv has twelve 1s, but only eleven of them are trunk layers.
///
/// A loop that walked 46 blocks would have run the prediction head as a trunk block and carried its output into the
/// next layer.  That is the silent-arithmetic failure mode this port keeps meeting, so the split is asserted here.
///
/// glm5next.attention.head_count_kv is the per-layer kind, 1 = MLA and 0 = KDA, and it reads
///
///     0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 0 0 1  0 1
///
/// so the TRUNK's MLA layers are every fourth starting at 3 - eleven of them - and the final 1 belongs to the MTP
/// layer.  An earlier note in this port said "block 3 is MLA", which was true only of the first occurrence.
///
/// The FFN kind comes from glm5next.leading_dense_block_count = 3, corroborated by the generated tensor table: three
/// blocks carry a dense ffn_gate/ffn_up/ffn_down and forty-three carry ffn_gate_inp, the router.  The routed blocks
/// also carry ffn_{gate,up,down}_shexp (the shared expert) and exp_probs_b (the selection bias).
///
/// The counts are asserted at compile time.  If the table and the artifact ever disagree this fails to build rather
/// than dispatching silently - the rule the rest of this port follows, applied to a table that is easy to mistype and
/// impossible to verify by eye.
#pragma once

#include <cstdint>

namespace strata::core::glm {

inline constexpr int GLM_BLOCK_COUNT = 46;      ///< including the MTP layer
inline constexpr int GLM_TRUNK_BLOCKS = 45;     ///< blocks 0..44; glm5next.nextn_predict_layers = 1
inline constexpr int GLM_LEADING_DENSE = 3;
inline constexpr bool GLM_TRUNK_MLA[] = {
    false, false, false, true,  false, false, false, true,  false, false, false, true,
    false, false, false, true,  false, false, false, true,  false, false, false, true,
    false, false, false, true,  false, false, false, true,  false, false, false, true,
    false, false, false, true,  false, false, false, true,  false,
};

/// The MTP layer is not a trunk block: it is `nextn_predict_layers` layers at the end, and running it in the trunk
/// loop would feed a prediction head's output forward as if it were a layer's.
constexpr bool glm_is_mtp_layer(int layer) { return layer >= GLM_TRUNK_BLOCKS && layer < GLM_BLOCK_COUNT; }

/// 1 = MLA (latent attention, one kv head), 0 = KDA (the recurrent attention this port verified first), -1 = out of
/// range.  Only meaningful for trunk layers; the MTP layer's own attention is MLA, but it is not part of the trunk.
constexpr int glm_attention_is_mla(int layer) {
    return (layer >= 0 && layer < GLM_TRUNK_BLOCKS) ? (GLM_TRUNK_MLA[layer] ? 1 : 0) : -1;
}

/// The leading blocks have a dense FFN; every later trunk block routes to 288 experts, 8 used.
constexpr bool glm_ffn_is_dense(int layer) { return layer < GLM_LEADING_DENSE; }

namespace detail {
constexpr int count_mla(int i = 0, int n = 0) {
    return i >= GLM_TRUNK_BLOCKS ? n : count_mla(i + 1, n + (GLM_TRUNK_MLA[i] ? 1 : 0));
}
constexpr int count_kda(int i = 0, int n = 0) {
    return i >= GLM_TRUNK_BLOCKS ? n : count_kda(i + 1, n + (GLM_TRUNK_MLA[i] ? 0 : 1));
}
}  // namespace detail

static_assert(detail::count_mla() == 11, "the trunk's MLA layer count disagrees with head_count_kv");
static_assert(detail::count_kda() == 34, "the trunk's KDA layer count disagrees with the 34 ssm_* tensor rows");
static_assert(detail::count_mla() + detail::count_kda() == GLM_TRUNK_BLOCKS, "every trunk layer is one kind or the other");
static_assert(sizeof(GLM_TRUNK_MLA) / sizeof(GLM_TRUNK_MLA[0]) == GLM_TRUNK_BLOCKS, "the table must cover the trunk");
static_assert(GLM_BLOCK_COUNT - GLM_TRUNK_BLOCKS == 1, "one MTP layer, per nextn_predict_layers");

}  // namespace strata::core::glm
