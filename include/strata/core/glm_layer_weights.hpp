/// The name-to-field mapping for one trunk layer's weights.
///
/// Separate from glm_trunk.hpp because it is the one part of the chain that has to know the artifact's tensor NAMES,
/// and keeping that knowledge in a single small file is what makes it checkable: the names here are the generated
/// table's own spellings, and two of them do not follow the obvious pattern (`ssm_a` has no suffix at all, and
/// `ssm_dt.bias` is a bias) - exactly the kind of thing that has cost this port time before.
#pragma once

#include <string>

#include "strata/core/glm_bind.hpp"
#include "strata/core/glm_trunk.hpp"

namespace strata::core::glm {

/// Whether this port applies the MLA indexer.  It does not.
///
/// mla_forward has no indexer fields, and the reference's own note says the indexer's selection is all-tokens below
/// 8192 positions - which is where this model's prompts sit.  Declared as a constant rather than left implicit so the
/// weight filler's check on the indexer's twelve tensors is a real gate: if a layer has them and this is false, the
/// filler refuses instead of quietly dropping them, and a long prompt that needed them would say so at that line.
inline constexpr bool kGlmAppliesIndexer = false;

/// Fills one trunk layer's weights from a BOUND block.
///
/// `kda_storage` and `mla_storage` are the caller's, one pair per layer, because GlmTrunkLayerWeights holds pointers
/// into them and they must outlive the loop.  `shexp_storage` is three slots the caller owns, filled with the shared
/// expert's {gate, up, down} for a routed layer.  The routed layer's geometry, native descriptor and blob source are
/// deliberately NOT set here: they come from the pack rather than from the bound block.
///
/// Pure assignment - no arithmetic, no dequantization, no layout fixups, because the binder already does those and a
/// second implementation would be a second thing to disagree with.
bool glm_fill_layer_weights(const GlmBoundBlock& b, int layer, const kernels::glm::KdaGeometry& kda_g,
                            const kernels::glm::MlaGeometry& mla_g, kernels::glm::KdaWeights& kda_storage,
                            kernels::glm::MlaWeights& mla_storage, const float** shexp_storage,
                            GlmTrunkLayerWeights& out, std::string& err);

}  // namespace strata::core::glm
