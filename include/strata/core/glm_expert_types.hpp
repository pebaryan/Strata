// include/strata/core/glm_expert_types.hpp - WHICH expert types the device kernels actually implement.
//
// The model's expert types are now fully covered by per-expert iq_mmvq: IQ1_S (19), Q2_K (10), and Q3_K (11)
// were added to the dispatch with measured row-byte gates and real-pack CPU/device parity tests. This helper records
// the explicit dispatch lists so future formats cannot be enabled by row-size support alone.
//
// The grouped native_expert_grouped kernel remains a distinct, narrower API; its down switch is still only 20, 23, 42.
// A capability must always be scoped to the entrypoint that will actually run.
//
// // The model's own types, for reference (tools/check_expert_rows.py):
//   gate/up : 19 on 28 layers, 16 on 14, 10 on 1        down : 18 on 39 layers, 23 on 3, 11 on 1
// so the per-expert device path can now serve all 43 routed layers; the grouped path remains narrower.
#pragma once

namespace strata {
namespace core {

/// Does iq_mmvq have a gate/up dispatch case for this type?
inline bool glm_expert_gu_supported(int ggml_type) {
    switch (ggml_type) {
        case 10:
        case 16:
        case 17:
        case 18:
        case 19:
        case 21:
        case 22:
        case 23:
        case 29:
        case 42:

            return true;
        default:
            return false;
    }
}

/// iq_mmvq's down dispatch - the PER-EXPERT path, and the set a per-expert stage must guard on.
///
/// CORRECTION: the first version of this function was generated from native_expert_grouped's DOWN switch, which covers
/// only 20, 23 and 42, and it therefore refused (16, 18) - the pair that 39 of this model's 43 layers use and that iq_mmvq
/// handles perfectly well.  tools/expert_parity_gate.cpp caught it by refusing to run, before any stage code depended on
/// it.  Both switches are now recorded, separately and honestly: this one is iq_mmvq's.
inline bool glm_expert_down_supported(int ggml_type) {
    switch (ggml_type) {
        case 11:
        case 16:
        case 17:
        case 18:
        case 20:
        case 21:
        case 22:
        case 23:
        case 29:
        case 42:
            return true;
        default:
            return false;
    }
}

/// native_expert_grouped's DOWN dispatch is only 20, 23 and 42 - so the GROUPED path serves three of this model's 43
/// layers.  Kept beside the per-expert list because the difference between them is what decided the stage's design.
inline bool glm_grouped_down_supported(int ggml_type) {
    switch (ggml_type) {
        case 20:
        case 23:
        case 42:
            return true;
        default:
            return false;
    }
}

/// Both projections of a layer servable on the device?  A layer is either entirely servable or not, because the type is
/// a property of the layer and not of the individual expert.
inline bool glm_expert_layer_supported(int gu_type, int d_type) {
    return glm_expert_gu_supported(gu_type) && glm_expert_down_supported(d_type);
}

}  // namespace core
}  // namespace strata
