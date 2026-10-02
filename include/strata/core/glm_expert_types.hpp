// include/strata/core/glm_expert_types.hpp - WHICH expert types the device kernels actually implement.
//
// THIS EXISTS BECAUSE iq_supported IS NOT A SAFE GUARD.  Three of the types this port needs are handled as follows:
//
//   type 19 (IQ1_S, the gate/up of 28 of 43 layers)   iq_supported false, iq_row_bytes 0, no kernel
//   type 10 (Q2_K, the gate/up of 1 layer)            iq_supported false, iq_row_bytes 0, no kernel
//   type 11 (Q3_K)                                    iq_supported TRUE, iq_row_bytes CORRECT, NO KERNEL CASE
//
// and every one of those dispatch switches ends in `default: ... std::exit(1)`.  So a guard written as
// `if (iq_supported(t))` passes for type 11 and TERMINATES THE PROCESS, and a guard written as "not supported, fall back"
// is right for 19 and 10 but would never have noticed 11.  The lists below are transcribed from the switch statements
// themselves, not from documentation, and the gate beside this header checks them against the live kernels.
//
// The model's own types, for reference (tools/check_expert_rows.py):
//   gate/up : 19 on 28 layers, 16 on 14, 10 on 1        down : 18 on 39 layers, 23 on 3, 11 on 1
// so the device path can serve every expert of 14 of the 43 layers today (16/18 and 16/23), and the rest need kernels.
#pragma once

namespace strata {
namespace core {

/// Does a native_expert_grouped gate/up dispatch case exist for this type?
inline bool glm_expert_gu_supported(int ggml_type) {
    switch (ggml_type) {
        case 16:
        case 17:
        case 18:
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

/// Does a native_expert_grouped down dispatch case exist for this type?
inline bool glm_expert_down_supported(int ggml_type) {
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
