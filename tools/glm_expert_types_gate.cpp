// tools/glm_expert_types_gate.cpp - which expert types the DEVICE path can actually serve, per the kernels' own switches.
//
// This gate exists because of a correction: the "14 of 43 layers need no new kernel" figure was derived from iq_mmvq's
// dispatch, and iq_mmvq DOES cover type 18.  But the path the stage is being built on is native_expert_grouped, whose
// DOWN switch covers only 20, 23 and 42 - and the model's down projection is type 18 on 39 of 43 layers.  So the grouped
// kernel can serve THREE layers, not fourteen, and every one of its switches ends in std::exit(1).
//
// The gate asserts the model's real type pairs, so the asymmetry cannot quietly change.
#include "strata/core/glm_expert_types.hpp"

#include <cstdio>
#include <cstring>

using strata::core::glm_expert_down_supported;
using strata::core::glm_expert_gu_supported;
using strata::core::glm_expert_layer_supported;

namespace {
int failures = 0;
void expect(bool got, bool want, const char* what) {
    const bool ok = (got == want);
    std::printf("  %-58s %s (got %s, want %s)\n", what, ok ? "ok" : "FAIL", got ? "true" : "false",
                want ? "true" : "false");
    if (!ok) ++failures;
}
}  // namespace

int main(int argc, char** argv) {
    const bool bad = (argc > 1 && std::strcmp(argv[1], "--bad") == 0);
    const bool inv = bad;      // --bad inverts every expectation

    std::printf("  THE TYPE PAIRS THE MODEL ACTUALLY USES, against iq_mmvq's dispatch (the PER-EXPERT path):\n");
    // gate/up 16 with down 23: 3 layers.  Both halves covered.
    expect(glm_expert_layer_supported(16, 23), !inv, "gu 16 / down 23 (3 layers)  servable");
    // gate/up 16 with down 18: 11 layers.  THIS IS THE PAIR THE FIRST VERSION OF THIS HEADER GOT WRONG - it was generated
    // from the GROUPED switch, which has no case for 18, and refused a pair iq_mmvq handles.  The parity gate caught it.
    expect(glm_expert_layer_supported(16, 18), !inv, "gu 16 / down 18 (11 layers) servable on the PER-EXPERT path");
    // IQ1_S has a new MMVQ implementation and is now covered for all 28 layers.
    expect(glm_expert_layer_supported(19, 18), !inv, "gu 19 / down 18 (28 layers) servable with IQ1_S MMVQ");
    // Q2_K and Q3_K now cover the final layer.
    expect(glm_expert_layer_supported(10, 11), !inv, "gu 10 / down 11 (1 layer) servable with K-quant MMVQ");

    std::printf("\n  AND THE ASYMMETRY THAT DECIDED THE STAGE'S DESIGN - the GROUPED path's down switch:\n");
    expect(glm_expert_down_supported(18), !inv, "down 18 IS in iq_mmvq's dispatch (so per-expert works)");
    expect(strata::core::glm_grouped_down_supported(18), inv, "down 18 is NOT in the GROUPED down switch (20, 23, 42)");
    expect(strata::core::glm_grouped_down_supported(23), !inv, "down 23 IS in the grouped switch");
    expect(glm_expert_gu_supported(16) && glm_expert_gu_supported(19) && glm_expert_gu_supported(10), !inv, "per-expert gu dispatch: 16, 19, 10 yes");
    expect(glm_expert_down_supported(11), !inv, "per-expert down dispatch: Q3_K type 11 yes");

    std::printf("\n  %d expectation(s) failed%s\n", failures, bad ? " (expected: all of them)" : "");
    if (bad) {
        std::printf("  %s - the gate's ability to fail is verified\n", failures > 0 ? "PASS" : "FAIL");
        return failures > 0 ? 0 : 1;
    }
    std::printf("  %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
