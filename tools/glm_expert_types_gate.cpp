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

    std::printf("  THE THREE TYPE PAIRS THE MODEL ACTUALLY USES, against the GROUPED kernel's dispatch:\n");
    // gate/up 16 with down 23: 3 layers.  Both halves have grouped cases, so this pair IS servable.
    expect(glm_expert_layer_supported(16, 23), !inv, "gu 16 / down 23 (3 layers)  servable");
    // gate/up 16 or 19 with down 18: 39 layers, and there is NO grouped down case for 18.
    expect(glm_expert_layer_supported(16, 18), inv, "gu 16 / down 18 (11 layers) NOT servable: no down case for 18");
    expect(glm_expert_layer_supported(19, 18), inv, "gu 19 / down 18 (28 layers) NOT servable: neither half");
    // gate/up 10 with down 11: 1 layer, neither half.
    expect(glm_expert_layer_supported(10, 11), inv, "gu 10 / down 11 (1 layer)  NOT servable: neither half");

    std::printf("\n  THE INDIVIDUAL LISTS, so a reader can see the asymmetry rather than infer it:\n");
    expect(glm_expert_gu_supported(16) && glm_expert_gu_supported(19) == false, !inv, "gu: 16 yes, 19 no");
    expect(glm_expert_down_supported(18), inv, "down: 18 (39 of 43 layers) IS supported by the grouped kernel");
    expect(glm_expert_down_supported(23), !inv, "down: 23 yes");
    expect(glm_expert_down_supported(11), inv, "down: 11 no");

    std::printf("\n  %d expectation(s) failed%s\n", failures, bad ? " (expected: all of them)" : "");
    if (bad) {
        std::printf("  %s - the gate's ability to fail is verified\n", failures > 0 ? "PASS" : "FAIL");
        return failures > 0 ? 0 : 1;
    }
    std::printf("  %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
