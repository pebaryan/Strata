// kolibri_guard_gate - phase 2 of KOLIBRI-PORT.md (branch kolibri-port).
//
// Runs check_kolibri1_architecture against the real artifact and reports the geometry it derives.
// The pass criterion is the GLM port's phase-2 gate: the engine opens the artifact, derives the
// layer table, and everything the guard requires agrees with what the file actually holds.  It
// also refuses, with a named reason, a file that is NOT a converted Kolibri 1 (the guard's value
// is the refusal as much as the acceptance).
//
// usage: kolibri_guard_gate <kolibri-gguf> [a-non-kolibri-gguf]

#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>

#include "strata/artifact/gguf_reader.hpp"

namespace {

// the sibling-shard union, the same way report_glm5next builds it
std::set<std::string> sibling_names(const std::string& path) {
    std::set<std::string> others;
    const size_t slash = path.find_last_of('/');
    const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t of = name.rfind("-of-");
    if (of == std::string::npos || of < 6) return others;
    const int n = std::atoi(name.substr(of + 4).c_str());
    const std::string stem = name.substr(0, of - 6);
    for (int i = 2; i <= n; ++i) {
        char sib[4096];
        std::snprintf(sib, sizeof sib, "%s-%05d-of-%05d.gguf", stem.c_str(), i, n);
        others.insert(sib);
    }
    return others;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: kolibri_guard_gate <kolibri-gguf> [a-non-kolibri-gguf]\n");
        return 2;
    }
    const std::string path = argv[1];
    const std::set<std::string> others = sibling_names(path);

    strata::GgufFile g(path);
    strata::Kolibri1Geometry geo;
    const std::string err = strata::check_kolibri1_architecture(g, geo, &others);
    if (!err.empty()) {
        std::printf("  kolibri1 guard: FAIL - %s\n", err.c_str());
        return 1;
    }

    uint32_t sliding = 0;
    for (uint8_t b : geo.swa_pattern) sliding += b != 0;
    std::printf("  kolibri1 guard: PASS\n");
    std::printf("    blocks %u, hidden %llu, heads %llu/%llu kv, head_dim %llu\n",
                geo.block_count, (unsigned long long) geo.hidden, (unsigned long long) geo.head_count,
                (unsigned long long) geo.head_count_kv, (unsigned long long) geo.head_dim);
    std::printf("    experts %llu, used %llu, shared %llu, ffn %llu (+ shared %llu)\n",
                (unsigned long long) geo.experts, (unsigned long long) geo.experts_used,
                (unsigned long long) geo.shared_experts, (unsigned long long) geo.expert_ffn,
                (unsigned long long) geo.shared_ffn);
    std::printf("    attention: SWA window %u, %u sliding / %u full (NoPE), rope base %.0f\n",
                geo.swa_window, sliding, geo.block_count - sliding, geo.rope_freq_base);
    std::printf("    norms: rms eps %.1e, sandwich; every block MoE: %s (first_moe %u)\n",
                geo.rms_eps, geo.every_block_moe ? "yes" : "no", geo.first_moe);
    std::printf("    context %llu, gating 5 (sigmoid_logit_add), weights_norm 0\n",
                (unsigned long long) geo.context_length);
    std::printf("    pattern[0..9]:");
    for (int i = 0; i < 10 && i < (int) geo.swa_pattern.size(); ++i)
        std::printf(" %d", geo.swa_pattern[(size_t) i]);
    std::printf(" ...\n");

    if (argc >= 3) {
        strata::GgufFile other(argv[2]);
        strata::Kolibri1Geometry throwaway;
        const std::string refuse = strata::check_kolibri1_architecture(other, throwaway, nullptr);
        std::printf("  refusal check (%s): %s\n", argv[2],
                    refuse.empty() ? "ACCEPTED (unexpected!)" : ("refused - " + refuse).c_str());
        return refuse.empty() ? 1 : 0;
    }
    return 0;
}
