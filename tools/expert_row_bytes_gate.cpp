// tools/expert_row_bytes_gate.cpp - does iq_row_bytes reproduce the expert sizes the pack actually contains?
//
// The device expert path is about to upload expert rows and hand them to native_mmvq.  Everything it does depends on
// iq_row_bytes(type, n_in) * n_out being the number of bytes the pack holds for that projection, and on iq_supported
// telling the truth about which kernels exist.  Neither is checked by the fact that the engine still runs: the routed
// experts have never gone through the native path, so this bridge has never been exercised.
//
// THE EXPECTED SIZES ARE MEASURED, NOT DERIVED FROM iq_row_bytes ITSELF.  They come from tools/check_expert_rows.py,
// which reproduces every layer's blob_bytes from the pack's own table (16 -> 66, 18 -> 98, 19 -> 50, 23 -> 136,
// 10 -> 84, 11 -> 110 bytes per 256 weights, all confirmed against the GGUF header for all 43 layers).  Comparing
// iq_row_bytes against a number it computed would be circular; comparing it against the pack is not.
//
// A GATE MUST BE SHOWN TO FAIL.  Pass --bad and it checks deliberately wrong expectations instead, and exits 0 only if
// the gate reports FAIL - so the gate's ability to fail is itself verified rather than assumed.
#include "strata/kernels/iq_kernels.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace {

struct Case {
    int type;
    long long n_in;
    long long n_out;
    unsigned long long expected;
    const char* what;
    bool supported_expected;
};

}  // namespace

int main(int argc, char** argv) {
    const bool bad = (argc > 1 && std::strcmp(argv[1], "--bad") == 0);

    const std::vector<Case> cases = {
        // type, n_in, n_out, bytes the pack holds for that projection, what it is, whether iq_supported should say yes
        {16, 4096, 2048, 2162688ull, "IQ2_XXS gate/up  (4096->2048)", true},
        {19, 4096, 2048, 1638400ull, "IQ1_S   gate/up  (4096->2048)", false},
        {10, 4096, 2048, 2752512ull, "Q2_K    gate/up  (4096->2048)", false},
        {18, 2048, 4096, 3211264ull, "IQ3_XXS down     (2048->4096)", true},
        {23, 2048, 4096, 4456448ull, "IQ4_XS  down     (2048->4096)", true},
        // TYPE 11 IS THE TRAP THIS GATE EXISTS FOR.  iq_supported(11) returns TRUE and iq_row_bytes(11) returns the
        // right size, but iq_mmvq's switch has no case 11 and its default calls std::exit(1) - so the guard a caller
        // would naturally write (if (iq_supported(type)) iq_mmvq(...)) passes the check and then TERMINATES THE PROCESS.
        // The expectation here is therefore the library's own claim (true), and the inconsistency is reported as a
        // warning below rather than hidden inside a pass: this gate cannot call iq_mmvq(11) to prove it, because doing
        // so would exit the gate itself.
        {11, 2048, 4096, 3604480ull, "Q3_K    down     (2048->4096)", true},
    };

    int failures = 0;
    std::printf("  %-34s %-8s %12s %12s  %s\n", "case", "support", "iq_row*b", "expected", "verdict");
    for (const Case& c : cases) {
        const unsigned long long got = (unsigned long long) strata::kernels::iq_row_bytes(c.type, c.n_in) * (unsigned long long) c.n_out;
        const bool supported = strata::kernels::iq_supported(c.type);
        const unsigned long long want = bad ? c.expected + 1ull : c.expected;     // the known-bad input
        const bool size_ok = (got == want);
        const bool supp_ok = bad ? (supported != c.supported_expected) : (supported == c.supported_expected);
        if (!size_ok || !supp_ok) ++failures;
        std::printf("  %-34s %-8s %12llu %12llu  %s%s\n", c.what, supported ? "yes" : "no", got, want,
                    size_ok ? "" : "SIZE," , supp_ok ? "ok" : (size_ok ? "SUPPORT" : "SIZE+SUPPORT"));
    }
    // units matter here and I got them wrong the first time: gu_row and d_row are BYTES PER ROW (1,056 and 784), while
    // up_off and down_off are whole-projection byte offsets inside the blob - up_off is n_ff rows of the gate, down_off
    // is that twice over, and bytes is the three concatenated, which is exactly the pack's blob_bytes.  The first run
    // compared a per-row figure against a whole-projection figure and reported the helper as wrong when it was not.
    std::printf("\n  native_expert_layout, for the two type pairs that matter (gu_row/d_row are BYTES PER ROW):\n");
    const struct { int gu, d; unsigned long long gu_row, d_row, bytes; const char* what; } layouts[] = {
        {16, 18, 1056ull, 784ull, 7536640ull, "IQ2_XXS / IQ3_XXS - 14 layers"},
        {19, 18, 0ull, 784ull, 3211264ull, "IQ1_S / IQ3_XXS   - 28 layers"},
    };
    for (const auto& L : layouts) {
        const strata::kernels::NativeExpertLayout lay = strata::kernels::native_expert_layout(L.gu, L.d, 4096, 2048);
        const bool rows_ok = ((unsigned long long) lay.gu_row == L.gu_row) && ((unsigned long long) lay.d_row == L.d_row);
        const bool bytes_ok = ((unsigned long long) lay.bytes == L.bytes);
        const unsigned long long gate_total = L.gu_row * 2048ull;                 // n_ff rows of the gate
        const bool off_ok = ((unsigned long long) lay.up_off == gate_total) &&
                            ((unsigned long long) lay.down_off == 2ull * gate_total);
        std::printf("    %-34s gu_row %6llu %s  d_row %6llu %s  up_off %8llu  down_off %8llu  bytes %9llu %s%s\n",
                    L.what, (unsigned long long) lay.gu_row, rows_ok ? "ok" : "MISMATCH",
                    (unsigned long long) lay.d_row, rows_ok ? "" : "MISMATCH", (unsigned long long) lay.up_off,
                    (unsigned long long) lay.down_off, (unsigned long long) lay.bytes,
                    bytes_ok ? "ok" : "MISMATCH", off_ok ? "" : "  OFFSETS NOT THE CONCATENATION");
        if (!rows_ok || !bytes_ok || !off_ok) {
            ++failures;
            std::printf("           ^ for the (19,18) pair a zero gu_row is CORRECT and expected: type 19 has no kernel,\n");
            std::printf("             so its row size is 0 and the blob's bytes collapse to the down projection alone.\n");
        }
    }

    std::printf("\n  %d of %zu case(s) disagree%s\n", failures, cases.size(), bad ? " (expected: all of them)" : "");

    // the known inconsistency, stated rather than buried
    std::printf("  WARNING: type 11 (Q3_K) is CLAIMED by iq_supported and has a row size, but iq_mmvq has no case for it\n");
    std::printf("           and its default calls std::exit(1) - so iq_supported is NOT a safe guard before iq_mmvq,\n");
    std::printf("           and the port must guard on the dispatch list itself.  Type 11 is layer 45's down projection.\n");
    if (bad) {
        std::printf("  %s - the gate's ability to fail is verified\n",
                    failures == (int) cases.size() ? "PASS" : "FAIL (it did not notice the wrong expectations)");
        return failures == (int) cases.size() ? 0 : 1;
    }
    std::printf("  %s - iq_row_bytes reproduces the pack's own expert sizes, and iq_supported names the right types\n",
                failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
