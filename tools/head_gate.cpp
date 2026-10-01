// head_gate - verify the head's first half (mean over the HC streams, then output_norm) against the oracle.
//
// The head did not exist in this engine at all until now, and the step worth testing is not the norm - it is the
// AVERAGING.  The trunk's output is four hyper-connection streams, and the reference reads their mean; a head that
// skipped that line would project one stream instead of the mean, produce no shape change, and emit plausible logits
// from the wrong vector.  That is invisible without a reference, which is why this gate compares against a value the
// oracle itself saved (glm5_full_run.py --save-hidden) rather than against a recomputation here.
//
// The trunk does not have to be 45 blocks deep for this: the head consumes whatever the trunk produced, so a 3-block
// run verifies the same code, and the oracle writes its hidden state for any --blocks value.  The argmax == 12089
// check DOES need the full model and is deliberately not attempted here - a 3-block hidden state projected through
// the head would print a mismatch that says nothing (the earlier 1-block run's "MISMATCH" was exactly that).
//
// usage: head_gate <dir>     reads l_out-2.bin, w_output_norm.bin, want_hidden.bin; optionally --full for l_out-44.bin

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/glm_layer.hpp"

namespace {

bool read_floats(const std::string& p, size_t n, std::vector<float>& out, const char* what) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::printf("  MISSING %s (%s)\n", p.c_str(), what); return false; }
    out.assign(n, 0.0f);
    const size_t got = std::fread(out.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { std::printf("  SHORT %s: %zu of %zu\n", p.c_str(), got, n); return false; }
    return true;
}

/// The dump container: five u32 [0, ne0..ne3], then ne0-fastest floats.
bool read_dump(const std::string& p, std::vector<float>& out, int ne[4], const char* what) {
    FILE* f = std::fopen(p.c_str(), "rb");
    if (!f) { std::printf("  MISSING %s (%s)\n", p.c_str(), what); return false; }
    unsigned hdr[5] = {0, 0, 0, 0, 0};
    if (std::fread(hdr, 4, 5, f) != 5) { std::fclose(f); std::printf("  BAD header %s\n", p.c_str()); return false; }
    for (int i = 0; i < 4; ++i) ne[i] = (int) hdr[i + 1];
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) (ne[i] > 0 ? ne[i] : 1);
    out.assign(n, 0.0f);
    const size_t got = std::fread(out.data(), 4, n, f);
    std::fclose(f);
    if (got != n) { std::printf("  SHORT %s: %zu of %zu\n", p.c_str(), got, n); return false; }
    std::printf("  %s ne=[%d,%d,%d,%d] %zu floats\n", what, ne[0], ne[1], ne[2], ne[3], out.size());
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: head_gate <dir>\n");
        return 2;
    }
    const std::string dir = argv[1];
    const int N_EMBD = 4096, HC = 4;

    std::vector<float> trunk, want, onorm;
    int ne_trunk[4] = {0, 0, 0, 0};
    // l_out-2 is the last block of a 3-block run; the file's layout is ne0-fastest, so token t occupies elements
    // [t * n_embd * HC, ...) and within a token the HC streams are contiguous blocks of n_embd - which is exactly the
    // engine's [stream][n_embd] layout, so the slice can be handed to the stage unchanged.
    if (!read_dump(dir + "/l_out-2.bin", trunk, ne_trunk, "l_out-2 (the trunk output, HC rows)")) return 2;
    if (!read_floats(dir + "/w_output_norm.bin", (size_t) N_EMBD, onorm, "output_norm")) return 2;
    if (!read_floats(dir + "/want_hidden.bin", (size_t) N_EMBD, want, "the oracle's hidden state")) return 2;

    if (trunk.size() < (size_t) N_EMBD * HC) {
        std::printf("  l_out-2 holds %zu floats, need at least %d for one token's %d streams\n", trunk.size(),
                    N_EMBD * HC, HC);
        return 2;
    }
    const int tokens = (int) (trunk.size() / ((size_t) N_EMBD * HC));
    const float* last_token = trunk.data() + (size_t) (tokens - 1) * N_EMBD * HC;
    std::printf("  %d token(s); using token %d, the last - the head reads the last prompt token\n", tokens, tokens - 1);

    std::vector<float> hidden((size_t) N_EMBD, 0.0f);
    std::string err;
    if (!strata::core::glm::glm_stage_head_mean_norm(last_token, HC, N_EMBD, onorm.data(), hidden.data(), err)) {
        std::printf("HEAD GATE: FAIL - the stage refused: %s\n", err.c_str());
        return 1;
    }

    double worst = 0.0, ssum = 0.0, wsum = 0.0;
    for (size_t i = 0; i < hidden.size(); ++i) {
        worst = std::max(worst, std::fabs((double) hidden[i] - (double) want[i]));
        ssum += (double) hidden[i] * hidden[i];
        wsum += (double) want[i] * want[i];
    }
    const double hr = std::sqrt(ssum / hidden.size()), wr = std::sqrt(wsum / want.size());
    std::printf("  engine hidden: rms %.9g\n", hr);
    std::printf("  oracle hidden: rms %.9g\n", wr);
    std::printf("  worst absolute difference %.6g   relative to rms %.6g   ratio %.9g\n", worst,
                worst / (wr ? wr : 1.0), hr / (wr ? wr : 1.0));

    // A head that AVERAGED NOTHING would be one stream rather than the mean; state what that would look like, so this
    // gate's sensitivity to the step it exists to test is on the record rather than assumed.
    double worst_no_mean = 0.0;
    {
        std::vector<float> one((size_t) N_EMBD);
        double ss = 0.0;
        for (int e = 0; e < N_EMBD; ++e) { one[e] = last_token[e]; ss += (double) one[e] * one[e]; }
        const float r = 1.0f / std::sqrt((float) (ss / N_EMBD) + 1e-5f);
        for (int e = 0; e < N_EMBD; ++e) {
            const double v = (double) one[e] * r * onorm[e];
            worst_no_mean = std::max(worst_no_mean, std::fabs(v - (double) want[e]));
        }
    }
    std::printf("  [sensitivity] skipping the mean would give worst %.6g - this gate separates the two by %.0fx\n",
                worst_no_mean, worst_no_mean / (worst > 0 ? worst : 1e-30));

    const bool pass = worst < 1e-4 && std::isfinite(hr);
    std::printf("HEAD GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
