// hc_post gate for block 0 of GLM-5.3 - the last unverified piece of the block, and the simplest gate in this port:
// no CUDA, no dequantizers, no weights.  Two vectors, a 4x4 matrix and four scalars.
//
// The reference is tools/glm5_hc_reference.py's hc_post, and the two implementations were read against each other
// before this file was written: the engine does
//
//     dst_row[e] = mix.post[dst] * site_out[e];   then   dst_row[e] += mix.comb[dst][src] * residual[src][e]
//
// and the oracle does the same, including the accumulation order.  That is why this is expected to agree to fp32
// precision rather than merely closely - but reading two texts and agreeing with yourself is exactly the trap this
// port has recorded twice, so it is gated anyway rather than left as an inspection.
//
//   usage: hcpost_gate [fixture-dir]      default /home/peb/moredata/glm5-hcpost-l0
//
// The fixture: site_out.bin (4096), residual.bin (4 rows of 4096, flat), post.bin (4), comb.bin (16, [dst][src]),
// result.bin (4096, the oracle's output).  All five were produced by the oracle on values read from the artifact
// and the block-0 dump, with site_out being the KDA gate's own result vector - a real attention output.
//
// NOTE ON WHAT THIS GATE DOES AND DOES NOT PROVE: it checks the residual combination arithmetic given a correct
// HcMix.  It does NOT check that the engine's hc_pre computes that mix - that is the stage-1 gate's job, and the
// mix arrives here as fixture data.  Joining the two - engine mix into this path - is the composition check, and
// it is a different test.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "strata/kernels/glm_hc.hpp"

static bool read_floats(const std::string& path, std::vector<float>& out) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", path.c_str()); return false; }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    out.resize((size_t) n / sizeof(float));
    if (std::fread(out.data(), 1, (size_t) n, f) != (size_t) n) {
        std::fprintf(stderr, "short read on %s\n", path.c_str());
        std::fclose(f);
        return false;
    }
    std::fclose(f);
    return true;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "/home/peb/moredata/glm5-hcpost-l0";
    const int n_embd = 4096;
    const int HC = 4;

    std::vector<float> site, residual, post, comb, ref;
    if (!read_floats(dir + "/site_out.bin", site)) return 1;
    if (!read_floats(dir + "/residual.bin", residual)) return 1;
    if (!read_floats(dir + "/post.bin", post)) return 1;
    if (!read_floats(dir + "/comb.bin", comb)) return 1;
    if (!read_floats(dir + "/result.bin", ref)) return 1;
    if ((int) site.size() != n_embd || (int) residual.size() != HC * n_embd ||
        (int) post.size() != HC || (int) comb.size() != HC * HC || (int) ref.size() != HC * n_embd) {
        std::fprintf(stderr, "fixture sizes wrong: site %zu residual %zu post %zu comb %zu ref %zu\n",
                     site.size(), residual.size(), post.size(), comb.size(), ref.size());
        return 1;
    }

    strata::kernels::glm::HcMix mix;
    for (int i = 0; i < HC; ++i) {
        mix.post[i] = post[(size_t) i];
        for (int j = 0; j < HC; ++j) mix.comb[i][j] = comb[(size_t) i * HC + j];
    }

    // hc_post writes one output row PER DESTINATION stream - HC rows, not one vector.  Allocating n_embd here
    // (as this file first did) was a 4x-undersized buffer; the size assertion above caught it before the call
    // rather than after, which is the reason the assertion exists at all.
    std::vector<float> out((size_t) HC * n_embd);
    strata::kernels::glm::hc_post(site.data(), residual.data(), mix, n_embd, out.data());

    double worst = 0.0, r2 = 0.0, sum = 0.0;
    int64_t bad = 0;
    for (size_t i = 0; i < out.size(); ++i) {
        const double d = std::fabs((double) out[i] - (double) ref[i]);
        if (d > worst) worst = d;
        r2 += (double) ref[i] * (double) ref[i];
        sum += (double) ref[i];
    }
    const double scale = std::sqrt(r2 / n_embd);
    for (size_t i = 0; i < out.size(); ++i)
        if (std::fabs((double) out[i] - (double) ref[i]) > 1e-4 * scale) ++bad;

    std::printf("  engine rms %.6g   oracle rms %.6g   oracle sum %.6g\n", std::sqrt(r2 / n_embd), scale, sum);
    std::printf("  worst absolute difference %.3e   relative to rms %.3e   elements off: %lld\n",
                worst, worst / (scale > 0 ? scale : 1.0), (long long) bad);

    // A gate must exercise what it tests.  With comb near the identity and post in (0,2), the residual term
    // dominates: assert that BOTH terms actually move the result, or the comparison is not testing the sum.
    double post_only = 0.0, res_only = 0.0;
    for (int i = 0; i < n_embd; ++i) {
        post_only += std::fabs((double) mix.post[0] * site[i]);
        res_only += std::fabs((double) mix.comb[0][0] * residual[i]);
    }
    const bool both_terms = (post_only > 0.0) && (res_only > 0.0) &&
                            (post_only > 1e-3 * (post_only + res_only)) &&
                            (res_only > 1e-3 * (post_only + res_only));
    std::printf("  term magnitudes: post-term %.6g   residual-term %.6g   both present: %s\n",
                post_only, res_only, both_terms ? "yes" : "NO");

    if (!both_terms) {
        std::printf("  HC_POST GATE: INCONCLUSIVE - one of the two terms is effectively absent, so this fixture "
                    "cannot distinguish the sum from a single term\n");
        return 2;
    }
    const bool pass = (bad == 0);
    std::printf("  HC_POST GATE: %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
