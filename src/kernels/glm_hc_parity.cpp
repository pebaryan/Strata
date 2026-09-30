// src/kernels/glm_hc_parity.cpp - does our mHC match the reference?
//
// The oracle is tools/glm5_hc_reference.py, which transcribes llama.cpp's GLM5-Next hyper-connection
// graph into numpy, and this test reads a fixture it writes with the REAL weights of a block:
//
//   python tools/glm5_hc_reference.py --gguf <shard1> --layer 0 --site attn --tokens 3 \
//          --raw-fixture /tmp/hc.bin
//   build-volta/glm_hc_parity /tmp/hc.bin
//
// Layout (little-endian, float32 unless stated): int32 n_embd, nt, hc, mix_dim; then fn[mix_dim*hc*n_embd],
// base[mix_dim], scale[3]; then per token x[hc*n_embd], site_out[n_embd], pre[hc], post[hc],
// comb[hc*hc] (comb[dst][src], row-major), layer_in[n_embd]; then per token streamed[hc*n_embd].
#include "strata/kernels/glm_hc.hpp"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

namespace glm = strata::kernels::glm;

namespace {

std::vector<float> read_floats(std::FILE* f, size_t n) {
    std::vector<float> v(n);
    if (n && std::fread(v.data(), sizeof(float), n, f) != n) {
        std::fprintf(stderr, "fixture is truncated\n");
        std::exit(1);
    }
    return v;
}

/// max abs and max relative error, with the relative one guarded against near-zero references.
///
/// A per-element relative tolerance is the wrong instrument for this operator: the Sinkhorn block has
/// entries at the eps floor (~1e-4), where float32 round-off alone is a few 1e-4 relative while being
/// 1e-7 absolute.  The gating criterion is therefore the absolute error relative to the ARRAY's own
/// scale (float32 round-off is proportional to magnitude); the per-element relative error is printed
/// for the record and, where the reference value is meaningful, is expected to be tiny too.
void compare(const char* what, const std::vector<float>& got, const float* want, size_t n, double& worst_abs,
             double& worst_rel, bool& ok) {
    double scale = 1e-30;
    for (size_t i = 0; i < n; ++i) scale = std::max(scale, std::fabs((double) want[i]));
    double ma = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        // only where the reference value is comparable to the array's own scale is a relative error
        // informative; below that it is round-off expressed as a large ratio of a tiny number
        if (std::fabs((double) want[i]) >= 0.1 * scale)
            mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const double rel_of_scale = ma / scale;
    const bool good = rel_of_scale < 1e-5 && mr < 1e-3;
    std::printf("  %-12s max abs %.3e (= %.2e of the array's scale)   worst element rel %.3e   %s\n", what, ma,
                rel_of_scale, mr, good ? "PASS" : "FAIL");
    worst_abs = std::max(worst_abs, ma);
    worst_rel = std::max(worst_rel, mr);
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_hc_parity <fixture.bin>   (write one with tools/glm5_hc_reference.py "
                    "--raw-fixture)\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }

    int32_t hdr[4] = {0, 0, 0, 0};
    if (std::fread(hdr, sizeof(int32_t), 4, f) != 4) { std::fprintf(stderr, "bad fixture header\n"); return 1; }
    const int n_embd = hdr[0], nt = hdr[1], hc = hdr[2], mix_dim = hdr[3];
    if (hc != glm::HC || mix_dim != glm::HC_MIX_DIM || n_embd <= 0 || nt <= 0) {
        std::fprintf(stderr, "fixture is for hc %d / mix_dim %d / n_embd %d; this engine is hc %d / %d\n",
                     hc, mix_dim, n_embd, glm::HC, glm::HC_MIX_DIM);
        return 1;
    }
    const size_t dim = (size_t) hc * (size_t) n_embd;

    const std::vector<float> fn = read_floats(f, (size_t) mix_dim * dim);
    const std::vector<float> base = read_floats(f, (size_t) mix_dim);
    const std::vector<float> scale = read_floats(f, 3);
    const std::vector<float> x = read_floats(f, (size_t) nt * dim);
    const std::vector<float> site_out = read_floats(f, (size_t) nt * (size_t) n_embd);
    const std::vector<float> e_pre = read_floats(f, (size_t) nt * (size_t) hc);
    const std::vector<float> e_post = read_floats(f, (size_t) nt * (size_t) hc);
    const std::vector<float> e_comb = read_floats(f, (size_t) nt * (size_t) hc * (size_t) hc);
    const std::vector<float> e_layer_in = read_floats(f, (size_t) nt * (size_t) n_embd);
    const std::vector<float> e_streamed = read_floats(f, (size_t) nt * dim);
    std::fclose(f);

    std::printf("mHC parity vs tools/glm5_hc_reference.py: n_embd %d, %d token(s), hc %d, mix_dim %d\n",
                n_embd, nt, hc, mix_dim);

    double worst_abs = 0.0, worst_rel = 0.0;
    bool ok = true;
    for (int t = 0; t < nt; ++t) {
        const float* xt = x.data() + (size_t) t * dim;
        glm::HcMix mix;
        std::vector<float> layer_in((size_t) n_embd, 0.0f);
        glm::hc_pre(xt, fn.data(), base.data(), scale.data(), n_embd, layer_in.data(), &mix);

        std::vector<float> got_pre(hc), got_post(hc), got_comb((size_t) hc * hc);
        for (int h = 0; h < hc; ++h) {
            got_pre[(size_t) h] = mix.pre[h];
            got_post[(size_t) h] = mix.post[h];
        }
        for (int dst = 0; dst < hc; ++dst)
            for (int src = 0; src < hc; ++src) got_comb[(size_t) (dst * hc + src)] = mix.comb[dst][src];

        std::vector<float> streamed((size_t) dim, 0.0f);
        glm::hc_post(site_out.data() + (size_t) t * n_embd, xt, mix, n_embd, streamed.data());

        std::printf("token %d\n", t);
        compare("pre", got_pre, e_pre.data() + (size_t) t * hc, hc, worst_abs, worst_rel, ok);
        compare("post", got_post, e_post.data() + (size_t) t * hc, hc, worst_abs, worst_rel, ok);
        compare("comb", got_comb, e_comb.data() + (size_t) t * hc * hc, (size_t) hc * hc, worst_abs, worst_rel, ok);
        compare("layer_in", layer_in, e_layer_in.data() + (size_t) t * n_embd, n_embd, worst_abs, worst_rel, ok);
        compare("streamed", streamed, e_streamed.data() + (size_t) t * dim, dim, worst_abs, worst_rel, ok);
    }

    // The invariants the reference's structure implies, checked against the ORACLE's own fixed point
    // rather than against an arbitrary constant: the eps-floored Sinkhorn does not reach a doubly
    // stochastic matrix exactly, and how far off its columns land varies by block (1e-3 .. 2.4e-2 over
    // this model's layers), so "columns within X of 1" would be a made-up bound.  What must hold is that
    // OUR sums are the ORACLE's sums, and that rows are near-exact because the iteration ends on a row
    // normalization.
    {
        const float* xt = x.data();
        glm::HcMix mix;
        std::vector<float> layer_in((size_t) n_embd, 0.0f);
        glm::hc_pre(xt, fn.data(), base.data(), scale.data(), n_embd, layer_in.data(), &mix);
        double rowmax = 0.0, colmax = 0.0, rowdiff = 0.0, coldiff = 0.0;
        for (int dst = 0; dst < hc; ++dst) {
            double s = 0.0, w = 0.0;
            for (int src = 0; src < hc; ++src) {
                s += mix.comb[dst][src];
                w += e_comb[(size_t) (dst * hc + src)];
            }
            rowmax = std::max(rowmax, std::fabs(s - 1.0));
            rowdiff = std::max(rowdiff, std::fabs(s - w));
        }
        for (int src = 0; src < hc; ++src) {
            double s = 0.0, w = 0.0;
            for (int dst = 0; dst < hc; ++dst) {
                s += mix.comb[dst][src];
                w += e_comb[(size_t) (dst * hc + src)];
            }
            colmax = std::max(colmax, std::fabs(s - 1.0));
            coldiff = std::max(coldiff, std::fabs(s - w));
        }
        std::printf("invariants: rows within %.3e of 1 (the iteration ends on a row normalization); "
                    "columns %.3e off 1, which the oracle's own columns share to within %.3e / %.3e\n",
                    rowmax, colmax, rowdiff, coldiff);
        if (rowmax > 1e-4 || rowdiff > 1e-5 || coldiff > 1e-5) { std::printf("  invariants FAIL\n"); ok = false; }
    }

    std::printf("glm_hc_parity: worst abs %.3e, worst rel %.3e -> %s\n", worst_abs, worst_rel,
                ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
