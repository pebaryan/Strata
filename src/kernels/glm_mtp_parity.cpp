// src/kernels/glm_mtp_parity.cpp - does the MTP head's own wiring match the reference?
//
//   python tools/glm5_mtp_reference.py --gguf <shard1> --raw-fixture /tmp/mtp.bin
//   build-volta/glm_mtp_parity /tmp/mtp.bin
//
// The block's MLA and MoE are covered by glm_mla_parity and glm_moe_parity, whose fixtures demonstrate
// that the MTP block's attention/FFN tensors have the trunk's shapes.  What is checked here is what is
// genuinely new: the enorm/hnorm norms, the concat, eh_proj, and the head norm - plus, independently of
// the oracle, that the concat really puts the EMBEDDING first.
#include "strata/kernels/glm_mtp.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

void compare(const char* what, const std::vector<float>& got, const std::vector<float>& want, bool& ok) {
    double scale = 1e-30, ma = 0.0, mr = 0.0;
    for (float v : want) scale = std::max(scale, std::fabs((double) v));
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        if (std::fabs((double) want[i]) >= 0.1 * scale) mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const bool good = ma / scale < 1e-4 && mr < 1e-3;
    std::printf("  %-8s max abs %.3e (= %.2e of scale)   worst element rel %.3e   %s\n", what, ma, ma / scale,
                mr, good ? "PASS" : "FAIL");
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_mtp_parity <fixture.bin>   (tools/glm5_mtp_reference.py --raw-fixture)\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int32_t hdr[4] = {0};
    if (std::fread(hdr, sizeof(int32_t), 4, f) != 4) { std::fprintf(stderr, "bad header\n"); return 1; }
    const int ne = hdr[0];

    std::vector<float> e = read_floats(f, (size_t) ne);
    std::vector<float> h = read_floats(f, (size_t) ne);
    std::vector<float> enorm = read_floats(f, (size_t) ne);
    std::vector<float> hnorm = read_floats(f, (size_t) ne);
    std::vector<float> eh_proj = read_floats(f, (size_t) ne * 2 * ne);
    std::vector<float> head_w = read_floats(f, (size_t) ne);
    const std::vector<float> e_enorm = read_floats(f, (size_t) ne);
    const std::vector<float> e_hnorm = read_floats(f, (size_t) ne);
    const std::vector<float> e_concat = read_floats(f, (size_t) 2 * ne);
    const std::vector<float> e_cur = read_floats(f, (size_t) ne);
    const std::vector<float> e_head = read_floats(f, (size_t) ne);
    std::fclose(f);

    std::printf("MTP parity vs tools/glm5_mtp_reference.py: blk.%d, n_embd %d\n", hdr[3], ne);

    std::vector<float> got_cur((size_t) ne), got_concat((size_t) 2 * ne), got_e((size_t) ne), got_h((size_t) ne);
    glm::mtp_input(enorm.data(), hnorm.data(), eh_proj.data(), ne, e.data(), h.data(), got_cur.data(),
                   got_concat.data(), got_e.data(), got_h.data());
    std::vector<float> got_head((size_t) ne);
    glm::mtp_head_norm(head_w.data(), ne, got_cur.data(), got_head.data());

    bool ok = true;
    compare("e norm", got_e, e_enorm, ok);
    compare("h norm", got_h, e_hnorm, ok);
    compare("concat", got_concat, e_concat, ok);
    compare("eh proj", got_cur, e_cur, ok);
    compare("head", got_head, e_head, ok);

    // the ordering claim, checked WITHOUT the oracle: the concat's first half must equal e_norm exactly and
    // its second half h_norm.  Swapping the halves produces a full-size plausible vector and a head that
    // is conditioned on the wrong half, so this is the one thing here worth asserting on its own.
    {
        double first = 0.0, second = 0.0;
        for (int i = 0; i < ne; ++i) {
            first = std::max(first, std::fabs((double) got_concat[i] - (double) got_e[i]));
            second = std::max(second, std::fabs((double) got_concat[ne + i] - (double) got_h[i]));
        }
        const bool good = first == 0.0 && second == 0.0;
        std::printf("  invariant  concat = [e_norm | h_norm] with e_norm FIRST (exact): "
                    "first half %.1e, second half %.1e   %s\n", first, second, good ? "PASS" : "FAIL");
        ok = ok && good;
    }

    std::printf("glm_mtp_parity: %s\n", ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
