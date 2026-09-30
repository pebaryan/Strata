// src/kernels/glm_indexer_parity.cpp - does our indexer match the reference?
//
// The oracle is tools/glm5_indexer_reference.py (transcribed from llama.cpp's GLM5-Next build_dsa_top_k
// and the pool inputs its memory module fills):
//
//   python tools/glm5_indexer_reference.py --gguf <shard1> --layer 3 --tokens 12 --raw-fixture /tmp/idx.bin
//   build-volta/glm_indexer_parity /tmp/idx.bin
//
// It checks the write side (the LayerNorm'd key plus the gate, recomputed from the same hidden states),
// the read side (pooled keys and scores), the SELECTION, and then the two invariants that are the whole
// point of this operator - that the cut is on whole pools and that the trailing incomplete pool is
// always visible - because those are the parts a plausible-looking implementation gets silently wrong.
#include "strata/kernels/glm_indexer.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
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

void compare(const char* what, const float* got, const float* want, size_t n, bool& ok) {
    double scale = 1e-30, ma = 0.0, mr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(want[i])) continue;
        scale = std::max(scale, std::fabs((double) want[i]));
    }
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(want[i])) continue;
        const double d = std::fabs((double) got[i] - (double) want[i]);
        ma = std::max(ma, d);
        if (std::fabs((double) want[i]) >= 0.1 * scale)
            mr = std::max(mr, d / std::fabs((double) want[i]));
    }
    const bool good = ma / scale < 1e-4 && mr < 1e-4;
    std::printf("  %-8s max abs %.3e (= %.2e of scale)   worst element rel %.3e   %s\n", what, ma, ma / scale,
                mr, good ? "PASS" : "FAIL");
    ok = ok && good;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: glm_indexer_parity <fixture.bin>   (tools/glm5_indexer_reference.py "
                    "--raw-fixture)\n");
        return 2;
    }
    std::FILE* f = std::fopen(argv[1], "rb");
    if (!f) { std::fprintf(stderr, "cannot open %s\n", argv[1]); return 1; }
    int32_t hdr[8] = {0};
    if (std::fread(hdr, sizeof(int32_t), 8, f) != 8) { std::fprintf(stderr, "bad header\n"); return 1; }

    glm::IdxGeometry g;
    g.n_embd = hdr[0];
    g.q_lora = hdr[1];
    g.d = hdr[2];
    g.nh = hdr[3];
    g.kpool = hdr[4];
    g.top_k = hdr[5];
    const int n_positions = hdr[6];
    const int n_sel_expected = hdr[7];
    const int n_pool = n_positions / g.kpool;

    std::vector<float> attn_k = read_floats(f, (size_t) g.d * g.n_embd);
    std::vector<float> k_norm_w = read_floats(f, (size_t) g.d);
    std::vector<float> k_norm_b = read_floats(f, (size_t) g.d);
    std::vector<float> attn_q_b = read_floats(f, (size_t) g.nh * g.d * g.q_lora);
    std::vector<float> proj = read_floats(f, (size_t) g.nh * g.n_embd);
    std::vector<float> c_gate = read_floats(f, (size_t) g.d * g.n_embd);
    std::vector<float> ape = read_floats(f, (size_t) g.kpool * g.d);
    std::vector<float> cur = read_floats(f, (size_t) g.n_embd);
    std::vector<float> qr = read_floats(f, (size_t) g.q_lora);
    std::vector<float> hidden = read_floats(f, (size_t) n_positions * g.n_embd);
    std::vector<float> cache = read_floats(f, (size_t) n_positions * 2 * g.d);
    const std::vector<float> e_pooled = read_floats(f, (size_t) n_pool * g.d);
    const std::vector<float> e_score = read_floats(f, (size_t) n_pool);
    const std::vector<float> e_bias = read_floats(f, (size_t) n_pool);
    int32_t n_cells_expected = 0;
    if (std::fread(&n_cells_expected, sizeof(int32_t), 1, f) != 1) { std::fprintf(stderr, "bad tail\n"); return 1; }
    std::vector<int32_t> e_cells((size_t) std::max(n_cells_expected, 0));
    if (n_cells_expected > 0 &&
        std::fread(e_cells.data(), sizeof(int32_t), (size_t) n_cells_expected, f) != (size_t) n_cells_expected) {
        std::fprintf(stderr, "fixture is truncated (cells)\n");
        return 1;
    }
    std::fclose(f);

    glm::IdxWeights w;
    w.attn_k = attn_k.data();
    w.k_norm_w = k_norm_w.data();
    w.k_norm_b = k_norm_b.data();
    w.attn_q_b = attn_q_b.data();
    w.proj = proj.data();
    w.c_gate = c_gate.data();
    w.ape = ape.data();

    std::printf("indexer parity vs tools/glm5_indexer_reference.py: %d positions, %d complete pool(s) of %d, "
                "d %d, nh %d, top_k %d\n", n_positions, n_pool, g.kpool, g.d, g.nh, g.top_k);

    bool ok = true;

    // ---- write side: the cache rows, recomputed from the same hidden states
    std::vector<float> got_cache((size_t) n_positions * 2 * g.d, 0.0f);
    for (int p = 0; p < n_positions; ++p)
        glm::idx_cache_row(w, g, hidden.data() + (size_t) p * g.n_embd, got_cache.data() + (size_t) p * 2 * g.d);
    compare("cache", got_cache.data(), cache.data(), cache.size(), ok);

    // ---- read side
    std::vector<float> got_pooled((size_t) std::max(n_pool, 1) * g.d, 0.0f);
    std::vector<float> got_score((size_t) std::max(n_pool, 1), 0.0f);
    std::vector<float> got_bias((size_t) std::max(n_pool, 1), 0.0f);
    const int cap = n_positions + g.kpool;
    std::vector<int32_t> got_cells((size_t) cap, -1);
    const glm::IdxResult res = glm::idx_select(w, g, cur.data(), qr.data(), n_positions, cache.data(),
                                               n_positions - 1, got_cells.data(), cap, got_pooled.data(),
                                               got_score.data(), got_bias.data());
    if (n_pool) {
        compare("pooled", got_pooled.data(), e_pooled.data(), e_pooled.size(), ok);
        compare("score", got_score.data(), e_score.data(), e_score.size(), ok);
        compare("bias", got_bias.data(), e_bias.data(), e_bias.size(), ok);
    }

    // ---- the selection itself: the SET of cells must match (the order among equal scores is not promised
    //      by the reference, which is exactly why the cut is on whole pools)
    got_cells.resize((size_t) res.n_cells);
    std::vector<int32_t> want_cells = e_cells;
    std::sort(got_cells.begin(), got_cells.end());
    std::sort(want_cells.begin(), want_cells.end());
    const bool same = got_cells == want_cells;
    std::printf("  selection  n_sel %d (expected %d), cells %d (expected %d), identical set %s   %s\n",
                res.n_sel, n_sel_expected, res.n_cells, n_cells_expected, same ? "yes" : "NO",
                same ? "PASS" : "FAIL");
    ok = ok && same && res.n_sel == n_sel_expected;

    // ---- invariant 1: whole pools only.  Every selected cell must belong to a pool whose r members are
    //      ALL selected, except the trailing incomplete pool's cells, which are always appended.
    {
        int broken = 0;
        const int tail_base = n_pool * g.kpool;
        for (int b = 0; b < n_pool; ++b) {
            int members = 0;
            for (int m = 0; m < g.kpool; ++m)
                members += (int) std::count(got_cells.begin(), got_cells.end(), b * g.kpool + m);
            if (members != 0 && members != g.kpool) ++broken;
        }
        int tail_selected = 0;
        for (int p = tail_base; p < n_positions; ++p)
            tail_selected += (int) std::count(got_cells.begin(), got_cells.end(), p);
        const int tail_size = n_positions - tail_base;
        std::printf("  invariant  pools cut whole: %d split   tail %d/%d cells always selected   %s\n", broken,
                    tail_selected, tail_size, (broken == 0 && tail_selected == tail_size) ? "PASS" : "FAIL");
        ok = ok && broken == 0 && tail_selected == tail_size;
    }

    std::printf("glm_indexer_parity: %s\n", ok ? "0 failures" : "FAILURES");
    return ok ? 0 : 1;
}
