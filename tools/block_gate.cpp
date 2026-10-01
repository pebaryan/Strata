// block_gate - run one whole block and compare each stage against the reference dump.
//
//     block_gate <block-dir> <kda-dir> <ffn-dir> <dump-dir> [tokens]
//
// The block loop (glm_block_forward) is the first chain code this port has written, and nothing has run it.  Its four
// constituent kernels each have passing gates, but the assembly does not - so this tool drives it with fixture files
// and prints every intermediate beside the dump's own value for that stage.
//
// THIS FIRST VERSION LOADS AND CHECKS ONLY.  It reads all 27 weight arrays, asserts each one's size against the size
// the geometry demands, reports where every quantity came from, and stops.  That is deliberate: this port has spent
// a whole session on defects that were purely a writer and a reader disagreeing about a format - a container header
// read as data, per-site slices taken for packed arrays, five derived sizes wrong by up to 4096x, a float64 file read
// as float32, a single-stream array taken for the stream set - and every one would have been caught by asserting the
// shape at the point of loading.  So the loading is verified before anything is computed from it.
//
// The run and the per-stage comparison come next, on top of a fixture whose shapes are known good.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/kernels/glm_kda.hpp"
#include "strata/kernels/glm_moe.hpp"

namespace {

constexpr int N_EMBD = 4096;
constexpr int HC = 4;

std::vector<uint8_t> read_file(const std::string& path, bool& ok) {
    std::vector<uint8_t> bytes;
    FILE* f = std::fopen(path.c_str(), "rb");
    ok = false;
    if (!f) {
        std::printf("  MISSING  %s\n", path.c_str());
        return bytes;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    bytes.resize((size_t) n);
    const size_t got = std::fread(bytes.data(), 1, (size_t) n, f);
    std::fclose(f);
    ok = (got == (size_t) n);
    return bytes;
}

/// Load n_floats from a raw float32 file, asserting the file is EXACTLY that size.  The assert is the point: a file
/// that is 2x or 8x the expected size is almost always a different tensor or a different dtype, and both would read
/// as plausible numbers.
bool load_f32(const std::string& path, size_t n_floats, std::vector<float>& out) {
    bool ok = false;
    std::vector<uint8_t> b = read_file(path, ok);
    if (!ok) return false;
    const size_t want = n_floats * 4;
    if (b.size() != want) {
        std::printf("  SIZE     %s: %zu bytes, expected %zu (%zu floats)\n", path.c_str(), b.size(), want, n_floats);
        return false;
    }
    out.resize(n_floats);
    std::memcpy(out.data(), b.data(), want);
    return true;
}

/// The dump's container: five u32 words [0, ne0, ne1, ne2, ne3] then ne0*ne1*ne2*ne3 floats, ne0 fastest.
bool load_dump(const std::string& path, std::vector<float>& out, int64_t ne[4]) {
    bool ok = false;
    std::vector<uint8_t> b = read_file(path, ok);
    if (!ok) return false;
    if (b.size() < 20) {
        std::printf("  SIZE     %s: %zu bytes, too small for a header\n", path.c_str(), b.size());
        return false;
    }
    uint32_t hdr[5];
    std::memcpy(hdr, b.data(), 20);
    for (int i = 0; i < 4; ++i) ne[i] = (int64_t) hdr[i + 1];
    const size_t n = (size_t) (ne[0] * ne[1] * ne[2] * ne[3]);
    if (b.size() != 20 + n * 4) {
        std::printf("  SIZE     %s: header ne=[%lld,%lld,%lld,%lld] implies %zu floats but the file holds %zu\n",
                    path.c_str(), (long long) ne[0], (long long) ne[1], (long long) ne[2], (long long) ne[3],
                    n, (b.size() - 20) / 4);
        return false;
    }
    out.resize(n);
    std::memcpy(out.data(), b.data() + 20, n * 4);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        std::fprintf(stderr, "usage: block_gate <block-dir> <kda-dir> <ffn-dir> <dump-dir> [tokens]\n");
        return 2;
    }
    const std::string bd = argv[1], kd = argv[2], fd = argv[3], dd = argv[4];
    const int tokens = (argc > 5) ? std::atoi(argv[5]) : 1;
    int bad = 0, checked = 0;

    std::printf("block_gate: loading.  block=%s kda=%s ffn=%s dump=%s tokens=%d\n", bd.c_str(), kd.c_str(),
                fd.c_str(), dd.c_str(), tokens);

    // ---- the block input: hc_init's streams, which the loop takes as x ----
    {
        std::vector<float> x;
        int64_t ne[4];
        checked++;
        if (!load_dump(dd + "/hc_init.bin", x, ne)) { bad++; }
        else if (ne[0] != N_EMBD || ne[1] != HC) {
            std::printf("  SHAPE    hc_init.bin: ne=[%lld,%lld,%lld,%lld], expected [%d,%d,...]\n", (long long) ne[0],
                        (long long) ne[1], (long long) ne[2], (long long) ne[3], N_EMBD, HC);
            bad++;
        } else {
            std::printf("  ok       x            %zu floats   %d streams x %d embd, %lld tokens available\n",
                        x.size(), HC, N_EMBD, (long long) ne[2]);
        }
        // and the array the loop actually gets: token 0's HC streams
        std::vector<float> x1;
        checked++;
        if (!load_f32(bd + "/x.bin", (size_t) HC * N_EMBD, x1)) bad++;
        else std::printf("  ok       x.bin        %zu floats   one token's %d streams\n", x1.size(), HC);
    }

    // ---- the eight pipeline weights the fixture provides ----
    struct { const char* name; size_t n; } w[] = {
        {"hc_attn_fn", (size_t) N_EMBD * 96}, {"hc_attn_base", 24}, {"hc_attn_scale", 3}, {"attn_norm", N_EMBD},
        {"hc_ffn_fn",  (size_t) N_EMBD * 96}, {"hc_ffn_base",  24}, {"hc_ffn_scale",  3}, {"ffn_norm",  N_EMBD},
    };
    for (const auto& e : w) {
        std::vector<float> v;
        checked++;
        if (!load_f32(bd + "/w_" + e.name + ".bin", e.n, v)) { bad++; continue; }
        double mn = v[0], mx = v[0];
        for (float f : v) { if (f < mn) mn = f; if (f > mx) mx = f; }
        std::printf("  ok       w_%-12s %9zu floats  [%.6g, %.6g]\n", e.name, v.size(), mn, mx);
    }

    // ---- the KDA weights: sixteen arrays, sizes fixed by the geometry ----
    {
        const strata::kernels::glm::KdaGeometry g;
        const size_t ne = (size_t) g.n_embd, inner = (size_t) g.d_inner();
        // The array sizes are MEASURED from the fixture, not inferred from the geometry.  The first version of this
        // tool inferred them and got six wrong by 2x to 128x: wq/wk/wv/wo are ne x d_inner (wq is NOT twice that),
        // ssm_a is per-head (nh) rather than d_inner, and f_a/f_b/beta are 64x/128x/32x d_inner.  Two of the
        // fixture's names also differ from the artifact's - the timestep bias file is dt_bias and the output norm is
        // o_norm - and there is no separate input_norm, because attn_norm serves that role.  These came from listing
        // the files and dividing by four, because the oracle's writer is the authority on what it emits.
        struct { const char* name; size_t n; } kw[] = {
            {"attn_norm", ne},                      {"wq", ne * inner}, {"wk", ne * inner}, {"wv", ne * inner},
            {"conv_q", (size_t) g.d_conv * inner},  {"conv_k", (size_t) g.d_conv * inner},
            {"conv_v", (size_t) g.d_conv * inner},
            {"ssm_a", (size_t) g.nh},               {"dt_bias", inner},
            {"ssm_f_a", inner * 64},                {"ssm_f_b", inner * 128},
            {"ssm_beta", inner * 32},
            {"ssm_g_a", inner * 64},                {"ssm_g_b", inner * 128},
            {"o_norm", (size_t) g.hd},              {"wo", inner * ne},
        };
        for (const auto& e : kw) {
            std::vector<float> v;
            checked++;
            if (!load_f32(kd + "/w_" + e.name + ".bin", e.n, v)) { bad++; continue; }
            std::printf("  ok       w_%-12s %9zu floats\n", e.name, v.size());
        }
        std::printf("  geometry n_embd=%d nh=%d hd=%d d_conv=%d d_inner=%zu\n", g.n_embd, g.nh,
                    g.hd, g.d_conv, inner);
    }

    // ---- the FFN's three raw block sets, kept raw: the proven dequantizers read them, not this loader ----
    {
        strata::kernels::glm::MoeGeometry fg;   // ff defaults to the ROUTED width (2048)
        fg.ff = 12288;                          // a leading dense block is 12288: set it, do not inherit
        struct { const char* name; size_t n; size_t block; } fw[] = {
            {"raw_gate", (size_t) N_EMBD, 176}, {"raw_up", (size_t) N_EMBD, 176}, {"raw_down", (size_t) N_EMBD, 210},
        };
        for (const auto& e : fw) {
            std::vector<uint8_t> raw;
            bool ok = false;
            raw = read_file(fd + "/" + e.name + ".bin", ok);
            checked++;
            if (!ok) { bad++; continue; }
            if (raw.size() % e.block != 0) {
                std::printf("  SIZE     %s: %zu bytes is not a multiple of the %zu-byte block\n", e.name, raw.size(),
                            e.block);
                bad++;
                continue;
            }
            std::printf("  ok       %-12s %9zu bytes  %zu blocks x %zu B = %zu elements\n", e.name, raw.size(),
                        raw.size() / e.block, e.block, raw.size() / e.block * 256);
        }
    }

    std::printf("block_gate: %d of %d inputs loaded and shape-checked", checked - bad, checked);
    if (bad) {
        std::printf("  -> %d BAD, refusing to run anything on this fixture\n", bad);
        return 1;
    }
    std::printf("  -> fixture is loadable and shape-consistent\n");
    std::printf("NOTE: this version loads only.  The run and the per-stage comparison against the dump come next -\n");
    std::printf("      the loop has not been executed yet, and loading being right does not make it right.\n");
    return 0;
}
