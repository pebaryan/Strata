// src/core/load_main.cpp - `strata-load`: the expert arena load, measurable at any scale.
//
// P2.S1 gives this a target (<= 60 s cold, <= 10 s warm for 33.97 GB) and a method (8 threads, 16 MB chunks,
// a checksum per layer).  The size is a FLAG rather than hard-coded because the honest way to measure a 34 GB
// load is to measure a smaller one first and project - and because pinning 34 GB evicts every page of page
// cache on the machine, which is not something a test should do while other work is running.
#include "strata/core/pinned.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

/// Read an integer out of the pack's manifest.json without a JSON parser: the manifests this tool reads
/// are written by the packers, one key per line, and a missing key must read as 0 rather than throw.
static int64_t manifest_int(const std::string& pack, const char* key) {
    std::ifstream in(pack + "/manifest.json");
    if (!in) return 0;
    const std::string needle = std::string("\"") + key + "\"";
    std::string line;
    while (std::getline(in, line)) {
        const size_t at = line.find(needle);
        if (at == std::string::npos) continue;
        const size_t colon = line.find(':', at + needle.size());
        if (colon == std::string::npos) continue;
        return std::strtoll(line.c_str() + colon + 1, nullptr, 10);
    }
    return 0;
}

// `--layout <pack>`: read a pack's expert table and report what it says, without touching the arena.
// This is how a pack for a model the engine has no kernels for yet (GLM-5.3-Flash) can still be shown
// to LOAD, which is what its port needs first: the table is what every expert consumer asks.
static int report_layout(const std::string& pack, int64_t n_layers, int64_t hidden, int64_t ffn) {
    if (hidden <= 0) hidden = manifest_int(pack, "hidden");
    if (ffn <= 0) ffn = manifest_int(pack, "expert_ffn");
    std::string err;
    if (!strata::kernels::cpu::expert_layout_load(pack, n_layers, 0, err, hidden, ffn)) {
        std::fprintf(stderr, "layout: %s\n", err.c_str());
        return 1;
    }
    const auto& L = strata::kernels::cpu::expert_layout();
    if (!L.native) {
        std::fprintf(stderr, "layout: %s has no native_experts.txt (canonical Q2_0 layout)\n", pack.c_str());
        return 1;
    }
    std::printf("native expert table: %s\n", pack.c_str());
    std::printf("  geometry      hidden %lld, expert ffn %lld%s\n", (long long) hidden, (long long) ffn,
                (hidden > 0 && ffn > 0) ? " (from the pack's manifest)" : " (compiled qwen4exp defaults)");
    std::printf("  layers %lld   experts/layer %lld   first layer with experts %lld   max blob %llu B\n",
                (long long) L.n_layers, (long long) L.n_expert, (long long) L.first_layer,
                (unsigned long long) L.max_blob);
    std::printf("  total         %.2f GB (all layers, all experts)\n", (double) L.total / 1e9);
    if (L.first_layer > 0)
        std::printf("  dense stem    layers 0-%lld carry no experts (this table starts at %lld)\n",
                    (long long) L.first_layer - 1, (long long) L.first_layer);
    std::printf("  blobs (run-length):\n");
    for (int64_t l = L.first_layer; l < L.n_layers;) {
        const uint64_t b = L.blob_bytes(l);
        int64_t j = l;
        while (j + 1 < L.n_layers && L.blob_bytes(j + 1) == b) ++j;
        std::printf("    layer %2lld-%-2lld  %9llu B/blob  %lld layer(s)\n", (long long) l, (long long) j,
                    (unsigned long long) b, (long long) (j - l + 1));
        l = j + 1;
    }
    // A pack's experts.bin must be exactly as long as the table says; a mismatch would decode as plausible
    // weights from the wrong offsets, so check it here rather than at the first token.
    std::FILE* f = std::fopen((pack + "/experts.bin").c_str(), "rb");
    if (f) {
        std::fseek(f, 0, SEEK_END);
        const long long size = std::ftell(f);
        std::fclose(f);
        std::printf("  experts.bin   %.2f GB  %s\n", (double) size / 1e9,
                    (uint64_t) size == L.total ? "matches the table" : "DOES NOT MATCH THE TABLE");
        if ((uint64_t) size != L.total) return 1;
    } else {
        std::printf("  experts.bin   absent: the experts are read from the model's own shards\n");
    }
    return 0;
}

int main(int argc, char** argv) {
    std::string path;
    uint64_t layers = 2;
    int threads = 8;
    uint64_t chunk = 16ull << 20;
    uint64_t blob_bytes = 1382400;      // architecture §3.2, asserted by tools/pack_layer.py
    uint64_t blobs_per_layer = 512;
    bool pin = true;
    bool stream = false;
    std::string layout_pack;
    int64_t layout_layers = 46;         // GLM-5.3-Flash; the flag exists so other models need no rebuild
    int64_t layout_hidden = 0, layout_ffn = 0;   // 0 = take them from the pack's manifest.json

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--file") path = val();
        else if (a == "--layout") layout_pack = val();
        else if (a == "--layout-layers") layout_layers = std::strtoll(val(), nullptr, 10);
        else if (a == "--layout-hidden") layout_hidden = std::strtoll(val(), nullptr, 10);
        else if (a == "--layout-ffn") layout_ffn = std::strtoll(val(), nullptr, 10);
        else if (a == "--layers") layers = std::strtoull(val(), nullptr, 10);
        else if (a == "--threads") threads = std::atoi(val());
        else if (a == "--chunk-mb") chunk = std::strtoull(val(), nullptr, 10) << 20;
        else if (a == "--no-pin") pin = false;
        else if (a == "--stream") stream = true;
        else if (a == "--help" || a == "-h") {
            std::printf("usage: strata-load --file experts.bin [--layers N] [--threads N] [--chunk-mb N]\n"
                        "                   [--no-pin]\n"
                        "       strata-load --layout <pack> [--layout-layers N]   (read the expert table)\n");
            return 0;
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", a.c_str());
            return 2;
        }
    }
    if (!layout_pack.empty()) return report_layout(layout_pack, layout_layers, layout_hidden, layout_ffn);
    if (path.empty()) { std::fprintf(stderr, "--file is required\n"); return 2; }

    const uint64_t bytes = layers * blobs_per_layer * blob_bytes;
    std::printf("expert arena: %llu layers x %llu blobs x %llu B = %.3f GiB\n", (unsigned long long) layers,
                (unsigned long long) blobs_per_layer, (unsigned long long) blob_bytes,
                (double) bytes / (1024.0 * 1024 * 1024));

    strata::core::PinnedArena arena;
    uint8_t* dst = nullptr;
    if (pin) {
        // constructed in place so the note from the constructor can be printed before the load
        new (&arena) strata::core::PinnedArena(bytes);
        std::printf("backing: %s\n", arena.note.c_str());
        if (!arena.valid()) { std::fprintf(stderr, "arena allocation failed\n"); return 1; }
        dst = arena.data();
    } else {
        dst = (uint8_t*) std::malloc((size_t) bytes);
        if (!dst) { std::fprintf(stderr, "malloc failed\n"); return 1; }
        std::printf("backing: malloc, NOT pinned (--no-pin)\n");
    }

    const strata::core::LoadStats st =
        strata::core::load_experts(path, dst, blob_bytes, blobs_per_layer, layers, threads, chunk);
    if (!st.ok) {
        std::fprintf(stderr, "loading the experts failed: %s\n", st.error.empty() ? "unknown" : st.error.c_str());
        return 1;
    }

    std::printf("loaded %.3f GiB in %.3f s with %d threads and %llu MiB chunks  ->  %.2f GiB/s\n",
                (double) st.bytes / (1024.0 * 1024 * 1024), st.seconds, threads,
                (unsigned long long) (chunk >> 20), st.gib_per_second());
    std::printf("per-layer checksums (FNV-1a 64):\n");
    for (size_t i = 0; i < st.layer_checksums.size() && i < 4; ++i) {
        std::printf("  layer %zu  %016llx\n", i, (unsigned long long) st.layer_checksums[i]);
    }
    if (st.layer_checksums.size() > 4) {
        std::printf("  ... %zu more (all layers loaded)\n", st.layer_checksums.size() - 4);
    }
    if (stream) {
        // The bus is the ceiling, so measure it at the granularity the engine actually uses.  Printed as a
        // sweep rather than one number because "bandwidth" without a transfer size is not a specification.
        std::printf("\nexpert stream over the pinned arena (host -> device, one reused destination):\n");
        const uint64_t granularities[] = {blob_bytes, 16 * blob_bytes, 512 * blob_bytes, bytes};
        const char* names[] = {"1 blob (1.38 MB)", "16 blobs (22 MB)", "1 layer (707 MB)", "whole arena"};
        for (int i = 0; i < 4; ++i) {
            const uint64_t g = granularities[i];
            if (g > bytes) continue;
            const int iters = (int) (g >= bytes ? 3 : (bytes / g >= 8 ? 8 : 3));
            const strata::core::StreamStats ss =
                strata::core::stream_bandwidth(dst, (bytes / g) * g, g, iters);
            if (ss.seconds < 0) continue;
            std::printf("  %-18s chunk %8.2f MB  %6.2f GB/s  (%.3f GiB in %.3f s)\n", names[i],
                        (double) g / 1e6, ss.gib_per_second() * 1.073741824,
                        (double) ss.bytes / (1024.0 * 1024 * 1024), ss.seconds);
        }
        // And what it means, using the design's own constants, stated with the assumption attached.
        const double best = 60.0;    // GB/s, replaced below by the measured best if it is larger
        (void) best;
        std::printf("\nper-token expert stream at E = 663.6 MB and hit rate h (VRAM-resident fraction):\n");
        std::printf("  %-8s %-14s %-14s %s\n", "h", "miss MB/token", "ms at 1 blob", "tok/s ceiling");
        const strata::core::StreamStats one = strata::core::stream_bandwidth(dst, blob_bytes, blob_bytes, 64);
        for (double h : {0.0, 0.5, 0.833, 0.9, 0.95}) {
            const double miss_mb = (1.0 - h) * 663.6;
            const double ms = miss_mb / 1000.0 / (one.gib_per_second() * 1.073741824) * 1000.0;
            std::printf("  %-8.3f %-14.1f %-14.2f %.1f\n", h, miss_mb, ms, ms > 0 ? 1000.0 / ms : 0.0);
        }
        std::printf("  h = 0.167 is the design's constant; h = 0.833 is its complement, shown because the\n"
                    "  plan's E = 663.6 MB is the FULL expert set per token, so which fraction is cached is\n"
                    "  the question the number hinges on.\n");
    }

    // The projection is the number the phase's <= 60 s target is about, and it is printed as a projection
    // rather than as the result, because at --layers 2 this program has NOT loaded 33.97 GB.
    const double full = 33973862400.0;
    std::printf("projection for the full %.2f GiB at this rate: %.1f s\n", full / (1024.0 * 1024 * 1024),
                full / (1024.0 * 1024 * 1024) / (st.gib_per_second() > 0 ? st.gib_per_second() : 1e-9));
    return 0;
}
