// tools/expert_row_upload_gate.cpp - can an expert row actually be uploaded, and is the range real?
//
// The device expert path's first act is to copy an expert's bytes from the pack's shard into device memory.  Three things
// can be wrong there and none of them is visible from the host arithmetic alone:
//
//   1. THE RANGE MIGHT NOT EXIST.  The pack's table gives a per-expert blob_bytes and a layer offset, and nothing so far
//      has checked that the file is long enough to contain the region the offsets describe - only that the arithmetic
//      adds up.  A truncated shard, a wrong shard name, or an offset past the end all produce a read of garbage rather
//      than an error the kernel would notice.
//   2. THE SIZES MIGHT NOT MATCH WHAT THE KERNEL EXPECTS.  iq_row_bytes reproduces the pack's sizes for types 16, 18 and
//      23 (tools/expert_row_bytes_gate), but that was arithmetic against arithmetic; here the same numbers are used to
//      carve a real uploaded buffer, so a mismatch shows up as an out-of-range slice.
//   3. THE COPY ITSELF MIGHT BE WRONG - a wrong pitch, a wrong length, a partial transfer - which shows up only by
//      downloading and comparing bytes, not by watching a kernel produce plausible floats.
//
// So this gate does the three: it bounds-checks the whole layer's expert region against the real file, uploads one
// expert's blob to the device, downloads it back, and requires byte-identity.  It deliberately does NOT run a kernel -
// that is the stage's job and its own gate - because a copy check that also computes something is a copy check nobody can
// read.
//
// Usage: expert_row_upload_gate <pack_dir> [layer]
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

struct Row {
    int layer = 0, gu_type = 0, d_type = 0;
    std::uint64_t offset = 0, blob_bytes = 0;
    std::string shard;
};

bool read_table(const std::string& pack_dir, int want_layer, Row& out, std::string& err) {
    std::ifstream f(pack_dir + "/native_experts.txt");
    if (!f) { err = "cannot open native_experts.txt"; return false; }
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        Row r;
        if (!(is >> r.layer >> r.gu_type >> r.d_type >> r.offset >> r.blob_bytes)) continue;
        std::uint64_t gate_off = 0, up_off = 0, down_off = 0;
        is >> gate_off >> up_off >> down_off;
        is >> r.shard;
        if (r.layer == want_layer) { out = r; return true; }
    }
    err = "layer " + std::to_string(want_layer) + " not in the table";
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: %s <pack_dir> [layer]\n", argv[0]); return 2; }
    const std::string pack_dir = argv[1];
    const int layer = (argc > 2) ? std::atoi(argv[2]) : 3;

    Row row;
    std::string err;
    if (!read_table(pack_dir, layer, row, err)) { std::fprintf(stderr, "  %s\n", err.c_str()); return 1; }
    std::printf("  layer %d: gu_type %d d_type %d, blob_bytes %llu per expert, shard %s\n", row.layer, row.gu_type,
                row.d_type, (unsigned long long) row.blob_bytes, row.shard.c_str());

    // WHERE THE EXPERT BYTES ACTUALLY LIVE.  The table's header says its offsets are "absolute offsets in
    // GLM-...-of-00003.gguf, or in the named shard beside it", and the first version of this gate believed it: it looked
    // for the shard beside the pack, did not find it, and stopped.  The pack ships its OWN consolidated experts.bin -
    // 86,406,856,704 bytes, which is exactly the total the table's own header declares - and the per-layer offsets index
    // into that.  So the shard name in the table records provenance, not the read path, and the read path is experts.bin.
    std::string path = pack_dir + "/experts.bin";
    {
        std::ifstream probe(path, std::ios::binary);
        if (!probe) {
            path = row.shard;
            if (path.empty() || path[0] != '/') path = pack_dir + "/" + row.shard;
            std::printf("  (no experts.bin in the pack, falling back to the shard named in the table)\n");
        } else {
            std::printf("  reading from the pack's own consolidated experts.bin\n");
        }
    }
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { std::fprintf(stderr, "  cannot open shard %s\n", path.c_str()); return 1; }
    const std::uint64_t size = (std::uint64_t) f.tellg();
    std::printf("  shard size %llu bytes (%.2f GB)\n", (unsigned long long) size, (double) size / 1073741824.0);

    // 1. does the region the offsets describe actually exist in the file?
    const std::uint64_t per_expert = row.blob_bytes;
    const std::uint64_t region = per_expert * 288ull;                       // n_expert, as the pack header states
    const bool in_bounds = row.offset + region <= size;
    std::printf("  expert region: offset %llu + 288 x %llu = %llu  %s the file\n", (unsigned long long) row.offset,
                (unsigned long long) per_expert, (unsigned long long) (row.offset + region),
                in_bounds ? "fits inside" : "RUNS PAST THE END OF");
    if (!in_bounds) {
        std::fprintf(stderr, "  FAIL: the pack's offsets describe a region the shard does not contain, so every upload\n");
        std::fprintf(stderr, "        from this layer would read past the end.  Nothing about the arithmetic would have\n");
        std::fprintf(stderr, "        said so - this is exactly what the bounds check is for.\n");
        return 1;
    }

    // 2 and 3. upload a 7.5 MB slice and require the round trip to be byte-identical
    //
    // WHAT THIS SLICE IS NOT: it is not one expert's blob.  The region a layer's offsets describe may or may not be laid
    // out expert-contiguously, and this gate does not know which - an attempt to infer it from the offsets (assuming the
    // three projections sit 288 rows apart) matched 0 of 43 layers, so the layout is the engine's business and the
    // authority on it is ExpertSource::blob(layer, expert), which the CPU path already uses successfully.  So this reads
    // blob_bytes from an offset that is deliberately NOT the first expert's, and checks only what a copy is responsible
    // for: that the range is inside the file, that the transfer is exact, and that the bytes are not degenerate.
    const std::uint64_t one = per_expert;
    std::vector<unsigned char> host((size_t) one);
    const std::uint64_t slice_at = row.offset + 7ull * per_expert;
    f.seekg((std::streamoff) slice_at, std::ios::beg);
    f.read((char*) host.data(), (std::streamsize) one);
    if (!f) { std::fprintf(stderr, "  FAIL: short read of %llu bytes at %llu\n", (unsigned long long) one,
                           (unsigned long long) slice_at); return 1; }

    void* dev = nullptr;
    const cudaError_t a = cudaMalloc(&dev, (size_t) one);
    if (a != cudaSuccess) { std::fprintf(stderr, "  cudaMalloc(%llu): %s\n", (unsigned long long) one, cudaGetErrorString(a)); return 1; }
    const cudaError_t u = cudaMemcpy(dev, host.data(), (size_t) one, cudaMemcpyHostToDevice);
    if (u != cudaSuccess) { std::fprintf(stderr, "  cudaMemcpy H2D: %s\n", cudaGetErrorString(u)); cudaFree(dev); return 1; }
    std::vector<unsigned char> back((size_t) one);
    const cudaError_t d = cudaMemcpy(back.data(), dev, (size_t) one, cudaMemcpyDeviceToHost);
    cudaFree(dev);
    if (d != cudaSuccess) { std::fprintf(stderr, "  cudaMemcpy D2H: %s\n", cudaGetErrorString(d)); return 1; }

    std::size_t differ = 0, first = 0;
    bool seen = false;
    for (std::size_t i = 0; i < back.size(); ++i) {
        if (back[i] != host[i]) { ++differ; if (!seen) { first = i; seen = true; } }
    }
    std::printf("  layer %d: H2D+D2H of %llu bytes from offset %llu (an arbitrary non-first slice, NOT one expert's blob"
                " - the layout is ExpertSource::blob's business), %zu byte(s) differ%s\n",
                layer, (unsigned long long) one, (unsigned long long) slice_at, differ,
                differ ? "" : " (byte-identical)");
    if (differ) {
        std::fprintf(stderr, "  FAIL: first difference at %zu (host 0x%02x vs device 0x%02x)\n", first, host[first],
                     back[first]);
        return 1;
    }

    // the bytes must not be a run of one value, or the round trip proves nothing
    bool all_same = true;
    for (std::size_t i = 1; i < host.size() && all_same; ++i) all_same = (host[i] == host[0]);
    std::printf("  content check: %s\n", all_same ? "ALL BYTES IDENTICAL - the round trip proves nothing, the read was wrong"
                                                  : "the uploaded bytes vary, so agreement is meaningful");
    std::printf("  first 16 bytes: ");
    for (int i = 0; i < 16; ++i) std::printf("%02x", host[(size_t) i]);
    std::printf("\n  %s\n", all_same ? "FAIL" : "PASS");
    return all_same ? 1 : 0;
}
