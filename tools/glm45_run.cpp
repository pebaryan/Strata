/// THE PORT'S LAST STEP: run the engine's OWN 45-block trunk and produce a token.
///
/// Everything this file assembles has been verified separately, and the point of writing it is to find the one thing
/// that cannot be verified separately - the assembly.  The acceptance test is a single number: argmax == 12089 for
/// prompt 785 6722 315 9621 374.  It is deliberately NOT a per-layer float comparison, because the engine's own
/// hidden state accumulates the quantization of 43 routed blocks; each routed layer differs from a float64 reference
/// by about 5% of rms at the worst element, measured, and that is the correct behaviour rather than a defect.
///
/// The plumbing is documented in tools/glm5_binding_contract.txt under "THE FINAL RUNNER".  Two traps are enforced in
/// code here because they are silent otherwise: the swiglu clamps are set EXPLICITLY (their defaults are 0, which
/// disables them) and expert_layout_load is called BEFORE FileExpertSource::open (open validates against the
/// process-wide layout, which is the canonical Q2_0 default until the pack is loaded).
///
/// usage: glm45_run <pack-dir> <gguf-shard1> [hc-init.bin] [w-output.bin] [w-output-norm.bin]

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/resource.h>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <fcntl.h>
#include <list>
#include <memory>
#include <mutex>
#include <unistd.h>
#include <thread>
#include <vector>

#include "strata/core/expert_source.hpp"
#include "strata/core/glm_bind.hpp"
#include <sys/mman.h>
#include <sys/stat.h>
#include "strata/core/expert_row_cache.hpp"
#include "strata/core/glm_expert_device.hpp"
#include "strata/core/glm_layer.hpp"
#include "strata/core/glm_layer_weights.hpp"
#include "strata/core/glm_moe_native.hpp"
#include "strata/core/glm_trunk.hpp"
#include "strata/core/layout.hpp"
#include "strata/core/native_dense.hpp"
#include "strata/core/native_head.hpp"
#include "strata/core/weights.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/iq_kernels.hpp"

namespace C = strata::core;
namespace K = strata::kernels::glm;
namespace KCPU = strata::kernels::cpu;
namespace KN = strata::kernels;

static const int N_LAYERS = 45;      // blocks 0..44; 45 is the MTP head and not part of the trunk
static const int N_EMBD = 4096;
static const int HC = 4;             // hyper-connection streams
static const int N_EXPERT = 288;
static const int N_USED = 8;
static const int FF_EXP = 2048;      // a routed expert / the shared expert
static const int FF_DENSE = 12288;   // the dense FFN of blocks 0..2
static const int NH = 64, HD = 128;  // KDA geometry
static const int KV_LORA = 512;      // MLA latent width
static const int TOKENS = 5;         // the prompt's length, which is also the MLA cache depth
static const float EPS = 1e-5f;
static const float CLAMP = 10.0f;    // swiglu_clamp_exp / _shexp: 10.0 for every layer of this model

struct StageCount {
    long calls = 0;
    int token = 0;
    std::map<std::string, long> by_name;
};

/// WALL-CLOCK PER LAYER, AT FILE SCOPE because the stage counter is local to the token lambda while a profile is a
/// property of the whole run.  The callback fires at known points inside each block, so the interval between two
/// consecutive calls is the cost of what ran between them, and attributing it to the layer that was live gives a
/// per-layer profile with no profiler attached - which is what decides where the CUDA port should start.
struct StageProfile {
    int last_layer = -1;
    const char* last_name = "";
    std::chrono::steady_clock::time_point last_time;
    std::map<int, double> ms;
    std::map<std::string, double> by_stage;   ///< keyed by the stage that was LIVE during the interval
    /// The interval between two consecutive callbacks is the cost of whatever ran between them, and the stage that ran
    /// in the gap is identified by the callback that CLOSED it - so the delta is charged to the name of the callback
    /// BEFORE it, which is what the trunk reports at the end of the work it just did.
    void note(int layer, const char* name) {
        const std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
        const double d = (last_layer >= 0) ? std::chrono::duration<double, std::milli>(now - last_time).count() : 0.0;
        if (last_layer >= 0) {
            ms[last_layer] += d;
            by_stage[last_name] += d;
        }
        last_layer = layer;
        last_name = name;
        last_time = now;
    }
};
static StageProfile g_prof;

/// HOW MANY BYTES THE STAGING MATERIALISES, which is the measurement that says what a device MMVQ path is worth.  The
/// provider dequantizes a layer's quantized tensors to host floats and frees them when the next layer is asked for, so
/// with the token loop outer EVERY LAYER IS RE-STAGED ON EVERY TOKEN: a device dequant plus a device-to-host copy per
/// tensor per token.  A device MMVQ path reads the artifact's own blocks in place - no dequantization, no transfer - so
/// this number is what it eliminates, and it is evidence rather than an argument.
static unsigned long long g_staged_bytes = 0;

static void stage_cb(void* ctx, int layer, const char* name, const float* data, int n) {
    StageCount* sc = (StageCount*) ctx;
    ++sc->calls;
    sc->by_name[name] += 1;
    // ---- WALL-CLOCK PER LAYER.  The callback fires at known points inside each block, so the interval between two
    // consecutive callbacks is the cost of what ran between them, and attributing that to the layer that was live gives
    // a per-layer profile with no profiler attached.  This is what decides which stage the CUDA port should take first,
    // and it is measured rather than guessed, like everything else in this port.
    g_prof.note(layer, name);
    // ---- AND REPORT THE MAGNITUDE, because counting stages proves the loop ran and says nothing about what it produced.
    // The oracle's own per-layer dumps for these same four families are on disk, and the runner's input IS the oracle's
    // hc_init - so the two runs are comparable stage by stage from block 0, which is the bisection that found block 0's
    // double-norm, block 1's chaining and block 3's four verdicts.  Token 0 only: that is where the lifetime and state
    // questions do not arise, so a disagreement there is a stage defect rather than a state one.
    const int dump_token = std::getenv("STRATA_STAGE_TOKEN") ? std::atoi(std::getenv("STRATA_STAGE_TOKEN")) : 0;
    if (sc->token == dump_token && data != nullptr && n > 0) {
        double ss = 0.0;
        for (int i = 0; i < n; ++i) ss += (double) data[i] * (double) data[i];
        std::printf("STAGE %d %s %.9g %d\n", layer, name, std::sqrt(ss / (double) n), n);
        // Optional raw stage capture for an exact, elementwise comparison with the oracle.  Keeping this behind an
        // environment variable means the normal runner remains read-only and avoids hundreds of diagnostic files.
        if (const char* dir = std::getenv("STRATA_STAGE_DUMP")) {
            const std::string path = std::string(dir) + "/" + std::to_string(layer) + "-" + name + ".raw";
            if (std::FILE* f = std::fopen(path.c_str(), "wb")) {
                std::fwrite(data, sizeof(float), (size_t) n, f);
                std::fclose(f);
            }
        }
    }
}

static bool read_dump(const std::string& p, std::vector<float>& out, int ne[4]) {
    std::ifstream f(p, std::ios::binary);
    if (!f) return false;
    unsigned hdr[5] = {0, 0, 0, 0, 0};
    f.read((char*) hdr, 20);
    if (!f) return false;
    for (int i = 0; i < 4; ++i) ne[i] = (int) hdr[i + 1];
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) (ne[i] > 0 ? ne[i] : 1);
    out.assign(n, 0.0f);
    f.read((char*) out.data(), (std::streamsize) (n * 4));
    return (bool) f;
}

static std::string layer_types(const std::string& pack, int layer, int& gu, int& dt) {
    std::ifstream f(pack + "/native_experts.txt");
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream is(line);
        int64_t l = -1, off = 0, nb = 0, go = 0, uo = 0, dob = 0;
        int g = 0, d = 0;
        if (!(is >> l >> g >> d >> off >> nb >> go >> uo >> dob)) continue;
        if (l == layer) { gu = g; dt = d; return ""; }
    }
    return "layer " + std::to_string(layer) + " is not in native_experts.txt";
}

struct Provider {
    // Per-layer storage.  glm_fill_layer_weights assigns pointers INTO the bound block and into these, so all of it
    // must outlive the run - which is why they are sized once and never resized.
    std::vector<C::GlmBoundBlock> bound;
    std::vector<K::KdaWeights> kda;
    std::vector<K::MlaWeights> mla;
    std::vector<std::array<const float*, 3> > shexp_store;
    std::vector<C::glm::GlmTrunkLayerWeights> w;
    std::vector<KCPU::NativeFmt> fmt;
    std::vector<bool> fmt_ready;
    K::KdaGeometry kda_g;
    K::MlaGeometry mla_g;
    K::MoeGeometry dense_g, moe_g, shexp_g;
    C::ExpertSource* src = nullptr;
    std::vector<std::vector<float*> > staged_by_layer;
    ~Provider() {
        for (std::vector<float*>& layer : staged_by_layer)
            for (float* f : layer) std::free(f);
    }
};

/// Dequantize one quantized bound tensor to HOST floats.  The kernels are device kernels (dequant_q5_K was gated
/// bit-exact against the oracle over 134 MB before being wired in), so the work happens on the device and the result is
/// copied back - what matters is that it ENDS UP IN HOST MEMORY, because every consumer in the trunk is a CPU kernel.
static float* dequant_to_host(const C::GlmBoundBlock::Tensor& t, std::string& err) {
    const size_t n = (size_t) t.ne0 * (size_t) (t.ne1 > 0 ? t.ne1 : 1);
    float* dev = nullptr;
    if (cudaMalloc(&dev, n * sizeof(float)) != cudaSuccess) {
        err = "dequant_to_host: cudaMalloc failed for " + t.name;
        return nullptr;
    }
    if (t.native_type == 8) {
        strata::kernels::dequant_q8_0((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else if (t.native_type == 13) {
        strata::kernels::dequant_q5_K((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else if (t.native_type == 14) {
        strata::kernels::dequant_q6_K((const uint8_t*) t.ptr, dev, (int64_t) n, nullptr);
    } else {
        err = "dequant_to_host: no dequantizer for type " + std::to_string(t.native_type) + " (" + t.name + ")";
        cudaFree(dev);
        return nullptr;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        err = "dequant_to_host: the kernel failed for " + t.name + ": " + cudaGetErrorString(cudaGetLastError());
        cudaFree(dev);
        return nullptr;
    }
    float* host = (float*) std::malloc(n * sizeof(float));
    if (host == nullptr) { err = "dequant_to_host: host allocation failed for " + t.name; cudaFree(dev); return nullptr; }
    const cudaError_t cp = cudaMemcpy(host, dev, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dev);
    if (cp != cudaSuccess) {
        std::free(host);
        err = "dequant_to_host: copy back failed for " + t.name + ": " + cudaGetErrorString(cp);
        return nullptr;
    }
    return host;
}

// LAYER STREAMER for the prompt path.  A long chunk routes to nearly every expert of every layer, and one layer's experts
// are contiguous in experts.bin, so the whole layer is read sequentially with large O_DIRECT reads into one of two
// anonymous staging buffers by a reader thread that runs up to two layers ahead of the compute.  That does two things the
// mmap path could not: the SSD stays busy through every layer's attention / hyper-connection / GPU work (it idled for
// ~4.8 s of each ~10 s layer), and the reads bypass the page cache, whose constant reclaim on a 31 GB box held buffered
// reads to ~205 MB/s against ~340 MB/s for the link.  The MoE stage takes its rows from the staging buffer
// (LayerStreamer::blob), blocking only until the bytes it needs have landed; anything not staged falls back to the mmap.
class LayerStreamer {
public:
    ~LayerStreamer() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        if (thread_.joinable()) thread_.join();
        for (int s = 0; s < 2; ++s) if (buf_[s]) munmap(buf_[s], buf_cap_);
        if (fd_ >= 0) ::close(fd_);
    }
    bool enabled() const { return enabled_; }

    /// `fd` is experts.bin opened O_DIRECT; the geometry comes from the expert source's own blob addresses.
    bool init(int fd, C::ExpertSource* src, uintptr_t map_start, uintptr_t map_end, off_t map_off, int first_layer, int n_layers) {
        fd_ = fd; first_ = first_layer; n_layers_ = n_layers;
        off_.assign((size_t) n_layers, 0); bytes_.assign((size_t) n_layers, 0); row_.assign((size_t) n_layers, 0);
        for (int l = first_layer; l < n_layers; ++l) {
            const uint8_t* b0 = src->blob(l, 0);
            const uint8_t* b1 = src->blob(l, 1);
            if (!b0 || !b1 || b1 <= b0) return false;
            const size_t row = (size_t) (b1 - b0), bytes = row * N_EXPERT;
            if ((uintptr_t) b0 < map_start || (uintptr_t) b0 + bytes > map_end) return false;
            const off_t off = map_off + (off_t) ((uintptr_t) b0 - map_start);
            if ((off % 4096) != 0 || (bytes % 4096) != 0) return false;     // O_DIRECT needs aligned offsets and lengths
            off_[(size_t) l] = off; bytes_[(size_t) l] = bytes; row_[(size_t) l] = row;
            buf_cap_ = std::max(buf_cap_, bytes);
        }
        for (int s = 0; s < 2; ++s) {
            void* p = mmap(nullptr, buf_cap_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
            if (p == MAP_FAILED) return false;
            buf_[s] = (uint8_t*) p;
        }
        enabled_ = true;
        thread_ = std::thread([this] { run(); });
        return true;
    }

    /// A chunk is starting: read layers first_layer.. in order, two ahead of the consumer.
    void begin(int first_layer) {
        std::unique_lock<std::mutex> l(m_);
        active_ = false;                                      // so the reader cannot start another stale layer meanwhile
        ++gen_;
        cv_.notify_all();
        cv_ready_.wait(l, [&] { return !reading_; });        // the reader abandons an old pass within one 8 MB read
        for (Slot& s : slot_) { s.layer = -1; s.filled = 0; s.failed = false; }
        next_ = std::max(first_layer, first_);
        consumed_ = next_;
        active_ = true;
        cv_.notify_all();
    }
    /// Every layer before `layer` is finished (its buffer may be reused).
    void layer_started(int layer) {
        std::lock_guard<std::mutex> l(m_);
        consumed_ = std::max(consumed_, layer);
        cv_.notify_all();
    }
    void end() {
        std::lock_guard<std::mutex> l(m_);
        active_ = false;
        ++gen_;
        cv_.notify_all();
    }
    bool active() const { return enabled_ && active_; }

    /// The staged row for (layer, expert), waiting for its bytes to arrive; nullptr when this layer is not being streamed.
    const uint8_t* blob(int layer, int expert) {
        if (!active() || layer < first_ || layer >= n_layers_ || expert < 0 || expert >= N_EXPERT) return nullptr;
        std::unique_lock<std::mutex> l(m_);
        const size_t need = ((size_t) expert + 1) * row_[(size_t) layer];
        int s = -1;
        cv_ready_.wait(l, [&] {
            if (!active_) return true;
            for (int i = 0; i < 2; ++i) if (slot_[i].layer == layer) { s = i; return true; }
            return layer < next_ ? true : false;           // a layer already passed over (consumed) will never be staged
        });
        if (s < 0 || !active_) return nullptr;
        cv_ready_.wait(l, [&] { return !active_ || slot_[s].filled >= need || slot_[s].failed || slot_[s].layer != layer; });
        if (!active_ || slot_[s].failed || slot_[s].layer != layer || slot_[s].filled < need) return nullptr;
        return buf_[s] + (size_t) expert * row_[(size_t) layer];
    }
    double read_seconds() const { return read_ns_.load() / 1e9; }
    double wait_seconds() const { return wait_ns_.load() / 1e9; }

private:
    struct Slot { int layer = -1; size_t filled = 0; bool failed = false; };
    int free_slot() const {   // a slot is free once its layer is consumed
        for (int s = 0; s < 2; ++s) if (slot_[s].layer < 0 || slot_[s].layer < consumed_) return s;
        return -1;
    }
    void run() {
        const size_t kChunk = 8u << 20;
        for (;;) {
            int s, layer; uint64_t gen;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || (active_ && next_ < n_layers_ && free_slot() >= 0); });
                if (stop_) return;
                s = free_slot(); layer = next_++; gen = gen_;
                slot_[s].layer = layer; slot_[s].filled = 0; slot_[s].failed = false;
                reading_ = true;
            }
            cv_ready_.notify_all();
            const size_t bytes = bytes_[(size_t) layer]; const off_t off = off_[(size_t) layer];
            size_t done = 0; bool fail = false;
            const auto t0 = std::chrono::steady_clock::now();
            while (done < bytes) {
                if (gen_snapshot() != gen) break;                       // a new pass began (or the chunk ended)
                const size_t n = std::min(kChunk, bytes - done);
                const ssize_t r = ::pread(fd_, buf_[s] + done, n, off + (off_t) done);
                if (r <= 0) { fail = true; break; }
                done += (size_t) r;
                { std::lock_guard<std::mutex> l(m_); slot_[s].filled = done; }
                cv_ready_.notify_all();
            }
            read_ns_ += (long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - t0).count();
            {
                std::lock_guard<std::mutex> l(m_);
                if (fail) slot_[s].failed = true;
                if (gen_ != gen) slot_[s].layer = -1;
                reading_ = false;
            }
            cv_ready_.notify_all();
        }
    }
    uint64_t gen_snapshot() { std::lock_guard<std::mutex> l(m_); return gen_; }

    int fd_ = -1;
    bool enabled_ = false;
    int first_ = 3, n_layers_ = 0;
    std::vector<off_t> off_;
    std::vector<size_t> bytes_, row_;
    size_t buf_cap_ = 0;
    uint8_t* buf_[2] = {nullptr, nullptr};
    std::thread thread_;
    std::mutex m_;
    std::condition_variable cv_, cv_ready_;
    Slot slot_[2];
    int next_ = 0, consumed_ = 0;
    uint64_t gen_ = 0;
    std::atomic<bool> active_{false};
    bool stop_ = false, reading_ = false;
    std::atomic<long> read_ns_{0}, wait_ns_{0};
};
static LayerStreamer g_stream;

static const uint8_t* blob_adapter(void* ctx, int layer, int expert) {
    if (g_stream.active())
        if (const uint8_t* staged = g_stream.blob(layer, expert)) return staged;
    return ((C::ExpertSource*) ctx)->blob(layer, expert);
}

// Expert rows are recycled through a size-keyed free list instead of cudaMalloc/cudaFree per miss: cudaFree
// synchronises the whole device, and a cold decode token misses on up to ~340 rows.  Row sizes take only a few distinct
// values (one per expert format), so the list stays tiny; its contents are at most the rows evicted and not yet reused.
static std::unordered_map<size_t, std::vector<void*>> g_row_pool;
static C::ExpertRowCache* g_reclaim_cache = nullptr;   // the GPU row cache; trimmed when an allocation cannot be met

// Device memory that rows must never take: everything else the engine allocates lazily (projection and KDA workspaces,
// MoE scratch that grows with the chunk, the CUDA runtime itself) needs room too, and a row allocation that leaves the
// card at exactly zero free turns the next small cudaMalloc elsewhere into a hard failure.
static const size_t kVramReserve = (size_t) 768 * 1024 * 1024;

static void row_pool_drain() {
    for (auto& kv : g_row_pool) {
        for (void* q : kv.second) (void) cudaFree(q);
        kv.second.clear();
    }
}

static bool vram_room(size_t bytes) {
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void) cudaGetLastError(); return true; }
    return free_b >= bytes + kVramReserve;
}

static void* row_alloc(size_t bytes) {
    std::vector<void*>& v = g_row_pool[bytes];
    if (!v.empty()) { void* p = v.back(); v.pop_back(); return p; }
    // Not enough room (counting the reserve): idle rows of OTHER sizes may be what holds it (a freed row only serves its
    // own size class, and layers use several), so give those back first; then the coldest cached rows, one at a time -
    // a prompt chunk needs every expert it selects at once (up to ~2.3 GB beyond the cache), a more urgent use of the
    // memory than the least recently used cache entries.
    if (!vram_room(bytes)) {
        row_pool_drain();
        for (int tries = 0; tries < 4096 && !vram_room(bytes) && g_reclaim_cache != nullptr && g_reclaim_cache->trim_one();
             ++tries)
            row_pool_drain();
    }
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) { (void) cudaGetLastError(); return nullptr; }
    return p;
}

static void row_release(void* p, size_t bytes) {
    if (p != nullptr) g_row_pool[bytes].push_back(p);
}

// Before a prompt chunk: make sure the device has room for its workspaces (projection/KDA/MoE scratch scale with the
// chunk length) by giving back the coldest cached rows if it does not.  Without this a cache sized for decode can fill
// the card and the chunk dies on an allocation the row pool knows nothing about.
static void ensure_prefill_headroom(int tokens) {
    const size_t need = (size_t) 600 * 1024 * 1024 + (size_t) tokens * 768 * 1024;
    size_t free_b = 0, total_b = 0;
    if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void) cudaGetLastError(); return; }
    int trimmed = 0;
    while (free_b < need && g_reclaim_cache != nullptr && g_reclaim_cache->trim_one()) {
        for (auto& kv : g_row_pool) {
            for (void* q : kv.second) (void) cudaFree(q);
            kv.second.clear();
        }
        ++trimmed;
        if (cudaMemGetInfo(&free_b, &total_b) != cudaSuccess) { (void) cudaGetLastError(); return; }
    }
    if (trimmed > 0)
        std::fprintf(stderr, "prefill headroom: trimmed %d cached expert rows (%.2f GB free, %.2f GB wanted)\n", trimmed,
                     (double) free_b / 1073741824.0, (double) need / 1073741824.0);
}

static void free_expert_row(void*, const C::ExpertRowKey&, const C::ExpertRowEntry& entry) {
    row_release(entry.dev, entry.bytes);
}

static C::ExpertRowCacheConfig expert_cache_config(size_t budget) {
    C::ExpertRowCacheConfig cfg;
    cfg.budget_bytes = budget;
    cfg.on_evict = &free_expert_row;
    return cfg;
}

struct GlmExpertDeviceRuntime {
    C::ExpertRowCache cache;
    C::GlmExpertDeviceScratch scratch;

    explicit GlmExpertDeviceRuntime(size_t budget) : cache(expert_cache_config(budget)) {}
    ~GlmExpertDeviceRuntime() {
        cache.clear();
        scratch.release();
    }
};

static bool run_cached_device_expert(void* raw, int layer, int expert, const uint8_t* blob,
                                     const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (layout.bytes == 0 || layout.bytes != fmt.bytes) {
        err = "device expert layout disagrees with NativeFmt byte count";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;

    const C::ExpertRowKey key{layer, expert};
    const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
    void* device_row = nullptr;
    bool temporary = false;
    if (state == C::ExpertRowState::resident) {
        const C::ExpertRowEntry* entry = runtime.cache.find(key);
        if (entry == nullptr || entry->dev == nullptr || entry->bytes != layout.bytes) {
            err = "expert cache reported resident without a matching device row";
            return false;
        }
        device_row = entry->dev;
    } else {
        if ((device_row = row_alloc(layout.bytes)) == nullptr) {
            err = std::string("cudaMalloc expert row: ") + cudaGetErrorString(cudaGetLastError());
            return false;
        }
        if (cudaMemcpy(device_row, blob, layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = std::string("upload expert row: ") + cudaGetErrorString(cudaGetLastError());
            row_release(device_row, layout.bytes);
            return false;
        }
        if (state == C::ExpertRowState::needs_upload) {
            runtime.cache.insert(key, C::ExpertRowEntry{device_row, layout.bytes});
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (entry == nullptr) {
                err = "expert cache did not retain a row it accepted";
                return false;
            }
            device_row = entry->dev;
        } else {
            temporary = true;  // correctness-preserving path for a row larger than the configured cache budget
        }
    }

    const bool ok = C::glm_expert_ffn_device_resident((const uint8_t*) device_row, layout, fmt.gu_type, fmt.d_type,
                                                       fmt.n_embd, fmt.n_ff, x, out, runtime.scratch, err);
    if (temporary) row_release(device_row, layout.bytes);
    if (!ok && !err.empty())
        err = "layer " + std::to_string(layer) + " expert " + std::to_string(expert) + ": " + err;
    return ok;
}

static void prefetch_host_rows(const uint8_t* p, size_t bytes);

// The upload-every-miss decode path: every selected expert not resident on the device is copied up (and normally cached)
// and the whole MoE runs on the GPU.  Kept as the fallback for STRATA_GLM_CPU_MISS=0 and for any CPU-tier failure.
static bool run_cached_device_moe_upload(void* raw, int layer, int n_experts, const int32_t* experts,
                                 const uint8_t* const* blobs, const float* weights,
                                 const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (layout.bytes == 0 || layout.bytes != fmt.bytes || n_experts <= 0 || n_experts > 64) {
        err = "device MoE layout/count disagrees with NativeFmt";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;
    std::vector<const uint8_t*> rows((size_t) n_experts);
    std::vector<void*> temporary;
    for (int i = 0; i < n_experts; ++i)
        if (blobs[i] && runtime.cache.find(C::ExpertRowKey{layer, (int) experts[i]}) == nullptr) prefetch_host_rows(blobs[i], layout.bytes);
    for (int i = 0; i < n_experts; ++i) {
        if (!blobs[i]) { err = "device MoE has a null host blob"; return false; }
        const C::ExpertRowKey key{layer, (int) experts[i]};
        const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
        void* device_row = nullptr;
        if (state == C::ExpertRowState::resident) {
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (!entry || !entry->dev || entry->bytes != layout.bytes) {
                err = "expert cache reported a resident row without a matching device allocation";
                return false;
            }
            device_row = entry->dev;
        } else {
            if ((device_row = row_alloc(layout.bytes)) == nullptr) {
                err = std::string("cudaMalloc expert row: ") + cudaGetErrorString(cudaGetLastError());
                return false;
            }
            if (cudaMemcpy(device_row, blobs[i], layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                err = std::string("upload expert row: ") + cudaGetErrorString(cudaGetLastError());
                row_release(device_row, layout.bytes);
                return false;
            }
            if (state == C::ExpertRowState::needs_upload) {
                runtime.cache.insert(key, C::ExpertRowEntry{device_row, layout.bytes});
                const C::ExpertRowEntry* entry = runtime.cache.find(key);
                if (!entry) { err = "expert cache did not retain an accepted row"; return false; }
                device_row = entry->dev;
            } else {
                temporary.push_back(device_row); // oversized rows still run correctly outside the cache
            }
        }
        rows[(size_t) i] = (const uint8_t*) device_row;
    }
    const bool ok = C::glm_expert_moe_device_resident(rows.data(), weights, n_experts, layout,
                                                       fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff,
                                                       x, out, runtime.scratch, err);
    for (void* row : temporary) row_release(row, layout.bytes);
    if (!ok && !err.empty()) err = "layer " + std::to_string(layer) + " device MoE: " + err;
    return ok;
}

// ---- CPU TIER: compute the GPU cache's misses in place, from host memory, beside the GPU ----------------------------
//
// The engine's own design for experts that do not fit the card: the routed experts the GPU holds run there, the rest are
// computed by the CPU straight from the RAM copy, concurrently, instead of being pushed over PCIe first.  An uploaded
// miss costs a synchronous 6.5-8.8 MB copy (~0.7 ms) per expert and then the row sits in the cache whether or not it is
// ever used again; the CPU needs ~1.3 ms of core time for the same expert and no bus traffic, and eight cores run eight
// misses side by side.  A row is promoted into the GPU cache only once it has missed `g_promote_after` times (or while
// the cache still has free room), so the cache converges on the experts the conversation actually reuses.
class CpuPool {
public:
    explicit CpuPool(int workers) {
        for (int i = 0; i < workers; ++i) threads_.emplace_back([this] { loop(); });
    }
    ~CpuPool() {
        { std::lock_guard<std::mutex> l(m_); stop_ = true; }
        cv_.notify_all();
        for (std::thread& t : threads_) t.join();
    }
    int size() const { return (int) threads_.size() + 1; }
    /// Runs fn(0..n-1) over the workers and the calling thread; returns when every index is done.
    void run(int n, const std::function<void(int)>& fn) {
        if (n <= 0) return;
        if (threads_.empty() || n == 1) { for (int i = 0; i < n; ++i) fn(i); return; }
        // Every run owns its job.  A worker still finishing the previous run holds the previous job (by shared_ptr), so
        // it can never take an index of, or be counted into, the next one - sharing next/done/n across runs let a
        // straggler run a task twice and overshoot the completion count, which hung the caller.
        std::shared_ptr<Job> job = std::make_shared<Job>();
        job->fn = &fn;
        job->n = n;
        {
            std::lock_guard<std::mutex> l(m_);
            job_ = job;
            ++gen_;
        }
        cv_.notify_all();
        work(*job);
        std::unique_lock<std::mutex> l(job->m);
        job->done_cv.wait(l, [&] { return job->done == job->n; });
    }

private:
    struct Job {
        const std::function<void(int)>* fn = nullptr;
        int n = 0;
        std::atomic<int> next{0};
        int done = 0;
        std::mutex m;
        std::condition_variable done_cv;
    };
    static void work(Job& job) {
        for (;;) {
            const int i = job.next.fetch_add(1);
            if (i >= job.n) break;
            (*job.fn)(i);
            std::lock_guard<std::mutex> l(job.m);
            if (++job.done == job.n) job.done_cv.notify_all();
        }
    }
    void loop() {
        uint64_t seen = 0;
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return stop_ || gen_ != seen; });
                if (stop_) return;
                seen = gen_;
                job = job_;
            }
            if (job) work(*job);
        }
    }
    std::vector<std::thread> threads_;
    std::mutex m_;
    std::condition_variable cv_;
    std::shared_ptr<Job> job_;
    uint64_t gen_ = 0;
    bool stop_ = false;
};

static CpuPool& cpu_pool() {
    static CpuPool pool([] {
        const char* e = std::getenv("STRATA_GLM_CPU_THREADS");
        const int n = e ? std::atoi(e) : (int) std::thread::hardware_concurrency() - 1;
        return std::max(0, n);
    }());
    return pool;
}

// HOST RAM TIER: an LRU of expert rows in ordinary anonymous (hugepage-eligible) memory, between the GPU cache and the
// SSD.  The mmap of experts.bin is a poor thing to compute from: its pages are file-backed, so each access pays page-table
// setup (about 0.8 ms per 7.5 MB expert even with every page cached) and the kernel may unmap them under pressure.  Rows
// are read in once with pread (the page-cache copy is dropped straight after, so RAM holds each row once) and then
// served at memory speed to the CPU kernels, and to the GPU when a row is promoted.  Fixed-size slots: every row class
// fits the largest, which wastes a little on the smaller classes and keeps eviction trivial.
class HostRowCache {
public:
    ~HostRowCache() {
        if (base_ != nullptr) munmap(base_, region_bytes_);
        if (fd_ >= 0) ::close(fd_);
    }
    bool enabled() const { return base_ != nullptr; }
    size_t slots() const { return n_slots_; }
    size_t slot_bytes() const { return slot_bytes_; }

    /// `typical_bytes` only sizes the "slots" figure reported at startup; rows are allocated at their exact size.
    bool init(size_t budget_bytes, size_t typical_bytes, const std::string& path, uintptr_t map_start, uintptr_t map_end,
              off_t map_offset) {
        typical_bytes = (typical_bytes + 4095) & ~(size_t) 4095;
        if (typical_bytes == 0 || budget_bytes / typical_bytes < 16) return false;
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) return false;
        region_bytes_ = budget_bytes & ~(size_t) 4095;
        void* p = mmap(nullptr, region_bytes_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
        if (p == MAP_FAILED) { ::close(fd_); fd_ = -1; return false; }
        (void) madvise(p, region_bytes_, MADV_HUGEPAGE);
        base_ = (uint8_t*) p;
        slot_bytes_ = typical_bytes;
        n_slots_ = region_bytes_ / typical_bytes;
        map_start_ = map_start; map_end_ = map_end; map_offset_ = map_offset;
        return true;
    }
    void set_drop_page_cache(bool drop) { drop_cache_ = drop; }
    /// Offset in experts.bin of a pointer into its mapping.
    bool file_offset(const uint8_t* blob, off_t& off) const {
        const uintptr_t a = (uintptr_t) blob;
        if (a < map_start_ || a >= map_end_) return false;
        off = map_offset_ + (off_t) (a - map_start_);
        return true;
    }
    /// A resident row (made most recently used), or nullptr.
    uint8_t* lookup(int layer, int expert) {
        auto it = map_.find(key(layer, expert));
        if (it == map_.end()) return nullptr;
        lru_.splice(lru_.begin(), lru_, it->second.pos);
        return base_ + it->second.offset;
    }
    /// Room for a row of `bytes` that is about to be filled: a freed block of exactly that size, else fresh space from
    /// the region, else the least recently used rows are dropped until a block of that size frees up (rows of other
    /// sizes freed on the way go to their own lists for reuse).  The row is recorded as resident immediately: the caller
    /// fills it before anything reads it.  nullptr only when `bytes` cannot fit the region at all.
    uint8_t* reserve(int layer, int expert, size_t bytes) {
        bytes = (bytes + 4095) & ~(size_t) 4095;
        size_t offset = 0;
        for (;;) {
            std::vector<size_t>& fl = free_[bytes];
            if (!fl.empty()) { offset = fl.back(); fl.pop_back(); break; }
            if (bump_ + bytes <= region_bytes_) { offset = bump_; bump_ += bytes; break; }
            if (lru_.empty()) return nullptr;
            const uint64_t victim = lru_.back();
            lru_.pop_back();
            const Entry& v = map_[victim];
            free_[v.bytes].push_back(v.offset);
            map_.erase(victim);
        }
        lru_.push_front(key(layer, expert));
        map_[key(layer, expert)] = Entry{offset, bytes, lru_.begin()};
        return base_ + offset;
    }
    void erase(int layer, int expert) {
        auto it = map_.find(key(layer, expert));
        if (it == map_.end()) return;
        free_[it->second.bytes].push_back(it->second.offset);
        lru_.erase(it->second.pos);
        map_.erase(it);
    }
    /// pread `bytes` at `off` into `dst`, then drop the page-cache copy so the row is held once.
    bool fill(uint8_t* dst, off_t off, size_t bytes) const {
        size_t done = 0;
        while (done < bytes) {
            const ssize_t n = ::pread(fd_, dst + done, bytes - done, off + (off_t) done);
            if (n <= 0) return false;
            done += (size_t) n;
        }
        if (drop_cache_) (void) posix_fadvise(fd_, off, (off_t) bytes, POSIX_FADV_DONTNEED);
        return true;
    }

private:
    struct Entry { size_t offset; size_t bytes; std::list<uint64_t>::iterator pos; };
    static uint64_t key(int layer, int expert) { return ((uint64_t) (uint32_t) layer << 32) | (uint32_t) expert; }
    uint8_t* base_ = nullptr;
    size_t region_bytes_ = 0, slot_bytes_ = 0, n_slots_ = 0;
    int fd_ = -1;
    uintptr_t map_start_ = 0, map_end_ = 0;
    off_t map_offset_ = 0;
    size_t bump_ = 0;
    bool drop_cache_ = false;
    std::unordered_map<size_t, std::vector<size_t>> free_;   // freed block offsets, by (page-rounded) size
    std::list<uint64_t> lru_;
    std::unordered_map<uint64_t, Entry> map_;
};
static HostRowCache g_host;
static long g_host_hits = 0, g_host_fills = 0;

static bool g_cpu_miss = false;         // STRATA_GLM_CPU_MISS=1 enables the hybrid CPU/GPU path once the cache is full
static int g_promote_after = 2;         // STRATA_GLM_PROMOTE_AFTER: misses before a row is uploaded into a full cache
static std::vector<uint8_t> g_miss_count((size_t) 46 * N_EXPERT, 0);
static long g_cpu_experts = 0, g_gpu_experts = 0, g_promoted = 0;
static double g_cpu_ms[7] = {0, 0, 0, 0, 0, 0, 0};   // populate, gate/up, quant h, down, wait for GPU, promote, host lookup
static std::atomic<long> g_fill_ns{0};                // summed over threads: time inside HostRowCache::fill

static double g_upload_frac = 0.5;      // STRATA_GLM_HYBRID_UPLOAD_FRAC: share of a full cache's misses that are uploaded

static bool run_cached_device_moe(void* raw, int layer, int n_experts, const int32_t* experts,
                                  const uint8_t* const* blobs, const float* weights,
                                  const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    if (!g_cpu_miss) return run_cached_device_moe_upload(raw, layer, n_experts, experts, blobs, weights, fmt, x, out, err);
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (layout.bytes == 0 || layout.bytes != fmt.bytes || n_experts <= 0 || n_experts > 64) {
        err = "device MoE layout/count disagrees with NativeFmt";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;
    // While the GPU cache still has room for every expert this call selects, uploading and caching them is strictly
    // better: the rows end up on the device and every later token hits.
    if (runtime.cache.bytes_used() + (size_t) n_experts * layout.bytes <= runtime.cache.budget())
        return run_cached_device_moe_upload(raw, layer, n_experts, experts, blobs, weights, fmt, x, out, err);

    // CACHE FULL: split the call three ways, all at once.  Resident rows and a share of the misses (which are uploaded and
    // cached, so they hit next time) run on the GPU; the remaining misses are computed on the CPU from host memory while
    // the PCIe copies and the GPU kernels run.  Neither side idles, and the cache still converges on the reused experts.
    std::vector<const uint8_t*> hit_rows;
    std::vector<float> hit_w;
    std::vector<int> pend;
    std::vector<C::ExpertRowState> pend_state;
    for (int i = 0; i < n_experts; ++i) {
        if (!blobs[i]) { err = "device MoE has a null host blob"; return false; }
        const C::ExpertRowKey key{layer, (int) experts[i]};
        const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
        if (state == C::ExpertRowState::resident) {
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (!entry || !entry->dev || entry->bytes != layout.bytes) {
                err = "expert cache reported a resident row without a matching device allocation";
                return false;
            }
            hit_rows.push_back((const uint8_t*) entry->dev);
            hit_w.push_back(weights[i]);
        } else {
            pend.push_back(i);
            pend_state.push_back(state);
            prefetch_host_rows(blobs[i], layout.bytes);
        }
    }
    const int nh = (int) hit_rows.size();
    g_gpu_experts += nh;

    // A row that keeps missing is worth a place on the card, so it is always uploaded (and cached); the first-time misses
    // are the ones shared between the CPU and the upload, since many of them are one-offs that would only evict
    // something useful.  Without this the CPU's share never becomes resident and a reused expert misses forever.
    std::vector<char> hot(pend.size(), 0);
    int cold = 0;
    for (size_t p = 0; p < pend.size(); ++p) {
        uint8_t& seen = g_miss_count[(size_t) layer * N_EXPERT + (size_t) experts[pend[p]]];
        if (seen < 255) ++seen;
        hot[p] = seen >= g_promote_after && pend_state[p] == C::ExpertRowState::needs_upload;
        if (!hot[p]) ++cold;
    }
    int cold_up_left = (int) std::ceil((double) cold * g_upload_frac);
    std::vector<int> up_idx, miss;
    std::vector<C::ExpertRowState> miss_state;
    for (size_t p = 0; p < pend.size(); ++p) {
        if (hot[p]) { up_idx.push_back(pend[p]); continue; }
        if (pend_state[p] == C::ExpertRowState::needs_upload && cold_up_left > 0) { --cold_up_left; up_idx.push_back(pend[p]); continue; }
        miss.push_back(pend[p]);
        miss_state.push_back(pend_state[p]);
    }
    const int nm = (int) miss.size();
    g_cpu_experts += nm;
    const int n_embd = (int) fmt.n_embd, n_ff = (int) fmt.n_ff;

    // GPU side: upload this call's share of misses, then run resident + uploaded rows in one go.
    struct Uploaded { C::ExpertRowKey key; void* dev; };
    std::vector<Uploaded> uploaded;
    std::vector<float> gpu_out((size_t) n_embd);
    std::string gpu_err;
    const bool gpu_has = nh + (int) up_idx.size() > 0;
    auto gpu_job = [&]() -> bool {
        std::vector<const uint8_t*> rows = hit_rows;
        std::vector<float> w = hit_w;
        for (int i : up_idx) {
            void* dev = row_alloc(layout.bytes);
            if (dev == nullptr) { gpu_err = std::string("cudaMalloc expert row: ") + cudaGetErrorString(cudaGetLastError()); return false; }
            if (cudaMemcpy(dev, blobs[i], layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                gpu_err = std::string("upload expert row: ") + cudaGetErrorString(cudaGetLastError());
                row_release(dev, layout.bytes);
                return false;
            }
            uploaded.push_back(Uploaded{C::ExpertRowKey{layer, (int) experts[i]}, dev});
            rows.push_back((const uint8_t*) dev);
            w.push_back(weights[i]);
        }
        if (rows.empty()) return true;
        return C::glm_expert_moe_device_resident(rows.data(), w.data(), (int) rows.size(), layout, fmt.gu_type, fmt.d_type,
                                                  fmt.n_embd, fmt.n_ff, x, gpu_out.data(), runtime.scratch, gpu_err);
    };
    auto release_uploaded = [&]() { for (const Uploaded& u : uploaded) row_release(u.dev, layout.bytes); uploaded.clear(); };

    std::vector<std::vector<float>> part((size_t) nm, std::vector<float>((size_t) n_embd));
    std::future<bool> gpu_done;
    if (nm > 0) gpu_done = std::async(std::launch::async, gpu_job);

    if (nm > 0) {
        std::vector<uint8_t> act(fmt.act_bytes);
        KCPU::native_quant_act(fmt, x, act.data());
        std::vector<std::vector<float>> ff((size_t) nm, std::vector<float>((size_t) n_ff));
        std::vector<std::vector<uint8_t>> hq((size_t) nm, std::vector<uint8_t>(fmt.h_bytes));

        // Where the CPU reads each expert from: the host RAM tier when enabled (a hit, or a slot to fill), else the mmap.
        std::vector<const uint8_t*> cblob((size_t) nm);
        std::vector<uint8_t*> fill_dst((size_t) nm, nullptr);
        std::vector<off_t> fill_off((size_t) nm, 0);
        std::vector<char> fill_failed((size_t) nm, 0);
        for (int m = 0; m < nm; ++m) {
            const int idx = miss[(size_t) m];
            cblob[(size_t) m] = blobs[idx];
            if (!g_host.enabled()) continue;
            if (uint8_t* p = g_host.lookup(layer, (int) experts[idx])) {
                cblob[(size_t) m] = p;
                ++g_host_hits;
            } else if (off_t off; g_host.file_offset(blobs[idx], off)) {
                if (uint8_t* slot = g_host.reserve(layer, (int) experts[idx], layout.bytes)) {
                    fill_dst[(size_t) m] = slot;
                    fill_off[(size_t) m] = off;
                    cblob[(size_t) m] = slot;
                    ++g_host_fills;
                }
            }
        }

        CpuPool& pool = cpu_pool();
        const int gu_chunk = 128, d_chunk = 256;
        const int gu_chunks = (n_ff + gu_chunk - 1) / gu_chunk, d_chunks = (n_embd + d_chunk - 1) / d_chunk;
        const void* act_p[1] = {act.data()};
        auto lap_t = std::chrono::steady_clock::now();
        auto lap = [&](int k) {
            const auto now = std::chrono::steady_clock::now();
            g_cpu_ms[k] += std::chrono::duration<double, std::milli>(now - lap_t).count();
            lap_t = now;
        };
        // Map each expert's pages in one call before the dot products (a per-page first-touch fault costs about 1,800
        // faults per expert and serialises the threads on the mapping lock); or, for the host tier, fill its slot.
        pool.run(nm, [&](int m) {
            if (fill_dst[(size_t) m] != nullptr) {
                const auto f0 = std::chrono::steady_clock::now();
                if (!g_host.fill(fill_dst[(size_t) m], fill_off[(size_t) m], layout.bytes)) {
                    fill_failed[(size_t) m] = 1;
                    cblob[(size_t) m] = blobs[miss[(size_t) m]];
                }
                g_fill_ns += (long) std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - f0).count();
                return;
            }
            if (cblob[(size_t) m] != blobs[miss[(size_t) m]]) return;   // host tier hit: nothing to map
            const uintptr_t pg = 4096;
            const uintptr_t a = (uintptr_t) blobs[miss[(size_t) m]] & ~(pg - 1);
            const uintptr_t e = ((uintptr_t) blobs[miss[(size_t) m]] + layout.bytes + pg - 1) & ~(pg - 1);
#ifndef MADV_POPULATE_READ
#define MADV_POPULATE_READ 22
#endif
            (void) madvise((void*) a, e - a, MADV_POPULATE_READ);
        });
        for (int m = 0; m < nm; ++m)
            if (fill_failed[(size_t) m]) g_host.erase(layer, (int) experts[miss[(size_t) m]]);
        lap(0);
        pool.run(nm * gu_chunks, [&](int t) {
            const int m = t / gu_chunks, c = t % gu_chunks;
            float* fp[1] = {ff[(size_t) m].data()};
            KCPU::native_gu_rows(fmt, cblob[(size_t) m], act_p, 1, fp, c * gu_chunk, std::min(n_ff, (c + 1) * gu_chunk));
        });
        lap(1);
        pool.run(nm, [&](int m) { KCPU::native_quant_h(fmt, ff[(size_t) m].data(), hq[(size_t) m].data()); });
        lap(2);
        pool.run(nm * d_chunks, [&](int t) {
            const int m = t / d_chunks, c = t % d_chunks;
            const void* hp[1] = {hq[(size_t) m].data()};
            float* op[1] = {part[(size_t) m].data()};
            KCPU::native_down_rows(fmt, cblob[(size_t) m], hp, 1, op, c * d_chunk, std::min(n_embd, (c + 1) * d_chunk));
        });
        lap(3);
    }

    bool gpu_ok = true;
    if (nm > 0) gpu_ok = gpu_done.get();
    else gpu_ok = gpu_job();
    if (!gpu_ok) {
        release_uploaded();
        err = "layer " + std::to_string(layer) + " device MoE: " + gpu_err;
        return false;
    }
    for (int j = 0; j < n_embd; ++j) {
        float acc = gpu_has ? gpu_out[(size_t) j] : 0.0f;
        for (int m = 0; m < nm; ++m) acc += part[(size_t) m][(size_t) j] * weights[miss[(size_t) m]];
        out[j] = acc;
    }

    // The uploaded share joins the cache (evicting the least recently used rows).
    for (const Uploaded& u : uploaded) {
        runtime.cache.insert(u.key, C::ExpertRowEntry{u.dev, layout.bytes});
        const C::ExpertRowEntry* entry = runtime.cache.find(u.key);
        if (!entry || entry->dev != u.dev) row_release(u.dev, layout.bytes);
        ++g_promoted;
    }
    return true;
}

static void prefetch_host_rows(const uint8_t* p, size_t bytes) {
    const uintptr_t pg = 4096;
    const uintptr_t a = (uintptr_t) p & ~(pg - 1);
    const uintptr_t e = ((uintptr_t) p + bytes + pg - 1) & ~(pg - 1);
    (void) madvise((void*) a, e - a, MADV_WILLNEED);
}

static bool run_cached_device_moe_batch(void* raw, int layer, int n_unique, const int32_t* experts,
                                       const uint8_t* const* blobs, int tokens, int n_used,
                                       const int32_t* route_slot, const float* weights,
                                       const KCPU::NativeFmt& fmt, const float* x, float* out, std::string& err) {
    GlmExpertDeviceRuntime& runtime = *(GlmExpertDeviceRuntime*) raw;
    const KN::NativeExpertLayout layout =
        KN::native_expert_layout(fmt.gu_type, fmt.d_type, fmt.n_embd, fmt.n_ff);
    if (!experts || !blobs || !route_slot || !weights || n_unique <= 0 || tokens <= 0 || tokens > 4096 ||
        layout.bytes == 0 || layout.bytes != fmt.bytes) {
        err = "batched device MoE layout/count disagrees with NativeFmt";
        return false;
    }
    if (runtime.scratch.n_embd == 0 && !runtime.scratch.alloc(fmt.n_embd, fmt.n_ff, err)) return false;
    struct PendingRow { C::ExpertRowKey key; void* dev; bool cacheable; };
    std::vector<PendingRow> pending;
    std::vector<const uint8_t*> rows((size_t) n_unique);
    for (int i = 0; i < n_unique; ++i)
        if (blobs[i] && runtime.cache.find(C::ExpertRowKey{layer, experts[i]}) == nullptr) prefetch_host_rows(blobs[i], layout.bytes);
    auto release_pending = [&]() { for (const PendingRow& p : pending) row_release(p.dev, layout.bytes); pending.clear(); };
    for (int i = 0; i < n_unique; ++i) {
        if (!blobs[i]) { err = "batched device MoE has a null host expert blob"; release_pending(); return false; }
        const C::ExpertRowKey key{layer, experts[i]};
        const C::ExpertRowState state = runtime.cache.lookup(key, layout.bytes);
        if (state == C::ExpertRowState::resident) {
            const C::ExpertRowEntry* entry = runtime.cache.find(key);
            if (!entry || !entry->dev || entry->bytes != layout.bytes) {
                err = "expert cache reported a resident batch row without a matching allocation";
                release_pending(); return false;
            }
            rows[(size_t) i] = (const uint8_t*) entry->dev;
            continue;
        }
        void* dev = row_alloc(layout.bytes);
        if (dev == nullptr ||
            cudaMemcpy(dev, blobs[i], layout.bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            if (dev) row_release(dev, layout.bytes);
            err = std::string("upload batched expert row: ") + cudaGetErrorString(cudaGetLastError());
            release_pending(); return false;
        }
        pending.push_back(PendingRow{key, dev, state == C::ExpertRowState::needs_upload});
        rows[(size_t) i] = (const uint8_t*) dev;
    }
    const bool ok = C::glm_expert_moe_device_batch_resident(rows.data(), n_unique, route_slot, weights, tokens,
                                                             n_used, layout, fmt.gu_type, fmt.d_type,
                                                             fmt.n_embd, fmt.n_ff, x, out, runtime.scratch, err);
    if (!ok) { release_pending(); if (err.empty()) err = "batched expert device FFN failed"; return false; }
    for (const PendingRow& p : pending) {
        if (!p.cacheable) { row_release(p.dev, layout.bytes); continue; }
        runtime.cache.insert(p.key, C::ExpertRowEntry{p.dev, layout.bytes});
        const C::ExpertRowEntry* entry = runtime.cache.find(p.key);
        if (!entry || entry->dev != p.dev) row_release(p.dev, layout.bytes);
    }
    pending.clear();
    return true;
}

static bool native_kda_project(int count, const void* const* weights, const int* types,
                               int tokens, const float* x, int n_in, int n_out, float* const* out) {
    struct Workspace {
        float* x = nullptr;
        float* y = nullptr;
        void* q = nullptr;
        cudaStream_t stream = nullptr;
        size_t x_bytes = 0, y_bytes = 0, q_bytes = 0;
        ~Workspace() { if (x) cudaFree(x); if (y) cudaFree(y); if (q) cudaFree(q); if (stream) cudaStreamDestroy(stream); }
    };
    static Workspace ws;
    if (tokens <= 0 || count < 1 || count > 3 || n_in <= 0 || n_out <= 0 || n_in > 16384 || n_out > 16384)
        return false;
    for (int i = 0; i < count; ++i) if (!strata::kernels::native_mmvq_supported(types[i])) return false;
    if (!ws.stream && cudaStreamCreateWithFlags(&ws.stream, cudaStreamNonBlocking) != cudaSuccess) return false;
    const int max_rows = std::min(tokens, 8);  // native_mmvq's supported multi-column width
    const size_t xb = (size_t) max_rows * n_in * sizeof(float);
    const size_t yb = (size_t) count * max_rows * n_out * sizeof(float);
    const size_t qb = strata::kernels::native_q8_1_bytes(n_in, max_rows);
    if (xb > ws.x_bytes || yb > ws.y_bytes || qb > ws.q_bytes) {
        if (cudaStreamSynchronize(ws.stream) != cudaSuccess) return false;
        if (ws.x) cudaFree(ws.x);
        if (ws.y) cudaFree(ws.y);
        if (ws.q) cudaFree(ws.q);
        ws.x = ws.y = nullptr; ws.q = nullptr;
        ws.x_bytes = ws.y_bytes = ws.q_bytes = 0;
        if (cudaMalloc(&ws.x, xb) != cudaSuccess || cudaMalloc(&ws.y, yb) != cudaSuccess ||
            cudaMalloc(&ws.q, qb) != cudaSuccess) return false;
        ws.x_bytes = xb; ws.y_bytes = yb; ws.q_bytes = qb;
    }
    for (int begin = 0; begin < tokens; begin += 8) {
        const int rows = std::min(8, tokens - begin);
        if (cudaMemcpyAsync(ws.x, x + (size_t) begin * n_in, (size_t) rows * n_in * sizeof(float),
                            cudaMemcpyHostToDevice, ws.stream) != cudaSuccess) return false;
        strata::kernels::native_quantize_q8_1(ws.x, ws.q, n_in, rows, (void*) ws.stream);
        for (int i = 0; i < count; ++i) {
            float* dy = ws.y + (size_t) i * rows * n_out;
            strata::kernels::native_mmvq(types[i], weights[i], ws.q, dy, n_in, n_out, rows, (void*) ws.stream);
            if (cudaMemcpyAsync(out[i] + (size_t) begin * n_out, dy, (size_t) rows * n_out * sizeof(float),
                                cudaMemcpyDeviceToHost, ws.stream) != cudaSuccess) return false;
        }
    }
    return cudaStreamSynchronize(ws.stream) == cudaSuccess;
}

static bool native_mla_project(int count, const void* const* weights, const int* types,
                               const float* x, int n_in, int n_out, float* const* out) {
    return native_kda_project(count, weights, types, 1, x, n_in, n_out, out);
}

static bool native_glm_ffn(const void* const* weights, const int* types, const K::MoeGeometry& g,
                           const float* x, float* out, float clamp_limit) {
    std::vector<float> gate((size_t) g.ff), up((size_t) g.ff), h((size_t) g.ff);
    float* gu_out[2] = {gate.data(), up.data()};
    if (!native_kda_project(2, weights, types, 1, x, g.n_embd, g.ff, gu_out)) return false;
    const bool clamp = clamp_limit > 1e-6f;
    for (int i = 0; i < g.ff; ++i) {
        float a = gate[(size_t) i], u = up[(size_t) i];
        if (clamp) { a = std::min(a, clamp_limit); u = std::max(-clamp_limit, std::min(u, clamp_limit)); }
        h[(size_t) i] = (a / (1.0f + std::exp(-a))) * u;
    }
    const void* down[1] = {weights[2]};
    const int down_type[1] = {types[2]};
    float* down_out[1] = {out};
    return native_kda_project(1, down, down_type, 1, h.data(), g.ff, g.n_embd, down_out);
}

static bool native_glm_ffn_batch(const void* const* weights, const int* types, const K::MoeGeometry& g,
                                int tokens, const float* x, float* out, float clamp_limit) {
    if (tokens <= 0 || !weights || !types || !x || !out) return false;
    std::vector<float> gate((size_t) tokens * g.ff), up((size_t) tokens * g.ff), h((size_t) tokens * g.ff);
    float* gu_out[2] = {gate.data(), up.data()};
    if (!native_kda_project(2, weights, types, tokens, x, g.n_embd, g.ff, gu_out)) return false;
    const bool clamp = clamp_limit > 1e-6f;
    for (int t = 0; t < tokens; ++t) for (int i = 0; i < g.ff; ++i) {
        const size_t j = (size_t) t * g.ff + i;
        float a = gate[j], u = up[j];
        if (clamp) { a = std::min(a, clamp_limit); u = std::max(-clamp_limit, std::min(u, clamp_limit)); }
        h[j] = (a / (1.0f + std::exp(-a))) * u;
    }
    const void* down[1] = {weights[2]};
    const int down_type[1] = {types[2]};
    float* down_out[1] = {out};
    return native_kda_project(1, down, down_type, tokens, h.data(), g.ff, g.n_embd, down_out);
}

static bool provider(void* raw, int layer, C::glm::GlmTrunkLayerWeights& out, std::string& err) {
    Provider* p = (Provider*) raw;
    if (layer < 0 || layer >= N_LAYERS) { err = "layer out of range"; return false; }
    // Retain CPU-consumed dequantized tensors per layer across tokens/requests. The current sweep materializes
    // about 1.55 GiB; freeing it each layer forced that same work to repeat for every prompt and decode token.
    {
        C::GlmBoundBlock& B = p->bound[(size_t) layer];
        const bool dense = (layer < 3);
        const bool kda_layer = C::glm::glm_attention_is_mla(layer) == 0;
        const bool routed_layer = !dense;
        for (size_t i = 0; i < B.tensors.size(); ++i) {
            C::GlmBoundBlock::Tensor& t = B.tensors[i];
            if (!t.quantized || t.ptr == nullptr) continue;
            const bool native_projection = kda_layer &&
                (t.name == "attn_q.weight" || t.name == "attn_k.weight" ||
                 t.name == "attn_v.weight" || t.name == "attn_output.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_shared = routed_layer &&
                (t.name == "ffn_gate_shexp.weight" || t.name == "ffn_up_shexp.weight" ||
                 t.name == "ffn_down_shexp.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_dense_ffn = dense &&
                (t.name == "ffn_gate.weight" || t.name == "ffn_up.weight" || t.name == "ffn_down.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            const bool native_mla = !kda_layer && !dense &&
                (t.name == "attn_q_a.weight" || t.name == "attn_q_b.weight" ||
                 t.name == "attn_kv_a_mqa.weight" || t.name == "attn_output.weight") &&
                strata::kernels::native_mmvq_supported(t.native_type);
            if (native_projection || native_shared || native_mla || native_dense_ffn) continue;
            // NO EXCEPTIONS, and the measurement is why: I first skipped the dense FFN's three tensors on blocks
            // 0..2, on the assumption that a GPU path consumed them.  It does not - glm_stage_ffn calls the CPU
            // expert_ffn, so the crash simply moved from kda_forward to expert_ffn when the rest were staged.  Every
            // quantized tensor is staged, and the per-layer lifetime is what keeps it affordable.
            (void) dense;
            float* h = dequant_to_host(t, err);
            if (h == nullptr) return false;
            p->staged_by_layer[(size_t) layer].push_back(h);
            g_staged_bytes += (unsigned long long) t.ne0 * (unsigned long long) (t.ne1 > 0 ? t.ne1 : 1) * 4ull;
            t.ptr = h;
            t.quantized = false;      // cached host representation remains valid for the provider lifetime
        }
        // re-map: glm_fill_layer_weights is pure assignment, so doing it per layer costs nothing
        if (!C::glm::glm_fill_layer_weights(B, layer, p->kda_g, p->mla_g, p->kda[(size_t) layer],
                                           p->mla[(size_t) layer], p->shexp_store[(size_t) layer].data(),
                                           p->w[(size_t) layer], err)) {
            err = "provider: re-mapping layer " + std::to_string(layer) + ": " + err;
            return false;
        }
        auto type_of = [&](const char* name) {
            for (const auto& t : B.tensors) if (t.name == name) return t.quantized ? t.native_type : 0;
            return 0;
        };
        if (kda_layer) {
            K::KdaWeights& kw = p->kda[(size_t) layer];
            kw.wq_type = type_of("attn_q.weight"); kw.wk_type = type_of("attn_k.weight");
            kw.wv_type = type_of("attn_v.weight"); kw.wo_type = type_of("attn_output.weight");
        } else {
            // THE MLA'S FOUR PROJECTIONS, typed the same way and for the same reason: a non-zero type tells mla_forward
            // that the pointer is the artifact's quantized blocks on the device, and the shared native projection hook
            // reads them in place.  These are the largest tensors in the model - q_b at 25,165,824 elements and wo at
            // 67,108,864 - so they are also the ones that stop being copied to host floats on every token.
            K::MlaWeights& mw = p->mla[(size_t) layer];
            mw.wq_a_type = type_of("attn_q_a.weight");
            mw.wq_b_type = type_of("attn_q_b.weight");
            mw.kv_a_type = type_of("attn_kv_a_mqa.weight");
            mw.wo_type = type_of("attn_output.weight");
        }
        if (dense) {
            C::glm::GlmTrunkLayerWeights& tw = p->w[(size_t) layer];
            tw.ffn_types[0] = type_of("ffn_gate.weight");
            tw.ffn_types[1] = type_of("ffn_up.weight");
            tw.ffn_types[2] = type_of("ffn_down.weight");
        }
        if (routed_layer) {
            C::glm::GlmTrunkLayerWeights& tw = p->w[(size_t) layer];
            tw.shexp_types[0] = type_of("ffn_gate_shexp.weight");
            tw.shexp_types[1] = type_of("ffn_up_shexp.weight");
            tw.shexp_types[2] = type_of("ffn_down_shexp.weight");
        }
    }
    out = p->w[(size_t) layer];                      // assembled from the staged host weights

    // ---- LAYER 0 BESIDE LAYER 4, because layer 0 runs and layer 4 stops on "null argument".  A dump of the failing
    // layer says which field is NULL; a differential says which field DIFFERS, and the difference is the defect.
    if ((layer == 0 || layer == 4) && out.kda != nullptr) {
        const K::KdaWeights& kw = *out.kda;
        const char* nm[16] = {"attn_norm", "wq", "wk", "wv", "conv_q", "conv_k", "conv_v", "ssm_a",
                              "dt_bias", "ssm_f_a", "ssm_f_b", "ssm_beta", "ssm_g_a", "ssm_g_b", "o_norm", "wo"};
        const float* pp[16] = {kw.attn_norm, kw.wq, kw.wk, kw.wv, kw.conv_q, kw.conv_k, kw.conv_v, kw.ssm_a,
                               kw.dt_bias, kw.ssm_f_a, kw.ssm_f_b, kw.ssm_beta, kw.ssm_g_a, kw.ssm_g_b, kw.o_norm, kw.wo};
        std::printf("  layer %d KdaWeights:\n", layer);
        for (int i = 0; i < 16; ++i) {
            std::printf("    %-10s %p%s\n", nm[i], (const void*) pp[i], pp[i] == nullptr ? "   <-- NULL" : "");
        }
    }
    const bool is_dense = (layer < 3);
    if (is_dense) {
        out.moe_g = &p->dense_g;
        out.clamp_limit = CLAMP;
    } else {
        if (!p->fmt_ready[(size_t) layer]) {
            err = "no expert descriptor for layer " + std::to_string(layer);
            return false;
        }
        out.moe_g = &p->moe_g;
        out.shexp_g = &p->shexp_g;
        out.shexp = p->shexp_store[(size_t) layer].data();
        out.moe_fmt = &p->fmt[(size_t) layer];
        out.blob_fn = &blob_adapter;
        out.blob_ctx = p->src;
        out.shexp_clamp = CLAMP;
    }
    return true;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: strata-glm <pack-dir> <gguf-shard1> [hc-init.bin | --serve] [w-output.bin] [w-norm.bin]\n");
        return 2;
    }
    const std::string pack = argv[1];
    const std::string shard1 = argv[2];
    const bool serve = argc > 3 && std::string(argv[3]) == "--serve";
    const std::string in_path = !serve && argc > 3 ? argv[3] : "/home/peb/moredata/glm5-oracle-full/hc_init.bin";
    const std::string wout_path = argc > (serve ? 4 : 4) ? argv[4] : "/home/peb/moredata/glm5-head-gate/w_output.bin";
    const std::string wnorm_path = argc > (serve ? 5 : 5) ? argv[5] : "/home/peb/moredata/glm5-head-gate/w_output_norm.bin";
    std::string err;

    // ---- all three GGUF shards: shard 1 is metadata-only, so a one-shard NativeDense sees an empty tensor list
    std::vector<std::string> shards;
    {
        const size_t slash = shard1.find_last_of('/');
        const std::string dir = slash == std::string::npos ? "" : shard1.substr(0, slash + 1);
        const std::string base = slash == std::string::npos ? shard1 : shard1.substr(slash + 1);
        const size_t pos = base.find("-00001-of-");
        if (pos != std::string::npos) {
            const std::string stem = base.substr(0, pos);
            const std::string tail = base.substr(base.find("-of-") + 4);
            const int n = std::atoi(tail.substr(0, tail.find('.')).c_str());
            for (int i = 1; i <= n; ++i) {
                char buf[4096];
                std::snprintf(buf, sizeof buf, "%s%s-%05d-of-%05d.gguf", dir.c_str(), stem.c_str(), i, n);
                shards.emplace_back(buf);
            }
        } else {
            shards.push_back(shard1);
        }
    }
    std::printf("shards: %zu\n", shards.size());

    std::set<std::string> served;
    if (!C::NativeDense::served_names(shards, false, served, err)) {
        std::fprintf(stderr, "served_names: %s\n", err.c_str());
        return 1;
    }
    C::WeightTable table;
    uint64_t bytes = 0;
    if (!table.pool_bytes(pack, bytes, err, &served)) { std::fprintf(stderr, "pool_bytes: %s\n", err.c_str()); return 1; }
    // MANAGED, not device-only.  The trunk's hc and norm stages are CPU kernels that read their weights directly
    // (hc_pre / rms_norm on the host), while the same bound block feeds the CUDA kernels - so the arena has to be
    // readable from both sides.  With cudaMalloc the host dereference is a SIGSEGV, which is exactly what the first
    // run of this tool produced, and the backtrace named it: hc_pre <- glm_stage_hc_norm <- glm_trunk_forward.  On a
    // V100 there is no host mapping for cudaMalloc memory, so the run must use managed memory (page-fault migration,
    // slower, correct) - a production loader would instead keep the small hc/norm weights in host memory.
    void* arena = nullptr;
    if (cudaMallocManaged(&arena, bytes, cudaMemAttachGlobal) != cudaSuccess) {
        std::fprintf(stderr, "managed arena alloc of %llu failed\n", (unsigned long long) bytes);
        return 1;
    }
    if (!table.load(pack, arena, bytes, err, &served)) { std::fprintf(stderr, "load: %s\n", err.c_str()); return 1; }
    C::NativeDense nd;
    if (!nd.load(shards, table, err)) { std::fprintf(stderr, "native: %s\n", err.c_str()); return 1; }
    C::NativeEmbed native_embed;
    if (serve && (shards.size() < 2 || !native_embed.load(shards, N_EMBD, 154880, err))) {
        std::fprintf(stderr, "embedding: %s\n", err.c_str());
        return 1;
    }
    // THE CUDA HEAD IS LOADED EVEN IN THE DIAGNOSTIC RUN.  It is not gated on serve any more, because the head is the
    // largest single item in the profile - 3.1 of 4.8 seconds per token for one 154,880 x 4,096 projection - and its
    // device path already exists here; gating it on serve meant the acceptance run measured the CPU projection and
    // reported it as the trunk's cost.  Unavailable is a fallback rather than a failure, so the gate still runs on a
    // machine or a shard set where the native head cannot load.
    C::NativeHead native_head;
    if (shards.size() < 2 || !native_head.load(shards, N_EMBD, 154880, err)) {
        std::fprintf(stderr, "output head: CUDA head unavailable (%s) - falling back to the CPU projection\n",
                     err.c_str());
    }
    const bool head_cuda = native_head.loaded();
    std::printf("head: %s\n", head_cuda ? "native (CUDA) over 154880 x 4096" : "CPU projection (fallback)");
    std::printf("arena %.2f GB, %zu tensors served from the GGUF\n", bytes / 1073741824.0, served.size());

    // ---- geometry.  The clamps are set EXPLICITLY: their defaults are 0.0f, which DISABLES them, and the header
    // says so in as many words.  A routed pre-activation past 10 would then be passed through unclamped, silently.
    Provider P;
    P.staged_by_layer.resize(N_LAYERS);
    K::kda_set_native_project(&native_kda_project);
    // The same function serves the MLA: the signature is already generic (weights, types, shapes), so there is no reason
    // for a second implementation to exist and drift from this one.
    K::mla_set_native_project(&native_mla_project);
    C::glm::glm_set_native_ffn(&native_glm_ffn);
    C::glm::glm_set_native_ffn_batch(&native_glm_ffn_batch);
    P.kda_g.n_embd = N_EMBD; P.kda_g.nh = NH; P.kda_g.hd = HD; P.kda_g.d_conv = 4;
    P.mla_g.n_embd = N_EMBD;                                   // its defaults are this artifact's MLA geometry
    P.dense_g.n_embd = N_EMBD; P.dense_g.ff = FF_DENSE; P.dense_g.n_expert = N_EXPERT; P.dense_g.n_used = N_USED;
    P.moe_g.n_embd = N_EMBD; P.moe_g.ff = FF_EXP; P.moe_g.n_expert = N_EXPERT; P.moe_g.n_used = N_USED;
    P.shexp_g.n_embd = N_EMBD; P.shexp_g.ff = FF_EXP;
    for (K::MoeGeometry* g : {&P.dense_g, &P.moe_g, &P.shexp_g}) {
        g->clamp_exp = CLAMP;
        g->clamp_shexp = CLAMP;
    }

    // ---- THE EXPERTS.  The layout must be loaded BEFORE the source is opened, and with a layer count that covers
    // the highest layer the file holds (45) - 43 fails as "malformed", because a row's layer number must be < n_layers.
    if (!KCPU::expert_layout_load(pack, 46, N_EXPERT, err, N_EMBD, FF_EXP)) {
        std::fprintf(stderr, "expert_layout_load: %s\n", err.c_str());
        return 1;
    }
    C::FileExpertSource src;
    if (!src.open(pack, 46, N_EXPERT, err)) { std::fprintf(stderr, "open: %s\n", err.c_str()); return 1; }
    P.src = &src;

    // RAM TIER.  Page-lock (cudaHostRegister) as much of the mapped expert file as the host can spare, so a GPU-cache
    // miss is a direct DMA at PCIe speed instead of a copy through the driver's pageable bounce buffer (about 4 GB/s
    // measured here against ~12 GB/s).  Automatic when the whole file fits in RAM with room to spare - the case the
    // engine is designed around - and otherwise opt-in via STRATA_GLM_PIN_GB, which pins the first N GB of layers.
    {
        const char* pin_env = std::getenv("STRATA_GLM_PIN_GB");
        double file_gb = 0.0, avail_gb = 0.0;
        struct stat sb;
        if (::stat((pack + "/experts.bin").c_str(), &sb) == 0) file_gb = (double) sb.st_size / 1073741824.0;
        if (FILE* mi = std::fopen("/proc/meminfo", "r")) {
            char line[256]; long kb = 0;
            while (std::fgets(line, sizeof line, mi))
                if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) { avail_gb = (double) kb / 1048576.0; break; }
            std::fclose(mi);
        }
        const double want_gb = pin_env ? std::atof(pin_env) : (file_gb > 0.0 && file_gb <= avail_gb * 0.8 ? file_gb : 0.0);
        if (want_gb > 0.0) {
            const int first_moe = 3;
            const uint8_t* lo = src.blob(first_moe, 0);
            const uint8_t* hi = lo;
            int pinned_layers = 0;
            for (int l = first_moe; lo && l < 46; ++l) {
                const uint8_t* b0 = src.blob(l, 0);
                const uint8_t* b1 = src.blob(l, 1);
                if (!b0 || !b1 || b0 != hi) break;     // stop at the first gap: one contiguous registration only
                const size_t layer_bytes = (size_t) (b1 - b0) * N_EXPERT;
                if ((double) ((b0 + layer_bytes) - lo) > want_gb * 1073741824.0) break;
                hi = b0 + layer_bytes;
                ++pinned_layers;
            }
            if (hi > lo) {
                const uintptr_t pg = 4096;
                const uintptr_t a = (uintptr_t) lo & ~(pg - 1);
                const uintptr_t e = ((uintptr_t) hi + pg - 1) & ~(pg - 1);
                cudaError_t pe = cudaHostRegister((void*) a, e - a, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
                if (pe != cudaSuccess) {
                    (void) cudaGetLastError();
                    pe = cudaHostRegister((void*) a, e - a, cudaHostRegisterPortable);
                }
                if (pe == cudaSuccess)
                    std::printf("RAM tier: %d MoE layers, %.1f GB page-locked (file %.1f GB, RAM available %.1f GB)\n",
                                pinned_layers, (double) (e - a) / 1073741824.0, file_gb, avail_gb);
                else {
                    (void) cudaGetLastError();
                    std::printf("RAM tier: cudaHostRegister of %.1f GB failed (%s); continuing pageable\n",
                                (double) (e - a) / 1073741824.0, cudaGetErrorString(pe));
                }
                std::fflush(stdout);
            }
        }
    }

    // Keep a bounded set of routed expert rows resident. The default leaves ample room for the model's other CUDA
    // allocations while fitting the measured hot set; misses upload once and then reuse the device pointer.
    const char* cache_gb_env = std::getenv("STRATA_GLM_EXPERT_CACHE_GB");
    const double cache_gb = cache_gb_env ? std::atof(cache_gb_env) : 16.0;
    GlmExpertDeviceRuntime expert_device_runtime((size_t) (cache_gb * 1024.0 * 1024.0 * 1024.0));
    g_reclaim_cache = &expert_device_runtime.cache;
    {
        const char* cm = std::getenv("STRATA_GLM_CPU_MISS");
        // Opt-in: with the 16 GB GPU cache the plain upload-and-cache path measured 5.1 t/s against 1.4 t/s for the
        // hybrid, which only wins (about +40%) when the conversation's experts are far larger than the card.
        g_cpu_miss = cm && std::strcmp(cm, "0") != 0;
        const char* pa = std::getenv("STRATA_GLM_PROMOTE_AFTER");
        if (pa) g_promote_after = std::max(1, std::atoi(pa));
        if (g_cpu_miss)
            std::printf("CPU tier: misses computed on %d threads beside the GPU; promoted to the GPU cache after %d misses\n",
                        cpu_pool().size(), g_promote_after);
        else
            std::printf("CPU tier: off (every miss is uploaded)\n");
    }
    // The prompt path's per-token hyper-connection stages are independent across tokens: spread them over the pool
    // (STRATA_GLM_CPU_THREADS sets its size; 0 workers = serial).
    C::glm::glm_set_parallel_for([](int n, const std::function<void(int)>& job) { cpu_pool().run(n, job); });
    // Prompt-path layer streamer (STRATA_GLM_STREAM_LAYERS=0 disables): see LayerStreamer.
    {
        const char* pn = std::getenv("STRATA_GLM_STREAM_LAYERS");
        if (!pn || std::strcmp(pn, "0") != 0) {
            uintptr_t ms = 0, me = 0;
            off_t mo = 0;
            if (FILE* mp = std::fopen("/proc/self/maps", "r")) {
                char line[1024];
                while (std::fgets(line, sizeof line, mp)) {
                    unsigned long s0, e0, off0;
                    if (std::strstr(line, "experts.bin") && std::sscanf(line, "%lx-%lx %*s %lx", &s0, &e0, &off0) == 3 &&
                        e0 - s0 > me - ms) { ms = s0; me = e0; mo = (off_t) off0; }
                }
                std::fclose(mp);
            }
            const int pfd = ::open((pack + "/experts.bin").c_str(), O_RDONLY | O_DIRECT);
            if (pfd >= 0 && me > ms && g_stream.init(pfd, &src, ms, me, mo, 3, 46)) {
                C::glm::glm_set_layer_prefetch(
                    [](void*, int layer) { if (layer == 0) g_stream.begin(3); g_stream.layer_started(layer); }, nullptr);
                std::printf("Layer streamer: O_DIRECT reads of whole layers, two ahead of the prompt path\n");
            } else {
                if (pfd >= 0) ::close(pfd);
                std::printf("Layer streamer: unavailable (open O_DIRECT / alignment); prompt path reads through the mmap\n");
            }
            std::fflush(stdout);
        }
    }
    // Host RAM tier (see HostRowCache).  Default: what the machine can spare beyond 9 GB of headroom.
    if (g_cpu_miss) {
        const char* hc = std::getenv("STRATA_GLM_HOST_CACHE_GB");
        double avail_gb = 0.0;
        if (FILE* mi = std::fopen("/proc/meminfo", "r")) {
            char line[256]; long kb = 0;
            while (std::fgets(line, sizeof line, mi))
                if (std::sscanf(line, "MemAvailable: %ld kB", &kb) == 1) { avail_gb = (double) kb / 1048576.0; break; }
            std::fclose(mi);
        }
        // Opt-in.  A tier only helps when RAM holds it AND the rest of the process (the GGUF-mapped non-expert weights
        // live in the page cache) without swapping: on the 31 GB development box a 16-21 GB tier made the kernel swap
        // it out and refault the other weights, 5x slower than no tier.  Size it to the expert working set with
        // several GB to spare, e.g. most of a 64 GB machine.
        (void) avail_gb;
        const double want_gb = hc ? std::atof(hc) : 0.0;
        if (want_gb > 0.0) {
            size_t slot = 0;
            if (std::ifstream ne(pack + "/native_experts.txt"); ne) {
                std::string ln;
                while (std::getline(ne, ln)) {
                    if (ln.empty() || ln[0] == '#') continue;
                    std::istringstream ls(ln);
                    long long f[5] = {0, 0, 0, 0, 0};
                    for (int i = 0; i < 5; ++i) ls >> f[i];
                    slot = std::max(slot, (size_t) f[4]);
                }
            }
            uintptr_t ms = 0, me = 0;
            off_t mo = 0;
            if (FILE* mp = std::fopen("/proc/self/maps", "r")) {
                char line[1024];
                while (std::fgets(line, sizeof line, mp)) {
                    unsigned long s0, e0, off0;
                    if (std::strstr(line, "experts.bin") && std::sscanf(line, "%lx-%lx %*s %lx", &s0, &e0, &off0) == 3 &&
                        e0 - s0 > me - ms) { ms = s0; me = e0; mo = (off_t) off0; }
                }
                std::fclose(mp);
            }
            if (slot > 0 && me > ms &&
                g_host.init((size_t) (want_gb * 1073741824.0), slot, pack + "/experts.bin", ms, me, mo)) {
                const char* drop = std::getenv("STRATA_GLM_HOST_DROP_CACHE");
                g_host.set_drop_page_cache(drop && std::strcmp(drop, "0") != 0);
                std::printf("Host RAM tier: %zu slots of %.1f MB (%.1f GB of anonymous memory)\n", g_host.slots(),
                            (double) g_host.slot_bytes() / 1048576.0,
                            (double) (g_host.slots() * g_host.slot_bytes()) / 1073741824.0);
            } else
                std::printf("Host RAM tier: unavailable (slot %zu, mapping %s); computing from the mmap\n", slot,
                            me > ms ? "found" : "not found");
            std::fflush(stdout);
        }
    }
    C::glm::glm_set_device_expert_ffn(&run_cached_device_expert, &expert_device_runtime);
    C::glm::glm_set_device_moe_ffn(&run_cached_device_moe, &expert_device_runtime);
    C::glm::glm_set_device_moe_batch_ffn(&run_cached_device_moe_batch, &expert_device_runtime);
    const char* kda_cuda_env = std::getenv("STRATA_GLM_KDA_CUDA");
    const bool kda_cuda = !kda_cuda_env || std::strcmp(kda_cuda_env,"0") != 0;
    K::kda_set_device_recurrence(kda_cuda);
    std::printf("KDA recurrence: %s\n",kda_cuda?"CUDA":"host");
    const char* kda_gates_env = std::getenv("STRATA_GLM_KDA_GATES");
    const bool kda_gates_cuda = !kda_gates_env || std::strcmp(kda_gates_env,"0") != 0;
    K::kda_set_device_gates(kda_gates_cuda);
    // Serve mode only: the resident KDA state stays on the device between calls (reset_state invalidates it).
    const char* kda_lazy_env = std::getenv("STRATA_GLM_KDA_LAZY_STATE");
    K::kda_set_lazy_state(serve && kda_cuda && (!kda_lazy_env || std::strcmp(kda_lazy_env,"0") != 0));
    std::printf("KDA gates: %s\n",kda_gates_cuda?"CUDA":"host");
    const char* mla_cuda_env = std::getenv("STRATA_GLM_MLA_CUDA");
    const bool mla_cuda = !mla_cuda_env || std::strcmp(mla_cuda_env,"0") != 0;
    K::mla_set_device_attention(mla_cuda);
    std::printf("MLA attention: %s\n",mla_cuda?"CUDA":"host");

    // ---- bind and map all 45 blocks
    P.bound.resize(N_LAYERS);
    P.kda.resize(N_LAYERS);
    P.mla.resize(N_LAYERS);
    P.shexp_store.resize(N_LAYERS);
    P.w.resize(N_LAYERS);
    P.fmt.resize(N_LAYERS);
    P.fmt_ready.assign(N_LAYERS, false);
    cudaStream_t stream = nullptr;
    for (int b = 0; b < N_LAYERS; ++b) {
        C::LayerView v(table, b);
        if (!bind_glm_block(v, b, 8192, 4, shards, P.bound[(size_t) b], (void*) stream, err)) {
            std::fprintf(stderr, "bind block %d: %s\n", b, err.c_str());
            return 1;
        }
        if (!C::glm::glm_fill_layer_weights(P.bound[(size_t) b], b, P.kda_g, P.mla_g, P.kda[(size_t) b],
                                            P.mla[(size_t) b], P.shexp_store[(size_t) b].data(),
                                            P.w[(size_t) b], err)) {
            std::fprintf(stderr, "fill block %d: %s\n", b, err.c_str());
            return 1;
        }
        if (b >= 3) {
            int gu = 0, dt = 0;
            const std::string te = layer_types(pack, b, gu, dt);
            if (!te.empty()) { std::fprintf(stderr, "%s\n", te.c_str()); return 1; }
            if (!KCPU::native_fmt(gu, dt, N_EMBD, FF_EXP, P.fmt[(size_t) b], err)) {
                std::fprintf(stderr, "native_fmt(layer %d, %d/%d): %s\n", b, gu, dt, err.c_str());
                return 1;
            }
            P.fmt_ready[(size_t) b] = true;
        }
    }
    std::printf("bound and mapped %d blocks\n", N_LAYERS);

    // ---- the hc weights need no workaround any more.
    //
    // This block used to copy hc_attn_fn and hc_ffn_fn to host, because the pack's dequantized weights came back as
    // device pointers while the trunk's kernels are CPU-side.  That copy moved the crash from hc_pre to kda_forward,
    // which proved the diagnosis; the fix now lives in the LOADER (glm_bind.cpp), where the dequantized floats land in
    // host memory for every tensor that needs it.  The runner no longer patches weights it did not load - and the
    // cudaMemcpy it used to do now fails with "invalid argument", because its source is host memory, which is the
    // signal that the loader is finally doing its job.

    // ---- DIAGNOSTIC: are the hc/norm weights the trunk's CPU stages read actually host-readable?
    //
    // Every previous caller of glm_trunk_forward passed FIXTURE arrays living in ordinary host memory, so this could
    // not come up before.  The bound pack puts those tensors somewhere, and hc_pre - a CPU kernel - segfaults on it.
    // An address inside the arena would mean the arena itself needs a host mapping; an address outside means the
    // binder allocated privately and that allocation is the thing to inspect.
    {
        const char* lo = (const char*) arena;
        const char* hi = lo + bytes;
        std::printf("arena range: %p .. %p\n", (const void*) lo, (const void*) hi);
        const int probe[2] = {0, 3};
        for (int k = 0; k < 2; ++k) {
            const int b = probe[k];
            const C::glm::GlmTrunkLayerWeights& ww = P.w[(size_t) b];
            const void* ptrs[4] = {(const void*) ww.hc_attn_fn, (const void*) ww.hc_attn_base,
                                   (const void*) ww.hc_attn_scale, (const void*) ww.attn_norm};
            const char* names[4] = {"hc_attn_fn", "hc_attn_base", "hc_attn_scale", "attn_norm"};
            std::printf("  layer %2d:", b);
            for (int j = 0; j < 4; ++j) {
                const char* where = "NULL";
                if (ptrs[j] != nullptr) {
                    const char* c = (const char*) ptrs[j];
                    where = (c >= lo && c < hi) ? "in-arena" : "OUTSIDE-arena";
                }
                std::printf(" %s=%p(%s)", names[j], ptrs[j], where);
            }
            std::printf("\n");
            std::fflush(stdout);
        }
    }

    // ---- WHICH TENSORS CAN A CPU KERNEL NOT READ?
    //
    // This is the list the loader has to be fixed against: every tensor the pack serves QUANTIZED comes back as a
    // device pointer, and the trunk's hc and KDA stages are CPU kernels.  Printed per tensor rather than inferred,
    // because the fields the weight structs point at are exactly the ones that have to move to host memory.
    {
        const char* lo = (const char*) arena;
        const char* hi = lo + bytes;
        const int probe[2] = {0, 3};
        for (int k = 0; k < 2; ++k) {
            const int b = probe[k];
            size_t outside = 0, inside = 0;
            std::printf("  layer %d bound tensors outside the arena:", b);
            for (const C::GlmBoundBlock::Tensor& t : P.bound[(size_t) b].tensors) {
                const char* c = (const char*) t.ptr;
                if (t.ptr != nullptr && (c < lo || c >= hi)) {
                    ++outside;
                    std::printf(" %s", t.name.c_str());
                } else {
                    ++inside;
                }
            }
            std::printf("\n    (%zu outside, %zu inside the arena)\n", outside, inside);
            std::fflush(stdout);
        }
    }

    // ---- state: one KDA state per KDA layer, one MLA cache per MLA layer, both keyed by the ARTIFACT's layer number
    const int cache_cells = serve ? 8192 : TOKENS;
    std::vector<std::vector<float> > kda_state(N_LAYERS);
    std::vector<std::vector<float> > kda_conv(N_LAYERS);
    std::vector<std::vector<float> > mla_cache(N_LAYERS);
    float* kda_ptrs[N_LAYERS];
    float* kda_conv_ptrs[N_LAYERS];
    float* mla_ptrs[N_LAYERS];
    int kda_len[N_LAYERS];
    int mla_len[N_LAYERS];
    int kda_index[N_LAYERS], mla_index[N_LAYERS];
    int n_kda = 0, n_mla = 0;
    for (int b = 0; b < N_LAYERS; ++b) {
        kda_index[b] = -1;
        mla_index[b] = -1;
        kda_len[b] = 0;
        mla_len[b] = 0;
        // FILLED BY SLOT, NOT BY LAYER, and that distinction is a real bug I had.
        //
        // The trunk reads state.kda_state[slot] and state.mla_cache[slot], where slot is kda_index[layer] /
        // mla_index[layer] - so those two arrays are keyed by SLOT.  This runner was filling them by LAYER.  For blocks
        // 0, 1 and 2 every block is KDA, so slot == layer and the two agree by accident; block 3 is MLA and takes no KDA
        // slot, so from block 4 onward the mapping shifts by one and kda_state[slot] lands on the index that block 3
        // would have had - never assigned, hence whatever the array held, and a fault named as a null argument.
        //
        // The same one-line shift applies to the MLA caches in the other direction, which is very likely what the
        // layer-3 pointer differential was showing when four device pointers came back identical across two layers.
        if (C::glm::glm_attention_is_mla(b) == 1) {
            mla_cache[(size_t) b].assign((size_t) cache_cells * KV_LORA, 0.0f);
            mla_index[b] = n_mla;
            mla_ptrs[n_mla] = mla_cache[(size_t) b].data();
            ++n_mla;
        } else {
            kda_state[(size_t) b].assign((size_t) NH * HD * HD, 0.0f);
            kda_conv[(size_t) b].assign((size_t) 3 * (P.kda_g.d_conv - 1) * P.kda_g.d_inner(), 0.0f);
            kda_index[b] = n_kda;
            kda_ptrs[n_kda] = kda_state[(size_t) b].data();
            kda_conv_ptrs[n_kda] = kda_conv[(size_t) b].data();
            ++n_kda;
        }
    }
    std::printf("state: %d KDA slots (%.0f MB), %d MLA caches (%d cells each)\n", n_kda,
                n_kda * (double) NH * HD * HD * 4.0 / 1048576.0, n_mla, cache_cells);

    C::glm::GlmTrunkState st;
    st.kda_state = kda_ptrs;
    st.kda_conv = kda_conv_ptrs;
    st.kda_index = kda_index;
    st.mla_cache = mla_ptrs;
    st.mla_len = mla_len;
    st.mla_index = mla_index;

        // ---- the head: load its small norm and large projection once.  Serve mode keeps both resident across requests.
    std::vector<float> onorm;
    int ne_n[4] = {0, 0, 0, 0};
    // ---- output_norm: read it, and REFUSE rather than pass an empty vector on.
    //
    // This is the whole of the head's null.  onorm.data() on an EMPTY std::vector is guaranteed to be nullptr - not
    // garbage, nullptr - so when read_dump parsed this file and produced no elements, the null the head reported was
    // created right here, one line earlier, and the head was telling the truth about an argument this code handed it.
    // The lesson is the same one this port keeps relearning: an argument that is null is not always a binding failure.
    bool have_norm = false;
    if (serve) {
        const C::WeightRef* nr = table.find("output_norm.weight");
        if (nr && nr->data && nr->bytes == (uint64_t) N_EMBD * sizeof(float)) {
            onorm.assign((const float*) nr->data, (const float*) nr->data + N_EMBD);
            have_norm = true;
        }
    } else {
        have_norm = read_dump(wnorm_path, onorm, ne_n) && (int) onorm.size() == N_EMBD;
    }
    if (!have_norm) {
        onorm.assign((size_t) N_EMBD, 0.0f);
        std::FILE* f = std::fopen(wnorm_path.c_str(), "rb");
        if (f != nullptr) {
            const size_t got = std::fread(onorm.data(), sizeof(float), (size_t) N_EMBD, f);
            std::fclose(f);
            have_norm = (got == (size_t) N_EMBD);
        }
    }
    if (!have_norm) {
        std::fprintf(stderr, "output_norm: %s is neither a dump with %d elements nor %d raw floats\n",
                     wnorm_path.c_str(), N_EMBD, N_EMBD);
        return 1;
    }
    {
        double s2 = 0.0;
        for (float v : onorm) s2 += (double) v * v;
        std::printf("output_norm: %zu floats, rms %.6g, first 3: %.6g %.6g %.6g\n", onorm.size(),
                    std::sqrt(s2 / (double) onorm.size()), onorm[0], onorm[1], onorm[2]);
    }
    const long vocab = 154880;
    std::vector<float> output_w;
    if (!serve) {
        std::ifstream wf(wout_path, std::ios::binary);
        if (!wf) { std::fprintf(stderr, "cannot read %s\n", wout_path.c_str()); return 1; }
        output_w.resize((size_t) vocab * N_EMBD);
        wf.read((char*) output_w.data(), (std::streamsize) (output_w.size() * sizeof(float)));
        if (!wf) { std::fprintf(stderr, "short read from %s\n", wout_path.c_str()); return 1; }
    }

    std::vector<float> l_out((size_t) HC * N_EMBD), hidden((size_t) N_EMBD), x((size_t) HC * N_EMBD);
    float *d_embed = nullptr, *d_hidden = nullptr, *d_logits = nullptr;
    cudaStream_t head_stream = nullptr;
    std::vector<float> logits(head_cuda ? (size_t) vocab : 0);
    if (head_cuda && (cudaMalloc(&d_embed, (size_t) N_EMBD * sizeof(float)) != cudaSuccess ||
                  cudaMalloc(&d_hidden, (size_t) N_EMBD * sizeof(float)) != cudaSuccess ||
                  cudaMalloc(&d_logits, (size_t) vocab * sizeof(float)) != cudaSuccess ||
                  cudaStreamCreateWithFlags(&head_stream, cudaStreamNonBlocking) != cudaSuccess)) {
        std::fprintf(stderr, "embedding/head scratch allocation failed\n"); return 1;
    }
    // Prompt embeddings do not depend on trunk state. Gather them in small batches so a prompt
    // pays one launch/sync/copy per 128 tokens instead of one per token. The trunk itself remains
    // sequential because KDA recurrence and MLA cache updates are causal.
    constexpr int EMBED_BATCH = 4096;
    const size_t trunk_row = (size_t) HC * N_EMBD;
    int32_t* d_prompt_tokens = nullptr;
    float* d_prompt_embeds = nullptr;
    bool prompt_batch_embed = false;
    std::vector<int32_t> prompt_token_ids(EMBED_BATCH);
    std::vector<float> prompt_embed_rows((size_t) EMBED_BATCH * N_EMBD);
    std::vector<float> prompt_inputs((size_t) EMBED_BATCH * trunk_row);
    std::vector<float> prompt_outputs((size_t) EMBED_BATCH * trunk_row);
    if (serve && head_cuda &&
        cudaMalloc(&d_prompt_tokens, (size_t) EMBED_BATCH * sizeof(int32_t)) == cudaSuccess &&
        cudaMalloc(&d_prompt_embeds, (size_t) EMBED_BATCH * N_EMBD * sizeof(float)) == cudaSuccess) {
        prompt_batch_embed = true;
    } else {
        if (d_prompt_tokens) cudaFree(d_prompt_tokens);
        if (d_prompt_embeds) cudaFree(d_prompt_embeds);
        d_prompt_tokens = nullptr;
        d_prompt_embeds = nullptr;
        cudaGetLastError();  // the scalar gather remains a valid fallback if this optional scratch did not fit
    }
    auto reset_state = [&]() {
        for (auto& v : kda_state) std::fill(v.begin(), v.end(), 0.0f);
        K::kda_invalidate_state();
        for (auto& v : kda_conv) std::fill(v.begin(), v.end(), 0.0f);
        for (auto& v : mla_cache) std::fill(v.begin(), v.end(), 0.0f);
        for (int i = 0; i < N_LAYERS; ++i) mla_len[i] = 0;
    };
    auto run_head = [&](const float* trunk_out, int& best, float& best_logit) -> bool {
        if (!C::glm::glm_stage_head_mean_norm(trunk_out, HC, N_EMBD, onorm.data(), hidden.data(), err)) return false;
        if (!head_cuda)
            return C::glm::glm_stage_head_project(output_w.data(), (int) vocab, N_EMBD, hidden.data(), best,
                                                  best_logit, err);
        if (cudaMemcpyAsync(d_hidden, hidden.data(), (size_t) N_EMBD * sizeof(float), cudaMemcpyHostToDevice,
                            head_stream) != cudaSuccess ||
            !native_head.run(d_hidden, d_logits, (void*) head_stream, err) ||
            cudaMemcpyAsync(logits.data(), d_logits, (size_t) vocab * sizeof(float), cudaMemcpyDeviceToHost,
                            head_stream) != cudaSuccess || cudaStreamSynchronize(head_stream) != cudaSuccess) {
            if (err.empty()) err = "native output head failed";
            return false;
        }
        best = (int) std::distance(logits.begin(), std::max_element(logits.begin(), logits.end()));
        best_logit = logits[(size_t) best];
        return true;
    };
    auto run_x = [&](const float* in, int pos, bool diagnostic, bool need_logits, int& best, float& best_logit) -> bool {
        StageCount sc; sc.token = pos;
        if (!C::glm::glm_trunk_forward(in, N_LAYERS, provider, &P, P.kda_g, P.mla_g, EPS, st, l_out.data(), nullptr,
                                       err, 0, diagnostic ? &stage_cb : nullptr, diagnostic ? &sc : nullptr)) return false;
        // During prompt prefill only the final position's logits are consumed.  Projecting the full
        // vocabulary at every earlier position adds a large redundant head pass and device sync.
        if (!need_logits) return true;
        return run_head(l_out.data(), best, best_logit);
    };
    auto embed = [&](int64_t tok) -> bool {
        if (tok < 0 || tok >= vocab) { err = "token outside vocabulary"; return false; }
        native_embed.gather_one(tok, d_embed, nullptr);
        if (cudaDeviceSynchronize() != cudaSuccess ||
            cudaMemcpy(x.data(), d_embed, (size_t) N_EMBD * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "token embedding gather failed"; return false;
        }
        for (int h = 1; h < HC; ++h)
            std::memcpy(x.data() + (size_t) h * N_EMBD, x.data(), (size_t) N_EMBD * sizeof(float));
        return true;
    };

    if (serve) {
        std::printf("READY glm-5.3-flash %ld 8192\n", vocab); std::fflush(stdout);
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            if (line.rfind("GEN ", 0) != 0) { std::printf("ERR expected GEN <max_new> <id,id,...>\n"); std::fflush(stdout); continue; }
            char* ep = nullptr; long max_new = std::strtol(line.c_str() + 4, &ep, 10);
            while (ep && *ep == ' ') ++ep;
            std::vector<int64_t> ids; std::string item; std::stringstream ss(ep ? ep : "");
            while (std::getline(ss, item, ',')) if (!item.empty()) ids.push_back(std::strtoll(item.c_str(), nullptr, 10));
            if (max_new < 1 || ids.empty() || ids.size() + (size_t) max_new > (size_t) cache_cells) {
                std::printf("ERR invalid request or context too long\n"); std::fflush(stdout); continue;
            }
            bool ids_valid = true;
            for (int64_t tok : ids) if (tok < 0 || tok >= vocab) { ids_valid = false; break; }
            if (!ids_valid) { std::printf("ERR token outside vocabulary\n"); std::fflush(stdout); continue; }
            reset_state(); int best = -1; float logit = 0.0f; bool ok = true; int pos = 0;
            for (size_t begin = 0; begin < ids.size() && ok; begin += EMBED_BATCH) {
                const int count = (int) std::min((size_t) EMBED_BATCH, ids.size() - begin);
                const float* gathered = nullptr;
                if (prompt_batch_embed) {
                    for (int i = 0; i < count; ++i) prompt_token_ids[(size_t) i] = (int32_t) ids[begin + (size_t) i];
                    if (cudaMemcpy(d_prompt_tokens, prompt_token_ids.data(), (size_t) count * sizeof(int32_t),
                                   cudaMemcpyHostToDevice) != cudaSuccess) {
                        err = "prompt token batch upload failed"; ok = false; break;
                    }
                    native_embed.gather_dev(d_prompt_tokens, count, d_prompt_embeds, nullptr);
                    if (cudaDeviceSynchronize() != cudaSuccess ||
                        cudaMemcpy(prompt_embed_rows.data(), d_prompt_embeds,
                                   (size_t) count * N_EMBD * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
                        err = "prompt embedding batch gather failed"; ok = false; break;
                    }
                    gathered = prompt_embed_rows.data();
                }
                for (int i = 0; i < count; ++i) {
                    const int64_t tok = ids[begin + (size_t) i];
                    if (gathered) {
                        std::memcpy(x.data(), gathered + (size_t) i * N_EMBD, (size_t) N_EMBD * sizeof(float));
                        for (int h = 1; h < HC; ++h)
                            std::memcpy(x.data() + (size_t) h * N_EMBD, x.data(), (size_t) N_EMBD * sizeof(float));
                    } else if (!embed(tok)) { ok = false; break; }
                    std::memcpy(prompt_inputs.data() + (size_t) i * trunk_row, x.data(), trunk_row * sizeof(float));
                }
                if (!ok) break;
                ensure_prefill_headroom(count);
                const bool chunk_ok = C::glm::glm_trunk_forward_batch(prompt_inputs.data(), count, N_LAYERS, provider, &P,
                                                                      P.kda_g, P.mla_g, EPS, st, prompt_outputs.data(),
                                                                      nullptr, err);
                g_stream.end();                    // stop reading ahead; decode goes back to the mmap
                if (!chunk_ok) { ok = false; break; }
                pos += count;
                if (begin + (size_t) count == ids.size() &&
                    !run_head(prompt_outputs.data() + (size_t) (count - 1) * trunk_row, best, logit)) ok = false;
            }
            long produced = 0;
            while (ok && produced < max_new) {
                std::printf("T %d\n", best); std::fflush(stdout); ++produced;
                if (best == 154820 || best == 154827 || produced == max_new) break;
                if (!embed(best) || !run_x(x.data(), pos++, false, true, best, logit)) ok = false;
            }
            if (std::getenv("STRATA_GLM_TIMING"))
                std::fprintf(stderr, "CPU tier ms (cumulative): populate %.0f  gate/up %.0f  quant %.0f  down %.0f  gpu-wait %.0f  promote %.0f  lookup %.0f  fill-thread-sum %.0f | cpu evals %ld gpu evals %ld promoted %ld | host hits %ld fills %ld\n",
                             g_cpu_ms[0], g_cpu_ms[1], g_cpu_ms[2], g_cpu_ms[3], g_cpu_ms[4], g_cpu_ms[5], g_cpu_ms[6],
                             (double) g_fill_ns.load() / 1e6, g_cpu_experts, g_gpu_experts, g_promoted, g_host_hits, g_host_fills);
            if (ok) std::printf("DONE %ld %zu\n", produced, ids.size());
            else std::printf("ERR %s\n", err.c_str());
            std::fflush(stdout);
        }
        cudaFree(d_embed);
        cudaFree(d_prompt_tokens);
        cudaFree(d_prompt_embeds);
        cudaFree(d_hidden);
        cudaFree(d_logits);
        cudaStreamDestroy(head_stream);
        std::fprintf(stderr, "%s\n", expert_device_runtime.cache.report().c_str());
        std::fprintf(stderr, "CPU tier: %ld expert evaluations on the CPU, %ld on the GPU, %ld rows promoted\n",
                     g_cpu_experts, g_gpu_experts, g_promoted);
        std::fprintf(stderr, "CPU tier ms: fill/populate %.0f  gate/up %.0f  quant %.0f  down %.0f  gpu-wait %.0f  promote %.0f\n",
                     g_cpu_ms[0], g_cpu_ms[1], g_cpu_ms[2], g_cpu_ms[3], g_cpu_ms[4], g_cpu_ms[5]);
        std::fprintf(stderr, "Host RAM tier: %ld hits, %ld fills\n", g_host_hits, g_host_fills);
        return 0;
    }

    // Diagnostic acceptance run: preserve the oracle fixture gate.
    std::vector<float> inp; int ne_in[4] = {0,0,0,0};
    if (!read_dump(in_path, inp, ne_in) || ne_in[0] != N_EMBD || ne_in[1] != HC || ne_in[2] < TOKENS) {
        std::fprintf(stderr, "cannot read compatible hc_init from %s\n", in_path.c_str()); return 1;
    }
    reset_state(); int best = -1; float best_logit = 0.0f;
    std::vector<float> batch_out(inp.size());
    if (!C::glm::glm_trunk_forward_batch(inp.data(), TOKENS, N_LAYERS, provider, &P, P.kda_g, P.mla_g, EPS, st,
                                         batch_out.data(), nullptr, err) ||
        !run_head(batch_out.data() + (size_t) (TOKENS - 1) * trunk_row, best, best_logit)) {
        std::fprintf(stderr, "batched trunk failed: %s\n", err.c_str()); return 1;
    }
    {
        double total = 0.0, stem = 0.0, kda_ms = 0.0, mla_ms = 0.0;
        for (const std::pair<const int, double>& kv : g_prof.ms) {
            total += kv.second;
            if (kv.first < 3) stem += kv.second;
            else if (C::glm::glm_attention_is_mla(kv.first) == 1) mla_ms += kv.second;
            else kda_ms += kv.second;
        }
        if (total > 0.0) {
            std::pair<int, double> worst(-1, -1.0);
            for (const std::pair<const int, double>& kv : g_prof.ms) if (kv.second > worst.second) worst = kv;
            std::fprintf(stderr, "PROFILE %.1f ms for %d tokens = %.1f ms/token (%.2f tok/s)\n",
                         total, TOKENS, total / TOKENS, 1000.0 * TOKENS / total);
            std::fprintf(stderr, "  stem 0-2 : %8.1f ms %5.1f%%\n", stem, 100.0 * stem / total);
            std::fprintf(stderr, "  34 KDA   : %8.1f ms %5.1f%%\n", kda_ms, 100.0 * kda_ms / total);
            std::fprintf(stderr, "  11 MLA   : %8.1f ms %5.1f%%\n", mla_ms, 100.0 * mla_ms / total);
            std::fprintf(stderr, "  heaviest block: layer %d at %.1f ms\n", worst.first, worst.second);
            std::fprintf(stderr, "  host staging: %.2f GB cached across layers and materialised once\n",
                         (double) g_staged_bytes / 1073741824.0);
                    {
            std::vector<std::pair<std::string, double> > v(g_prof.by_stage.begin(), g_prof.by_stage.end());
            std::sort(v.begin(), v.end(), [](const std::pair<std::string, double>& a,
                                             const std::pair<std::string, double>& b) { return a.second > b.second; });
            std::fprintf(stderr, "  by stage (interval closed by the named callback):\n");
            for (size_t i = 0; i < v.size() && i < 10; ++i)
                std::fprintf(stderr, "    %-16s %8.1f ms %5.1f%%\n", v[i].first.c_str(), v[i].second,
                             100.0 * v[i].second / total);
        }
        std::fprintf(stderr, "  per layer ms:");
            for (const std::pair<const int, double>& kv : g_prof.ms) std::fprintf(stderr, " %d:%.1f", kv.first, kv.second);
            std::fprintf(stderr, "\n");
        }
    }
    // MAJOR VS MINOR FAULTS, because it is the number that says whether the routed-expert stage's 861.7 ms per token is
    // really disk I/O: a major fault is a page that had to come from the device, a minor fault is one that was already
    // in the page cache.  0.44 GB per token at 4 KB pages is about 115,000 major faults per token if the expert rows are
    // genuinely cold every time; far fewer means they are cache hits and the stage is CPU dequantisation after all,
    // which would send the work back to a device MMVQ kernel rather than to a residency plan.
    {
        struct rusage ru;
        std::memset(&ru, 0, sizeof ru);
        if (getrusage(RUSAGE_SELF, &ru) == 0) {
            std::fprintf(stderr, "FAULTS minflt %ld majflt %ld inblock %ld oublock %ld\n", ru.ru_minflt, ru.ru_majflt,
                         ru.ru_inblock, ru.ru_oublock);
            std::fprintf(stderr, "       majflt x 4096 = %.2f GB of pages faulted in from the device\n",
                         (double) ru.ru_majflt * 4096.0 / 1073741824.0);
        }
    }
    K::kda_print_profile();
    std::printf("\nargmax = %d   (logit %.6f)   expected 12089\n", best, best_logit);
    std::printf("ENGINE TOKEN: %s\n", best == 12089 ? "PASS" : "MISMATCH");
    std::fprintf(stderr, "%s\n", expert_device_runtime.cache.report().c_str());
    return best == 12089 ? 0 : 1;
}
