// include/strata/core/expert_row_cache.hpp - which expert rows are resident on the device, and which to drop.
//
// WHY A CACHE AT ALL, and it is a measured decision rather than a guess.  The routed experts use 8 of 288 experts per
// layer per token, and one expert's gate+up+down is 6.5-9.1 MB, so uploading every selected expert for every token is
// 2.40 GB per token: 300 ms at 8 GB/s of PCIe and 200 ms at 12 GB/s, against the roughly 360 ms of CPU dequantisation a
// device path would replace.  That is a wash at best and a loss if the MMVQ calls are less than perfectly overlapped -
// exactly the mistake the host staging already cost this port, one bus further out.  With the hot set RESIDENT, the same
// 2.40 GB is a one-off: it fits thirteen times over in the V100's 32 GB, and after a few tokens the PCIe traffic is
// nearly zero while the dequantisation saving is collected in full.  So the cache is not an optimisation on top of the
// device expert path; it is the part that makes the device expert path worth having.
//
// WHAT IS DELIBERATELY NOT HERE: no CUDA, no device allocation, no upload.  This is the policy - keys, bytes, a budget,
// least-recently-used eviction - and nothing else, so it can be tested without a GPU and so the upload half cannot hide a
// policy bug behind a cudaMalloc.  The two are joined by the stage that drives them, which is where a reader looking for
// "why is this row on the device" should end up.
#pragma once

#include <cstddef>
#include <cstdint>
#include <list>
#include <string>
#include <unordered_map>

namespace strata {
namespace core {

/// What identifies a resident row: one expert of one layer.  The types travel with it because a row's byte length is
/// type-dependent and the load path must be able to re-derive the ranges without consulting the pack again.
struct ExpertRowKey {
    int layer = -1;
    int expert = -1;
    bool operator==(const ExpertRowKey& o) const { return layer == o.layer && expert == o.expert; }
};

struct ExpertRowKeyHash {
    std::size_t operator()(const ExpertRowKey& k) const noexcept {
        return (std::size_t) ((std::uint64_t) (std::uint32_t) k.layer << 32 | (std::uint32_t) k.expert);
    }
};

/// One expert's three projections, as they sit on the device.  `dev` is whatever the uploader returned; this class never
/// interprets it.
struct ExpertRowEntry {
    void* dev = nullptr;       ///< the uploader's handle, opaque here
    std::size_t bytes = 0;     ///< gate + up + down, the amount charged against the budget
};

struct ExpertRowCacheConfig {
    /// How many bytes of device memory the cache may hold.  The measured hot set is about 2.40 GB for all 43 layers'
    /// eight experts; a budget below one layer's eight experts (about 60 MB) makes the cache useless, so the class
    /// reports that rather than silently thrashing.
    std::size_t budget_bytes = 0;
    /// Optional sink for eviction notices, so the caller can free whatever `dev` points at.  Never null-checked for
    /// correctness: a cache that cannot free has a caller bug, not a cache bug.
    void (*on_evict)(void* ctx, const ExpertRowKey& key, const ExpertRowEntry& entry) = nullptr;
    void* evict_ctx = nullptr;
};

/// Result of asking for a row.
enum class ExpertRowState {
    resident,       ///< already on the device, nothing to do
    needs_upload,   ///< not resident and there is room once the LRU entries are dropped
    cannot_fit,     ///< the row is larger than the entire budget, so it can never be cached
};

class ExpertRowCache {
public:
    explicit ExpertRowCache(const ExpertRowCacheConfig& cfg) : cfg_(cfg) {}

    ExpertRowState lookup(const ExpertRowKey& key, std::size_t bytes);
    /// Read a resident entry after lookup() reports resident; nullptr when the key is absent.
    const ExpertRowEntry* find(const ExpertRowKey& key) const;

    /// Record a row as resident.  Pays for it by dropping least-recently-used entries; the caller is expected to have
    /// called lookup() first and to upload only when it said needs_upload.
    void insert(const ExpertRowKey& key, const ExpertRowEntry& entry);

    /// Make an existing row the most recently used.  Called on every use, which is what makes this LRU and not FIFO.
    void touch(const ExpertRowKey& key);

    void clear();

    /// Drop the least recently used row (through on_evict) to give its memory back; false when the cache is empty.
    /// For a caller that ran out of device memory for something more urgent than the cache.
    bool trim_one();

    std::size_t bytes_used() const { return used_; }
    std::size_t budget() const { return cfg_.budget_bytes; }
    std::size_t entries() const { return map_.size(); }
    std::uint64_t hits() const { return hits_; }
    std::uint64_t misses() const { return misses_; }
    std::uint64_t evictions() const { return evictions_; }
    std::uint64_t refusals() const { return refusals_; }

    /// A one-line summary for a run's report.
    std::string report() const;

private:
    void evict_until(std::size_t want);

    ExpertRowCacheConfig cfg_;
    std::unordered_map<ExpertRowKey, ExpertRowEntry, ExpertRowKeyHash> map_;
    std::list<ExpertRowKey> lru_;                                  ///< front = most recently used
    std::unordered_map<ExpertRowKey, std::list<ExpertRowKey>::iterator, ExpertRowKeyHash> pos_;
    std::size_t used_ = 0;
    std::uint64_t hits_ = 0, misses_ = 0, evictions_ = 0, refusals_ = 0;
};

}  // namespace core
}  // namespace strata
