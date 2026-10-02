// src/core/expert_row_cache.cpp - the policy half of the device-resident expert cache.  See the header.
#include "strata/core/expert_row_cache.hpp"

#include <cstdio>

namespace strata {
namespace core {

ExpertRowState ExpertRowCache::lookup(const ExpertRowKey& key, std::size_t bytes) {
    if (bytes > cfg_.budget_bytes) {
        ++refusals_;
        return ExpertRowState::cannot_fit;
    }
    if (map_.find(key) != map_.end()) {
        ++hits_;
        touch(key);
        return ExpertRowState::resident;
    }
    ++misses_;
    return ExpertRowState::needs_upload;
}

void ExpertRowCache::insert(const ExpertRowKey& key, const ExpertRowEntry& entry) {
    const auto found = map_.find(key);
    if (found != map_.end()) {
        // re-inserting the same key re-charges it: the caller is replacing the row, so the old bytes are released first
        used_ -= found->second.bytes;
        map_.erase(found);
        const auto p = pos_.find(key);
        if (p != pos_.end()) {
            lru_.erase(p->second);
            pos_.erase(p);
        }
    }
    if (entry.bytes > cfg_.budget_bytes) {          // never admit something that cannot fit; lookup() already refuses
        ++refusals_;
        return;
    }
    evict_until(entry.bytes);
    map_[key] = entry;
    lru_.push_front(key);
    pos_[key] = lru_.begin();
    used_ += entry.bytes;
}

void ExpertRowCache::touch(const ExpertRowKey& key) {
    const auto p = pos_.find(key);
    if (p == pos_.end()) return;
    lru_.splice(lru_.begin(), lru_, p->second);
    p->second = lru_.begin();
}

void ExpertRowCache::evict_until(std::size_t want) {
    while (used_ + want > cfg_.budget_bytes && !lru_.empty()) {
        const ExpertRowKey victim = lru_.back();
        lru_.pop_back();
        pos_.erase(victim);
        const auto it = map_.find(victim);
        if (it == map_.end()) continue;
        used_ -= it->second.bytes;
        ++evictions_;
        if (cfg_.on_evict != nullptr) cfg_.on_evict(cfg_.evict_ctx, victim, it->second);
        map_.erase(it);
    }
}

void ExpertRowCache::clear() {
    for (const auto& kv : map_) {
        if (cfg_.on_evict != nullptr) cfg_.on_evict(cfg_.evict_ctx, kv.first, kv.second);
    }
    map_.clear();
    lru_.clear();
    pos_.clear();
    used_ = 0;
}

std::string ExpertRowCache::report() const {
    char buf[256];
    std::snprintf(buf, sizeof buf,
                  "expert rows: %zu resident, %.2f of %.2f GB used, hits %llu misses %llu evictions %llu refusals %llu",
                  map_.size(), (double) used_ / 1073741824.0, (double) cfg_.budget_bytes / 1073741824.0,
                  (unsigned long long) hits_, (unsigned long long) misses_, (unsigned long long) evictions_,
                  (unsigned long long) refusals_);
    return std::string(buf);
}

}  // namespace core
}  // namespace strata
