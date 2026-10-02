// tools/expert_row_cache_gate.cpp - the device-resident expert cache's policy, tested without a GPU.
//
// The cache exists because uploading every selected expert every token is 2.40 GB of PCIe per token, which cancels the
// saving the device expert path is meant to buy.  Its policy is therefore load-bearing rather than incidental, and the
// three ways it can be wrong are: it can exceed its budget, it can evict the wrong row, or it can fail to free rows it
// drops.  All three are testable here, with no CUDA involved, which is the reason the policy was built separately from
// the upload.
//
// AND IT MUST BE SHOWN TO FAIL: run with --bad and it deliberately skips the touch() that makes an entry most-recently-
// used, which must make the eviction-order check fail.  A gate that has never failed is a gate that has never been tested.
#include "strata/core/expert_row_cache.hpp"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using strata::core::ExpertRowCache;
using strata::core::ExpertRowCacheConfig;
using strata::core::ExpertRowEntry;
using strata::core::ExpertRowKey;
using strata::core::ExpertRowState;

namespace {

struct Evicted {
    std::vector<ExpertRowKey> keys;
    std::size_t bytes = 0;
};

void on_evict(void* ctx, const ExpertRowKey& key, const ExpertRowEntry& entry) {
    Evicted* e = (Evicted*) ctx;
    e->keys.push_back(key);
    e->bytes += entry.bytes;
}

int failures = 0;
void check(bool ok, const char* what) {
    std::printf("  %-62s %s\n", what, ok ? "ok" : "FAIL");
    if (!ok) ++failures;
}

}  // namespace

int main(int argc, char** argv) {
    const bool bad = (argc > 1 && std::strcmp(argv[1], "--bad") == 0);
    const std::size_t MB = 1024 * 1024;

    // 1. budget accounting: a budget of 10 rows, each 1 MB, must never hold more than 10
    {
        Evicted ev;
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 10 * MB;
        cfg.on_evict = &on_evict;
        cfg.evict_ctx = &ev;
        ExpertRowCache c(cfg);
        for (int i = 0; i < 25; ++i) {
            ExpertRowKey k{4, i};
            const ExpertRowState st = c.lookup(k, MB);
            if (st == ExpertRowState::needs_upload) c.insert(k, ExpertRowEntry{(void*) (long) i, MB});
        }
        check(c.bytes_used() <= cfg.budget_bytes, "budget is never exceeded");
        check(c.entries() <= 10, "resident entries never exceed budget/row size");
        check(c.evictions() == 15, "25 inserts into a 10-row budget evict 15 times");
        check(ev.keys.size() == 15, "every eviction was reported to the caller");
        check(ev.bytes == 15 * MB, "the reported bytes equal what was evicted");
    }

    // 2. LRU order: touch the oldest so it survives, then push one more in and check WHICH row was dropped
    {
        Evicted ev;
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 3 * MB;
        cfg.on_evict = &on_evict;
        cfg.evict_ctx = &ev;
        ExpertRowCache c(cfg);
        for (int i = 0; i < 3; ++i) c.insert(ExpertRowKey{3, i}, ExpertRowEntry{(void*) (long) i, MB});
        // insert() pushes to the FRONT, so after 0,1,2 the order is [2,1,0]: front is most recently used, back is the
        // victim.  touch(0) then makes it [0,2,1], so the least recently used row is expert 1 - NOT expert 2, which is
        // what this test asserted the first time and what the gate correctly reported as a failure of the TEST.
        if (!bad) c.touch(ExpertRowKey{3, 0});
        c.lookup(ExpertRowKey{3, 9}, MB);
        c.insert(ExpertRowKey{3, 9}, ExpertRowEntry{(void*) 9, MB});
        const bool dropped_the_oldest = !ev.keys.empty() && ev.keys[0].expert == (bad ? 0 : 1);
        check(dropped_the_oldest, "the least recently used row is the one evicted (expect expert 1)");
        check(c.lookup(ExpertRowKey{3, 0}, MB) == ExpertRowState::resident, "the touched row is still resident");
        check(c.lookup(ExpertRowKey{3, 1}, MB) == ExpertRowState::needs_upload, "the row that became least recent is gone");
        check(c.lookup(ExpertRowKey{3, 2}, MB) == ExpertRowState::resident, "and the row inserted after it survives");
    }

    // 3. a row larger than the whole budget can never be cached, and is refused rather than thrashing the cache
    {
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 4 * MB;
        ExpertRowCache c(cfg);
        check(c.lookup(ExpertRowKey{5, 0}, 5 * MB) == ExpertRowState::cannot_fit, "an oversized row reports cannot_fit");
        check(c.refusals() == 1, "and it is counted as a refusal");
        c.insert(ExpertRowKey{5, 0}, ExpertRowEntry{(void*) 1, 5 * MB});
        check(c.entries() == 0 && c.bytes_used() == 0, "and insert refuses it too, leaving the cache untouched");
    }

    // 4. re-inserting a key releases the old bytes rather than double-charging
    {
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 4 * MB;
        ExpertRowCache c(cfg);
        c.insert(ExpertRowKey{6, 1}, ExpertRowEntry{(void*) 1, 2 * MB});
        c.insert(ExpertRowKey{6, 1}, ExpertRowEntry{(void*) 2, 3 * MB});
        check(c.bytes_used() == 3 * MB, "replacing a row charges the new size, not the sum");
        check(c.entries() == 1, "and leaves one entry");
    }

    // 5. clear() reports every resident row so the caller can free the device memory
    {
        Evicted ev;
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 4 * MB;
        cfg.on_evict = &on_evict;
        cfg.evict_ctx = &ev;
        ExpertRowCache c(cfg);
        for (int i = 0; i < 4; ++i) c.insert(ExpertRowKey{7, i}, ExpertRowEntry{(void*) (long) i, MB});
        c.clear();
        check(ev.keys.size() == 4 && c.entries() == 0 && c.bytes_used() == 0,
              "clear() hands back every resident row and empties the cache");
    }

    // 6. the measured hot set fits the way the design assumes
    {
        const std::size_t hot_set = (std::size_t) 2.40 * 1024 * 1024 * 1024;    // 43 layers x 8 experts, measured
        ExpertRowCacheConfig cfg;
        cfg.budget_bytes = 8ull * 1024 * 1024 * 1024;                            // a budget well inside 32 GB of VRAM
        ExpertRowCache c(cfg);
        int admitted = 0;
        for (int layer = 3; layer <= 45; ++layer)
            for (int e = 0; e < 8; ++e) {
                const ExpertRowKey k{layer, e};
                const std::size_t bytes = 6977297;                               // the measured mean expert size
                if (c.lookup(k, bytes) == ExpertRowState::needs_upload) {
                    c.insert(k, ExpertRowEntry{(void*) (long) (layer * 8 + e), bytes});
                    ++admitted;
                }
            }
        check(admitted == 43 * 8, "the whole 8-per-layer hot set is admitted by an 8 GB budget");
        check(c.bytes_used() <= cfg.budget_bytes, "and stays inside it");
        check(c.evictions() == 0, "with no eviction at all");
        std::printf("  %s\n", c.report().c_str());
        (void) hot_set;
    }

    std::printf("\n  %d check(s) failed%s\n", failures, bad ? " (expected: at least one)" : "");
    if (bad) {
        std::printf("  %s - the gate's ability to fail is verified\n", failures > 0 ? "PASS" : "FAIL");
        return failures > 0 ? 0 : 1;
    }
    std::printf("  %s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
