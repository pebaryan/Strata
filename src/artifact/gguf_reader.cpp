// src/artifact/gguf_reader.cpp - the `strata-gguf` CLI.  The reader itself is header-only in
// include/strata/artifact/gguf_reader.hpp; this file is just the entry point, so it cannot drift
// from the library it exercises.
#include "strata/artifact/gguf_reader.hpp"

// ---- GLM-5.3-Flash: report the block table the phase-2 guard derived, and the cross-checks that
// make it trustworthy (branch glm5next-port).  Phase 2's gate is exactly this: the engine opens the
// artifact and reports its layer table instead of refusing it.
namespace {

std::string glm_block_kind(const strata::Glm5Block& b) {
    std::string s = (b.attn == strata::AttnKind::Linear) ? "linear" : "mla";
    s += (b.mlp == strata::MlpKind::Moe) ? "+moe" : "+dense";
    if (b.indexer) s += "+indexer";
    if (b.mtp) s += "+mtp";
    return s;
}

int report_glm5next(const std::string& path) {
    strata::GgufFile g(path);
    // A split model keeps most tensors in its other shards: union their names so the block table
    // describes the model rather than the shard that happens to hold the metadata.
    std::set<std::string> others;
    const size_t slash = path.find_last_of('/');
    const std::string dir = (slash == std::string::npos) ? std::string() : path.substr(0, slash + 1);
    const std::string name = (slash == std::string::npos) ? path : path.substr(slash + 1);
    const size_t of = name.rfind("-of-");
    if (of != std::string::npos && of >= 6) {
        const std::string total = name.substr(of + 4);          // "00003.gguf"
        const std::string stem = name.substr(0, of - 6);        // everything before "-00001-of-..."
        const int n_shards = std::atoi(total.c_str());
        for (int i = 2; i <= n_shards; ++i) {
            char sib[4096];
            std::snprintf(sib, sizeof sib, "%s%s-%05d-of-%s", dir.c_str(), stem.c_str(), i, total.c_str());
            try {
                strata::GgufFile s(sib);
                for (const auto& t : s.tensors()) others.insert(t.name);
            } catch (const std::exception& e) {
                std::printf("  shard %s: %s\n", sib, e.what());
            }
        }
    }
    strata::Glm5NextGeometry geo;
    const std::string err = strata::check_glm5next_architecture(g, geo, &others);
    if (!err.empty()) {
        std::printf("  glm5next guard: FAIL - %s\n", err.c_str());
        return 1;
    }
    uint32_t n_linear = 0, n_mla = 0, n_moe = 0, n_kv = 0;
    for (const auto& b : geo.blocks) {
        n_linear += (b.attn == strata::AttnKind::Linear);
        n_mla += (b.attn == strata::AttnKind::Mla);
        n_moe += (b.mlp == strata::MlpKind::Moe);
        n_kv += (b.kv_heads > 0);
    }
    std::printf("  glm5next guard: PASS (metadata and tensors agree)\n");
    std::printf("  blocks          %u (dense stem %u, MoE %u, MTP block %u)\n", geo.block_count,
                geo.leading_dense, n_moe, geo.mtp_block);
    std::printf("  attention       %u linear (kda head_dim %u), %u MLA (kv_lora_rank %u, "
                "key_length_mla %u), %u blocks carry KV\n", n_linear, geo.kda_head_dim, n_mla,
                geo.kv_lora_rank, geo.key_length_mla, n_kv);
    std::printf("  experts         %u routed (%u used) + %u shared, ffn %u\n", geo.experts,
                geo.experts_used, geo.shared_experts, geo.expert_ffn);
    std::printf("  hyper-conn      %u residual streams, %u sinkhorn iterations\n", geo.hc_count,
                geo.hc_sinkhorn);
    std::printf("  indexer         %u heads, key_length %u, kpool %u, top_k %u\n", geo.indexer_heads,
                geo.indexer_key_length, geo.indexer_kpool, geo.indexer_top_k);
    std::printf("  expert layout   first MoE block %u: blocks 0-%u carry no experts, so the pack's "
                "expert table starts there\n", geo.first_moe, geo.first_moe ? geo.first_moe - 1 : 0);
    std::printf("  block table (run-length):\n");
    for (uint32_t i = 0; i < geo.block_count;) {
        const std::string kind = glm_block_kind(geo.blocks[i]);
        uint32_t j = i;
        while (j + 1 < geo.block_count && glm_block_kind(geo.blocks[j + 1]) == kind) ++j;
        std::printf("    blk %2u-%-2u  %-22s  %2u block(s)  kv_heads %u\n", i, j, kind.c_str(), j - i + 1,
                    geo.blocks[i].kv_heads);
        i = j + 1;
    }
    return 0;
}

}  // namespace

#ifndef STRATA_GGUF_MAIN_DISABLED
int main(int argc, char** argv) {
    if (argc < 2) {
        std::printf("usage: gguf_reader <file.gguf> [--check | --glm5next]\n");
        return 2;
    }
    const std::string mode = (argc > 2) ? std::string(argv[2]) : std::string();
    const bool check = mode == "--check";
    try {
        if (mode == "--glm5next") return report_glm5next(argv[1]);
        strata::GgufFile g(argv[1]);
        std::printf("%s\n", argv[1]);
        std::printf("  version %u   tensors %zu   metadata %zu   data_start %llu   size %llu\n", g.version(),
                    g.tensors().size(), g.metadata().size(), (unsigned long long)g.data_start(),
                    (unsigned long long)g.file_size());

        if (check) {
            // Independent expectations, all established by earlier phases.
            size_t n_blk = 0;
            for (const auto& t : g.tensors())
                if (t.name.rfind("blk.", 0) == 0) ++n_blk;
            std::printf("  blk.* tensors %zu   globals %zu\n", n_blk, g.tensors().size() - n_blk);
            std::printf("  Q2_0 %llu   BF16 %llu   F32 %llu\n", (unsigned long long)g.count_type("Q2_0"),
                        (unsigned long long)g.count_type("BF16"), (unsigned long long)g.count_type("F32"));

            // Every tensor's byte size must land inside the gap to the next tensor's offset, since
            // GGUF aligns every tensor. That is the same bracket test tools/verify_q2_0_geometry.py
            // uses, and it validates block_geometry() against the file rather than asserting it.
            std::vector<const strata::TensorInfo*> ordered;
            for (const auto& t : g.tensors()) ordered.push_back(&t);
            std::sort(ordered.begin(), ordered.end(), [](auto a, auto b) { return a->offset < b->offset; });
            size_t bad = 0, unknown = 0;
            for (size_t i = 0; i < ordered.size(); ++i) {
                const auto* t = ordered[i];
                int el = 0, by = 0;
                if (!strata::block_geometry(t->type, el, by)) {
                    ++unknown;
                    continue;
                }
                if (t->elements() % (uint64_t)el) {
                    ++bad;
                    continue;
                }
                const uint64_t nb = t->elements() / (uint64_t)el * (uint64_t)by;
                // The bracket is on the GAP to the next tensor's offset, not on that offset itself -
                // offsets are relative to data_start, so the gap is a difference. Comparing against
                // the raw offset flagged almost every tensor (213/214) while the same test in Python
                // (tools/verify_q2_0_geometry.py) passed.
                const uint64_t end = (i + 1 < ordered.size()) ? ordered[i + 1]->offset - t->offset
                                                              : g.file_size() - g.data_start() - t->offset;
                if (!(nb <= end && end < nb + 32)) ++bad;
            }
            std::printf("  geometry bracket: %zu out of range, %zu types unknown\n", bad, unknown);
            const std::string err = strata::check_architecture(g);
            std::printf("  architecture guard: %s\n", err.empty() ? "PASS" : ("FAIL - " + err).c_str());
            if (bad) return 1;
        }
        return 0;
    } catch (const std::exception& e) {
        std::printf("ERROR: %s\n", e.what());
        return 1;
    }
}
#endif // STRATA_GGUF_MAIN_DISABLED
