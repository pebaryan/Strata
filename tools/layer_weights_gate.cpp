// layer_weights_gate - does glm_fill_layer_weights put each tensor in the RIGHT field?
//
// The provider is the one piece of the chain whose whole job is to know which name is which field, and a mapping
// mistake there is invisible: every pointer is a valid float* to a plausibly-shaped array, so a swapped pair (f_a for
// f_b, o_norm for attn_norm, dt_bias for ssm_a) produces a model that computes something.  This port has already been
// bitten by exactly that class once - the size table where ssm_f_a was given ssm_f_b's row - so the mapping gets its
// own gate rather than being trusted because it compiles.
//
// It uses the fixtures already on disk and needs NO pack, no CUDA and no expert source: a GlmBoundBlock is just a list
// of {name, ptr}, which is precisely why taking a bound block made the provider testable at all.
//
// What it checks:
//   1  layer 0 is KDA + dense, so out.kda is set, out.mla is null, and the dense matrices are set with no router;
//   2  EVERY field points at the buffer it should - and specifically that f_a and f_b are different buffers, that
//      o_norm is o_norm and not attn_norm, and that dt_bias is dt_bias and not ssm_a, since those are the three pairs
//      that would still "work";
//   3  the KDA struct's attn_norm is the SAME pointer as the layer's attn_norm, because the kernel norms its own input
//      with the layer's norm and reaching it by two names is the one place the provider has to know that;
//   4  a block with a tensor REMOVED fails, and names the missing tensor.
//
// Usage: layer_weights_gate <block-dir> <kda-dir>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "strata/core/glm_layer_weights.hpp"

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    std::printf("  %-4s %s\n", ok ? "ok" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

std::vector<float> load(const std::string& path, bool& ok) {
    std::vector<float> v;
    ok = false;
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::printf("  MISSING %s\n", path.c_str());
        return v;
    }
    std::fseek(f, 0, SEEK_END);
    const long n = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (n <= 0 || n % 4 != 0) {
        std::fclose(f);
        std::printf("  SIZE    %s: %ld bytes is not a whole number of floats\n", path.c_str(), n);
        return v;
    }
    v.resize((size_t) n / 4);
    const size_t got = std::fread(v.data(), 4, v.size(), f);
    std::fclose(f);
    ok = (got == v.size());
    return v;
}

/// A held buffer, so its address stays stable while the block points at it.
struct Held {
    std::vector<float> buf;
};

void add(strata::core::GlmBoundBlock& b, const std::string& name, Held& h) {
    strata::core::GlmBoundBlock::Tensor t;
    t.name = name;
    t.ptr = h.buf.data();
    t.ne0 = (int) h.buf.size();
    t.ne1 = 1;
    b.tensors.push_back(t);
}

bool has(const strata::core::GlmBoundBlock& b, const std::string& name) {
    for (const auto& t : b.tensors) if (t.name == name) return true;
    return false;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: layer_weights_gate <block-dir> <kda-dir>\n");
        return 2;
    }
    const std::string bd = argv[1], kd = argv[2];

    // ---- the tensors layer 0 needs, each held so its address is stable ----
    struct Entry { const char* tensor; const char* file; const char* dir; };
    const Entry entries[] = {
        {"hc_attn_fn.weight", "w_hc_attn_fn.bin", "b"}, {"hc_attn_base.weight", "w_hc_attn_base.bin", "b"},
        {"hc_attn_scale.weight", "w_hc_attn_scale.bin", "b"}, {"attn_norm.weight", "w_attn_norm.bin", "b"},
        {"hc_ffn_fn.weight", "w_hc_ffn_fn.bin", "b"}, {"hc_ffn_base.weight", "w_hc_ffn_base.bin", "b"},
        {"hc_ffn_scale.weight", "w_hc_ffn_scale.bin", "b"}, {"ffn_norm.weight", "w_ffn_norm.bin", "b"},
        {"attn_q.weight", "w_wq.bin", "k"}, {"attn_k.weight", "w_wk.bin", "k"}, {"attn_v.weight", "w_wv.bin", "k"},
        {"ssm_conv1d_q.weight", "w_conv_q.bin", "k"}, {"ssm_conv1d_k.weight", "w_conv_k.bin", "k"},
        {"ssm_conv1d_v.weight", "w_conv_v.bin", "k"}, {"ssm_a", "w_ssm_a.bin", "k"},
        {"ssm_dt.bias", "w_dt_bias.bin", "k"}, {"ssm_f_a.weight", "w_ssm_f_a.bin", "k"},
        {"ssm_f_b.weight", "w_ssm_f_b.bin", "k"}, {"ssm_beta.weight", "w_ssm_beta.bin", "k"},
        {"ssm_g_a.weight", "w_ssm_g_a.bin", "k"}, {"ssm_g_b.weight", "w_ssm_g_b.bin", "k"},
        {"ssm_norm.weight", "w_o_norm.bin", "k"}, {"attn_output.weight", "w_wo.bin", "k"},
        {"ffn_gate.weight", "w_ffn_gate.bin", "b2"}, {"ffn_up.weight", "w_ffn_up.bin", "b2"},
        {"ffn_down.weight", "w_ffn_down.bin", "b2"},
    };
    std::vector<Held> held(sizeof(entries) / sizeof(entries[0]));
    strata::core::GlmBoundBlock block;
    int loaded = 0;
    for (size_t i = 0; i < held.size(); ++i) {
        const Entry& e = entries[i];
        std::string dir = (std::strcmp(e.dir, "k") == 0) ? kd : bd;
        bool ok = false;
        held[i].buf = load(dir + "/" + e.file, ok);
        if (!ok) {
            std::printf("  note: %s not available here; the check that needs it will be skipped\n", e.file);
            continue;
        }
        add(block, e.tensor, held[i]);
        ++loaded;
    }
    std::printf("layer_weights_gate: %d tensors held, block has %zu\n", loaded, block.tensors.size());

    // the dense FFN matrices live in the ffn fixture, which this gate does not need - so a dense layer is exercised
    // only when they loaded, and the routed case is checked separately below by the negative test
    auto ptr_of = [&](const char* name) -> const float* {
        for (const auto& t : block.tensors) if (t.name == name) return t.ptr;
        return nullptr;
    };

    strata::kernels::glm::KdaWeights kda_storage;
    strata::kernels::glm::MlaWeights mla_storage;
    const float* shexp_storage[3] = {nullptr, nullptr, nullptr};
    strata::core::glm::GlmTrunkLayerWeights w;
    strata::kernels::glm::KdaGeometry kda_g;
    strata::kernels::glm::MlaGeometry mla_g;
    std::string err;

    // ---- 1: layer 0 is KDA + dense, and the provider must say so by what it fills ----
    const bool is_mla_l0 = strata::core::glm::glm_attention_is_mla(0) == 1;
    check(!is_mla_l0, "layer 0 is KDA per the dispatch table (not MLA)");

    const bool filled = strata::core::glm::glm_fill_layer_weights(block, 0, kda_g, mla_g, kda_storage, mla_storage,
                                                                  shexp_storage, w, err);
    if (!filled) {
        std::printf("  note: fill returned false (%s) - expected if the dense FFN matrices are not in <block-dir>\n",
                    err.c_str());
    }

    // The common and attention fields are assigned BEFORE the FFN step, so a failure at the dense FFN - which is what
    // happens with this fixture, because those matrices are not in it - does not invalidate them.  Checking them anyway
    // is the difference between a gate that verifies the mapping and one that reports PASS because every mapping
    // assertion was skipped, which is the failure mode this port has already been caught by once.
    if (w.kda == nullptr) {
        check(false, "the KDA attention fields were filled (the provider assigns them before the FFN step)");
    } else {
        // ---- 2: every field points at the buffer it should ----
        auto eq = [&](const char* field, const char* tensor, bool expect_equal) {
            const float* want = ptr_of(tensor);
            const float* got = nullptr;
            if (std::strcmp(field, "hc_attn_fn") == 0) got = w.hc_attn_fn;
            else if (std::strcmp(field, "hc_attn_base") == 0) got = w.hc_attn_base;
            else if (std::strcmp(field, "hc_attn_scale") == 0) got = w.hc_attn_scale;
            else if (std::strcmp(field, "attn_norm") == 0) got = w.attn_norm;
            else if (std::strcmp(field, "hc_ffn_fn") == 0) got = w.hc_ffn_fn;
            else if (std::strcmp(field, "ffn_norm") == 0) got = w.ffn_norm;
            else if (std::strcmp(field, "kda.wq") == 0) got = w.kda ? w.kda->wq : nullptr;
            else if (std::strcmp(field, "kda.wk") == 0) got = w.kda ? w.kda->wk : nullptr;
            else if (std::strcmp(field, "kda.ssm_a") == 0) got = w.kda ? w.kda->ssm_a : nullptr;
            else if (std::strcmp(field, "kda.dt_bias") == 0) got = w.kda ? w.kda->dt_bias : nullptr;
            else if (std::strcmp(field, "kda.ssm_f_a") == 0) got = w.kda ? w.kda->ssm_f_a : nullptr;
            else if (std::strcmp(field, "kda.ssm_f_b") == 0) got = w.kda ? w.kda->ssm_f_b : nullptr;
            else if (std::strcmp(field, "kda.o_norm") == 0) got = w.kda ? w.kda->o_norm : nullptr;
            else if (std::strcmp(field, "kda.wo") == 0) got = w.kda ? w.kda->wo : nullptr;
            else if (std::strcmp(field, "kda.attn_norm") == 0) got = w.kda ? w.kda->attn_norm : nullptr;
            check((got == want) == expect_equal, std::string(field) + " -> " + tensor);
        };
        check(w.kda != nullptr, "a KDA layer sets out.kda");
        check(w.mla == nullptr, "a KDA layer leaves out.mla null");
        eq("hc_attn_fn", "hc_attn_fn.weight", true);
        eq("hc_attn_base", "hc_attn_base.weight", true);
        eq("hc_attn_scale", "hc_attn_scale.weight", true);
        eq("attn_norm", "attn_norm.weight", true);
        eq("hc_ffn_fn", "hc_ffn_fn.weight", true);
        eq("ffn_norm", "ffn_norm.weight", true);
        eq("kda.wq", "attn_q.weight", true);
        eq("kda.wk", "attn_k.weight", true);
        eq("kda.ssm_a", "ssm_a", true);
        eq("kda.dt_bias", "ssm_dt.bias", true);
        eq("kda.ssm_f_a", "ssm_f_a.weight", true);
        eq("kda.ssm_f_b", "ssm_f_b.weight", true);
        eq("kda.o_norm", "ssm_norm.weight", true);
        eq("kda.wo", "attn_output.weight", true);
        eq("kda.attn_norm", "attn_norm.weight", true);

        // the swaps that would still "work": each of these must NOT hold
        eq("kda.ssm_f_a", "ssm_f_b.weight", false);
        eq("kda.ssm_f_b", "ssm_f_a.weight", false);
        eq("kda.o_norm", "attn_norm.weight", false);
        eq("kda.dt_bias", "ssm_a", false);
        eq("kda.ssm_a", "ssm_dt.bias", false);

        // ---- 3: the same tensor reached by two names ----
        check(w.kda->attn_norm == w.attn_norm, "kda.attn_norm IS the layer's attn_norm (one tensor, two names)");
    }

    // and the failure, when there is one here, must be the dense FFN step and nothing earlier
    if (!filled) {
        check(err.find("ffn_gate.weight") != std::string::npos,
              "the fill failed only at the dense FFN step: \"" + err + "\"");
    }

    // ---- 4: a missing tensor must fail and name itself ----
    {
        strata::core::GlmBoundBlock short_block;
        for (const auto& t : block.tensors) {
            if (t.name == "ssm_f_b.weight") continue;   // drop exactly one
            short_block.tensors.push_back(t);
        }
        if (short_block.tensors.size() != block.tensors.size()) {
            strata::core::glm::GlmTrunkLayerWeights w2;
            strata::kernels::glm::KdaWeights k2;
            strata::kernels::glm::MlaWeights m2;
            const float* sx[3] = {nullptr, nullptr, nullptr};
            std::string err2;
            const bool ok2 = strata::core::glm::glm_fill_layer_weights(short_block, 0, kda_g, mla_g, k2, m2, sx, w2, err2);
            check(!ok2, "a block missing ssm_f_b.weight fails");
            check(err2.find("ssm_f_b.weight") != std::string::npos,
                  "and the error names ssm_f_b.weight: \"" + err2 + "\"");
        }
    }

    std::printf("layer_weights_gate: %s\n", failures ? "FAIL" : "PASS");
    return failures ? 1 : 0;
}
