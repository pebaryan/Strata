#include <filesystem>
#include <cstdio>
#include "strata/core/native_dense.hpp"
#include <cstdlib>
#include "strata/core/weights.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>
#include <algorithm>
#include <climits>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <set>

namespace strata::core {
namespace {
/// GLM-5.3's natively-served matrices.  The two architectures name their projections differently, so the
/// qwen4exp patterns below match nothing in a glm5next artifact and NativeDense declared it served nothing -
/// which is how the binding driver came to be told "no supported GDN/QSA matrices in supplied shards".
/// A glm5next name is eligible when it is one of these and, as before, the quant type has a native MMVQ
/// kernel and the tensor is 2-D; the caller applies those two conditions.
static bool glm5_eligible_name(const std::string& name) {
    // DATA-DRIVEN, not a name list.  What NativeDense must serve is exactly "quantized tensors the pack did
    // not put in dense.bin", which is a property of the artifact, not a list someone maintains: an attempt
    // at a list grew 558 -> 560 -> 596 -> 632 and still met a name it had never heard of
    // (blk.3.indexer_compressor_gate.weight).  The caller still applies the two real conditions - an MMVQ
    // kernel exists for the type, and the shape is 2-D or (for glm5next) 3-D.
    //
    // The one exclusion is the expert tensors: those are owned by the expert cache reading the pack's
    // experts.bin, and their block geometry is not this class's business.
    if (name.find("_exps.") != std::string::npos) return false;
    return true;
}

/// True when the artifact is glm5next: then ONLY glm5_eligible_name decides what is served natively.
/// The qwen4exp patterns would otherwise claim GLM's expert tensors (ffn_*_exps.weight), whose geometry is
/// different and which belong to the expert cache reading the pack's experts.bin, and their block-geometry
/// check rejects them outright.
static bool g_glm5_only = false;
static int g_layer_lb = -1, g_layer_le = -1;   // set_layer_range; -1 means every layer
static bool in_range(const std::string& name) {
    if (g_layer_lb < 0 || name.rfind("blk.", 0) != 0) return true;
    const int l = std::atoi(name.c_str() + 4);
    return l >= g_layer_lb && l < g_layer_le;
}

bool eligible(const strata::TensorInfo& tensor, bool include_ple_key) {
    if (g_glm5_only) return glm5_eligible_name(tensor.name);
    const auto& name = tensor.name;
    if (name.rfind("blk.", 0) != 0) return false;
    // Match the native PLE kernel: Q2_0, IQ3_XXS, IQ4_XS and Q8_0 (UD-Q4_K_XL). Other keys retain the packed BF16
    // fallback.
    if (name == "blk.1.ple_key.weight")
        return include_ple_key && (tensor.type == 42 || tensor.type == 18 || tensor.type == 23 || tensor.type == 8);
    static const char* suffixes[] = {".attn_qkv.weight", ".attn_gate.weight", ".ssm_out.weight",
        ".attn_q.weight", ".attn_k.weight", ".attn_v.weight", ".attn_output.weight",
        ".ffn_gate_shexp.weight", ".ffn_up_shexp.weight", ".ffn_down_shexp.weight"};
    for (const char* suffix : suffixes) if (name.ends_with(suffix)) return true;
    return false;
}
struct DeviceFree { void operator()(void* p) const { if (p) cudaFree(p); } };
using DevicePtr = std::unique_ptr<void, DeviceFree>;
struct Pending {
    WeightRef* ref;
    int type;
    uint64_t bytes;
    DevicePtr data;
};
}

bool NativeDense::served_names(const std::vector<std::string>& shards, bool include_ple_key,
                               std::set<std::string>& out, std::string& err) {
    // served_names runs BEFORE load(), so it has to establish the naming regime itself - otherwise
    // g_glm5_only is still false here and glm5next's 3-D MLA tensors are rejected as if they were qwen4exp's
    // 2-D-only set, which is exactly how they came to be missing from the served set.
    try {
        strata::GgufFile first(shards.empty() ? std::string() : shards.front());
        const strata::MetaValue* a = first.get("general.architecture");
        g_glm5_only = (a && a->s == "glm5next");
    } catch (const std::exception&) { g_glm5_only = false; }
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& tensor : gguf.tensors()) {
                // glm5next's MLA tensors are legitimately 3-D; qwen4exp's native set is all 2-D.
                const bool shape_ok = tensor.shape.size() == 2 ||
                                      (g_glm5_only && tensor.shape.size() == 3);
                if (eligible(tensor, include_ple_key) && strata::kernels::native_mmvq_supported(tensor.type) &&
                    shape_ok)
                    out.insert(tensor.name);
            }
        }
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}

void NativeDense::set_layer_range(int lb, int le) { g_layer_lb = lb; g_layer_le = le; }
bool NativeDense::keep_unquantized_ple_key(const std::string& pack_dir, std::set<std::string>& skip,
                                           std::string& err) {
    const std::string key = "blk.1.ple_key.weight";
    if (!skip.count(key)) return true;
    int code_bits = -1;
    if (!WeightTable::index_code_bits(pack_dir, key, code_bits, err)) return false;
    if (code_bits == 0) skip.erase(key);
    return true;
}

NativeDense::~NativeDense() {
    if (scratch_) cudaFree(scratch_);
    for (void* p : weights_) cudaFree(p);
}

bool NativeDense::load(const std::vector<std::string>& shards, WeightTable& table, std::string& err,
                       bool include_ple_key, int64_t layer_lo, int64_t layer_hi) {
    auto outside = [&](const std::string& name) {
        if (layer_hi >= 0 && name.rfind("blk.", 0) == 0) {
            const long l = std::strtol(name.c_str() + 4, nullptr, 10);
            return l < layer_lo || l >= layer_hi;
        }
        return false;
    };
    // The artifact's OWN metadata decides which naming applies, before anything is served.
    try {
        strata::GgufFile first(shards.empty() ? std::string() : shards.front());
        const strata::MetaValue* a = first.get("general.architecture");
        g_glm5_only = (a && a->s == "glm5next");
    } catch (const std::exception&) { g_glm5_only = false; }
    if (scratch_ || !weights_.empty()) { err = "native dense: already loaded"; return false; }
    if (shards.empty()) { err = "native dense: at least one GGUF shard is required"; return false; }
    try {
        std::vector<Pending> pending;
        std::set<std::string> seen;
        int max_in = 0;
        uint64_t total = 0;
        uint64_t split_count = 0, split_tensors = 0;
        std::set<uint64_t> split_numbers;
        bool have_architecture = false;
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            const auto* count = gguf.get("split.count");
            const auto* number = gguf.get("split.no");
            const auto* tensors = gguf.get("split.tensors.count");
            if (gguf.get("general.architecture")) {
                // This branch builds the engine for GLM-5.3 as well as qwen4exp.  glm5next has its own,
                // stricter guard (check_glm5next_architecture, which validates the block table, the
                // per-layer MLA/KDA kinds, the expert counts and the indexer geometry), so route an
                // glm5next artifact to it instead of failing the generic qwen4exp field checks.  The
                // generic path is untouched for qwen4exp.
                const strata::MetaValue* arch = gguf.get("general.architecture");
                if (arch && arch->s == "glm5next") {
                    // This artifact's shard 1 is METADATA ONLY - it holds no tensors at all - so the
                    // per-block kind check has to see the siblings' tensor names, exactly as the phase-0
                    // preflight tool does.  Without this the guard reports "blk.0 has neither ssm_a nor
                    // attn_kv_a_mqa.weight" for every block, which is a property of the sharding, not of
                    // the model.
                    std::set<std::string> others;
                    {
                        const std::string fn = std::filesystem::path(path).filename().string();
                        const size_t of = fn.rfind("-of-");
                        if (of != std::string::npos && of >= 6) {
                            const std::string dir = std::filesystem::path(path).parent_path().string() + "/";
                            const std::string stem = fn.substr(0, of - 6);
                            const int n_shards = std::atoi(fn.substr(of + 4).c_str());
                            for (int i = 2; i <= n_shards; ++i) {
                                char sib[4096];
                                std::snprintf(sib, sizeof sib, "%s%s-%05d-of-%05d.gguf", dir.c_str(),
                                              stem.c_str(), i, n_shards);
                                try {
                                    strata::GgufFile s2(sib);
                                    for (const auto& t : s2.tensors()) others.insert(t.name);
                                } catch (const std::exception&) { /* a missing sibling is reported by the guard */ }
                            }
                        }
                    }
                    strata::Glm5NextGeometry geo;
                    err = strata::check_glm5next_architecture(gguf, geo, &others);
                } else {
                    err = strata::check_architecture(gguf);
                }
                if (!err.empty()) return false;
                have_architecture = true;
                if (count && number && tensors && number->u == 0 && count->u > 1) {
                    split_count = count->u;
                    split_tensors = tensors->u;
                }
            } else if (!have_architecture || !split_count || !count || !number || !tensors ||
                       count->u != split_count || number->u == 0 || number->u >= split_count ||
                       tensors->u != split_tensors) {
                err = "native dense: additional shard must match the architecture-validated first shard's split metadata";
                return false;
            }
            if (number && !split_numbers.insert(number->u).second) {
                err = "native dense: duplicate split shard number"; return false;
            }
            std::vector<uint64_t> offsets;
            for (const auto& tensor : gguf.tensors()) offsets.push_back(tensor.offset);
            std::sort(offsets.begin(), offsets.end());
            if (std::adjacent_find(offsets.begin(), offsets.end()) != offsets.end()) {
                err = "native dense: tensor payload offsets overlap"; return false;
            }
            // Validate every directory span, including tensors we do not upload:
            // an ignored tensor must not overlap the native matrix that follows it.
            const uint64_t payload = gguf.file_size() - gguf.data_start();
            for (const auto& tensor : gguf.tensors()) {
                int block_elements = 0, block_bytes = 0;
                uint64_t elements = 1;
                // Only tensors this class will actually SERVE need their quant blocks validated.  The
                // unfiltered loop tripped over glm5next's expert tensors, which NativeDense never serves
                // (the expert cache reads them from the pack's experts.bin).
                if (!eligible(tensor, include_ple_key)) continue;
                if (tensor.shape.empty() || !strata::block_geometry(tensor.type, block_elements, block_bytes) ||
                    tensor.shape[0] % (uint64_t) block_elements != 0) {
                    err = "native dense: invalid block geometry " + tensor.name; return false;
                }
                for (uint64_t dimension : tensor.shape) {
                    if (!dimension || elements > (std::numeric_limits<uint64_t>::max)() / dimension) {
                        err = "native dense: invalid tensor extent " + tensor.name; return false;
                    }
                    elements *= dimension;
                }
                const uint64_t blocks = elements / (uint64_t) block_elements;
                if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) block_bytes) {
                    err = "native dense: tensor byte count overflow " + tensor.name; return false;
                }
                const uint64_t bytes = blocks * (uint64_t) block_bytes;
                if (tensor.offset > payload || bytes > payload - tensor.offset) {
                    err = "native dense: truncated payload " + tensor.name; return false;
                }
                const auto next = std::upper_bound(offsets.begin(), offsets.end(), tensor.offset);
                if (next != offsets.end() && bytes > *next - tensor.offset) {
                    err = "native dense: overlapping payload " + tensor.name; return false;
                }
            }
            for (const auto& tensor : gguf.tensors()) {
                if (!eligible(tensor, include_ple_key) || outside(tensor.name)) continue;
                if (!in_range(tensor.name) && tensor.name.find("ple") == std::string::npos) continue;
                // 3-D tensors (glm5next's MLA k_b/v_b): mark only, do not upload.  The MMVQ
                // upload path is 2-D by construction - "incompatible matrix" is that assumption
                // firing - and the MLA kernel wants its own layout anyway, so glm_bind.cpp fetches
                // these from the GGUF itself.
                if (tensor.shape.size() != 2) continue;
                if (!seen.insert(tensor.name).second) {
                    err = "native dense: duplicate tensor " + tensor.name; return false;
                }
                auto found = table.table_.find(tensor.name);
                if (found == table.table_.end()) {
                    err = "native dense: tensor absent from canonical table: " + tensor.name; return false;
                }
                auto& ref = found->second;
                if (ref.native_data) { err = "native dense: override already attached"; return false; }
                if (!strata::kernels::native_mmvq_supported(tensor.type)) continue;
                // #326: the pack keeps an unquantized (--compat-bf16) key, which the PLE reads from the arena
                if (tensor.name == "blk.1.ple_key.weight" && !ref.quantized()) continue;
                if (!ref.quantized() || tensor.shape.size() != 2 ||
                    ref.ne0 <= 0 || ref.ne0 > INT_MAX || ref.ne1 <= 0 || ref.ne1 > INT_MAX ||
                    tensor.shape[0] != (uint64_t) ref.ne0 || tensor.shape[1] != (uint64_t) ref.ne1) {
                    err = "native dense: incompatible matrix " + tensor.name; return false;
                }
                const auto bytes = strata::kernels::native_mmvq_weight_bytes(
                    tensor.type, (int) ref.ne0, (int) ref.ne1);
                if (std::getenv("STRATA_NATIVE_TRACE") &&
                    tensor.name.find("blk.0.") != std::string::npos) {
                    // What was ALLOCATED versus what a Q8_0 reader with the plan row's element count would READ.
                    std::fprintf(stderr,
                                 "native upload %s: gguf shape=%llu,%llu  ref=%d,%d  alloc=%llu B  "
                                 "a q8_0 reader over %llu elems reads %llu B\n",
                                 tensor.name.c_str(), (unsigned long long) tensor.shape[0],
                                 (unsigned long long) tensor.shape[1], ref.ne0, ref.ne1,
                                 (unsigned long long) bytes,
                                 (unsigned long long) tensor.shape[0] * (unsigned long long) tensor.shape[1],
                                 (unsigned long long) tensor.shape[0] * (unsigned long long) tensor.shape[1] / 32 * 34);
                }
                void* allocation = nullptr;
                // MANAGED, NOT cudaMalloc, and this is the last of the port's device-vs-host defects.  A bound block
                // feeds CPU kernels and CUDA kernels from the SAME pointers - hc_pre, rms_norm, kda_forward and
                // mla_forward read what the GPU matmuls also read - and only managed memory is readable from both.
                // With a private device allocation a CPU kernel's first dereference is a SIGSEGV, which is what the
                // engine's 45-block run has been doing since it reached layer 3: seven of the eight MlaWeights
                // pointers came back 0x7fe9... while the one the runner had dequantized itself was 0x5623..., and
                // since the managed ARENA was demonstrably host-readable and these still faulted, they were provably
                // not in the arena.  The same reasoning the arena's own comment already records, one allocation down.
                auto status = cudaMallocManaged(&allocation, bytes, cudaMemAttachGlobal);
                DevicePtr data(allocation);
                if (status == cudaSuccess)
                    status = cudaMemcpy(data.get(), gguf.tensor_data(tensor), bytes, cudaMemcpyHostToDevice);
                if (status != cudaSuccess) {
                    err = "native dense upload " + tensor.name + ": " + cudaGetErrorString(status); return false;
                }
                max_in = (std::max)(max_in, (int) ref.ne0);
                total += bytes;
                pending.push_back(Pending{&ref, (int) tensor.type, bytes, std::move(data)});
            }
        }
        if (pending.empty()) { err = "native dense: no supported GDN/QSA matrices in supplied shards"; return false; }
        void* allocation = nullptr;
        // managed for the same reason: this scratch is written by the GPU quantizer and read by whatever consumes
        // the result, and the port has already been bitten once by assuming which side of the bus a consumer is on.
        const auto status = cudaMallocManaged(&allocation, strata::kernels::native_q8_1_bytes(max_in));
        DevicePtr scratch(allocation);
        if (status != cudaSuccess) { err = std::string("native dense scratch: ") + cudaGetErrorString(status); return false; }
        // All checks and allocations finish before publishing any reference.
        weights_.reserve(pending.size());
        for (auto& item : pending) {
            item.ref->native_data = item.data.get();
            item.ref->native_type = item.type;
            item.ref->native_q8_1 = scratch.get();
            weights_.push_back(item.data.release());
        }
        scratch_ = scratch.release();
        bytes_ = total;
        return true;
    } catch (const std::exception& error) {
        err = std::string("native dense: ") + error.what();
        return false;
    }
}
} // namespace strata::core
