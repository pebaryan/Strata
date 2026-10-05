// include/strata/artifact/gguf_reader.hpp - generated from src/artifact/gguf_reader.cpp by
// scripts/split_artifact.py.  Header-only on purpose: the reader is one translation unit's worth of
// code with no state to hide, and a header-only split cannot introduce a duplicate-symbol or
// missing-declaration bug in code that is already validated.
#pragma once
// src/artifact/gguf_reader.cpp - P1.S2: GGUF v3 reader, mmap, no ggml dependency.
//
// The C++ counterpart of tools/gguf_reader.py, which has been the reference since P0.S6 (written
// because gguf-py cannot represent type 42 / Q2_0). Per P1.S2 it must:
//   * mmap the file and parse header, metadata KV (ALL value types incl. arrays), tensor directory
//   * be multi-shard aware (split.* keys)
//   * carry an architecture guard: general.architecture == "qwen4exp" and the compiled-in constants
//     must match, refusing with a precise error otherwise
//   * have no ggml dependency
//
// This file is the reader plus a `--check` mode that validates it the way the Python one is
// validated: parse the tiny model AND both real shards, and report counts that other harnesses have
// already established independently (1,223 / 1 tensors; 202 Q2_0 in shard 1). Agreement with those
// numbers is the test.
//
// Build: scripts/build_artifact.bat        Run: gguf_reader.exe <file.gguf> [--check]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <stdexcept>
#include <algorithm>

#include "strata/artifact/gguf_split.hpp"

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX   // windows.h's min/max macros would break std::min/std::max in every file that includes this one
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace strata {

// ---- ggml type ids we care about. 42 = Q2_0, the PrismML ternary 2-bit encoding this engine targets.
inline const char* ggml_type_name(uint32_t t) {
    switch (t) {
    case 0:
        return "F32";
    case 1:
        return "F16";
    case 2:
        return "Q4_0";
    case 3:
        return "Q4_1";
    case 6:
        return "Q5_0";
    case 7:
        return "Q5_1";
    case 8:
        return "Q8_0";
    case 9:
        return "Q8_1";
    case 10:
        return "Q2_K";
    case 11:
        return "Q3_K";
    case 12:
        return "Q4_K";
    case 13:
        return "Q5_K";
    case 14:
        return "Q6_K";
    case 15:
        return "Q8_K";
    case 16:
        return "IQ2_XXS";
    case 17:
        return "IQ2_XS";
    case 18:
        return "IQ3_XXS";
    case 19:
        return "IQ1_S";
    case 20:
        return "IQ4_NL";
    case 21:
        return "IQ3_S";
    case 22:
        return "IQ2_S";
    case 23:
        return "IQ4_XS";
    case 24:
        return "I8";
    case 30:
        return "BF16";
    case 34:
        return "TQ1_0";
    case 35:
        return "TQ2_0";
    case 39:
        return "MXFP4";
    case 40:
        return "NVFP4";
    case 41:
        return "Q1_0";
    case 42:
        return "Q2_0";
    default:
        return "?";
    }
}

// Block geometry: (elements per block, bytes per block). Q2_0 is 64/18 - proven from this artifact's
// own offset brackets in P0.S6 and recorded in docs/q2_0-contract.md.
inline bool block_geometry(uint32_t t, int& elems, int& bytes) {
    switch (t) {
    case 0:
        elems = 1;
        bytes = 4;
        return true;
    case 1:
    case 30:
        elems = 1;
        bytes = 2;
        return true;
    case 2:
        elems = 32;
        bytes = 18;
        return true;
    case 3:
        elems = 32;
        bytes = 20;
        return true;
    case 6:
        elems = 32;
        bytes = 22;
        return true;
    case 7:
        elems = 32;
        bytes = 24;
        return true;
    case 8:
        elems = 32;
        bytes = 34;
        return true;
    case 9:
        elems = 32;
        bytes = 36;
        return true;
    case 10:
        elems = 256;
        bytes = 84;
        return true;
    case 11:
        elems = 256;
        bytes = 110;
        return true;
    case 12:
        elems = 256;
        bytes = 144;
        return true;
    case 13:
        elems = 256;
        bytes = 176;
        return true;
    case 14:
        elems = 256;
        bytes = 210;
        return true;
    case 16:
        elems = 256;
        bytes = 66;
        return true;
    case 17:
        elems = 256;
        bytes = 74;
        return true;
    case 18:
        elems = 256;
        bytes = 98;
        return true;
    case 20:
        elems = 32;
        bytes = 18;
        return true;
    case 21:   // IQ3_S
        elems = 256;
        bytes = 110;
        return true;
    case 22:   // IQ2_S
        elems = 256;
        bytes = 82;
        return true;
    case 23:
        elems = 256;
        bytes = 136;
        return true;
    case 29:   // IQ1_M
        elems = 256;
        bytes = 56;
        return true;
    case 24:   // I8: raw bytes (the FP8 PLE table of tools/ple_fp8_pack.py)
        elems = 1;
        bytes = 1;
        return true;
    case 42:
        elems = 64;
        bytes = 18;
        return true;
    default:
        return false;
    }
}

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> shape; // GGUF order: dim 0 varies fastest
    uint32_t type = 0;
    uint64_t offset = 0;
    uint64_t elements() const {
        uint64_t n = 1;
        for (auto d : shape) n *= d;
        return n;
    }
    const char* type_name() const { return ggml_type_name(type); }
};

// A bounds-checked cursor over the mmapped header region. Every read is checked, so a truncated or
// corrupt file produces a precise error rather than a segfault.
class Cursor {
public:
    Cursor(const uint8_t* base, size_t size) : base_(base), size_(size) {}
    void need(size_t n) const {
        if (pos_ + n > size_) throw std::runtime_error("GGUF: unexpected end of file in header");
    }
    template <class T> T read() {
        need(sizeof(T));
        T v;
        std::memcpy(&v, base_ + pos_, sizeof(T));
        pos_ += sizeof(T);
        return v;
    }
    std::string str() {
        const uint64_t n = read<uint64_t>();
        need((size_t)n);
        std::string s(reinterpret_cast<const char*>(base_ + pos_), (size_t)n);
        pos_ += (size_t)n;
        return s;
    }
    size_t pos() const { return pos_; }

private:
    const uint8_t* base_;
    size_t size_;
    size_t pos_ = 0;
};

enum class MetaType : uint32_t { U8 = 0, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 };

struct MetaValue {
    MetaType type = MetaType::U32;
    uint64_t u = 0;                // integer payloads
    double f = 0;                  // float payloads
    std::string s;                 // string payloads
    MetaType elem = MetaType::U32; // arrays
    uint64_t count = 0;
    std::vector<MetaValue> items;
    bool is_num() const { return type != MetaType::STRING && type != MetaType::ARRAY; }
    double num() const { return (type == MetaType::F32 || type == MetaType::F64) ? f : (double)u; }
};

inline MetaValue read_value(Cursor& c, MetaType t, int depth = 0) {
    if (depth > 2) throw std::runtime_error("GGUF: array nesting too deep");
    MetaValue v;
    v.type = t;
    switch (t) {
    case MetaType::U8:
        v.u = c.read<uint8_t>();
        break;
    case MetaType::I8:
        v.u = (uint64_t)(int64_t)c.read<int8_t>();
        break;
    case MetaType::U16:
        v.u = c.read<uint16_t>();
        break;
    case MetaType::I16:
        v.u = (uint64_t)(int64_t)c.read<int16_t>();
        break;
    case MetaType::U32:
        v.u = c.read<uint32_t>();
        break;
    case MetaType::I32:
        v.u = (uint64_t)(int64_t)c.read<int32_t>();
        break;
    case MetaType::F32: {
        float x = c.read<float>();
        v.f = x;
        v.u = 0;
        break;
    }
    case MetaType::BOOL:
        v.u = c.read<uint8_t>() ? 1 : 0;
        break;
    case MetaType::STRING:
        v.s = c.str();
        break;
    case MetaType::U64:
        v.u = c.read<uint64_t>();
        break;
    case MetaType::I64:
        v.u = (uint64_t)c.read<int64_t>();
        break;
    case MetaType::F64:
        v.f = c.read<double>();
        break;
    case MetaType::ARRAY: {
        v.elem = (MetaType)c.read<uint32_t>();
        v.count = c.read<uint64_t>();
        if (v.count > (1u << 24)) throw std::runtime_error("GGUF: implausible array length");
        v.items.reserve((size_t)std::min<uint64_t>(v.count, 64));
        for (uint64_t i = 0; i < v.count; ++i) {
            MetaValue e = read_value(c, v.elem, depth + 1);
            if (i < 64) v.items.push_back(std::move(e)); // keep a sample; the count is what matters
        }
        break;
    }
    default:
        throw std::runtime_error("GGUF: unknown metadata value type");
    }
    return v;
}

class GgufFile {
public:
    explicit GgufFile(const std::string& path) : path_(path) {
        try { open(); }
        catch (...) { close(); throw; }
    }
    ~GgufFile() { close(); }
    GgufFile(const GgufFile&) = delete;
    GgufFile& operator=(const GgufFile&) = delete;

    const std::vector<TensorInfo>& tensors() const { return tensors_; }
    const std::map<std::string, MetaValue>& metadata() const { return meta_; }
    uint32_t version() const { return version_; }
    uint64_t data_start() const { return data_start_; }
    uint64_t file_size() const { return size_; }
    uint64_t alignment() const { return alignment_; }
    const std::string& path() const { return path_; }

    const TensorInfo* find(const std::string& name) const {
        for (const auto& t : tensors_)
            if (t.name == name) return &t;
        return nullptr;
    }
    uint64_t count_type(const char* tn) const {
        uint64_t n = 0;
        for (const auto& t : tensors_)
            if (std::strcmp(t.type_name(), tn) == 0) ++n;
        return n;
    }
    const MetaValue* get(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? nullptr : &it->second;
    }
    const uint8_t* tensor_data(const TensorInfo& t) const { return base_ + data_start_ + t.offset; }

private:
    void open() {
#ifdef _WIN32
        HANDLE h = CreateFileA(path_.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                               FILE_ATTRIBUTE_NORMAL, nullptr);
        if (h == INVALID_HANDLE_VALUE) throw std::runtime_error("cannot open " + path_);
        LARGE_INTEGER li{};
        GetFileSizeEx(h, &li);
        size_ = (uint64_t)li.QuadPart;
        HANDLE m = CreateFileMappingA(h, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (!m) {
            CloseHandle(h);
            throw std::runtime_error("CreateFileMapping failed");
        }
        base_ = (const uint8_t*)MapViewOfFile(m, FILE_MAP_READ, 0, 0, 0);
        if (!base_) {
            CloseHandle(m);
            CloseHandle(h);
            throw std::runtime_error("MapViewOfFile failed");
        }
        map_ = m;
        file_ = h;
#else
        int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open " + path_);
        fd_ = fd;
        struct stat st{};
        if (fstat(fd, &st) != 0) throw std::runtime_error("fstat failed");
        size_ = (uint64_t)st.st_size;
        void* p = mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, fd, 0);
        if (p == MAP_FAILED) throw std::runtime_error("mmap failed");
        base_ = (const uint8_t*)p;
#endif
        parse();
    }
    void close() {
#ifdef _WIN32
        if (base_) UnmapViewOfFile(base_);
        if (map_) CloseHandle((HANDLE)map_);
        if (file_) CloseHandle((HANDLE)file_);
#else
        if (base_) munmap((void*)base_, size_);
        if (fd_ >= 0) ::close(fd_);
#endif
    }

    void parse() {
        Cursor c(base_, size_);
        const uint32_t magic = c.read<uint32_t>();
        if (magic != 0x46554747u) throw std::runtime_error("not a GGUF file (bad magic)");
        version_ = c.read<uint32_t>();
        if (version_ != 3)
            throw std::runtime_error("GGUF v" + std::to_string(version_) + ", this reader handles v3");
        const uint64_t n_tensors = c.read<uint64_t>();
        const uint64_t n_kv = c.read<uint64_t>();

        for (uint64_t i = 0; i < n_kv; ++i) {
            std::string key = c.str();
            MetaType t = (MetaType)c.read<uint32_t>();
            meta_.emplace(std::move(key), read_value(c, t));
        }
        tensors_.reserve((size_t)n_tensors);
        // GGUF has no index to arbitrate between two tensors of one name: find() is first-match, so a
        // duplicate would silently win by position.  Refuse the file at open instead, naming both.
        std::set<std::string> names;
        for (uint64_t i = 0; i < n_tensors; ++i) {
            TensorInfo t;
            t.name = c.str();
            if (!names.insert(t.name).second)
                throw std::runtime_error("GGUF: duplicate tensor name '" + t.name + "' in " + path_);
            const uint32_t nd = c.read<uint32_t>();
            if (nd == 0 || nd > 4) throw std::runtime_error("GGUF: bad n_dims for " + t.name);
            t.shape.resize(nd);
            for (uint32_t d = 0; d < nd; ++d) t.shape[d] = c.read<uint64_t>();
            t.type = c.read<uint32_t>();
            t.offset = c.read<uint64_t>();
            tensors_.push_back(std::move(t));
        }
        uint64_t align = 32;
        if (const MetaValue* a = get("general.alignment"))
            if (a->u) align = a->u;
        alignment_ = align;
        data_start_ = (c.pos() + align - 1) / align * align;
        // A metadata-only shard ends at its tensor table, so aligning past it lands just beyond EOF.
        // GLM-5.3-Flash's shard 1 is exactly that: 72 metadata keys, zero tensors, 9.4 MB.  Nothing can
        // be read out of a file that holds no tensors, so clamp instead of refusing - while keeping the
        // strict check when tensors DO exist, since then a past-EOF data start is real corruption.
        if (data_start_ > size_) {
            if (!tensors_.empty()) throw std::runtime_error("GGUF: data section starts past EOF");
            data_start_ = size_;
        }
    }

    std::string path_;
    const uint8_t* base_ = nullptr;
    uint64_t size_ = 0, data_start_ = 0, alignment_ = 32;
    uint32_t version_ = 0;
#ifdef _WIN32
    void* file_ = nullptr;
    void* map_ = nullptr;
#else
    int fd_ = -1;
#endif
    std::vector<TensorInfo> tensors_;
    std::map<std::string, MetaValue> meta_;
};

// Bytes of a tensor's payload from its shape and block geometry; 0 when the type is unknown, a row is not whole
// blocks, or the count overflows.
inline uint64_t tensor_payload_bytes(const TensorInfo& t) {
    int be = 0, bb = 0;
    if (t.shape.empty() || !block_geometry(t.type, be, bb) || t.shape[0] % (uint64_t) be) return 0;
    uint64_t elements = 1;
    for (uint64_t d : t.shape) {
        if (d == 0 || elements > (std::numeric_limits<uint64_t>::max)() / d) return 0;
        elements *= d;
    }
    const uint64_t blocks = elements / (uint64_t) be;
    if (blocks > (std::numeric_limits<uint64_t>::max)() / (uint64_t) bb) return 0;
    return blocks * (uint64_t) bb;
}

// The shards of one model (strata::gguf_split_paths), opened together; tensors are looked up across all of them.
// From eddoursul/Strata 8029fa9, with the split-key validation of #255 (gopinath87607) made a property of the
// model rather than of one loader: a split GGUF carries the model's metadata (general.architecture and the
// qwen4exp keys) in its FIRST shard only, and the later shards just declare split.count / split.no /
// split.tensors.count.  Unsloth's UD-Q4_K_XL is the extreme case: shard 1 holds the metadata and no tensor at
// all, and output.weight, token_embd.weight and the PLE table live in shard 2.  So the architecture is checked
// on `meta()` and a tensor is read from the shard that holds it.
//
// Refused at construction: a later shard whose split keys disagree with shard 1's (another model's shard, or a
// shard renamed into the family), a split model whose tensor directories do not add up to split.tensors.count,
// and a tensor name present in two shards (GGUF has no index to say which one is meant).
class GgufModel {
public:
    explicit GgufModel(const std::vector<std::string>& paths) {
        if (paths.empty()) throw std::runtime_error("GGUF: a model needs at least one shard");
        for (const auto& p : paths) shards_.push_back(std::make_unique<GgufFile>(p));
        validate_split();
        for (size_t i = 0; i < shards_.size(); ++i)
            for (const auto& t : shards_[i]->tensors()) {
                const auto ins = index_.emplace(t.name, std::make_pair(i, &t));
                if (!ins.second)
                    throw std::runtime_error("GGUF: tensor " + t.name + " is in two shards (" +
                                             shards_[ins.first->second.first]->path() + " and " +
                                             shards_[i]->path() + ")");
            }
    }
    /// Opens every shard of the model that `any_shard` belongs to (throws when one is missing).
    static GgufModel open(const std::string& any_shard) { return GgufModel(gguf_split_paths(any_shard)); }

    size_t size() const { return shards_.size(); }
    const GgufFile& shard(size_t i) const { return *shards_[i]; }
    /// The metadata shard: general.architecture and the model's keys.
    const GgufFile& meta() const { return *shards_[0]; }
    /// The tensor named `name` (its shard index in `*shard`), or nullptr.
    const TensorInfo* find(const std::string& name, size_t* shard = nullptr) const {
        const auto it = index_.find(name);
        if (it == index_.end()) return nullptr;
        if (shard) *shard = it->second.first;
        return it->second.second;
    }
    /// Whether `t` (a tensor of shard `s`) has a known byte count that lies inside its file.
    bool in_bounds(const TensorInfo& t, size_t s) const {
        const GgufFile& g = *shards_[s];
        const uint64_t bytes = tensor_payload_bytes(t);
        const uint64_t payload = g.file_size() - g.data_start();
        return bytes != 0 && t.offset <= payload && bytes <= payload - t.offset;
    }

private:
    void validate_split() const {
        const size_t n = shards_.size();
        const MetaValue* count0 = shards_[0]->get("split.count");
        if (n == 1) {
            if (count0 && count0->u > 1)
                throw std::runtime_error("GGUF: " + shards_[0]->path() + " is shard 1 of " + std::to_string(count0->u) +
                                         ", but it was opened as a whole model");
            return;
        }
        if (!shards_[0]->get("general.architecture"))
            throw std::runtime_error("GGUF: " + shards_[0]->path() + " has no general.architecture; the first shard "
                                     "of a split model carries the metadata");
        const MetaValue* total = shards_[0]->get("split.tensors.count");
        uint64_t tensors = 0;
        for (size_t i = 0; i < n; ++i) {
            const GgufFile& g = *shards_[i];
            const MetaValue* count = g.get("split.count");
            const MetaValue* no = g.get("split.no");
            const MetaValue* tc = g.get("split.tensors.count");
            if (!count || !no || count->u != n || no->u != i || (total && (!tc || tc->u != total->u)))
                throw std::runtime_error("GGUF: " + g.path() + " does not declare itself shard " + std::to_string(i + 1) +
                                         " of " + std::to_string(n) + " of this model (split.count / split.no / "
                                         "split.tensors.count)");
            tensors += g.tensors().size();
        }
        if (total && tensors != total->u)
            throw std::runtime_error("GGUF: the " + std::to_string(n) + " shards hold " + std::to_string(tensors) +
                                     " tensors, but split.tensors.count is " + std::to_string(total->u));
    }

    std::vector<std::unique_ptr<GgufFile>> shards_;
    std::map<std::string, std::pair<size_t, const TensorInfo*>> index_;
};

// ---- architecture guard (P1.S2). The engine is specialised to ONE model; anything else must be
// refused with a precise error rather than silently mis-run.
// experts = 0 means "accept the file's own count" (LOCAL PRUNED-MODEL SUPPORT, peb, 2026-09-28): the
// pruned GSQ-RCO releases carry 256 of the base model's 512 experts, and the engine reads the real
// count from the file anyway.  Every other field here still defines the architecture and is asserted.
struct Qwen4ExpGuard {
    uint32_t block_count = 48, hidden = 2560, experts = 0, experts_used = 0, head_count = 24,
             head_count_kv = 2;   // 0 = presence-only: pruned variants (GSQ-RCO Coder) legitimately ship
                                  // fewer experts than the canonical 512; the graph reads the true value
};

inline std::string check_architecture(const GgufFile& g, const Qwen4ExpGuard& want = {}) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "qwen4exp") return "architecture is '" + arch->s + "', this engine requires 'qwen4exp'";
    struct Req {
        const char* key;
        uint64_t want;
    };
    const Req reqs[] = {
        {"qwen4exp.block_count", want.block_count},
        {"qwen4exp.embedding_length", want.hidden},
        {"qwen4exp.expert_count", want.experts},
        {"qwen4exp.expert_used_count", want.experts_used},
        {"qwen4exp.attention.head_count", want.head_count},
        {"qwen4exp.attention.head_count_kv", want.head_count_kv},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (r.want && v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    return {}; // empty == ok
}

// ---- GLM-5.3-Flash (branch glm5next-port).  The qwen4exp guard above asserts one scalar geometry,
// which cannot express this model: two attention kinds inside one file, a dense MLP stem, an MTP
// block, and a per-layer KV head count that GGUF stores as an ARRAY.  So the block table is derived
// from the tensor names and then CROSS-CHECKED against the metadata - a guard that trusts only the
// header cannot catch a file whose tensors disagree with it.
enum class AttnKind : uint8_t { None, Linear, Mla };
enum class MlpKind : uint8_t { Dense, Moe };

struct Glm5Block {
    AttnKind attn = AttnKind::None;
    MlpKind mlp = MlpKind::Dense;
    bool mtp = false, indexer = false;
    uint32_t kv_heads = 0;     // from the metadata array: 0 means this block keeps no per-token KV
};

struct Glm5NextGeometry {
    uint32_t block_count = 46, hidden = 4096, head_count = 64, experts = 288, experts_used = 8,
             shared_experts = 1, expert_ffn = 2048, leading_dense = 3, nextn = 1,
             q_lora_rank = 1536, kv_lora_rank = 512, key_length = 512, value_length = 512,
             key_length_mla = 256, value_length_mla = 256, kda_head_dim = 128,
             hc_count = 4, hc_sinkhorn = 20,
             indexer_heads = 32, indexer_key_length = 128, indexer_kpool = 4, indexer_top_k = 2048;
    uint32_t first_moe = 0, mtp_block = 0;   // the expert layout starts at first_moe, not at 0
    std::vector<Glm5Block> blocks;
};

inline std::string check_glm5next_architecture(const GgufFile& g, Glm5NextGeometry& out,
                                               const std::set<std::string>* other_shards = nullptr) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "glm5next") return "architecture is '" + arch->s + "', expected 'glm5next'";

    struct Req { const char* key; uint64_t want; };
    const Req reqs[] = {
        {"glm5next.block_count", 46},
        {"glm5next.embedding_length", 4096},
        {"glm5next.attention.head_count", 64},
        {"glm5next.attention.key_length", 512},
        {"glm5next.attention.value_length", 512},
        {"glm5next.attention.q_lora_rank", 1536},
        {"glm5next.attention.kv_lora_rank", 512},
        {"glm5next.attention.key_length_mla", 256},
        {"glm5next.attention.value_length_mla", 256},
        {"glm5next.kda.head_dim", 128},
        {"glm5next.expert_count", 288},
        {"glm5next.expert_used_count", 8},
        {"glm5next.expert_shared_count", 1},
        {"glm5next.expert_feed_forward_length", 2048},
        {"glm5next.leading_dense_block_count", 3},
        {"glm5next.nextn_predict_layers", 1},
        {"glm5next.hyper_connection.count", 4},
        {"glm5next.hyper_connection.sinkhorn_iterations", 20},
        {"glm5next.attention.indexer.head_count", 32},
        {"glm5next.attention.indexer.key_length", 128},
        {"glm5next.attention.indexer.kpool", 4},
        {"glm5next.attention.indexer.top_k", 2048},
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (v->type == MetaType::ARRAY) return std::string(r.key) + " is an array, expected a scalar";
        if (v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    out.block_count = 46;
    out.hidden = 4096;   out.head_count = 64;
    out.experts = 288;   out.experts_used = 8;   out.shared_experts = 1;   out.expert_ffn = 2048;
    out.leading_dense = 3;   out.nextn = 1;
    out.q_lora_rank = 1536;  out.kv_lora_rank = 512;  out.key_length = 512;  out.value_length = 512;
    out.key_length_mla = 256;  out.value_length_mla = 256;  out.kda_head_dim = 128;
    out.hc_count = 4;  out.hc_sinkhorn = 20;
    out.indexer_heads = 32;  out.indexer_key_length = 128;  out.indexer_kpool = 4;  out.indexer_top_k = 2048;

    // The per-layer KV head count is an array here, and it is the one field that says which blocks
    // attend over real KV.  Its length must equal the block count or the two disagree about the model.
    const MetaValue* kv = g.get("glm5next.attention.head_count_kv");
    if (!kv) return "missing glm5next.attention.head_count_kv";
    if (kv->type != MetaType::ARRAY) return "glm5next.attention.head_count_kv is not an array";
    if (kv->count != out.block_count)
        return "head_count_kv has " + std::to_string(kv->count) + " entries for " +
               std::to_string(out.block_count) + " blocks";
    for (const auto& item : kv->items)
        if (!item.is_num()) return "head_count_kv holds a non-numeric entry";

    // derive the block table from the tensors, then cross-check it against that array.  A split model
    // keeps most tensors in the other shards, so the caller passes their names in and presence here
    // means "declared by the model", not "in this file".
    out.blocks.assign(out.block_count, Glm5Block{});
    auto has = [&](uint32_t b, const char* suffix) {
        char name[128];
        std::snprintf(name, sizeof name, "blk.%u.%s", b, suffix);
        return g.find(name) != nullptr || (other_shards && other_shards->count(name) != 0);
    };
    out.first_moe = out.block_count;   // ~0: "no MoE block yet"
    uint32_t mtp_seen = 0, mla_seen = 0, moe_seen = 0;
    for (uint32_t b = 0; b < out.block_count; ++b) {
        Glm5Block& blk = out.blocks[b];
        const bool ssm = has(b, "ssm_a");
        const bool mla = has(b, "attn_kv_a_mqa.weight");
        blk.mtp = has(b, "nextn.eh_proj.weight");
        blk.indexer = has(b, "indexer.attn_k.weight");
        blk.mlp = has(b, "ffn_gate_exps.weight") ? MlpKind::Moe : MlpKind::Dense;
        blk.kv_heads = (uint32_t)kv->items[b].u;
        if (ssm && mla) return "blk." + std::to_string(b) + " has both the linear and the MLA tensors";
        if (ssm) blk.attn = AttnKind::Linear;
        else if (mla) blk.attn = AttnKind::Mla;
        else return "blk." + std::to_string(b) + " has neither ssm_a nor attn_kv_a_mqa.weight";
        if (blk.attn == AttnKind::Mla) ++mla_seen;
        if (blk.mlp == MlpKind::Moe) {
            ++moe_seen;
            if (b < out.first_moe) out.first_moe = b;
        }
        if (blk.mtp) { ++mtp_seen; out.mtp_block = b; }
        // the cross-check: the array must agree with the tensors about which blocks keep KV
        if ((blk.attn == AttnKind::Mla) != (blk.kv_heads > 0))
            return "blk." + std::to_string(b) + ": the metadata says " + std::to_string(blk.kv_heads) +
                   " KV head(s) but the tensors say " +
                   (blk.attn == AttnKind::Mla ? "MLA" : "linear attention");
    }
    if (mla_seen != (uint32_t)std::count_if(kv->items.begin(), kv->items.end(),
                                            [](const MetaValue& v) { return v.u > 0; }))
        return "the KV-head array and the MLA tensors disagree on how many blocks carry KV";
    if (mtp_seen != out.nextn)
        return std::to_string(mtp_seen) + " block(s) carry nextn tensors, the metadata says " +
               std::to_string(out.nextn);
    if (out.mtp_block != out.block_count - 1)
        return "the MTP block is " + std::to_string(out.mtp_block) + ", not the last block";
    if (out.first_moe != out.leading_dense)
        return "the first MoE block is " + std::to_string(out.first_moe) + " but the metadata's leading "
               "dense block count is " + std::to_string(out.leading_dense);
    for (uint32_t b = out.first_moe; b < out.block_count; ++b)
        if (out.blocks[b].mlp != MlpKind::Moe)
            return "blk." + std::to_string(b) + " has no experts although every block from the stem on "
                   "must be a Mixture-of-Experts block";
    return {};   // empty == ok
}

/// Kolibri 1's guard (branch kolibri-port): architecture, geometry and tensor agreement for
/// Aleph Alpha's Kolibri-1 as converted by kolibri1-llama.cpp.patch.  The metadata keys are
/// arch-prefixed (kolibri1.*) exactly as llama.cpp writes them.  Shape mirrors
/// check_glm5next_architecture: a Kolibri1Geometry out-param, required scalars, then the two
/// arrays cross-checked against the tensors.
struct Kolibri1Geometry {
    uint32_t block_count = 50;
    uint64_t hidden = 2560, head_count = 48, head_count_kv = 4, head_dim = 128;
    uint64_t experts = 384, experts_used = 6, shared_experts = 1, expert_ffn = 512, shared_ffn = 512;
    uint64_t context_length = 262144;
    float rms_eps = 1e-6f, rope_freq_base = 10000.0f;
    uint32_t swa_window = 513;
    std::vector<uint8_t> swa_pattern;    // per layer: 1 = sliding (RoPE + window), 0 = full (NoPE)
    // the tensor-derived facts the trunk needs beside the geometry
    bool every_block_moe = true;
    uint32_t first_moe = 0;
};

inline std::string check_kolibri1_architecture(const GgufFile& g, Kolibri1Geometry& out,
                                               const std::set<std::string>* other_shards = nullptr) {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return "missing general.architecture";
    if (arch->s != "kolibri1") return "architecture is '" + arch->s + "', expected 'kolibri1'";

    // required scalars; the trailing bool says the value must match exactly (geometry is compile-
    // time in the kernels: 2560 % 256 == 0 and 512 % 256 == 0 are what native_fmt's block checks need)
    struct Req { const char* key; uint64_t want; };
    const Req reqs[] = {
        {"kolibri1.block_count", out.block_count},
        {"kolibri1.embedding_length", out.hidden},
        {"kolibri1.attention.head_count", out.head_count},
        {"kolibri1.attention.head_count_kv", out.head_count_kv},
        {"kolibri1.attention.key_length", out.head_dim},
        {"kolibri1.expert_count", out.experts},
        {"kolibri1.expert_used_count", out.experts_used},
        {"kolibri1.expert_shared_count", out.shared_experts},
        {"kolibri1.expert_feed_forward_length", out.expert_ffn},
        {"kolibri1.expert_shared_feed_forward_length", out.shared_ffn},
        {"kolibri1.expert_gating_func", 5},   // SIGMOID_LOGIT_ADD; the loader asserts this too
    };
    for (const auto& r : reqs) {
        const MetaValue* v = g.get(r.key);
        if (!v) return std::string("missing ") + r.key;
        if (v->type == MetaType::ARRAY) return std::string(r.key) + " is an array, expected a scalar";
        if (v->u != r.want)
            return std::string(r.key) + " = " + std::to_string(v->u) + ", expected " + std::to_string(r.want);
    }
    if (const MetaValue* v = g.get("kolibri1.expert_weights_norm"); v && v->u != 0)
        return "kolibri1.expert_weights_norm = 1; this port pins the artifact's no-renormalisation "
               "router (weights_norm is carried in the pack manifest, not flipped in the engine)";
    out.context_length = g.get("kolibri1.context_length") ? g.get("kolibri1.context_length")->u
                                                          : out.context_length;
    if (const MetaValue* v = g.get("kolibri1.attention.layer_norm_rms_epsilon")) out.rms_eps = v->f;
    if (const MetaValue* v = g.get("kolibri1.rope.freq_base")) out.rope_freq_base = v->f;

    // the SWA window and the per-layer pattern: both required (the patch's loader refuses
    // sliding_window == 0 too), the pattern's length must be the block count
    const MetaValue* win = g.get("kolibri1.attention.sliding_window");
    if (!win) return "missing kolibri1.attention.sliding_window";
    if (win->u == 0) return "kolibri1.attention.sliding_window must be > 0";
    out.swa_window = (uint32_t) win->u;
    const MetaValue* pat = g.get("kolibri1.attention.sliding_window_pattern");
    if (!pat) return "missing kolibri1.attention.sliding_window_pattern";
    if (pat->type != MetaType::ARRAY) return "kolibri1.attention.sliding_window_pattern is not an array";
    if (pat->count != out.block_count)
        return "sliding_window_pattern has " + std::to_string(pat->count) + " entries for " +
               std::to_string(out.block_count) + " blocks";
    out.swa_pattern.assign((size_t) out.block_count, 0);
    for (uint32_t b = 0; b < out.block_count; ++b)
        out.swa_pattern[(size_t) b] = pat->items[b].u != 0;

    // tensor agreement: every block carries the MoE quartet, the sandwich norms, QK-norm, the
    // router bias, and the shared expert.  A split model's other shards count as present.
    auto has = [&](uint32_t b, const char* suffix) {
        char name[128];
        std::snprintf(name, sizeof name, "blk.%u.%s", b, suffix);
        return g.find(name) != nullptr || (other_shards && other_shards->count(name) != 0);
    };
    for (uint32_t b = 0; b < out.block_count; ++b) {
        const char* need[] = {"ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight",
                              "ffn_gate_inp.weight", "exp_probs_b.bias",
                              "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                              "attn_norm.weight", "post_attention_norm.weight",
                              "ffn_norm.weight", "post_ffw_norm.weight",
                              "attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight",
                              "attn_q_norm.weight", "attn_k_norm.weight"};
        for (const char* suffix : need)
            if (!has(b, suffix))
                return "blk." + std::to_string(b) + " is missing " + suffix;
    }
    out.first_moe = 0;
    out.every_block_moe = true;
    return {};   // empty == ok
}

} // namespace strata
