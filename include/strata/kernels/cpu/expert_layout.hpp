// include/strata/kernels/cpu/expert_layout.hpp - plan v0.3 P6: where each routed expert lives in experts.bin.
//
// A Q2_0 pack (tools/strata_pack.py) has one blob size for every layer, `BLOB`, in the Strata expert form.  A
// native pack (tools/iq_pack.py, the IQ2_XS / IQ3_XXS files) keeps each expert's raw GGUF slices, so the blob
// size and the formats change from layer to layer; `native_experts.txt` says how.  Everything that touches an
// expert blob - the arena, the VRAM tier, the prompt path, the CPU pool, the GPU window - asks this table.
#pragma once

#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/native_expert.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace strata::kernels::cpu {

struct ExpertLayout {
    int version = 0;
    bool native = false;
    int64_t n_layers = 0, n_expert = NE;
    /// The lowest layer that HAS experts.  A model whose first blocks are a dense MLP stem (GLM-5.3-Flash:
    /// blocks 0-2) starts its native_experts.txt here, so layers [0, first_layer) hold no expert bytes and
    /// no consumer may ask this table for them.  The v3 header cannot express the stem, so the file's own
    /// lowest layer is taken as the authority.
    int64_t first_layer = 0;
    std::vector<NativeFmt> fmt;           ///< per layer (native packs)
    std::vector<uint64_t> offset, bytes;  ///< per layer: where its 512 blobs start, bytes per blob
    /// Plan v0.3 P6: per layer, the absolute offsets of the gate / up / down tensors in the model's shard 1, so
    /// the arena can be filled from the GGUF itself when the pack has no experts.bin (3 x n_layers, 0 = unknown).
    std::vector<uint64_t> gguf_off;
    /// Per layer, the GGUF file (a name beside the --native shard) that holds its experts when the model's
    /// shards split the layers (Swift's GGUFs: layers 13-47 in shard 2).  Empty = the --native shard itself.
    std::vector<std::string> gguf_file;
    uint64_t max_blob = BLOB;
    uint64_t total = 0;                   ///< experts.bin size

    uint64_t blob_bytes(int64_t layer) const { return native ? bytes[(size_t) layer] : (uint64_t) BLOB; }
    uint64_t layer_offset(int64_t layer) const {
        return native ? offset[(size_t) layer] : (uint64_t) layer * (uint64_t) n_expert * (uint64_t) BLOB;
    }
    uint64_t blob_offset(int64_t layer, int64_t expert) const {
        return layer_offset(layer) + (uint64_t) expert * blob_bytes(layer);
    }
};

/// Plan v0.3 P6: whether this CPU (and its OS) runs the AVX-512 kernels (F, BW, VL, VNNI, VBMI).  Probed in a
/// file compiled without AVX-512, so asking is safe everywhere; STRATA_FORCE_AVX2=1 answers no (for tests).
bool cpu_avx512_ok();
/// AVX-512BW / AVX-512VL / AVX-512DQ support used by the i-quant AVX-512 kernels.
bool cpu_avx512bw_ok();
/// AVX2 + FMA + F16C support.
bool cpu_avx2_ok();
/// AVX support for the AVX1 router path and older CPU builds.
bool cpu_avx1_ok();
/// SSE4.2 + POPCNT support for older CPU builds.
bool cpu_sse42_ok();
/// STRATA_FORCE_ISA cap: 3=none, 2=AVX2, 1=AVX, 0=SSE4.2.
int cpu_isa_cap();
/// The experimental CPU build's ISA floor, or empty for the normal build.
const char* isa_floor_build();
/// STRATA_IQ256_GATHER, the AVX-2 i-quant kernels' gathered grid decode (iq_avx2.cpp): -1 unset (auto, see
/// cpu_gather_fast_here), 0 the scalar decode on every core, 1 the gathered one on every core.
int iq256_gather_setting();
/// Whether this CPU's performance cores gather the IQ grid entries faster than they assemble them from scalar loads:
/// an Intel CPU with AVX-VNNI, i.e. Alder Lake / Sapphire Rapids or newer, and not one of the E-core-only parts.
/// The older Intel cores gather slowly (Haswell, Broadwell) or under the Downfall (GDS) microcode (Skylake to Tiger
/// Lake), AMD Zen 2/3 gather slowly, and Zen 4/5 run the AVX-512 kernels.  STRATA_FORCE_ISA answers no.
bool cpu_gather_fast();
/// cpu_gather_fast() and the CALLING THREAD runs on a performance core: CPUID 1Ah core type 40h on a hybrid CPU (its
/// E-cores, 20h, gather slower than they assemble).  Probed once per thread: the pool pins each worker to one core.
bool cpu_gather_fast_here();
/// Whether the AVX-2 expert kernels take their AVX-VNNI forms (vpdpwssd / vpdpbusd: Alder Lake, Sapphire Rapids and
/// later, P- and E-cores alike; the same integer sums): cpu_avx2_ok() and CPUID 7.1:EAX[4], not capped by
/// STRATA_FORCE_ISA, and not an older-CPU build (STRATA_ISA_FLOOR, which stays on its floor and AVX2).
/// STRATA_NO_AVXVNNI=1 answers no.
bool cpu_avxvnni_ok();
/// Whether a native pack's layer of this ggml type runs on Strata's own Q2_0 kernels (AVX2 / AVX-512).  On a
/// CPU without AVX2 (a native, non-portable build: ggml-cpu compiled for this CPU's SSE) a Q2_0 layer takes
/// ggml-cpu's own Q2_0 vec_dot like every other native type: slower, but it runs.
inline bool q2_native_kernels(int type) { return type == 42 && cpu_avx2_ok(); }
/// CPU brand string, or "unknown".
std::string cpu_name();

/// Q2_0 GGUF rows / activation quantizer on the kernels this CPU has.
void q2_rows_any(const uint8_t* w, size_t row_bytes, int nblocks, const ActQ* const* a, int nt, float* const* out,
                 int r0, int r1);
void act_quant_any(const float* x, int n, ActQ& a);

/// The process-wide layout (canonical Q2_0 until `expert_layout_load` finds a native pack).
const ExpertLayout& expert_layout();
/// Reads `<pack_dir>/native_experts.txt` when it exists (a native pack), else sets the canonical layout.
/// `hidden` and `expert_ffn` are the model's expert geometry; 0 means the compiled-in qwen4exp defaults
/// (2560 / 640).  GLM-5.3-Flash needs 4096 / 2048, and with the wrong pair the block-size check refuses
/// the layer instead of decoding it wrongly (native_fmt), so this is a refusal to get right, not a guess.
inline constexpr int kExpertLayoutVersion = 4;
bool expert_layout_load(const std::string& pack_dir, int64_t n_layers, int64_t n_expert, std::string& err,
                        int64_t hidden = 0, int64_t expert_ffn = 0);

}  // namespace strata::kernels::cpu
