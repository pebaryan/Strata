/// GLM-5.3 weight binding for one block, in the engine.
///
/// Everything here is transcription of tools/glm5_binding_contract.txt, which is generated from the kernel
/// headers, the pack's index, and the ONE loader proven correct by reproducing llama.cpp's activations.
/// The three transforms that file documents are the whole reason it exists; all three are silent when wrong:
///
///   1. ssm_conv1d_{q,k,v}.weight is stored (d_inner, 1, d_conv) with d_conv FASTEST, so the
///      (d_conv, d_inner) that conv1d_silu indexes as conv_w[k*d_inner + ch] is reshape(d_inner, d_conv).T.
///      Getting this wrong took attn_output-0 from corr +1.00000 / 0.02% to +0.52073 / 90.56%.
///   2. the pack records every matrix as (in, out); the verified oracle works in the GGUF's (out, in).
///   3. the swiglu clamps and the KDA's contraction convention come from the artifact, never from a
///      neighbouring constant or a remembered convention.
#include <cstdlib>

#include <cuda_runtime.h>

#include <memory>

#include <cstdint>
#include <string>
#include <vector>

#include "strata/core/glm_bind.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/glm5_blocks_gen.hpp"
#include "strata/core/layout.hpp"
#include "strata/kernels/quantize_act.hpp"

namespace strata::core {

const float* GlmBoundBlock::find(const std::string& n) const {
    for (const Tensor& t : tensors) if (t.name == n) return t.ptr;
    return nullptr;
}

GlmBoundBlock::~GlmBoundBlock() {
    for (float* p : owned) if (p) cudaFree(p);
    for (void* p : host_stage) if (p) free(p);
}

/// Dequantize a quantized tensor to device, then transpose (in,out) -> (out,in).
/// The transpose is done on the host once per layer: this runs at bind time, not per token.
static bool dequant_transposed(const WeightRef& w, int ne0, int ne1, void* stream, float** out_dev,
                               void** out_host, std::string& err) {
    const int64_t n = (int64_t) ne0 * ne1;
    void* stage = malloc((size_t) n * sizeof(float));
    if (!stage) { err = "glm_bind: host staging allocation failed"; return false; }
    float* dev = nullptr;
    if (cudaMalloc(&dev, (size_t) n * sizeof(float)) != cudaSuccess) { free(stage); err = "glm_bind: cudaMalloc failed"; return false; }
    strata::kernels::dequant_q8_0((const uint8_t*) w.data, dev, n, stream);
    if (cudaMemcpy(stage, dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        free(stage); cudaFree(dev); err = "glm_bind: staging copy failed"; return false;
    }
    // ne0 is the FAST dim, so the pack's flat order is [ne1][ne0]; emit [ne0][ne1].
    float* src = (float*) stage;
    float* dst = (float*) malloc((size_t) n * sizeof(float));
    if (!dst) { free(stage); cudaFree(dev); err = "glm_bind: transpose allocation failed"; return false; }
    for (int64_t i1 = 0; i1 < ne1; ++i1)
        for (int64_t i0 = 0; i0 < ne0; ++i0) dst[i0 * ne1 + i1] = src[i1 * ne0 + i0];
    if (cudaMemcpy(dev, dst, (size_t) n * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
        free(stage); free(dst); cudaFree(dev); err = "glm_bind: transpose upload failed"; return false;
    }
    free(dst);
    *out_dev = dev;
    *out_host = stage;
    return true;
}

/// Repack ssm_conv1d_{q,k,v}.weight from the artifact's (d_inner, 1, d_conv) to (d_conv, d_inner).
static float* repack_conv(const float* src, int d_inner, int d_conv) {
    float* dst = nullptr;
    if (cudaMalloc(&dst, (size_t) d_conv * d_inner * sizeof(float)) != cudaSuccess) return nullptr;
    std::vector<float> tmp((size_t) d_conv * d_inner);
    for (int ch = 0; ch < d_inner; ++ch)
        for (int k = 0; k < d_conv; ++k) tmp[(size_t) k * d_inner + ch] = src[(size_t) ch * d_conv + k];
    cudaMemcpy(dst, tmp.data(), tmp.size() * sizeof(float), cudaMemcpyHostToDevice);
    return dst;
}

/// Bind one GLM block's weights.  `block` only selects the `blk.<n>.` suffix, so this works for every layer.
bool bind_glm_block(const LayerView& v, int block, int d_inner, int d_conv,
                    const std::vector<std::string>& gguf_shards, GlmBoundBlock& out, void* stream,
                    std::string& err) {
    // opened once per call and held for its duration, so a tensor can be found by name
    std::vector<std::unique_ptr<strata::GgufFile>> ggufs;
    for (const std::string& p : gguf_shards) {
        try { ggufs.push_back(std::make_unique<strata::GgufFile>(p)); }
        catch (const std::exception&) { /* a missing shard is reported per tensor */ }
    }
    const std::string prefix = "blk." + std::to_string(block) + ".";
    const int n_rows = glm_block_row_count(block);
    if (!n_rows) { err = "glm_bind: no tensor table for block " + std::to_string(block); return false; }
    const GlmTensorRow* rows = glm_block_rows(block);
    for (int i = 0; i < n_rows; ++i) {
        const GlmTensorRow& t = rows[i];
        const std::string full = t.name;
        const std::string suffix = full.substr(prefix.size());
        const WeightRef* w = v.get(suffix.c_str());
        if (!w) { err = v.name(suffix.c_str()) + " is missing"; return false; }

        GlmBoundBlock::Tensor got;
        got.name = suffix;
        got.ne0 = t.ne0;
        got.ne1 = t.ne1;

        if (t.ttype == 0) {
            const float* p = (const float*) w->data;
            // transform 1: the conv kernel wants (d_conv, d_inner)
            if (suffix.compare(0, 10, "ssm_conv1d") == 0 && t.ne0 == d_inner && t.ne1 == d_conv) {
                float* repacked = repack_conv(p, d_inner, d_conv);
                if (!repacked) { err = "glm_bind: conv repack failed for " + full; return false; }
                out.owned.push_back(repacked);
                got.ptr = repacked;
                got.ne0 = d_conv;
                got.ne1 = d_inner;
            } else {
                got.ptr = p;
            }
        } else if (std::getenv("STRATA_HC_DEQUANT") && w->native_type == 8 &&
                   ((full.size() >= 10 && full.compare(full.size() - 10, 10, "_fn.weight") == 0) ||
                    (std::getenv("STRATA_DEQUANT_ATTN_Q") &&
                     full.compare(0, full.find('.'), "blk") == 0 &&
                     full.find(".attn_q.weight") != std::string::npos))) {
            // The hc function matrices (hc_attn_fn / hc_ffn_fn) are consumed as FLOATS by hc_pre, which reads
            // them as the ggml weight [hc*n_embd, (2+hc)*hc] - see the note in glm_hc.hpp.  Native serving
            // hands over Q8_0 blocks in the swapped orientation the ATTENTION kernels want, which is both the
            // wrong type (hc_pre would read ~1.5 MB out of a ~418 KB buffer: the segfault) and the wrong
            // layout.  So take the artifact's own bytes and dequantize them once at bind time.
            if (!w->native_data) { err = "glm_bind: " + full + " is not natively served"; return false; }
            const int64_t n = (int64_t) t.ne0 * (t.ne1 > 0 ? t.ne1 : 1);
            float* dev_f = nullptr;
            if (cudaMalloc(&dev_f, (size_t) n * sizeof(float)) != cudaSuccess) {
                err = "glm_bind: cudaMalloc failed for " + full; return false;
            }
            // OPT-IN ONLY (STRATA_HC_DEQUANT).  EARLIER NOTE, NOW RETRACTED: this branch was recorded here as
            // one that "CORRUPTS THE ARENA", because with the flag on the F32 rows next to it read back
            // cudaMemoryTypeUnregistered and any read faulted.  That was WRONG, and the real cause is known:
            // the driver ran this stage block AFTER its own cudaFree(arena), so every arena pointer was a freed
            // allocation - cudaMemoryTypeUnregistered is what that looks like.  The flag only appeared to be the
            // cause because it was the only configuration that REACHED the stage at all.  No corruption was ever
            // demonstrated, and the branch produces exactly what hc_pre needs (floats, unswapped).
            //
            // What is still unverified for this branch, and is the next thing to measure rather than assume:
            // n is taken from the PLAN ROW (t.ne0 * t.ne1), while the bytes being read are the NATIVE tensor's.
            // For the hc function matrices those agree.  For a swapped native attention matrix they need not, and
            // widening this branch to attn_q/k/v produced a poisoned context - cudaMalloc failing on a later
            // tensor with the GPU free - so the count, and what the native blocks actually are, is the thing to
            // check before widening it again.
            //
            // w->native_data is the artifact's own Q8_0 blocks, already resident on the DEVICE and in the
            // artifact's orientation (NativeDense copies them as they are).  dequant_q8_0 takes device blocks
            // - handing it a host pointer faults and poisons the context, which is how the second call here
            // came back as "cudaMalloc failed".
            // Clear any error the process set EARLIER (a failed upload, a failed allocation) so the report below
            // is about this dequant and not about something that happened before it - cudaGetLastError() is
            // sticky, which is exactly how a downstream failure gets misattributed to this step.
            (void) cudaGetLastError();
            strata::kernels::dequant_q8_0((const uint8_t*) w->native_data, dev_f, n, stream);
            if (stream) cudaStreamSynchronize((cudaStream_t) stream);
            // INSTRUMENTATION: name the step rather than theorise.  If a tensor's dequant is what poisons the
            // context, the error is already set here - and reporting it per tensor says WHICH tensor and at which
            // step, instead of leaving the next allocation to report it indirectly.
            {
                const cudaError_t err_after = cudaGetLastError();
                std::fprintf(stderr, "dequant: %-30s n=%lld want=%.1f MB  alloc+dequant: %s\n",
                             full.c_str(), (long long) n, (double) n * 4.0 / 1048576.0,
                             cudaGetErrorString(err_after));
                if (err_after != cudaSuccess) {
                    err = std::string("glm_bind: ") + full + ": " + cudaGetErrorString(err_after);
                    return false;
                }
            }
            // Discriminating probe: run the kernel, then discard its result.  If the arena still corrupts,
            // the kernel's write is the cause; if it does not, holding this second allocation is.
            if (std::getenv("STRATA_HC_DEQUANT_KERNEL_ONLY")) {
                cudaFree(dev_f);
                continue;
            }
            out.owned.push_back(dev_f);
            got.ptr = dev_f;
            got.quantized = false;
            got.ne0 = t.ne0;                 // the artifact's orientation, NOT swapped
            got.ne1 = t.ne1;
            out.tensors.push_back(got);
            continue;
        } else if (!w->native_data && w->bytes == 0) {
            // Marked natively served but not uploaded: NativeDense skips 3-D tensors (glm5next's MLA k_b/v_b,
            // whose blocks the MMVQ upload path cannot take), so their row exists and their data does not.
            // The binding has to fetch these from the GGUF itself, in the layout the MLA kernel wants - that
            // is the remaining half of 9.1 rather than something to paper over.
            // Option (c): fetch it from the artifact itself.  NativeDense will not upload a 3-D tensor
            // (the MMVQ path is 2-D by construction) and the pack does not hold it either, so the GGUF is
            // the only source.  Copied once at bind time, not per token.
            const strata::TensorInfo* found = nullptr;
            strata::GgufFile* owner = nullptr;
            for (auto& g : ggufs) {
                for (const auto& ti : g->tensors()) {
                    if (ti.name == full) { found = &ti; owner = g.get(); break; }
                }
                if (found) break;
            }
            if (!found) { err = "glm_bind: " + full + " is in no shard"; return false; }
            const size_t nbytes = (size_t) w->ne0 * (size_t) (w->ne1 > 0 ? w->ne1 : 1) * 4;
            float* dev3 = nullptr;
            if (cudaMalloc(&dev3, nbytes) != cudaSuccess) { err = "glm_bind: alloc failed for " + full; return false; }
            if (cudaMemcpy(dev3, owner->tensor_data(*found), nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(dev3); err = "glm_bind: upload failed for " + full; return false;
            }
            out.owned.push_back(dev3);
            got.ptr = dev3;
            got.quantized = true;
            out.tensors.push_back(got);
            continue;
        } else if (w->native_data) {
            // Served straight from the GGUF, which is the orientation the verified oracle works in, so this
            // must NOT be transposed.  Transposing it - which an earlier version of this file did - is trap 7
            // misapplied: trap 7 is about the PACK's dense.bin rows, not the GGUF's own tensors.
            got.ptr = (const float*) w->native_data;
            got.quantized = true;             // the GGUF's own blocks: NOT floats, do not read as floats
            got.ne0 = t.ne1;
            got.ne1 = t.ne0;
        } else {
            // transform 2: a pack-resident quantized row: dequantize, then (in,out) -> (out,in)
            float* dev = nullptr;
            void* host = nullptr;
            if (!dequant_transposed(*w, t.ne0, t.ne1, stream, &dev, &host, err)) return false;
            out.owned.push_back(dev);
            out.host_stage.push_back(host);
            got.ptr = dev;
            got.ne0 = t.ne1;      // after the transpose the fast dim is the old ne1
            got.ne1 = t.ne0;
        }
        out.tensors.push_back(got);
    }
    return true;
}

}  // namespace strata::core
