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
#include <cstring>
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
    // The KDA implementation is host-side, while GlmBoundBlock owns CUDA allocations.  Managed memory satisfies
    // both requirements and can still be released by the block's cudaFree-based destructor.
    if (cudaMallocManaged(&dst, (size_t) d_conv * d_inner * sizeof(float), cudaMemAttachGlobal) != cudaSuccess)
        return nullptr;
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
            // dense.bin may hold an unquantized tensor as BF16.  The generated row's ttype says that no block
            // dequantizer is needed; it does not imply four bytes per element.  In particular every MoE router is
            // 4096*288*2 bytes.  Casting those pairs of BF16 values to float made the router consume every other
            // oracle value as one invented float and consequently select an unrelated set of experts.
            const size_t n = (size_t) t.ne0 * (size_t) (t.ne1 > 0 ? t.ne1 : 1);
            if (w->bytes == n * sizeof(uint16_t)) {
                float* expanded = nullptr;
                if (cudaMallocManaged(&expanded, n * sizeof(float), cudaMemAttachGlobal) != cudaSuccess) {
                    err = "glm_bind: BF16 expansion allocation failed for " + full;
                    return false;
                }
                const uint16_t* src = (const uint16_t*) w->data;
                for (size_t j = 0; j < n; ++j) {
                    const uint32_t bits = (uint32_t) src[j] << 16;
                    std::memcpy(&expanded[j], &bits, sizeof(float));
                }
                out.owned.push_back(expanded);
                p = expanded;
            }
            // transform 1: the conv kernel wants (d_conv, d_inner)
            // GGUF's ne0 is the fastest dimension.  These tensors are [d_conv, d_inner] in the generated table,
            // but their flat payload is channel-major (src[ch*d_conv+k]); the kernel expects tap-major
            // (dst[k*d_inner+ch]).  The old guard tested the dimensions backwards and silently skipped this repack.
            if (suffix.compare(0, 10, "ssm_conv1d") == 0 && t.ne0 == d_conv && t.ne1 == d_inner) {
                float* repacked = repack_conv(p, d_inner, d_conv);
                if (!repacked) { err = "glm_bind: conv repack failed for " + full; return false; }
                out.owned.push_back(repacked);
                got.ptr = repacked;
                got.ne0 = d_conv;
                got.ne1 = d_inner;
            } else {
                got.ptr = p;
            }
        } else if (std::getenv("STRATA_HC_DEQUANT") &&
                   (w->native_type == 8 || w->native_type == 13 || w->native_type == 14) &&
                   ((full.size() >= 10 && full.compare(full.size() - 10, 10, "_fn.weight") == 0) ||
                    (std::getenv("STRATA_DEQUANT_ALL_Q8_0")))) {
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
            // Dispatch by TYPE, and only for types a dequantizer implements.  Type 8 is Q8_0 (34-byte blocks,
            // 32 elements) and type 13 is Q5_K (176-byte blocks, 256 elements) - the type this artifact uses for
            // attn_q/k/v, attn_output, ffn_gate and ffn_up.  dequant_q5_K was gated BIT-EXACT against the oracle
            // over a 134 MB tensor before being wired in here.
            if (w->native_type == 8) {
                strata::kernels::dequant_q8_0((const uint8_t*) w->native_data, dev_f, n, stream);
            } else if (w->native_type == 13) {
                strata::kernels::dequant_q5_K((const uint8_t*) w->native_data, dev_f, n, stream);
            } else if (w->native_type == 14) {
                strata::kernels::dequant_q6_K((const uint8_t*) w->native_data, dev_f, n, stream);
            } else {
                err = std::string("glm_bind: no dequantizer for type ") + std::to_string(w->native_type) +
                      " (" + full + ")";
                return false;
            }
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
            // HOST COPY, and this is the whole defect.  These dequantized floats are read by CPU kernels - hc_pre,
            // rms_norm, kda_forward and mla_forward; the entire trunk is CPU-side, which is why its own parity gates
            // link no CUDA at all.  Leaving the result in a CUDA buffer means the first host dereference is a
            // SIGSEGV, and that is exactly what the engine's first 45-block run produced: hc_pre first (the hc stage
            // precedes the KDA), then kda_forward (the KDA precedes the MLA).  Every previous caller of
            // glm_trunk_forward was fed fixture arrays in ordinary host memory, so no amount of per-kernel testing
            // could have found it - only a caller that builds its weights from the real pack.
            //
            // Dequantize on the device (the kernels exist and are bit-exact), copy back, and point the tensor at the
            // host copy.  The host buffer is intentionally NOT freed: GlmBoundBlock::owned is released with cudaFree,
            // so putting a host pointer there would be worse than leaking it, and a bind happens once per process.
            {
                float* host_f = (float*) std::malloc((size_t) n * sizeof(float));
                if (host_f == nullptr) {
                    err = "glm_bind: " + full + ": host dequant buffer allocation failed";
                    return false;
                }
                const cudaError_t cp = cudaMemcpy(host_f, dev_f, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost);
                if (cp != cudaSuccess) {
                    std::free(host_f);
                    err = "glm_bind: " + full + ": host copy of the dequantized weights: " +
                          cudaGetErrorString(cp);
                    return false;
                }
                cudaFree(dev_f);
                // host_stage, NOT owned: `owned` is released with cudaFree, so a malloc'd host pointer there would be
                // freed by the wrong allocator the moment a GlmBoundBlock is destroyed.  The struct already carries a
                // host_stage vector for exactly this, which is what a binder expecting host-side weights would use.
                out.host_stage.push_back(host_f);
                got.ptr = host_f;
            }
            got.quantized = false;
            got.native_type = 0;
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
            // KEEP THE HOST POINTER.  tensor_data() hands back FLOATS out of the artifact's own mapping, and these
            // tensors are consumed by CPU kernels (mla_forward), so uploading them to a device buffer produced a copy
            // the CPU could not read and the original - already in host memory - was thrown away.  The device upload
            // is gone entirely.
            //
            // AND THEY ARE NOT QUANTIZED, which was the defect that surfaced as "dequant_to_host: no dequantizer for
            // type -1 (attn_k_b.weight)": this branch was marking a FLOAT buffer quantized, with the sentinel type -1
            // carried over from a WeightRef that has no native blocks, so a dequantizer was handed floats.  Recording
            // quantized = false says what the data IS, and that is what makes the staging pass leave it alone.
            // AND THEY ARE Q8_0 BLOCKS, not floats - which is what the arithmetic says and what I had backwards for one
            // commit.  These tensors are 64 x 512 x 256 = 8,388,608 elements; as floats that would be 33,554,432
            // bytes, but the artifact declares 8,912,896, and 8,388,608 / 32 * 34 = 8,912,896 exactly - Q8_0 is 34
            // bytes per 32 values.  The type is 8.  So tensor_data() hands back BLOCKS, the consumer is a CPU kernel,
            // and the staging must dequantize them (type 8 is one of the three dequantizers gated bit-exact).
            //
            // AND THE ELEMENT COUNT MUST COME FROM THE ARTIFACT'S OWN SHAPE.  A product of two dimensions cannot
            // express a three-dimensional tensor: the previous version sized its copy as ne0 * ne1 * 4 = 524,288
            // bytes, fourteen times short of 8,912,896, and nobody noticed because nothing ever read it.  elements()
            // multiplies every dimension the file declares.
            // AND THE BLOCKS HAVE TO BE UPLOADED, because the dequantizer is a CUDA kernel and CUDA kernels cannot
            // dereference host memory on this device.  Leaving the source in the artifact's mapping - which is what the
            // previous commit did - produced "the kernel failed for attn_k_b.weight: an illegal memory access was
            // encountered": the right blocks, the right type, and a pointer the kernel was never able to read.
            //
            // The byte count comes from ggml_row_size over the artifact's OWN type and element count, which is the
            // whole point of taking it from found rather than from the bound tensor: 8,388,608 elements at type 8 is
            // 8,912,896 bytes, and a two-dimensional ne0 * ne1 * 4 would have been 524,288 - fourteen times short.
            // The block sizes are the ones the dequantizers themselves are gated on - Q8_0 34 bytes per 32 values,
            // Q5_K 176 per 256, Q6_K 210 per 256 - and anything else is REFUSED rather than guessed, because the
            // previous four versions of this line each assumed a size and each was wrong.  ggml_row_size would be the
            // tidy answer and is not available to this translation unit, which is why the numbers are stated here with
            // the sentence that pins them.
            const size_t nelem = (size_t) found->elements();
            size_t nbytes = 0;
            switch (found->type) {
                case 8:  nbytes = nelem / 32 * 34;  break;     // Q8_0
                case 13: nbytes = nelem / 256 * 176; break;    // Q5_K
                case 14: nbytes = nelem / 256 * 210; break;    // Q6_K
                default:
                    err = "glm_bind: " + full + ": type " + std::to_string(found->type) +
                          " is not a block type this fetch can upload";
                    return false;
            }
            float* dev3 = nullptr;
            if (cudaMalloc(&dev3, nbytes) != cudaSuccess) {
                err = "glm_bind: alloc failed for " + full + " (" + std::to_string(nbytes) + " bytes)";
                return false;
            }
            if (cudaMemcpy(dev3, owner->tensor_data(*found), nbytes, cudaMemcpyHostToDevice) != cudaSuccess) {
                cudaFree(dev3);
                err = "glm_bind: upload failed for " + full;
                return false;
            }
            out.owned.push_back(dev3);
            got.ptr = (const float*) dev3;
            got.quantized = true;
            got.native_type = (int) found->type;           // 8 = Q8_0, read from the artifact rather than assumed
            got.ne0 = (int) found->elements();             // the FULL element count; ne1 stays 1
            got.ne1 = 1;
            out.tensors.push_back(got);
            continue;
        } else if (w->native_data) {
            // Served straight from the GGUF, which is the orientation the verified oracle works in, so this
            // must NOT be transposed.  Transposing it - which an earlier version of this file did - is trap 7
            // misapplied: trap 7 is about the PACK's dense.bin rows, not the GGUF's own tensors.
            got.ptr = (const float*) w->native_data;
            got.quantized = true;
            got.native_type = (int) w->native_type;             // the GGUF's own blocks: NOT floats, do not read as floats
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
