#pragma once
/// GLM-5.3 per-block weight binding.  See src/core/glm_bind.cpp for the three transforms that are silent
/// when wrong, and tools/glm5_binding_contract.txt for the generated, verified spec they implement.
#include <string>
#include <vector>

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

namespace strata::core {

struct GlmBoundBlock {
    /// `native_type` is the artifact's ggml type for a quantized tensor (0 for floats).  It is carried here because a
    /// caller that has to DEQUANTIZE one cannot decide how from the bound block alone - the type lived only on the
    /// WeightRef the binder was holding, so a per-layer streamer (which must not dequantize all forty-five layers at
    /// once: 20-30 GB of host floats, measured as an OOM kill) had no way to dispatch.  Recording it costs four bytes
    /// and removes the only reason that streamer would have had to re-query the weight table per layer.
    struct Tensor { std::string name; const float* ptr = nullptr; int ne0 = 0; int ne1 = 0;
                    bool quantized = false; int native_type = 0; };   ///< native GGUF blocks, not floats
    std::vector<Tensor> tensors;
    std::vector<float*> owned;
    std::vector<void*> host_stage;
    ~GlmBoundBlock();
    const float* find(const std::string& n) const;
};

/// `gguf_shards` is needed for the tensors NativeDense marks but cannot upload: glm5next's 3-D MLA tensors
/// (attn_k_b / attn_v_b and friends), whose MMVQ-path blocks are not the MLA kernel's layout.  The binding
/// fetches those straight from the artifact, which is the one place the GGUF is the source rather than the
/// pack.  Pass an empty vector and those tensors report that they are unavailable instead.
bool bind_glm_block(const LayerView& v, int block, int d_inner, int d_conv,
                    const std::vector<std::string>& gguf_shards, GlmBoundBlock& out, void* stream,
                    std::string& err);

}  // namespace strata::core
