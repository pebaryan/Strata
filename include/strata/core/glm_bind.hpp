#pragma once
/// GLM-5.3 per-block weight binding.  See src/core/glm_bind.cpp for the three transforms that are silent
/// when wrong, and tools/glm5_binding_contract.txt for the generated, verified spec they implement.
#include <string>
#include <vector>

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

namespace strata::core {

struct GlmBoundBlock {
    struct Tensor { std::string name; const float* ptr = nullptr; int ne0 = 0; int ne1 = 0;
                    bool quantized = false; };   ///< native GGUF blocks, not floats
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
