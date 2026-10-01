#pragma once
/// GLM-5.3 per-block weight binding.  See src/core/glm_bind.cpp for the three transforms that are silent
/// when wrong, and tools/glm5_binding_contract.txt for the generated, verified spec they implement.
#include <string>
#include <vector>

#include "strata/core/layout.hpp"
#include "strata/core/weights.hpp"

namespace strata::core {

struct GlmBoundBlock {
    struct Tensor { std::string name; const float* ptr = nullptr; int ne0 = 0; int ne1 = 0; };
    std::vector<Tensor> tensors;
    std::vector<float*> owned;
    std::vector<void*> host_stage;
    ~GlmBoundBlock();
    const float* find(const std::string& n) const;
};

bool bind_glm_block(const LayerView& v, int block, int d_inner, int d_conv, GlmBoundBlock& out,
                    void* stream, std::string& err);

}  // namespace strata::core
