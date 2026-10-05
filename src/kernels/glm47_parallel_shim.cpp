// src/kernels/glm47_parallel_shim.cpp
//
// A serial glm_parallel_for for the GLM-4.7 parity gates.
//
// Production defines this in src/core/glm_trunk.cpp (a worker pool over independent index ranges), and the
// only reason glm_moe_native.cpp needs it is the batched native stage.  A parity gate must not drag in the
// whole GLM-5.3 trunk engine (glm_trunk.cpp -> glm_layer.cpp -> glm_bind.cpp -> ...) just to satisfy one
// symbol, and the parallel version is numerically IDENTICAL: it distributes independent jobs over threads
// with no cross-range reduction, so running them in order on one thread gives bit-for-bit the same result.
// A serial definition is therefore the honest stand-in for a CPU parity build - not a stub that skips work.
//
// If glm_trunk.cpp is ever linked alongside this file, the duplicate definition is a deliberate loud error.
#include <functional>

namespace strata::core::glm {

void glm_parallel_for(int n, const std::function<void(int)>& job) {
    for (int i = 0; i < n; ++i) job(i);
}

}  // namespace strata::core::glm
