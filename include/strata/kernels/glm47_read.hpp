/// Read a whole GGUF tensor into a flat float array, dequantizing block by block.
///
/// The layout is the GGUF's OWN: GGUF dim 0 varies fastest, so the flat array is exactly what the
/// references (tools/glm47_*_reference.py, which dequantize row by row through gguf-py) produce - and
/// exactly the order the MLA weights are consumed in.  Every tensor this port reads has dim 0 divisible
/// by its block size (Q4_K/Q6_K 256, Q5_0 32), so a linear block walk is exact - no per-row stride game.
///
/// This is the C++ side of what the gates had only ever seen as Python-written fixtures: the runner reads
/// the artifact directly.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "strata/artifact/dequant.hpp"      // dequantize_* (inline), block_geometry
#include "strata/artifact/gguf_reader.hpp"  // GgufFile, TensorInfo

namespace strata::kernels::glm {

/// Loads `name` from `g` into `out` (resized to the tensor's element count), dequantizing to float.
/// Returns false and sets `err` if the tensor is missing or its type is unsupported.
inline bool load_tensor_f32(const GgufFile& g, const std::string& name, std::vector<float>& out,
                            std::string& err) {
    const TensorInfo* t = g.find(name);
    if (t == nullptr) { err = "GGUF has no tensor " + name; return false; }
    int elems = 0, bytes = 0;
    if (!block_geometry(t->type, elems, bytes)) {
        err = name + ": unsupported GGML type " + std::to_string(t->type);
        return false;
    }
    const uint64_t n = t->elements();
    const uint8_t* p = g.tensor_data(*t);
    out.assign((size_t) n, 0.0f);
    if (elems == 1) {   // element types
        if (t->type == 0) { dequantize_f32(p, out.data(), (int) n); return true; }
        if (t->type == 1) { dequantize_f16(p, out.data(), (int) n); return true; }
        if (t->type == 30) { dequantize_bf16(p, out.data(), (int) n); return true; }
        err = name + ": unsupported scalar type " + std::to_string(t->type);
        return false;
    }
    if (n % (uint64_t) elems != 0) {
        err = name + ": " + std::to_string(n) + " elements is not a multiple of the block " +
              std::to_string(elems);
        return false;
    }
    for (uint64_t i = 0; i < n; i += (uint64_t) elems) {
        const uint8_t* blk = p + (i / (uint64_t) elems) * (uint64_t) bytes;
        float* o = out.data() + (size_t) i;
        switch (t->type) {
        case 6:  dequantize_q5_0(blk, o); break;
        case 12: dequantize_q4_K(blk, o); break;
        case 14: dequantize_q6_K(blk, o); break;
        default: err = name + ": unsupported block type " + std::to_string(t->type); return false;
        }
    }
    return true;
}

}  // namespace strata::kernels::glm
