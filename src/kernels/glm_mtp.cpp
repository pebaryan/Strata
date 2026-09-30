// src/kernels/glm_mtp.cpp - the MTP head's own arithmetic (branch glm5next-port).  Host code, float, no
// CUDA.  The block's MLA and MoE are the already-verified operators; this covers the prologue, the
// concat order, and the head norm.  The reason each is worth its own fixture is in the header.
#include "strata/kernels/glm_mtp.hpp"

#include <cmath>
#include <cstring>

namespace strata::kernels::glm {

namespace {

/// rms over the whole vector with a gain, as ggml_rms_norm followed by a multiply.
void rms_norm_gain(const float* w, int n, const float* x, float* out) {
    double ss = 0.0;
    for (int i = 0; i < n; ++i) ss += (double) x[i] * (double) x[i];
    const float inv = 1.0f / std::sqrt((float) (ss / n) + MTP_RMS_EPS);
    for (int i = 0; i < n; ++i) out[i] = x[i] * inv * w[i];
}

}  // namespace

void mtp_input(const float* enorm, const float* hnorm, const float* eh_proj, int n_embd, const float* e,
               const float* h, float* cur_out, float* concat_out, float* e_norm_out, float* h_norm_out) {
    // reusable scratch rather than a stack buffer: n_embd is a build-time constant in this engine but not
    // here, and the reference path must not depend on a particular width.
    static thread_local float* e_buf = nullptr;
    static thread_local float* h_buf = nullptr;
    static thread_local int buf_n = 0;
    if (buf_n < n_embd) {
        delete[] e_buf;
        delete[] h_buf;
        e_buf = new float[(size_t) n_embd];
        h_buf = new float[(size_t) n_embd];
        buf_n = n_embd;
    }
    rms_norm_gain(enorm, n_embd, e, e_buf);
    rms_norm_gain(hnorm, n_embd, h, h_buf);
    if (e_norm_out) std::memcpy(e_norm_out, e_buf, (size_t) n_embd * sizeof(float));
    if (h_norm_out) std::memcpy(h_norm_out, h_buf, (size_t) n_embd * sizeof(float));
    // concat(e_norm, h_norm) along dim 0: the embedding occupies the FIRST n_embd entries
    if (concat_out) {
        std::memcpy(concat_out, e_buf, (size_t) n_embd * sizeof(float));
        std::memcpy(concat_out + n_embd, h_buf, (size_t) n_embd * sizeof(float));
    } else {
        for (int i = 0; i < n_embd; ++i) {
            const float* row_e = eh_proj + (size_t) i * (2 * n_embd);
            const float* row_h = row_e + n_embd;
            float acc = 0.0f;
            for (int j = 0; j < n_embd; ++j) acc += row_e[j] * e_buf[j] + row_h[j] * h_buf[j];
            cur_out[i] = acc;
        }
        return;
    }
    for (int i = 0; i < n_embd; ++i) {
        const float* row = eh_proj + (size_t) i * (2 * n_embd);
        float acc = 0.0f;
        for (int j = 0; j < 2 * n_embd; ++j) acc += row[j] * concat_out[j];
        cur_out[i] = acc;
    }
}

void mtp_head_norm(const float* head_norm_w, int n_embd, const float* x, float* out) {
    rms_norm_gain(head_norm_w, n_embd, x, out);
}

}  // namespace strata::kernels::glm
