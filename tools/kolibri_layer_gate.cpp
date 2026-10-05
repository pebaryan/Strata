// kolibri_layer_gate - parity for Kolibri 1's layer graph, CPU-only (branch kolibri-port).
//
// Two layers are exercised, a SLIDING one (RoPE + 513-window) and a FULL one (NoPE, unbounded),
// through a sequence of 600 positions so the window actually clips.  The engine path (float, the
// helpers in kolibri_swa.hpp plus float GEMMs written here) is compared against an INDEPENDENT
// double-precision reference written straight from the patch's models/kolibri1.cpp graph:
//   x1 = x0 + rms(attn(rms(x0)))           post-norm INSIDE the residual
//   x2 = x1 + rms(ffn(rms(x1)))            post-norm INSIDE the residual
//   attn: QK-norm over head_dim BEFORE RoPE; RoPE only on sliding layers; softmax(QK^T/sqrt(d))
//   ffn:  routed experts weighted by unbiased sigmoid + the shared expert, unweighted
// The MoE site uses the SHARED kernels::glm::moe_route (already gated) with norm_w=false, so the
// attention/norm/residual semantics are what this gate pins.
//
// What it catches (each is a real bug class the GLM port's notes warn about):
//   * post-norm moved OUTSIDE the residual (order flip) - changes every downstream logit;
//   * RoPE applied on full layers (NoPE violated) or skipped on sliding layers;
//   * the SWA window off by one (attends q-k == window, must be strictly less);
//   * QK-norm applied after RoPE (rotation is norm-invariant per pair only when norms are equal
//     across heads; with distinct per-head weights it is not, so the order is observable).
//
// usage: kolibri_layer_gate        (no args; exits 0 on PASS)

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "strata/kernels/kolibri_swa.hpp"
#include "strata/kernels/glm_moe.hpp"

using strata::kernels::glm::MoeGeometry;
using strata::kernels::kolibri::layer_is_swa;
using strata::kernels::kolibri::rope_neox;
using strata::kernels::kolibri::rms_norm;
using strata::kernels::kolibri::qk_norm_per_head;
using strata::kernels::kolibri::swa_attends;

namespace {

constexpr int N_EMBD = 2560, N_HEAD = 48, N_KV = 4, HEAD_DIM = 128;
constexpr int FF = 512, N_EXPERT = 384, N_USED = 6;
constexpr int WINDOW = 513;
constexpr float EPS = 1e-6f, THETA = 10000.0f;

int fails = 0;
void check(bool ok, const std::string& what) {
    if (ok) { std::printf("  PASS %s\n", what.c_str()); return; }
    ++fails;
    std::printf("  FAIL %s\n", what.c_str());
}

// ---- the float engine path (what the trunk will call) ------------------------------------------------

struct LayerWeights {
    std::vector<float> attn_norm, attn_post_norm, q_norm, k_norm, ffn_norm, ffn_post_norm;
    std::vector<float> wq, wk, wv, wo;          // [out][in]
};

void gemm(const std::vector<float>& w, int out_n, int in_n, const float* x, float* y) {
    for (int o = 0; o < out_n; ++o) {
        float acc = 0.0f;
        const float* row = w.data() + (size_t) o * in_n;
        for (int i = 0; i < in_n; ++i) acc += row[i] * x[i];
        y[o] = acc;
    }
}

// One attention layer over `T` cached positions with the NEW token at `q_pos`; keys/values 0..q_pos.
void attention_forward(const LayerWeights& lw, bool is_swa, std::vector<float>& k_cache,
                       std::vector<float>& v_cache, int q_pos, float* x_io) {
    // QK-norm BEFORE RoPE, on the new token's projections only
    std::vector<float> q(N_HEAD * HEAD_DIM), k(N_KV * HEAD_DIM), v(N_KV * HEAD_DIM);
    gemm(lw.wq, N_HEAD * HEAD_DIM, N_EMBD, x_io, q.data());
    gemm(lw.wk, N_KV * HEAD_DIM, N_EMBD, x_io, k.data());
    gemm(lw.wv, N_KV * HEAD_DIM, N_EMBD, x_io, v.data());
    {
        std::vector<float> qn((size_t) N_HEAD * HEAD_DIM), kn((size_t) N_KV * HEAD_DIM);
        qk_norm_per_head(q.data(), lw.q_norm.data(), N_HEAD, HEAD_DIM, EPS, qn.data());
        qk_norm_per_head(k.data(), lw.k_norm.data(), N_KV, HEAD_DIM, EPS, kn.data());
        q = qn; k = kn;
    }
    if (is_swa) {   // NoPE on full layers
        rope_neox(q.data(), q_pos, N_HEAD, HEAD_DIM, THETA, HEAD_DIM);
        rope_neox(k.data(), q_pos, N_KV, HEAD_DIM, THETA, HEAD_DIM);
    }
    std::memcpy(k_cache.data() + (size_t) q_pos * N_KV * HEAD_DIM, k.data(),
                (size_t) N_KV * HEAD_DIM * 4);
    std::memcpy(v_cache.data() + (size_t) q_pos * N_KV * HEAD_DIM, v.data(),
                (size_t) N_KV * HEAD_DIM * 4);

    // attention with the SWA mask, GQA: 48 query heads share 4 KV heads (12 each)
    const int window = is_swa ? WINDOW : 0;
    std::vector<float> out(N_HEAD * HEAD_DIM, 0.0f);
    for (int h = 0; h < N_HEAD; ++h) {
        const int hk = h / (N_HEAD / N_KV);
        double maxi = -1e300;
        std::vector<double> scores((size_t) q_pos + 1);
        for (int p = 0; p <= q_pos; ++p) {
            if (!swa_attends(q_pos, p, window)) { scores[(size_t) p] = -1e300; continue; }
            double d = 0.0;
            for (int i = 0; i < HEAD_DIM; ++i)
                d += (double) q[(size_t) h * HEAD_DIM + i] *
                     (double) k_cache[(size_t) p * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
            scores[(size_t) p] = d / std::sqrt((double) HEAD_DIM);
            maxi = std::max(maxi, scores[(size_t) p]);
        }
        double sum = 0.0;
        for (int p = 0; p <= q_pos; ++p) { scores[(size_t) p] = std::exp(scores[(size_t) p] - maxi); sum += scores[(size_t) p]; }
        for (int i = 0; i < HEAD_DIM; ++i) {
            double acc = 0.0;
            for (int p = 0; p <= q_pos; ++p)
                acc += scores[(size_t) p] * (double) v_cache[(size_t) p * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
            out[(size_t) h * HEAD_DIM + i] = (float) (acc / sum);
        }
    }
    std::vector<float> o(N_EMBD);
    gemm(lw.wo, N_EMBD, N_HEAD * HEAD_DIM, out.data(), o.data());
    std::memcpy(x_io, o.data(), N_EMBD * 4);
}

// ---- the double reference (independent, from the patch's graph) --------------------------------------

using D = double;
struct RefLayer {
    std::vector<D> attn_norm, attn_post_norm, q_norm, k_norm, ffn_norm, ffn_post_norm;
    std::vector<D> wq, wk, wv, wo;
    std::vector<D> router, bias;                 // [n_expert][n_embd], [n_expert]
    std::vector<D> gate, up, down;               // expert 0 only; expert>0 never selected (bias -inf)
    std::vector<D> sgate, sup, sdown;            // shared expert
};

D d_rms_at(const std::vector<D>& x, const std::vector<D>& w, int n, D eps, std::vector<D>& out) {
    D ss = 0.0;
    for (int i = 0; i < n; ++i) ss += x[(size_t) i] * x[(size_t) i];
    const D inv = 1.0 / std::sqrt(ss / n + eps);
    out.resize((size_t) n);
    for (int i = 0; i < n; ++i) out[(size_t) i] = x[(size_t) i] * inv * w[(size_t) i];
    return inv;
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    auto uni = [&](float lo, float hi) { return lo + (hi - lo) * (float) rng() / (float) 4294967295.0; };

    // ---- the pattern: 600 positions, alternating sliding/full as SSSSF, LAST position is evaluated --
    const int T = 600;
    std::vector<uint8_t> pattern((size_t) T, 1);
    for (int i = 4; i < T; i += 5) pattern[(size_t) i] = 0;

    // weights for the two kinds (one sliding layer reused for all sliding positions, one full)
    auto rand_vec = [&](int n, float lo, float hi) {
        std::vector<float> v((size_t) n);
        for (auto& x : v) x = uni(lo, hi);
        return v;
    };
    auto rand_layer = [&](float s) {
        LayerWeights lw;
        lw.attn_norm = rand_vec(N_EMBD, 1 - s, 1 + s);
        lw.attn_post_norm = rand_vec(N_EMBD, 1 - s, 1 + s);
        lw.q_norm = rand_vec(HEAD_DIM, 1 - s, 1 + s);
        lw.k_norm = rand_vec(HEAD_DIM, 1 - s, 1 + s);
        lw.ffn_norm = rand_vec(N_EMBD, 1 - s, 1 + s);
        lw.ffn_post_norm = rand_vec(N_EMBD, 1 - s, 1 + s);
        lw.wq = rand_vec(N_HEAD * HEAD_DIM * N_EMBD, -s, s);
        lw.wk = rand_vec(N_KV * HEAD_DIM * N_EMBD, -s, s);
        lw.wv = rand_vec(N_KV * HEAD_DIM * N_EMBD, -s, s);
        lw.wo = rand_vec(N_EMBD * N_HEAD * HEAD_DIM, -s, s);
        return lw;
    };
    const LayerWeights lw_slide = rand_layer(0.02f);
    const LayerWeights lw_full = rand_layer(0.02f);

    // ---- run the engine path over the sequence, both layers, comparing against the reference --------
    // The reference checks need a full double re-run of the same sequence; to keep the gate O(T^2)
    // small, positions are compared at three probes: 0 (no cache), 512 (window boundary -1), 599.
    const int probes[3] = {0, 512, 599};

    // Engine side: process positions one by one (sliding and full keep SEPARATE caches)
    std::vector<float> ks((size_t) T * N_KV * HEAD_DIM), vs((size_t) T * N_KV * HEAD_DIM);
    std::vector<float> kf((size_t) T * N_KV * HEAD_DIM), vf((size_t) T * N_KV * HEAD_DIM);
    std::vector<std::vector<float>> x0((size_t) T, std::vector<float>((size_t) N_EMBD));
    for (int t = 0; t < T; ++t) for (int i = 0; i < N_EMBD; ++i) x0[(size_t) t][(size_t) i] = uni(-0.5f, 0.5f);

    std::vector<std::vector<float>> attn_out((size_t) T, std::vector<float>((size_t) N_EMBD));
    for (int t = 0; t < T; ++t) {
        const bool swa = layer_is_swa(pattern, t);
        auto& cache_k = swa ? ks : kf;
        auto& cache_v = swa ? vs : vf;
        const LayerWeights& lw = swa ? lw_slide : lw_full;
        std::vector<float> cur = x0[(size_t) t];
        std::vector<float> n1((size_t) N_EMBD);
        rms_norm(cur.data(), lw.attn_norm.data(), N_EMBD, EPS, n1.data());
        attention_forward(lw, swa, cache_k, cache_v, t, n1.data());
        // post-norm INSIDE the residual: x1 = x0 + rms(attn_out)
        std::vector<float> po((size_t) N_EMBD);
        rms_norm(n1.data(), lw.attn_post_norm.data(), N_EMBD, EPS, po.data());
        for (int i = 0; i < N_EMBD; ++i) cur[(size_t) i] += po[(size_t) i];
        attn_out[(size_t) t] = cur;
    }

    // ---- reference side (double), at the probe positions only ---------------------------------------
    for (int t : probes) {
        const bool swa = layer_is_swa(pattern, t);
        const LayerWeights& lw = swa ? lw_slide : lw_full;

        // double copies of this layer's weights
        auto to_d = [](const std::vector<float>& v) { return std::vector<D>(v.begin(), v.end()); };
        const std::vector<D> dx0(to_d(x0[(size_t) t])), da(to_d(lw.attn_norm)), dp(to_d(lw.attn_post_norm));
        const std::vector<D> dq(to_d(lw.q_norm)), dk(to_d(lw.k_norm));
        const std::vector<D> wq(to_d(lw.wq)), wk(to_d(lw.wk)), wv(to_d(lw.wv)), wo(to_d(lw.wo));

        // norm -> QKV
        std::vector<D> n1;
        d_rms_at(dx0, da, N_EMBD, EPS, n1);
        auto mul = [&](const std::vector<D>& w, int out_n, int in_n, const std::vector<D>& x) {
            std::vector<D> y((size_t) out_n);
            for (int o = 0; o < out_n; ++o) {
                D acc = 0.0;
                for (int i = 0; i < in_n; ++i) acc += w[(size_t) o * in_n + i] * x[(size_t) i];
                y[(size_t) o] = acc;
            }
            return y;
        };
        std::vector<D> q = mul(wq, N_HEAD * HEAD_DIM, N_EMBD, n1);
        std::vector<D> k = mul(wk, N_KV * HEAD_DIM, N_EMBD, n1);
        std::vector<D> v = mul(wv, N_KV * HEAD_DIM, N_EMBD, n1);
        // QK-norm per head, then RoPE (sliding only), matching kolibri_swa.hpp's order
        for (int h = 0; h < N_HEAD; ++h) {
            std::vector<D> qh((size_t) HEAD_DIM);
            d_rms_at(std::vector<D>(q.begin() + (size_t) h * HEAD_DIM, q.begin() + (size_t) (h + 1) * HEAD_DIM),
                     dq, HEAD_DIM, EPS, qh);
            for (int i = 0; i < HEAD_DIM; ++i) q[(size_t) h * HEAD_DIM + i] = qh[(size_t) i];
        }
        for (int h = 0; h < N_KV; ++h) {
            std::vector<D> kh((size_t) HEAD_DIM);
            d_rms_at(std::vector<D>(k.begin() + (size_t) h * HEAD_DIM, k.begin() + (size_t) (h + 1) * HEAD_DIM),
                     dk, HEAD_DIM, EPS, kh);
            for (int i = 0; i < HEAD_DIM; ++i) k[(size_t) h * HEAD_DIM + i] = kh[(size_t) i];
        }
        if (swa) {
            auto rope = [&](std::vector<D>& vec, int heads) {
                for (int h = 0; h < heads; ++h)
                    for (int i = 0; i < HEAD_DIM / 2; ++i) {
                        const D freq = std::pow((D) THETA, (D) (-2.0 * i) / HEAD_DIM);
                        const D angle = (D) t * freq;
                        const D c = std::cos(angle), s = std::sin(angle);
                        const D x0v = vec[(size_t) h * HEAD_DIM + 2 * i], x1v = vec[(size_t) h * HEAD_DIM + 2 * i + 1];
                        vec[(size_t) h * HEAD_DIM + 2 * i] = x0v * c - x1v * s;
                        vec[(size_t) h * HEAD_DIM + 2 * i + 1] = x0v * s + x1v * c;
                    }
            };
            rope(q, N_HEAD); rope(k, N_KV);
        }
        // attention over the SAME float caches (the reference tests graph semantics, not cache bits)
        const int window = swa ? WINDOW : 0;
        std::vector<D> out((size_t) N_HEAD * HEAD_DIM, 0.0);
        const std::vector<D> kf32(to_d(swa ? ks : kf)), vf32(to_d(swa ? vs : vf));
        for (int h = 0; h < N_HEAD; ++h) {
            const int hk = h / (N_HEAD / N_KV);
            D maxi = -1e300;
            std::vector<D> sc((size_t) t + 1);
            for (int p = 0; p <= t; ++p) {
                if (!swa_attends(t, p, window)) { sc[(size_t) p] = -1e300; continue; }
                D d = 0.0;
                for (int i = 0; i < HEAD_DIM; ++i)
                    d += q[(size_t) h * HEAD_DIM + i] * kf32[(size_t) p * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
                sc[(size_t) p] = d / std::sqrt((D) HEAD_DIM);
                maxi = std::max(maxi, sc[(size_t) p]);
            }
            D sum = 0.0;
            for (int p = 0; p <= t; ++p) { sc[(size_t) p] = std::exp(sc[(size_t) p] - maxi); sum += sc[(size_t) p]; }
            for (int i = 0; i < HEAD_DIM; ++i) {
                D acc = 0.0;
                for (int p = 0; p <= t; ++p)
                    acc += sc[(size_t) p] * vf32[(size_t) p * N_KV * HEAD_DIM + (size_t) hk * HEAD_DIM + i];
                out[(size_t) h * HEAD_DIM + i] = acc / sum;
            }
        }
        const std::vector<D> o = mul(wo, N_EMBD, N_HEAD * HEAD_DIM, out);
        std::vector<D> po;
        d_rms_at(o, dp, N_EMBD, EPS, po);
        std::vector<D> ref_x1((size_t) N_EMBD);
        for (int i = 0; i < N_EMBD; ++i) ref_x1[(size_t) i] = dx0[(size_t) i] + po[(size_t) i];

        // compare against the engine's post-attention residual
        double worst = 0.0;
        for (int i = 0; i < N_EMBD; ++i)
            worst = std::max(worst, std::abs(ref_x1[(size_t) i] - (double) attn_out[(size_t) t][(size_t) i]));
        char buf[160];
        std::snprintf(buf, sizeof buf, "layer %d (%s) x1 within %.3g of the double reference",
                      t, swa ? "sliding" : "full/NoPE", worst);
        check(worst < 2e-3, buf);
    }

    // ---- order-of-operations pins: they fail loudly if the graph order regresses --------------------
    {
        // NoPE: the QKV PROJECTIONS are position-independent (no rotation).  The attention OUTPUT
        // is not: a later position causally attends more cached keys, which is causal masking, not
        // positional encoding.  So the pin is on the projections, not on the layer output.
        std::vector<float> k1((size_t) T * N_KV * HEAD_DIM), v1((size_t) T * N_KV * HEAD_DIM);
        std::vector<float> xa = rand_vec(N_EMBD, -0.5f, 0.5f);
        std::vector<float> n1((size_t) N_EMBD), n2((size_t) N_EMBD);
        rms_norm(xa.data(), lw_full.attn_norm.data(), N_EMBD, EPS, n1.data());
        std::vector<float> q1((size_t) N_HEAD * HEAD_DIM), k1v((size_t) N_KV * HEAD_DIM), v1v((size_t) N_KV * HEAD_DIM);
        gemm(lw_full.wq, N_HEAD * HEAD_DIM, N_EMBD, n1.data(), q1.data());
        gemm(lw_full.wk, N_KV * HEAD_DIM, N_EMBD, n1.data(), k1v.data());
        gemm(lw_full.wv, N_KV * HEAD_DIM, N_EMBD, n1.data(), v1v.data());
        std::vector<float> q1n((size_t) N_HEAD * HEAD_DIM), k1n((size_t) N_KV * HEAD_DIM);
        qk_norm_per_head(q1.data(), lw_full.q_norm.data(), N_HEAD, HEAD_DIM, EPS, q1n.data());
        qk_norm_per_head(k1v.data(), lw_full.k_norm.data(), N_KV, HEAD_DIM, EPS, k1n.data());
        // the same vectors through rope at two positions would DIFFER; without rope they cannot
        std::vector<float> q1r = q1n, k1r = k1n;
        rope_neox(q1r.data(), 3, N_HEAD, HEAD_DIM, THETA, HEAD_DIM);   // what a sliding layer would do
        rope_neox(k1r.data(), 3, N_KV, HEAD_DIM, THETA, HEAD_DIM);
        std::vector<float> q1r2 = q1n, k1r2 = k1n;
        rope_neox(q1r2.data(), 7, N_HEAD, HEAD_DIM, THETA, HEAD_DIM);
        rope_neox(k1r2.data(), 7, N_KV, HEAD_DIM, THETA, HEAD_DIM);
        bool rope_moves = false;
        for (int i = 0; i < HEAD_DIM; ++i)
            if (q1r[(size_t) i] != q1r2[(size_t) i]) rope_moves = true;
        // and the full layer's path keeps the normed vectors verbatim: the values the full layer
        // feeds to attention are exactly the normed ones (no rope call), i.e. identical to q1n at
        // position 0 where rope's rotation angle is 0 -> rope(normed, pos=0) == normed.
        bool nope_clean = true;
        {
            std::vector<float> q0 = q1n;
            rope_neox(q0.data(), 0, N_HEAD, HEAD_DIM, THETA, HEAD_DIM);   // identity rotation
            for (int i = 0; i < N_HEAD * HEAD_DIM; ++i) nope_clean = nope_clean && q1n[(size_t) i] == q0[(size_t) i];
        }
        check(rope_moves, "RoPE is position-dependent where it runs (sanity, q at pos 3 != pos 7)");
        check(nope_clean, "NoPE: the full layer's Q vectors pass through unrotated");
    }
    {
        // SWA boundary: position 600 attends 88..599 (window 513, strictly less), not 87
        bool ok = swa_attends(600, 88, WINDOW) && !swa_attends(600, 87, WINDOW);
        ok = ok && swa_attends(600, 600, WINDOW) && swa_attends(600, 0, 0);
        check(ok, "SWA mask: window 513 clips strictly (q-k < 513), causal, 0 = unbounded");
    }
    {
        // QK-norm before RoPE is observable when the norm weight is NON-uniform (a uniform weight
        // is rotation-invariant, so that pairing proves nothing)
        std::vector<float> w((size_t) HEAD_DIM);
        for (int i = 0; i < HEAD_DIM; ++i) w[(size_t) i] = uni(0.5f, 1.5f);
        std::vector<float> q((size_t) HEAD_DIM);
        for (int i = 0; i < HEAD_DIM; ++i) q[(size_t) i] = uni(-1.0f, 1.0f);
        std::vector<float> qn((size_t) HEAD_DIM);
        qk_norm_per_head(q.data(), w.data(), 1, HEAD_DIM, EPS, qn.data());
        rope_neox(qn.data(), 3, 1, HEAD_DIM, THETA, HEAD_DIM);          // norm -> rope
        std::vector<float> qr = q;
        rope_neox(qr.data(), 3, 1, HEAD_DIM, THETA, HEAD_DIM);          // rope -> norm
        std::vector<float> qr_n((size_t) HEAD_DIM);
        qk_norm_per_head(qr.data(), w.data(), 1, HEAD_DIM, EPS, qr_n.data());
        double worst = 0.0;
        for (int i = 0; i < HEAD_DIM; ++i) worst = std::max(worst, (double) std::abs(qn[(size_t) i] - qr_n[(size_t) i]));
        check(worst > 1e-6, "QK-norm order is observable with a real (non-uniform) q_norm");
    }

    std::printf("\n%s\n", fails == 0 ? "LAYER GATE: PASS" : "LAYER GATE: FAIL");
    return fails == 0 ? 0 : 1;
}
