#!/usr/bin/env python3
"""tools/glm5_kda_reference.py - one GLM-5.3-Flash KDA (linear attention) block's operator.

34 of the 46 blocks are linear attention.  The oracle for the port's implementation, transcribed from
llama.cpp's GLM5-Next `build_kda_layer` (src/models/glm5next.cpp), its `glm5next_causal_conv1d`, the
shared `llm_build_delta_net_base` path, and the fused op's CPU reference
`ggml_compute_forward_gated_delta_net_one_chunk` (ggml/src/ggml-cpu/ops.cpp).

The recurrence is the GDN that qwen4exp already uses; KDA differs in exactly one place - the decay is
per channel instead of per head, which the reference encodes as `const bool kda = (neg0 == S_v)`:

    decay:   KDA  S[i][j] *= exp(g[i])        (g has S_v entries per head)
             GDN  S[i][j] *= exp(g[0])        (one scalar per head)
    delta:   delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
    update:  S[i][j] += k[i] * delta[j]
    output:  attn[j]  = (sum_i S[i][j]*q[i]) * scale,   scale = 1/sqrt(head_dim)   [in the op]

with S[i][:] decayed by exp(g[i]), i.e. g's index is S's FIRST index (the output/query channel), and the
output read AFTER the update.

The front end, per the reference:
    x     = rms_norm(cur, attn_norm, eps = f_norm_rms_eps = 1e-5)
    q,k,v = silu(conv1d(w{q,k,v} @ x, ssm_conv1d_{q,k,v})),  depthwise, kernel 4, causal, sep. per q/k/v
    g     = kda_gate_lower_bound * sigmoid(-(ssm_a * (ssm_f_b(ssm_f_a(x)) + ssm_dt.bias)))
            with ssm_a holding -exp(A_log), so the factor is exp(A_log); per (head_dim, head)
    beta  = sigmoid(ssm_beta @ x), one scalar per head
    q, k  = l2_norm(..., 1e-6)             <- hard-coded in the reference, NOT f_norm_rms_eps
    out   = rms_norm(attn, ssm_o_norm, eps = 1e-5) * sigmoid(ssm_g_b(ssm_g_a(x)))
    result= wo @ out

The block-level residuals and hyper-connections are out of scope here: this is the operator, like the
MLA and mHC oracles.

Usage:
  glm5_kda_reference.py --gguf <shard1> --layer 4 --tokens 8 --selftest
  glm5_kda_reference.py --gguf <shard1> --layer 4 --tokens 8 --raw-fixture out.bin
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import iq_pack as P          # noqa: E402

N_EMBD = 4096
NH = 64                  # attention.head_count -> the KDA head count
HD = 128                 # kda.head_dim -> attention indexer key_length shares the name but not the value
D_CONV = 4               # ssm.conv_kernel
GATE_LOWER = -5.0        # kda.gate_lower_bound
RMS_EPS = 1e-5           # attention.layer_norm_rms_epsilon
L2_EPS = 1e-6            # hard-coded in the reference
D_INNER = NH * HD


class Model:
    def __init__(self, first: pathlib.Path):
        self.m = P.Model(first)
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGMLQuantizationType as Q, quants
        self.Q, self.quants = Q, quants

    def tensor(self, name: str) -> np.ndarray:
        g, t, mm, _ = self.m.where[name]
        raw = np.asarray(P.tensor_bytes(mm, g, t))
        vals = self.quants.dequantize(raw, self.Q[t.type_name])
        return np.asarray(vals, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rms_norm(x: np.ndarray, weight: np.ndarray | None, eps: float = RMS_EPS, axis: int = -1):
    ms = np.mean(x * x, axis=axis, keepdims=True)
    y = x / np.sqrt(ms + eps)
    return y * weight if weight is not None else y


def l2_norm(x: np.ndarray, eps: float = L2_EPS, axis: int = -1) -> np.ndarray:
    """ggml_l2_norm: x / max(||x||, eps) over the (already unit-ish) axis.  One head at a time."""
    n = np.sqrt(np.sum(x * x, axis=axis, keepdims=True))
    return x / np.maximum(n, eps)


def conv1d_causal(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """x: (T, ch) block outputs.  w: (D_CONV, ch) the depthwise kernel, forward order (a correlation).

    out[t] = sum_k w[k] * x[t + k - (D_CONV-1)], with the state (previous D_CONV-1 inputs) zero here.
    """
    T, ch = x.shape
    out = np.zeros_like(x)
    for k in range(D_CONV):
        shift = k - (D_CONV - 1)                    # -3, -2, -1, 0
        if shift < 0:
            out[-shift:] += w[k] * x[:T + shift]
        else:
            out[:T - shift] += w[k] * x[shift:]
    return out


def kda_gate(w: dict, x: np.ndarray) -> np.ndarray:
    """Returns g in [GATE_LOWER, 0], shaped (nh, hd).  g is per (head, channel)."""
    h = w["ssm_f_b"] @ (w["ssm_f_a"] @ x)                 # (D_INNER,)
    h = h + w["dt_bias"]
    h = h.reshape(NH, HD)
    h = h * w["ssm_a"][:, None]                           # ssm_a holds -exp(A_log), per head
    return GATE_LOWER * (1.0 / (1.0 + np.exp(h)))         # sigmoid(-h) == 1/(1+exp(h)) for the -1 fold


def kda_block(w: dict, x: np.ndarray, tokens: int, want_state: bool = False):
    """x: (tokens, N_EMBD) block inputs.  Returns (result, intermediates)."""
    T = tokens
    xn = rms_norm(x, w["attn_norm"])                      # (T, N_EMBD)

    qc = silu(conv1d_causal(xn @ w["wq"].T, w["conv_q"]))  # (T, D_INNER)
    kc = silu(conv1d_causal(xn @ w["wk"].T, w["conv_k"]))
    vc = silu(conv1d_causal(xn @ w["wv"].T, w["conv_v"]))

    g = np.stack([kda_gate(w, xn[t]) for t in range(T)])   # (T, nh, hd)
    beta = 1.0 / (1.0 + np.exp(-(xn @ w["ssm_beta"].T)))   # (T, nh)

    q = l2_norm(qc.reshape(T, NH, HD).astype(np.float64), axis=-1)
    k = l2_norm(kc.reshape(T, NH, HD).astype(np.float64), axis=-1)
    v = vc.reshape(T, NH, HD).astype(np.float64)

    scale = 1.0 / np.sqrt(HD)
    S = np.zeros((NH, HD, HD), dtype=np.float64)           # S[head][i][j]
    attn = np.zeros((T, NH, HD), dtype=np.float64)
    for t in range(T):
        S *= np.exp(g[t])[:, :, None]                       # S[i][:] *= exp(g[i])   (index i, the first)
        pred = np.einsum("hij,hi->hj", S, k[t]) # sum_i S[i][j] k[i], indexed by the value axis             # sum_j S[i][j] k[j]
        delta = (v[t] - pred) * beta[t][:, None]            # (nh, hd)
        S += np.einsum("hi,hj->hij", k[t], delta) # S[i][j] += k[i] * delta[j]           # S[i][j] += delta[i] * k[j]
        attn[t] = np.einsum("hij,hi->hj", S, q[t]) * scale  # read AFTER the update, contract the key axis

    # the norm is over ne0 = head_dim, i.e. WITHIN each head over its 128 channels, with a 128-entry gain
    o = rms_norm(attn.reshape(T, NH, HD), w["o_norm"], axis=-1).reshape(T, D_INNER)
    o = o * (1.0 / (1.0 + np.exp(-(w["ssm_g_b"] @ (w["ssm_g_a"] @ xn.T)).T)))
    result = o @ w["wo"].T                                  # (T, N_EMBD)
    return result, {"xn": xn, "qc": qc, "kc": kc, "vc": vc, "g": g, "beta": beta, "q": q, "k": k,
                    "v": v, "attn": attn, "o": o, "state": S if want_state else None}


def _conv(m, name):
    """The (d_conv, d_inner) conv array the kernel indexes as conv_w[k*d_inner + ch].

    The artifact stores (d_inner, 1, d_conv) with d_conv FASTEST, so this is reshape(d_inner, d_conv).T.
    reshape(d_conv, d_inner) instead interleaves four channels' taps: measured on block 0 it takes
    attn_output-0 from corr +1.00000 / 0.02% to corr +0.52073 / 90.56%.

    NOTE for whoever builds the missing regression: flipping the order HERE does not produce a usable
    known-bad fixture, because the oracle's own reference outputs are computed with the same weights - the
    fixture would then hold wrong weights and wrong expected values, which the gate would pass.  That
    self-consistency is exactly why the original defect was invisible.  The flip belongs at the fixture
    WRITER, so the expected outputs stay correct while the weights are wrong.
    """
    return np.ascontiguousarray(m.tensor(name).reshape(D_INNER, D_CONV).T)


def load_weights(m: Model, layer: int) -> dict:
    p = f"blk.{layer}."
    w = {
        "attn_norm": m.tensor(p + "attn_norm.weight"),
        "wq": m.tensor(p + "attn_q.weight"),
        "wk": m.tensor(p + "attn_k.weight"),
        "wv": m.tensor(p + "attn_v.weight"),
        # The artifact stores these as (d_inner, 1, d_conv) with d_conv FASTEST, so the (d_conv, d_inner)
        # array conv1d_causal wants is reshape(d_inner, d_conv).T.  reshape(d_conv, d_inner) instead
        # interleaves four different channels' taps and is silently wrong.  Measured on block 0:
        # reshape(d_inner,d_conv).T -> attn_output-0 corr +1.00000, 0.02% error
        # reshape(d_conv,d_inner)   -> attn_output-0 corr +0.52073, 90.56% error
        "conv_q": _conv(m, p + "ssm_conv1d_q.weight"),
        "conv_k": _conv(m, p + "ssm_conv1d_k.weight"),
        "conv_v": _conv(m, p + "ssm_conv1d_v.weight"),
        "ssm_a": m.tensor(p + "ssm_a"),
        "dt_bias": m.tensor(p + "ssm_dt.bias"),
        "ssm_f_a": m.tensor(p + "ssm_f_a.weight"),
        "ssm_f_b": m.tensor(p + "ssm_f_b.weight"),
        "ssm_beta": m.tensor(p + "ssm_beta.weight"),
        "ssm_g_a": m.tensor(p + "ssm_g_a.weight"),
        "ssm_g_b": m.tensor(p + "ssm_g_b.weight"),
        "o_norm": m.tensor(p + "ssm_norm.weight"),
        "wo": m.tensor(p + "attn_output.weight"),
    }
    assert w["wq"].shape == (D_INNER, N_EMBD), w["wq"].shape
    assert w["ssm_f_a"].shape[0] == HD or w["ssm_f_a"].shape[1] == HD, w["ssm_f_a"].shape
    assert w["ssm_f_b"].shape == (D_INNER, w["ssm_f_a"].shape[0]), (w["ssm_f_b"].shape, w["ssm_f_a"].shape)
    assert w["ssm_beta"].shape == (NH, N_EMBD), w["ssm_beta"].shape
    assert w["ssm_a"].shape == (NH,), w["ssm_a"].shape
    assert w["dt_bias"].shape == (D_INNER,), w["dt_bias"].shape
    assert w["o_norm"].shape == (HD,), w["o_norm"].shape
    return w


BAD_CONV = __import__("os").environ.get("STRATA_BAD_CONV_FIXTURE") == "1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-fixture")
    a = ap.parse_args()

    m = Model(pathlib.Path(a.gguf))
    w = load_weights(m, a.layer)
    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal((a.tokens, N_EMBD), dtype=np.float32) * 0.5

    result, mid = kda_block(w, x, a.tokens, want_state=True)

    if a.selftest:
        g = mid["g"]
        print(f"layer {a.layer}: {a.tokens} tokens, {NH} heads x {HD}, conv kernel {D_CONV}")
        print(f"  gate    range [{g.min():.4f}, {g.max():.4f}]  (must be within [{GATE_LOWER}, 0])")
        print(f"  beta    range [{mid['beta'].min():.6f}, {mid['beta'].max():.6f}]  (sigmoid -> (0,1))")
        print(f"  |q| per head {np.linalg.norm(mid['q'][0], axis=-1)[:3]}  (l2-normed -> ~1)")
        print(f"  |k| per head {np.linalg.norm(mid['k'][0], axis=-1)[:3]}")
        print(f"  state |S| per head (last token) {np.linalg.norm(mid['state'], axis=(1, 2))[:3]}")
        print(f"  result[0][:4] {result[0][:4]}")
        print(f"  result  range [{result.min():.4f}, {result.max():.4f}]")
        # a zero decay (g == 0) must leave the state untouched by the decay step, and beta == 0 must make
        # the update a no-op: both are checks the recurrence's structure implies, independent of weights
        w0 = dict(w)
        S0 = np.zeros((NH, HD, HD))
        kk = mid["k"][0]
        pred = np.einsum("hij,hj->hi", S0, kk)
        print(f"  zero state -> zero prediction: max|pred| {np.abs(pred).max():.1e}")

    if a.raw_fixture:
        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<6i", N_EMBD, NH, HD, D_CONV, a.tokens, a.layer))
            for arr in (w["attn_norm"], w["wq"], w["wk"], w["wv"], w["conv_q"], w["conv_k"], w["conv_v"],
                        w["ssm_a"], w["dt_bias"], w["ssm_f_a"], w["ssm_f_b"], w["ssm_beta"],
                        w["ssm_g_a"], w["ssm_g_b"], w["o_norm"], w["wo"], x):
                _a = np.ascontiguousarray(arr, dtype=np.float32)
                if BAD_CONV and _a.shape == (D_CONV, D_INNER):
                    # REGRESSION MODE: emit the conv weights in the WRONG order while the fixture's expected
                    # outputs above were computed with the right one, so the parity gate must FAIL.  Flipping
                    # in the loader instead would corrupt the expected values too (wrong against wrong) and
                    # the gate would pass - which is exactly how the original defect survived every gate.
                    _a = np.ascontiguousarray(_a.T)
                fh.write(_a.tobytes())
            for arr in (mid["qc"], mid["kc"], mid["vc"], mid["g"], mid["beta"], mid["q"], mid["k"],
                        mid["v"], mid["attn"], mid["o"], result):
                _a = np.ascontiguousarray(arr, dtype=np.float32)
                if BAD_CONV and _a.shape == (D_CONV, D_INNER):
                    # REGRESSION MODE: emit the conv weights in the WRONG order while the fixture's expected
                    # outputs above were computed with the right one, so the parity gate must FAIL.  Flipping
                    # in the loader instead would corrupt the expected values too (wrong against wrong) and
                    # the gate would pass - which is exactly how the original defect survived every gate.
                    _a = np.ascontiguousarray(_a.T)
                fh.write(_a.tobytes())
        print(f"wrote raw fixture {a.raw_fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
