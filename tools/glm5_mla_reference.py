#!/usr/bin/env python3
"""tools/glm5_mla_reference.py - the MLA block (nope-only, with absorption), as a reference.

The oracle for the port's MLA implementation, transcribed from llama.cpp's GLM5-Next graph
(`build_mla_layer`, src/models/glm5next.cpp).  Attention runs in the LATENT space:

    qr      = rms_norm(wq_a @ x)                     q_lora 1536
    q       = wq_b @ qr            -> [64][256]      64 heads x head_dim 256
    Qcur[h] = wk_b[h]^T @ q[h]     -> [64][512]      the ABSORBED query, in the latent space
    kv      = rms_norm(wkv_a_mqa @ x) -> [512]       the latent KV: ONE head, no rope, no k_pe
    attn[h] = softmax_j((Qcur[h] . kv_j) * 1/sqrt(256)) . kv_j        -> [64][512]
    v[h]    = wv_b[h] @ attn[h]    -> [64][256]      the un-absorption, per head
    out     = wo @ concat(v)       -> [4096]         head-major, the per-head dim innermost

so K and V are the SAME 512-dim vector and the head structure lives in the two absorbed projections.
Note the scale uses the MLA head size (256), not the latent rank.

Shapes are ggml's ([ne0, ne1, ...] with ne0 fastest); the tensors are loaded in that order, so a weight
of ggml shape [a, b] arrives here as (b, a) with a contiguous, exactly as ggml_mul_mat uses it.

Usage:
  glm5_mla_reference.py --gguf <shard1> --layer 3 --tokens 5 --raw-fixture out.bin
  glm5_mla_reference.py --gguf <shard1> --layer 3 --selftest
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

N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA = 4096, 64, 256, 512, 1536
KQ_SCALE = 1.0 / np.sqrt(np.float32(HEAD_DIM))


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


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def load_weights(m: Model, layer: int) -> dict:
    p = f"blk.{layer}."
    w = {
        "wq_a": m.tensor(p + "attn_q_a.weight"),          # (1536, 4096)
        "q_a_norm": m.tensor(p + "attn_q_a_norm.weight"),  # (1536,)
        "wq_b": m.tensor(p + "attn_q_b.weight"),          # (16384, 1536)
        "wk_b": m.tensor(p + "attn_k_b.weight"),          # (64, 512, 256)
        "kv_a": m.tensor(p + "attn_kv_a_mqa.weight"),     # (512, 4096)
        "kv_a_norm": m.tensor(p + "attn_kv_a_norm.weight"),  # (512,)
        "wv_b": m.tensor(p + "attn_v_b.weight"),          # (64, 256, 512)
        "wo": m.tensor(p + "attn_output.weight"),         # (4096, 16384)
    }
    assert w["wq_a"].shape == (Q_LORA, N_EMBD), w["wq_a"].shape
    assert w["wq_b"].shape == (N_HEAD * HEAD_DIM, Q_LORA), w["wq_b"].shape
    assert w["wk_b"].shape == (N_HEAD, KV_LORA, HEAD_DIM), w["wk_b"].shape
    assert w["kv_a"].shape == (KV_LORA, N_EMBD), w["kv_a"].shape
    assert w["wv_b"].shape == (N_HEAD, HEAD_DIM, KV_LORA), w["wv_b"].shape
    assert w["wo"].shape == (N_EMBD, N_HEAD * HEAD_DIM), w["wo"].shape
    return w


def mla_forward(w: dict, x: np.ndarray, cache: np.ndarray):
    """x: (4096,) one token.  cache: (T, 512) the latents attention may read, the current one last."""
    qr = rms_norm(w["wq_a"] @ x, w["q_a_norm"])                    # (1536,)
    q = (w["wq_b"] @ qr).reshape(N_HEAD, HEAD_DIM)                 # (64, 256) head-major, dim inner
    # Qcur[h] = wk_b[h]^T (512x256) @ q[h]
    qcur = np.einsum("hij,hj->hi", w["wk_b"], q)                   # (64, 512)
    kv = rms_norm(w["kv_a"] @ x, w["kv_a_norm"])                   # (512,) one head
    # attention over the cache, in the latent space
    scores = (qcur @ cache.T) * KQ_SCALE                           # (64, T)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    attn = p @ cache                                               # (64, 512)
    # un-absorb: v[h] = wv_b[h] (256x512) @ attn[h]
    v = np.einsum("hij,hj->hi", w["wv_b"], attn)                   # (64, 256)
    out = w["wo"] @ v.reshape(-1)                                  # (4096,)
    return out, qr, qcur, kv, attn, p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=5, help="cache length (the current token is the last)")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-fixture")
    a = ap.parse_args()

    m = Model(pathlib.Path(a.gguf))
    w = load_weights(m, a.layer)
    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal(N_EMBD, dtype=np.float32)
    cache = rng.standard_normal((a.tokens, KV_LORA), dtype=np.float32)
    out, qr, qcur, kv, attn, p = mla_forward(w, x, cache)

    if a.selftest:
        print(f"layer {a.layer}: cache {cache.shape}, kq_scale {KQ_SCALE:.6f}")
        print(f"  qr[:4]      {qr[:4]}")
        print(f"  kv[:4]      {kv[:4]}")
        print(f"  qcur[0,:4]  {qcur[0,:4]}")
        print(f"  attn[0,:4]  {attn[0,:4]}")
        print(f"  attn probs row0 sums to {p[0].sum():.8f}")
        print(f"  out[:4]     {out[:4]}  |out| {np.linalg.norm(out):.4f}")

    if a.raw_fixture:
        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<7i", N_EMBD, 1, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA, a.tokens))
            for arr in (w["wq_a"], w["q_a_norm"], w["wq_b"], w["wk_b"], w["kv_a"], w["kv_a_norm"],
                        w["wv_b"], w["wo"], x, cache, qr, qcur, kv, attn, out):
                fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        print(f"wrote raw fixture {a.raw_fixture}: layer {a.layer}, cache {a.tokens} latent(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
