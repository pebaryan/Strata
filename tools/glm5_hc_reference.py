#!/usr/bin/env python3
"""tools/glm5_hc_reference.py - the manifold-constrained hyper-connection (mHC), as a reference.

This is the ORACLE for the port's mHC implementation: numpy here, C++ in src/kernels/glm_hc.cpp, and
src/kernels/glm_hc_parity.cpp compares the two on the real weights of a block.  The semantics are taken
from llama.cpp's GLM5-Next graph (src/models/glm5next.cpp + deepseek4.cpp), line by line:

    inpL = repeat_4d(embed, n_embd, hc=4, n_tokens)          # the 4 streams start as 4 copies
    for each site (attn, ffn):
        residual = inpL
        cur, post, comb = hc_pre(inpL, fn, scale, base)
        cur = rms_norm(cur); cur = the site's computation
        inpL = hc_post(cur, residual, post, comb)
    head: mean over the 4 streams -> rms_norm(output_norm) -> output projection

hc_pre:  flat = x.reshape(hc*n_embd, nt); flat_norm = rms_norm(flat, eps)
         mixes[24, nt] = fn^T @ flat_norm                      # fn is [hc*n_embd, (2+hc)*hc]
         pre  = sigmoid(mixes[0:hc]  * scale[0] + base[0:hc])  + eps      (selects the layer input)
         post = sigmoid(mixes[hc:2hc]* scale[1] + base[hc:2hc]) * 2       (scales the layer output)
         comb = sinkhorn(reshape(mixes[2hc:] * scale[2] + base[2hc:], (hc, hc)))
         layer input = sum_h pre[h] * x[h]
hc_post: out[dst] = post[dst]*cur + sum_src comb[dst, src] * residual[src]

sinkhorn: softmax over dst, +eps, one column normalization, then (iters-1) x (row, column) - all with
eps added to every divisor, exactly as the reference does.

Usage:
  glm5_hc_reference.py --gguf <shard1> --layer 0 --fixture out.npz     write an oracle fixture
  glm5_hc_reference.py --gguf <shard1> --layer 0 --selftest            print a few values
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G      # noqa: E402
import iq_pack as P          # noqa: E402

HC = 4          # hyper_connection.count
SINKHORN_ITERS = 20
EPS = 1e-6      # hyper_connection.epsilon (metadata: 9.999999974752427e-07)
RMS_EPS = 1e-5  # attention.layer_norm_rms_epsilon (metadata: 9.999999747378752e-06)


class Model:
    """All shards of the model: a metadata-only shard 1 means the tensors are elsewhere."""

    def __init__(self, first: pathlib.Path):
        self.m = P.Model(first)
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGMLQuantizationType as Q, quants
        self.Q, self.quants = Q, quants

    def tensor(self, name: str) -> np.ndarray:
        """Dequantized, in ggml's own index order: ne0 fastest, so shape (ne1, ne0, ...) as C rows."""
        g, t, mm, _shard = self.m.where[name]
        raw = np.asarray(P.tensor_bytes(mm, g, t))
        vals = self.quants.dequantize(raw, self.Q[t.type_name])
        return np.asarray(vals, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))


def rms_norm(x: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    """ggml_rms_norm over ne0, unscaled (the reference calls it without a weight).

    eps here is the MODEL's rms eps (attention.layer_norm_rms_epsilon = 1e-5, passed to the graph
    context as norm_rms_eps), NOT hyper_connection.epsilon - the two differ and using the wrong one
    is invisible to a parity gate, because the oracle and the kernel then share the error.
    """
    return x / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)


def sinkhorn(comb: np.ndarray) -> np.ndarray:
    """comb[dst, src, nt]; softmax over dst, +eps, one column normalization, then (iters-1)x(row, col).

    The names come from the reference's ggml calls, where `sum_rows` sums over ne0 (= dst here):
      norm_cols (the permuted sum_rows) divides by the sum over src  -> per-dst divisor
      norm_rows divides by the sum over dst                          -> per-src divisor
    and the order is norm_cols, then (iters-1) x (norm_rows, norm_cols), so the last step is norm_cols.
    """
    e = np.exp(comb - comb.max(axis=0, keepdims=True))
    comb = e / e.sum(axis=0, keepdims=True)                    # softmax over dst (ggml_soft_max normalizes ne0)
    comb = comb + EPS
    comb = comb / (comb.sum(axis=1, keepdims=True) + EPS)      # norm_cols: sum over src
    for _ in range(1, SINKHORN_ITERS):
        comb = comb / (comb.sum(axis=0, keepdims=True) + EPS)  # norm_rows: sum over dst
        comb = comb / (comb.sum(axis=1, keepdims=True) + EPS)  # norm_cols: sum over src
    return comb


def hc_pre(x, fn, scale, base):
    """x: [hc, n_embd, nt] floats.  fn: [(2+hc)*hc, hc*n_embd].  scale: [3].  base: [(2+hc)*hc].

    fn is in ggml's index order: hc_fn is [hc*n_embd, 24], so as C rows it is (24, hc*n_embd) and
    ggml_mul_mat(hc_fn, flat) - which reduces over hc_fn's ne0 - is fn @ flat with no transpose.
    """
    hc, n_embd, nt = x.shape
    flat = x.reshape(hc * n_embd, nt)
    mixes = fn @ rms_norm(flat)                       # [(2+hc)*hc, nt]
    pre = 1.0 / (1.0 + np.exp(-(mixes[0:hc] * scale[0] + base[0:hc][:, None]))) + EPS
    post = 2.0 / (1.0 + np.exp(-(mixes[hc:2 * hc] * scale[1] + base[hc:2 * hc][:, None])))
    comb_raw = mixes[2 * hc:] * scale[2] + base[2 * hc:][:, None]
    # ggml reshapes the flat [hc*hc, nt] block to [hc, hc, nt] with ne0 (= dst) FASTEST, so
    # comb[dst][src] = flat[dst + src*hc].  numpy's C-order reshape assumes the first index is slowest,
    # which silently transposes the matrix - hence order="F" (the first axis fastest), which is exactly
    # ggml's convention.  The kernel got this right; this line is where the oracle had it wrong.
    comb = sinkhorn(comb_raw.reshape(hc, hc, nt, order="F"))
    layer_in = np.einsum("st,set->et", pre, x)        # sum_h pre[h] * x[h]
    return layer_in, pre, post, comb


def hc_mean(x: np.ndarray) -> np.ndarray:
    """The head's collapse: the mean over the streams.

    x is [hc, n_embd, nt] in this module's orientation.  The reference (glm5next.cpp:336,
    glm5next_hc_mean) sums ne1 - the hc axis - and scales by 1/hc, so the collapse is the arithmetic
    mean, not a weighted one.  It was on the path to the output without being covered by any gate.
    """
    return x.mean(axis=0)


def hc_post(cur, residual, post, comb):
    """cur: [n_embd, nt] the site's output.  residual: [hc, n_embd, nt].  Returns [hc, n_embd, nt]."""
    hc = residual.shape[0]
    out = np.empty_like(residual)
    for dst in range(hc):
        acc = post[dst][None, :] * cur
        for src in range(hc):
            acc = acc + comb[dst, src][None, :] * residual[src]
        out[dst] = acc
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=0)
    ap.add_argument("--site", default="attn", choices=("attn", "ffn"))
    ap.add_argument("--n-embd", type=int, default=4096)
    ap.add_argument("--tokens", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--fixture", help="write the oracle fixture here (.npz)")
    ap.add_argument("--raw-fixture", help="write the flat fixture the C++ parity test reads")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    shard = pathlib.Path(a.gguf)
    model = Model(shard)
    pre_name = f"blk.{a.layer}.hc_{a.site}_"
    fn = model.tensor(pre_name + "fn.weight")           # (24, hc*n_embd) as C rows
    base = model.tensor(pre_name + "base.weight").reshape(-1)    # [24]
    scale = model.tensor(pre_name + "scale.weight").reshape(-1)  # [3]
    if fn.shape != ((2 + HC) * HC, HC * a.n_embd):
        raise SystemExit(f"unexpected hc_fn shape {fn.shape}")
    if base.size != (2 + HC) * HC or scale.size != 3:
        raise SystemExit(f"unexpected base/scale shapes {base.shape} {scale.shape}")

    rng = np.random.default_rng(a.seed)
    x = rng.standard_normal((HC, a.n_embd, a.tokens), dtype=np.float32)
    layer_in, pre, post, comb = hc_pre(x, fn, scale, base)
    site_out = rng.standard_normal((a.n_embd, a.tokens), dtype=np.float32)   # a stand-in for attention
    streamed = hc_post(site_out, x, post, comb)

    if a.selftest:
        print(f"layer {a.layer} {a.site}: fn {fn.shape}  base {base.shape}  scale {scale}")
        print(f"  layer_in  [0,:3] {layer_in[0,:3]}")
        print(f"  post      [{post.min():.6f}, {post.max():.6f}] (2*sigmoid, so within (0,2))")
        print(f"  comb col0 sum {comb[:,0,0].sum():.6f}  row0 sum {comb[0,:,0].sum():.6f}")
        print(f"  streamed  [0,0,:3] {streamed[0,0,:3]}")

    if a.fixture:
        np.savez(a.fixture, fn=fn, base=base, scale=scale, x=x, site_out=site_out,
                 layer_in=layer_in, post=post, comb=comb, streamed=streamed)
        print(f"wrote {a.fixture}: fn {fn.shape}, x {x.shape}, {a.tokens} token(s)")

    if a.raw_fixture:
        # A flat little-endian fixture for the C++ test (src/kernels/glm_hc_parity.cpp).  Every tensor is
        # written one token at a time, in the layout the C++ side documents:
        #   int32 n_embd, nt, hc, mix_dim
        #   float fn[mix_dim * hc*n_embd]  base[mix_dim]  scale[3]
        #   then per token: x[hc*n_embd]  site_out[n_embd]
        #   then per token: pre[hc]  post[hc]  comb[hc*hc] (comb[dst][src], row-major)  layer_in[n_embd]
        #   then per token: streamed[hc*n_embd]
        import struct
        tn = lambda arr, axes: np.ascontiguousarray(np.transpose(arr, axes), dtype=np.float32)

        def write(fh, arr):
            fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())

        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<4i", a.n_embd, a.tokens, HC, (2 + HC) * HC))
            write(fh, fn)
            write(fh, base)
            write(fh, scale)
            write(fh, tn(x, (2, 0, 1)).reshape(-1))                  # per token: [hc][n_embd]
            write(fh, tn(site_out, (1, 0)).reshape(-1))              # per token: n_embd
            write(fh, tn(pre, (1, 0)).reshape(-1))                   # per token: hc
            write(fh, tn(post, (1, 0)).reshape(-1))
            write(fh, tn(comb, (2, 0, 1)).reshape(-1))               # per token: comb[dst][src]
            write(fh, tn(layer_in, (1, 0)).reshape(-1))
            write(fh, tn(streamed, (2, 0, 1)).reshape(-1))           # per token: [hc][n_embd]
        print(f"wrote raw fixture {a.raw_fixture} (n_embd {a.n_embd}, {a.tokens} token(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
