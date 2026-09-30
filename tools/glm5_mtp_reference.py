#!/usr/bin/env python3
"""tools/glm5_mtp_reference.py - GLM-5.3-Flash's NextN/MTP draft head, and what it reuses.

Transcribed from `llama_model_glm5next::graph_mtp`:

    e_norm = rms_norm(tok_embd(token), nextn.enorm)      the token embedding, normed
    h_norm = rms_norm(h,              nextn.hnorm)        the PREVIOUS hidden state, normed
    concat = concat(e_norm, h_norm)                       e_norm FIRST, along dim 0 -> [2*n_embd]
    cur    = nextn.eh_proj @ concat                       -> [n_embd], and this is the residual base
    ... one plain pre-norm decoder block ...
    h_nextn = rms_norm(cur, nextn.shared_head_norm)       feeds the LM head

The block itself is NOT new arithmetic here, and the check below asserts that rather than assuming it:

  * its attention is the SAME MLA with the same tensor shapes as the trunk (wq_a [4096,1536],
    wq_b [1536,16384], wk_b [256,512,64], wkv_a_mqa [4096,512], wv_b [512,256,64], wo [16384,4096]),
    kq_scale = 1/sqrt(256) = 1/16 - which the phase-4a oracle and kernel already verify;
  * its FFN is the SAME sigmoid-gated MoE with a shared expert that phase 6 verifies;
  * what differs is that the block has NO hc_* tensors: plain residuals instead of the mHC, so
    `cur = attn_out + eh_proj_out` and `cur = moe_out + cur` are ordinary adds.

The implementation deliberately runs its MTP attention DENSELY - the reference's own comment: the
separate MTP context cannot see the trunk's indexer state, so it attends every cached cell. That
costs acceptance rate, never correctness.

Usage:
  glm5_mtp_reference.py --gguf <shard1> --selftest
  glm5_mtp_reference.py --gguf <shard1> --raw-fixture out.bin
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
RMS_EPS = 1e-5               # attention.layer_norm_rms_epsilon
MLA_LAYER = 3                # a trunk MLA block, to prove the MTP's attention shapes equal the trunk's
MTP_LAYER = 45


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

    def has(self, name: str) -> bool:
        return name in self.m.where


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = RMS_EPS) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def mtp_input(w: dict, e: np.ndarray, h: np.ndarray):
    """The MTP head's prologue.  Returns (cur, e_norm, h_norm, concat)."""
    e_norm = rms_norm(e, w["enorm"])
    h_norm = rms_norm(h, w["hnorm"])
    concat = np.concatenate([e_norm, h_norm])          # e_norm FIRST
    return w["eh_proj"] @ concat, e_norm, h_norm, concat


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-fixture")
    a = ap.parse_args()

    m = Model(pathlib.Path(a.gguf))
    p = f"blk.{MTP_LAYER}."
    w = {"enorm": m.tensor(p + "nextn.enorm.weight"), "hnorm": m.tensor(p + "nextn.hnorm.weight"),
         "eh_proj": m.tensor(p + "nextn.eh_proj.weight"),
         "head_norm": m.tensor(p + "nextn.shared_head_norm.weight")}
    assert w["eh_proj"].shape == (N_EMBD, 2 * N_EMBD), w["eh_proj"].shape

    rng = np.random.default_rng(a.seed)
    e = (rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)      # a token embedding
    h = (rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)      # the previous hidden state
    cur, e_norm, h_norm, concat = mtp_input(w, e, h)
    head_norm = rms_norm(cur, w["head_norm"])

    if a.selftest:
        print(f"MTP block blk.{MTP_LAYER}: {w['eh_proj'].shape} eh_proj")
        print(f"  has nextn.embed_tokens: {m.has(p + 'nextn.embed_tokens.weight')}  "
              f"(false -> the head reuses the model's tok_embd, as the reference's fallback does)")
        print(f"  has hc_* tensors: {any(m.has(p + 'hc_' + s + '_fn.weight') for s in ('attn', 'ffn'))}  "
              f"(false -> plain residuals, no hyper-connections)")
        print(f"  has nextn.shared_head_head: {m.has(p + 'nextn.shared_head_head.weight')}  "
              f"(false -> the head reuses the model's output projection)")
        print(f"  e_norm |.| {np.linalg.norm(e_norm):.4f}   h_norm |.| {np.linalg.norm(h_norm):.4f}")
        print(f"  concat[0:2] {concat[:2]}  (the first half must be e_norm: {e_norm[:2]})")
        print(f"  cur range [{cur.min():.4f}, {cur.max():.4f}]  |cur| {np.linalg.norm(cur):.4f}")
        # the reuse claim, asserted rather than assumed: the MTP's attention tensors must have exactly
        # the trunk's shapes, or the verified MLA kernel would not apply to them
        def attn_shapes(prefix):
            out = {}
            for n in m.m.where:
                if n.startswith(prefix) and "attn" in n[len(prefix):]:
                    out[n[len(prefix):]] = tuple(int(d) for d in m.m.where[n][1].shape)
            return out
        a3, a45 = attn_shapes(f"blk.{MLA_LAYER}."), attn_shapes(p)
        differs = {k: (a3[k], a45.get(k)) for k in a3 if a45.get(k) != a3[k]}
        print(f"  reuse asserted: MTP attention tensors equal the trunk's -> "
              f"{'yes' if not differs else differs}")
        hc = [k for k in a45 if k.startswith("hc_")]
        print(f"  hc_* tensors in the MTP block: {len(hc)}  (0 -> plain residuals)")
        idx = [k for k in a45 if k.startswith("indexer.")]
        print(f"  indexer tensors present: {len(idx)}  (the head runs DENSELY: the MTP context cannot see "
              f"the trunk's index, per the reference's own comment)")

    if a.raw_fixture:
        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<4i", N_EMBD, 0, 0, MTP_LAYER))
            for arr in (e, h, w["enorm"], w["hnorm"], w["eh_proj"], w["head_norm"],
                        e_norm, h_norm, concat, cur, head_norm):
                fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        print(f"wrote raw MTP fixture {a.raw_fixture}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
