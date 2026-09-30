#!/usr/bin/env python3
"""tools/glm5_indexer_reference.py - the sparse indexer: its 4:1 pool and its selection.

The oracle for the port's indexer implementation, transcribed from llama.cpp's GLM5-Next
`build_dsa_top_k` (src/models/glm5next.cpp) and the pool inputs filled by
`llama_memory_hybrid_idx_context::set_input_kpool` (src/llama-memory-hybrid-idx.cpp):

    pooled[b][ch] = sum_m softmax_over_m(g_m[ch] + ape[m][ch]) * k_m[ch]     a 4:1 pool, channel-wise
    q[head]       = indexer.attn_q_b @ qr          -> 32 heads x 128
    wts[head]     = (indexer.proj @ cur)[head] / sqrt(d*nh)                  a per-head gate
    score[b]      = sum_head wts[head] * relu(q[head] . pooled[b]) + bias[b]
    bias[b]       = 0 when the pool is complete AND its last member is at/before the query, else -inf
    select        = the n_sel = min(n_pool, top_k/r) HIGHEST-scoring POOLS expanded to their cells,
                    plus the trailing incomplete pool's cells (always)

No softmax over the pools: the scores are a mask (which cells attention may read), not weights.

Two behaviours that are silent-wrong if missed, both documented by the reference itself:
  * the cut is on whole pools, never on cells - relu sends many pools to exactly 0.0 and top_k is
    unordered among equal keys, so a cell-level cut splits pools apart (PR #27754);
  * an incomplete pool has no pool key and can never win on score, so its cells are ALWAYS appended
    instead of consuming pool budget.

Fixture layout (little-endian; float32 unless stated):
  int32  n_embd, q_lora, d, nh, r, top_k, n_positions, n_sel
  float  attn_k[d*n_embd], k_norm_w[d], k_norm_b[d], attn_q_b[nh*d*q_lora], proj[nh*n_embd],
         c_gate[d*n_embd], ape[r*d], cur[n_embd], qr[q_lora], cache[n_positions*2d], pooled[n_pool*d]
  float  expected score[n_pool], expected bias[n_pool]
  int32  n_cells, then the selected cell indices (sorted; the set is what must match, since the
         reference itself cannot promise an order among equal scores)

Usage:
  glm5_indexer_reference.py --gguf <shard1> --layer 3 --tokens 12 --raw-fixture out.bin
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

N_EMBD, Q_LORA = 4096, 1536
D, NH, R, TOP_K = 128, 32, 4, 2048
SCALE = 1.0 / np.sqrt(np.float64(D * NH))


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


def layer_norm(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """The reference norm for the indexer key is LLM_NORM (layer norm, not rms), with weight AND bias."""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * weight + bias


def pool_keys(cache: np.ndarray, ape: np.ndarray, n_pool: int) -> np.ndarray:
    """cache: (n_pool*R, 2D) rows of [key(D) | gate(D)] -> (n_pool, D) pooled keys."""
    members = cache.reshape(n_pool, R, 2 * D)
    k, g = members[:, :, :D], members[:, :, D:]
    logits = g + ape[None, :, :]                                   # the intra-pool position bias
    logits = logits - logits.max(axis=1, keepdims=True)
    w = np.exp(logits)
    w /= w.sum(axis=1, keepdims=True)                              # softmax over the r members, per channel
    return (w * k).sum(axis=1)


def index_score(w: dict, cur: np.ndarray, qr: np.ndarray, pooled: np.ndarray) -> np.ndarray:
    q = (w["attn_q_b"] @ qr).reshape(NH, D)                        # (32, 128)
    wts = (w["proj"] @ cur) * SCALE                                # (32,) the per-head gate, scale folded in
    dots = q @ pooled.T                                            # (NH, n_pool)
    return (wts[:, None] * np.maximum(dots, 0.0)).sum(axis=0)      # relu, then the weighted head sum


def pool_bias(n_pool: int, n_positions: int, q_pos: int) -> np.ndarray:
    """A query may pick a pool only if the pool is complete and its last member is visible."""
    bias = np.zeros(n_pool, dtype=np.float64)
    for b in range(n_pool):
        ok = ((b + 1) * R <= n_positions) and ((b + 1) * R - 1 <= q_pos)
        bias[b] = 0.0 if ok else -np.inf
    return bias


def select(score: np.ndarray, bias: np.ndarray, n_pool: int, n_positions: int):
    """Whole pools only; ties break by index (deterministic here, not promised by the reference)."""
    total = score + bias
    n_sel = min(n_pool, TOP_K // R)
    order = np.lexsort((np.arange(n_pool), -total))
    cells = []
    for b in sorted(order[:n_sel].tolist()):
        cells.extend(range(b * R, b * R + R))
    if n_positions % R:                                            # the trailing incomplete pool
        cells.extend(range((n_positions // R) * R, n_positions))
    return np.array(sorted(set(cells)), dtype=np.int32), n_sel


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=12)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-fixture")
    a = ap.parse_args()

    m = Model(pathlib.Path(a.gguf))
    p = f"blk.{a.layer}."
    w = {
        "attn_k": m.tensor(p + "indexer.attn_k.weight"),
        "k_norm_w": m.tensor(p + "indexer.k_norm.weight"),
        "k_norm_b": m.tensor(p + "indexer.k_norm.bias"),
        "attn_q_b": m.tensor(p + "indexer.attn_q_b.weight"),
        "proj": m.tensor(p + "indexer.proj.weight"),
        "c_gate": m.tensor(p + "indexer_compressor_gate.weight"),
        "ape": m.tensor(p + "indexer_compressor_ape.weight"),
    }
    assert w["attn_k"].shape == (D, N_EMBD), w["attn_k"].shape
    assert w["attn_q_b"].shape == (NH * D, Q_LORA), w["attn_q_b"].shape
    assert w["proj"].shape == (NH, N_EMBD), w["proj"].shape
    assert w["c_gate"].shape == (D, N_EMBD), w["c_gate"].shape
    assert w["ape"].shape == (R, D), w["ape"].shape

    rng = np.random.default_rng(a.seed)
    hidden = rng.standard_normal((a.tokens, N_EMBD), dtype=np.float32)
    keys = layer_norm(hidden @ w["attn_k"].T, w["k_norm_w"], w["k_norm_b"])
    gates = hidden @ w["c_gate"].T
    cache = np.ascontiguousarray(np.concatenate([keys, gates], axis=1), dtype=np.float32)

    n_pool = a.tokens // R
    pooled = pool_keys(cache[:n_pool * R], w["ape"], n_pool) if n_pool else np.zeros((0, D), np.float32)

    cur = rng.standard_normal(N_EMBD, dtype=np.float32)
    qr = rng.standard_normal(Q_LORA, dtype=np.float32)
    score = index_score(w, cur, qr, pooled) if n_pool else np.zeros(0)
    bias = pool_bias(n_pool, a.tokens, a.tokens - 1)
    cells, n_sel = select(score, bias, n_pool, a.tokens)

    if a.selftest:
        print(f"layer {a.layer}: {a.tokens} positions -> {n_pool} complete pools of {R}, "
              f"n_sel {n_sel} (top_k {TOP_K} / r {R}), tail {a.tokens % R} cell(s)")
        print(f"  valid pools (bias 0): {int(np.isfinite(bias).sum())} of {n_pool}")
        print(f"  score[:4] {score[:4] if n_pool else 'n/a'}")
        print(f"  selected cells: {len(cells)} -> {cells[:14].tolist()}{' ...' if len(cells) > 14 else ''}")

    if a.raw_fixture:
        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<8i", N_EMBD, Q_LORA, D, NH, R, TOP_K, a.tokens, n_sel))
            for arr in (w["attn_k"], w["k_norm_w"], w["k_norm_b"], w["attn_q_b"], w["proj"], w["c_gate"],
                        w["ape"], cur, qr, hidden, cache, pooled, score.astype(np.float32),
                        bias.astype(np.float32)):
                fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
            fh.write(struct.pack("<i", int(cells.size)))
            fh.write(np.ascontiguousarray(cells, dtype=np.int32).tobytes())
        print(f"wrote raw fixture {a.raw_fixture}: {a.tokens} positions, {cells.size} selected cells, "
              f"n_sel {n_sel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
