#!/usr/bin/env python3
"""tools/glm47_block_reference.py - one full GLM-4.7-Flash (deepseek2) decoder block, float64 oracle.

The trunk is 47 of these: `attn_norm -> MLA -> +x -> ffn_norm -> MoE -> +x`.  The MLA and the MoE are
already gated separately (tools/glm47_mla_reference.py, tools/glm47_moe_reference.py); this composes
them and pins the two things the composition adds, which are the ones that are silent if wrong:

  * the block norms are `rms_norm(x, w, eps=1e-5)` - ggml_rms_norm then a plain multiply by the gain,
    NOT a folded `(1 + w)`, and `eps` is the model's `attention.layer_norm_rms_epsilon`, not the
    hyper-connection epsilon;
  * the residual is `x + op(norm(x))` at BOTH sites, and the shared expert is part of the FFN output
    (`x2 + (moe + shexp)`), not added after the residual.

    xb   = rms_norm(x,  attn_norm_w, 1e-5)
    attn = MLA(xb, cache)                       # the phase-3 operator
    x2   = x + attn
    ff   = rms_norm(x2, ffn_norm_w, 1e-5)
    ffn  = MoE(ff)                              # moe + shared expert, the phase-4 operator
    out  = x2 + ffn

Block 0 is the leading dense stem (a plain FFN, ff 10240, no router); blocks 1..46 are the MoE above.
This reference covers the MoE blocks; the dense stem reuses the same `ffn_norm + residual` shell with
`MoE` replaced by `expert_ffn`.

Shapes are the REAL GLM-4.7-Flash shapes.  Weights are the model's own when --gguf is given, otherwise
seeded random weights of the same shapes; both are stored float32 and the oracle computes in float64
from those exact bits.

Usage:
  glm47_block_reference.py --raw-fixture out.bin [--seed 47] [--cases 3] [--selftest]
  glm47_block_reference.py --gguf <model.gguf> --layer 1 --raw-fixture out.bin
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm47_mla_reference as MLA   # noqa: E402
import glm47_moe_reference as MOE   # noqa: E402

N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA = 2048, 20, 256, 512, 768
N_ROT = 64
EPS = 1e-5
FREQ_BASE = 1_000_000.0
NOPE = HEAD_DIM - N_ROT
Q_DIM = N_HEAD * HEAD_DIM          # 5120
KV_DIM = KV_LORA + N_ROT           # 576

MAGIC = 0x47423437                 # 'GB47'
FIXTURE_VERSION = 1


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float = EPS) -> np.ndarray:
    x64 = x.astype(np.float64)
    return x64 / np.sqrt(np.mean(x64 * x64) + eps) * w.astype(np.float64)


def block_forward(x, mla_w, attn_norm_w, ffn_norm_w, moe_m, cache_latent, cache_kpe, pos):
    """One deepseek2 MoE block.  All math in float64; returns the stage dict the gate compares."""
    xb = rms_norm(x, attn_norm_w).astype(np.float32)
    stage = MLA.mla_forward(mla_w, xb, cache_latent, cache_kpe, pos, FREQ_BASE)
    attn = stage["out"]
    x2 = x.astype(np.float64) + attn
    ff = rms_norm(x2.astype(np.float32), ffn_norm_w)
    mo = MOE.moe_forward(moe_m, ff.astype(np.float32))
    out = x2 + mo["out"]
    return {"xb": xb, "attn": attn, "x2": x2, "ff": ff, "moe": mo["moe"], "shexp": mo["shexp"],
            "out": out, "ids": mo["ids"], "weights": mo["weights"], "experts": mo["experts"]}


def random_block(seed, layer_seed):
    rng = np.random.default_rng(seed)
    mla_w = MLA.random_weights(rng)
    attn_norm_w = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
    ffn_norm_w = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
    moe_m = MOE.RandomModel(layer_seed)
    return mla_w, attn_norm_w, ffn_norm_w, moe_m


def real_block(gguf, layer):
    mla_w = MLA.load_real_weights(gguf, layer)
    mm = MOE.GGUFModel(gguf, layer)
    attn_norm_w = mm.tensor(f"blk.{layer}.attn_norm.weight")
    ffn_norm_w = mm.tensor(f"blk.{layer}.ffn_norm.weight")
    return mla_w, attn_norm_w, ffn_norm_w, mm


def make_case(rng, mla_w, attn_norm_w, ffn_norm_w, moe_m, n_cache, pos):
    x = rng.standard_normal(N_EMBD).astype(np.float32)
    hist_lat = rng.standard_normal((n_cache - 1, KV_LORA)).astype(np.float32)
    hist_kpe = rng.standard_normal((n_cache - 1, N_ROT)).astype(np.float32)
    xb = rms_norm(x, attn_norm_w).astype(np.float32)
    # the reference computes the current token's kv / k_pe and appends them (the last cache row)
    z = np.zeros((1, KV_LORA), np.float32)
    zk = np.zeros((1, N_ROT), np.float32)
    s1 = MLA.mla_forward(mla_w, xb, np.vstack([hist_lat, z]), np.vstack([hist_kpe, zk]), pos, FREQ_BASE)
    cache_latent = np.vstack([hist_lat, s1["kv"].astype(np.float32)])
    cache_kpe = np.vstack([hist_kpe, s1["k_pe"].astype(np.float32)])
    stage = block_forward(x, mla_w, attn_norm_w, ffn_norm_w, moe_m, cache_latent, cache_kpe, pos)
    stage.update({"x": x, "cache_latent": cache_latent, "cache_kpe": cache_kpe, "pos": pos})
    return stage


def write_fixture(path, mla_w, attn_norm_w, ffn_norm_w, moe_m, cases):
    o = moe_m
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", MAGIC, FIXTURE_VERSION))
        fh.write(struct.pack("<10i", N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA, N_ROT,
                             MOE.N_EXPERT, MOE.N_USED, MOE.FF_EXP, len(cases)))
        fh.write(struct.pack("<2f", FREQ_BASE, EPS))
        for k in ("wq_a", "q_a_norm", "wq_b", "wk_b", "kv_a", "kv_a_norm", "wv_b", "wo"):
            fh.write(np.ascontiguousarray(mla_w[k], np.float32).tobytes())
        fh.write(np.ascontiguousarray(attn_norm_w, np.float32).tobytes())
        fh.write(np.ascontiguousarray(ffn_norm_w, np.float32).tobytes())
        fh.write(np.ascontiguousarray(o.w_router, np.float32).tobytes())
        fh.write(np.ascontiguousarray(o.b, np.float32).tobytes())
        for t in o.shexp:
            fh.write(np.ascontiguousarray(t, np.float32).tobytes())
        for c in cases:
            fh.write(struct.pack("<2i", int(c["pos"]), int(c["cache_latent"].shape[0])))
            fh.write(np.ascontiguousarray(c["x"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["cache_latent"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["cache_kpe"], np.float32).tobytes())
            for e in c["ids"].tolist():
                for t in c["experts"][e]:
                    fh.write(np.ascontiguousarray(t, np.float32).tobytes())
            for nm in ("xb", "attn", "x2", "ff"):
                fh.write(np.ascontiguousarray(c[nm], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["ids"], np.int32).tobytes())
            fh.write(np.ascontiguousarray(c["weights"], np.float32).tobytes())
            for nm in ("moe", "shexp", "out"):
                fh.write(np.ascontiguousarray(c[nm], np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-fixture")
    ap.add_argument("--gguf")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=7)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.gguf:
        mla_w, attn_norm_w, ffn_norm_w, moe_m = real_block(pathlib.Path(a.gguf), a.layer)
        src = f"REAL weights layer {a.layer} of {a.gguf}"
    else:
        mla_w, attn_norm_w, ffn_norm_w, moe_m = random_block(a.seed, a.seed)
        src = f"seeded random weights (seed {a.seed})"
    print(f"glm47 block reference: {src}")
    print(f"  block = attn_norm -> MLA -> +x -> ffn_norm -> MoE(+shexp) -> +x ; eps {EPS:g}")

    specs = [(a.tokens, None), (1, 0), (5, 4)]
    cases = []
    for i in range(a.cases):
        rng = np.random.default_rng(a.seed + 1000 + i)
        nc, pos = specs[min(i, len(specs) - 1)]
        p = (nc - 1) if pos is None else pos
        cases.append(make_case(rng, mla_w, attn_norm_w, ffn_norm_w, moe_m, nc, p))

    if a.selftest:
        c = cases[0]
        print(f"  case0 n_cache {c['cache_latent'].shape[0]} pos {c['pos']}")
        print(f"    ids      {c['ids'].tolist()}  weights {np.array2string(c['weights'], precision=6)}")
        print(f"    |xb| {np.linalg.norm(c['xb']):.4f}  |attn| {np.linalg.norm(c['attn']):.4f}  "
              f"|x2| {np.linalg.norm(c['x2']):.4f}  |ff| {np.linalg.norm(c['ff']):.4f}")
        print(f"    |moe| {np.linalg.norm(c['moe']):.4f}  |shexp| {np.linalg.norm(c['shexp']):.4f}  "
              f"|out| {np.linalg.norm(c['out']):.4f}")

    if a.raw_fixture:
        write_fixture(pathlib.Path(a.raw_fixture), mla_w, attn_norm_w, ffn_norm_w, moe_m, cases)
        sz = pathlib.Path(a.raw_fixture).stat().st_size
        print(f"wrote raw fixture {a.raw_fixture}: {len(cases)} cases, {sz/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
