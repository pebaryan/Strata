#!/usr/bin/env python3
"""tools/glm47_mla_reference.py - GLM-4.7-Flash (GGUF arch `deepseek2`) MLA, as a float64 oracle.

The oracle for the port's decoupled-RoPE MLA path, transcribed from llama.cpp / Kolibri's
deepseek2 `build_mla` (src/models/deepseek2.cpp, the `is_mla` branch).  Unlike GLM-5.3's nope-only
MLA, the query and key carry a rope half:

    qr      = rms_norm(wq_a @ x)                    q_lora 768
    q       = wq_b @ qr            -> [20][256]     20 heads x key_length_mla 256
    q_nope  = q[..., :192]                          192 = key_length_mla - rope.dimension_count
    q_pe    = rope_neox(q[..., 192:256])            the DECOUPLED rope half, 64 dims
    kv_mqa  = wkv_a_mqa @ x        -> [576]         kv_lora_rank 512 + 64 rope dims
    kv      = rms_norm(kv_mqa[:512]) -> [512]       the latent (V, and K's nope part)
    k_pe    = rope_neox(kv_mqa[512:576])            the key's roped half, ONE head
    Qcur[h] = wk_b[h] (512 x 192) @ q_nope[h] -> [512]      the absorbed query
    score[h,t] = (Qcur[h] . kv_t + q_pe[h] . k_pe_t) * kq_scale        kq_scale = mscale^2/sqrt(256)
    attn[h] = softmax_t(score) . kv_t                               V is the SAME 512 latent
    v[h]    = wv_b[h] (256 x 512) @ attn[h]  -> [20][256]
    out     = wo @ concat(v)      -> [2048]

RoPE is GGML_ROPE_TYPE_NEOX: rotate the pair (i, i + n_rot/2) by theta = pos * freq_base^(-2i/n_rot).
deepseek2 carries no YaRN (freq_scale 1, ext_factor 0), so mscale = attn_factor = 1 and
kq_scale = 1/sqrt(256) = 1/16; n_ctx_orig and the yarn factors do not enter.

Shapes are the REAL GLM-4.7-Flash shapes (n_embd 2048, 20 heads, key_length_mla 256, kv_lora 512,
q_lora 768, rope.dimension_count 64, rope.freq_base 1e6).  Weights and inputs are RANDOM (seeded)
and stored as float32 - the oracle then computes in float64 from those exact float32 bits, so the
C++ (float32) result can be diffed stage by stage.

Usage:
  glm47_mla_reference.py --raw-fixture out.bin [--seed 4321] [--tokens 7] [--selftest]
  glm47_mla_reference.py --layer 1 --gguf <model.gguf> --raw-fixture out.bin   # real weights (optional)
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

# Real GLM-4.7-Flash geometry (D:/aimodels/strata-pack-glm47/manifest.json).
N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA = 2048, 20, 256, 512, 768
N_ROT = 64
NOPE = HEAD_DIM - N_ROT
FREQ_BASE = 1_000_000.0
KQ_SCALE = 1.0 / np.sqrt(np.float64(HEAD_DIM))   # mscale^2 / sqrt(key_length_mla), mscale = 1

MAGIC = 0x47344D4C   # 'GLM4'
FIXTURE_VERSION = 1


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps) * weight


def rope_neox(x: np.ndarray, pos: int, freq_base: float) -> np.ndarray:
    """GGML_ROPE_TYPE_NEOX on the last axis: rotate (i, i + d/2), theta_i = pos * freq_base^(-2i/d)."""
    d = x.shape[-1]
    half = d // 2
    i = np.arange(half, dtype=np.float64)
    theta = np.float64(pos) * np.power(np.float64(freq_base), -2.0 * i / d)
    c, s = np.cos(theta), np.sin(theta)
    x0, x1 = x[..., :half], x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x0 * c - x1 * s
    out[..., half:] = x0 * s + x1 * c
    return out


def random_weights(rng: np.random.Generator) -> dict:
    """Unit-variance-ish weights so every intermediate stays O(1) and float32 round-off is ~1e-6."""
    def w(shape, fan_in):
        return (rng.standard_normal(shape) / np.sqrt(fan_in)).astype(np.float32)
    return {
        "wq_a": w((Q_LORA, N_EMBD), N_EMBD),
        "q_a_norm": (1.0 + 0.1 * rng.standard_normal(Q_LORA)).astype(np.float32),
        "wq_b": w((N_HEAD * HEAD_DIM, Q_LORA), Q_LORA),
        "wk_b": w((N_HEAD, KV_LORA, NOPE), NOPE),
        "kv_a": w((KV_LORA + N_ROT, N_EMBD), N_EMBD),
        "kv_a_norm": (1.0 + 0.1 * rng.standard_normal(KV_LORA)).astype(np.float32),
        "wv_b": w((N_HEAD, HEAD_DIM, KV_LORA), KV_LORA),
        "wo": w((N_EMBD, N_HEAD * HEAD_DIM), N_HEAD * HEAD_DIM),
    }


def mla_forward(w: dict, x: np.ndarray, cache_latent: np.ndarray, cache_kpe: np.ndarray,
                pos: int, freq_base: float = FREQ_BASE) -> dict:
    """One token through one deepseek2 MLA block.  All math in float64.

    x: (N_EMBD,).  cache_latent: (n_cache, KV_LORA) the normed latents.  cache_kpe: (n_cache, N_ROT)
    the roped k_pe, the current token's row last.  Returns the stage dict the parity gate compares.
    """
    wq_a, q_a_norm, wq_b = w["wq_a"].astype(np.float64), w["q_a_norm"].astype(np.float64), w["wq_b"].astype(np.float64)
    wk_b, kv_a, kv_a_norm = w["wk_b"].astype(np.float64), w["kv_a"].astype(np.float64), w["kv_a_norm"].astype(np.float64)
    wv_b, wo = w["wv_b"].astype(np.float64), w["wo"].astype(np.float64)
    x64 = x.astype(np.float64)
    lat = cache_latent.astype(np.float64)
    kpe = cache_kpe.astype(np.float64)

    qr = rms_norm(wq_a @ x64, q_a_norm)                       # (Q_LORA,)
    q = (wq_b @ qr).reshape(N_HEAD, HEAD_DIM)                 # (N_HEAD, HEAD_DIM)
    q_nope = q[:, :NOPE]                                      # (N_HEAD, NOPE)
    q_pe = rope_neox(q[:, NOPE:], pos, freq_base)             # (N_HEAD, N_ROT), after rope

    kv_mqa = kv_a @ x64                                        # (KV_LORA + N_ROT,)
    kv = rms_norm(kv_mqa[:KV_LORA], kv_a_norm)                # (KV_LORA,)
    k_pe = rope_neox(kv_mqa[KV_LORA:], pos, freq_base)        # (N_ROT,), after rope

    qcur = np.einsum("hij,hj->hi", wk_b, q_nope)              # (N_HEAD, KV_LORA)

    scores = (qcur @ lat.T + q_pe @ kpe.T) * KQ_SCALE         # (N_HEAD, n_cache)
    scores -= scores.max(axis=1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=1, keepdims=True)
    attn = p @ lat                                            # (N_HEAD, KV_LORA)

    v = np.einsum("hij,hj->hi", wv_b, attn)                   # (N_HEAD, HEAD_DIM)
    out = wo @ v.reshape(-1)                                  # (N_EMBD,)
    return {"qr": qr, "q_nope": q_nope, "q_pe": q_pe, "kv": kv, "k_pe": k_pe,
            "qcur": qcur, "attn": attn, "v": v, "out": out, "p": p}


def make_case(rng: np.random.Generator, w: dict, n_cache: int, pos: int, freq_base: float):
    x = rng.standard_normal(N_EMBD, dtype=np.float32)
    # history rows 0..n_cache-2 are the caller's already-appended latents / k_pe
    hist_lat = rng.standard_normal((n_cache - 1, KV_LORA), dtype=np.float32)
    hist_kpe = rng.standard_normal((n_cache - 1, N_ROT), dtype=np.float32)
    # the reference computes the current token's kv/k_pe and appends them (the last row)
    dummy = np.zeros((1, KV_LORA), np.float32)
    dummy_kpe = np.zeros((1, N_ROT), np.float32)
    stage = mla_forward(w, x, np.vstack([hist_lat, dummy]), np.vstack([hist_kpe, dummy_kpe]), pos, freq_base)
    cache_latent = np.vstack([hist_lat, stage["kv"].astype(np.float32)])
    cache_kpe = np.vstack([hist_kpe, stage["k_pe"].astype(np.float32)])
    stage = mla_forward(w, x, cache_latent, cache_kpe, pos, freq_base)
    return x, cache_latent, cache_kpe, pos, stage


def load_real_weights(gguf: pathlib.Path, layer: int) -> dict:
    """Optional: the model's own MLA weights, dequantized to float32.  Requires gguf + iq_pack."""
    here = pathlib.Path(__file__).resolve().parent
    sys.path.insert(0, str(here))
    import iq_pack as P
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    m = P.Model(gguf)

    def tensor(name):
        g, t, mm, _ = m.where[name]
        raw = np.asarray(P.tensor_bytes(mm, g, t))
        return np.asarray(quants.dequantize(raw, Q[t.type_name]), dtype=np.float32).reshape(
            tuple(int(d) for d in reversed(t.shape)))
    p = f"blk.{layer}."
    return {
        "wq_a": tensor(p + "attn_q_a.weight"),
        "q_a_norm": tensor(p + "attn_q_a_norm.weight"),
        "wq_b": tensor(p + "attn_q_b.weight"),
        "wk_b": tensor(p + "attn_k_b.weight"),
        "kv_a": tensor(p + "attn_kv_a_mqa.weight"),
        "kv_a_norm": tensor(p + "attn_kv_a_norm.weight"),
        "wv_b": tensor(p + "attn_v_b.weight"),
        "wo": tensor(p + "attn_output.weight"),
    }


def write_fixture(path: pathlib.Path, w: dict, cases: list, freq_base: float) -> None:
    n_cases = len(cases)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<9if", MAGIC, FIXTURE_VERSION, N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA,
                             N_ROT, n_cases, freq_base))
        for k in ("wq_a", "q_a_norm", "wq_b", "wk_b", "kv_a", "kv_a_norm", "wv_b", "wo"):
            fh.write(np.ascontiguousarray(w[k], dtype=np.float32).tobytes())
        for x, cache_latent, cache_kpe, pos, stage in cases:
            fh.write(struct.pack("<2i", int(pos), int(cache_latent.shape[0])))
            fh.write(np.ascontiguousarray(x, np.float32).tobytes())
            fh.write(np.ascontiguousarray(cache_latent, np.float32).tobytes())
            fh.write(np.ascontiguousarray(cache_kpe, np.float32).tobytes())
            for k in ("qr", "q_nope", "q_pe", "kv", "k_pe", "qcur", "attn", "v", "out"):
                fh.write(np.ascontiguousarray(stage[k], np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-fixture", help="write the parity fixture here")
    ap.add_argument("--seed", type=int, default=4321)
    ap.add_argument("--tokens", type=int, default=7, help="cache length of the primary case")
    ap.add_argument("--gguf", help="load REAL weights from this GGUF (default: seeded random)")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    if a.gguf:
        w = load_real_weights(pathlib.Path(a.gguf), a.layer)
        print(f"loaded real weights from {a.gguf} layer {a.layer}")
    else:
        w = random_weights(rng)
    for k, v in w.items():
        print(f"  weight {k:10s} {tuple(v.shape)}")

    # A spread of cache depths and positions: 1 row (the current token only), a short context, a
    # context where the rope has rotated far (pos in the thousands), and pos != n_cache-1 (context shift).
    specs = [(a.tokens, None), (1, 0), (5, 4), (33, None), (17, 4000)]
    cases = []
    for nc, pos in specs:
        p = (nc - 1) if pos is None else pos
        cases.append(make_case(rng, w, nc, p, FREQ_BASE))

    if a.selftest:
        st = cases[0][4]
        print(f"case0 cache {cases[0][1].shape}: kq_scale {KQ_SCALE:.6f}")
        print(f"  qr[:4]        {st['qr'][:4]}")
        print(f"  q_pe[0,:4]    {st['q_pe'][0,:4]}")
        print(f"  k_pe[:4]      {st['k_pe'][:4]}")
        print(f"  kv[:4]        {st['kv'][:4]}")
        print(f"  attn[0,:4]    {st['attn'][0,:4]}")
        print(f"  attn probs row0 sums to {st['p'][0].sum():.8f}")
        print(f"  out[:4]       {st['out'][:4]}  |out| {np.linalg.norm(st['out']):.4f}")

    if a.raw_fixture:
        write_fixture(pathlib.Path(a.raw_fixture), w, cases, FREQ_BASE)
        print(f"wrote raw fixture {a.raw_fixture}: {len(cases)} cases, "
              f"depths {[c[1].shape[0] for c in cases]}, pos {[c[3] for c in cases]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
