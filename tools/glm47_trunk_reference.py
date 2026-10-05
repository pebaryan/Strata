#!/usr/bin/env python3
"""tools/glm47_trunk_reference.py - the GLM-4.7-Flash (deepseek2) trunk, float64 oracle.

The trunk is the loop the two block kinds compose into:

    x = hidden state
    for l in 0..L-1:   x = block_l(x, cache_l)          # block 0 dense, 1..46 MoE
    hn = rms_norm(x, output_norm, eps=1e-5)
    logits = W_out @ hn                                 # tied-style projection
    token  = argmax(logits)

Every stage below the loop is already gated block by block (glm47_block_parity, phases 4b/4c).  This
pins the three things the LOOP adds, which are the ones that are silent if wrong:

  * EACH LAYER OWNS ITS OWN MLA CACHE and every token appends to all of them.  A loop that reuses one
    cache for every layer hands layer l the keys layer l-1 wrote - a finite, plausible, wrong hidden
    state, and nothing per-block catches it;
  * the residual/norm order across a block boundary: the next block's input is the previous block's
    `out` (post second residual), not its pre-FFN `x2`;
  * the head is `rms_norm -> project -> argmax` on the LAST token's hidden state, eps 1e-5, a plain
    gain (not a folded `1+w`).

The token embedding is deliberately NOT here: x_0..x_{T-1} are the post-embedding hidden states, so
the loop is gated without a tokenizer or a 154880-row vocab.  Layer count and vocab are fixture
parameters - a 45-block / 154880-vocab fixture is not a unit test - but the geometry that matters
(MLA, MoE, the dense stem, the norms) is the model's own.

Usage:
  glm47_trunk_reference.py --raw-fixture out.bin [--layers 2] [--tokens 2] [--vocab 512]
                           [--seed 47] [--selftest]
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm47_block_reference as BR   # noqa: E402  (rms_norm, MLA, MOE, constants)
import glm47_mla_reference as MLA    # noqa: E402
import glm47_moe_reference as MOE    # noqa: E402

N_EMBD, N_HEAD, HEAD_DIM, KV_LORA, Q_LORA, N_ROT = BR.N_EMBD, BR.N_HEAD, BR.HEAD_DIM, BR.KV_LORA, BR.Q_LORA, BR.N_ROT
EPS = BR.EPS
FREQ_BASE = BR.FREQ_BASE
DENSE_FF = BR.DENSE_FF
rms_norm = BR.rms_norm

MAGIC = 0x47543437          # 'GT47'
FIXTURE_VERSION = 1

MLA_KEYS = ("wq_a", "q_a_norm", "wq_b", "wk_b", "kv_a", "kv_a_norm", "wv_b", "wo")


def make_layers(rng, n_layer):
    """Block 0 is the dense stem (ff 10240); every later block is the MoE (64 experts, top-4, +1 shared)."""
    layers = []
    for l in range(n_layer):
        mla = MLA.random_weights(rng)
        attn_norm = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
        ffn_norm = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
        if l == 0:
            fan = N_EMBD
            def rw(shape, fan=fan):
                return (rng.standard_normal(shape) / np.sqrt(fan)).astype(np.float32)
            layers.append({"kind": "dense", "mla": mla, "attn_norm": attn_norm, "ffn_norm": ffn_norm,
                           "wg": rw((DENSE_FF, N_EMBD)), "wu": rw((DENSE_FF, N_EMBD)), "wd": rw((N_EMBD, DENSE_FF)),
                           "dff": DENSE_FF, "kept": {}})
        else:
            layers.append({"kind": "moe", "mla": mla, "attn_norm": attn_norm, "ffn_norm": ffn_norm,
                           "moe": MOE.RandomModel(rng.integers(1, 2**31 - 1)), "kept": {}})
    return layers


def trunk_forward(layers, xs, pos_list, output_norm, W):
    """Run the loop.  xs[t] is token t's hidden state; pos_list[t] its position.  Returns the per-token
    per-layer hidden states, the final head pieces, and the per-layer caches (for the gate to re-derive)."""
    T, L = len(xs), len(layers)
    caches = [{"lat": np.zeros((0, KV_LORA), np.float32), "kpe": np.zeros((0, N_ROT), np.float32)} for _ in range(L)]
    hiddens = np.zeros((T, L, N_EMBD), np.float64)
    sel = {}
    for t in range(T):
        x = xs[t].astype(np.float64)
        pos = pos_list[t]
        for l, ly in enumerate(layers):
            xb = rms_norm(x.astype(np.float32), ly["attn_norm"]).astype(np.float32)
            z = np.zeros((1, KV_LORA), np.float32)
            zk = np.zeros((1, N_ROT), np.float32)
            s1 = MLA.mla_forward(ly["mla"], xb, np.vstack([caches[l]["lat"], z]),
                                 np.vstack([caches[l]["kpe"], zk]), pos, FREQ_BASE)
            caches[l]["lat"] = np.vstack([caches[l]["lat"], s1["kv"].astype(np.float32)])
            caches[l]["kpe"] = np.vstack([caches[l]["kpe"], s1["k_pe"].astype(np.float32)])
            attn = MLA.mla_forward(ly["mla"], xb, caches[l]["lat"], caches[l]["kpe"], pos, FREQ_BASE)["out"]
            x2 = x + attn
            ff = rms_norm(x2.astype(np.float32), ly["ffn_norm"])
            if ly["kind"] == "dense":
                ffn = MOE.expert_ffn(ly["wg"], ly["wu"], ly["wd"], ff.astype(np.float32))
            else:
                mo = MOE.moe_forward(ly["moe"], ff.astype(np.float32))
                ffn = mo["out"]
                sel[(t, l)] = mo["ids"].astype(np.int32)
                for id_ in mo["ids"].tolist():
                    ly["kept"][int(id_)] = mo["experts"][int(id_)]     # (gate, up, down) of the selected expert
            x = x2 + ffn
            hiddens[t, l] = x
    hn = rms_norm(hiddens[T - 1, L - 1].astype(np.float32), output_norm)
    logits = W.astype(np.float64) @ hn.astype(np.float64)
    argmax = int(np.argmax(logits))
    return hiddens, hn, logits, argmax, caches, sel


def write_fixture(path, layers, xs, pos_list, output_norm, W, hiddens, hn, logits, argmax, sel):
    T, L, V = len(xs), len(layers), W.shape[0]
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", MAGIC, FIXTURE_VERSION))
        fh.write(struct.pack("<9i", N_EMBD, L, V, N_HEAD, HEAD_DIM, KV_LORA, N_ROT, Q_LORA, T))
        fh.write(struct.pack("<2f", FREQ_BASE, EPS))
        for ly in layers:
            fh.write(struct.pack("<i", 0 if ly["kind"] == "dense" else 1))
            for k in MLA_KEYS:
                fh.write(np.ascontiguousarray(ly["mla"][k], np.float32).tobytes())
            fh.write(np.ascontiguousarray(ly["attn_norm"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(ly["ffn_norm"], np.float32).tobytes())
            if ly["kind"] == "dense":
                fh.write(struct.pack("<i", ly["dff"]))
                for t in (ly["wg"], ly["wu"], ly["wd"]):
                    fh.write(np.ascontiguousarray(t, np.float32).tobytes())
            else:
                o = ly["moe"]
                fh.write(struct.pack("<4i", MOE.N_EXPERT, MOE.N_USED, MOE.FF_EXP, MOE.FF_EXP))
                fh.write(np.ascontiguousarray(o.w_router, np.float32).tobytes())
                fh.write(np.ascontiguousarray(o.b, np.float32).tobytes())
                for t in o.shexp:
                    fh.write(np.ascontiguousarray(t, np.float32).tobytes())
                kept = sorted(ly["kept"])
                fh.write(struct.pack("<i", len(kept)))
                for e in kept:
                    fh.write(struct.pack("<i", e))
                    for t in ly["kept"][e]:
                        fh.write(np.ascontiguousarray(t, np.float32).tobytes())
        fh.write(np.ascontiguousarray(output_norm, np.float32).tobytes())
        fh.write(np.ascontiguousarray(W, np.float32).tobytes())
        for t in range(T):
            fh.write(np.ascontiguousarray(xs[t], np.float32).tobytes())
        for t in range(T):
            for l in range(L):
                fh.write(np.ascontiguousarray(hiddens[t, l], np.float32).tobytes())
        for t in range(T):
            for l in range(L):
                if (t, l) in sel:
                    fh.write(np.ascontiguousarray(sel[(t, l)], np.int32).tobytes())
        fh.write(np.ascontiguousarray(hn, np.float32).tobytes())
        fh.write(np.ascontiguousarray(logits, np.float32).tobytes())
        fh.write(struct.pack("<i", argmax))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-fixture")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--tokens", type=int, default=2)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    layers = make_layers(rng, a.layers)
    xs = [rng.standard_normal(N_EMBD).astype(np.float32) for _ in range(a.tokens)]
    pos_list = list(range(a.tokens))
    output_norm = (1.0 + 0.1 * rng.standard_normal(N_EMBD)).astype(np.float32)
    W = (rng.standard_normal((a.vocab, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32)

    hiddens, hn, logits, argmax, _, sel = trunk_forward(layers, xs, pos_list, output_norm, W)

    print(f"glm47 trunk reference: {a.layers} layers (0 dense, rest MoE), {a.tokens} tokens, vocab {a.vocab} (seed {a.seed})")
    print(f"  loop = per layer: attn_norm -> MLA(cache_l) -> +x -> ffn_norm -> {'dense' if a.layers == 1 else 'MoE'} -> +x ; then head")
    if a.selftest:
        for l, ly in enumerate(layers):
            if ly["kind"] == "moe":
                print(f"  layer {l} MoE kept experts: {sorted(ly['kept'])} "
                      f"(union over {a.tokens} tokens)")
        for t in range(a.tokens):
            print(f"  token {t}: |h| per layer " +
                  " ".join(f"{np.linalg.norm(hiddens[t, l]):.3f}" for l in range(a.layers)))
        print(f"  |hn| {np.linalg.norm(hn):.4f}  logits range [{logits.min():.3f}, {logits.max():.3f}]  argmax {argmax}")
    if a.raw_fixture:
        write_fixture(pathlib.Path(a.raw_fixture), layers, xs, pos_list, output_norm, W, hiddens, hn, logits, argmax, sel)
        sz = pathlib.Path(a.raw_fixture).stat().st_size
        print(f"wrote raw trunk fixture {a.raw_fixture}: {sz/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
