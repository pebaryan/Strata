#!/usr/bin/env python3
"""tools/glm5_full_run.py - the trunk, all 46 blocks, on the REAL prompt, to the logits.

What this checks that the per-kernel gates cannot: the ASSEMBLY.  Every operator has its own
oracle-backed gate, but nothing until now ran the model's own chain - the layer-kind switch that picks
KDA or MLA out of `attention.head_count_kv`, the two sites per block, the caches, and the head - and
nothing produced a number the REFERENCE can be compared against.  The reference's own greedy
generation for this prompt is 12089 13 1084 374 (tools/glm5-ref/dump.tsv, tokens.txt), and its logits
argmax on the last prompt token is that first token.

So the gate is: argmax(logits) == 12089.  That is the same gate the C++ port must pass, which is the
point - this run separates "the assembly is wrong" from "the port is wrong", and it is cheap next to
debugging both at once.

Layer kinds come from the artifact: glm5next.attention.head_count_kv is a per-layer array, 1 = MLA
(latent attention, one kv head) and 0 = KDA.  For this prompt every indexer pool is selected (5 tokens
is 2 pool cells against top_k 2048), so the indexer's selection is all-tokens; that stops being true
past 8192 tokens and the run says so rather than assuming it.

Usage:
  glm5_full_run.py --gguf <shard1> --tokens "785 6722 315 9621 374" [--blocks 46]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import glm5_block_driver as DRV          # noqa: E402   hc_pre_3d / hc_post_3d / rms_norm_ne0
import glm5_moe_reference as MOE         # noqa: E402   the MoE / dense FFN oracle
import glm5_mla_reference as MLA         # noqa: E402   the MLA oracle
import glm5_kda_reference as KDA         # noqa: E402   the KDA oracle
import glm5_hc_reference as HC           # noqa: E402   hc_mean

N_EMBD = 4096
HC_STREAMS = 4
DENSE_LEAD = 3
GOLDEN = [12089, 13, 1084, 374]          # the reference's greedy generation for this prompt
N_CTX = 5


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--blocks", type=int, default=0,
                    help="how many trunk blocks (default: block_count - nextn_predict_layers, i.e. the "
                         "artifact's own trunk length)")
    ap.add_argument("--vocab", type=int, default=154880)
    ap.add_argument("--dump", help="also write per-block l_out-N tensors here, so the trunk can be "
                                    "bisected against the reference block by block")
    ap.add_argument("--save-hidden", help="write the post-trunk hidden state (last token, post "
                                          "output_norm) to this .npy so a head fix need not re-run the "
                                          "whole trunk")
    a = ap.parse_args()

    toks = [int(t) for t in a.tokens.split()]
    assert len(toks) == N_CTX, len(toks)
    gguf = pathlib.Path(a.gguf)
    w = DRV.Weights(gguf)
    moe_m = MOE.Model(gguf)
    mla_m = MLA.Model(gguf)

    kinds = moe_m.meta["glm5next.attention.head_count_kv"]
    n_all = int(moe_m.meta["glm5next.block_count"])
    n_nextn = int(moe_m.meta["glm5next.nextn_predict_layers"])
    # The trunk is NOT block_count blocks: the last block_count - nextn_predict_layers are the trunk and
    # the trailing ones are the NextN/MTP head.  For this artifact that is 45, not 46, and block 45 has
    # no hc_* tensors at all - running it as a trunk block fails on the first mHC weight, which is how
    # this was caught.  Derive the split from the artifact rather than assuming it.
    n_layer = n_all - n_nextn
    blocks = a.blocks if a.blocks else n_layer
    assert blocks <= n_layer, f"{blocks} exceeds the {n_layer} trunk blocks (block {n_layer} is the MTP head)"
    print(f"prompt {toks}; artifact: {n_all} blocks, {n_nextn} nextn -> {n_layer} trunk blocks, dense lead "
          f"{DENSE_LEAD}; running {blocks}")
    print(f"layer kinds: {sum(1 for k in kinds[:n_layer] if k == 1)} MLA, "
          f"{sum(1 for k in kinds[:n_layer] if k == 0)} KDA")
    print(f"swiglu clamps: exp {moe_m.meta['glm5next.swiglu_clamp_exp'][0]:g}, "
          f"shexp {moe_m.meta['glm5next.swiglu_clamp_shexp'][0]:g}")

    # ---- embedding, four-stream init
    emb = np.stack([w.embedding(t, a.vocab) for t in toks])            # (T, n_embd)
    inpL = np.repeat(emb, HC_STREAMS, axis=0).reshape(len(toks), HC_STREAMS, N_EMBD).transpose(2, 1, 0)

    latents: dict[int, list[np.ndarray]] = {}      # MLA layers: the per-token latent cache

    dump = pathlib.Path(a.dump) if getattr(a, "dump", None) else None
    if dump is not None:
        dump.mkdir(parents=True, exist_ok=True)

    for il in range(blocks):
        p = f"blk.{il}."
        is_mla = kinds[il] == 1
        residual = inpL

        # ---- attention site
        cur, post, comb = DRV.hc_pre_3d(inpL, w.tensor(p + "hc_attn_fn.weight"),
                                        w.tensor(p + "hc_attn_scale.weight"),
                                        w.tensor(p + "hc_attn_base.weight"))
        cur_raw = cur                                                    # pre-norm residual
        cur = DRV.rms_norm_ne0(cur, w.tensor(p + "attn_norm.weight"))   # [n_embd, T]

        if is_mla:
            mw = MLA.load_weights(mla_m, il)
            cache = latents.setdefault(il, [])
            outs = []
            for t in range(len(toks)):
                x = cur[:, t].astype(np.float64)
                kv = MLA.rms_norm(mw["kv_a"] @ x, mw["kv_a_norm"])       # the token's own latent
                cache.append(kv)
                assert len(cache) <= 2048 * 4, "the indexer's selection would stop being all-tokens"
                out, _qr, _qc, _kv, _attn, _p = MLA.mla_forward(mw, x, np.stack(cache))
                outs.append(out)
            attn_out = np.stack(outs).T                                  # [n_embd, T]
        else:
            kw = KDA.load_weights(w.m, il)
            kda_out, _kmid = KDA.kda_block(kw, cur_raw.T, len(toks), want_state=True)
            attn_out = kda_out.T                                         # [n_embd, T]

        inpL = DRV.hc_post_3d(attn_out, residual, post, comb)

        # ---- FFN site
        residual = inpL
        cur, post, comb = DRV.hc_pre_3d(inpL, w.tensor(p + "hc_ffn_fn.weight"),
                                        w.tensor(p + "hc_ffn_scale.weight"),
                                        w.tensor(p + "hc_ffn_base.weight"))
        cur = DRV.rms_norm_ne0(cur, w.tensor(p + "ffn_norm.weight"))

        if il < DENSE_LEAD:
            lim = float(moe_m.meta["glm5next.swiglu_clamp_shexp"][il])
            ffn = DRV.dense_ffn(w.tensor(p + "ffn_gate.weight"), w.tensor(p + "ffn_up.weight"),
                                w.tensor(p + "ffn_down.weight"), cur.T, lim).T
        else:
            outs = [MOE.moe_forward(moe_m, il, cur[:, t].astype(np.float64))[0] for t in range(len(toks))]
            ffn = np.stack(outs).T

        inpL = DRV.hc_post_3d(ffn, residual, post, comb)
        if dump is not None:
            DRV.write_tensor(dump, f"l_out-{il}", inpL)
        print(f"  block {il:2d} ({'MLA' if is_mla else 'KDA'}, "
              f"{'dense' if il < DENSE_LEAD else 'MoE'}) done; |l_out| {np.linalg.norm(inpL):.4f}",
              flush=True)

    # ---- head: mean over the streams, then output_norm, then the tied output projection
    cur = HC.hc_mean(np.transpose(inpL, (1, 0, 2)))                     # [n_embd, T]
    cur = DRV.rms_norm_ne0(cur, w.tensor("output_norm.weight"))
    x = cur[:, -1].astype(np.float64)                                    # the last prompt token
    if a.save_hidden:
        # keep the post-trunk state so a HEAD fix can be tested in seconds instead of re-running 45
        # blocks: the expensive part is the trunk, and it does not change when the head does.
        np.save(a.save_hidden, np.asarray(x, dtype=np.float32))
        print(f"saved the post-trunk hidden state to {a.save_hidden}")

    best_v, best_s = -1, -np.inf
    # the reference prefers the artifact's own output.weight and falls back to the TIED token_embd only
    # when it is absent (llama-model.cpp: TENSOR_NOT_REQUIRED, then TENSOR_DUPLICATED).  Using the
    # embedding here when output.weight exists silently computes the wrong projection.
    out_name = "output.weight" if "output.weight" in w.m.m.where else "token_embd.weight"
    print(f"output projection: {out_name}")
    for lo in range(0, a.vocab, 4096):                                   # in chunks: 154880 x 4096
        hi = min(lo + 4096, a.vocab)
        rows = np.stack([w.embedding(v, a.vocab, out_name) for v in range(lo, hi)]).astype(np.float64)
        s = rows @ x
        i = int(np.argmax(s))
        if s[i] > best_s:
            best_s, best_v = float(s[i]), lo + i

    print(f"\nclamp activity (count of pre-activations past the limit): {MOE.CLAMP_STATS}")
    print(f"logits argmax = {best_v}   expected {GOLDEN[0]}   "
          f"{'MATCH' if best_v == GOLDEN[0] else 'MISMATCH'}")
    print(f"|hidden| {np.linalg.norm(x):.4f}   logit range "
          f"[{best_s:.4f}, ...]")
    return 0 if best_v == GOLDEN[0] else 1


if __name__ == "__main__":
    raise SystemExit(main())
