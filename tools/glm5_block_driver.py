#!/usr/bin/env python3
"""tools/glm5_block_driver.py - run the trunk's blocks on the REAL prompt, for comparison with the
reference dump.

This is phase 8's wiring check.  Every operator it composes already has its own oracle-backed gate
(mHC, KDA, MLA, indexer, MoE, the dense FFN); what is unverified until now is the GRAPH: the embedding
and the four-stream init, the hc_pre/attn_norm/site/hc_post order, and the two sites per block.  So this
driver reproduces the reference's own tensor names - in the reference's own order - and writes them for
tools/glm5_compare_dump.py, which reports the FIRST divergence.

Block 0's chain, as the reference names it:

    inp_embd      [4096,5]       the prompt's token embeddings
    hc_init       [4096,4,5]     the embedding broadcast into hc = 4 streams
    hc_attn_pre-0 [4096,5]       hc_pre: the layer input selected out of the streams
    attn_norm-0   [4096,5]
    kda_gate-0    [128,64,5]     per (channel, head) forget gate
    kda_beta-0    [1,64,5]       per head
    attn_output-0 [128,64,5]     the KDA recurrence's output, BEFORE the gated norm and wo
    hc_attn_post-0[4096,4,5]
    hc_ffn_pre-0  [4096,5]
    ffn_norm-0    [4096,5]
    ffn_out-0     [4096,5]       the leading blocks are DENSE (ffn_gate/up/down, ff 12288)
    l_out-0       [4096,4,5]

Orientations: the port's operators use [n_embd, T] (mHC), (T, n_embd) (KDA/dense FFN) and (T, nh, hd)
(the recurrence); the reference dumps [ne0, ne1, ...] with ne0 fastest, i.e. [n_embd, T], [hd, nh, T].
Every write below transposes explicitly, and the comment says which axis is which.

Usage:
  glm5_block_driver.py --gguf <shard1> --tokens "785 6722 315 9621 374" --blocks 1 \
      --dump /home/peb/moredata/glm5-port-dump
  python tools/glm5_compare_dump.py --ref /home/peb/moredata/glm5-ref-dump2 --port /home/peb/moredata/glm5-port-dump
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import iq_pack as P                              # noqa: E402
import glm5_hc_reference as HC                   # noqa: E402   the verified mHC
import glm5_kda_reference as KDA                 # noqa: E402   the verified KDA block
import glm5_moe_reference as MOE                 # noqa: E402   expert_ffn == the dense FFN arithmetic

N_EMBD = 4096
HC_STREAMS = 4
DENSE_LEAD = 3          # the leading blocks use a dense FFN, not the MoE


class Weights:
    """Loads only what the trunk prefix needs, from the artifact (not from the pack, so this stays an
    independent check of the artifact rather than of my own packer)."""

    def __init__(self, gguf: pathlib.Path):
        self.m = KDA.Model(gguf)          # shares the gguf/iq_pack loading used by every oracle

    def tensor(self, name: str) -> np.ndarray:
        return self.m.tensor(name)

    def embedding(self, token: int, vocab: int, name: str = "token_embd.weight") -> np.ndarray:
        """One row of the (quantized) token embedding table, dequantized.  ne0 = n_embd, so a row is a
        whole number of quant blocks and can be sliced directly.

        `name` picks the table: the reference's output projection prefers the artifact's own
        output.weight and only falls back to the tied token_embd (llama-model.cpp's TENSOR_NOT_REQUIRED
        then TENSOR_DUPLICATED), so a head that always uses token_embd is wrong whenever output.weight
        exists - which it does in this artifact."""
        g, t, mm, _ = self.m.m.where[name]
        row_bytes = t.expected_bytes() // vocab
        raw = np.asarray(P.tensor_bytes(mm, g, t))[token * row_bytes:(token + 1) * row_bytes]
        vals = self.m.quants.dequantize(raw, self.m.Q[t.type_name])
        v = np.asarray(vals, dtype=np.float32)
        assert v.size == N_EMBD, (name := "token_embd", v.size)
        return v


def write_tensor(dump: pathlib.Path, name: str, arr: np.ndarray) -> None:
    """fp32 with the reference dumper's header: type, ne0..ne3, data with ne0 fastest."""
    a = np.ascontiguousarray(arr, dtype=np.float32)
    ne = list(a.shape) + [1] * (4 - a.ndim)
    with open(dump / f"{name}.bin", "wb") as fh:
        fh.write(struct.pack("<5i", 0, ne[0], ne[1], ne[2], ne[3]))
        fh.write(a.tobytes())


def rms_norm_ne0(x: np.ndarray, w: np.ndarray | None = None, eps: float = 1e-5) -> np.ndarray:
    """ggml_rms_norm over ne0.  x is [n_embd, T] here, so the reduction is over axis 0."""
    y = x / np.sqrt(np.mean(x * x, axis=0, keepdims=True) + eps)
    return y * w[:, None] if w is not None else y


def dense_ffn(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, x: np.ndarray, limit: float) -> np.ndarray:
    """x: (T, n_embd).  The leading blocks' FFN - the same arithmetic as an expert, ff 12288 - and the
    SAME clamp: the reference reaches this site through build_ffn, which reads swiglu_clamp_shexp[il]
    (llama-graph.cpp:1816) for every il >= 0.  The limit is passed in, never defaulted, so this cannot
    silently become an unclamped second implementation of the oracle."""
    gate = x @ wg.T
    up = x @ wu.T
    if limit > 1e-6:
        gate = np.minimum(gate, limit)
        up = np.clip(up, -limit, limit)
    return (MOE.silu(gate) * up) @ wd.T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", required=True, help="space-separated token ids (from the reference dump)")
    ap.add_argument("--blocks", type=int, default=1, help="how many trunk blocks to run")
    ap.add_argument("--dump", required=True)
    ap.add_argument("--vocab", type=int, default=154880)
    a = ap.parse_args()

    dump = pathlib.Path(a.dump)
    dump.mkdir(parents=True, exist_ok=True)
    toks = [int(t) for t in a.tokens.split()]
    w = Weights(pathlib.Path(a.gguf))
    meta = MOE._read_meta(pathlib.Path(a.gguf))
    shexp_clamp = meta["glm5next.swiglu_clamp_shexp"]
    exp_clamp = meta["glm5next.swiglu_clamp_exp"]
    print(f"prompt tokens {toks} ({len(toks)} tokens), running {a.blocks} block(s); "
          f"swiglu clamps exp {exp_clamp[0]:g} shexp {shexp_clamp[0]:g}")

    # ---- the embedding and the four-stream init
    emb = np.stack([w.embedding(t, a.vocab) for t in toks])          # (T, n_embd)
    inp_embd = emb.T.copy()                                          # [n_embd, T]
    write_tensor(dump, "inp_embd", inp_embd)
    # the reference: inputs_embeds.unsqueeze(2).expand(-1, -1, hc, -1) - all hc streams start identical
    inpL = np.repeat(emb, HC_STREAMS, axis=0).reshape(len(toks), HC_STREAMS, N_EMBD).transpose(2, 1, 0)
    write_tensor(dump, "hc_init", inpL)                              # [n_embd, hc, T]

    for il in range(a.blocks):
        p = f"blk.{il}."
        hc_attn_fn = w.tensor(p + "hc_attn_fn.weight")               # (16384, 24) as C rows
        hc_attn_scale = w.tensor(p + "hc_attn_scale.weight")         # (3,)
        hc_attn_base = w.tensor(p + "hc_attn_base.weight")           # (24,)

        # ---- attention site: hc_pre -> attn_norm -> attention -> hc_post
        residual = inpL
        cur, post, comb = hc_pre_3d(inpL, hc_attn_fn, hc_attn_scale, hc_attn_base)
        write_tensor(dump, f"hc_attn_pre-{il}", cur)                 # [n_embd, T]
        # the normed value is dumped for comparison, but the KDA block gets the RAW residual: kda_block
        # normalizes internally (`xn = rms_norm(x, attn_norm)`), so feeding it the normed tensor normalizes
        # TWICE.  That cost 56% on the gate and inherited into every tensor downstream - measured, not
        # reasoned: the double-normed gate is 56.07% off the reference and the driver's was 56.1%.
        attn_normed = rms_norm_ne0(cur, w.tensor(p + "attn_norm.weight"))
        write_tensor(dump, f"attn_norm-{il}", attn_normed)

        kind = "kda" if il not in (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43) else "mla"
        if kind != "kda":
            print(f"  block {il} is an MLA block; this driver only does KDA blocks so far")
            return 3
        kw = KDA.load_weights(w.m, il)
        kda_out, kmid = KDA.kda_block(kw, cur.T, len(toks), want_state=True)
        g = kmid["g"]                                                # (T, nh, hd)
        beta = kmid["beta"]                                          # (T, nh)
        attn = kmid["attn"]                                          # (T, nh, hd), pre-norm/wo
        write_tensor(dump, f"kda_gate-{il}", np.transpose(g, (2, 1, 0)))        # [hd, nh, T]
        write_tensor(dump, f"kda_beta-{il}", beta.T[None])                      # [1, nh, T]
        # the reference's attn_output-N is [hd, nh, T]: a PER-HEAD tensor, so it is the value AFTER the
        # within-head RMS norm and the sigmoid gate and BEFORE wo - not the raw recurrence output.  The raw
        # one is kept under its own name because it is still useful for localizing inside the recurrence.
        # The reference's attn_output-N IS the raw recurrence output: correlating every intermediate
        # against it gives attn +1.00000 with the conv layout fixed, o +0.24, q/k/g ~0.  Part 16 said
        # otherwise and was wrong.
        write_tensor(dump, f"attn_output-{il}", np.transpose(attn, (2, 1, 0)))    # [hd, nh, T]
        _o = kmid["o"].reshape(len(toks), KDA.NH, KDA.HD)                         # post norm+gate, pre-wo
        write_tensor(dump, f"kda_o-{il}", np.transpose(_o, (2, 1, 0)))            # [hd, nh, T]

        inpL = hc_post_3d(kda_out.T, residual, post, comb)           # [n_embd, hc, T]
        write_tensor(dump, f"hc_attn_post-{il}", inpL)

        # ---- FFN site
        residual = inpL
        hc_ffn_fn = w.tensor(p + "hc_ffn_fn.weight")
        hc_ffn_scale = w.tensor(p + "hc_ffn_scale.weight")
        hc_ffn_base = w.tensor(p + "hc_ffn_base.weight")
        cur, post, comb = hc_pre_3d(inpL, hc_ffn_fn, hc_ffn_scale, hc_ffn_base)
        write_tensor(dump, f"hc_ffn_pre-{il}", cur)
        cur = rms_norm_ne0(cur, w.tensor(p + "ffn_norm.weight"))
        write_tensor(dump, f"ffn_norm-{il}", cur)

        if il < DENSE_LEAD:
            ffn = dense_ffn(w.tensor(p + "ffn_gate.weight"), w.tensor(p + "ffn_up.weight"),
                            w.tensor(p + "ffn_down.weight"), cur.T, shexp_clamp[il])
        else:
            print(f"  block {il} is a MoE block; this driver only does the dense leading blocks so far")
            return 3
        write_tensor(dump, f"ffn_out-{il}", ffn.T)                   # [n_embd, T]

        inpL = hc_post_3d(ffn.T, residual, post, comb)
        write_tensor(dump, f"l_out-{il}", inpL)                      # [n_embd, hc, T]
        print(f"  block {il} ({kind} attention, dense FFN) done; |l_out| {np.linalg.norm(inpL):.4f}")

    print(f"wrote {len(list(dump.glob('*.bin')))} tensors to {dump}")
    return 0


def hc_pre_3d(x: np.ndarray, fn: np.ndarray, scale: np.ndarray, base: np.ndarray):
    """The mHC oracle works on [hc, n_embd, nt]; the trunk keeps [n_embd, hc, T].  Its layer input comes
    back as [n_embd, T].  Both transposes are explicit here so the axis meaning is never implicit."""
    layer_in, _pre, post, comb = HC.hc_pre(np.transpose(x, (1, 0, 2)), fn, scale, base)
    return layer_in, post, comb


def hc_post_3d(cur_nt: np.ndarray, residual: np.ndarray, post: np.ndarray, comb: np.ndarray) -> np.ndarray:
    """cur_nt is [n_embd, T]; the oracle takes and returns [hc, n_embd, nt]."""
    out = HC.hc_post(cur_nt, np.transpose(residual, (1, 0, 2)), post, comb)
    return np.transpose(out, (1, 0, 2))


if __name__ == "__main__":
    raise SystemExit(main())
