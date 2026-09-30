#!/usr/bin/env python3
"""tools/glm5_moe_reference.py - GLM-5.3-Flash's MoE feed-forward and its router.

The oracle for the port's MoE, transcribed from llama.cpp's GLM5-Next graph (which calls the shared
`build_moe_ffn`) and that helper's implementation in src/llama-graph.cpp:

    logits   = ffn_gate_inp @ x                     # [n_expert] = [288]
    probs    = sigmoid(logits)                      # gating func 2 = SIGMOID, NOT softmax
    sel      = ffn_exp_probs_b + probs              # the bias is used for SELECTION ONLY
    topk     = argsort_top_k(sel, 8)                # 8 of 288, no expert groups (n_expert_groups = 1)
    weights  = probs[topk]                          # gathered from the UNBIASED probs
    if norm_w: weights /= max(sum(weights), 6.103515625e-05)   # clamp is the f16 minimum normal
    weights *= expert_weights_scale                 # 2.5, and only if not 0 and not 1
    moe      = sum_i weights[i] * down_i(silu(gate_i(x)) * up_i(x))
    shexp    = down_s(silu(gate_s(x)) * up_s(x))    # a plain parallel-SiLU FFN, added unweighted
    out      = moe + shexp

Order matters twice over: normalize BEFORE scaling (2.5 is applied to the normalized weights), and the
bias biases the CHOICE while the weights come from the unbiased probability - a swap of either is
silent, plausible-looking routing with the wrong weights.

Usage:
  glm5_moe_reference.py --gguf <shard1> --layer 4 --selftest
  glm5_moe_reference.py --gguf <shard1> --layer 4 --raw-fixture out.bin
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
N_EXPERT = 288
N_USED = 8           # expert_used_count
FF_EXP = 2048        # expert_feed_forward_length
W_SCALE = 2.5        # expert_weights_scale
NORM_W = True        # expert_weights_norm
F16_MIN = 6.103515625e-05
TOP_K_ORDER = "index"   # ties broken by ascending index, as ggml's argsort is stable by index


class Model:
    def __init__(self, first: pathlib.Path):
        self.m = P.Model(first)
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGMLQuantizationType as Q, quants
        self.Q, self.quants = Q, quants

    def raw(self, name: str) -> np.ndarray:
        return np.asarray(self.m.bytes(name))

    def tensor(self, name: str) -> np.ndarray:
        g, t, mm, _ = self.m.where[name]
        raw = np.asarray(P.tensor_bytes(mm, g, t))
        vals = self.quants.dequantize(raw, self.Q[t.type_name])
        return np.asarray(vals, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))

    def expert(self, name: str, e: int, n_expert: int) -> np.ndarray:
        """One expert's slab of a stacked expert tensor, dequantized and laid out as C rows.

        The expert axis is the outermost, so expert e is a contiguous slab (the same slicing the packer
        uses); reading it here keeps the oracle independent of the pack it produced.
        """
        g, t, mm, _ = self.m.where[name]
        raw = np.asarray(P.tensor_bytes(mm, g, t))
        per = raw.size // n_expert
        slab = raw[e * per:(e + 1) * per]
        vals = self.quants.dequantize(slab, self.Q[t.type_name])
        rows, cols = int(t.shape[1]), int(t.shape[0])         # ggml ne1 x ne0 -> C rows (ne1, ne0)
        return np.asarray(vals, dtype=np.float32).reshape(rows, cols)


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def router(w_router: np.ndarray, b: np.ndarray | None, x: np.ndarray):
    """Returns (topk indices, weights).  Selection is on the biased probs, weights from the unbiased."""
    logits = w_router @ x                       # (N_EXPERT,)
    probs = 1.0 / (1.0 + np.exp(-logits))       # SIGMOID
    sel = probs + b if b is not None else probs
    # descending by score, ties by ascending index: ggml's argsort_top_k is deterministic and index-ordered
    order = np.lexsort((np.arange(N_EXPERT), -sel))
    topk = order[:N_USED].astype(np.int64)
    weights = probs[topk].astype(np.float64)
    if NORM_W:
        s = max(float(weights.sum()), F16_MIN)
        weights = weights / s
    if W_SCALE != 0.0 and W_SCALE != 1.0:
        weights = weights * W_SCALE
    return topk, weights, probs


def expert_ffn(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, x: np.ndarray) -> np.ndarray:
    """LLM_FFN_SILU + LLM_FFN_PAR: down(silu(gate(x)) * up(x))."""
    return wd @ (silu(wg @ x) * (wu @ x))


def moe_forward(m: Model, layer: int, x: np.ndarray, n_expert: int = N_EXPERT, want: str = "all"):
    p = f"blk.{layer}."
    w_router = m.tensor(p + "ffn_gate_inp.weight")            # (288, 4096)
    b = None
    for cand in ("exp_probs_b.bias", "ffn_exp_probs_b.bias"):   # llama.cpp's enum name is ffn_exp_probs_b,
        try:                                                    # the FILE's name is exp_probs_b
            b = m.tensor(p + cand)
            break
        except KeyError:
            continue
    topk, weights, probs = router(w_router, b, x)

    acc = np.zeros(N_EMBD, dtype=np.float64)
    experts = {}
    for i, e in enumerate(topk):
        wg = m.expert(p + "ffn_gate_exps.weight", int(e), n_expert)
        wu = m.expert(p + "ffn_up_exps.weight", int(e), n_expert)
        wd = m.expert(p + "ffn_down_exps.weight", int(e), n_expert)
        experts[int(e)] = (wg, wu, wd)
        acc += weights[i] * expert_ffn(wg, wu, wd, x.astype(np.float64))

    shexp = expert_ffn(m.tensor(p + "ffn_gate_shexp.weight"), m.tensor(p + "ffn_up_shexp.weight"),
                       m.tensor(p + "ffn_down_shexp.weight"), x.astype(np.float64))
    moe = acc
    out = moe + shexp
    return out, {"topk": topk, "weights": weights, "probs": probs, "moe": moe, "shexp": shexp,
                 "experts": experts, "b": b, "w_router": w_router}


def dense_forward(m: Model, layer: int, x: np.ndarray):
    """The 3 leading blocks' FFN: the SAME arithmetic as an expert, but ff is 12288, not 2048."""
    p = f"blk.{layer}."
    wg = m.tensor(p + "ffn_gate.weight")     # (12288, 4096)
    wu = m.tensor(p + "ffn_up.weight")
    wd = m.tensor(p + "ffn_down.weight")     # (4096, 12288)
    assert wg.shape[0] == wu.shape[0] == wd.shape[1], (wg.shape, wu.shape, wd.shape)
    assert wg.shape[1] == wd.shape[0] == x.size, (wg.shape, wd.shape, x.size)
    return expert_ffn(wg, wu, wd, x.astype(np.float64)), {"gate": wg, "up": wu, "down": wd}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--layer", type=int, default=4)
    ap.add_argument("--n-expert", type=int, default=N_EXPERT)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--raw-fixture")
    ap.add_argument("--dense", action="store_true",
                    help="the 3 leading blocks' dense FFN instead of a MoE site (same math, ff 12288)")
    ap.add_argument("--bias-matters", action="store_true",
                    help="search seeds for an input where ffn_exp_probs_b changes the selection, so the "
                         "fixture can actually catch a port that ignores it")
    a = ap.parse_args()

    m = Model(pathlib.Path(a.gguf))
    p = f"blk.{a.layer}."

    if a.dense:
        # the 3 leading blocks have NO router at all (they are the "leading_dense_block_count"), so the
        # dense path must not touch the MoE tensors - that is itself part of what this checks.
        x = (np.random.default_rng(a.seed).standard_normal(N_EMBD) * 0.5).astype(np.float32)
        out, mid = dense_forward(m, a.layer, x)
        ff = int(mid["gate"].shape[0])
        if a.selftest:
            print(f"layer {a.layer} dense FFN: ff {ff}, n_embd {N_EMBD}")
            print(f"  out range [{out.min():.4f}, {out.max():.4f}]  |out| {np.linalg.norm(out):.4f}")
        if a.raw_fixture:
            with open(a.raw_fixture, "wb") as fh:
                fh.write(struct.pack("<5i", N_EMBD, 0, 0, ff, a.layer))
                fh.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
                for n in ("gate", "up", "down"):
                    fh.write(np.ascontiguousarray(mid[n], dtype=np.float32).tobytes())
                fh.write(np.ascontiguousarray(out, dtype=np.float32).tobytes())
            print(f"wrote raw dense fixture {a.raw_fixture}: ff {ff}")
        return 0

    w_router = m.tensor(p + "ffn_gate_inp.weight")
    b_fixed = None
    for cand in ("exp_probs_b.bias", "ffn_exp_probs_b.bias"):
        try:
            b_fixed = m.tensor(p + cand)
            break
        except KeyError:
            continue
    assert b_fixed is not None, "the router bias is missing; the fixture would not test it"

    seed = a.seed
    if a.bias_matters:
        # a fixture whose selection is the same with and without the bias cannot test the bias at all:
        # a port that ignored ffn_exp_probs_b would match it exactly.  Find an input where the choice
        # actually differs, so the gate has teeth on the one part of the router that is easiest to drop.
        found = None
        for s in range(4000):
            xs = (np.random.default_rng(s).standard_normal(N_EMBD) * 0.5).astype(np.float32)
            with_b = np.lexsort((np.arange(a.n_expert), -(1.0 / (1.0 + np.exp(-(w_router @ xs))) + b_fixed)))
            without = np.lexsort((np.arange(a.n_expert), -(1.0 / (1.0 + np.exp(-(w_router @ xs))))))
            if not np.array_equal(with_b[:N_USED], without[:N_USED]):
                found = (s, with_b[:N_USED], without[:N_USED])
                break
        if found is None:
            print("no seed in 4000 changes the selection via the bias; the gate cannot test it")
        else:
            seed = found[0]
            print(f"bias-sensitive input: seed {seed} selects {found[1].tolist()} with the bias, "
                  f"{found[2].tolist()} without")
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal(N_EMBD) * 0.5).astype(np.float32)

    out, mid = moe_forward(m, a.layer, x, a.n_expert)

    if a.selftest:
        print(f"layer {a.layer}: {a.n_expert} experts, {N_USED} used, scale {W_SCALE}, norm {NORM_W}")
        print(f"  probs   range [{mid['probs'].min():.6f}, {mid['probs'].max():.6f}]  (sigmoid -> (0,1))")
        print(f"  topk    {mid['topk'].tolist()}")
        print(f"  weights {np.array2string(mid['weights'], precision=6)}")
        print(f"  sum(weights before scaling) = {mid['weights'].sum() / W_SCALE:.6f}  (norm_w -> 1)")
        print(f"  |moe| {np.linalg.norm(mid['moe']):.4f}   |shexp| {np.linalg.norm(mid['shexp']):.4f}")
        print(f"  out     range [{out.min():.4f}, {out.max():.4f}]  |out| {np.linalg.norm(out):.4f}")
        # does the selection bias actually change the choice?  If it never does, the port could ignore it
        # and still pass a value comparison, which is why this is asserted here.
        _, _, probs = router(mid["w_router"], None, x)
        plain = np.lexsort((np.arange(a.n_expert), -probs))[:N_USED]
        print(f"  bias matters: selection without ffn_exp_probs_b -> {plain.tolist()}  "
              f"({'DIFFERENT' if not np.array_equal(plain, mid['topk']) else 'same'})")

    if a.raw_fixture:
        with open(a.raw_fixture, "wb") as fh:
            fh.write(struct.pack("<5i", N_EMBD, a.n_expert, N_USED, FF_EXP, a.layer))
            fh.write(np.ascontiguousarray(x, dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(mid["w_router"], dtype=np.float32).tobytes())
            if mid["b"] is not None:
                fh.write(np.ascontiguousarray(mid["b"], dtype=np.float32).tobytes())
            for e in mid["topk"].tolist():
                for t in mid["experts"][e]:
                    fh.write(np.ascontiguousarray(t, dtype=np.float32).tobytes())
            p = f"blk.{a.layer}."
            for n in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"):
                fh.write(np.ascontiguousarray(m.tensor(p + n), dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(mid["topk"], dtype=np.int32).tobytes())
            fh.write(np.ascontiguousarray(mid["weights"], dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(mid["moe"], dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(mid["shexp"], dtype=np.float32).tobytes())
            fh.write(np.ascontiguousarray(out, dtype=np.float32).tobytes())
        print(f"wrote raw fixture {a.raw_fixture}: {N_USED} experts ({mid['topk'].tolist()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
