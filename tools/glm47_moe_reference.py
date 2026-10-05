#!/usr/bin/env python3
"""tools/glm47_moe_reference.py - GLM-4.7-Flash (GGUF arch `deepseek2`) router + MoE, float64 oracle.

The oracle for the port's MoE site, transcribed from llama.cpp/Kolibri's shared `build_moe_ffn`
(src/llama-graph.cpp) AS THE deepseek2 GRAPH CALLS IT (src/models/deepseek2.cpp).

WHICH GATING MODE?  deepseek2.cpp reads `expert_gating_func` with `false` (optional) and, when the
key is absent, falls back:

    if (hparams.expert_gating_func == LLAMA_EXPERT_GATING_FUNC_TYPE_NONE) {
        if ((hparams.n_layer() == 47 || hparams.n_layer() == 48) && n_vocab == 154880) {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID;   // GLM 4.7 Lite
        } else {
            hparams.expert_gating_func = LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX;
        }
    }

The GLM-4.7-Flash artifact has NO `deepseek2.expert_gating_func` key, `block_count` 47 and
`vocab_size` 154880, so the oracle selects **SIGMOID**.  build_moe_ffn's SIGMOID branch is:

    logits   = ffn_gate_inp @ x                     # [n_expert] = [64]
    probs    = sigmoid(logits)                      # the gating function
    sel      = probs + exp_probs_b                  # SIGMOID THEN +bias  (selection only)
    ids      = argsort_top_k(sel, 4)                # 4 of 64, no groups (expert_group_count = 1)
    weights  = probs[ids]                           # from the UNBIASED sigmoid probs
    if norm_w: weights /= max(sum(weights), 6.103515625e-05)   # clamp is the f16 minimum normal
    weights *= expert_weights_scale                 # 1.8, only if != 0 and != 1, AFTER normalising
    moe      = sum_i weights[i] * down_i(silu(gate_i(x)) * up_i(x))
    shexp    = down_s(silu(gate_s(x)) * up_s(x))    # the +1 shared expert, added UNWEIGHTED
    out      = moe + shexp

Two orderings are silent if wrong.  (1) The bias is added to the *sigmoid probs* and drives SELECTION
ONLY; the weights come from the unbiased probs.  This is the DEEPSEEK2/SIGMOID path, NOT Kolibri's own
`SIGMOID_LOGIT_ADD` variant (graph:2085-2094), which would select on `logits + bias` = bias-then-sigmoid
and give a DIFFERENT top-4.  (2) The 1.8 scale is applied AFTER the normalisation, so sum(weights) = 1.8,
not 1.  A swap of either is silent, plausible-looking routing with the wrong experts or the wrong scale.

Shapes are the REAL GLM-4.7-Flash expert shapes (n_embd 2048, 64 experts, top-4, ff 1536, +1 shared,
w_scale 1.8, norm true).  Weights and inputs are stored as float32; the oracle computes in float64 from
those exact float32 bits, so the C++ (float32) result is diffed against it.  Weights are the model's
OWN weights when --gguf is given (the strongest fixture); otherwise seeded random weights of the same
shapes (so the gate runs without the 18 GB artifact).

Usage:
  glm47_moe_reference.py --raw-fixture out.bin [--seed 47] [--cases 3] [--selftest]
  glm47_moe_reference.py --gguf <model.gguf> --layer 1 --raw-fixture out.bin
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np

# Real GLM-4.7-Flash geometry (from the GGUF header, not inferred).
N_EMBD = 2048
N_EXPERT = 64
N_USED = 4            # expert_used_count
N_SHARED = 1          # expert_shared_count
FF_EXP = 1536         # expert_feed_forward_length
W_SCALE = 1.8         # expert_weights_scale
NORM_W = True         # expert_weights_norm
F16_MIN = 6.103515625e-05
N_GROUP = 1           # expert_group_count -> no group routing

MAGIC = 0x474D3437    # 'GM47'
FIXTURE_VERSION = 1


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def router(w_router: np.ndarray, b: np.ndarray | None, x: np.ndarray):
    """The SIGMOID router, exactly as build_moe_ffn's SIGMOID branch.

    Returns (topk_ids, weights, probs, logits, sel).  Selection is on `probs + b`; the weights are
    gathered from the UNBIASED `probs`; normalise then scale.
    """
    logits = w_router.astype(np.float64) @ x.astype(np.float64)   # (N_EXPERT,)
    probs = sigmoid(logits)                                       # SIGMOID
    sel = probs + b.astype(np.float64) if b is not None else probs  # SIGMOID then +bias (selection only)
    # descending by score, ties by ascending index: ggml's argsort_top_k is index-ordered
    order = np.lexsort((np.arange(w_router.shape[0]), -sel))
    topk = order[:N_USED].astype(np.int64)
    weights = probs[topk].astype(np.float64)
    if NORM_W:
        s = max(float(weights.sum()), F16_MIN)
        weights = weights / s
    if W_SCALE != 0.0 and W_SCALE != 1.0:
        weights = weights * W_SCALE
    return topk, weights, probs, logits, sel


def expert_ffn(wg: np.ndarray, wu: np.ndarray, wd: np.ndarray, x: np.ndarray) -> np.ndarray:
    """LLM_FFN_SILU + LLM_FFN_PAR: down(silu(gate(x)) * up(x)).  No clamp: the artifact has no key."""
    x = x.astype(np.float64)
    return wd.astype(np.float64) @ (silu(wg.astype(np.float64) @ x) * (wu.astype(np.float64) @ x))


# ---------------------------------------------------------------------------------------------------
# weights: random of the real shapes, or the model's own
# ---------------------------------------------------------------------------------------------------

def random_expert(seed: int, e: int):
    """One expert's {gate, up, down}, deterministic in (seed, e).  Unit-ish so float32 round-off is ~1e-6."""
    rng = np.random.default_rng((seed * 1_000_003 + e) & 0xFFFFFFFF)
    wg = (rng.standard_normal((FF_EXP, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32)
    wu = (rng.standard_normal((FF_EXP, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32)
    wd = (rng.standard_normal((N_EMBD, FF_EXP)) / np.sqrt(FF_EXP)).astype(np.float32)
    return wg, wu, wd


class RandomModel:
    """Seeded random weights of the real shapes.  Bias has the model's characteristic: a large shared
    offset with a small spread, so it drives the selection (matching exp_probs_b ~ 9.03 on the artifact)."""
    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.w_router = (rng.standard_normal((N_EXPERT, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32)
        self.b = (9.0 + 0.02 * rng.standard_normal(N_EXPERT)).astype(np.float32)
        self._seed = seed
        self._cache: dict[int, tuple] = {}
        self.shexp = (
            (rng.standard_normal((FF_EXP, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32),
            (rng.standard_normal((FF_EXP, N_EMBD)) / np.sqrt(N_EMBD)).astype(np.float32),
            (rng.standard_normal((N_EMBD, FF_EXP)) / np.sqrt(FF_EXP)).astype(np.float32),
        )

    def expert(self, e: int):
        if e not in self._cache:
            self._cache[e] = random_expert(self._seed, e)
        return self._cache[e]


class GGUFModel:
    """The artifact's own router, bias, shared expert and routed experts (dequantized to float32)."""
    def __init__(self, gguf: pathlib.Path, layer: int):
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
        import iq_pack as P
        from _paths import add_gguf_py
        add_gguf_py()
        from gguf import GGMLQuantizationType as Q, quants
        self._m = P.Model(gguf)
        self._Q, self._quants = Q, quants
        self.layer = layer
        p = f"blk.{layer}."
        self.w_router = self.tensor(p + "ffn_gate_inp.weight")
        self.b = self.tensor(p + "exp_probs_b.bias")
        self.shexp = (self.tensor(p + "ffn_gate_shexp.weight"),
                      self.tensor(p + "ffn_up_shexp.weight"),
                      self.tensor(p + "ffn_down_shexp.weight"))
        self._cache: dict[int, tuple] = {}

    def tensor(self, name: str) -> np.ndarray:
        g, t, mm, _ = self._m.where[name]
        raw = np.asarray(self._m.bytes(name))
        vals = self._quants.dequantize(raw, self._Q[t.type_name])
        return np.asarray(vals, dtype=np.float32).reshape(tuple(int(d) for d in reversed(t.shape)))

    def expert(self, e: int):
        """One expert's slab, dequantized.  gate/up reshape to (ff, n_embd), down to (n_embd, ff)."""
        if e in self._cache:
            return self._cache[e]
        out = []
        for nm in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            g, t, mm, _ = self._m.where[f"blk.{self.layer}.{nm}"]
            raw = np.asarray(self._m.bytes(f"blk.{self.layer}.{nm}"))
            per = raw.size // N_EXPERT
            slab = raw[e * per:(e + 1) * per]
            vals = self._quants.dequantize(slab, self._Q[t.type_name])
            out.append(np.asarray(vals, dtype=np.float32).reshape(int(t.shape[1]), int(t.shape[0])))
        self._cache[e] = tuple(out)
        return self._cache[e]


# ---------------------------------------------------------------------------------------------------
# the site
# ---------------------------------------------------------------------------------------------------

def moe_forward(m, x: np.ndarray):
    """One token through one MoE site: route, weighted experts, + shared expert."""
    topk, weights, probs, logits, sel = router(m.w_router, m.b, x)
    x64 = x.astype(np.float64)
    acc = np.zeros(N_EMBD, dtype=np.float64)
    experts = {}
    for i, e in enumerate(topk.tolist()):
        wg, wu, wd = m.expert(e)
        experts[e] = (wg, wu, wd)
        acc += float(weights[i]) * expert_ffn(wg, wu, wd, x)
    sg, su, sd = m.shexp
    shexp = expert_ffn(sg, su, sd, x)
    return {"ids": topk, "weights": weights, "probs": probs, "logits": logits, "sel": sel,
            "moe": acc, "shexp": shexp, "out": acc + shexp, "experts": experts}


def alt_mode_ids(logits: np.ndarray, b: np.ndarray):
    """What Kolibri's SIGMOID_LOGIT_ADD (bias-then-sigmoid) would select: sigmoid(logits + bias).
    Used to prove the fixture can tell the two orders apart."""
    sel = sigmoid(logits.astype(np.float64) + b.astype(np.float64))
    return np.lexsort((np.arange(N_EXPERT), -sel))[:N_USED]


# ---------------------------------------------------------------------------------------------------
# naming an input that makes the gate have teeth
# ---------------------------------------------------------------------------------------------------

def case_quality(m, x: np.ndarray):
    """(ids, probs, margin, bias_flips, logitadd_flips) for a candidate input.

    margin          selection-score gap between the 4th and 5th expert (a tiny margin would let
                    float32-vs-float64 round-off change the ids, making the gate flaky)
    bias_flips      the top-4 changes when exp_probs_b is dropped -> the bias is exercised
    logitadd_flips  the top-4 changes under bias-then-sigmoid -> the fixture distinguishes the orders
    """
    topk, _, probs, logits, sel = router(m.w_router, m.b, x)
    order = np.lexsort((np.arange(N_EXPERT), -sel))
    margin = float(sel[order[N_USED - 1]] - sel[order[N_USED]])
    no_bias = np.lexsort((np.arange(N_EXPERT), -probs))[:N_USED]
    alt = alt_mode_ids(logits, m.b)
    return topk, probs, margin, (not np.array_equal(topk, no_bias)), (not np.array_equal(topk, alt))


def find_input(m, base_seed: int, want_bias_flip: bool, want_mode_flip: bool, margin_min: float):
    """Search seeds for an input whose selection has a clear margin and exercises the bias / the mode."""
    best = None
    for s in range(base_seed, base_seed + 8000):
        x = np.random.default_rng(s).standard_normal(N_EMBD).astype(np.float32)
        topk, probs, margin, flip_b, flip_m = case_quality(m, x)
        if margin < margin_min:
            continue
        if want_bias_flip and not flip_b:
            continue
        if want_mode_flip and not flip_m:
            continue
        return s, x, topk, probs, margin, flip_b, flip_m
        # (best would only be needed if we could not satisfy the constraints; we require them)
    return best


def write_fixture(path: pathlib.Path, m, cases: list):
    with open(path, "wb") as fh:
        fh.write(struct.pack("<II", MAGIC, FIXTURE_VERSION))
        fh.write(struct.pack("<iiiiifi", N_EMBD, N_EXPERT, N_USED, N_SHARED, FF_EXP, float(W_SCALE),
                             int(NORM_W)))
        fh.write(struct.pack("<i", len(cases)))
        fh.write(np.ascontiguousarray(m.w_router, np.float32).tobytes())
        fh.write(np.ascontiguousarray(m.b, np.float32).tobytes())
        for t in m.shexp:
            fh.write(np.ascontiguousarray(t, np.float32).tobytes())
        for c in cases:
            fh.write(np.ascontiguousarray(c["x"], np.float32).tobytes())
            for e in c["ids"].tolist():
                for t in c["experts"][e]:
                    fh.write(np.ascontiguousarray(t, np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["ids"], np.int32).tobytes())
            fh.write(np.ascontiguousarray(c["weights"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["probs"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["moe"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["shexp"], np.float32).tobytes())
            fh.write(np.ascontiguousarray(c["out"], np.float32).tobytes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-fixture", help="write the parity fixture here")
    ap.add_argument("--gguf", help="use REAL weights from this GGUF (default: seeded random)")
    ap.add_argument("--layer", type=int, default=1)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--cases", type=int, default=3)
    ap.add_argument("--margin-min", type=float, default=1e-3,
                    help="minimum 4th-vs-5th selection-score gap, so float round-off cannot flip the ids")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    m = GGUFModel(pathlib.Path(a.gguf), a.layer) if a.gguf else RandomModel(a.seed)
    src = f"REAL weights layer {a.layer} of {a.gguf}" if a.gguf else f"seeded random weights (seed {a.seed})"
    print(f"glm47 MoE reference: {src}")
    print(f"  geometry: n_embd {N_EMBD}, {N_EXPERT} experts, top-{N_USED}, ff {FF_EXP}, "
          f"+{N_SHARED} shared, w_scale {W_SCALE}, norm_w {NORM_W}, groups {N_GROUP}")
    print(f"  gating mode: SIGMOID (sigmoid then +exp_probs_b; selection on probs+bias, weights unbiased)")

    # Three inputs: (1) bias-sensitive with a clear margin, (2) a mode discriminator (the two bias
    # orders disagree), (3) a plain margin-clean input.  All three must satisfy the margin.
    specs = [("bias-flip", True, False), ("mode-flip", True, True), ("plain", False, False)]
    cases = []
    for i in range(a.cases):
        tag, want_b, want_m = specs[min(i, len(specs) - 1)]
        got = find_input(m, a.seed + i * 100000, want_b, want_m, a.margin_min)
        if got is None:
            print(f"  case {i}: could NOT find a {tag} input with margin >= {a.margin_min:g}")
            return 1
        s, x, topk, probs, margin, flip_b, flip_m = got
        mid = moe_forward(m, x)
        cases.append(mid | {"x": x})
        cases[-1]["experts"] = mid["experts"]
        print(f"  case {i} ({tag}, seed {s}): ids {topk.tolist()}  margin {margin:.3e}  "
              f"bias flips {flip_b}  bias-then-sigmoid flips {flip_m}")

    if a.selftest:
        c = cases[0]
        print(f"  probs   range [{c['probs'].min():.6f}, {c['probs'].max():.6f}]  (sigmoid -> (0,1))")
        print(f"  weights {np.array2string(c['weights'], precision=8)}  sum {c['weights'].sum():.6f} "
              f"(== w_scale {W_SCALE}, i.e. normalised THEN scaled)")
        print(f"  |moe| {np.linalg.norm(c['moe']):.4f}   |shexp| {np.linalg.norm(c['shexp']):.4f}   "
              f"|out| {np.linalg.norm(c['out']):.4f}")

    if a.raw_fixture:
        write_fixture(pathlib.Path(a.raw_fixture), m, cases)
        sz = pathlib.Path(a.raw_fixture).stat().st_size
        print(f"wrote raw fixture {a.raw_fixture}: {len(cases)} cases, {sz/1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
