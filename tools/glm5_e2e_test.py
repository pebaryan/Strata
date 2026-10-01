#!/usr/bin/env python3
"""End-to-end test on the OUTPUT: hidden state -> output projection -> logits -> argmax.

The reference (llama.cpp) is the oracle.  For the prompt "The capital of France is" (tokens
785 6722 315 9621 374) its greedy continuation is 12089 13 1084 374 ("Paris" and friends), and its own
post-trunk hidden state and logits are in the reference dump, so every step is checkable:

    hidden state    vs the dump's result_norm        (post output_norm, last token)
    logits          vs the dump's result_output      argmax, correlation, rank of the reference's argmax
    next token      vs 12089                         the actual product being tested

Usage:  glm5_e2e_test.py [--hidden PATH]
"""
import argparse
import pathlib
import sys

import numpy as np

REF = pathlib.Path("/home/peb/moredata/glm5-ref-dump5")
SHARD = pathlib.Path("/home/peb/moredata/models/glm-5.3-flash-gguf/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf")
GOLDEN = 12089


def ne_of(n):
    for line in (REF / "dump.tsv").read_text().splitlines():
        f = line.split("\t")
        if f[0] == n:
            return tuple(int(x) for x in f[1:5])
    raise KeyError(n)


def dump(n):
    """Reference tensor as a flat array, in the oracle's [tokens, ...] convention."""
    ne = ne_of(n)
    flat = np.frombuffer((REF / f"{n}.bin").read_bytes()[20:], dtype="<f4")
    return flat[: int(np.prod(ne))].reshape(*reversed(ne)).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", default="/home/peb/moredata/glm5-trunk-hidden-fixed.npy")
    a = ap.parse_args()

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import glm5_kda_reference as K

    h = np.load(a.hidden).astype(np.float64).ravel()
    ref_h = dump("result_norm").astype(np.float64).ravel()
    ref_logits = dump("result_output").astype(np.float64).ravel()

    print(f"hidden state : n={h.size}   vs reference: corr {np.corrcoef(h, ref_h)[0,1]:+.5f}   "
          f"max|d| {np.abs(h-ref_h).max():.3e}  (ref scale {np.abs(ref_h).max():.3f})")

    model = K.Model(SHARD)
    W = np.asarray(model.tensor("output.weight"), dtype=np.float64)
    logits = W @ h if W.shape[1] == h.size else h @ W
    assert logits.size == ref_logits.size, f"{logits.size} vs {ref_logits.size}"
    print(f"output.weight: {W.shape} -> logits {logits.size}")

    r_ref_am = int(np.argmax(ref_logits))
    r_am = int(np.argmax(logits))
    rank = int((logits > logits[r_ref_am]).sum()) + 1     # rank of the REFERENCE's argmax in ours
    top5 = set(np.argsort(logits)[-5:]) & set(np.argsort(ref_logits)[-5:])

    print(f"reference    : argmax {r_ref_am}  (golden {GOLDEN})  {'OK' if r_ref_am == GOLDEN else 'MISMATCH'}")
    print(f"ours         : argmax {r_am}   corr {np.corrcoef(logits, ref_logits)[0,1]:+.5f}   "
          f"relative err {np.abs(logits-ref_logits).max()/np.abs(ref_logits).max()*100:.1f}%")
    print(f"the reference's own top token ranks {rank} in our logits; top-5 overlap {len(top5)}/5")
    print()
    print("E2E VERDICT:", "PASS" if r_am == r_ref_am == GOLDEN else
          f"FAIL - our argmax {r_am} != reference {r_ref_am} (golden {GOLDEN})")
    print(f"  our top-5 tokens: {sorted(np.argsort(logits)[-5:].tolist(), key=lambda t: -logits[t])}")
    print(f"  ref top-5 tokens: {sorted(np.argsort(ref_logits)[-5:].tolist(), key=lambda t: -ref_logits[t])}")


if __name__ == "__main__":
    main()
