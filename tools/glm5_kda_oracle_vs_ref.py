#!/usr/bin/env python3
"""Is the KDA gate formula right, or only the C++ kernel?  Reference data decides.

The port's first error is kda_gate-0 (2.8e+00 of a 4.95 scale, r +0.9856) and kda_beta-0.  Both are
computed from the attention input, which the dump proves correct (attn_norm-0 agrees to 1.1e-04).  So the
fault is in the parameter path or the formula, and this script separates the two:

  * the numpy oracle reproduces the reference's kda_gate-0 / kda_beta-0  -> the FORMULA is right and the
    port's C++ kernel is wrong;
  * the oracle disagrees too                                            -> the formula or the weight
    mapping is wrong, and both sides need fixing.

GUARD FIRST, deliberately: the oracle's own internal rms_norm(hc_attn_pre-0) must reproduce the reference's
attn_norm-0 before any gate number is believed.  That is a free end-to-end check of the oracle's input path
and axis convention against data already in the dump - the specific defence against repeating the
transpose error that invalidated parts 3-5.

Axis conventions, stated once so they cannot bite again:
  dump -> ggml memory order (i0 fastest), so reshape ne[:3] then transpose(2,1,0) gives numpy axis order
  oracle -> [tokens, dim] arrays; g is (T, nh, hd); beta is (T, nh)
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, "/home/peb/code/Strata/tools")
import glm5_kda_reference as K

REF = pathlib.Path("/home/peb/moredata/glm5-ref-dump5")
SHARD = pathlib.Path("/home/peb/moredata/models/glm-5.3-flash-gguf/"
                     "GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf")
T = 5


def dump(name: str) -> np.ndarray:
    """A dumped tensor as a numpy array in the ORACLE's convention.

    The payload is ggml memory order (i0 fastest), so `reshape(*reversed(ne))` is what lines the axes up
    with a [tokens, ..., i0] array - the same mapping the guard verifies rather than assumes.
    """
    ne = None
    for line in (REF / "dump.tsv").read_text().splitlines():
        f = line.split("\t")
        if f[0] == name:
            ne = tuple(int(x) for x in f[1:5])
            break
    raw = (REF / f"{name}.bin").read_bytes()[20:]
    flat = np.frombuffer(raw, dtype="<f4")
    return flat[: int(np.prod(ne))].reshape(*reversed(ne)).squeeze()


def report(tag: str, got: np.ndarray, ref: np.ndarray) -> bool:
    d = float(np.abs(got - ref).max())
    scale = float(np.abs(ref).max()) or 1.0
    r = float(np.corrcoef(got.ravel(), ref.ravel())[0, 1])
    rel = d / scale
    ok = rel < 1e-4
    near = rel < 2e-2          # the gate is a sharply nonlinear function of h, so ~1% is rounding,
    print(f"  {tag:<26} max|d| {d:.3e}  (scale {scale:.3g})  rel {rel:.2e}  r {r:+.4f}   "
          f"{'MATCHES' if ok else ('matches to ~1%' if near else 'DIFFERS')}")
    return near


m = K.Model(SHARD)
w = K.load_weights(m, 0)

# the oracle's input is the RESIDUAL; its internal norm must land on the dump's attn_norm-0
x = dump("hc_attn_pre-0").reshape(T, -1)
print(f"input hc_attn_pre-0: {x.shape}, scale {np.abs(x).max():.3g}")
print("\nGUARD - the oracle's internal norm against the reference's own attn_norm-0:")
xn = K.rms_norm(x, w["attn_norm"])
guard = report("rms_norm(hc_attn_pre-0)", xn, dump("attn_norm-0").reshape(T, -1))
if not guard:
    print("\nGUARD FAILED: the oracle's input path disagrees with the reference, so a gate comparison")
    print("would be meaningless.  Fix the input convention first.")
    raise SystemExit(1)

print("\nTHE GATE (the port's first error):")
g = np.stack([K.kda_gate(w, xn[t]) for t in range(T)])          # (T, nh, hd)
g_ok = report("oracle kda_gate", g, dump("kda_gate-0"))
print("\nBETA:")
beta = 1.0 / (1.0 + np.exp(-(xn @ w["ssm_beta"].T)))            # (T, nh)
b_ok = report("oracle kda_beta", beta, dump("kda_beta-0"))

print()
if g_ok and b_ok:
    print("VERDICT: the oracle reproduces the reference on both.  The KDA formula is right and the")
    print("PORT'S C++ KERNEL (or its weight mapping) is wrong - the fix belongs in src/kernels/glm_kda.*")
else:
    print("VERDICT: the oracle disagrees with the reference too, so my formula or weight mapping is")
    print("wrong - oracle and kernel need fixing together.  This is the more serious of the two.")
