#!/usr/bin/env python3
"""Bisect the GLM-5.3-Flash port's trunk against the reference activation dump, per tensor.

WHY THIS FILE EXISTS AND WHAT IT COST: an earlier comparison reported "every tensor diverges".  That was
a measurement artifact, not a finding.  The two dumps are written by different code with different axis
conventions:

  * the reference (tools/glm5_ref_dump.cpp) writes the ggml tensor's raw bytes, so the fp32 payload is in
    ggml MEMORY order: index = i0 + ne0*i1 + ne0*ne1*i2 ...  (i0 fastest)
  * the port (tools/glm5_block_driver.py) writes numpy arrays shaped in ggml AXIS order with C-order
    strides, i.e. index = i0*(ne1*ne2) + i1*ne2 + i2  - THE REVERSE.

So the port's payload must be reshaped to ne and FULLY TRANSPOSED, not reshaped, before comparing.  Get
this wrong and every tensor "diverges" with the right magnitudes and a correlation near zero - which is
exactly what a genuinely broken port looks like.  Two turns went into blaming the runtime's read path for
it; the check that settles it is cheap: a tensor whose value is known by construction (the reference's
hc_init is a 4x broadcast; the port's copies must be identical along the hc axis) and the FIRST element,
which the two conventions agree on.

Usage:  glm5_bisect.py [chain of tensor names]
"""
import pathlib
import sys

import numpy as np

REF = pathlib.Path("/home/peb/moredata/glm5-ref-dump5")
PORT = pathlib.Path("/home/peb/moredata/glm5-port-dump")

# the block-0 chain, in graph order; l_out-0 is the block's output
CHAIN = ["hc_init", "hc_attn_pre-0", "attn_norm-0", "attn_output-0", "hc_attn_post-0",
         "hc_ffn_pre-0", "ffn_norm-0", "ffn_out-0", "l_out-0"]


def payload(p: pathlib.Path) -> np.ndarray:
    """fp32 values behind the 20-byte (5 x int32) header both writers use."""
    return np.frombuffer(p.read_bytes()[20:], dtype="<f4")


def to_ggml_order(flat: np.ndarray, ne) -> np.ndarray:
    """The port's payload, re-expressed in ggml memory order."""
    return flat.reshape(ne[:3]).transpose(2, 1, 0).ravel()


def main() -> int:
    meta = {}
    for line in (REF / "dump.tsv").read_text().splitlines():
        f = line.split("\t")
        if len(f) >= 8:
            meta[f[0]] = tuple(int(x) for x in f[1:5])

    chain = sys.argv[1:] or CHAIN
    print(f"{'tensor':<16} {'ne':<18} {'max|d| flat':>12} {'max|d| ggml':>12} {'r':>8}  verdict")
    first = None
    for n in chain:
        if n not in meta or not (PORT / f"{n}.bin").exists():
            print(f"{n:<16} (present in only one dump)")
            continue
        ne = meta[n]
        r, p = payload(REF / f"{n}.bin"), payload(PORT / f"{n}.bin")
        if r.size != p.size:
            print(f"{n:<16} SIZE MISMATCH ref={r.size} port={p.size}")
            continue
        flat = float(np.abs(r - p).max())
        q = to_ggml_order(p, ne)
        d = float(np.abs(r - q).max())
        corr = float(np.corrcoef(r, q)[0, 1])
        scale = float(np.abs(r).max()) or 1.0
        if d / scale < 1e-4:
            verdict = "AGREES"
        elif d / scale < 1e-2:
            verdict = "close"
        else:
            verdict = "DIVERGES"
            if first is None:
                first = n
        print(f"{n:<16} {str(ne):<18} {flat:>12.3e} {d:>12.3e} {corr:>+8.4f}  {verdict}"
              f"  [scale {scale:.3g}]")

    print("\nfirst divergence in the chain:", first if first else "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
