#!/usr/bin/env python3
"""tools/glm5_compare_dump.py - compare a port dump against the reference dump, layer by layer.

The reference dump comes from tools/glm5_ref_dump.cpp, which runs the fork's own GLM5-Next graph with an
eval callback and writes every named tensor.  The port writes the same tensors (5-int32 header + fp32
data).  This reports the FIRST tensor in the reference's own evaluation order whose error exceeds the
tolerance - which is the whole point: "the logits differ" localizes nothing, but "l_out-3 diverges while
l_out-2 matches" points at block 3.

Usage:
  glm5_compare_dump.py --ref /home/peb/moredata/glm5-ref-dump --port /path/to/port-dump
  glm5_compare_dump.py --ref DIR --port DIR --only l_out --tol 1e-2
"""
from __future__ import annotations

import argparse
import pathlib
import struct
import sys

import numpy as np


def load(path: pathlib.Path):
    raw = path.read_bytes()
    if len(raw) < 20:
        return None
    typ, n0, n1, n2, n3 = struct.unpack("<5i", raw[:20])
    shape = tuple(d for d in (n0, n1, n2, n3) if d > 1 or d == n0)
    n = int(np.prod([n0, n1, n2, n3]))
    vals = np.frombuffer(raw[20:20 + 4 * n], dtype="<f4")
    if vals.size != n:
        return None
    return vals.reshape((n0, n1, n2, n3)[: max(1, len(shape))])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--port", required=True)
    ap.add_argument("--only", default=None, help="substring filter on tensor names")
    ap.add_argument("--tol", type=float, default=1e-3, help="relative tolerance on the array's scale")
    ap.add_argument("--top", type=int, default=20, help="how many rows of the summary to print")
    a = ap.parse_args()

    ref_dir, port_dir = pathlib.Path(a.ref), pathlib.Path(a.port)
    man = ref_dir / "dump.tsv"
    if not man.exists():
        print(f"no manifest at {man}; run tools/glm5_ref_dump.cpp first")
        return 2

    rows, first_bad, checked, missing = [], None, 0, 0
    for line in man.read_text().splitlines():
        f = line.split("\t")
        if len(f) < 7:
            continue
        name, tfile = f[0], f[6]
        if a.only and a.only not in name:
            continue
        rp, pp = ref_dir / tfile, port_dir / tfile
        if not rp.exists() or not pp.exists():
            missing += 1
            continue
        r, p = load(rp), load(pp)
        if r is None or p is None or r.shape != p.shape:
            rows.append((name, float("nan"), float("nan"), r.shape if r is not None else None,
                         p.shape if p is not None else None))
            continue
        d = np.abs(r - p)
        scale = max(float(np.abs(r).max()), 1e-30)
        ma = float(d.max())
        rel = ma / scale
        rows.append((name, ma, rel, r.shape, p.shape))
        checked += 1
        if first_bad is None and rel > a.tol:
            first_bad = (name, ma, rel)

    print(f"compared {checked} tensors ({missing} missing from one side), tol {a.tol:g} of each array's scale")
    if first_bad:
        print(f"\nFIRST DIVERGENCE in evaluation order: {first_bad[0]}  "
              f"max abs {first_bad[1]:.4e}  (= {first_bad[2]:.4e} of scale)")
    else:
        print("\nno divergence above tolerance")

    bad = [r for r in rows if r[2] == r[2] and r[2] > a.tol]
    print(f"\n{len(bad)} of {len(rows)} over tolerance; worst {a.top}:")
    for name, ma, rel, rs, ps in sorted([r for r in rows if r[2] == r[2]], key=lambda r: -r[2])[: a.top]:
        print(f"  {name:34s} rel {rel:9.3e}  abs {ma:9.3e}  ref{rs} port{ps}")
    shaped = [r for r in rows if r[2] != r[2]]
    if shaped:
        print(f"\n{len(shaped)} tensors with mismatched shapes:")
        for name, _, _, rs, ps in shaped[:a.top]:
            print(f"  {name:34s} ref {rs}  port {ps}")
    return 0 if not bad and not shaped else 1


if __name__ == "__main__":
    sys.exit(main())
