"""Validate the regenerated block-3 stage references against the oracle's own outputs.

The stage tensors that glm5_block_driver.py writes were produced by a writer that packed C order under an
ne0-fastest header.  That writer is fixed, so the stage families are regenerated - and the point of this
script is to show the regenerated file agrees with a source that never went through that writer at all:
the oracle's own l_out-3.  A stage reference that disagrees with the oracle is not a reference.
"""
import pathlib
import struct
import sys


def load(p):
    b = pathlib.Path(p).read_bytes()
    ne = struct.unpack("<5I", b[:20])[1:]
    n = 1
    for k in ne:
        n *= max(k, 1)
    return ne, struct.unpack("<%df" % n, b[20:20 + 4 * n])


def main():
    d = pathlib.Path(sys.argv[1])
    oracle = pathlib.Path(sys.argv[2])
    if not d.is_dir():
        print("  no dump at %s - the driver did not run" % d)
        return 1

    for stage in ["l_out-0", "l_out-3", "ffn_out-3", "attn_norm-3", "ffn_norm-3", "hc_attn_pre-3"]:
        p = d / (stage + ".bin")
        if not p.exists():
            print("  %-14s NOT PRODUCED" % stage)
            continue
        ne, v = load(p)
        rms = (sum(float(x) * x for x in v) / len(v)) ** 0.5
        print("  %-14s ne=%-20s %8d floats  rms %.9g" % (stage, str(list(ne)), len(v), rms))

    # The decisive check: block 3's output from the fixed driver must equal the oracle's own l_out-3.
    # The oracle never used this writer, so agreement here is evidence about the writer, not a tautology.
    a = oracle / "l_out-3.bin"
    b = d / "l_out-3.bin"
    if not (a.exists() and b.exists()):
        print("  cannot compare: %s or %s missing" % (a, b))
        return 1
    nea, va = load(a)
    neb, vb = load(b)
    print("  oracle l_out-3 ne=%s (%d floats)" % (list(nea), len(va)))
    print("  driver l_out-3 ne=%s (%d floats)" % (list(neb), len(vb)))
    if len(va) != len(vb):
        print("  DISAGREE: different extents - the containers differ")
        return 1
    worst = max(abs(x - y) for x, y in zip(va, vb))
    verdict = "AGREE" if worst < 1e-9 else "DISAGREE"
    print("  worst |driver - oracle| over l_out-3: %.6g   %s" % (worst, verdict))
    return 0 if verdict == "AGREE" else 1


if __name__ == "__main__":
    sys.exit(main())
