#!/usr/bin/env python3
"""IQ1_S reference oracle for the Strata GLM-5.3-Flash port.

Transcribed from llama.cpp's own CUDA path (ggml-cuda/vecdotq.cuh:1244, vec_dot_iq1_s_q8_1) and its CPU decode
(ggml-quants.c:2650, dequantize_row_iq1_s).  It exists because bit-exactness for the kernel has to be defined against
THIS function - an integer DP4A accumulation over packed nibbles - and not against a dequantise-then-dot, which would
agree only to within rounding and would therefore pass a subtly wrong kernel.

Two representations of the same 2048-entry grid are involved and the script checks that they agree before using either:
the CPU table is uint64 (8 signed bytes, the decode's grid[0..7]) and the device table is uint32 with the same values
packed as four bytes of two 4-bit nibbles.  Note the stored nibble is VALUE + 1, because the device path folds the
"+1" into delta = -1 + IQ1S_DELTA - ... and relies on ds.y (the sum of the quantised activations) to carry it; that is
the reason the two tables are not byte-identical and the reason this check is worth doing rather than assuming.

Usage:
    python3 tools/iq1s_reference.py            # tables + self-checks + a synthetic block
    python3 tools/iq1s_reference.py PACK BLOB  # also a real 50-byte block from a pack expert blob
"""
import re
import struct
import sys

import numpy as np

REF_COMMON = "/home/peb/llama.cpp-glm5/ggml/src/ggml-common.h"
QK_K = 256
IQ1S_DELTA = 0.125


def extract_table(text, name):
    """Pull a GGML_TABLE_BEGIN(...) table out by name, in order."""
    lines = text.split("\n")
    start = next(i for i, l in enumerate(lines) if ("GGML_TABLE_BEGIN" in l and name in l))
    end = next(i for i in range(start, len(lines)) if "GGML_TABLE_END" in lines[i])
    body = "\n".join(lines[start:end + 1])
    vals = [int(v, 16) if v.lower().startswith("0x") else int(v)
            for v in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", body)]
    if len(vals) > 2048:                      # the macro's own argument list contributes numbers
        vals = vals[-2048:]
    assert len(vals) == 2048, "%s: got %d entries" % (name, len(vals))
    return vals


class Iq1s:
    """Both tables, plus the cross-check that makes them interchangeable."""

    def __init__(self, common_path=REF_COMMON):
        text = open(common_path, errors="ignore").read()
        self.cpu = extract_table(text, "iq1s_grid")           # uint64: 8 signed bytes per entry
        self.gpu = extract_table(text, "iq1s_grid_gpu")        # uint32: 4 bytes of nibbles per entry
        # CPU form: 8 signed bytes, little-endian inside the uint64
        self.cpu_bytes = np.array([[(v >> (8 * j)) & 0xFF for j in range(8)] for v in self.cpu], dtype=np.uint8)
        self.cpu_int8 = self.cpu_bytes.view(np.int8)
        # device form: four bytes, each holding two 4-bit values
        gb = np.array([[(v >> (8 * j)) & 0xFF for j in range(4)] for v in self.gpu], dtype=np.uint8)
        self.gpu_bytes = gb
        self.gpu_lo = (gb & 0x0F).astype(np.int16)            # grid0 in the CUDA kernel
        self.gpu_hi = ((gb >> 4) & 0x0F).astype(np.int16)     # grid1

    def check(self):
        """The two representations must decode to the same eight values, differing only by the +1 the device folds in.

        THE PERMUTATION IS THE IDENTITY - lo[0..3] then hi[0..3] - and that is a measured fact, not an assumption: an
        earlier version of this function built the device sequence interleaved (lo0,hi0,lo1,hi1,...) and therefore
        reported 7,830 mismatches out of 16,384, which said the tables disagreed when in fact the CHECK was wrong.  The
        instrument's own refusal to continue is what stopped a kernel being built on an unverified ordering, so the
        refusal stays even though the fix was one line.
        """
        lo = self.gpu_lo.reshape(2048, 4)
        hi = self.gpu_hi.reshape(2048, 4)
        dev_seq = np.hstack([lo, hi]).astype(np.int16)     # low nibbles of the four words, then the high nibbles
        cpu_plus1 = self.cpu_int8.astype(np.int16) + 1
        ok_direct = bool((cpu_plus1 == dev_seq).all())
        return ok_direct, cpu_plus1, dev_seq


def unpack_block(blk):
    """One 50-byte block_iq1_s -> (d, qs[32], qh[8] as uint16)."""
    assert len(blk) == 50, "block_iq1_s is 50 bytes for QK_K=256, got %d" % len(blk)
    d = np.frombuffer(blk[0:2], dtype=np.float16)[0]
    qs = np.frombuffer(blk[2:34], dtype=np.uint8)
    qh = np.frombuffer(blk[34:50], dtype=np.uint16)
    return float(d), qs, qh


def dot_int(blk, q8):
    """The device path, transcribed: integer accumulation over packed nibbles.

    q8 must be a per-32-group quantisation: for each of the eight groups, (d, s, int8 values[32]).
    Returns the eight group results (the kernel's per-sub-block outputs).
    """
    d, qs, qh = unpack_block(blk)
    out = []
    for iqs in range(8):
        w32 = qs[4 * iqs:4 * iqs + 4]                      # four qs bytes -> eight 8-weight sub-blocks
        qhv = int(qh[iqs])
        q8d, q8s, q8v = q8[iqs]
        sumi = 0
        for w in range(4):                                  # four words of two nibble-halves
            idx = int(w32[w]) | (((qhv >> 3 * w) & 0x07) << 8)
            g = int(grid.gpu[idx])
            grid0 = g & 0x0F0F0F0F
            grid1 = (g >> 4) & 0x0F0F0F0F
            u0 = int.from_bytes(q8v[8 * w:8 * w + 4].tobytes(), "little")
            u1 = int.from_bytes(q8v[8 * w + 4:8 * w + 8].tobytes(), "little")
            sumi += _dp4a(grid0, u0) + _dp4a(grid1, u1)
        d1q = d * (((qhv >> 11) & 0x0E) + 1)
        delta = -1.0 + IQ1S_DELTA - (qhv & 0x8000) * (2.0 * IQ1S_DELTA / 0x8000)
        out.append(d1q * (q8d * sumi + q8s * delta))
    return np.array(out)


def _dp4a(a, b):
    """Four int8 products summed, exactly as the DP4A instruction does."""
    s = 0
    for k in range(4):
        va = (a >> (8 * k)) & 0xFF
        vb = (b >> (8 * k)) & 0xFF
        if va > 127: va -= 256
        if vb > 127: vb -= 256
        s += va * vb
    return s


def dot_float(blk, q8):
    """The decode path, for an independent sanity comparison: dequantise, then a plain float dot."""
    d, qs, qh = unpack_block(blk)
    out = []
    for iqs in range(8):
        qhv = int(qh[iqs])
        dl = d * (2 * ((qhv >> 12) & 7) + 1)
        delta = -IQ1S_DELTA if (qhv & 0x8000) else IQ1S_DELTA
        w = np.empty(32, dtype=np.float64)
        for l in range(4):
            idx = int(qs[4 * iqs + l]) | (((qhv >> 3 * l) & 7) << 8)
            g = grid.cpu_int8[idx].astype(np.float64)
            w[8 * l:8 * l + 8] = dl * (g + delta)
        q8d, q8s, q8v = q8[iqs]
        out.append(float(np.dot(w, q8v.astype(np.float64)) * q8d))
    return np.array(out)


def quantise_q8(x):
    """A per-32-group q8_1 quantisation: (d, s, int8[32]) per group, as block_q8_1 holds.

    s IS d * sum(q), NOT THE RAW INTEGER SUM - that is the convention llama.cpp's block_q8_1 uses, and the device
    formula depends on it: d1q * (ds.x*sumi + ds.y*delta) only reduces to the decode path's d1q * ds.x * sum((g +/-
    0.125) * u) when ds.y = ds.x * sum(u).  With s as the bare integer sum the integer path came out ~1e4 against the
    float path's ~1e1, a pure scale factor, which is what a wrong convention looks like and not what a wrong kernel
    transpose would look like.
    """
    groups = []
    for g in range(len(x) // 32):
        v = x[32 * g:32 * g + 32]
        amax = float(np.abs(v).max())
        d = amax / 127.0 if amax > 0 else 1.0
        q = np.clip(np.round(v / d), -127, 127).astype(np.int8)
        groups.append((float(d), float(d) * float(q.astype(np.int32).sum()), q))
    return groups


def main():
    ok, cpu_plus1, dev = None, None, None
    ok, cpu_plus1, dev = grid.check()
    print("  grid: %d entries" % len(grid.cpu))
    print("  CPU table values in %s" % sorted(set(grid.cpu_int8.flatten().tolist())))
    mismatch = int((cpu_plus1 != dev).sum())
    print("  cross-representation check (CPU int8 + 1 == device unpacked): %s  (%d mismatching of %d)"
          % ("AGREE" if ok else "DISAGREE", mismatch, cpu_plus1.size))
    if not ok:
        bad = np.argwhere(cpu_plus1 != dev)[:5]
        for b in bad:
            print("    entry %d: cpu+1 %s vs device %s" % (b[0], cpu_plus1[b[0]].tolist(), dev[b[0]].tolist()))
        print("  REFUSING to continue: the two representations disagree, so a kernel and an oracle built on them would")
        print("  not be comparable, and any parity result would be about that disagreement rather than about IQ1_S.")
        return 2

    rng = np.random.default_rng(12345)
    blk = bytes(rng.integers(0, 256, 50, dtype=np.uint8))
    x = rng.standard_normal(QK_K).astype(np.float32)
    q8 = quantise_q8(x)
    di, df = dot_int(blk, q8), dot_float(blk, q8)
    print("  synthetic block, integer path vs float decode path:")
    print("    integer: %s" % np.array2string(di, precision=5, max_line_width=200))
    print("    float  : %s" % np.array2string(df, precision=5, max_line_width=200))
    rel = np.abs(di - df) / np.maximum(np.abs(df), 1e-9)
    print("    max relative difference %.3e  (these are different arithmetics, so they should be CLOSE, not equal)"
          % rel.max())

    if len(sys.argv) >= 3:
        pack, blob = sys.argv[1], sys.argv[2]
        with open(blob, "rb") as f:
            head = f.read(64)
        print("  pack blob %s: first 16 bytes %s" % (blob, head[:16].hex()))
        real = head[:50]
        d, qs, qh = unpack_block(real)
        print("    first block d=%g qs[0:4]=%s qh[0]=0x%04x" % (d, qs[:4].tolist(), int(qh[0])))
        di, df = dot_int(real, q8), dot_float(real, q8)
        rel = np.abs(di - df) / np.maximum(np.abs(df), 1e-9)
        print("    integer %s" % np.array2string(di, precision=5, max_line_width=200))
        print("    float   %s" % np.array2string(df, precision=5, max_line_width=200))
        print("    max relative difference %.3e" % rel.max())
    return 0


grid = Iq1s()
if __name__ == "__main__":
    sys.exit(main())
