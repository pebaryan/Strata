"""Is the pack's blob for (layer 3, expert E) the GGUF's expert E?

The engine has never been asked this.  glm_moe_parity passed for layer 3, but it fed the stage blobs from a RAW
FIXTURE built by glm5_moe_reference.py - not through FileExpertSource - so the pack's own bytes have never been
compared with the artifact's.  The observed signature (a mixture of the right magnitude drawn from the wrong
members, with the routing, input, norms, hc sites and MLA branch all independently verified) is exactly what
serving the wrong expert data looks like, so this is the test that settles it.

The layout is pinned by arithmetic rather than assumed:
    IQ2_XXS is 66 bytes per 256 weights; a 2048x4096 expert is 8,388,608 weights = 32,768 blocks = 2,162,688 bytes
    IQ3_XXS is 98 bytes per 256 weights; the down matrix is the same weight count = 3,211,264 bytes
    2,162,688 (gate) + 2,162,688 (up) + 3,211,264 (down) = 7,536,640  ==  the pack's declared blob_bytes, exactly
so the per-expert stride is provably right and only the CONTENT is in question.  Neither the pack nor the GGUF
breaks a 256-weight block across a row boundary (4096 is a multiple of 256), so the byte streams are directly
comparable.

usage: check_blob_bytes.py <pack-dir> <gguf-shard1> <layer> <expert> [more experts...]
"""
import os
import struct
import sys

GGUF_PY = "/home/peb/llama.cpp-glm5/gguf-py"

GU_BYTES = 2162688      # one 2048x4096 matrix in IQ2_XXS
D_BYTES = 3211264       # one 4096x2048 matrix in IQ3_XXS
BLOB_BYTES = GU_BYTES + GU_BYTES + D_BYTES


def pack_blob(expert, layer_base, n_expert):
    """The bytes FileExpertSource serves: experts.bin at the layer's base plus expert * blob_bytes."""
    path = os.path.join(os.path.dirname(pack), "experts.bin")
    off = layer_base + expert * BLOB_BYTES
    with open(path, "rb") as f:
        f.seek(off)
        return f.read(BLOB_BYTES), off


def gguf_pieces(path, layer):
    """The artifact's own bytes for the three expert tensors of this layer."""
    sys.path.insert(0, GGUF_PY)
    from gguf import GGUFReader
    r = GGUFReader(path)
    want = {"blk.%d.ffn_gate_exps.weight" % layer: GU_BYTES,
            "blk.%d.ffn_up_exps.weight" % layer: GU_BYTES,
            "blk.%d.ffn_down_exps.weight" % layer: D_BYTES}
    out = {}
    for t in r.tensors:
        if t.name in want:
            raw = t.data.tobytes()
            got = len(raw)
            per = want[t.name]
            n_expert = got // per
            out[t.name] = (raw, per, n_expert, t.tensor_type)
    return out


def main():
    global pack
    if len(sys.argv) < 5:
        print(__doc__)
        return 2
    pack = os.path.abspath(sys.argv[1])
    gguf = sys.argv[2]
    layer = int(sys.argv[3])
    experts = [int(x) for x in sys.argv[4:]]

    # the layer's base offset, straight out of the pack's own row
    layer_base = None
    with open(os.path.join(pack, "native_experts.txt")) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            c = line.split()
            if int(c[0]) == layer:
                layer_base = int(c[3])
                print("  pack row for layer %d: base offset %d, blob_bytes %s" % (layer, layer_base, c[4]))
                break
    if layer_base is None:
        print("  layer %d is not in native_experts.txt" % layer)
        return 1

    pieces = gguf_pieces(gguf, layer)
    if len(pieces) != 3:
        print("  expected three expert tensors in the GGUF, found %d" % len(pieces))
        return 1
    for name, (raw, per, n_expert, tt) in sorted(pieces.items()):
        print("  %-32s %6d bytes/expert  n_expert %d  type %d" % (name, per, n_expert, tt))

    # the GGUF's per-expert bytes, concatenated in the pack's order: gate, up, down
    gname = "blk.%d.ffn_gate_exps.weight" % layer
    uname = "blk.%d.ffn_up_exps.weight" % layer
    dname = "blk.%d.ffn_down_exps.weight" % layer
    n_expert = pieces[gname][2]

    for e in experts:
        blob, off = pack_blob(e, layer_base, n_expert)
        ref = (pieces[gname][0][e * GU_BYTES:(e + 1) * GU_BYTES]
               + pieces[uname][0][e * GU_BYTES:(e + 1) * GU_BYTES]
               + pieces[dname][0][e * D_BYTES:(e + 1) * D_BYTES])
        if len(ref) != BLOB_BYTES:
            print("  expert %d: the GGUF slice is %d bytes, expected %d" % (e, len(ref), BLOB_BYTES))
            continue
        same = (blob == ref)
        # where they first differ, and how much of each third agrees
        first = next((i for i in range(BLOB_BYTES) if blob[i] != ref[i]), None)
        thirds = []
        for lo, hi, nm in ((0, GU_BYTES, "gate"), (GU_BYTES, 2 * GU_BYTES, "up"),
                           (2 * GU_BYTES, BLOB_BYTES, "down")):
            eq = sum(1 for i in range(lo, hi, 101) if blob[i] == ref[i])          # sparse: 1% sample
            thirds.append("%s %d%%" % (nm, eq * 100 // ((hi - lo) // 101 + 1)))
        print("  expert %-4d pack offset %-12d %s   first difference at %s   [%s]"
              % (e, off, "IDENTICAL" if same else "DIFFERENT",
                 "none" if first is None else str(first), ", ".join(thirds)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
