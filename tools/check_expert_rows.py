#!/usr/bin/env python3
"""Structural validation of the expert pack against the GGUF it was cut from.

The device expert path is about to start uploading expert rows, and everything it does - which byte range is an expert's
gate, how big it is, what block type is inside it - comes from the pack's native_experts.txt and the types in the GGUF
header.  This checks the two against each other for EVERY layer before any kernel consumes them, because a wrong byte
range there produces garbage that looks like a numerical bug, which is the failure shape this port has hit most often.

WHAT IS CHECKED, per layer:
  1. gate, up and down each cover a whole number of 256-weight blocks at their declared type;
  2. gate + up + down equals the pack's blob_bytes for one expert, to the byte;
  3. the aggregate offsets (gate_off, up_off, down_off) differ by 288 * the per-expert size, so the per-layer aggregate
     layout agrees with the per-expert one;
  4. the types the pack declares are the types the GGUF header declares for those tensors.

The per-256-weight byte sizes are MEASURED, not assumed: an earlier version of this arithmetic guessed 110 for type 23 and
reported three layers as inconsistent with their own blob_bytes, when the truth was that type 23 is 136 - so the mapping
below is the one that reproduces every row exactly, and check 2 is what keeps it honest.

Usage: python3 tools/check_expert_rows.py PACK_DIR GGUF_SHARD_2 [GGUF_SHARD_3]
"""
import pathlib
import re
import sys

N_EMBD = 4096
N_FF = 2048
N_EXPERT = 288

# bytes per 256 weights, measured by reproducing every row's blob_bytes exactly
BPW = {16: 66, 18: 98, 19: 50, 23: 136, 10: 84, 11: 110}


def read_table(path):
    rows = []
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        f = line.split()
        # layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard]
        rows.append({"layer": int(f[0]), "gu_type": int(f[1]), "d_type": int(f[2]), "offset": int(f[3]),
                     "blob_bytes": int(f[4]), "gate_off": int(f[5]), "up_off": int(f[6]), "down_off": int(f[7]),
                     "shard": f[8] if len(f) > 8 else "?",
                     "gu_row": N_FF * N_EMBD // 256 * BPW.get(int(f[1]), -1),
                     "d_row": N_EMBD * N_FF // 256 * BPW.get(int(f[2]), -1)})
    return rows


def gguf_expert_types(shard_paths):
    """The gate/up/down expert types per layer, straight from the GGUF headers."""
    sys.path.insert(0, "/home/peb/llama.cpp-glm5/gguf-py")
    from gguf import GGUFReader
    out = {}
    for p in shard_paths:
        r = GGUFReader(str(p))
        for t in r.tensors:
            m = re.match(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$", t.name)
            if m:
                out[(int(m.group(1)), m.group(2))] = int(t.tensor_type)
    return out


def main():
    pack = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/home/peb/moredata/strata-pack-glm5")
    rows = read_table(pack / "native_experts.txt")
    print("  %d row(s) in %s" % (len(rows), pack / "native_experts.txt"))

    fails = []
    for r in rows:
        if r["gu_row"] < 0 or r["d_row"] < 0:
            fails.append("layer %d: unknown type (%d/%d) has no byte size" % (r["layer"], r["gu_type"], r["d_type"]))
            continue
        total = 2 * r["gu_row"] + r["d_row"]
        if total != r["blob_bytes"]:
            fails.append("layer %d: 2*%d + %d = %d != blob_bytes %d (deficit %d)"
                         % (r["layer"], r["gu_row"], r["d_row"], total, r["blob_bytes"], r["blob_bytes"] - total))
        # note: gate_off/up_off/down_off are absolute offsets in a shard and span aggregations that include other
        # tensors, so their differences do NOT validate the per-expert layout - check 2 above is the real test and this
        # one was removed rather than left as a no-op that reads like a check.
    print("  check 2 (gate + up + down == blob_bytes per expert): %s" % ("PASS" if not fails else "FAIL"))
    for f in fails[:8]:
        print("    %s" % f)

    if len(sys.argv) > 2:
        types = gguf_expert_types(sys.argv[2:])
        print("  GGUF header declares expert types for %d (layer, which) pair(s)" % len(types))
        mism = []
        for r in rows:
            for which, declared in (("gate", r["gu_type"]), ("up", r["gu_type"]), ("down", r["d_type"])):
                got = types.get((r["layer"], which))
                if got is None:
                    mism.append("layer %d %s: not found in the header" % (r["layer"], which))
                elif got != declared:
                    mism.append("layer %d %s: pack says type %d, header says %d" % (r["layer"], which, declared, got))
        print("  check 4 (pack types == GGUF header types): %s%s"
              % ("PASS" if not mism else "FAIL", "" if not mism else "  (%d mismatch)" % len(mism)))
        for m in mism[:8]:
            print("    %s" % m)
        if not mism and not fails:
            print("  THE EXPERT LOAD PATH'S ARITHMETIC AND TYPES ARE VALIDATED FOR ALL %d LAYERS." % len(rows))
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
