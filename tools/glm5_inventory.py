#!/usr/bin/env python3
"""Inventory a GLM-5.3-Flash GGUF split without loading tensor data.

Written for the glm5next-port branch: the work list for teaching Strata a new
architecture should come from the file that will actually be served, not from a paper.
Reads only the metadata + tensor-info tables of each shard (kilobytes, not gigabytes).

Usage: glm5_inventory.py <shard1.gguf> [shard2.gguf ...]
"""
from __future__ import annotations

import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

# GGUF metadata value types
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# ggml tensor types -> (name, block elements, bytes per block); enough to size an inventory
GGML_TYPES = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 40), 10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136), 24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8),
    28: ("F64", 1, 8), 29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66), 39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18),
}


def read_str(fh) -> str:
    n, = struct.unpack("<Q", fh.read(8))
    return fh.read(n).decode("utf-8", "replace")


def read_value(fh, t):
    if t == U8: return struct.unpack("<B", fh.read(1))[0]
    if t == I8: return struct.unpack("<b", fh.read(1))[0]
    if t == U16: return struct.unpack("<H", fh.read(2))[0]
    if t == I16: return struct.unpack("<h", fh.read(2))[0]
    if t == U32: return struct.unpack("<I", fh.read(4))[0]
    if t == I32: return struct.unpack("<i", fh.read(4))[0]
    if t == F32: return struct.unpack("<f", fh.read(4))[0]
    if t == BOOL: return struct.unpack("<B", fh.read(1))[0] != 0
    if t == STRING: return read_str(fh)
    if t == U64: return struct.unpack("<Q", fh.read(8))[0]
    if t == I64: return struct.unpack("<q", fh.read(8))[0]
    if t == F64: return struct.unpack("<d", fh.read(8))[0]
    if t == ARRAY:
        et, = struct.unpack("<I", fh.read(4))
        n, = struct.unpack("<Q", fh.read(8))
        return [read_value(fh, et) for _ in range(n)]
    raise ValueError(f"unknown metadata value type {t}")


def read_header(path: Path):
    with open(path, "rb") as fh:
        magic = fh.read(4)
        if magic != b"GGUF":
            raise ValueError(f"{path}: not a GGUF (magic {magic!r})")
        version, = struct.unpack("<I", fh.read(4))
        n_tensors, = struct.unpack("<Q", fh.read(8))
        n_kv, = struct.unpack("<Q", fh.read(8))
        kv = {}
        for _ in range(n_kv):
            key = read_str(fh)
            vtype, = struct.unpack("<I", fh.read(4))
            kv[key] = read_value(fh, vtype)
        tensors = []
        for _ in range(n_tensors):
            name = read_str(fh)
            nd, = struct.unpack("<I", fh.read(4))
            dims = [struct.unpack("<Q", fh.read(8))[0] for _ in range(nd)]
            ttype, = struct.unpack("<I", fh.read(4))
            off, = struct.unpack("<Q", fh.read(8))
            tensors.append((name, dims, ttype, off))
        return version, kv, tensors


def role_of(name: str) -> str:
    """Group a tensor name into the engine work item it implies."""
    parts = name.split(".")
    if len(parts) >= 3 and parts[0] == "blk":
        body = ".".join(parts[2:])
    else:
        body = name
    hints = [
        # order matters: the specific names win over the generic ones ("ffn_gate_exps" is MoE, "ffn_gate" is
        # the dense stem; "gate" alone is the MoE router's input projection, "hc_*" is the mHC mixer)
        ("mtp_nextn", ("nextn", "eh_proj", "enorm", "hnorm", "shared_head")),
        ("hyper_connection", ("hc_attn", "hc_ffn", "hc_norm", "hyper_connection", "input_mix", "block_inject")),
        ("sparse_indexer", ("indexer", "index_compressor", "k_pool", "indexer_compressor")),
        ("linear_attn_ssm", ("ssm_", "dt_bias", "conv1d", "f_a_proj", "f_b_proj", "g_a_proj", "g_b_proj", "kda")),
        ("mla_attention", ("attn_q_a", "attn_q_b", "attn_kv_a", "attn_k_b", "attn_v_b", "attn_q", "attn_k",
                           "attn_v", "attn_output", "attn_output_norm", "q_a_proj", "q_b_proj", "kv_a_proj",
                           "kv_b_proj")),
        ("moe_experts", ("exps",)),
        ("moe_shared_expert", ("shexp", "shared_expert")),
        ("moe_router", ("gate_inp", "exp_probs", "router", "e_score")),
        ("moe_dense_stem", ("ffn_gate", "ffn_up", "ffn_down")),
        ("norm", ("attn_norm", "ffn_norm", "ssm_norm", "norm", "ln_", "layernorm")),
        ("embedding_head", ("token_embd", "output_norm", "output")),
        ("vision", ("v_", "vision", "patch_embed", "mmproj")),
    ]
    low = body.lower()
    for role, needles in hints:
        if any(n in low for n in needles):
            return role
    return "other"


def main() -> int:
    files = [Path(p) for p in sys.argv[1:]]
    if not files:
        print(__doc__)
        return 2

    total = Counter()
    role_bytes = defaultdict(int)
    role_types = defaultdict(Counter)
    layer_roles = defaultdict(Counter)
    kvs = {}
    n_tensors = 0
    print("# tensor inventory\n")
    for f in files:
        version, kv, tensors = read_header(f)
        kvs.update({k: v for k, v in kv.items() if k not in kvs or kvs[k] == v})
        print(f"## {f.name}\n")
        print(f"- GGUF v{version}, {len(tensors)} tensors, {len(kv)} metadata keys")
        per_role = Counter()
        for name, dims, ttype, off in tensors:
            tname, block, tbytes = GGML_TYPES.get(ttype, (f"type{ttype}", 1, 4))
            n = 1
            for d in dims:
                n *= int(d)
            size = n / block * tbytes
            role = role_of(name)
            per_role[role] += 1
            role_bytes[role] += size
            role_types[role][tname] += 1
            total[tname] += 1
            if name.startswith("blk."):
                layer_roles[name.split(".")[1]][role] += 1
            n_tensors += 1
        for role, c in sorted(per_role.items()):
            print(f"  - {role:<20} {c:>5} tensors")
        print()
    print("## totals by role (all shards)\n")
    print("| role | tensors | approx bytes | distinct dtypes |")
    print("|---|---:|---:|---|")
    for role, b in sorted(role_bytes.items(), key=lambda kv: -kv[1]):
        print(f"| {role} | {sum(role_types[role].values())} | {b / 1e9:.2f} GB | "
              f"{', '.join(f'{t}x{c}' for t, c in role_types[role].most_common(6))} |")
    print(f"\n* {n_tensors} tensors total; quant types seen: "
          f"{', '.join(f'{t}x{c}' for t, c in total.most_common())}")
    print("\n## architecture metadata (the guard must know these)\n")
    for k in sorted(kvs):
        if any(s in k.lower() for s in ("architecture", "block_count", "expert", "head", "embedding",
                                        "context", "rope", "vocab", "attention", "nextn", "kda", "index",
                                        "hyper", "pool", "mla", "shared")):
            v = kvs[k]
            s = str(v)
            print(f"- `{k}` = {s[:120]}{'...' if len(s) > 120 else ''}")
    print("\n## per-layer role mix (first 3 and last 2 layers)\n")
    keys = sorted(layer_roles, key=lambda x: int(x) if x.isdigit() else 0)
    for layer in keys[:3] + keys[-2:]:
        print(f"- blk.{layer}: " + ", ".join(f"{r} x{c}" for r, c in sorted(layer_roles[layer].items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
