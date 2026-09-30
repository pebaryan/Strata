#!/usr/bin/env python3
"""tools/glm5_pack.py - pack a GLM-5.3-Flash GGUF split into a Strata pack (branch glm5next-port).

Phase 1 of GLM5NEXT-PORT.md.  Reuses iq_pack.py's machinery, because the pack format is
an index plus a small float arena, not a requantizer:

  dense.bin           the FLOAT tensors (F32/F16/BF16) this model has, in index order
  index.txt           one 19-field row per non-expert tensor; quantized tensors are
                      served natively from the source GGUF, so their rows carry shape only
  native_experts.txt  one line per MoE layer pointing at the GGUF's own expert blobs
  manifest.json       the layer table + metadata snapshot + tensor accounting (phase 2 needs it)

What is different from qwen4exp, and therefore handled explicitly here:

  * NOT every block has experts.  GLM-5.3-Flash has a 3-layer dense MLP stem (blocks 0-2)
    and its block `block_count` (45) is the MTP layer, which DOES have experts.  The
    existing expert layout assumes layer 0..n-1 are all MoE and contiguous - that gap is a
    phase-2 engine change, and this packer records the truth in manifest.json meanwhile.
  * Two attention kinds in one model: 34 linear-attention blocks (tensors named `ssm_*`
    plus attn_q/k/v/output) and 11 MLA blocks (attn_q_a/q_b, attn_kv_a_mqa, attn_k_b,
    attn_v_b, indexer.*).  Head count per layer comes from the metadata ARRAY
    `attention.head_count_kv`, not a scalar.
  * Some tensors are 3-D (attn_k_b [256,512,64], attn_v_b, the conv1d [4,1,N]).  The index
    holds two dimensions, so a 3-D tensor is folded to (ne0, ne1*ne2) and listed in
    manifest.json's `folded`, since its inner strides must be preserved when read.
  * Routers are F32 like qwen4exp; the engine's router is BF16, so the same policy applies
    (refuse unless the F32 values are exactly BF16, or --compat-bf16 rounds them).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G       # noqa: E402
import iq_pack as P           # noqa: E402

ROLES = ("gate", "up", "down")
ALIGN = P.ALIGN


def layer_table(T, block_count: int) -> list[dict]:
    """One entry per model block: its kind and the geometry phase 2 will need.

    `block_count` is a COUNT (46 blocks, 0..45 here), not a last index: blocks 0-2 are the dense
    MLP stem, 3..44 mix 34 linear-attention and 11 MLA blocks, and the last block carries the
    MTP/NextN projections (which is what identifies it - its index is not a constant).
    """
    table = []
    for b in range(block_count):              # 0 .. block_count-1
        has_experts = f"blk.{b}.ffn_gate_exps.weight" in T
        has_ssm = f"blk.{b}.ssm_a" in T
        has_mla = f"blk.{b}.attn_kv_a_mqa.weight" in T
        has_indexer = f"blk.{b}.indexer.attn_k.weight" in T
        is_mtp = f"blk.{b}.nextn.eh_proj.weight" in T
        if is_mtp:
            kind = "mtp"
        elif has_experts:
            kind = "moe"
        else:
            kind = "dense_stem"
        table.append({"block": b, "kind": kind, "ssm": has_ssm, "mla": has_mla,
                      "indexer": has_indexer, "experts": has_experts, "mtp": is_mtp})
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compat-bf16", action="store_true",
                    help="round the small F32 projections the engine reads as BF16")
    ap.add_argument("--experts-bin", action="store_true",
                    help="write experts.bin (required when a shard boundary falls inside a block: "
                         "v3 native_experts.txt names one shard per layer)")
    ap.add_argument("--dry-run", action="store_true", help="write nothing, report the accounting")
    a = ap.parse_args()

    src = pathlib.Path(a.gguf).absolute()
    out = pathlib.Path(a.out)
    model = P.Model(src)
    T = {n: w[1] for n, w in model.where.items()}
    g = G.GGUFFile(src)
    meta = {k: v for k, v in g.metadata.items()} if hasattr(g, "metadata") else {}

    def m(key, default=None):
        v = meta.get(key, default)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v

    block_count = int(m("glm5next.block_count", 45))
    head_count_kv = m("glm5next.attention.head_count_kv", [])
    table = layer_table(T, block_count)
    routers = [f"blk.{b}.ffn_gate_inp.weight" for b in range(block_count)
               if f"blk.{b}.ffn_gate_inp.weight" in T]
    n_expert = int(T[routers[0]].shape[-1]) if routers else 0

    print(f"source      : {src.name}")
    print(f"shards      : {', '.join(p.name for p in model.paths)}")
    print(f"blocks      : 0..{block_count} ({sum(1 for t in table if t['kind'] == 'moe')} MoE, "
          f"{sum(1 for t in table if t['kind'] == 'dense_stem')} dense stem, "
          f"{sum(1 for t in table if t['kind'] == 'mtp')} MTP)")
    print(f"experts     : {n_expert}/layer,  routed attention: "
          f"{sum(1 for t in table if t['ssm'])} linear (ssm_*), {sum(1 for t in table if t['mla'])} MLA")
    if head_count_kv:
        ones = sum(1 for x in head_count_kv if int(x) > 0)
        print(f"head_count_kv: per-layer ARRAY, {ones} of {len(head_count_kv)} layers carry KV")

    rows, folded, converted = [], [], []
    at = 0
    served = 0
    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
    dense_path = out / "dense.bin"
    fo = None if a.dry_run else open(dense_path, "wb")
    try:
        for name, (gg, t, mm, _shard) in model.where.items():
            if P.is_expert(t.name):
                continue
            nd = len(t.shape)
            if nd > 3:
                print(f"{t.name}: {nd} dimensions; the index holds two after folding ne1*ne2")
                return 1
            ne0 = int(t.shape[0])
            ne1 = int(t.shape[1]) if nd > 1 else 0
            ne2 = int(t.shape[2]) if nd > 2 else 0
            if ne2:
                folded.append({"name": t.name, "shape": [ne0, ne1, ne2], "stored_as": [ne0, ne1 * ne2]})
                ne1 *= ne2
            convert = a.compat_bf16 and P.needs_bf16(t.name, t.type_name) and t.type_name != "BF16"
            if t.type_name in P.FLOAT or convert:
                raw = P.tensor_bytes(mm, gg, t).tobytes()
                if convert:
                    raw = P.bf16_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "4"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw)})
                else:
                    kind = {"BF16": "4", "F16": "5", "F32": "2"}[t.type_name]
                if not convert and t.type_name == "F32" and t.name.endswith(P.ROUTERS):
                    u = np.frombuffer(raw, dtype=np.uint32)
                    if np.count_nonzero(u & 0xFFFF):
                        print(f"router {t.name} is F32 whose values are not BF16; pass --compat-bf16")
                        return 1
                    raw = (u >> 16).astype(np.uint16).tobytes()
                    kind = "4"
                rows.append([t.name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)),
                             str(ne0), str(ne1), "0", "0", "1"] + ["0"] * 7)
                if fo is not None:
                    fo.write(raw)
                    pad = (-len(raw)) % ALIGN
                    fo.write(b"\0" * pad)
                at += len(raw) + (-len(raw)) % ALIGN
            else:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1),
                             "8", "0", "32"] + ["0"] * 7)
    finally:
        if fo is not None:
            fo.close()

    # ---- the experts: one line per MoE block, pointing at the GGUF's own blobs
    layout, offset = [], 0
    for entry in table:
        b = entry["block"]
        if not entry["experts"]:
            continue
        ts = [T[f"blk.{b}.ffn_{r}_exps.weight"] for r in ROLES]
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            print(f"block {b}: gate and up differ in type or per-expert size")
            return 1
        blob = per[0] + per[1] + per[2]
        layout.append((b, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * n_expert

    if not a.dry_run:
        out.mkdir(parents=True, exist_ok=True)
        P.write_index(out, rows, src, served, 0)
        # the tokenizer the server needs, exported from the GGUF the same way iq_pack does it
        tok = HERE / "strata_tokenizer.py"
        if tok.is_file() and not (out / "tokenizer" / "vocab.json").exists():
            import subprocess
            rc = subprocess.run([sys.executable, str(tok), "--gguf", str(src), "--out", str(out)],
                                capture_output=True, text=True)
            print("tokenizer   : " + ("exported" if rc.returncode == 0 else
                                      f"export FAILED ({(rc.stderr or rc.stdout).strip()[:120]})"))
        elif not (out / "tokenizer" / "vocab.json").exists():
            print("tokenizer   : tools/strata_tokenizer.py is absent; the pack has no tokenizer/ yet")
        tmp = out / "native_experts.txt.tmp"
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                     "[shard] (n_expert %d, total %d; absolute offsets in %s, or in the named shard beside it)\n"
                     % (n_expert, offset, src.name))
            for b, gt, dt, off, blob, ts in layout:
                ws = [model.where[t.name] for t in ts]
                shards = {w[3].name for w in ws}
                if len(shards) != 1 and not a.experts_bin:
                    # v3 native_experts.txt names ONE shard per layer, so a split whose boundary falls
                    # inside a layer (gate/up in one file, down in another) cannot be expressed.  With
                    # experts.bin the blobs are self-contained and the GGUF offsets are not read.
                    print(f"block {b}: gate/up/down are in different shards ({', '.join(sorted(shards))}); "
                          f"this pack needs --experts-bin")
                    tmp.unlink()
                    return 1
                gg = ws[0][0]
                shard = model.where[ts[0].name][3]
                fh.write("%d %d %d %d %d %d %d %d" % (b, gt, dt, off, blob,
                                                      *[gg.data_start + t.offset for t in ts])
                         + ("" if shard == src else " " + shard.name) + "\n")
        tmp.replace(out / "native_experts.txt")

    if a.experts_bin and not a.dry_run:
        # The blob layout expert_source.cpp expects: per layer, n_expert rows of [gate | up | down].
        path = out / "experts.bin"
        if path.exists() and path.stat().st_size == offset:
            print("experts.bin : exists with the right size; not rewritten")
        else:
            with open(path, "wb") as fh:
                for b, gt, dt, off, blob, ts in layout:
                    parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
                    chunk = np.concatenate(parts, axis=1)
                    assert chunk.shape == (n_expert, blob), (chunk.shape, blob)
                    fh.write(chunk.tobytes())
                    print(f"  block {b:2d}  blob {blob:9d}  at {off / 2**30:6.2f} GiB", flush=True)
            print(f"experts.bin : {len(layout)} blocks, {offset / 1e9:.2f} GB")

    manifest = {
        "packer": "tools/glm5_pack.py (branch glm5next-port)",
        "source": {"shard1": str(src), "shards": [str(p) for p in model.paths]},
        "architecture": "glm5next",
        "geometry": {
            "block_count": block_count,
            "n_expert": n_expert,
            # the expert geometry: the engine validates every layer against this pair, and its compiled
            # defaults are the qwen4exp ones (2560 / 640), so a GLM pack must state its own.
            "hidden": int(m("glm5next.embedding_length", 4096)),
            "expert_ffn": int(m("glm5next.expert_feed_forward_length", 2048)),
            "head_count_kv": head_count_kv,
            "kv_lora_rank": m("glm5next.attention.kv_lora_rank"),
            "key_length_mla": m("glm5next.attention.key_length_mla"),
            "indexer": {"head_count": m("glm5next.attention.indexer.head_count"),
                        "key_length": m("glm5next.attention.indexer.key_length"),
                        "kpool": m("glm5next.attention.indexer.kpool"),
                        "top_k": m("glm5next.attention.indexer.top_k")},
        },
        "layers": table,
        "index_rows": len(rows),
        "native_only_rows": served,
        "dense_bin_bytes": at,
        "expert_bytes": offset,
        "folded": folded,
        "compat_bf16": converted,
        "note": ("blocks 0-2 carry no experts; native_experts.txt therefore starts at block 3, "
                 "which the engine's contiguous-layer assumption cannot express yet (phase 2)."),
    }
    if not a.dry_run:
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")

    print(f"index       : {len(rows)} rows ({served} served natively from the GGUF, "
          f"{len(rows) - served} in dense.bin)")
    print(f"dense.bin   : {at / 1e6:.1f} MB" + (" (dry run)" if a.dry_run else ""))
    print(f"experts     : {len(layout)} MoE blocks, {offset / 1e9:.2f} GB pointed at in the source GGUF")
    print(f"folded 3-D  : {len(folded)} tensors; compat-bf16: {len(converted)}")
    if not a.dry_run:
        print(f"wrote       : {out}/index.txt, native_experts.txt, dense.bin, manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
