#!/usr/bin/env python3
"""tools/glm47_pack.py - pack a GLM-4.7-Flash GGUF into a Strata pack (branch glm47-port).

Phase 1 of GLM47-PORT.md.  Reuses iq_pack.py's machinery (the pack format is an index plus a
small float arena, not a requantizer) and follows glm5_pack.py's structure:

  dense.bin           the FLOAT tensors (F32/F16/BF16) this model has, in index order
  index.txt           one 19-field row per non-expert tensor; quantized tensors are
                      served natively from the source GGUF, so their rows carry shape only
  native_experts.txt  one line per MoE layer pointing at the GGUF's own expert blobs
  manifest.json       the layer table + metadata snapshot + tensor accounting (phase 2 needs it)

What is different from GLM-5.3 (`glm5_pack.py`), and therefore handled explicitly here:

  * Every non-expert tensor is MLA.  There is no `ssm_*` (KDA), no `indexer.*`, no `ssm_conv1d`,
    and no MTP block: GLM-4.7-Flash (`deepseek2`) is a plain MLA MoE.
  * Exactly one dense stem block (block 0) with a plain `ffn_gate/up/down` (width
    `deepseek2.feed_forward_length`); blocks 1..block_count-1 are MoE.  The expert table
    therefore starts at block 1, not 0 - the same stem gap GLM-5.3 had at blocks 0-2.
  * `attention.head_count_kv` is a SCALAR (1, the MLA latent), not a per-layer array.
  * gate and up are SEPARATE 3-D tensors (`ffn_gate_exps` / `ffn_up_exps`) as in Kolibri, so a
    per-expert blob is [gate rows | up rows | down rows], concatenating two source tensors.
  * Routers are F32 (`ffn_gate_inp`); the engine's router is BF16, so the same policy applies
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
ARCH = "deepseek2"


def layer_table(T, block_count: int) -> list[dict]:
    """One entry per model block: its kind.  Block 0 is the dense MLP stem; every other block is
    MoE.  `block_count` is a COUNT (47 blocks, 0..46 here)."""
    table = []
    for b in range(block_count):
        has_experts = f"blk.{b}.ffn_gate_exps.weight" in T
        has_mla = f"blk.{b}.attn_kv_a_mqa.weight" in T
        table.append({"block": b,
                      "kind": "moe" if has_experts else "dense_stem",
                      "mla": has_mla,
                      "experts": has_experts})
    return table


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's GGUF")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compat-bf16", action="store_true",
                    help="round the small F32 projections the engine reads as BF16")
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

    block_count = int(m(f"{ARCH}.block_count", 47))
    table = layer_table(T, block_count)
    routers = [f"blk.{b}.ffn_gate_inp.weight" for b in range(block_count)
               if f"blk.{b}.ffn_gate_inp.weight" in T]
    n_expert = int(T[routers[0]].shape[-1]) if routers else 0
    n_moe = sum(1 for t in table if t["kind"] == "moe")
    first_expert = min((t["block"] for t in table if t["experts"]), default=-1)

    print(f"source      : {src.name}")
    print(f"shards      : {', '.join(p.name for p in model.paths)}")
    print(f"blocks      : 0..{block_count - 1} ({n_moe} MoE, {block_count - n_moe} dense stem; "
          f"expert table starts at block {first_expert})")
    print(f"experts     : {n_expert}/layer; attention: "
          f"{sum(1 for t in table if t['mla'])} MLA, {sum(1 for t in table if not t['mla'])} other")

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
                folded.append({"name": t.name, "shape": [ne0, ne1, ne2],
                               "stored_as": [ne0, ne1 * ne2]})
                ne1 *= ne2
            convert = a.compat_bf16 and P.needs_bf16(t.name, t.type_name) and t.type_name != "BF16"
            if t.type_name in P.FLOAT or convert:
                raw = P.tensor_bytes(mm, gg, t).tobytes()
                if convert:
                    raw = P.bf16_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "4"
                    converted.append({"name": t.name, "source_type": t.type_name,
                                      "bytes": len(raw)})
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
                    fo.write(b"\0" * ((-len(raw)) % ALIGN))
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
            fh.write("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off "
                     "up_off down_off [shard] (n_expert %d, total %d; absolute offsets in %s, or in "
                     "the named shard beside it)\n" % (n_expert, offset, src.name))
            for b, gt, dt, off, blob, ts in layout:
                ws = [model.where[t.name] for t in ts]
                shards = {w[3].name for w in ws}
                if len(shards) != 1:
                    print(f"block {b}: gate/up/down are in different shards; this single-file "
                          f"artifact should not hit this")
                    tmp.unlink()
                    return 1
                gg = ws[0][0]
                shard = model.where[ts[0].name][3]
                fh.write("%d %d %d %d %d %d %d %d" % (b, gt, dt, off, blob,
                                                      *[gg.data_start + t.offset for t in ts])
                         + ("" if shard == src else " " + shard.name) + "\n")
        tmp.replace(out / "native_experts.txt")

    manifest = {
        "packer": "tools/glm47_pack.py (branch glm47-port)",
        "source": {"shard1": str(src), "shards": [str(p) for p in model.paths]},
        "architecture": ARCH,
        "geometry": {
            "block_count": block_count,
            "n_expert": n_expert,
            "first_expert_block": first_expert,
            # the expert geometry: the engine validates every layer against this pair, and its
            # compiled defaults are the qwen4exp ones (2560 / 640), so a GLM pack must state its own.
            "hidden": int(m(f"{ARCH}.embedding_length", 2048)),
            "expert_ffn": int(m(f"{ARCH}.expert_feed_forward_length", 1536)),
            "head_count": int(m(f"{ARCH}.attention.head_count", 20)),
            "head_count_kv": int(m(f"{ARCH}.attention.head_count_kv", 1)),
            "q_lora_rank": m(f"{ARCH}.attention.q_lora_rank"),
            "kv_lora_rank": m(f"{ARCH}.attention.kv_lora_rank"),
            "key_length_mla": m(f"{ARCH}.attention.key_length_mla"),
            "value_length_mla": m(f"{ARCH}.attention.value_length_mla"),
            "rope_dimension_count": m(f"{ARCH}.rope.dimension_count"),
            "rope_freq_base": m(f"{ARCH}.rope.freq_base"),
            "expert_used_count": m(f"{ARCH}.expert_used_count"),
            "expert_weights_scale": m(f"{ARCH}.expert_weights_scale"),
            "expert_weights_norm": m(f"{ARCH}.expert_weights_norm"),
        },
        "layers": table,
        "index_rows": len(rows),
        "native_only_rows": served,
        "dense_bin_bytes": at,
        "expert_bytes": offset,
        "folded": folded,
        "compat_bf16": converted,
        "note": (f"block 0 carries no experts; native_experts.txt therefore starts at block "
                 f"{first_expert}, which the engine's contiguous-layer assumption cannot express "
                 f"yet (phase 2)."),
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
