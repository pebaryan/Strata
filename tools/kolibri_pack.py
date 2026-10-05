#!/usr/bin/env python3
"""tools/kolibri_pack.py - pack Aleph Alpha's Kolibri-1 GGUF into a Strata pack (branch kolibri-port).
Phase 1 of KOLIBRI-PORT.md.  Reuses iq_pack.py's machinery wholesale (the pack format is an index
plus a float arena plus a native-expert table, not a requantizer); what is Kolibri-specific:

  * EVERY block is MoE and contiguous (no GLM dense stem, no MTP block) - the existing expert
    layout's assumption holds as-is.
  * gate and up are SEPARATE 3-D tensors (ffn_gate_exps, ffn_up_exps), as in qwen4exp - the blob
    is still [gate rows | up rows | down rows] per expert, cut the same way.
  * down mixes Q4_K and Q6_K across layers (native_experts.txt's per-layer d_type covers it).
  * Kolibri-only metadata goes to manifest.json for phase 2: the per-layer sliding/full attention
    pattern (attention.sliding_window_pattern, 513-token window; full layers are NoPE), the
    sigmoid_logit_add router (enum 5), expert_weights_norm false, and the sandwich-norm tensor
    names (post_attention_norm / post_ffw_norm).

usage:
  python3 tools/kolibri_pack.py --gguf Kolibri-1-Q4_K_M.gguf --out <pack-dir> [--experts-bin]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import iq_pack  # noqa: E402  (the pack machinery: Model, index, expert layout, writers)

KOLIBRI_ARCH = "kolibri1"


def kolibri_manifest(model, expert_info) -> dict:
    """Kolibri-specific manifest fields phase 2's arch guard and trunk read.  Metadata comes from
    the first shard's GGUFFile.metadata dict (iq_pack's Model exposes it as model.files[0].metadata)."""
    m = model.files[0].metadata
    # llama.cpp writes arch-prefixed keys (kolibri1.attention.*); accept the bare form too
    ARCH = KOLIBRI_ARCH + "."

    def lookup(name):
        return m.get(ARCH + name, m.get(name))

    def need_int(name):
        v = lookup(name)
        if v is None:
            raise SystemExit("the GGUF lacks %s (as '%s%s' or '%s'); not a converted Kolibri 1"
                             % (name, ARCH, name, name))
        return int(v)

    pattern = lookup("attention.sliding_window_pattern")
    if pattern is None:
        raise SystemExit("the GGUF lacks attention.sliding_window_pattern; not a converted Kolibri 1")
    block_count = need_int("block_count")
    if len(pattern) != block_count:
        raise SystemExit("attention.sliding_window_pattern has %d entries for %d blocks"
                         % (len(pattern), block_count))

    gating = need_int("expert_gating_func")
    if gating != 5:
        raise SystemExit("expert_gating_func is %d; Kolibri 1 packs need sigmoid_logit_add (5)" % gating)

    return {
        "architecture": KOLIBRI_ARCH,
        "attention": {
            "swa_window": need_int("attention.sliding_window"),
            "swa_pattern": [bool(b) for b in pattern],   # true = sliding (RoPE, window); false = full (NoPE)
            "full_attention_nope": True,                  # the patch's graph: rope only on sliding layers
            "head_count": need_int("attention.head_count"),
            "head_count_kv": need_int("attention.head_count_kv"),
            "head_dim": need_int("attention.key_length"),
            "rope_freq_base": float(lookup("rope.freq_base") or 10000.0),
            "qk_norm": "per_head_dim_before_rope",
        },
        "norms": {
            "sandwich": True,
            "tensors": ["attn_norm", "post_attention_norm", "ffn_norm", "post_ffw_norm"],
        },
        "moe": {
            "n_expert": need_int("expert_count"),
            "n_used": need_int("expert_used_count"),
            "expert_ffn": need_int("expert_feed_forward_length"),
            "shared_experts": int(lookup("expert_shared_count") or 1),
            "shared_ffn": int(lookup("expert_shared_feed_forward_length") or 0),
            "gating": "sigmoid_logit_add",                # select on logits + bias, weight unbiased sigmoid
            "weights_norm": bool(lookup("expert_weights_norm") or False),
            "router_bias": "blk.N.exp_probs_b.bias",
        },
        "context_length": need_int("context_length"),
        "rms_eps": float(lookup("attention.layer_norm_rms_epsilon") or 1e-6),
        "experts": expert_info,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the first shard of Kolibri-1's GGUF")
    ap.add_argument("--out", required=True, help="the pack directory to write")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also cut experts.bin (the GGUF serves experts natively either way)")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="round F32 tensors that are not exactly BF16 (the routers are F32)")
    args = ap.parse_args()

    src = pathlib.Path(args.gguf)
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "native_experts.txt").unlink(missing_ok=True)   # the completion marker goes back last

    model = iq_pack.Model(src)
    T = {n: w[1] for n, w in model.where.items()}

    # the arch guard, before any work: this packer is for kolibri1 and says so
    arch = model.files[0].metadata.get("general.architecture")  # unprefixed by convention
    if arch != KOLIBRI_ARCH:
        raise SystemExit("%s is '%s', not '%s' (this packer is Kolibri-specific; the tensor layout "
                         "would silently mismatch)" % (src.name, arch, KOLIBRI_ARCH))

    # the experts: iq_pack's expert_layout does everything (per-shard offsets, per-layer types,
    # gate==up type check, v3/v4 header).  Kolibri's names are exactly its qwen4exp names.
    expert = iq_pack.expert_layout(model, src)
    if isinstance(expert, str):
        raise SystemExit("expert layout: " + expert)
    layout, ne_text, n_expert, total = expert
    print("experts: %d layers x %d experts, %.2f GB of blobs" % (len(layout), n_expert, total / 1e9))

    # dense/index: the standalone index path serves every quantized tensor natively from the GGUF
    iq_pack.index_standalone(src, out, model, compat_bf16=args.compat_bf16)

    # write the expert table last-but-one, then the manifest (the completion marker is native_experts.txt;
    # keep iq_pack's convention: it is unlinked at the start and rewritten at the very end)
    (out / "native_experts.txt").write_text(ne_text)

    manifest = kolibri_manifest(model, {
        "n_expert": n_expert,
        "total_bytes": total,
        "cut": bool(args.experts_bin),
        "source": src.name,
    })
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.experts_bin:
        raise SystemExit("--experts-bin: cutting experts.bin is not wired yet; the GGUF serves the "
                         "experts natively (native_experts.txt points at the shards), which is what "
                         "phase 2-6 use.  Extend iq_pack's cut for a single-file source if needed.")

    print("pack: %s (native_experts.txt, index.txt, dense.bin, manifest.json)" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
