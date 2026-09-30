# GLM-5.3-Flash on Strata — port plan (branch `glm5next-port`)

Started 2026-09-30. This branch exists so `main` stays a clean sm_70 port of upstream; nothing here is
merged back until it loads a GLM pack and passes a parity gate.

**Status: planning complete, no kernel work started.** This document plus `tools/glm5_inventory.py` and
`docs/glm5next-tensor-inventory.md` (both generated from the artifact that will actually be served, not
from the model card) are the branch's first commit.

## Verdict first: what porting buys, and what it cannot

The artifact measured here (`unsloth/GLM-5.3-Flash-GGUF`, `UD-IQ1_S`) is **93.1 GB on disk: 86.4 GB of
it is MoE experts (93%), and the entire *novel* architecture — the linear-attention, MLA, indexer, mHC
and MTP tensors together — is ~5.5 GB.** That is the encouraging half: the bulk of the weights is a MoE
the engine already handles count-driven, with IQ quant types it already dequantizes.

The discouraging half is capacity, and no port changes it: 93 GB of weights against this box's **63 GB**
(32 GB VRAM + 31 GB RAM). llama.cpp already runs that exact artifact here and measured **0.16 t/s cold /
0.86 t/s warm**, and Strata's profile-ranked expert cache would fight for the same few GB. To *serve*
this model the requirement is ~**120 GB of fast memory** (a 121 GiB unified-memory box measured 17.6 t/s
on a 109 GB quant with the MTP head working). So the port is worth doing for correctness and quant
research, with a bigger machine as the deployment target — not as a way to serve GLM-5 on this V100.

## What the artifact actually contains (measured)

From the GGUF metadata and every tensor info block (`tools/glm5_inventory.py`, 1,412 tensors):

| role | tensors | approx bytes | dtypes seen |
|---|---:|---:|---|
| MoE experts (288/layer) | 129 | 86.41 GB | IQ1_S, IQ3_XXS, IQ2_XXS, IQ4_XS, Q2_K, Q3_K |
| MLA attention | 232 | 4.31 GB | Q5_K, Q8_0, F32, Q6_K |
| MoE shared expert (the 1) | 129 | 0.80 GB | Q5_K, Q6_K |
| embeddings + head (untied) | 2 | 0.71 GB | Q4_K |
| dense MLP stem (3 layers) | 9 | 0.33 GB | Q5_K, Q6_K |
| MoE routers + `exp_probs_b` bias | 86 | 0.20 GB | F32 |
| linear attention (`ssm_*`) | 374 | 0.14 GB | F32, Q8_0 |
| sparse indexer + compressor | 84 | 0.10 GB | F32, Q8_0 |
| mHC hyper-connections (`hc_*`) | 270 | 0.04 GB | F32, Q8_0 |
| MTP (`nextn.*`) | 4 | 0.04 GB | F32, Q8_0 |

Geometry, read from the metadata keys the engine's guard would have to accept:

- `block_count` 45 (blocks 0-44) **plus block 45**, which is the MTP layer (full MLA + MoE + `nextn.*`)
- `attention.head_count` 64, `attention.key_length_mla` 256, `attention.kv_lora_rank` **512**
- `attention.head_count_kv` is an **array**, not a scalar: `[0,0,0,1,0,0,0,1,…]` — one layer in four
  carries per-token KV (11 MLA layers), the rest are linear attention with no KV at all
- `attention.indexer`: `head_count` 32, `key_length` 128, **`kpool` 4**, **`top_k` 2048**
- vocab 154,880 (text + image + video share it), `token_embd` and `output` are untied

Two layer kinds, by tensor names in the file (not by the paper):

- **Linear-attention layer** (34 of them): `attn_q/k/v` + `attn_output` **plus an `ssm_*` block** —
  `ssm_a`, `ssm_dt.bias`, `ssm_conv1d_q/k/v` (kernel 4), `ssm_f_a/ssm_f_b` and `ssm_g_a/ssm_g_b`
  (low-rank 4096->128->8192 gate pairs), `ssm_norm`, `ssm_beta`. The converter named them after the
  Mamba/SSM convention, which is also how llama.cpp models them.
- **MLA layer** (11 of them, every 4th): `attn_q_a`/`attn_q_a_norm`/`attn_q_b` (low-rank Q),
  `attn_kv_a_mqa`/`attn_kv_a_norm` (the 512-dim latent KV), `attn_k_b`, `attn_v_b`, `attn_output`,
  then `indexer.*` and `indexer_compressor_*` (the kpool-4 compressor).
- **Every layer** carries six `hc_attn_*`/`hc_ffn_*` tensors — the mHC mixer: `hc_fn` (a low-rank map
  from 24 to 16384), `hc_base` (24), `hc_scale` (3).

## What already maps onto Strata

- **MoE + shared expert + router**: present, count-driven since the sm_70 port (`native_router_*`,
  `shared_expert`, `native_moe_combine`). 288 experts and 1 shared is a parameter, not a rewrite.
- **Delta-rule linear attention**: `gdn` (Gated Delta Net) spans 37 files with its own parity tests —
  the same family as this layer's `ssm_*` block (chunked scan, recurrent state, conv1d, gates).
- **Sparse attention with an indexer**: QSA has an indexer, a cell-selection kernel and split-K decode
  over selected cells — the same shape as DSA's lightning indexer, minus the kpool stage.
- **Hyper-connections**: the engine reads `*_hyper_connection.*` tensors for its current model, so the
  multi-stream residual path exists; GLM's mHC is a different parameterization of the same idea.
- **MTP/speculation**: the `--mtp` arm and its runtime format already exist.

## The gap list (each item anchored in the file)

1. **Linear attention whose math matches this model** — adapt `gdn_layer`/`native_gdn*` to GLM's decay
   and gate parameterization (`ssm_a`, `ssm_dt`, the f/g low-rank pairs, conv1d on q/k/v), or write it
   fresh. Needs its own parity test; correctness here decides whether long-context generation works.
2. **MLA** — lowest-rank KV (512) with absorbed per-head `attn_k_b`/`attn_v_b`, noPE, head dim 256.
   Nothing in the engine today has a latent KV: `mla`, `ckv`, `kv_lora`, `low_rank`, `nope` all return
   zero files. This is the largest single kernel job.
3. **The kpool=4 indexer compressor** on top of the existing indexer (top_k 2048 selected cells).
4. **mHC** — the `hc_fn`/`hc_base`/`hc_scale` mixer with its manifold-constrained (Sinkhorn-style)
   normalization, over the existing hyper-connection path.
5. **Top-8 routing** — the router is `native_router_top10` with `k=10` baked in and asserted; GLM routes
   8 of 288 with a bias (`exp_probs_b`).
6. **An architecture identity + geometry** — `gguf_reader.hpp` hard-rejects anything that is not
   `qwen4exp` and asserts a scalar geometry; GLM needs its own arch string, an **array-valued**
   `head_count_kv`, and a layer-type table (34 linear + 11 MLA + 3 dense stems + the MTP block).
7. **The packer** — `tools/iq_pack.py` must learn these tensor names and the per-layer kind, and emit the
   pack layout the loader expects (`dense.bin`, `index.txt`, `native_experts.txt`, `tokenizer/`).
8. **MTP packing** — `nextn.eh_proj`/`enorm`/`hnorm`/`shared_head_norm` into the runtime draft layout.

**Not needed for this artifact: an FP8 path.** The card model was dequantized to IQ/Q_K already, so the
engine's existing dequant table covers every tensor type in the file. FP8 (`e4m3`) only matters if the
*official* `zai-org` checkpoint is ingested, and the V100 has no FP8 hardware regardless.

## Order of work, with the gate that closes each phase

The oracle is the point: llama.cpp's GLM5-Next implementation is **merged and validated**, and this box
already has it built for sm_70 at `/home/peb/llama.cpp-glm5/build-glm5/bin/llama-cli` with the exact
artifact — so every phase can be checked against a known-good implementation instead of re-deriving math
from a paper. That is what makes this an open-ended task rather than an unbounded one.

| phase | work | gate |
|---|---|---|
| 0 | this inventory + tooling | *done* — tensor inventory matches the metadata |
| 1 | packer: names, per-layer kind, pack layout | a GLM pack is produced and its index matches the inventory |
| 2 | arch guard + geometry (array `head_count_kv`, layer kinds) | engine loads the pack headers and reports the layer table |
| 3 | mHC mixer | logits match the oracle for a 2-layer slice |
| 4 | MLA + kpool indexer | attention outputs match the oracle, prefill and decode |
| 5 | linear-attention layer (`ssm_*`) | same, including a long-context check that the recurrent state is right |
| 6 | router top-8 + `exp_probs_b` | expert ids match the oracle exactly on fixed prompts |
| 7 | MTP arm | draft acceptance in the project's own range (~0.7), not ~1.0 |
| 8 | end-to-end | greedy tokens identical to the oracle on a fixed prompt set |

Phases 3-5 are independent of each other and can be validated in isolation with a synthetic two-layer
pack, which is how the engine's existing parity tests are already structured.

## What this branch does not do

It does not make GLM-5 servable on this box. The capacity arithmetic above is the whole answer to that
question, and it is a hardware fact, not an engineering one: 93 GB of weights, 63 GB of memory.
