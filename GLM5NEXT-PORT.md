# GLM-5.3-Flash on Strata — port plan (branch `glm5next-port`)

Started 2026-09-30. This branch exists so `main` stays a clean sm_70 port of upstream; nothing here is
merged back until it loads a GLM pack and passes a parity gate.

**Status: phases 0-7 done** (inventory, packer, arch guard + layer table, expert-table stem + build-time
model geometry, the mHC, the MLA block with its indexer, the KDA linear-attention block, the MoE site with
its router and the leading blocks' dense FFN, and the MTP draft head - each verified against an oracle).
What remains is phase 8: the graph that wires the 46 blocks together and the first end-to-end greedy-token
comparison against the reference.  Nothing is half-finished: every commit on this branch builds and passes
its own gate.

The MTP head's reuse, checked rather than assumed: blk.45's attention tensors are shape-identical to a
trunk MLA block's, so the verified MLA kernel applies unchanged; it has no `hc_*` tensors (plain
residuals); it has no `nextn.embed_tokens` or `nextn.shared_head_head`, so it reuses the model's embedding
table and output projection; and it carries indexer tensors that its graph does NOT use, because the
reference runs the MTP attention densely - a separate MTP context cannot see the trunk's indexer state,
which costs acceptance rate, never correctness.

```
cmake -S . -B build-glm5 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-14 -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DSTRATA_EXPERIMENTAL_SM75=ON -DSTRATA_MODEL_H=4096 -DSTRATA_MODEL_FF=2048
ninja -C build-glm5 strata
./build-glm5/strata-load --layout /home/peb/moredata/strata-pack-glm5 --layout-layers 46
```

That last command is the phase-3 gate: it reports 46 layers, 288 experts/layer, the table starting at
block 3 (the dense stem), 86.41 GB total, and `experts.bin` matching the table byte for byte.


## What the phases found that the plan did not predict

- **`general.architecture` is `glm5next`, and the metadata is self-describing**: `block_count` 46 (a
  COUNT, not a last index), `leading_dense_block_count` 3, `nextn_predict_layers` 1,
  `hyper_connection.count` 4 with `sinkhorn_iterations` 20, `kda.head_dim` 128, `expert_used_count` 8,
  `expert_shared_count` 1, `indexer.kpool` 4 / `top_k` 2048, `kv_lora_rank` 512. The mHC shape I had
  inferred from tensor bytes (4 streams, Sinkhorn-normalised) is stated outright in the header.
- **`rope.dimension_count` is 0**: no rotary embeddings anywhere in this model. The linear-attention
  blocks use conv1d + decay, the MLA blocks are noPE. So the port needs no RoPE path for GLM at all.
- **The reader refused the artifact for a reason that had nothing to do with GLM's architecture**:
  shard 1 holds 72 metadata keys and **zero tensors**, so aligning past its tensor table lands 29 bytes
  beyond EOF. Fixed by clamping when there are no tensors, keeping the strict refusal when there are.
- **A shard boundary falls inside block 25**: gate/up in one file, down in another. v3
  `native_experts.txt` names one shard per layer, so the pack is written with `--experts-bin` instead.
  A v4 with per-tensor shards is the general fix; it is NOT written speculatively, because the current
  reader would take a v4 line as v3 and read the wrong bytes silently.
- **The expert table cannot start at block 0**: blocks 0-2 are the dense stem. `expert_layout_load`
  now checks contiguity from the file's own lowest layer and reports it as `first_layer`, and it takes
  the model's expert geometry (`hidden`, `expert_ffn`) instead of assuming the compiled qwen4exp pair
  (2560 / 640). Verify with `strata-load --layout <pack> --layout-layers 46` (it reads `hidden` and
  `expert_ffn` from the pack's manifest.json when they are not passed).
- **The CPU expert pool's geometry is now a BUILD option** (chosen and implemented, branch
  `glm5next-port`): `-DSTRATA_MODEL_H` / `-DSTRATA_MODEL_FF` (defaults 2560 / 640, the qwen4exp model)
  are configured into a generated `strata/model_geometry.hpp`, which `expert.hpp` uses for `H`/`FF` and
  `native_expert.hpp` uses to size the pooled activation buffers from the geometry itself (the widest
  activation these kernels quantize to is Q8_1-shaped, 1.25 B/element, plus a row of slack, with the
  historical 4096/1024 kept as floors so the default build is unchanged). A GLM build is
  `-DSTRATA_MODEL_H=4096 -DSTRATA_MODEL_FF=2048`. Measured cost of the larger buffers: ~100 KB per thread.
  The alternative — runtime-sized pool buffers — was rejected because it puts indirection into the hottest
  loops and into the AVX-512 kernels' compile-time unrolling, to avoid a second build directory.
  **One model per build**, which is the engine's own design: in a GLM-configured build the qwen4exp pack is
  refused at the first layer (`blob is 2176000 B but its formats make 11141120`).


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

## Reference notes for the phases that remain (read from llama.cpp's GLM5-Next graph)

These are the semantics phase 4 onward must reproduce, taken from the reference implementation rather
than from prose. They are recorded because reading them again costs more than typing them once.

**MLA block (phase 4)** - 11 blocks, nope-only, with the absorption folding:

```
qr      = rms_norm(wq_a @ cur)                 wq_a [4096,1536]      q_lora_rank 1536
q       = wq_b @ qr        -> [256, 64, nt]    wq_b [1536,16384]     64 heads, head dim 256
Qcur    = wk_b^T @ q       -> [512, 64, nt]    wk_b [256,512,64]     the ABSORBED query: in latent space
kv_cmpr = rms_norm(wkv_a_mqa @ cur) -> [512]   wkv_a_mqa [4096,512]  kv_lora_rank 512, ONE head
out     = attn(Qcur, kv_cmpr, kv_cmpr, wv_b, top_k, kq_scale=1/sqrt(256))
```

K and V are the *same* 512-dim latent; the head structure lives in `wk_b` on the query and `wv_b`
[512,256,64] on the output. There is no rope anywhere and no k_pe, so nothing needs a position.
`top_k` comes from the indexer (below); a GGUF without indexer weights passes null and attends the
whole cache.

**The indexer and its 4:1 pool (phase 4)** - per MLA block, using `indexer_head_size` d = 128,
`indexer_n_head` 32, `indexer_block_size` r = 4 (`kpool`), `top_k` 2048:

* `k = LayerNorm(indexer.attn_k @ cur)` using `indexer.k_norm.weight` **and** `.bias`
* `g = indexer_compressor_gate @ cur`, and k and g are packed into ONE cache row ([2d] = 256)
* each pool of r = 4 consecutive keys is combined into one pooled key by a **gate-weighted average over
  the members, channel by channel**: `w = softmax_over_members(g + ape)`, `pooled = sum_m w . k_m`
* `indexer_compressor_ape` [128, 4] is the intra-pool position bias and, with no rope in the model at
  all, the only ordering signal the indexer has
* the query side uses `indexer.attn_q_b` [1536,4096] from `qr`, and `indexer.proj` [4096,32] scores
  the pooled keys; the top `top_k` = 2048 cells are what attention may read

**The indexer's scoring and selection (phase 4b - read from the reference, not yet implemented)**

The pieces that are easy to get subtly wrong, with the reference's own reasoning:

* a pooled key is a **channel-wise gate-weighted average over its r = 4 members**:
  `pooled[channel] = sum_member softmax_over_members(gate + ape)[member][channel] * key[member][channel]`
* the indexer query is `indexer.attn_q_b @ qr` -> `d = 128` per head times `nh = 32` heads per token
  (so 4096 values, from the *q_lora* vector, while the per-head gate below comes from the block input)
* `indexer.proj @ cur` gives `nh` per-head weights, scaled by `1/sqrt(d*nh)`; the reference folds both
  positive scalars into those small weights rather than into the big score tensor, using
  `relu(x*s) == s*relu(x)` for `s > 0` - i.e. the heads are combined as a ReLU-gated weighted sum
* selection is `n_sel = min(n_pool, top_k / r)` = min(n_pool, 512) **whole pools**, then the pools are
  expanded into their members
* **the cut is on whole pools, never on single cells.** Scoring cells with their pool's score and
  cutting there is NOT the same thing: ReLU sends many distinct pools to exactly 0.0 and `ggml_top_k`
  is unordered among equal keys, so cutting on cells splits pools apart. The reference cites PR #27754
  for the diagnosis and the reference-free check (count partly selected pools).
* **the trailing incomplete pool is always selected** (`index_kpool_always_select_tail`): it has no pool
  key and so can never be picked by score, so its cells are appended instead of consuming pool budget.
* the reference chunks the token loop to bound scratch memory (the score tensor is materialised twice,
  `2*n_pool*nh*n_tokens*4 B` per device); no reduction runs across tokens, so chunking is exact - worth
  copying if a device-side implementation ever needs the bound.

**The FFN, per block (phase 6)** - the leading 3 blocks (`leading_dense_block_count`) use a plain
parallel SILU FFN (`ffn_gate`/`ffn_up`/`ffn_down`), every other block uses MoE + a shared expert:

```
build_moe_ffn(cur, ffn_gate_inp, ffn_up_exps, ffn_gate_exps, ffn_down_exps, ffn_exp_probs_b,
              n_expert=288, n_expert_used=8, SILU, expert_weights_norm, expert_weights_scale,
              expert_gating_func=2)
shexp = build_ffn(cur, ffn_up_shexp, ffn_gate_shexp, ffn_down_shexp, ..., SILU, PAR)
```

`ffn_exp_probs_b` is a per-expert routing BIAS on the router logits (the `[288]` F32 tensor), and both
`expert_weights_norm` and `expert_weights_scale` are metadata values that must reach the kernels - the
engine's current router has neither, and it hardcodes k = 10 where GLM needs 8.

**The MTP block (phase 7)**: `blk.45` is a full MLA + MoE block with **no hyper-connections at all**
(unlike every trunk block), plus `nextn.enorm`, `nextn.hnorm`, `nextn.eh_proj` and
`nextn.shared_head_norm`; the trunk's head mean-collapses the streams, rms_norms and projects.

**KDA blocks (phase 5) - and why this phase is SMALLER than the plan assumed**

Reading the reference changed the estimate, so this is worth stating precisely. 34 of the 46 blocks are
linear attention, and llama.cpp builds them through `llm_build_delta_net_base::build_recurrent_attn`
(delta-net-base.cpp), which ends in the fused op `ggml_gated_delta_net`. The CPU reference for that op
(ggml/src/ggml-cpu/ops.cpp) contains this branch:

```cpp
const bool kda = (neg0 == S_v);        // the ONLY structural difference in the recurrence
```

So the recurrence is shared with the GDN that **qwen4exp already uses** - the model this engine serves -
and KDA differs from GDN in exactly one place: the decay is per-channel instead of per-head.

```
decay:   KDA:  S[i][j] *= exp(g[i])     (g has S_v entries per head)
         GDN:  S[i][j] *= exp(g[0])     (one scalar per head)
delta:   delta[j] = (v[j] - sum_i S[i][j]*k[i]) * beta
update:  S[i][j] += k[i] * delta[j]
output:  attn[j]  = (sum_i S[i][j]*q[i]) * scale        (scale = 1/sqrt(head_dim), applied to q)
```

Strata already has `gdn_step(state, q, k, v, gate, ...)` in include/strata/kernels/gdn.hpp with
`dec = exp(gate)` and a gate of shape `(h_v,)` - i.e. the scalar variant. **The kernel work phase 5 needs
is the per-channel decay: a gate of `(h_v, S_v)` and a row-wise elementwise multiply instead of one
scalar scale.** That is 8 lines of math in the reference, not a new recurrence.

What IS new is GLM's front end, and it differs from qwen4exp's in ways that matter:

* **three separate causal convolutions** (q, k, v each with their own `ssm_conv1d_*`, kernel 4), each
  followed by **SiLU** — the activation is there, but applied per-conv. qwen4exp instead convolves
  q|k|v concatenated in ONE conv and applies SiLU once to the result. Either way the conv is depthwise
  over the sequence (`ggml_ssm_conv`: `out[t][ch] = sum_k w[k][ch] * x[t+k][ch]` over the concatenated
  `[state(d_conv-1) | tokens]`, so the kernel is applied in forward order — a correlation, not a
  flipped convolution — and the state holds the previous `d_conv-1` inputs per channel);
* the gate: `g = gate_lower_bound * sigmoid(-(ssm_a * (ssm_f_b(ssm_f_a(x)) + dt_bias)))` with
  `ssm_a` holding `-exp(A_log)`, so the effective factor is `exp(A_log)`; `gate_lower_bound` is -5.0,
  so the decay exponent lives in [-5, 0]. It is per (head_dim, head), which is what makes it KDA;
* `beta = sigmoid(ssm_beta @ x)`, one scalar per head;
* `q` and `k` are L2-normalised with a hard-coded eps of **1e-6** (qwen4exp uses the model's
  `f_norm_rms_eps` instead);
* the output is an RMS norm **gated by a sigmoid** (`ssm_o_norm` then `sigmoid(ssm_g_b(ssm_g_a(x)))`),
  where qwen4exp's path gates with SiLU - a sigmoid-vs-SiLU swap is exactly the kind of detail that
  silently changes generations.




The oracle is the point: llama.cpp's GLM5-Next implementation is **merged and validated**, and this box
already has it built for sm_70 at `/home/peb/llama.cpp-glm5/build-glm5/bin/llama-cli` with the exact
artifact — so every phase can be checked against a known-good implementation instead of re-deriving math
from a paper. That is what makes this an open-ended task rather than an unbounded one.

| phase | work | gate |
|---|---|---|
| 0 | this inventory + tooling | *done* - tensor inventory matches the metadata |
| 1 | packer: names, per-layer kind, pack layout | *done* - the pack's index accounts for all 1412 tensors; 43 expert blocks contiguous; experts.bin matches its header byte for byte |
| 2 | arch guard + geometry (array `head_count_kv`, layer kinds) | *done* - the guard reads the artifact, reports the block table, and refuses 5 distinct tampered copies with precise messages |
| 3 | expert-layout stem offset + build-time model geometry | *done* - a GLM-configured build loads the pack's expert table (46 layers, 288 experts, stem at block 3, experts.bin matching); the default build's constants are unchanged |
| 3b | the mHC mixer | *done* - 8/8 blocks match the numpy oracle (worst 4.5e-06 absolute, i.e. float32 round-off), with the Sinkhorn shown to sit at the oracle's own fixed point |
| 4 | MLA + the kpool indexer | *done* - MLA 4/4 layers (worst 1.2e-05, every intermediate checked) and the indexer 4/4 (cache rows, pooled keys, scores, bias, the selection set, plus the whole-pool and always-select-tail invariants) |
| 5 | the linear-attention block (`ssm_*`, KDA) | *done* - 12 stage checks pass on layers 4 and 20 at 1, 5 and 8 tokens (worst 1.8e-05 relative), covering the single-token decode step as well as prefill; the gate stays in [gate_lower_bound, 0] and &#124;q&#124; comes out 1.0 per head |
| 6 | router top-8, `ffn_exp_probs_b`, `expert_weights_norm/scale`; dense stem FFN; shared expert | *done* - the router on a fixture chosen BECAUSE the selection bias changes the answer (ids exact, weights 1.9e-07, sum(weights) exactly 2.5), the MoE sum / shared expert / site output to 1.5e-05, and the leading blocks' dense FFN (ff 12288, not an expert's 2048) to 1.6e-05 |
| 7 | MTP arm (blk.45, which has no hyper-connections) | *done* - the head's own wiring verified (e/h norms, the concat with e_norm FIRST exactly, eh_proj, the head norm, worst 7.1e-06); the block's MLA/MoE are the trunk's operators, asserted by comparing tensor shapes rather than assumed, and blk.45 confirmed to have 0 hc_* tensors |
| 8 | end-to-end | **ground truth established, graph still to write** - the reference's own GLM5-Next graph now dumps 947 named tensors plus the golden greedy tokens (`tools/glm5_ref_dump.cpp`, manifest in `tools/glm5-ref/`, comparison harness in `tools/glm5_compare_dump.py`), so a mismatch localizes to a layer and a site |

## Phase 8: the ground truth, and what it gives the port

Before writing the trunk graph, the thing that had to exist is a way to compare against the reference
OTHER than its final tokens.  "The logits differ" localizes nothing, and with 46 blocks and four cache
kinds to wire, a single opaque mismatch would mean bisecting by hand.

`tools/glm5_ref_dump.cpp` runs the fork's own GLM5-Next graph with `cb_eval` set and writes every tensor
the graph NAMES - the graph's `cb(...)` calls are the hooks.  A fixed prompt yields 947 named tensors and
the golden greedy generation:

    prompt    "The capital of France is"  ->  785 6722 315 9621 374
    greedy    12089 13 1084 374

and the granularity is what matters:

  * `l_out-N` [4096, 4, 5, 1] for every trunk block - the four hyper-connection streams after block N,
    which is exactly the array shape the verified mHC kernel consumes;
  * `hc_head`, `result_norm`, `result_output` [154880] for the last prompt token;
  * per-SITE tensors inside each block, so a mismatch localizes further than a layer: for a KDA block
    `hc_attn_pre-N`, `kda_gate-N`, `kda_beta-N`, `attn_output-N`, `hc_attn_post-N`, `hc_ffn_pre-N`,
    `ffn_out-N`; for an MLA block `Qcur-N` [512,64,5,1], `kv_cmpr-N` [512,1,5,1], `hc_attn_post-N`,
    `ffn_moe_out-N`.

The dump is ~116 MB and lives in `/home/peb/moredata/glm5-ref-dump/`; what is committed is the manifest
and the golden tokens (`tools/glm5-ref/`) plus the regeneration recipe.
`tools/glm5_compare_dump.py` compares a port dump against it and reports the FIRST tensor in the
reference's own evaluation order that exceeds tolerance - so the port's failures come back as "block 3
diverges while block 2 matched".

Running the reference needs the same attention to memory as the earlier llama.cpp work: the experts stay
on the CPU (one tensor-buft override per layer, the CLI's `-ncmoe`), because 86 GB of experts cannot live
in 32 GB of VRAM.  Note that `token_embd.weight` and 126 other tensors land on the CPU anyway
(`cannot be used with preferred buffer type CUDA_Host`), so the reference run is not fully offloaded
either - worth knowing when comparing timings.

Phases 3-5 are independent of each other and can be validated in isolation with a synthetic two-layer
pack, which is how the engine's existing parity tests are already structured.

## What this branch does not do

It does not make GLM-5 servable on this box. The capacity arithmetic above is the whole answer to that
question, and it is a hardware fact, not an engineering one: 93 GB of weights, 63 GB of memory.
