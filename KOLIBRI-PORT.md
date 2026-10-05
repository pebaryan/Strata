# Kolibri-1 on Strata — port plan (branch `kolibri-port`)

Feasibility and phase plan for running Aleph Alpha's Kolibri-1 (78B MoE, 3.46B active) on the Strata
engine with the qwen4exp tricks (native expert rows, the GPU row cache with its CPU tier, resident
budgets, mmap experts). Written 2026-10-05. Branch: `kolibri-port`, cut from `glm5next-port` at
`6c3013c`, because that branch already carries the V100 build fixes (SM60 opt-in, gcc-14) and the
arch-guard pattern this port copies.

## Verdict first: this is the cheap kind of port

Kolibri's GGUF uses qwen4exp tensor conventions almost everywhere. The expensive GLM machinery
(KDA linear attention, MLA with its indexer, hyper-connections, MTP draft head) is NOT needed:
Kolibri is a standard-attention MoE. Measured from the artifact itself
(`/home/peb/moredata/models/Kolibri-1-GGUF/Kolibri-1-Q4_K_M.gguf`, 903 tensors, 44.19 GiB of tensor
bytes, sha256 `c2ac1301424441ef210b6de50ce25e8ccf69f86494df53d6ba52ed558456062e`):

- **What maps onto Strata unmodified**
  - Expert blobs: gate/up are Q4_K (13.18 GiB each over 50 layers), down is Q4_K (6.59 GiB) and
    Q6_K (9.61 GiB). `native_expert_supported` on CUDA already takes Q4_K and Q6_K
    (`src/kernels/cuda/iq_kernels.cu`, `Fmt<12>`, the Q6_K dequant is in `dequant_bf16.cu`), and the
    CPU path has the vec_dots (`src/kernels/cpu/native_expert.cpp`, `kq_avx2.cpp`).
  - Geometry fits the native kernel constraints: n_embd 2560 and expert n_ff 512 are both divisible
    by the 256 block size of Q4_K and Q6_K, and `(n_ff * n_embd) % 256 == 0`.
  - Shared expert: `ffn_{gate,up,down}_shexp.weight` in every layer, and the generic layer already
    has the shared-expert site with a gate (`src/core/layer.cpp`, `moe_shared`,
    `include/strata/kernels/shared_expert.hpp`).
  - Router weights are F32 (`ffn_gate_inp`, 0.18 GiB total), same policy as qwen4exp/glm5next:
    refuse unless the F32 values are exactly BF16, or narrow with `--compat-bf16`.
  - QK-norm: `attn_q_norm` / `attn_k_norm` over head_dim 128; the GLM path already has
    RMSNorm-over-head-dim kernels for exactly this shape (`verify.cpp` consumes them for glm5next).
  - No MTP, no indexer, no conv/SSM state: every layer is MoE and contiguous, which is SIMPLER than
    GLM's 3-layer dense stem. The expert-table code needs no stem special case.
- **What is genuinely new (the whole kernel-side port)**
  1. The router rule: top-6 selected on `logits + exp_probs_b.bias`, expert weights = the UNBIASED
     `sigmoid(logits)`, no renormalisation (`norm_topk_prob` false). `exp_probs_b.bias` needs a slot
     in the WeightTable (a 384-vector per layer). The routing math itself is a few lines in the
     router stage, gated by a manifest flag.
  2. Attention layout: 50 layers in an SSSS-F interleave (full attention at layers 4, 9, ... 49).
     The 40 sliding layers use a 513-token window with RoPE (theta 10000); the 10 full layers are
     NoPE (skip RoPE entirely, a per-layer flag). SWA is the only real new kernel work: window
     masking in the attention path, CPU and CUDA, and a KV policy that stops accumulating beyond the
     window on sliding layers (KV stays tiny: the full layers' KV dominates).
  3. Sandwich norms: `attn_norm` + `post_attention_norm` and `ffn_norm` + `post_ffw_norm` per layer.
     RMSNorm primitives exist; this is plumbing in the layer's forward, not a new kernel.
- **Ground truth for every gate**: the patched llama.cpp at `/home/peb/llama.cpp-kolibri-try`
  (build `b11396-142c471a5`, master `46847e615` + the four `kolibri1` commits) already loads and
  generates with this exact file. Its logits are the oracle, the same role llama.cpp's GLM5-Next
  graph played for phases 8-9 of the GLM port.
- **Memory fit is the point of the port**: 44.19 GiB of tensors against a 32 GiB card and 31 GiB of
  RAM. The V100 takes attention + router + shared experts + KV, and the expert rows go through the
  row cache with a resident budget of roughly 20 GiB, misses served from mmap'd pages. This is the
  qwen4exp Coder configuration with the numbers scaled up.

## Measured geometry (from the GGUF header, not inferred)

- `general.architecture` = `kolibri1`, GGUF v3, block_count 50 (a count).
- hidden 2560, 48 query heads / 4 KV heads, head_dim 128 (attention q width 6144), context 262144.
- 384 routed experts, top-6, expert FFN 512, 1 shared expert (FFN 512), `expert_gating_func` =
  sigmoid-logit-add (llama.cpp enum 5, added by the patch).
- Vocab 131072, untied embeddings, `token_embd` Q4_K, `output` Q6_K.
- Quant mix by bytes: experts 42.56 GiB (94% of the model), attention 0.9 GiB, embeddings+head 0.42
  GiB, routers 0.18 GiB, norms/biases the F32 remainder.

## Phases, each with its own pass criterion

Discipline from `GLM5NEXT-PORT.md` carries over: every phase ends in a committed, passing gate, and
nothing proceeds on an unverified assumption.

**Status (2026-10-05, evening):** branch `kolibri-port`, four phases DONE and gated:

- `d7cb66a` router: `MoeGeometry::Gating` (SIGMOID_LOGIT_ADD vs pinned SIGMOID_BIASED),
  `tools/kolibri_router_gate.cpp` 7/7 PASS. The gate caught two real kernel bugs (unfilled logit
  sort key; dropped sigmoid in the else branch).
- `957d0fc` layer graph: `kolibri_swa.hpp` (pattern reader, SWA mask, RoPE, QK-norm before RoPE)
  + `tools/kolibri_layer_gate.cpp` 7/7 PASS (600 positions, probes at 0/512/599).
- `eaa84ed` packer: `tools/kolibri_pack.py`; pack at
  `/home/peb/moredata/strata-pack-kolibri`; phase-1 gate
  `strata-load --layout <pack> --layout-layers 50 --layout-hidden 2560 --layout-ffn 512` PASS
  (50 layers x 384 experts, 45.71 GB; the hidden/ffn flags are required - defaults are the
  binary's compiled GLM geometry).
- `0d1ba6b` arch guard: `check_kolibri1_architecture` + `tools/kolibri_guard_gate.cpp`
  (acceptance on the real artifact AND refusal of a glm5next file, both PASS).

Oracle: `llama.cpp-kolibri-try/examples/simple/kolibri-oracle.cpp` (built as
`build-oracle/bin/llama-kolibri-oracle`), experts pinned to CPU via
`tensor_buft_overrides`; verified end to end on "The capital of Austria is" ->
greedy `29116 46 1914 262 3515 735` = "Vienna. So the answer is"; logits dumps at
`/home/peb/moredata/kolibri-oracle/`. NOTE: the oracle's n_vocab is 128,000, not the header's
131,072 - phase-6 logit comparisons must use the oracle's size.

- `62872a6` trunk gate: `tools/kolibri_trunk_gate.cpp` - one token through the engine's own
  50-layer trunk (straight from the GGUF; pack index semantics deferred to serving), compared
  with the oracle's position-0 logits for token 325: max |err| 0.34 over 128k vocab, greedy
  argmax identical (1646), 3.9 s single-thread CPU.

Remaining: a multi-token run (exercises RoPE/SWA/KV through the window), then the serving
integration (pack index path, expert row cache + mmap) and the measured tok/s comparison.

### Phase 0 - inventory (done, this document)
Tensor names, quants, shapes and metadata read from the artifact with gguf-py. Regenerate with
`tools/glm5_inventory.py`'s approach (a `kolibri1` variant is phase 1's first output).

### Phase 1 - the packer (`tools/kolibri_pack.py`)
Reuse `glm5_pack.py`'s machinery. Deltas: gate and up are SEPARATE 3-D tensors
(`ffn_gate_exps` [512, 2560, 384], `ffn_up_exps`), so a blob is cut as `[gate rows | up rows |
down rows]` per expert, concatenating two source tensors; down mixes Q4_K and Q6_K per layer, which
`native_experts.txt` already expresses (per-layer `gu_type d_type`). No dense stem. Router bias
`exp_probs_b.bias` goes to `dense.bin` (F32).
Gate: `strata-load --layout <pack> --layout-layers 50` reports 50 layers, 384 experts/layer, table
starting at block 0, ~42.6 GB total, `experts.bin` matching the table byte for byte.

### Phase 2 - arch guard + geometry
Accept `kolibri1` beside `glm5next`/`qwen4exp` in the guards (`native_head.cpp`,
`native_dense.cpp`, `check_architecture`), per-layer attention-kind table from a new
`attention.layer_type` metadata array (or derived from the SSSSF rule), per-layer RoPE on/off and
window flags in the manifest.
Gate: the reader loads the pack and reports the geometry above; wrong-arch packs still refuse.

### Phase 3 - expert path parity
The native expert rows run on CUDA for Q4_K gate/up and Q4_K/Q6_K down at [512, 2560].
Gate: one layer's MoE output, logits-compared against the llama.cpp oracle within the usual
tolerance, for a fixed input; the CPU path likewise (the existing parity-harness pattern:
`tools/glm5_moe_reference.py` has the shape of it).

### Phase 4 - the router
Sigmoid-logit-add mode: selection on `logits + bias`, weights `sigmoid(logits)`, no renorm, top-6.
Manifest flag (`routing: sigmoid_logit_add`), WeightTable slot for the bias.
Gate: routing decisions (selected ids AND weights) for 64 random hidden states match the oracle's
expert choices bit-for-bit on ids and within fp32 tolerance on weights.

### Phase 5 - attention: SWA, NoPE, QK-norm, sandwich norms
Per-layer flags from phase 2. NoPE = skip RoPE on full layers. SWA = window 513 masking plus a KV
policy that drops positions beyond the window on sliding layers (start at context 8192; 262144 is a
later measurement, not a launch requirement). QK-norm reuses the GLM head-dim RMSNorm kernels.
Sandwich norms wire the two post-norms into the layer forward.
Gate: one full layer (sliding and full variant) logits-compared against the oracle; then the
embedding + one block end to end.

### Phase 6 - trunk graph + end-to-end parity
Wire the 50 blocks (all MoE), the head, and greedy decode for 64 tokens at context 512.
Gate: 64/64 greedy tokens identical to the llama.cpp oracle on the fixed prompt; per-token logits
max-abs-diff reported and within the GLM port's accepted band.

### Phase 7 - the tricks, then serve
Expert-row cache on (`--expert-cache 16g` semantics via resident budget ~20 GiB), mmap experts,
CPU miss tier, `--kv-resident` for the full layers. Measure: tok/s and the cache report
(hits/misses/evictions) at context 512 and 8192, against the llama.cpp baseline measured on this box
(0.9 t/s all-CPU, 2.4 t/s at -ngl 24).
Gate: Strata serves the model on the V100 with the row cache report showing steady-state reuse, and
decode is faster than the llama.cpp baseline or the reason it is not is written down.

## Risks and open questions

- **SWA correctness at depth**: window masking is easy, but the KV eviction policy interacting with
  `--kv-resident` and the prompt cache is where subtle bugs live. Mitigation: phase 5 gates on
  oracle logits before any KV trickery; phase 7 re-gates after.
- **Mixed down-quant per layer** (Q4_K vs Q6_K): the table supports it, but the CUDA grouped path
  wants one format per call. If it does not, fall back to per-layer grouping (two calls) and
  measure; the GLM port's grouped-MoE gate has the harness.
- **F32 router narrowing**: 0.18 GiB of F32; if the values are not exactly BF16, `--compat-bf16`
  rounds them, which perturbs routing near ties. Phase 4's gate must check id-identity, not just
  weight tolerance.
- **Sigmoid gating numerics**: `exp(logits + bias)` can overflow in fp32 for extreme logits; the
  oracle uses sigmoid, so implement it as sigmoid, never as softmax-of-something.
- **Vocabulary and chat template**: 131072 tokens, no BPE surprises known, but the tokenizer lives
  in llama.cpp, not Strata; the port reuses Strata's tokenizer story for qwen4exp (verify it in
  phase 2; if Strata's server needs the chat template, take it from the patched llama.cpp's
  `--jinja` behaviour).
- **What this port does NOT do**: no requantisation (the Q4_K_M file is served natively), no merge
  back to `main` before the parity gates pass, no Windows/HIP claims.

## Effort estimate

The GLM port took phases 0-9 over five days because of KDA, MLA, the indexer, hyper-connections and
MTP. Kolibri has none of those; the new work is SWA, the router mode, sandwich norms and the packer
cut. Roughly a third of the GLM effort: about two focused days to phase 6 on this box, with the
V100 free and `iq_pack` finished (the packer and gates read the same disk).
