# GLM-4.7-Flash on Strata — port plan (branch `glm47-port`)

Cut from `glm5next-port` at `6c3013c` (2026-10-05). That base already carries the MLA kernel, the
router/MoE kernels, the packer machinery and the arch-guard pattern from the GLM-5.3 port, so this
branch starts from a working sm_70/Windows build rather than upstream `main`'s qwen4exp-only tree.

## Verdict first: a standard MLA MoE, and a much cheaper port than GLM-5.3

GLM-4.7-Flash is `general.architecture = deepseek2`: **47 blocks, every one MLA attention, MoE after a
single dense stem.** The expensive GLM-5.3 machinery is absent — no KDA linear attention, no kpool
indexer, no hyper-connections, no MTP head, no SWA interleave. Two things differ from the GLM-5.3 MLA,
and they are the whole kernel-side delta:

1. **Geometry** — 20 heads (not 64) and `q_lora_rank` 768 (not 1536); `kv_lora_rank` 512, MLA head dim
   256 and the 1/sqrt(256) scale are the same.
2. **A 64-dimensional decoupled RoPE** (`rope.dimension_count = 64`, `freq_base = 1e6`). GLM-5.3 is
   noPE; the base `glm_mla` has no rope path at all, so this is the one genuinely new kernel job.

Measured from the artifact (`Huihui-GLM-4.7-Flash-abliterated.Q4_K_M.gguf`, 844 tensors, 18.13 GB,
GGUF v3; official base `zai-org/GLM-4.7-Flash`, MIT).

## Measured geometry (from the GGUF header, not inferred)

- `block_count` 47, `leading_dense_block_count` 1, context 202752, `embedding_length` 2048.
- `attention.head_count` 20, `head_count_kv` **1** (MLA latent), `key_length` 576
  (= kv_lora 512 + rope 64), `value_length` 512, `key_length_mla`/`value_length_mla` 256.
- `q_lora_rank` 768, `kv_lora_rank` 512, `rope.dimension_count` 64, `rope.freq_base` 1e6,
  `layer_norm_rms_epsilon` 1e-5.
- MoE: `expert_count` 64, `expert_used_count` 4, `expert_shared_count` 1, `expert_group_count` 1,
  `expert_feed_forward_length` 1536, **`expert_weights_norm` true, `expert_weights_scale` 1.8**.
  No `expert_gating_func` key and no swiglu-clamp key.
- Dense stem FFN `feed_forward_length` 10240; vocab 154880 (untied `token_embd` Q4_K / `output` Q6_K);
  tokenizer gpt2-BPE, `pre = glm4`, chat template `[gMASK]<sop>`.
- Bytes by role: experts 16.77 GB (all Q4_K/Q6_K — already in Strata's dequant table), attention
  0.59 GB, embd/head 0.44 GB, routers 0.02 GB. No FP8, no new quant type.

Per-block wiring: `attn_norm → MLA → +x → ffn_norm → MoE → +x`. Block 0 replaces the MoE with a dense
`ffn_gate/up/down` (ff 10240). Blocks 1-46 carry `ffn_gate_inp` (router) + `exp_probs_b.bias` +
`ffn_{gate,up,down}_exps` + `ffn_{gate,up,down}_shexp`.

## What maps onto Strata unmodified (reuse)

- **MLA** — `glm_mla.hpp` computes exactly this absorption: the wk_b-absorbed query, K and V the same
  512-dim latent with one head, wv_b un-absorbing per head on the way out, scale 1/sqrt(head_dim).
  Reuse with `MlaGeometry{n_head 20, q_lora 768, kv_lora 512, head_dim 256}` and the new rope path.
- **MoE + router** — `glm_moe.hpp` is the sigmoid router with `exp_probs_b` biasing **selection only**,
  weights from the unbiased probabilities, normalize + clamp, then `expert_weights_scale` **after**
  normalisation. GLM-5.3 already reads `expert_weights_norm`/`expert_weights_scale` — the exact two
  metadata values this model needs (1.8 / true). Reuse with `n_expert 64, n_used 4, ff 1536`.
- **Dense stem** — the existing parallel-SiLU `build_ffn` at ff 10240.
- **QK-norm** — RMSNorm over head_dim; the GLM path already has the kernel.
- **Packer** — `glm5_pack.py`'s machinery (dense.bin / index.txt / native_experts.txt / manifest.json),
  simplified.
- **Arch guard** — `check_glm5next_architecture` is the template for a `deepseek2` variant.

## The gap list (each item anchored in the file)

1. **MLA rope** — the 64-dim decoupled RoPE. The base MLA is noPE; this needs a rope path on the query
   and the latent's rope slice, matching llama.cpp's `deepseek2` graph. Largest new kernel job (and the
   one to gate first, because a wrong rope slice/order is silent).
2. **MLA geometry** — 20 heads, `q_lora` 768 threaded through the packing + binding + kernel geometry.
3. **Router/MoE geometry** — `n_expert 64`, `n_used 4`, `ff 1536`, `w_scale 1.8`, `norm true`,
   `clamp_exp`/`clamp_shexp` 0 (no clamp key). Gating mode SETTLED (phase 4): SIGMOID, sigmoid-then-bias.
4. **Arch guard + geometry** — accept `deepseek2`; vocab 154880; refuse everything else precisely.
5. **Packer** — `tools/glm47_pack.py`: all-MLA, dense stem block 0, 46 MoE blocks; gate/up are separate
   expert tensors (the Kolibri cut), down is per-layer Q4_K/Q6_K (the mixed-down table already exists).
6. **Trunk graph + head** — wire the 47 blocks, the final norm and the untied `output`.
7. **Serving** — the `--serve` path, tokenizer (`pre = glm4`, `[gMASK]<sop>` template), endpoints.

**Not needed:** KDA, the kpool indexer, hyper-connections, MTP, SWA, FP8 — none are in this artifact.

## Oracle

Our patched `llama.cpp` (`llama-kolibri` build, b11385) registers `deepseek2` and **runs this exact
file** (measured ~96 t/s short-context / ~28 t/s at 60K). Its logits are the ground truth for every
gate below, exactly as llama.cpp's GLM5-Next graph was for phases 8-9 of the GLM port.

## Phases, each with its own pass criterion

| phase | work | gate |
|---|---|---|
| 0 | this inventory | tensor set + metadata match the header |
| 1 ✅ | packer (`tools/glm47_pack.py`) | **PASSED** — `strata-load --layout --layout-layers 47`: 47 layers, 64 experts/layer, first expert block 1, 16.77 GB, dense stem block 0, Q4_K/Q6_K down mix |
| 2 ✅ | arch guard + geometry (`deepseek2`) | **PASSED** — `strata-gguf <gguf> --glm47` reports 47 blocks / 47 MLA / 64 experts / rope 64 and accepts the artifact; a non-deepseek2 file refuses (`architecture is 'gpt-oss', expected 'deepseek2'`) |
| 3 ✅ | MLA parity (incl. the new rope) | **PASSED** — `glm47_mla_parity` vs `tools/glm47_mla_reference.py`: all 9 stages (qr, q_nope, q_pe-after-rope, kv, k_pe-after-rope, Qcur, attn, v, out) under 1e-4 over 5 cases (depths 1–33, pos 0–4000); worst stage max abs 8.0e-06 |
| 4 ✅ | router/MoE parity | **PASSED** — `glm47_moe_parity` vs `tools/glm47_moe_reference.py` (both gating tasks): top-4 ids EXACT on all 3 cases, weights 5.96e-08 (<1e-5), probs 1.25e-06, moe 3.58e-06 / shexp 1.61e-06 / out 3.96e-06 (<1e-4), over the model's REAL layer-1 `ffn_gate_inp` + `exp_probs_b` + experts; teeth 3/3 (ids flip without the bias and under bias-then-sigmoid) |
| 4b ✅ | full decoder block | **PASSED** — `glm47_block_parity` vs `tools/glm47_block_reference.py`: `attn_norm → MLA → +x → ffn_norm → MoE(+shexp) → +x`, all 7 stages (xb, attn, x2, ff, moe, shexp, out) < 1.1e-05 (random) and < 8.3e-06 (REAL layer-1 weights) over 3 cases, top-4 ids EXACT. Adds the block RMSNorm (`glm_norm.cpp`: eps 1e-5, gain is NOT folded as `1+w`) and the two residuals — the unit the trunk is made of |
| 4c ✅ | dense stem block (blk.0) | **PASSED** — `glm47_block_parity <dense fixt>` vs `tools/glm47_block_reference.py --dense`: `attn_norm → MLA → +x → ffn_norm → dense FFN (ff 10240) → +x`, all 6 stages (xb, attn, x2, ff, ffn, out) < 1.1e-05 (random) and < 5.3e-06 (REAL block-0 weights); the one block with no router and no shared expert |
| 5a ✅ | trunk loop + head (engine code) | **PASSED** — the loop now lives in engine code (`strata/core/glm47_trunk.cpp` + `glm47_trunk.hpp`: a weights-provider + `expert fn` interface) and `glm47_trunk_parity` drives THAT function vs `tools/glm47_trunk_reference.py`: block 0 dense, the rest MoE, **each layer its own MLA cache**, then `output_norm` → project → argmax. 2 layers × 2 tokens AND 5 layers × 4 tokens both PASS: per-(token,layer) hidden rel < 3.7e-06, top-4 expert ids EXACT (16/16 at depth), logits rel < 3.9e-06, argmax exact. With the **ARTIFACT's own weights** (blk.0 dense + blk.1 MoE + the real 154880-row head): hidden rel 6.5e-08, hn/logits rel ~1e-07, top-4 ids EXACT, argmax 70799 exact. **Teeth:** aliasing every layer to ONE shared cache (the classic bug) moves the hidden state by rel 0.86–1.10 (random) and 5.8e-03 on real weights — ~1e5× the parity noise, so the gate really exercises the per-layer cache, the thing no per-block gate can see |
| 5 | trunk + end-to-end | 64/64 greedy tokens identical to the oracle on a fixed prompt. **Recon (done):** the engine's `generate` path is PLE/QSA-specific (qwen4exp) and the GLM-5.3 port's own trunk is *also* unwritten (`GLM5NEXT-PORT.md` phase 8 = "graph still to write"), so there is no trunk to adapt — this is a from-scratch `deepseek2` CPU trunk: the 47-block loop (block 0 dense stem), final `output_norm`, untied `output`, the tokenizer, then the greedy-token comparison. Phase 4b verified the block unit and `glm_norm.cpp` supplies the block norm; the block loop, the embedding/head, and the oracle harness remain. |
| 6 | serving | `/v1/chat/completions` returns a valid response through the exported tokenizer |

## Risks and open questions

- **MLA rope layout** — SETTLED (phase 3): q splits per head into nope `q[..., :192]` (absorbed by wk_b) and
  roped `q[..., 192:256]`; `wkv_a_mqa` yields `kv_cmpr = [0:512]` (rms-normed, the latent) and `k_pe = [512:576]`
  (roped, one head). NEOX rope, theta_i = pos·freq_base^(-2i/64); kq_scale 1/16. The C++ path is verified
  against the float64 oracle stage by stage, so the layout is anchored to the reference, not the paper.
- **Router gating mode — SETTLED (phase 4): `SIGMOID`, i.e. sigmoid-then-bias.** The artifact has no
  `deepseek2.expert_gating_func` key, and `src/models/deepseek2.cpp:22-32`'s fallback maps
  (`n_layer` 47, `n_vocab` 154880) → `LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID`. In `build_moe_ffn`
  (`src/llama-graph.cpp:2064-2094`) the SIGMOID branch is `probs = sigmoid(logits)`, then
  `selection_probs = probs + exp_probs_b` — the bias is added **AFTER** the sigmoid, selection is on the
  **biased** probs, and the weights are gathered from the **unbiased** `sigmoid(logits)`, normalised,
  then ×1.8. This is *not* Kolibri's own `SIGMOID_LOGIT_ADD` variant (`selection_probs = logits +
  exp_probs_b`, i.e. bias-then-sigmoid), reserved for other models by the `:2085` branch. `glm_moe.hpp/cpp`
  already implements exactly the SIGMOID order, so phase 4 reused it unmodified; the gate proves the
  fixture distinguishes the two orders (all 3 cases change ids under bias-then-sigmoid) and that the bias
  drives selection (all 3 change ids when it is dropped). Its magnitude is real: `exp_probs_b` ≈ 9.03 on
  layer 1.
- **F32 routers** — refuse unless the F32 values are exactly BF16, or narrow with `--compat-bf16` (the
  same policy as qwen4exp/glm5next); check id-identity, not just weight tolerance.
- **The selection bias must NOT be narrowed to BF16** — measured, and the sharpest new risk of this
  port. The pack's router path stores routers BF16 (`glm_bind.cpp:113-117` expands BF16→F32), which is
  lossless here: `ffn_gate_inp.weight` is exactly BF16-representable (131072/131072 values). But
  `exp_probs_b.bias` is F32 and is **not** (0/64 values; BF16 truncation error up to 5.7e-2 on values
  ≈9.03). The bias spread that orders the 64 experts is ≈2e-2, so a BF16 bias flips the top-4. The
  packer (phase 5/6) must keep `exp_probs_b` in F32, and the trunk gate must check id-identity, not just
  output tolerance — a BF16-narrowed bias can leave the outputs looking plausible while routing to the
  wrong experts.
- **Mixed down-quant** — down mixes Q4_K and Q6_K per layer; the table supports it but the CUDA grouped
  path may want one format per call (fall back to per-layer grouping and measure if so).

## Build notes (Windows, this box — needed before any gate can run)

The base `strata` build here predates these, so they are build-config facts, not port code:

- **CCCL / CUB vs MSVC.** `glm_indexer_device.cu` includes `<cub/...>`; CCCL emits a fatal `#error` under
  MSVC's traditional preprocessor. Fixed by adding `/Zc:preprocessor` (CXX) and `-Xcompiler=/Zc:preprocessor`
  (CUDA) in `CMakeLists.txt`, guarded by `if(MSVC)`.
- **Compile-time expert geometry.** The CPU expert path is built for one geometry pair,
  `STRATA_MODEL_H` / `STRATA_MODEL_FF` (`CMakeLists.txt:60-61`). The defaults are qwen4exp's (2560 / 640)
  and reject a GLM pack ("activation larger than the pool's buffers"). **GLM-4.7-Flash must be built with
  `-DSTRATA_MODEL_H=2048 -DSTRATA_MODEL_FF=1536`.** A Strata build therefore serves one model geometry;
  a future clean fix is to drive these from the pack's manifest instead of a compile-time constant.
- **`--layout-layers`.** `load_main.cpp:94` defaults `layout_layers = 46` (GLM-5.3). GLM-4.7 needs
  `--layout-layers 47`. Same future fix: read `block_count` from the manifest.
- Build recipe: `build_strata_local.bat` + the two `-D` values above; the CUDA v13.3 `bin/x64` must be on
  `PATH` when running, or the exe exits immediately.

## What this branch does NOT do

No requantisation (the Q4_K_M file is served natively), no merge back to `main` before the parity gates
pass, no Windows/HIP claims beyond the build the base already produces.
