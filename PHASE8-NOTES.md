# Phase 8: the ground truth, the wiring check, and the blocker found in the instrument

## What works

* `tools/glm5_ref_dump.cpp` runs the FORK's own GLM5-Next graph with `cb_eval` set and writes every
  named tensor: 947 names for the fixed prompt `"The capital of France is"`, whose golden greedy
  generation is `12089 13 1084 374`.
* `tools/glm5_compare_dump.py` compares a port dump against that manifest and reports the FIRST tensor
  in evaluation order over tolerance.
* `tools/glm5_block_driver.py` runs block 0 of the trunk on the real prompt with the port's verified
  operators and writes the reference's own names, so the comparison is layer- and site-resolved.

Two instrument bugs were found and fixed on the way, both silent:

* **non-contiguous views**: `ggml_backend_tensor_get` on a broadcast/permute returns the underlying
  buffer bytes, not what the graph sees through the view.  It produced a `hc_init` whose four
  broadcast streams differed from each other - impossible for a broadcast, which is how it was
  caught.  Views are now skipped and marked in the manifest.
* **duplicate names**: this graph names two different tensors `inp_embd` (the inputs-embeds
  placeholder `inp->embd` and the real embedding lookup).  One file silently overwrote the other.
  Duplicates are now recorded and the first is kept.

## The blocker

The dumped tensors are **not mutually consistent**, so a comparison against them is not yet meaningful.
Three independent facts, all from the reference's own four dumps:

1. `hc_attn_pre-0` and `attn_norm-0` ARE related as `rms_norm(x) * f` for a per-channel `f` that is
   token-INDEPENDENT (per-token means agree to 1.4e-3), i.e. those two came from the same evaluation;
2. but `f` is NOT `blk.0.attn_norm.weight` - correlation 0.000.  `f` has mean 0.1304 and std 0.0475
   while the stored tensor has mean 0.1027 and std 0.0052, and the stored tensor was confirmed by an
   INDEPENDENT reader (`gguf` + `quants.dequantize`, F32, 4096 values), so the reading is not at fault.
   No value of the norm's eps makes the two agree (tested 0, 1e-8, 1e-6, 1e-5, 1e-4);
3. and `attn_norm-0` does not produce `kda_beta-0` through the documented chain
   (`sigmoid(ssm_beta @ attn_norm)`), failing by 5.8e-01.

Facts 1 and 3 together are the signature of **tensors from more than one graph evaluation**: the two
that agree were captured in the same pass, the third in another.  The likely mechanism is the CUDA graph
warmup, which runs during the same decode window with the same tensor names and dummy data - the log
shows `CUDA graph warmup complete` interleaved with the dump.  The dumper's "first occurrence wins" rule
then keeps whichever pass happened to evaluate a given node first, so the manifest can hold a mix.

## The fix to apply next

Dump exactly ONE evaluation, and make it unambiguous which:

* do not dump during the first decode; dump on a later decode, after the warmups are complete, and
  record the step index in the manifest;
* or disable CUDA graph capture for the dumping run (`GGML_CUDA_DISABLE_GRAPHS=1`) so the warmup cannot
  interleave;
* and record, per tensor, which evaluation produced it, so a mixed dump is detectable instead of looking
  like a wiring bug in the port.

Once the reference's own chain is verifiably self-consistent (`attn_norm-0 == rms_norm(hc_attn_pre-0) *
attn_norm.weight` to float precision, and `kda_beta-0 == sigmoid(ssm_beta @ attn_norm-0)`), a mismatch in
the block-0 comparison means the port, not the measurement.

A self-consistency check belongs in the harness, not in my head: the comparison should first assert the
reference dump's internal relations, and refuse to report a port divergence if the reference itself is
inconsistent.  Episodes like this are exactly why the harness has to check the oracle too.
