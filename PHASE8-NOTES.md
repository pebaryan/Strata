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

## The blocker, resolved as far as it goes: the ACTIVATION dump is not trustworthy

Two runs with identical arguments produced **879/879 byte-identical tensors**, so the dump is
deterministic and evaluation mixing is ruled out. The tensors are reproducible - and still wrong:

* `hc_init`'s four "streams" have **zero correlation with each other** (-0.0006, +0.0323, +0.0005),
  i.e. four unrelated vectors. The graph builds that tensor as
  `repeat_4d(reshape_3d(inp), n_embd, hc, n_tokens, 1)` - a broadcast. No read of a real broadcast
  can produce unrelated streams, and no input data can either.
* my token-embedding rows correlate ~0.02 with the dumped `inp_embd` columns and ~0.01 with
  `hc_init` - i.e. not the same vectors at all, not merely scaled differently.
* `attn_norm-0` is only 0.977-correlated with `rms_norm(hc_attn_pre-0)*w` per channel, where a single
  evaluation would make that ratio *be* the weight vector, identical to machine precision.

The most likely mechanism is the read path, not the graph: with the experts on the CPU (`-ncmoe`) this
graph is SPLIT across the CUDA and CPU backends, and `ggml_backend_tensor_get` on a tensor that belongs
to the other backend can return stale bytes from the buffer rather than the computed values.  That would
also explain why each run reads the same wrong (but deterministic) region.

**What is trustworthy**: `tokens.txt` (the sampled generation, produced through llama's own sampler) and
the graph's OUTPUT tensors (`result_output`, the 154880 logits) - which is what the generated tokens were
sampled from, so they are validated by the tokens themselves.

## What this means for phase 8

The pass/fail gate stays what the plan always said: **greedy tokens identical to the reference**, plus a
logits comparison on the last prompt token.  That is measurable with the paths that demonstrably work.

The layer-by-layer dump would be a *localization* aid, not the gate, and it needs the backend read fixed
first: dump only tensors on the backend that computed them (pass the backend to the callback, or
synchronize and read per-backend), or run the reference with a single backend (no `-ncmoe`, e.g. a
smaller model) so no split exists.  Until then, a port divergence cannot be attributed to a layer by this
mechanism - and pretending otherwise would send me hunting bugs that do not exist in the port.

The lesson, stated for the next person: I built the instrument for the port and did not check the
instrument.  Two rounds of "the port disagrees with the reference" were actually "the reference's dump
disagrees with the reference's graph", and the thing that caught it was a relation that must hold by
construction (a broadcast whose copies differ), not a tolerance.

