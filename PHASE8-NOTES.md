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

## Part 2: the read path, and where the diagnosis stands

The scheduler's callback fires AFTER the node is computed and after `ggml_backend_synchronize`
(ggml-backend.cpp:1824-1834: with a callback set, each node is computed alone as a one-node graph view,
then synchronized, then reported).  The "it fires pre-compute" hypothesis is dead.

The dumper now records PROVENANCE for every tensor - `view=Y/N`, `contig=Y/N`, strides, and the backend
buffer name - because a dump that is wrong cannot otherwise be told from a dump of the wrong tensor.
That immediately settled two of the anomalies:

* `inp_embd` is a PLACEHOLDER, not data: the graph names two tensors `inp_embd` (the inputs-embeds
  buffer and the real embedding lookup), the placeholder is written first, and for a token-based batch
  nothing fills it.  Not a read bug.
* the weights are dumped as the loader's VIEWS (`blk.0.hc_attn_scale.weight (view)`, view=Y contig=Y).
  Comparing them against the file gives 180 exact agreements and 180 "disagreements" that are slices of
  stacked per-layer tables, where the view's offset is not 0 - inconclusive, not a verdict.

STILL UNEXPLAINED, and it is what blocks per-block localization: `hc_init` is recorded `view=N contig=Y
nb=[4,16384,65536,327680]`, a MATERIALISED contiguous [4096,4,5,1] tensor whose four hc slices must be
identical (the graph builds it with `repeat_4d`), yet they differ by 6.3e-02 - and dump5 is
byte-identical to dump3, so it is not a torn write.  A contiguous materialised broadcast cannot have
differing slices.  Three attempts to settle it from ggml's internals did not, and a fourth is not worth
it: the measurement needs to be PROVEN right by construction, not argued right by inference.

THE FIX TO BUILD NEXT: patch the fork's `graph::build` to append a `ggml_dup` COPY node for each tensor
we want, with a unique name we choose, and read them after `llama_decode` returns (the graph's buffers
stay valid until the next decode).  A copy materialises the value and gives a tensor whose `data` is
unambiguous, and it removes the placeholder/duplicate-name problem at the same time.  An end-to-end read
test belongs in the dumper either way: write a known pattern into the embedding input and confirm the
dump returns it, BEFORE trusting any activation from the same path.

Port state meanwhile: the trunk runs all 45 blocks to the logits; against the reference's
`result_output` (validated: its argmax reproduces the sampled token) the correlation is +0.46, the right
token ranks 331, top-50 overlap 2/50.  A partial semantic error accumulating over 45 blocks - and
localizing it needs exactly the per-block ground truth this section is about.

## Part 3: SOLVED, and it was the reader all along

The read path is broken, and the signature is measured rather than argued.  Method: `--embd-test` feeds
the graph a pattern of EMBEDDINGS whose values are chosen exactly (build_inp_embd takes the embd path
when `ubatch.token` is null, so the graph's placeholder IS its input).  Then every downstream tensor can
be checked against data I chose:

    inp_embd (reshaped)   buf=CUDA_Host  off=32      -> matched the pattern EXACTLY (max|d| 0.000e+00)
    result_output         buf=CUDA0      off=0       -> argmax reproduces the sampled token (correct)
    hc_init               buf=CUDA0      off=441088  -> WRONG: not the pattern broadcast at all
    hc_attn_pre-0         buf=CUDA0      off=768768  -> wrong
    l_out-0               buf=CUDA0      off=441088  -> wrong

So: device tensors at offset 0 and host tensors read correctly; device tensors at a NONZERO offset
inside their buffer do not.  `ggml_backend_tensor_get` on those returns a different region of the same
buffer - deterministically, which is why every dump reproduced itself byte for byte and why the values
always looked plausible.

This explains everything that came before it:

* `hc_init`'s four "unequal streams": it was never the broadcast, it was another tensor's data.  The
  relation that caught it (a broadcast whose copies must be identical) was the right test.
* the earlier "attn_norm-0 is not rms_norm(hc_attn_pre-0) * attn_norm.weight" mismatch: BOTH tensors
  came from wrong regions, so that comparison was meaningless, and the "the graph must apply a
  different weight" conclusion was an artifact.
* every `l_out-N` comparison in the first phase-8 attempt was comparing garbage to garbage.

**Consequence, stated plainly: the port's divergence is NOT yet established.**  The +0.46 correlation
against `result_output` stands (that tensor is at offset 0 and is validated by the sampled token), but
nothing about WHICH PART of the trunk is wrong has been measured, because the only per-block reference
was unreadable.

**The fix**: never read a device tensor with `ggml_backend_tensor_get`.  Copy it into a host tensor
(`ggml_backend_tensor_copy` into a CPU buffer attached to a host tensor, the same path llama.cpp itself
uses to bring logits back) and read the host copy.  The disk compare harness then needs no changes; only
the dumper's read does.  After that, the provenance column makes every dump self-checking: a tensor at a
nonzero device offset is only trustworthy if it came through the copy path.


