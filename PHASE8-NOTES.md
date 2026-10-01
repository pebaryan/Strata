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



## Part 4: the CPU-forced read is also falsified - the last suspect is the interpretation

The fork is patched (STRATA_DUMP_FILTER forces matched intermediates onto the CPU backend, reusing the
same ggml_backend_sched_set_tensor_backend pattern the file already applies to "norm"/"l_last").  It
works: hc_init's provenance moved from CUDA0 to CUDA_Host.  The values did not move at all:

    hc_init (CUDA0)                  stream 0 vs 1 = 4.091e+00
    hc_init (CUDA_Host, CPU-computed) stream 0 vs 1 = 4.091e+00   <- bit-identical
    input read back (inp_embd)        max|d| against the pattern = 0.000e+00 (exact)

Hypotheses now dead, in order of burial: callback fires pre-compute (source), mixed evaluations (source),
the inp_embd placeholder (real, but not the cause), non-contiguous views (real, not the cause), the read
API on device tensors (a real symptom, but not the cause), scheduler liveness under per-node evaluation
(falsified by --only), and now tensor PLACEMENT (falsified by the CPU force).

What remains, and it is a different kind of suspect: the assumption that this tensor IS what the source
says it is.  hc_init is repeat_4d(reshape_3d(inp)) - a broadcast, so its four hc slices are identical BY
CONSTRUCTION for any input, any backend, any read that addresses the right bytes.  The input reads back
exactly.  The read of this tensor is stable under every variation I can make from outside.  So either the
bytes are not this tensor's, or this tensor is not a broadcast.

Next test, and it is cheap: stop inferring from the source and measure the tensor itself.  Ask for a
tensor whose value is a KNOWN FUNCTION OF THE KNOWN INPUT and nothing else, and for hc_init ask the
runtime for its shape AND its memory offset together in the same run, then check whether a read at
t.data - off (i.e. the buffer base) is what another named tensor holds.  If the buffer base holds a
neighbour, the read is losing the offset for host buffers too and the fix belongs in the dumper's read.

## Part 5: CORRECTION - the dump was never broken.  My reshape was.

The read-path investigation is closed, and the answer is not the one parts 3 and 4 concluded.  The
reference dump is trustworthy and always was.  The bug was in the analysis scripts.

ggml stores a tensor of ne=[4096,4,5,1] with nb=[4,16384,65536,327680] bytes, so the fp32 payload index
is i_embd + 4096*i_hc + 16384*i_tok.  The right reshape is (tok, hc, embd).  My checks used
(embd, hc, tok), whose C-order strides are 20*e + 5*h + t - a different mapping, i.e. I was comparing the
wrong axis.  Both readings on the SAME file:

    reshape(embd, hc, tok)  ->  hc spread 4.091e+00   <- the "four streams differ" claim
    reshape(tok, hc, embd)  ->  hc spread 0.000e+00   <- correct: a perfect broadcast, as constructed

So hc_init - the anchor the whole investigation started from - reads back exactly, in every dump, from the
first one on.  Consequences, stated plainly:

  * Everything in parts 3 and 4 is an artifact of this error: the "device tensors at nonzero offsets read
    wrong", the "liveness hypothesis", the host-copy and CPU-forced fixes.  The offset correlation was
    real in the manifest but never the cause - a wrong axis produces mismatched numbers whatever the
    buffer is.
  * The "attn_norm-0 is not rms_norm(hc_attn_pre-0) * attn_norm.weight and no eps reconciles it"
    conclusion used the same helper, on the same shape of tensor, and is equally void.  There is no
    evidence that the graph applies a different weight.
  * No instrument change was needed.  The fork's STRATA_DUMP_FILTER extension and the --only filter are
    harmless (env-gated, inert by default) but unnecessary; keep or drop as convenience.

LESSON, and it is the same one this file already records twice: check the instrument by construction, and
when a relation that MUST hold fails, suspect the measurement machinery before the graph.  Here the
relation that must hold was a broadcast having identical slices; it did hold, and the tool that said
otherwise was mine.  A five-minute check against ggml's own strides (nb, printed in the provenance column
I had already added) would have caught it - the data was in the manifest the whole time.

The reference dump is usable for per-block bisection.  That is the next step, and it is the one the port
has been waiting for since phase 8 started.

## Part 6: the bisection - block 0 agrees exactly until the KDA attention

With the axis convention fixed (tools/glm5_bisect.py), the block-0 chain is unambiguous:

    hc_init          0.000e+00   r +1.0000   exact
    hc_attn_pre-0    1.760e-05   r +1.0000   fp noise - the mHC pre-mix is right
    attn_norm-0      1.121e-04   r +1.0000   fp noise - the RMS norm is right
    attn_output-0    2.077e-02   r +0.5136   FIRST DIVERGENCE - the KDA attention
    hc_attn_post-0   8.497e-02   r +0.4938
    hc_ffn_pre-0     1.084e-01   r +0.5647
    ffn_norm-0       9.781e-01   r +0.6024
    ffn_out-0        5.167e-02   r +0.4657
    l_out-0          7.126e-02   r +0.4080   everything downstream inherits it

So the embedding, the hyper-connection pre-mix, and the attention norm are correct, and the port's first
error is the KDA attention itself - in the FIRST block, which is a KDA layer per attention.head_count_kv.
Correlation +0.51 rather than ~0 says it is partially right: consistent with one wrong term in the KDA
recurrence or front end (decay g, beta, the conv1d state, the L2 norms, the within-head norm, or wo)
rather than a structural failure.

The two earlier "everything diverges" reports were flat-comparison artifacts: the port writes numpy
arrays in ggml AXIS order with C-order strides, the reference writes raw memory order, so the payloads
must be transposed, not reshaped. Both reports had the right magnitudes and near-zero correlation - which
is exactly what a genuinely broken port looks like, hence two turns spent on the wrong suspect.

Next: bisect INSIDE the KDA block. hc_attn_pre-0 and attn_norm-0 are known-good inputs, so dump the KDA
front end (kda_gate-0, kda_beta-0, the conv1d outputs, the q/k/v norms) and find the first wrong term.

## Part 7: inside the KDA block - the decay gate and beta are wrong, the input is not

Bisecting inside block 0's attention (the port and reference dumps both carry the KDA internals, so no new
dump was needed):

    attn_norm-0     1.121e-04  r +1.0000   the KDA input x is correct
    kda_gate-0      2.777e+00  r +0.9856   FIRST DIVERGENCE - the decay gate g   [scale 4.95]
    kda_beta-0      1.190e-01  r +0.9890   beta is wrong too                    [scale 0.98]
    attn_output-0   2.077e-02  r +0.5136   inherits both

Reading: both SSM front-end outputs are wrong while the x they are computed FROM is right, so the error is
in the parameter path - ssm_a, ssm_dt.bias, ssm_f_a/ssm_f_b, ssm_beta - or in the formula itself. The
correlation is +0.99 for both, so this is a systematic error (a wrong or missing term, a wrong constant, a
wrong tensor), not a layout or indexing failure: a bad layout would decorrelate, as attn_output-0's +0.51
shows further down.

Next experiment, and it discriminates the two remaining possibilities: compute g and beta with the NUMPY
ORACLE from the reference's own verified attn_norm-0 input and compare against the reference's kda_gate-0
and kda_beta-0.

  * oracle matches the reference  -> my FORMULA is right and the PORT KERNEL is wrong: fix in C++.
  * oracle also disagrees         -> my formula or weight mapping is wrong: fix oracle and kernel together.

Care before running it: the dump is in ggml memory order (i0 fastest) whereas the oracle's helpers use
[tokens, dim] numpy arrays, so the input must be transposed deliberately - the exact class of mistake that
cost two turns in parts 3-5. Verify the oracle's own convention first by checking that its internal
rms_norm(hc_attn_pre-0) reproduces the reference's attn_norm-0, which the dump already contains: that is a
free end-to-end check of the oracle's input path before any gate comparison is believed.

## Part 8: the formula is right; the PORT's KDA kernel is wrong - by a factor of 64

tools/glm5_kda_oracle_vs_ref.py feeds the reference's own verified input into the numpy oracle and checks
its internal norm FIRST (the guard), then compares the gate and beta against the reference's dumps:

    GUARD  rms_norm(hc_attn_pre-0) -> attn_norm-0     5.960e-08   r +1.0000   MATCHES
    ORACLE kda_gate vs reference                      4.35e-02    r +1.0000   0.88% of a 4.95 scale
    ORACLE kda_beta vs reference                      6.90e-04    r +1.0000   0.07% of a 0.98 scale
    PORT   kda_gate vs reference                      2.78e+00    r +0.9856   56%
    PORT   kda_beta vs reference                      1.19e-01    r +0.9890   12%

The oracle is 64x closer than the port on the gate and 172x closer on beta, both at r = +1.0000, so the KDA
FORMULA and the weight mapping in the oracle are right and the port's C++ kernel is wrong. The residual
~1% on the gate is consistent with rounding: g = GATE_LOWER * sigmoid(-h) is sharply nonlinear, and h
reaches ~10, so a fraction of a percent in h shows up as ~1% in g. Judging a 4.95-scale gate with an
absolute 1e-4 threshold had mislabelled it as a mismatch; the script now judges on RELATIVE error.

Narrowed further: the port's error is systematic (r +0.9856) rather than a layout failure, which would
decorrelate. So look at the gate/beta term itself in src/kernels/glm_kda.*: ssm_a (== -exp(A_log)),
ssm_dt.bias, ssm_f_a/ssm_f_b, ssm_beta, the reshape to (nh, hd) before the ssm_a multiply, and whether the
port applies the same -1 fold that makes sigmoid(-h) == 1/(1+exp(h)).

Method note, since this is the third convention trap in this phase: the guard is what makes the gate
comparison trustworthy, and it is cheap - it uses a tensor the dump already contains and a step the oracle
performs internally anyway. Run it BEFORE interpreting any comparison that follows.

## Part 9: FOUND IT - dt_bias is bound with 64 entries and read as if it had 8192

The port's KDA formula is identical to the oracle's, line for line: xn (normed, and the dump proves the
norm right), the low-rank ssm_f_a -> ssm_f_b composition, +dt_bias, the per-head ssm_a multiply, the same
-h fold, and beta = sigmoid(ssm_beta @ xt).  So the formula was never the suspect; the WIRING was.  The
GGUF shapes decide it:

    blk.0.ssm_a          (64,)        the kernel indexes [h]                consistent
    blk.0.ssm_beta       (64, 4096)   the kernel walks h*ne + c             consistent
    blk.0.ssm_f_a        (128, 4096)  matvec(out=hd, in=ne)                 consistent
    blk.0.ssm_f_b        (8192, 128)  matvec(out=di, in=hd)                 consistent
    blk.0.ssm_dt.bias    (8192,)      the kernel walks h*hd + i  <-- BOUND AS 64

include/strata/kernels/glm_kda.hpp:49 states the contract:  `const float* dt_bias;  ///< [d_inner]`.
src/core/layout.cpp:101, the ONLY declaration of ssm_dt.bias in the port, sizes it g.ssm_v_heads - the Qwen
GDN convention, where dt genuinely is per-head.  In the GLM artifact ssm_dt.bias is d_inner = hd*nh = 8192
wide and the formula adds it BEFORE the (nh, hd) reshape, so the kernel reads ~8128 floats past the end of
a 64-float weight.  Whatever the packer placed next is what the gate has been adding.

That single defect explains the measured signature exactly: the gate is wrong by 56% with r +0.9856 (a
deterministic neighbour, not noise), while beta - which touches no dt_bias - is only 12% off and
attn_norm/hc_attn_pre/hc_init are exact.

WHY THE KDA PARITY GATE PASSED 3/3: it builds its own fixture weights, including an 8192-float dt_bias
array, and calls the kernel directly.  It validates the kernel and never the wiring, so a loader that
binds 64 floats is invisible to it.  Same shape of gap as the swiglu clamp and the router bias: the gate
must exercise the term AND take its inputs from the path under test.

FIX: size ssm_dt.bias (and the GLM KDA weight layout generally) from the artifact, not from the Qwen
per-head convention - 8192 here, model-dependent, so the GLM layout needs its own entry rather than
reusing gdn_v's.  Then assert at bind time that the length matches the kernel's documented [d_inner]
contract, so the next mismatch of this kind fails loudly instead of silently reading a neighbour.

## Part 10: PART 9 IS WRONG - dt_bias is fine.  What the pack actually contains, measured.

Part 9 blamed `src/core/layout.cpp:101` sizing ssm_dt.bias as g.ssm_v_heads.  That is the Qwen4-Exp layout
table, not the GLM path's, and the claim was never checked against the pack before it was written down.
Measured now, byte-for-byte, by searching strata-pack-glm5/dense.bin for the artifact's own values:

    ssm_a          (64,)      fp32 BYTE-IDENTICAL to the GGUF, at offset 49536
    ssm_dt.bias    (8192,)    fp32 BYTE-IDENTICAL to the GGUF, at offset 443008
    attn_norm.weight, ssm_norm.weight   likewise identical
    ssm_f_a.weight (128,4096)  NOT PRESENT in dense.bin, as fp32 or fp16
    ssm_f_b.weight (8192,128)  NOT PRESENT
    ssm_beta.weight (64,4096)  NOT PRESENT

So every KDA length the kernel asks for is right (ssm_a 64, dt_bias 8192, f_a hd*ne, f_b di*hd, beta
nh*ne), the two 1-D inputs to the gate are exact, and the gate formula in src/kernels/glm_kda.cpp is
identical to the oracle's.  The matrices are not in the dense arena at all: they are served natively from
the GGUF, i.e. still quantized (the index row carries a quant type and a 32-byte block size for them),
which is also why an earlier attempt to read them at a byte offset read nothing.

That leaves one specific question, and it is exactly where the 56% lives: the gate's value is
v = ssm_f_b @ (ssm_f_a @ xn) + dt_bias, and the two matvecs are the only part of it whose weights the port
gets from a natively-served, still-quantized source.  If the loader treats those blobs as float while the
pack serves them quantized, the error would be confined to v - which is what the measurement shows: g
wrong by 56% at r +0.9856, beta (same xn, no matvec in its path) wrong by only 12%, and everything upstream
exact.

Next: read what the port's loader does with ssm_f_a / ssm_f_b - the dequantization path and the sraw
convention - and check it against the GGUF type.  Also verify the port's beta path, which uses ssm_beta
with the same xn and no dt_bias, since a 12% error there needs its own explanation once the gate is fixed.

Method note worth keeping: part 9 was written from a grep hit in the wrong code path and disproved by one
cheap measurement.  Read the table that the path under test actually uses, and measure the artifact before
blaming a declaration.

## Part 11: the artifact's KDA matrices are Q8_0, and the pack stores them TRANSPOSED

The index's field 9 is the ggml type and field 11 the block size:

    blk.0.ssm_a           type 0 (F32), block 1        <- and byte-identical to the GGUF
    blk.0.ssm_dt.bias     type 0 (F32), block 1        <- likewise
    blk.0.ssm_beta.weight type 8 (Q8_0), block 32      listed as (4096, 64)
    blk.0.ssm_f_a.weight  type 8 (Q8_0), block 32      listed as (4096, 128)
    blk.0.ssm_f_b.weight  type 8 (Q8_0), block 32      listed as (128, 8192)

Two consequences, and the second is testable immediately:

1. The gate's v = ssm_f_b @ (ssm_f_a @ xn) + dt_bias is the only term in the KDA front end whose weights
   the port reads QUANTIZED, which is why the error concentrates there: g 56% at r +0.9856, beta 12%, and
   everything upstream exact.  (The oracle reads the same tensors dequantized and matches the reference to
   0.9%, so the values in the artifact are not in question.)

2. The pack's listed shape is the TRANSPOSE of the GGUF's in every case - beta (4096,64) vs the GGUF's
   (64,4096), f_a (4096,128) vs (128,4096), f_b (128,8192) vs (8192,128).  That is consistent with the
   port's matvec contract `matvec(W, x, y, out, in)` (f_a as out=hd/in=ne wants (ne, hd) = (4096,128)),
   so the transposition is deliberate for the matvec path - BUT the beta access in glm_kda.cpp is
   `b_row = ssm_beta + h * ne` with `acc += b_row[c] * xt[c]`, which requires (nh, ne) with ne contiguous
   - the GGUF's orientation, NOT the pack's.  Read in the pack's (ne, nh) layout that walks columns, which
   is a wrong-weights read, and beta is precisely the output that is wrong with no other explanation yet.

Next, in order: (a) confirm how the port's loader dequantizes and transposes the Q8_0 KDA matrices, block
order included, against the GGUF; (b) fix the beta access to match whatever orientation the loader
actually produces, rather than the one the kernel was written against.

## Part 12: the double-norm fix is in, and the trunk is still wrong - bisect it per block

glm5_full_run.py ran all 45 blocks with the double-norm removed and wrote the post-trunk hidden state
(last token, post output_norm - verified against the flag's own help text, so it is the same quantity the
reference's result_norm holds).  Compared against the reference:

    OLD (double norm)   corr +0.20802   max|d| 5.702e+00
    NEW (raw residual)  corr +0.07921   max|d| 1.349e+01
    argmax 421 vs expected 12089   MISMATCH

So the fix did not help; it made the end-to-end worse.  This CONTRADICTS the block-0 measurement, which is
direct and unambiguous (raw residual -> gate 0.88% off the reference; double normed -> 56.07%; the
driver's dumped gate was 56.1%).  Both statements are measured, so the honest reading is that they are
measuring different things: block 0 in the driver matches the reference with the raw residual, while the
45-block trunk was already wrong before this change and something else dominates it.  Do not treat the
end-to-end delta as a verdict on the normalization until the trunk is bisected - a broken trunk cannot
score a change.

WHAT IS MISSING, and it is now the only thing in the way: the reference dump carries l_out-N for every
block (90 of them, 45 blocks x 2 variants), but glm5_full_run.py dumps no per-block tensors, so the real
trunk has never been checked block by block.  The driver does dump per block, which is how block 0 was
localized, but it stops at block 2 ("block 3 is an MLA block; this driver only does KDA blocks so far").

Next: have glm5_full_run.py write l_out-{il} for every block (the same [n_embd, hc, T] transpose the
driver uses) and bisect against the reference's l_out-N.  That yields the FIRST wrong block of the real
trunk deterministically, the same way block 0's gate was pinned down.  Until that number exists, any
claim about which operator is wrong is inference, not measurement.

Also recorded from this run: clamp activity over 1890 calls was exp_up 1, shexp_up 10, everything else 0 -
consistent with the earlier note that the swiglu clamp barely fires on real data even though it is a
required term.

## Part 13: the front end is CORRECT and the first error is the KDA RECURRENCE

With the double-norm removed, the same block-0 chain that had been failing now reads:

    hc_init         0.000e+00   r +1.0000   AGREES
    hc_attn_pre-0   1.760e-05   r +1.0000   close
    attn_norm-0     1.121e-04   r +1.0000   close
    kda_gate-0      4.373e-02   r +1.0000   close      (was 56%: the double-norm, now 0.88%)
    kda_beta-0      6.887e-04   r +1.0000   close      (was 12%, now 0.07%)
    attn_output-0   2.187e-02   r +0.5267   DIVERGES   <-- the recurrence
    l_out-0         6.960e-02   r +0.4130   DIVERGES

So the fix in part 12 is vindicated on direct evidence (the gate and beta both drop to r +1.0000 and under
1%), and the FIRST genuine error is the KDA recurrence itself: attn = (S @ q) * scale, with S updated by
the delta rule.  The front end (conv1d+silu projections, the decay gate, beta, the L2 norms) is verified
against the reference; the state update or the read is not.

This also says something important about the KDA parity gate that passes 3/3: its fixtures were produced
from the same understanding of the recurrence as the kernel, so a formula error agrees with itself and the
gate cannot see it.  The authority for this operator is llama.cpp's ggml_compute_forward_kda in
ggml/src/ggml-cpu/ops.cpp (kda = (neg0 == S_v); decay S[i][j] *= exp(g[i]); delta = (v - S k) * beta;
S += delta (x) k; read AFTER the update, scale = 1/sqrt(S_v)).  Re-derive the oracle from THAT text, line
by line, and compare each term against the reference dump - do not re-derive it from our own kernel.

Suspects in order, all inside the recurrence: the decay's axis (per head i, not per channel j), whether the
read happens before or after the update, the scale (1/sqrt(head_dim) = 1/sqrt(128)), and the placement of
beta.  attn_output-0 is the pre-norm, pre-wo output, so the gated norm and wo are NOT yet implicated.

## Part 14: the RECURRENCE bug, found by reading llama.cpp's own loop

ggml/src/ggml-cpu/ops.cpp, ggml_compute_forward_kda's inner loop (the kda branch at :11002), with the state
stored transposed as `s_out[j*S_v + i] = S[i][j]`:

    decay    delta[i] = expf(g_d[i]);  for each row j: M[j][:] *= delta   ->  S[i][j] *= exp(g[i])
    pred     sum = dot(row j of M, k)                                     ->  sum_i S[i][j]*k[i]   indexed by j
    delta    delta[j] = (v_d[j] - sum) * beta_val                         ->  indexed by j
    update   M[j][:] += k_d * delta[j]                                    ->  S[i][j] += k[i]*delta[j]
    read     attn[j] = dot(row j of M, q) * scale                         ->  sum_i S[i][j]*q[i]   indexed by j

So in the state, i is the KEY axis (it pairs with k in pred and q in the read) and j is the VALUE axis (it
pairs with v and delta).  Our oracle's recurrence has them swapped:

    pred  = np.einsum("hij,hj->hi", S, k)          contracts k on the VALUE axis
    delta = (v[t] - pred) * beta[t][:, None]       indexed by i, not j
    S    += np.einsum("hi,hj->hij", delta, k)      k and delta swapped
    attn  = np.einsum("hij,hj->hi", S, q) * scale  contracts q on the VALUE axis

This is not a relabeling and cannot be one, because the decay ties the axes to specific tensors: g is a
per-key-channel quantity and llama.cpp decays i (the axis that pairs with k and q) while ours decays an
axis that pairs with nothing.  Decaying one axis while contracting the other is a different recurrence.

Corrected oracle (S[i][j] = key i, value j, matching the reference):

    S    *= np.exp(g[t])[:, :, None]                    # decay the key axis
    pred  = np.einsum("hij,hi->hj", S, k[t])            # contract the key axis with k, indexed by j
    delta = (v[t] - pred) * beta[t][:, None]            # indexed by the value axis
    S    += np.einsum("hi,hj->hij", k[t], delta)        # S[i][j] += k[i] * delta[j]
    attn[t] = np.einsum("hij,hi->hj", S, q[t]) * scale  # read AFTER the update, scale = 1/sqrt(S_v)

Note the decay line is unchanged - it already multiplies axis 1 of S, which is correct once the contractions
are fixed.  The C++ kernel in src/kernels/glm_kda.cpp mirrors the oracle and needs the same correction.

Why the parity gate passed 3/3: its fixtures were generated from this same oracle, so the wrong recurrence
agreed with itself.  Third instance of the pattern (swiglu clamp, router bias, and now this): a gate whose
inputs come from the implementation cannot detect a wrong implementation.

Verification after the fix: attn_output-0 must drop from r +0.5267 (2.187e-02 on a 0.0243 scale) to
agreement, and l_out-0 must follow.  That is the test; nothing less counts.
