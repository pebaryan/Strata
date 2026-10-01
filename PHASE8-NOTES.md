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

## Part 15: PART 14 IS WRONG - the axis swap is equivalent.  And the reference names attn_output elsewhere

Part 14 claimed the recurrence contracts the wrong axes and that this was the bug.  Applied and measured:

    kda_gate-0     4.373e-02   r +1.0000   unchanged (correct, as before)
    kda_beta-0     6.887e-04   r +1.0000   unchanged
    attn_output-0  2.201e-02   r +0.5207   was 2.187e-02 r +0.5267  -> NO CHANGE
    l_out-0        6.881e-02   r +0.4134   was 6.960e-02 r +0.4130  -> NO CHANGE

So the two index conventions produce the same recurrence and the swap is a relabeling after all.  Part 14
is retracted; the source-reading was careful but the conclusion was wrong, and one run settled it.

Two things follow, and the second is the real lead:

1. tools/glm5_kda_reference.py now has an inconsistency to resolve: line 213 still contracts the old way
   (`einsum("hij,hj->hi", S0, kk)`) in whatever helper that is.  Either make it consistent or establish that
   it is a genuinely different operator.  Left alone deliberately rather than blind-patched.

2. The bug for attn_output-0 is NOT in the recurrence's indexing.  Since the recurrence is validated by a
   dump we cannot see inside, the next thing to question is the COMPARISON ITSELF: attn_output-0 is
   assumed (asserted in glm5_block_driver.py's docstring, which is our annotation, not evidence) to be the
   pre-norm, pre-wo recurrence output.  If llama.cpp attaches the name "attn_output-%d" AFTER the gated
   RMS norm or after wo, then we have been comparing a recurrence output against a post-processed tensor -
   which would produce exactly this signature: the right magnitude, a partial correlation, and complete
   insensitivity to changes in the recurrence.

   Check the naming site in llama.cpp's graph construction for the kda path (llama-graph.cpp / the model's
   build functions) and see which tensor `cb(..., "attn_output-%d")` actually names, and in what order
   relative to the gated norm (ssm_norm / the sigmoid gate) and wo.

Also worth stating plainly: this is the fourth hypothesis this phase that measurement killed.  The lesson is
not "read more source" but "before comparing two tensors, establish by construction that they are the same
quantity" - the guard that worked for rms_norm and for the gate, and the one nobody has run for
attn_output-0.

## Part 16: we have been comparing the WRONG TENSOR - attn_output-0 is not the recurrence output

The reference's attn_output-0 has ne = (128, 64, 5) = [hd, nh, T]: a PER-HEAD tensor.  A post-wo value would
be [4096, 5], so attn_output-0 sits AFTER the within-head RMS norm and AFTER the sigmoid gate, but BEFORE
wo.  glm5_block_driver.py dumps kmid["attn"], the raw recurrence output, pre-norm and pre-gate, and calls it
attn_output-0.  Those are different quantities, and the docstring asserting otherwise was our annotation,
not evidence.

That single mismatch explains every observation that has been confusing this phase: the right magnitude (the
norm and gate are monotone-ish scalings), a partial correlation of about +0.52, and COMPLETE INSENSITIVITY
to changing the recurrence - including the axis swap in part 14, which measured identically because the
compared tensor is not the recurrence's output at all.

Correct next comparison: the oracle's `o` - rms_norm(attn) over the head dim, then the sigmoid gate, before
wo - reshaped to (hd, nh, T) and written as attn_output-N.  One line in the driver:

    write_tensor(dump, f"attn_output-{il}", np.transpose(o.reshape(T, NH, HD), (2, 1, 0)))

Then: match means the recurrence, the within-head norm and the gate are all correct and the error lives in
wo or later (l_out-0 is the unambiguous block output and it diverges, so something after this point is
still wrong); mismatch means the bug is inside the norm/gate or the recurrence.

THE RULE this phase keeps re-teaching, now four times over: before comparing two tensors, establish by
construction that they are the same quantity - the same guard that worked for rms_norm (verified to 6e-08)
and for the gate (verified to 0.88%).  A tensor's SHAPE is the cheapest such evidence and it was sitting in
the manifest the whole time: [hd, nh, T] cannot be a post-wo tensor.

## Part 17: part 16 is also refuted, and the recurrence is where the half-agreement lives

Part 16 argued attn_output-0 must be post-norm/post-gate because its shape is [hd, nh, T].  Measured by
dumping that value instead:

    kda_gate-0                4.373e-02   r +1.0000   (correct)
    kda_beta-0                6.887e-04   r +1.0000   (correct)
    attn_output-0 (raw rec.)  2.201e-02   r +0.5207
    attn_output-0 (post-gate) 5.228e-02   r +0.2431   <- WORSE
    l_out-0                   6.881e-02   r +0.4134

So post-processing it made the agreement worse, which means attn_output-0 is the RAW recurrence output after
all and the shape argument was wrong: a per-head tensor can be pre-norm, and [hd, nh, T] does not by itself
imply "after the gated norm".  Part 16 retracted as well.

What survives, and it is thin but measured:
  * Everything up to and including kda_beta is correct (r +1.0000 to under 1% error).
  * The recurrence output agrees only about half way (r +0.52), and neither candidate quantity matches.
  * The block output is wrong (r +0.41), unambiguously.
  * Changing the recurrence's contraction axes changes nothing (part 14), and post-processing the comparison
    makes it worse (this part) - so the error is inside the recurrence's arithmetic or its inputs.

THE NEXT TEST, and it is the one that separates the two remaining readings: run the oracle's recurrence with
g REPLACED by the reference's own dumped kda_gate-0 instead of the oracle-computed gate.

  * If attn then matches the reference, the recurrence code is right and the 0.88% gate error is amplified by
    the dynamics (exp of a log-decay, applied 5 times, with g near -5 - a 0.04 absolute error in g is a 4%
    compounding factor per step).
  * If it still disagrees, the recurrence's arithmetic or its other inputs (q/k/v, the L2 norms, the v-path
    silu, the scale) are wrong, and the search moves into those terms.

That is one small script and it is cheap.  It also removes the gate from the equation entirely, which is what
makes it decisive.

Standing rule, now with four retractions behind it: a hypothesis about THIS operator is only worth acting on
after the quantity being compared has been established by construction.  Parts 14, 16 and 9 were each
killed by one cheap measurement; the pattern is that source-reading and shape-reasoning both feel like
evidence and neither is.

## Part 18: attn_output-0 IS the recurrence output; the error is in q/k/v, not the recurrence's form

Measured, in this order:

1. WHICH INTERMEDIATE IS IT.  Correlating every oracle intermediate against the reference's attn_output-0:
   attn +0.52075 (best), v/vc +0.20, o +0.243, q -0.005, k +0.003, g -0.002.  So attn_output-0 IS the raw
   recurrence output.  Parts 14 and 16 were both wrong; the original annotation was right.

2. THE GATE AND BETA DO NOT MATTER.  Running the recurrence with the reference's own dumped gate and beta,
   in all four combinations, gives corr +0.52073..+0.52075 and 90.56% - identical to four decimals.  The
   gate's 0.88% error and beta's 0.07% are irrelevant to this discrepancy.

3. THE RECURRENCE'S FORM IS NOT THE DOMINANT ERROR.  Sixteen variants (decay in {exp(g), exp(-g), g, none}
   x read {after, before} x scale {1/sqrt(hd), 1}): the best is still the current formulation (exp(g),
   after, 1/sqrt(hd)) at +0.52073/90.56%.  Decisive detail: REMOVING THE DECAY ENTIRELY gives +0.49962 -
   almost the same.  The decay, the read order and the scale are all secondary; nothing in the recurrence's
   form explains a 90% magnitude error.

4. THE FRONT END IS NOT A SILU/L2 PERMUTATION.  Six variants (silu on v or not, L2 on q/k or q/k/v or q
   only, no silu) land between +0.34 and +0.54, worst errors 86-225%.

CONCLUSION: ~90% magnitude error with a +0.52 correlation, insensitive to the gate, the decay, the read order
and the scale, means the INPUTS q/k/v are wrong and wrong in a way that none of the six placements explains.
So the next thing to check is how the oracle OBTAINS them from the artifact: attn_q/attn_k/attn_v.weight are
Q8_0 (the pack's index shows type 8 for them), and the oracle dequantizes them through its own Model.tensor
path - the same path the parity gates validated against fixtures the oracle itself generated.  Verify the
oracle's dequantization of one of those matrices against an independent reader (gguf-py's own dequantizer),
the same way the pack's bytes were checked against the GGUF earlier.  If that dequantization is wrong, the
gate could never have seen it.

Standing rule, now with five falsified hypotheses behind it this phase: measure which quantity a dumped
tensor actually is BEFORE reasoning about it (done here, and it cleared two wrong conclusions), and prefer a
bounded variant grid with an unambiguous pass criterion (+1.0 correlation) over further source-reading.

## Part 19: end-to-end test on the output - built, and it FAILS

tools/glm5_e2e_test.py runs the whole output path: hidden state -> output.weight -> logits -> argmax, against
llama.cpp as the oracle.  Result:

    hidden state : corr +0.07921   max|d| 1.349e+01   (reference scale 9.262)
    reference    : argmax 12089  (golden 12089)  OK      <- the oracle side checks out
    ours         : argmax 421    corr +0.24460   relative err 87.6%
    the reference's own top token ranks 11705 in our logits; top-5 overlap 0/5
    E2E VERDICT: FAIL

The reference's own logits reproduce the golden token 12089 exactly, so the test harness, the logits
extraction and the output.weight orientation are all correct - what fails is the trunk that feeds them.

CONTRADICTION TO RESOLVE, and it is now the highest-value item: this run used the RAW residual for the KDA
(the part-12 fix), and its hidden state is WORSE than the pre-fix run (+0.079 vs +0.208).  Yet the isolated
block-0 front end measured CLEARLY better with raw (gate 56% -> 0.88%).  Both are measurements, so one of the
two call sites is not doing what I think.  The prime suspect is where `cur_raw = cur` is captured in
glm5_full_run.py's loop: if `cur` has already been normed by an earlier step at that point in the loop, or if
the loop reuses the variable afterwards, then the trunk is still getting a normed input (or now gets
something else) while the driver's single-block path does not.  Print cur vs cur_raw at the KDA call for
block 0 in the full run and compare against the driver's - that settles it in one run.

The e2e test is the right top-level gate from here: it is cheap, it fails loudly, and its oracle side is
verified against the golden tokens rather than assumed.  Chase every fix through it rather than through the
per-tensor bisect alone.

## Part 20: END TO END PASS - the trunk produces the reference's output

With the conv1d layout fixed (part 19's diagnosis, applied in load_weights), the full 45-block trunk run and
tools/glm5_e2e_test.py:

    hidden state : corr +0.99338   max|d| 6.438e-01   (reference scale 9.262)
    reference    : argmax 12089  (golden 12089)  OK
    ours         : argmax 12089   logits corr +0.99628   relative err 8.9%
    the reference's top token ranks 1 in our logits; top-5 overlap 5/5
    E2E VERDICT: PASS
      ours: [12089, 279, 825, 264, 7407]
      ref : [12089, 279, 825, 264, 7407]

Per-block l_out against the reference, same run: l_out-0 1.460e-03 r +0.9999, l_out-1 r +1.0000,
l_out-2 1.605e-03 r +1.0000, l_out-3 (MLA) 2.190e-02 r +0.9999, l_out-4/5 (MoE) 2.2e-02 r +0.9999 - so the
trunk is correct at every block, including the MLA and MoE blocks that had never been checked. What remains is
fp32-vs-quantized rounding compounding with depth (0.6-3% per block).

Clamp activity over this run: exp_gate 5, exp_up 3, shexp_gate 2, shexp_up 15 of 1890 calls.

WHAT THE BUG WAS: one reshape. The artifact stores ssm_conv1d_{q,k,v}.weight as (d_inner, 1, d_conv) with
d_conv fastest, so the (d_conv, d_inner) array the kernel wants is reshape(d_inner, d_conv).T; we loaded
reshape(d_conv, d_inner), interleaving four channels' taps. It survived a long investigation because the gate
and beta never touch the conv, so they measured perfect (r +1.0000) while q, k and v were scrambled - and
because every parity gate's fixtures were generated from the same broken loader.

REMAINING FOR A WORKING GLM-5.3 IN THE ENGINE (not the oracle):
  1. src/kernels/glm_kda.cpp - the same conv layout fix in C++.
  2. Regenerate the KDA fixtures and re-derive the parity gate from llama.cpp, not from our loader.
  3. Wire the engine path and chase every step through tools/glm5_e2e_test.py.
  4. Serving reality: ~75-85 GB resident vs 63 GB of VRAM+RAM on this box, so expect disk-streamed, low
     tokens/sec. Correctness is now established; that part is throughput.

## Part 21: the KDA parity gate re-verified against the corrected oracle - and the rule for the engine

Regenerating the fixture from the corrected oracle (tools/glm5_kda_reference.py --layer 4 --tokens 1
--raw-fixture ...) and running build-volta/glm_kda_parity on it gives 0 failures, with v/attn/gated/result
all at ~1e-06 relative and the invariants (gate in [-5,0], |q| per head = 1) holding.  So the C++ kernel's
conv1d_silu - which indexes conv_w[k*d_inner + ch], i.e. the (d_conv, d_inner) layout - was ALWAYS correct;
the fixtures fed it a transposed array.  The defect lived entirely in the Python loader, which is also why the
gate certified it: same loader, same wrong order, agreement at 1e-06.

THE RULE FOR THE ENGINE'S LOADER, to be applied when the engine path is wired: the artifact stores
ssm_conv1d_{q,k,v}.weight as (d_inner, 1, d_conv) with d_conv FASTEST.  The (d_conv, d_inner) array the kernel
expects is `reshape(d_inner, d_conv).T` - the transpose, not the reshape.  Get this wrong and q/k/v are
scrambled by interleaved taps, everything downstream is wrong, and every operator-level gate still passes.

REGRESSION TO ADD (not yet done, and it is the point of the whole lesson): the parity gate should be given a
fixture built with the WRONG order and must be shown to FAIL on it.  A gate that cannot reject the known-bad
input is not evidence.  This is the fourth time in this phase that a gate passed while the model was wrong
(swiglu clamp, router bias, the double norm, and now the conv layout), so the assertion belongs in the gate
itself rather than in a note.

Also worth regenerating for completeness: the t=8 and layer-20 fixtures, since they were built from the
pre-fix oracle.  The parity path is otherwise settled: kernel correct, oracle correct, agreement at 1e-06.

## Part 22: the KDA gate can now FAIL - known-bad input is rejected

The fixture writer gained a regression mode (STRATA_BAD_CONV_FIXTURE=1) that emits the (d_conv, d_inner) conv
array transposed AT THE BYTE-WRITING STEP ONLY, so the fixture holds a deliberately wrong conv layout while
its expected outputs were computed with the correct one.  Same gate, same kernel, same oracle:

    correct fixture   v/attn/gated/result all ~1e-06 of scale, 0 failures
    bad fixture       v 1.95x of scale (worst element 10x), attn/gated 1.00x, result 1.18x -> FAILURES

So the defect that survived this entire phase is now caught by the gate, and a PASS carries information.

The instruction that makes this repeatable: flipping the order in the LOADER (tried first) does not work,
because the oracle's own expected outputs are computed with the same weights - the fixture then holds wrong
weights AND wrong expected values, and the gate passes.  That wrong-against-wrong self-consistency is exactly
how the original bug survived four gates.  The flip must be at the writer.

FOUR GATES PASSED WHILE THE MODEL WAS WRONG IN THIS PHASE: the swiglu clamp, the router bias, the double
normalization, and the conv layout.  Every one had its inputs derived from the implementation under test.
Any new operator gate should ship with a known-bad fixture it is required to reject.

Remaining: regenerate the t=8 and layer-20 fixtures from the corrected oracle (t=1 is done and passes), then
wire the engine path with the loader rule reshape(d_inner, d_conv).T and chase it through glm5_e2e_test.py.
