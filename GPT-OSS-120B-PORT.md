# GPT-OSS-120B on Strata — port plan

Started 2026-10-06 on branch `gpt-oss-120b-port`, based on upstream Strata
`82f46a8`. This port starts from llama.cpp's existing GPT-OSS graph and ggml
operations. Model math and quantized matrix operations must remain ggml/llama.cpp
operations unless profiling later proves a specific operation is the bottleneck.

## Rule for this port

The pinned llama.cpp revision is `3cf03257f219afbe7334045ff7c6a06ac68c627d`,
the revision Strata already uses for ggml. Its `src/models/openai-moe.cpp`
implements the GPT-OSS graph, including attention sinks, sliding-window layers,
expert biases, and the GPT-OSS MoE activation and routing. We will first preserve
that graph and operation ordering, and wrap it in the smallest Strata integration
that can run and be compared end to end. Replacing a ggml operation with Strata
CUDA/HIP code is out of scope until profiling demonstrates a bottleneck and a
parity gate demonstrates equivalence.

## Model contract to verify against the GGUF

The upstream `openai/gpt-oss-120b` config reports:

- 36 transformer blocks, hidden width 2,880, vocabulary 201,088.
- 64 query heads, 8 key/value heads, head width 64; grouped-query attention.
- Alternating sliding-window and full-attention layers; sliding window 128.
- RoPE with base 150,000 and YaRN scaling factor 32 (original context 4,096;
  model context 131,072).
- 128 routed experts, top 4 per token, expert intermediate width 2,880.
- SwiGLU with the model's 7.0 clamp; attention and expert projections include
  biases, and each attention layer has learned attention sinks.
- The official GGUF inference weights use MXFP4 for the large expert matrices;
  the tokenizer, dense tensors, and exact tensor types must be read from the
  selected GGUF rather than assumed from the source checkpoint.

The llama.cpp graph is the semantic authority for the port. In particular it
uses RMSNorm, Q/K RoPE, sink-aware attention, an attention residual followed by
`attn_post_norm`, softmax-weight routing, the GPT-OSS SwiGLU operation, expert
projection biases, then the FFN residual. These details are not to be recreated
from a paper or inferred from tensor names.

## Phases and acceptance gates

| Phase | Work | Gate |
|---|---|---|
| 0 | Record pinned llama.cpp revision and model contract | GGUF metadata and tensor inventory agree with this document; unsupported variants refuse with a specific reason |
| 1 | Establish an unmodified llama.cpp/ggml GPT-OSS-120B baseline | Same GGUF, prompt token IDs, context, KV type, and greedy settings; save logits/tokens and throughput separately for prompt and decode |
| 2 | Add a Strata adapter that calls the existing ggml operations and retains GGUF/MXFP4 tensors | Single-op tests use ggml references; no copied or new model kernels |
| 3 | Match one full block, including alternating attention-window behavior | Per-stage hidden states and routed expert IDs match the pinned llama.cpp path within documented tolerances |
| 4 | Match full-depth teacher-forced logits and greedy generation | Fixed prompts produce identical greedy tokens; near-tie cases additionally compare logits/margins |
| 5 | Integrate Strata memory management and serving around the verified graph | Correct output under the selected target's memory budget; no performance claim without same-machine A/B measurements |
| 6 | Profile before any kernel specialization | Each proposed replacement names the measured bottleneck, carries a parity test, and beats the baseline on the target hardware |

## Local artifact and target

The available artifact is `D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf` (63,387,346,208 bytes). Strata's
header reader confirms GGUF v3, `general.architecture = gpt-oss`, 36 blocks, context 131,072, hidden 2,880,
64 query heads / 8 KV heads, YaRN factor 32, 128 experts with top-4 routing, expert width 2,880, and
window 128. It contains 687 tensors: 108 MXFP4, 146 Q8_0, and 433 F32. All tensor byte counts are
consistent with their declared shape/type; each layer has the expected expert gate/up/down matrices and
biases, router+bias, attention projections+biases, sink vector, and post-attention norm.

The live machine probe found two RTX 5060 Ti devices (16,283 MiB available to CUDA each at test time) and 63.9 GiB system RAM. CUDA
13.3 and MSVC 19.44 are installed. The previous RTX 5070/12 GB figures elsewhere in Strata's benchmark
docs refer to a different PC and are not this port's target.

## First Strata API serving path

The resident `strata_gpt_oss --serve` adapter now speaks Strata's existing line
protocol. `serve/server.py` remains the HTTP/API process; it starts one resident
GPT-OSS engine, passes token IDs, and streams token IDs back. The adapter keeps
the model loaded between requests, clears per-request KV state, supports STOP,
and uses the pinned llama.cpp graph and CUDA kernels. The server config can set
`temperature`, `top_k`, `top_p`, `min_p`, repetition/frequency/presence penalties,
and `seed`; it does not enable batching, disk sessions, vision, or tool calling
for this adapter yet.

Prompt prefill is split into chunks no larger than llama.cpp's `n_batch`, while
still using its ordinary graph and kernels. This avoids the former hard failure
when a request prompt exceeded the runtime's 2,048-token batch limit. The
one-shot runner uses the same chunking. A 2,281-token API prompt at 4,096
context returned the expected answer; the same long prompt also completed in
the one-shot runner. Longer prompts remain bounded by context minus the
requested output tokens.

GPT-OSS declares the `gpt-4o` tokenizer pre-tokenizer, not Qwen35. The exporter
and Strata API tokenizer now select the pinned llama.cpp GPT4O pattern by the
GGUF `tokenizer.ggml.pre` value. The extracted tokenizer passed its round-trip
corpus, and on the same fully rendered GPT-OSS prompt Python and llama.cpp
produced exactly the same 81 token IDs. The server strips GPT-OSS channel and
message boundaries into reasoning/content events and recognizes `<|return|>`
as the turn-ending token.

For a local build, configure with `-DSTRATA_GPT_OSS_RUNTIME=ON` and the existing
CUDA/native-experts options, then build `strata_gpt_oss`. Extract the GGUF's
tokenizer and template:

```powershell
python tools/strata_tokenizer.py `
  --gguf D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf `
  --out build-gptoss\gptoss-serve --check
```

Use a Strata server config with the equivalent of these fields (adjust absolute
paths for the checkout and model):

```json
{
  "exe": "D:/code/Strata-gpt-oss-120b/build-gptoss/strata_gpt_oss.exe",
  "args": ["--model", "D:/aimodels/gpt-oss/gpt-oss-120b-MXFP4.gguf",
           "--gpu-layers", "17", "--context", "4096", "--ubatch", "1024",
           "--tensor-split", "0.42,0.58"],
  "cwd": "D:/code/Strata-gpt-oss-120b",
  "backend": "cuda",
  "gpu": [0, 1],
  "tokenizer": "D:/code/Strata-gpt-oss-120b/build-gptoss/gptoss-serve/tokenizer",
  "model_name": "gpt-oss-120b"
}
```

Start only on loopback unless an API key is configured:

```powershell
python -m serve.server --engine strata `
  --config build-gptoss\gptoss-serve\strata.json --host 127.0.0.1 --port 8095
```

The local service passed `/health`, OpenAI Chat Completions, and Anthropic
Messages smoke tests. Both API dialects returned text `2` and routed the
reasoning separately for “What is 1 + 1? Answer with one digit only.” A repeat
OpenAI check at 17 layers, 4,096 context, and microbatch 64 used an 81-token
prompt, generated 48 tokens, and reported 3.13 s prompt evaluation and 4.33 s
decode evaluation (11.1 generated tokens/s). This is a single smoke test, not a
speed or answer-quality claim. Multi-turn requests are accepted as a complete
prompt each time; the adapter currently clears KV state rather than reusing a
conversation cache.

## Current status

Phase 0 is passed for the local GGUF. The pinned llama.cpp/ggml CUDA baseline and
the Strata-linked `strata_gpt_oss` runner both build for `sm_120`. The runner uses
the pinned llama.cpp C API for model loading, chat formatting, tokenization, graph
construction, sampling, and execution; it adds no model kernels. With the same
model, 20-token formatted prompt, greedy sampling, context 512, microbatch 128,
and 17 GPU layers split 42:58, the runner and upstream CLI in `--no-jinja` mode
emitted the same 32 special-token pieces. The runner logged all generated token
IDs. The default Jinja CLI mode is not a parity target for this C API harness,
because it formats the assistant prefix differently.

The initial MXFP4 operator parity gate now runs one real 2,880×2,880 routed-expert
matrix from the GGUF through `ggml_mul_mat` on CPU and CUDA0. Layer 0/expert 0
measured max absolute error `2.38e-7` and relative L2 `8.62e-8`; layer 17/expert 7
measured `4.77e-7` and `8.72e-8`. The harness fails above `1e-4` max absolute
or `1e-5` relative L2. This verifies only this MXFP4 matrix operation and its
GGUF slice/layout; it does not establish full-block, expert-routing, or full-model
parity. Run it with `build-gptoss\strata_gpt_oss_mxfp4_parity.exe
D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf [layer] [expert]`.

The runtime also has an opt-in reference capture for a selected graph block. It
records the formatted prompt token IDs, attention-normalized input, Q/K/V projections and Q/K RoPE outputs,
attention output, post-attention norm, top-4 expert IDs, normalized expert
weights, MoE output, and post-residual block output from llama.cpp's own named
graph checkpoints. Captures use a binary header with GGML type, dimensions, byte
count, and strides, followed by the tensor storage bytes;
normal runs do not enable the evaluation callback. For example, add
`--dump-block 0 --dump-prefix build-gptoss\block0ref` to the runner command above.
A local capture of the 20-token prompt produced finite F32 `2880×20` tensors at
the hidden-state checkpoints. It is reference instrumentation; normal runs do not
enable the callback.

The first isolated MoE branch comparison now passes for prompt tokens 0, 5, and
19. A Strata-side ggml graph uses the captured normalized activations, expert IDs,
and weights, plus the model's expert matrices/biases, and compares its reduction
to the captured `ffn_moe_out`. Maximum relative L2 across those three tokens was
`8.61e-8` (maximum absolute error `1.91e-6`). For these same three positions, a
separate ggml router graph reproduced the captured top-4 expert IDs exactly and
the normalized weights with zero measured absolute error. The graph uses pinned
ggml's `mul_mat`, `argsort_top_k`, `get_rows`, `soft_max`, `mul_mat_id`, `add_id`,
and `swiglu_oai`; it adds no replacement kernels. In the composed run below, the
expert branch instead consumes independently recomputed routing from the candidate
activation.
Reproduce with `build-gptoss\strata_gpt_oss_moe_parity.exe MODEL.gguf
CAPTURE_PREFIX TOKEN_INDEX` after making the block-0 capture described above.

Block-0 Q/K/V projections and Q/K RoPE outputs match the pinned reference exactly
for the captured 20-token prompt. A decomposed causal-attention graph then applies
the 64-to-8 grouped-query mapping, F16 K/V, causal mask, learned attention sinks,
softmax, and output projection using stock ggml operations. Its final attention
output differs from llama.cpp's fused path by max absolute `0.021224` and relative
L2 `0.00155346` on this prompt; the diagnostic gate is `0.05` and `0.003` to allow
for the fused/decomposed reduction difference. This is one reference prompt, not
a general error bound.

The components are now composed into the complete first transformer block using
the model's token embeddings, both RMSNorms, attention residual, attention sinks,
router, expert FFN, and final residual. Across all 20 tokens in this prompt, the
candidate router selected the same top-4 experts as llama.cpp for every token.
The largest candidate-vs-reference top-4 weight error was `3.51e-4`; the largest
MoE-output relative L2 was `0.00611`. After the residual, the largest full-block
output relative L2 was `0.0020464` (max absolute error `0.114307`). The diagnostic
gates are `0.0025` relative L2 and `0.15` max absolute for block output. Thus this
single block and prompt pass the current gate, with a visible but bounded
fused-versus-decomposed attention difference. It does not establish accuracy on
other prompts or parity through all 36 layers.

A second 20-token prompt, “Explain why the sky looks blue in one sentence. Keep
it simple.”, also passes the same block-0 checks at all 20 positions. Its
decomposed attention output differs by max absolute `0.023993` and relative L2
`0.001577`; the full block's worst relative L2 is `0.00204637` and worst max
absolute error is `0.117939`. Router top-4 IDs match at all positions, with
maximum weight error `4.52e-4`. This adds a second prompt to the diagnostic sample,
not full-depth or production validation.

Reproduce after capturing the block:

```powershell
build-gptoss\strata_gpt_oss_attention_parity.exe `
  D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf build-gptoss\block0ref
for ($i = 0; $i -lt 20; $i++) {
  build-gptoss\strata_gpt_oss_moe_parity.exe `
    D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf build-gptoss\block0ref $i `
    build-gptoss\block0ref.candidate.attn_post_norm-0.bin
}
```

Run
`build-gptoss\strata_gpt_oss_attention_parity.exe MODEL.gguf CAPTURE_PREFIX`
after making the same capture.

The earlier 17-layer configuration loaded the GGUF with 33,512 MiB in the CPU
mapped buffer, 13,170 MiB on CUDA0, and 13,757 MiB on CUDA1. It generated 32
tokens successfully. The runner's synchronized llama.cpp counters report 15,915
ms for 20 prompt tokens and 7,095 ms for 31 subsequent decode runs (about 1.26
prompt tokens/s and 4.37 decode runs/s). Under the same settings, the upstream
CLI reported 1.5 prompt tokens/s and 1.9 generation tokens/s. The output pieces
match, but the throughput summaries do not; the timing discrepancy is unresolved,
so there is no speed claim. The test response itself is not a quality evaluation.

The automatic split succeeds at 16 GPU layers, but fails during compute-buffer
allocation at 17. Explicit `--tensor-split 0.42,0.58` makes 17 layers work. This
setting passed direct-run tests at context 4,096 and 8,192 with microbatch 64;
the 8,192 test evaluated a 20-token prompt in 15.74 s and one decode step in
0.329 s. The 4,096 setting also passed a full Strata API smoke test and is the
documented default to leave more memory headroom. At 18 layers, balanced split
`0.5,0.5` and microbatch 64 worked at context 512 and a 32-token direct-run test
reported 5.12 s over 31 decode runs (6.05 runs/s), versus 7.28 s (4.26 runs/s)
at 17 layers with the same balanced split. These are single runs, not a reliable
performance comparison. At 18 layers, context 4,096 failed during CUDA
workspace setup; at 19 layers and balanced split, weight allocation failed on
CUDA0 requesting 16,462 MiB when 16,311 MiB was available. Leave VRAM headroom
for other applications; these configurations are not guarantees under GPU load.

A repeated OpenAI API check at 17 layers, context 4,096, microbatch 64, and
split `0.42,0.58` returned “Jupiter” on all three requests. The first request
reported 28.8 s prompt and 5.45 s decode evaluation; the next two reported
3.29/3.02 s prompt and 3.39/3.31 s decode evaluation, or 12.1/12.4 generated
tokens/s. The large first-request outlier means startup/warm-up and steady-state
numbers must be reported separately; this three-request sample is not a general
benchmark. Changing only the split to `0.5,0.5` at the same 17-layer/4,096/
64-token-batch setting repeatedly failed when CUDA0 could not allocate its
cuBLAS workspace. Keep the asymmetric split for this context on this machine.

With the same 741-token prompt and 17-layer `0.42,0.58` split, increasing
`--ubatch` from 128 to 256, 512, and 1,024 reduced the second-request prompt
time from 10.68 s to 6.82 s, 5.78 s, and 3.77 s, respectively. Decode was
8.8–9.2 generated tokens/s in these single steady-state samples. At 1,024, both
4,096 and 8,192 context served the request successfully; at 8,192, the second
request took 4.80 s for the 741-token prompt and 5.32 s to decode 47 tokens.
That 8,192-context process used about 14,973 MiB and 14,955 MiB on the two
16,311-MiB cards, so leave the safer 4,096 context as the default. Use
`--ubatch 1024` to improve long-prompt prefill on this setup, but benchmark your
own prompt mix; these are small single-machine samples, not a general speed
claim. Greedy runs returned the same short answer while generating different
reasoning lengths, so this is not a token-for-token parity result.

Reproduce the working runner configuration after building with
`STRATA_GPT_OSS_RUNTIME=ON` from a Visual Studio developer prompt:

```powershell
build-gptoss\strata_gpt_oss.exe `
  --model D:\aimodels\gpt-oss\gpt-oss-120b-MXFP4.gguf `
  --prompt "What is 1 + 1? Answer with one digit only." `
  --tokens 32 --gpu-layers 17 --context 4096 --ubatch 1024 `
  --tensor-split 0.42,0.58
```

The combined nominal memory is enough to investigate a 59-GiB MXFP4 artifact,
but not enough to assume the full model plus KV/cache/work buffers will be
resident. This is a two-GPU/PCIe port; do not assume peer-to-peer or unified
memory. Phase 1 remains open: capture baseline token IDs/logits directly over
representative prompts and investigate the throughput-counter discrepancy, then
measure actual allocation and usable context before making a fit claim. The
runner currently accepts one user message; multi-turn history and tool calls
remain pending.

## References used for the contract

- OpenAI model config: `https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json`
- Pinned graph: `https://github.com/ggml-org/llama.cpp/blob/3cf03257f219afbe7334045ff7c6a06ac68c627d/src/models/openai-moe.cpp`
- Pinned MXFP4 block and ggml implementation: `third_party/ggml/ggml-common.h` and the matching llama.cpp `ggml` sources.
