# The phase-8 ground truth

Produced by `tools/glm5_ref_dump.cpp`, which runs the FORK's own GLM5-Next graph (the reference build) with
an eval callback and writes every tensor the graph names.  The .bin payloads (116 MB) live in
`/home/peb/moredata/glm5-ref-dump/` and are regenerable; what is committed here is the small part needed to
identify a run:

  tokens.txt   the golden greedy generation for the fixed prompt below
  dump.tsv     the manifest: every tensor's name, shape, type and payload file, in EVALUATION ORDER

## The fixed prompt

  "The capital of France is"   ->  tokens 785 6722 315 9621 374
  generation (greedy, 4):      12089 13 1084 374

Any port run must reproduce that sequence, and its `result_output` (154880 logits) must match the
reference's for the last prompt token.

## Regenerating

  cd /home/peb/code/Strata
  g++-14 -O2 -std=c++17 -I/home/peb/llama.cpp-glm5/include -I/home/peb/llama.cpp-glm5/ggml/include \
      tools/glm5_ref_dump.cpp -o build-volta/glm5_ref_dump \
      -L/home/peb/llama.cpp-glm5/build-glm5/bin -lllama -lggml -lggml-base \
      -Wl,-rpath,/home/peb/llama.cpp-glm5/build-glm5/bin
  ./build-volta/glm5_ref_dump \
      /home/peb/moredata/models/glm-5.3-flash-gguf/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf \
      /home/peb/moredata/glm5-ref-dump "The capital of France is" 4 46

The last argument is the number of layers whose expert weights stay on the CPU, which is what makes an
86 GB model run on a 32 GB card; it matches the CLI's `-ncmoe` exactly (one tensor-buft override per layer
with the pattern `blk\.N\.ffn_(up|down|gate|gate_up)_(ch|)exps` on the CPU buffer type).

## Comparing a port

  python tools/glm5_compare_dump.py --ref /home/peb/moredata/glm5-ref-dump --port <port dump dir>

It reports the first tensor in evaluation order whose error exceeds the tolerance, so a mismatch
localizes to a layer and a site rather than to "the logits differ".
