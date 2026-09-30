# ADJUSTMENTS.md — the sm_70 Volta port of Strata

What this fork changes, why, and what the upstream merge on 2026-09-29 decided. Written for the next
session on this box; every number here was measured on this machine unless it says otherwise.

## The box (measured)

| Fact | Value |
|---|---|
| GPU | `Tesla PG503-216`, compute capability **7.0**, 32768 MiB, driver 580.173.02 |
| Toolkit | CUDA 12.9 (`/opt/cuda/bin/nvcc`) |
| Host compiler | `/usr/bin/gcc-14` (14.3.1) — the distro default (GCC 16) breaks nvcc's host pass |
| OS | CachyOS, kernel 7.1.8-1-cachyos |
| Served artifact | `strata-pack-coder` (256 experts/layer, pruned GSQ-RCO Coder) + IQ1_M GGUF shards |
| Notes | The board reports an OEM part number, not `Tesla V100-SXM2-32GB`; the die is the same V100 |

## Build

```bash
cmake -S . -B build-volta \
  -DSTRATA_EXPERIMENTAL_SM75=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DBUILD_TESTING=OFF \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-14 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-14 -DCMAKE_C_COMPILER=/usr/bin/gcc-14
ninja -C build-volta -k 30 strata        # the CLI and the server are the same binary
```

`-DSTRATA_EXPERIMENTAL_SM75=ON` is **required**: upstream's arch guard is
`if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM75)`, i.e. its own escape hatch for a sub-75 build —
after 0.1.29 upstream's floor is 7.5 (Turing), because below sm_80 the QSA scorer takes its portable
fp32-FMA fallback and the tensor-core prompt kernels refuse. The floor is reused rather than deleted,
so a normal build of this fork still refuses < 7.5.

Verify the result is really an sm_70 binary before believing a green build:

```bash
cuobjdump build-volta/strata | grep -m3 'arch = sm_70'
```

## What the port changes (by capability, not by file)

- **Architecture gate** — `STRATA_EXPERIMENTAL_SM75=ON` instead of a deleted floor (CMakeLists.txt).
- **Runtime device floor** — 7.0 instead of upstream's 7.5 (src/core/device.cu).
- **The pruned 256-expert artifact** — upstream's kernels pinned 512 experts into the warp layout.
  The count is now the caller's everywhere: the native router (`route<NEXP>`, 512 -> 16 values per
  lane, 256 -> 8) for both the single-row and batched calls, the prefill `route` dispatch
  (`route_kernel<REG>`), and the pack-geometry reader, which takes the count from the pack header.
- **`NE` in the prefill path** — sized buffers and host-side expert loops (`host_res[l * NE + e]`),
  which are allocated as `n_layers * n_expert`. Hardcoded 512, this indexes past its own arrays on a
  256-expert pack, so it resolves to the loaded layout's count (`#define NE ...`).
- **Geometry from the model, not a default** — the MoE shape is read from the model GGUF
  (`qwen4exp.expert_count` / `expert_used_count`) rather than assumed.
- **Tooling** — `tools/mtp_fetch.py` pins name resolution to IPv4 (this box has no working IPv6 route
  and huggingface.co publishes CloudFront AAAA records first, so urllib dies with `Errno 113` while
  curl survives); `tools/make_profile.py` takes the pruned model's layer/expert counts on the CLI.
- **Excluded, not ported** — `native_qsa_score.cu` needs `ldmatrix` (sm_75+) and tf32; it is excluded
  from the sm_70 build (CMakeLists.txt).

## Merge log: 2026-09-29 (engine 0.1.14 -> 0.1.24)

187 upstream commits, 13 conflicted files, 26 hunks. Local pre-merge state was tagged
`pre-sync-2026-09-29` and branched to `backup-pre-merge`.

Upstream landed equivalents of several local fixes, and **upstream's version won** where it was equal
or better — the fork should carry as little as possible:

- `src/platform/direct_file.cpp` — upstream's own thread pool (perf-review F-2) replaces the port's
  read pool. Same design; upstream's `io_threads()` also reads `STRATA_IO_THREADS` (clamped 1..64)
  and routes `wake()` through the completion queue like the Windows port. **Theirs, wholesale.**
- `src/prefill/kernels.cu` — upstream parameterized the warp layout as `template <int REG>` and added
  a generic top-k fallback. **Theirs, wholesale.**
- `include/strata/artifact/gguf_reader.hpp` — upstream's "0 = presence-only" guard scheme achieves the
  same thing more cleanly. **Theirs.**
- `tools/iq_pack.py` — upstream now derives the expert count from the router weight's own shape and
  validates every layer against it. **Theirs.**
- `src/program/generate.cpp` — upstream reads the MoE shape from the model GGUF; that supersedes
  taking it from the loaded layout. **Theirs.**

Kept, because upstream does not have it:

- `NE` resolving to the loaded layout in `src/prefill/prefill.cpp` (upstream still hardcodes 512 there).
- The count-aware native-router gate in `src/core/layer.cpp`.
- `--layers` on `tools/make_profile.py` (upstream added only `--n-expert`; the profile's layer count is
  a module constant upstream).
- The dual pack-header spelling in `src/kernels/cpu/expert_layout.cpp`: upstream parses `(n_expert N`
  (v3 packs), this fork's packs write `n_expert N`. Both are accepted.

Extended rather than duplicated:

- `native_router_top10_multi` — upstream's batched window validated `[n,512]` and launched the 512
  kernel; it now takes the expert count like the single-row call, so a 256-expert pack gets the same
  batched decode. `src/core/verify.cpp` passes the count and its gate accepts 512 or 256.

### The one that nearly slipped through

Git resolved `src/prefill/kernels.cu` such that two **local-only** lines (`static_assert(NEXP % 32 ...)`
and `constexpr int VEC = NEXP / 32;`) ended up *outside* the conflict markers, after upstream's
`template <int REG>` had replaced `NEXP`. A merge can therefore be marker-free and still not compile —
or, worse, compile and mean something else. Marker counts are not verification; the build is.

## Merge log: 2026-09-30 (engine 0.1.24 -> 0.1.29)

67 upstream commits (tags v0.1.25 … v0.1.29), 97 files, **two** conflicts. Pre-merge state tagged
`pre-sync-2026-09-30` and branched to `backup-pre-sync-2026-09-30`.

Upstream headline: sampled answers up to 40% faster with the same tokens (#197), faster prompt kernels
(#187, #188, #207), correctness fixes (#154); a fatal CUDA fault in the prompt path now exits instead of
hanging (#224); CMake warns when sm_120 is built with CUDA < 13.0 (#220); `iq_pack` writes
`native_experts.txt` under a temporary name so an interrupted pack leaves nothing partial (#172).

Both conflicts were the arch floor, changed on *both* sides at once:

- `CMakeLists.txt` — upstream lowered its floor 80 -> 75 (Turing). Resolution keeps the port's hatch and
  re-points it at the new floor: `if(_base LESS 75 AND NOT STRATA_EXPERIMENTAL_SM75)`.
- `src/core/device.cu` — took upstream's rewritten runtime check **including its new HIP/gfx1100 branch**,
  lowered the CUDA minor 75 -> 70 for the port, and kept an `sm_7x` warning because upstream validates
  7.5 and newer: this port's speed and numerics are the port's own.

### Post-merge verification (2026-09-30, measured)

- Merge commit `f880896`; `git rev-list --count` says **11 ahead / 0 behind** `origin/main`.
- No conflict markers; the port's markers survive (`LOCAL VOLTA PORT` x4, `LOCAL PRUNED-MODEL SUPPORT` x11);
  the router stride fix, the count-driven router, `enable_testing()` and the sm75 hatch are all present.
- Build: `190/190` targets, 0 errors, `Result=success`; version line now `0.1.29`.
- Suite: `ctest -j1` → 28 tests, 25 pass. The three failures are (a) `ple_parity`, which needs a PLE table
  absent from this checkout, (b) `platform_memory_test`, which needs a raised `ulimit -l`, and (c)
  `kv_hybrid_parity`'s `qsa_prompt_attn` check — see below.
- `kv_hybrid_parity` [5/5]: the tensor-core prompt path refuses cc < 8 **by design**
  (`src/kernels/cuda/qsa_prompt_attn.cu`: `if (cc_major[dev] < 8) return false;`), so on this card the old
  prompt kernel runs and is still checked against the dequant reference (checks [1-4] pass). Upstream's test
  carved out HIP but not sub-sm_80, so the port adds the same carve-out for `cc_major < 8` — with the FAIL
  branch kept for sm_80+, where a refusal would be a real regression. Closing this gap properly means writing
  the Volta fp16-mma prompt scorer recorded under "Excluded, not ported".

## Post-merge verification (2026-09-29, engine 0.1.24)

- Build: `120/120` targets, 0 errors, `Result=success`.
- Artifact: `build-volta/strata`, 36,575,128 bytes, `cuobjdump` reports `arch = sm_70`.
- Load + generate on the pruned pack (`--tokens 1,2,3 --max-new 8 --greedy`, 256-expert pack,
  `--expert-cache auto`, `--kv q4_0`, `--mtp`, `--spec 4`): exit 0, engine reports `session is up
  (engine 0.1.24)`, output `61 5 31 0 4 61946 198 55742`.
- Throughput from that run is a micro-test, not a figure to quote: 8 tokens in 341.2 ms (23.45 tok/s)
  including speculation warm-up, prefill 2 tokens. The served configuration's numbers are the ones in
  `bench/results/`.

## Not changed

The engine's algorithm, its quantization formats, and every number under `bench/results/` (vendor
data, produced on other hardware). Nothing here is presented as the validated sm_80+ path: sm_70 has
no tf32 mma, no `ldmatrix` and no bf16 tensor-core math, so this port's speed and numerics are its own
and are measured, never assumed.
