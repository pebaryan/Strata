# Kolibri-1 on the V100: what was actually slow, and the rules that came out of it

Written after a session that started with "why is it so slow compared to llama.cpp" and ended with
decode at 47.6 tok/s on a short generation and 26.9 tok/s sustained, parity unchanged. Everything
below is a measurement made on this box, with the commands to reproduce it. The wrong turns are
included because they are the more useful half.

## The ladder

Same pack, same GGUF, same card, `--prefill` then `--gen`, 24-31 GB row budget:

| state | decode | notes |
|---|---|---|
| starting point, 8 GB cache | 0.86 tok/s | contention: llama-server held the card |
| starting point, 24 GB cache, cold | 2.91 tok/s | 179 MB per token from the SSD |
| after the expert-row fix | 15.8 tok/s (63 ms/token) | no more host copies on cache hits |
| after the router rewrite | 33.8 tok/s (30 ms/token) | short generation, warm cache |
| after `pread` row assembly | 22.2 tok/s (45 ms/token) | 128-token decode, misses now ~5 ms not 70 ms |
| with 24 layers pinned | 26.9 tok/s (37 ms/token) | 20.4 GB pinned + 3.8 GB LRU resident |
| with 30 layers pinned | 47.6 tok/s (21 ms/token) | gen 16; collapses to 5.7 tok/s over 128 tokens |
| llama.cpp `-ngl 99 -ncmoe 18` | ~50 tok/s | 31.4 GB VRAM **plus** 17.2 GB RAM, 18 layers on the CPU |

Where the time went at each stage, from `STRATA_KOLIBRI_PROFILE=1`:

    start   pre+sync 48.3 ms | experts 13.9 | post 0.3 | head 0.5   = 125 ms
    end     pre+sync  7.0 ms | experts 13.2 | post 0.2 | head 0.5   =  21 ms

## Defect 1: expert rows were copied on every lookup, including cache hits

`run_experts` called `src.blob(layer, expert)` before consulting the row cache, and in gguf mode
(a pack with no `experts.bin`) that call assembles the row from three non-adjacent slices. So a warm
token copied ~765 MB (300 rows) of host bytes for rows already resident in VRAM.

Fix: consult the cache first and fetch host bytes only on a miss.

## Defect 2: the staging pool was zero bytes, and every miss wrote 2.55 MB into it

`stage_blob_` is sized from `layer_blob_bytes_` (expert_source.cpp:706), which nothing fills for a
pack with no `experts.bin`. The staging buffers were therefore allocated with size 0, and
`copy_from_files` wrote a full row into them. A heap overflow that had been present all along; it
only became fatal when unrelated heap changes in this session moved the arena, which is why an
instrumented build started segfaulting mid-session with the stack pointing at `claim_stage`.

Fix: the serving path assembles rows itself, from its own read-only mmap, and no longer depends on
that pool at all. Root cause worth keeping: *a size derived from an empty container is zero, not an
error*; the code had no assertion that the buffer was at least one row.

## Defect 3: the router was 68 percent of decode

`route_k` ran as a single block of 384 threads: each thread reduced one 2560-element bf16 row
serially, and thread 0 ran the top-6 selection as a serial scan of 384 x 6 entries. Measured 42.86 ms
per token, 0.86 ms per layer, to multiply a 384x2560 matrix. The whole decode was 63 ms.

Fix: `route_dots_k` (one block per expert, block reduction, a real GEMV) plus `route_topk_k` (a
parallel reduction). The selection rule is a total order - ranked descending, lower expert index on
a tie - so a tree reduction returns exactly the ids the serial scan returned; the parity gate
confirmed it. Decode went 63 ms to 30 ms.

## Defect 4: a miss cost 70 ms because the rows came through page faults

Rows were assembled by `memcpy` out of a file-backed mmap: ~640 minor page faults per 2.4 MB row,
each a 4 KB synchronous read of a DRAM-less SATA SSD. Measured 70 ms per row, 34 MB/s, and a
64-token generation that misses ~1.5 rows per token therefore ran at 135 ms/token *no matter how
fast the hits were*. This is the defect that made every residency experiment look pointless.

Fix: `pread` the three slices into a pinned staging buffer. 136 ms/token to 45 ms/token at the same
cache budget.

## Defect 5: the device was 80 percent idle

The engine held 4.58 GB of a 32 GB card after a 52-token run, because an LRU only stores what
happens to be touched. llama.cpp fills 31.4 GB. Idle VRAM is capacity that could be holding experts.

Fix: `--pin-layers N` allocates one block per layer holding all 384 rows, and a pinned layer
resolves a row by address arithmetic: no lookup, no host bytes. Measured on a 30-31 GB budget over a
128-token decode: pin 0 = 45 ms/token, pin 24 = 37 ms/token, pin 30 = 21 ms/token at gen 16 but
176 ms/token sustained, pin 32 = out of memory (27.0 GB pinned + 4 GB LRU + trunk exceeds 32.8 GB).

The pin-30 collapse is I/O, not compute, and it is instructive: `pre+sync` is 7.0 ms at gen 16 and
151.7 ms at gen 128 under the identical configuration, because the remaining misses on the unpinned
layers stall the host thread and the cost lands in the *next* layer's sync bucket. More pinning
cannot fix it; it hits the VRAM wall at 32 layers.

## Three hypotheses that were wrong, and what killed them

1. **"It is disk-bound, 179 MB per token at 520 MB/s."** Wrong for the warm case: at a 91 percent
   hit rate with 200 misses across a whole 64-token decode, decode was still 339 ms/token. A hit-rate
   measurement is what disproved it.

2. **"The per-layer host round trips are the bottleneck: 100 GPU pipeline drains per token."**
   Disproved by a micro-benchmark (`~/.hermes/cache/scratch/drain_cost.cu`): 750 launches plus 100
   syncs is 1.64 ms/token, 16 us per drain, and a D2H+H2D+sync round trip is 2.5 us. 100 drains
   cannot explain 125 ms. *Estimate the primitive before blaming it.*

3. **"The expert mmvq kernels are free"**, from a skip-kernels A/B that showed 125 vs 127 ms. Invalid:
   skipping the expert kernels changes the hidden states, therefore the routing, therefore which rows
   are fetched (1,990 rows versus 1,012 in the two runs). It compared two different workloads.

The pattern in all three: an inference that sounded right, presented before the instrument existed.
The per-op timers (`STRATA_KOLIBRI_PROFILE=1`, `STRATA_KOLIBRI_NO_GRAPH=1`) are what finally named
the router in one run after two wrong reports.

## Method that worked

- **Attribute per operation, not per phase.** Phase timers said "experts 71 ms"; the per-op table
  said `p_route 42.86` and settled it.
- **Report prefill and decode separately.** Averaging them produced a wrong reading early on; the
  same counters now reset at the boundary and print twice.
- **A counter that moves with generation length points at host I/O**, because a host stall in one
  phase is attributed to the next phase's sync.
- **A/B only what you can hold constant.** Prefer an A/B inside one process, or an instrument that
  does not change the workload at all.
- **Report the residency split with every rate.** "24 GB budget" is meaningless if 4.6 GB is resident
  and the rest streams from disk.

## Still open

- **No host-RAM tier.** This is the remaining structural gap and the reason llama.cpp does not
  degrade on long generations: its 18 non-VRAM layers live in RAM and are computed on the CPU, so
  they never touch the SSD. A VRAM-only policy degrades as the sequence grows. The engine has the
  pieces (`HostRowCache`, the CPU expert path, `std::async` overlap in the GLM route) and the
  Kolibri path uses none of them.
- **Prefill is unbatched.** The serve path calls `run_token` once per prompt token, so prefill runs
  at decode speed: ~25-30 tok/s against llama.cpp's 340 tok/s on pp2048. Any prefill benchmark will
  show this, and it is a design gap, not a tuning issue.
- **The per-layer split at the routing** still costs a host round trip per layer, and the expert
  chain is 6 sequential 4-kernel chains where the engine already has
  `glm_expert_moe_device_resident` for one grouped call over all selected rows.
- **Pin-30 sustained behavior** is understood qualitatively (I/O tail attributable to host stalls)
  but not fixed.
