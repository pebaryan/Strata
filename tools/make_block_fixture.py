#!/usr/bin/env python3
"""Collect the fixture a whole-block run needs, for one block.

    tools/make_block_fixture.py <block> <outdir>

The block loop (glm_block_forward) needs fifteen weight arrays plus the block's input and, for a first run, an
expected value per stage.  Most of that already exists on disk:

    the 16 KDA weights          glm5-kda-l{0}-t1/w_*.bin        (from the oracle's raw fixture)
    the FFN's three matrices    glm5-ffn-l0/raw_*.bin           (raw blocks, dequantized by the gate)
    every expected stage        glm5-ref-dump5/<family>-0.bin   (the phase-8 dump)

What is NOT already anywhere is the two hyper-connection sites and the two site norms, because hc_attn_fn and
hc_ffn_fn are quantized and so absent from the dump, and their base/scale entries in the dump are per-site SLICES
rather than the packed arrays the oracle consumes.  This tool fetches those eight from the artifact through the
model's own tensor accessor - the same path that produced the KDA fixture's weights - and writes them, plus the
block's input x (hc_attn_pre-0, which is already [HC][n_embd]-shaped: what the loop takes).

It deliberately does NOT fabricate the expected stages.  Those come from the dump, and a driver should read them
there rather than from a copy, because a copy is one more place for the two sides to drift apart - which is the
defect this port spent most of a session on.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm5_kda_reference as K

GGUF = pathlib.Path('/home/peb/moredata/models/glm-5.3-flash-gguf')
DUMP = pathlib.Path('/home/peb/moredata/glm5-ref-dump5')
HC = 4

# the oracle's key -> the artifact's tensor name.  hc_*_fn has no ".weight" suffix anywhere else in this model's
# naming but does here; measured, not assumed.
WANT = {
    'hc_attn_fn':    'hc_attn_fn.weight',
    'hc_attn_base':  'hc_attn_base.weight',
    'hc_attn_scale': 'hc_attn_scale.weight',
    'attn_norm':     'attn_norm.weight',
    'hc_ffn_fn':     'hc_ffn_fn.weight',
    'hc_ffn_base':   'hc_ffn_base.weight',
    'hc_ffn_scale':  'hc_ffn_scale.weight',
    'ffn_norm':      'ffn_norm.weight',
}


def read_dump(path: pathlib.Path):
    """[0, ne0..ne3] as five u32, then ne0*ne1*ne2*ne3 floats with ne0 fastest.  The header shapes the data."""
    b = path.read_bytes()
    ne = np.frombuffer(b[:20], dtype='<u4')[1:].astype(np.int64)
    if int(np.prod(ne)) * 4 != len(b) - 20:
        raise SystemExit(f'{path.name}: header ne={ne.tolist()} does not match {len(b)-20} data bytes')
    return np.frombuffer(b[20:], dtype='<f4'), ne


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    block = int(sys.argv[1])
    out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)

    model = K.Model(pathlib.Path(sorted(GGUF.glob('*.gguf'))[0]))
    total = 0
    for key, name in WANT.items():
        a = np.asarray(model.tensor(f'blk.{block}.{name}'), dtype=np.float32).ravel()
        a.astype('<f4').tofile(out / f'w_{key}.bin')
        total += a.size
        print(f'    w_{key:<14} {a.size:>9,} floats   range [{a.min():.5g}, {a.max():.5g}]')
    print(f'  {len(WANT)} weights fetched from the artifact ({total:,} floats)')

    # The loop's x is the BLOCK INPUT: HC streams of n_embd.  That is hc_init, whose header reads
    # ne=[4096, 4, 5, 1] - embd 4096, FOUR streams, five tokens.  hc_attn_pre-N is NOT this: its header is
    # ne=[4096, 5, 1, 1], i.e. one stream over five tokens, the normed layer input rather than the stream set.
    # Both are float*, so only a shape assertion distinguishes them - which is exactly why the first attempt
    # against hc_attn_pre stopped here instead of producing a plausible wrong answer.
    b = (DUMP / 'hc_init.bin').read_bytes()
    ne = np.frombuffer(b[:20], dtype='<u4')[1:].astype(np.int64)
    raw = np.frombuffer(b[20:], dtype='<f4')
    if int(np.prod(ne)) != raw.size:
        raise SystemExit(f'hc_init: header ne={ne.tolist()} does not match {raw.size} floats')
    hc, nt = int(ne[1]), int(ne[2])
    if hc != HC:
        raise SystemExit(f'expected HC={HC} streams in the block input, the header says {hc}')
    print(f'  hc_init: ne={ne.tolist()} -> {hc} streams x {nt} tokens x {ne[0]} embd')
    x = raw[:hc * ne[0]].copy()          # token 0, stream-major: the first HC*n_embd floats
    x.astype('<f4').tofile(out / 'x.bin')
    print(f'  wrote x.bin and {len(WANT)} w_*.bin to {out}')
    print(f'  expected stages are NOT copied here - read them from {DUMP} so the two sides cannot drift')
    return 0


if __name__ == '__main__':
    sys.exit(main())
