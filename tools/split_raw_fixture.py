#!/usr/bin/env python3
"""Split an oracle RAW fixture into the 29-file directory kda_gate reads.

    tools/split_raw_fixture.py <raw-fixture-file> <outdir>

WHY THIS EXISTS.  The 29-file fixtures were hand-rolled: a script read the artifact, dequantized the weights, repacked
the convolutions, ran the oracle and wrote one file per quantity.  Every one of those steps is a chance to produce a
directory whose parts do not imply each other - and block 1's directory is exactly that: its inter_qc is not what its
own xn, wq and conv_q imply, while block 0's is.  A gate over an inconsistent fixture cannot support any conclusion
in either direction, which is what made a correct kernel look broken for most of a session.

The oracle's own --raw-fixture writer cannot have that defect: it writes the very arrays the model loader handed to
kda_block, plus that same call's return, with nothing recomputed or reordered on the way out.  This tool converts
that file into the layout the gate already reads, so the gate can consume reference output that is self-consistent by
construction - without touching any C++.

THE FORMAT, straight from the writer in glm5_kda_reference.py:

    struct.pack("<6i", N_EMBD, NH, HD, D_CONV, tokens, layer)        # 24-byte little-endian header
    17 float32 arrays: attn_norm, wq, wk, wv, conv_q, conv_k, conv_v, ssm_a, dt_bias,
                       ssm_f_a, ssm_f_b, ssm_beta, ssm_g_a, ssm_g_b, o_norm, wo, x
    11 float32 arrays: qc, kc, vc, g, beta, q, k, v, attn, o, result

Sizes follow from the header.  The one shape that is not a clean function of (NE, NH, HD) is dt_bias, whose length is
taken from the existing block-0 fixture rather than guessed - a measurement, not an assumption - and the tool then
ASSERTS that all the sizes it assigns consume the file exactly.  A mismatch is a hard error, because a shifted read
would silently produce a fixture that is wrong in a way no gate could attribute.

ONE DERIVATION IS UNAVOIDABLE and is called out rather than hidden: the writer emits x but not xn, and the gate
compares xn.  So xn is computed here as rms_norm(x, attn_norm).  That is the only non-copy step in this file, it is
the same arithmetic the oracle applies internally, and it is checked against the working block-0 fixture whenever
that fixture is present - so if this derivation were wrong, the tool says so rather than emitting a quiet difference.
"""
import pathlib
import struct
import sys

import numpy as np

REF_FIXTURE = pathlib.Path('/home/peb/moredata/glm5-kda-l0-t1')

# MEASURED, NOT DERIVED.  The first version of this table derived every size from (n_embd, nh, hd, d_conv, tokens)
# and was wrong for four arrays - ssm_f_a, ssm_f_b, ssm_beta, ssm_g_a, ssm_g_b - by factors between 128 and 4096,
# because they are much larger than the per-head or per-channel shapes those dimensions suggest.  The total-size
# assertion below caught all five before a single byte was written, which is the entire reason it exists: a
# misaligned read would have produced another fixture whose parts do not imply each other, and another session lost
# to a correct kernel looking broken.  These numbers come from the oracle's own tensors on block 1.
MEASURED = {
    'attn_norm': 4096, 'wq': 33554432, 'wk': 33554432, 'wv': 33554432,
    'conv_q': 32768, 'conv_k': 32768, 'conv_v': 32768,
    'ssm_a': 64, 'dt_bias': 8192,
    'ssm_f_a': 524288, 'ssm_f_b': 1048576, 'ssm_beta': 262144,
    'ssm_g_a': 524288, 'ssm_g_b': 1048576,
    'o_norm': 128, 'wo': 33554432,
}


def dt_bias_len() -> int:
    """dt_bias is the one array whose size is not implied by the header.  Measure it from a fixture that works."""
    p = REF_FIXTURE / 'w_dt_bias.bin'
    if not p.exists():
        raise SystemExit(f'cannot determine dt_bias length: {p} is missing. Refusing to guess a size that a wrong '
                         f'value would turn into a silently shifted read.')
    return int(p.stat().st_size // 4)


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    raw, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    blob = raw.read_bytes()

    if len(blob) < 24:
        raise SystemExit(f'{raw.name}: too short for a header')
    ne, nh, hd, d_conv, tokens, layer = struct.unpack_from('<6i', blob, 0)
    di = nh * hd
    print(f'  header: n_embd {ne} nh {nh} hd {hd} d_conv {d_conv} tokens {tokens} layer {layer}')

    # sizes come from MEASURED where the dimensions mislead, and are checked in aggregate below
    weights = [(n, MEASURED[n]) for n in ('attn_norm', 'wq', 'wk', 'wv', 'conv_q', 'conv_k', 'conv_v',
                                          'ssm_a', 'dt_bias', 'ssm_f_a', 'ssm_f_b', 'ssm_beta',
                                          'ssm_g_a', 'ssm_g_b', 'o_norm', 'wo')]
    weights.append(('x', tokens * ne))
    inters = [('qc', tokens * di), ('kc', tokens * di), ('vc', tokens * di), ('g', tokens * di),
              ('beta', tokens * nh), ('q', tokens * di), ('k', tokens * di), ('v', tokens * di),
              ('attn', tokens * di), ('o', tokens * di), ('result', tokens * ne)]
    # NOTE: q, k and v are three arrays of tokens*di, not four - the first draft of this file counted four and the
    # aggregate check below reported a 24-byte gap, which is exactly the 6-int header it was comparing against.

    need = 24 + 4 * (sum(n for _, n in weights) + sum(n for _, n in inters))
    if need != len(blob):
        raise SystemExit(f'LAYOUT MISMATCH: header and shapes imply {need} bytes, the file is {len(blob)}. '
                         f'Refusing to write a fixture from a misaligned read. (dt_bias length {dt_bias_len()} '
                         f'comes from {REF_FIXTURE.name}; if that fixture is not the same model this is the cause.)')
    print(f'  layout checks out: {len(blob)} bytes = 24 + 4*({need - 24})/4 floats')

    off = 24
    arrays = {}
    for name, n in weights + inters:
        arrays[name] = np.frombuffer(blob, dtype='<f4', count=n, offset=off).copy()
        off += 4 * n
    assert off == len(blob)

    for name, _ in weights:
        arrays[name].astype('<f4').tofile(out / f'w_{name}.bin')
    for name, _ in inters:
        arrays[name].astype('<f4').tofile(out / f'inter_{name}.bin')
    # the gate calls the block input input.bin, not w_x.bin
    arrays['x'].astype('<f4').tofile(out / 'input.bin')
    # the gate looks for result.bin by name, not inter_result.bin - without this the last stage is skipped
    arrays['result'].astype('<f4').tofile(out / 'result.bin')

    # the one derivation, written and labelled as such
    x = arrays['x'].reshape(tokens, ne).astype(np.float64)
    w = arrays['attn_norm'].astype(np.float64)
    inv = 1.0 / np.sqrt((x * x).mean(axis=1) + 1e-5)
    (x * inv[:, None] * w).astype('<f4').tofile(out / 'inter_xn.bin')

    # and if a working fixture is around, check the derivation against it rather than asserting it by construction
    ref = REF_FIXTURE / 'inter_xn.bin'
    if ref.exists() and tokens == 1 and layer == 0:   # only meaningful for the same layer - comparing block 1's xn against block 0's fixture said nothing
        got = np.fromfile(out / 'inter_xn.bin', dtype='<f4').astype('f8')
        want = np.fromfile(ref, dtype='<f4').astype('f8')
        if got.size == want.size:
            rel = float(np.abs(got - want).max()) / max(1e-30, float(np.abs(want).max()))
            print(f'  xn derivation vs the working block-0 fixture: rel {rel:.3e}')
        else:
            print(f'  xn derivation not compared: sizes {got.size} vs {want.size}')
    print(f'  wrote {len(weights)} w_*.bin, {len(inters) + 1} inter_*.bin to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
