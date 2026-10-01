#!/usr/bin/env python3
"""Build a KDA fixture in the oracle's own raw format, but from a REAL input rather than the oracle's seed.

    tools/make_kda_fixture_from.py <attn_norm.bin> <out.raw> [token]

Why: block_gate demonstrated that the KDA kernel disagrees with the oracle by 1.391x on a real activation, while
kda_gate passes twelve of twelve on the oracle's own seed.  The twelve-intermediate gate is already built and already
has its failure mode demonstrated, so pointing it at this input turns "the kernel is wrong by a factor somewhere" into
"this operation is wrong".

The writing below is the oracle's own writer, copied verbatim from glm5_kda_reference.py's --raw-fixture branch.  The
ONLY difference is that x comes from a file here instead of from the oracle's construction - which is the entire
point of the exercise, and the one difference that has to be right.  Reproducing more than that would risk the
wrong-against-wrong defect that a hand-rolled fixture caused earlier in this port; the layout is asserted by
tools/split_raw_fixture.py, which checks that the arrays consume the file exactly.
"""
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm5_kda_reference as K

GGUF = pathlib.Path('/home/peb/moredata/models/glm-5.3-flash-gguf')


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    token = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    tokens = 1

    # the input, in the oracle's container format: [0, ne0..ne3] then ne0-fastest floats
    b = src.read_bytes()
    ne = np.frombuffer(b[:20], dtype='<u4')[1:].astype(np.int64)
    a = np.frombuffer(b[20:], dtype='<f4')
    if int(np.prod(ne)) != a.size:
        raise SystemExit(f'{src.name}: header ne={ne.tolist()} does not match {a.size} floats')
    width = int(ne[0])
    if a.size < width * (token + 1):
        raise SystemExit(f'{src.name}: no token {token}')
    # (T, D) token-major: conv1d_causal unpacks `T, ch = x.shape` and rejects a flat vector
    x = a[token * width:(token + 1) * width].astype(np.float64).reshape(tokens, width)
    print(f'  x: {src.name} token {token}, shape {x.shape}, rms {np.sqrt((x**2).mean()):.6g}')

    model = K.Model(pathlib.Path(sorted(GGUF.glob('*.gguf'))[0]))
    w = K.load_weights(model, 0)                      # the same 16 entries the oracle uses
    result, mid = K.kda_block(w, x, tokens, want_state=True)
    print(f'  intermediates available: {sorted(mid.keys())}')
    print(f'  result rms {np.sqrt((np.asarray(result, dtype=np.float64)**2).mean()):.6g}')

    # ---- verbatim from the oracle's writer ----
    with open(out, 'wb') as fh:
        fh.write(struct.pack('<6i', K.N_EMBD, K.NH, K.HD, K.D_CONV, tokens, 0))
        for arr in (w['attn_norm'], w['wq'], w['wk'], w['wv'], w['conv_q'], w['conv_k'], w['conv_v'],
                    w['ssm_a'], w['dt_bias'], w['ssm_f_a'], w['ssm_f_b'], w['ssm_beta'],
                    w['ssm_g_a'], w['ssm_g_b'], w['o_norm'], w['wo'], x):
            fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        for arr in (mid['qc'], mid['kc'], mid['vc'], mid['g'], mid['beta'], mid['q'], mid['k'],
                    mid['v'], mid['attn'], mid['o'], result):
            fh.write(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
    print(f'  wrote raw fixture {out}  ({out.stat().st_size} bytes)')
    print(f'  next: tools/split_raw_fixture.py {out} <outdir>   then   ./build-glm5/kda_gate <outdir> {tokens}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
