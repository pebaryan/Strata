#!/usr/bin/env python3
"""Run the ORACLE on the real block input, and write its stages in the dump's own container format.

    tools/make_oracle_stages.py <outdir>

Why this exists: block_gate's attention-site run compared four stages against the phase-8 dump and every one failed,
but the engine's stage-1 output is verified against the ORACLE to seven significant digits while the dump's
attn_norm token 0 (-0.2749/0.37) matches no slice of the dump and disagrees with the oracle.  So the dump and the
oracle disagree at that point - a discrepancy this project already recorded as unresolved - and comparing against the
dump cannot distinguish "the engine is wrong" from "the dump holds something else".

Feeding the oracle the SAME real input the engine got, and writing its output in the dump's container format
([0, ne0..ne3] then ne0-fastest floats), lets block_gate compare engine against oracle with no change to the C++:
point its dump directory argument at this output.

Note kda_block applies its own rms_norm, so the input is hc_attn_pre-0's token 0 - the PRE-norm value - not the
normed one the engine's stage 1 hands to the kernel.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm5_kda_reference as K

GGUF = pathlib.Path('/home/peb/moredata/models/glm-5.3-flash-gguf')
DUMP = pathlib.Path('/home/peb/moredata/glm5-ref-dump5')


def read_dump(path: pathlib.Path):
    b = path.read_bytes()
    ne = np.frombuffer(b[:20], dtype='<u4')[1:].astype(np.int64)
    a = np.frombuffer(b[20:], dtype='<f4')
    if int(np.prod(ne)) != a.size:
        raise SystemExit(f'{path.name}: header ne={ne.tolist()} does not match {a.size} floats')
    return a, ne


def write_dump(path: pathlib.Path, a: np.ndarray, ne):
    """The dump's container: five u32 [0, ne0..ne3], then the floats.  Written so block_gate can read it as-is."""
    hdr = np.array([0, ne[0], ne[1], ne[2], ne[3]], dtype='<u4')
    with open(path, 'wb') as fh:
        fh.write(hdr.tobytes())
        fh.write(np.asarray(a, dtype='<f4').tobytes())
    print(f'    wrote {path.name:<22} ne={[int(v) for v in ne]}  {a.size} floats  rms {np.sqrt((a.astype(np.float64)**2).mean()):.6g}')


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else '/home/peb/moredata/glm5-oracle-l0')
    out.mkdir(parents=True, exist_ok=True)

    pre, ne = read_dump(DUMP / 'hc_attn_pre-0.bin')
    # (T, D) token-major: kda_block's conv1d_causal unpacks `T, ch = x.shape`, so a flat vector is rejected.
    x = pre[:4096].astype(np.float64).reshape(1, -1)   # token 0 of the PRE-norm value (kda_block norms it)
    print(f'  input: hc_attn_pre-0 token 0, {x.size} floats, rms {np.sqrt((x**2).mean()):.6g}')

    model = K.Model(pathlib.Path(sorted(GGUF.glob('*.gguf'))[0]))
    w = K.load_weights(model, 0)
    print(f'  oracle weights loaded: {len(w)} entries')

    # the oracle's own view of the two upstream stages, so the engine's stage 1 can be compared on the SAME input
    xn = K.rms_norm(x, w["attn_norm"])   # the dict key is attn_norm
    write_dump(out / 'attn_norm-0.bin', xn.ravel(), [4096, 1, 1, 1])
    write_dump(out / 'hc_attn_pre-0.bin', x.ravel(), [4096, 1, 1, 1])

    res = K.kda_block(w, x, 1)
    print(f'  kda_block returned {type(res).__name__}'
          + (f' of {len(res)}' if isinstance(res, tuple) else ''))

    def emit(stage: str, value, n: int):
        a = np.asarray(value, dtype=np.float64).ravel()
        if a.size != n:
            print(f'    SKIP {stage}: {a.size} floats, the engine produces {n}')
            return
        write_dump(out / f'{stage}-0.bin', a, [n, 1, 1, 1])

    # the oracle's own intermediates, where it exposes them, plus the final output
    if isinstance(res, tuple):
        for i, part in enumerate(res):
            try:
                emit('attn_output' if i == 0 else f'oracle_part{i}', part, 4096)   # the KDA returns n_embd
            except Exception as exc:                                   # noqa: BLE001 - report, do not mask
                print(f'    no emit for part {i}: {exc}')
    else:
        emit('attn_output', res, 4096)   # the KDA returns n_embd

    print(f'  oracle stages written to {out}')
    print('  now run: block_gate <block-dir> <kda-dir> <ffn-dir> ' + str(out))
    return 0


if __name__ == '__main__':
    sys.exit(main())
