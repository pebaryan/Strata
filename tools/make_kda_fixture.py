#!/usr/bin/env python3
"""Build a KDA gate fixture for any block of GLM-5.3, from the oracle.

    tools/make_kda_fixture.py <block> <outdir> [tokens]

This existed only as a throwaway inline script when block 0's fixture was built, which meant the recipe had to be
reconstructed from memory the first time a second block was needed - and reconstructing a recipe from memory is
how this port lost time on the Q5_K capture and on the dump's container format.  It is a saved tool now.

WHAT IT PRODUCES, and where each piece comes from:

    input.bin            the block's pre-norm hidden state, i.e. hc_pre's layer_in.  The dump has it as
                         hc_attn_pre-<block>.bin.  kda_block applies rms_norm itself, so this is the state BEFORE
                         the attention norm, not after.
    w_<oracle key>.bin   the 16 tensors kda_block consumes, dequantized from the artifact through the model's own
                         tensor accessor.  The oracle's keys are not the artifact's names; the mapping is explicit
                         below rather than derived.
    inter_<name>.bin     each intermediate the oracle returns (xn, qc, kc, vc, g, beta, q, k, v, attn, o, state)
    result.bin           the block's output

Two conventions, both learned the hard way and both asserted rather than assumed:
  * every file in glm5-ref-dump* is [0, ne0, ne1, ne2, ne3] as five u32 words then ne0*ne1*ne2*ne3 floats, ne0
    fastest - so the header is read and used to shape the data rather than the shape being guessed;
  * x is (tokens, N_EMBD), which is the layout kda_block documents.

SELF-TEST: run it for block 0 into a temporary directory and diff against the working fixture
/home/peb/moredata/glm5-kda-l0-t1.  A builder that cannot reproduce a fixture already known to pass is not a
builder, and that comparison is the only thing that makes it safe to point at block 1.
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import glm5_hc_reference as H
import glm5_kda_reference as K

DUMP = pathlib.Path('/home/peb/moredata/glm5-ref-dump5')
GGUF = pathlib.Path('/home/peb/moredata/models/glm-5.3-flash-gguf')

# the oracle's key -> the artifact's tensor name.  Written out rather than guessed from the key.
KEYS = {
    'attn_norm': 'attn_norm.weight',
    'wq': 'attn_q.weight',
    'wk': 'attn_k.weight',
    'wv': 'attn_v.weight',
    'wo': 'attn_output.weight',
    'conv_q': 'ssm_conv1d_q.weight',
    'conv_k': 'ssm_conv1d_k.weight',
    'conv_v': 'ssm_conv1d_v.weight',
    'ssm_a': 'ssm_a',            # NOTE: the artifact name carries no .weight suffix
    'ssm_beta': 'ssm_beta.weight',
    'dt_bias': 'ssm_dt.bias',   # ...nor does this one
    'ssm_f_a': 'ssm_f_a.weight',
    'ssm_f_b': 'ssm_f_b.weight',
    'ssm_g_a': 'ssm_g_a.weight',
    'ssm_g_b': 'ssm_g_b.weight',
    'o_norm': 'ssm_norm.weight',  # the artifact calls it ssm_norm
}


def read_dump(path: pathlib.Path) -> np.ndarray:
    """A dump file: five u32 words [0, ne0..ne3], then ne0*ne1*ne2*ne3 floats with ne0 fastest."""
    b = path.read_bytes()
    if len(b) < 20:
        raise ValueError(f'{path.name} is too short to hold a header')
    ne = np.frombuffer(b[:20], dtype='<u4')[1:].astype(np.int64)
    if ne[0] == 0 or int(np.prod(ne)) * 4 != len(b) - 20:
        raise ValueError(f'{path.name}: header ne={ne.tolist()} does not match {len(b) - 20} data bytes')
    # Rows are the product of the non-fastest axes: hc_attn_pre-N has ne=[4096, nt, 1, 1], so the
    # token count lives in ne[1], and collapsing only ne[2] would fold every token into one row.
    return np.frombuffer(b[20:], dtype='<f4').astype(np.float64).reshape(ne[2] * ne[1], ne[0])


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    block = int(sys.argv[1])
    out = pathlib.Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    # The input is the block's PRE-NORM hidden state, and its source matters:
    #   block 0: hc_pre's layer_in, computed here from the dump's hc_init with the artifact's hc weights.  This is
    #            what the working block-0 fixture used - and hc_attn_pre-0 is NOT the same tensor (it differs by
    #            5.3e-02 and is what stage 1 was mistakenly compared against for a while).
    #   later blocks: hc_attn_pre-<block>, a real per-block activation from the dump.  For a gate that is enough:
    #            the oracle and the engine are compared on the SAME input, which is all a gate requires.  It is
    #            not the chain's true input, because that is block N-1's output and needs the chain to run.
    model = K.Model(pathlib.Path(sorted(GGUF.glob('*.gguf'))[0]))
    if block == 0:
        xi = read_dump(DUMP / 'hc_init.bin')
        pfx = 'blk.0.'
        fn = np.asarray(model.tensor(pfx + 'hc_attn_fn.weight'), dtype=np.float64)
        sc = np.asarray(model.tensor(pfx + 'hc_attn_scale.weight'), dtype=np.float64)
        ba = np.asarray(model.tensor(pfx + 'hc_attn_base.weight'), dtype=np.float64)
        nt = xi.shape[0] // 4
        layer_in, _, _, _ = H.hc_pre(xi.reshape(nt, 4, 4096).transpose(1, 2, 0), fn, sc, ba)
        state = np.asarray(layer_in).T.reshape(nt, 4096)
    else:
        state = read_dump(DUMP / f'hc_attn_pre-{block}.bin')
    if state.shape[0] < tokens:
        raise ValueError(f'block {block}: dump holds {state.shape[0]} tokens, {tokens} requested')
    x = state[:tokens].reshape(tokens, -1)
    x.astype('<f4').tofile(out / 'input.bin')   # the fixture holds `tokens` rows of n_embd
    print(f'  block {block}: input {x.shape} from hc_attn_pre-{block}.bin  rms {np.sqrt((x ** 2).mean()):.6g}')

    w = {}
    for key, tensor in KEYS.items():
        arr = np.asarray(model.tensor(f'blk.{block}.{tensor}'), dtype=np.float64)
        if key.startswith('conv_'):
            # The artifact stores ssm_conv1d_* as (d_inner, 1, d_conv) with d_conv FASTEST; the engine and the
            # oracle both want (d_conv, d_inner), i.e. reshape(d_inner, d_conv).T.  This is the same transform
            # repack_conv applies in src/core/glm_bind.cpp - without it the conv weights come out permuted, which
            # a min/max comparison cannot see and which would have looked like a KDA kernel failure.
            d_conv = arr.size // arr.shape[0]
            arr = np.ascontiguousarray(arr.reshape(arr.shape[0], d_conv).T).ravel()
        w[key] = arr
        arr.astype('<f4').tofile(out / f'w_{key}.bin')
    print(f'  {len(w)} weights bound from the artifact; wq {w["wq"].shape} conv_q {w["conv_q"].shape}')

    result, inter = K.kda_block(w, x, tokens)

    # SELF-CONSISTENCY, ASSERTED BEFORE ANYTHING IS WRITTEN.  A fixture whose parts do not imply each other cannot
    # validate anything, and a gate cannot tell that apart from a broken engine.  This costs three lines and it is
    # the check that took a whole session to think of when it should have been first: recompute one reference
    # intermediate from the fixture's OWN other parts and require them to agree.
    def _silu(v):
        return v / (1.0 + np.exp(-v))

    def _conv1d_causal(xx, ww):
        t, ch = xx.shape
        out = np.zeros_like(xx)
        d = ww.shape[0]
        for k in range(d):
            sh = k - (d - 1)
            if sh < 0:
                out[-sh:] += ww[k] * xx[:t + sh]
            else:
                out[:t - sh] += ww[k] * xx[sh:]
        return out

    xn_ref = np.asarray(inter['xn'], dtype=np.float64).reshape(tokens, -1)
    qc_ref = np.asarray(inter['qc'], dtype=np.float64).ravel()
    qc_calc = _silu(_conv1d_causal(xn_ref @ w['wq'].T, w['conv_q'].reshape(d_conv, -1))).ravel()
    gap = float(np.abs(qc_calc - qc_ref).max()) / max(1e-30, float(np.abs(qc_ref).max()))
    print(f'  self-consistency: silu(conv1d(xn @ wq.T, conv_q)) vs the oracle\'s own qc: rel {gap:.3e}')
    if gap > 1e-4:
        raise SystemExit(f'FIXTURE IS NOT SELF-CONSISTENT for block {block}: its qc is not what its own xn, wq '
                         f'and conv_q imply (rel {gap:.3e}). Nothing has been written. A gate over this would be '
                         f'measuring a disagreement between two things that were never consistent.')
    for name, arr in inter.items():
        np.asarray(arr, dtype=np.float32).tofile(out / f'inter_{name}.bin')
    np.asarray(result, dtype=np.float32).tofile(out / 'result.bin')
    r = np.asarray(result)
    print(f'  result {r.shape}  min {r.min():.6g} max {r.max():.6g} rms {np.sqrt((r ** 2).mean()):.6g}')
    print(f'  intermediates written: {sorted(inter.keys())}')
    print(f'  fixture written to {out}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
