"""The oracle's own dequantization of blk.0.ffn_down.weight, for the Q6_K parity gate.

Built for the same reason the Q5_K gate exists: a parity gate for TRANSCRIBED code must not compare against the source
it was transcribed from.  dequant_q6_K was written by transcribing llama.cpp's C, so comparing it to llama.cpp's C
would only confirm the copying was faithful, not that the copy is right.  This produces the tensor's values through
gguf-py's numpy implementation instead - a separate implementation in a separate language, which is the reference that
can actually disagree.

Written because this is not a hypothetical: the block gate's 3.0% discrepancy against the oracle is consistent with a
wrong ffn_down and nothing else in the block has ever been ungated, and dequant_q6_K is the one kernel in the chain
that has never been compared to anything.

usage: q6k_reference.py <out-dir>
writes <out-dir>/oracle_dequantised.bin (float32, ne0-fastest, no header) and <out-dir>/raw_q5k_blocks.bin
"""

import os
import sys
import pathlib

import numpy as np

sys.path.insert(0, os.environ.get("STRATA_GGUF_PY", "/home/peb/llama.cpp-glm5/gguf-py"))

from gguf import GGUFReader  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

GGUF = "/home/peb/moredata/models/glm-5.3-flash-gguf/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf"
TENSOR = "blk.0.ffn_down.weight"


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: q6k_reference.py <out-dir>", file=sys.stderr)
        return 2
    out = pathlib.Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)

    r = GGUFReader(GGUF)
    hit = [t for t in r.tensors if t.name == TENSOR]
    if not hit:
        print(f"{TENSOR} not found in {GGUF}", file=sys.stderr)
        return 1
    t = hit[0]
    print(f"  {TENSOR}: type {t.tensor_type.name}, shape {t.shape}, raw {t.n_bytes} bytes")

    # the raw blocks, exactly as the file holds them, so the C++ gate can be run over the same bytes
    raw = t.data.tobytes()
    (out / "raw_q5k_blocks.bin").write_bytes(raw)
    print(f"  wrote raw_q5k_blocks.bin: {len(raw)} bytes")

    # the reference values, one element per 256-element block
    vals = dequantize(t.data, t.tensor_type).astype(np.float32)
    n_expected = int(np.prod(t.shape))
    if vals.size != n_expected:
        print(f"  dequantized {vals.size} values but the shape implies {n_expected}", file=sys.stderr)
        return 1
    # ne0-fastest: the gate's own convention, and the convention glm5_block_driver now honours
    vals = np.asfortranarray(vals.reshape(tuple(reversed(t.shape))), dtype=np.float32)
    (out / "oracle_dequantised.bin").write_bytes(vals.tobytes(order="F"))
    d = vals.astype(np.float64)
    print(f"  wrote oracle_dequantised.bin: {vals.size} floats  rms {np.sqrt((d * d).mean()):.9g}  "
          f"min {vals.min():.9g}  max {vals.max():.9g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
