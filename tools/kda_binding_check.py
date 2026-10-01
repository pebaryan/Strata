#!/usr/bin/env python3
"""Check the engine's binding of the KDA tensors against the oracle's own values.

Why this is a file and not a shell one-liner: the driver separates its fields with the two characters
backslash + t (a literal '\\t' in the printf), and every attempt to express that inside `python3 -c "..."` lost
a backslash to shell quoting and silently matched nothing - which reports as "0 parsed" or as absurd values taken
from the hash field.  In a real source file '\\t' means backslash-t, which is what the driver actually emits.

Usage:
    ./glm5_bind_check <pack> <gguf> 0 > /tmp/bind.txt      # the driver's own output
    kda_binding_check.py /tmp/bind.txt /path/to/glm5-kda-l0-t1

What it compares: the driver prints min / max / sum for every tensor it binds as floats.  For the F32 tensors that
is a real check; tensors the artifact serves natively quantized print no stats, so they are reported as QUANTIZED
rather than as failures.  It is deliberately NOT an elementwise check - the driver does not print elements - so a
pass here means the right tensor arrived at the right scale, not that every value agrees.
"""
import pathlib
import re
import sys

import numpy as np

# oracle weight key -> the artifact tensor name the engine binds
TENSORS = {
    "attn_norm": "attn_norm.weight",
    "conv_q": "ssm_conv1d_q.weight",
    "conv_k": "ssm_conv1d_k.weight",
    "conv_v": "ssm_conv1d_v.weight",
    "ssm_a": "ssm_a",
    "dt_bias": "ssm_dt.bias",
    "ssm_f_a": "ssm_f_a.weight",
    "ssm_f_b": "ssm_f_b.weight",
    "ssm_beta": "ssm_beta.weight",
    "ssm_g_a": "ssm_g_a.weight",
    "ssm_g_b": "ssm_g_b.weight",
    "o_norm": "ssm_norm.weight",
    "wo": "attn_output.weight",
}

# name \t ne=.. \t n=<digits> \t min \t max \t sum \t hash
ROW = re.compile(
    r"^blk\.(?P<blk>\d+)\.(?P<name>[^\\]+)\\tne=[^\\]+\\tn=\d+\\t"
    r"(?P<min>[^\\]+)\\t(?P<max>[^\\]+)\\t(?P<sum>[^\\]+)\\t"
)


def parse(driver_output):
    """Fields: name \\t ne=.. \\t n=<digits> \\t min \\t max \\t sum \\t hash, separated by REAL TABS.

    The trap that cost seven attempts: repr() renders a real tab as '\\t' and so does od -c, so reading either and
    concluding the file contains a literal backslash-t is wrong - a literal backslash-t shows as '\\\\t' in repr and
    as two separate columns in od -c.  These lines use ordinary tabs; split on '\\t' in this file, which is one.
    """
    rows = {}
    for line in pathlib.Path(driver_output).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) < 6 or not parts[0].startswith("blk."):
            continue
        try:
            rows[parts[0].split(".", 2)[2]] = (float(parts[3]), float(parts[4]), float(parts[5]))
        except ValueError:
            continue
    return rows


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    rows = parse(sys.argv[1])
    fixture = pathlib.Path(sys.argv[2])
    print(f"parsed {len(rows)} float rows from {sys.argv[1]}")
    if not rows:
        print("nothing parsed: check that the driver output is the file you think it is")
        return 2

    n_pass = n_fail = n_quant = 0
    print(f"  {'oracle key':<12} {'engine [min max]':<34} {'oracle [min max]':<34} rel      verdict")
    for key, tensor in TENSORS.items():
        f = fixture / f"w_{key}.bin"
        if not f.exists():
            print(f"  {key:<12} no fixture file {f.name}")
            continue
        a = np.fromfile(f, dtype="<f4")
        ref = (float(a.min()), float(a.max()), float(a.sum()))
        if tensor not in rows:
            print(f"  {key:<12} {'QUANTIZED in the artifact - no float values to compare':<34}")
            n_quant += 1
            continue
        got = rows[tensor]
        scale = max(1e-9, abs(ref[0]), abs(ref[1]))
        rel = max(abs(got[0] - ref[0]), abs(got[1] - ref[1]), abs(got[2] - ref[2]) / a.size) / scale
        verdict = "PASS" if rel < 1e-6 else "FAIL"
        n_pass += verdict == "PASS"
        n_fail += verdict == "FAIL"
        print(f"  {key:<12} [{got[0]:+.8g} {got[1]:+.8g}]{'':<12} [{ref[0]:+.8g} {ref[1]:+.8g}]{'':<12}"
              f" {rel:.1e}  {verdict}")

    print(f"  -> {n_pass} PASS, {n_fail} FAIL, {n_quant} quantized in the artifact (not comparable as floats)")
    print("  note: min/max/sum only - this does not compare the conv tensors elementwise, so it cannot detect a")
    print("  permutation inside ssm_conv1d_*.weight, only a wrong tensor or a wrong scale.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
