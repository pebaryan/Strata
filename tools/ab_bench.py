#!/usr/bin/env python3
"""Interleaved A/B benchmark for the GLM-5.3-Flash Strata runner.

WHY THIS EXISTS.  The engine is precise inside a batch (per-stage spreads of 0.3-1.3% over three consecutive runs) but
the whole level moves by up to 1.8x BETWEEN batches, and the per-stage shares move with it - ffn_norm is 55.6% of a fast
batch and 69.2% of a slow one, which is the signature of the I/O-bound expert stage following the page cache's state.
That makes any comparison of two runs taken at different times worthless, including one of mine: the claim that keeping
the expert pages is 1.8x slower compared two runs in one batch against a run in another, and 1.8x is exactly the size of
the between-batch swing.

WHAT THIS DOES INSTEAD.  It alternates the two arms - A, B, A, B - inside a single session, so both see the same cache
state and any drift is shared rather than attributed to the flag.  It reports the PAIRED differences as well as the arm
means, and it refuses a verdict it cannot support: if the paired difference is not larger than the within-arm spread, the
answer is NOT DISTINGUISHABLE, which is a result rather than a failure.

USAGE
    python3 tools/ab_bench.py --pairs 3 --a STRATA_EXPERT_KEEP=1 --b STRATA_EXPERT_KEEP= \\
        -- ./build-glm5/glm45_run /path/to/pack /path/to/model.gguf

`--b X=` (empty value) means the variable is removed for that arm.
"""
import argparse
import os
import re
import statistics
import subprocess
import sys


def one_run(cmd, env_extra):
    """Run once, return (ms per token, wall seconds, argmax, returncode)."""
    env = dict(os.environ)
    for k, v in env_extra.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    m = re.search(r"=\s*([\d.]+)\s*ms/token", proc.stderr)
    tok = re.search(r"argmax\s*=\s*(\d+)", proc.stdout)
    return (float(m.group(1)) if m else None, (tok.group(1) if tok else "?"), proc.returncode)


def parse_env(spec):
    if spec is None:
        return {}
    out = {}
    for item in spec.split(","):
        if not item.strip():
            continue
        k, _, v = item.partition("=")
        out[k.strip()] = v if v != "" else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=3, help="number of A/B pairs (each pair is two runs)")
    ap.add_argument("--a", default=None, help="arm A environment, e.g. 'VAR=1,OTHER=2'")
    ap.add_argument("--b", default=None, help="arm B environment; 'VAR=' means unset")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- then the command to run")
    args = ap.parse_args()
    cmd = [c for c in args.cmd if c != "--"]
    if not cmd:
        print("ab_bench: give the command after --", file=sys.stderr)
        return 2

    env_a, env_b = parse_env(args.a), parse_env(args.b)
    print("arm A: %s" % (env_a if env_a else "{} (unchanged environment)"))
    print("arm B: %s" % (env_b if env_b else "{} (unchanged environment)"))
    print("command: %s" % " ".join(cmd))
    print()

    A, B, pairs = [], [], []
    for i in range(args.pairs):
        # A then B within the round, so each pair brackets the same stretch of wall time
        a_ms, a_tok, a_rc = one_run(cmd, env_a)
        b_ms, b_tok, b_rc = one_run(cmd, env_b)
        if a_ms is None or b_ms is None:
            print("pair %d: FAILED to parse a time (rc %d/%d) - stopping" % (i + 1, a_rc, b_rc), file=sys.stderr)
            return 1
        A.append(a_ms)
        B.append(b_ms)
        pairs.append(a_ms - b_ms)
        print("  pair %d: A %9.1f ms  B %9.1f ms   A-B %+9.1f ms   argmax %s/%s" % (i + 1, a_ms, b_ms, a_ms - b_ms, a_tok, b_tok))

    def spread(v):
        return (max(v) - min(v)) / statistics.mean(v) * 100.0 if len(v) > 1 else 0.0

    ma, mb, mp = statistics.mean(A), statistics.mean(B), statistics.mean(pairs)
    within = max(spread(A), spread(B))
    print()
    print("  arm A mean %9.1f ms/token  (%.2f tok/s)  spread %.1f%%" % (ma, 1000.0 / ma, spread(A)))
    print("  arm B mean %9.1f ms/token  (%.2f tok/s)  spread %.1f%%" % (mb, 1000.0 / mb, spread(B)))
    print("  paired mean difference (A-B) %+9.1f ms  (%.1f%% of the mean)" % (mp, 100.0 * mp / ((ma + mb) / 2.0)))
    print("  largest within-arm spread   %9.1f %%" % within)
    if abs(mp) / ((ma + mb) / 2.0) * 100.0 > within:
        print("  VERDICT: DISTINGUISHABLE - the paired difference is larger than the within-arm spread.")
        print("           %s is %s." % ("A", "slower" if mp > 0 else "faster"))
    else:
        print("  VERDICT: NOT DISTINGUISHABLE - the paired difference is no larger than the within-arm spread,")
        print("           so this experiment does not support a claim either way.  Add pairs or change the thing measured.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
