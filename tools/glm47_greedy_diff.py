#!/usr/bin/env python3
"""Diff the Strata runner's greedy tokens against a llama.cpp oracle's, token by token.

    python tools/glm47_greedy_diff.py --prompt "The quick brown fox jumps over the lazy" --n 40

This is the phase-5 acceptance test made reproducible: it tokenizes the prompt on the oracle, asks
the oracle for `n` greedy tokens AND runs `build/glm47_run.exe` on the same ids, then reports how many
greedy tokens agree and where they first diverge.  (The oracle's own `/completion` returns an empty
`tokens` array in this build, so the oracle's ids are recovered by re-tokenising its text - stable for
greedy output.)
"""
import argparse
import json
import subprocess
import urllib.request


def post(base, path, body, timeout=180):
    req = urllib.request.Request(base + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="build/glm47_run.exe")
    ap.add_argument("--oracle", default="http://127.0.0.1:8080")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    ids = post(args.oracle, "/tokenize", {"content": args.prompt})["tokens"]
    oracle = post(args.oracle, "/completion",
                  {"prompt": args.prompt, "n_predict": args.n, "temperature": 0, "cache_prompt": False})
    oracle_text = oracle.get("content", "")
    oracle_ids = post(args.oracle, "/tokenize", {"content": oracle_text})["tokens"]

    run = subprocess.run([args.runner, "--tokens", ",".join(map(str, ids)), "--gen", str(args.n)],
                         capture_output=True, text=True)
    mine = [int(line.split(":")[1].split()[0])
            for line in run.stdout.splitlines() if line.strip().startswith("step ")]

    match = 0
    for a, b in zip(mine, oracle_ids):
        if a != b:
            break
        match += 1

    my_text = post(args.oracle, "/detokenize", {"tokens": mine}).get("content", "")
    print(f"prompt tokens : {len(ids)}   n : {args.n}")
    print(f"match         : {match}/{min(len(mine), len(oracle_ids))} greedy tokens identical to the oracle")
    if match < min(len(mine), len(oracle_ids)):
        print(f"first divergence at index {match}: mine={mine[match]} oracle={oracle_ids[match]}")
    print(f"oracle : {oracle_text!r}")
    print(f"mine   : {my_text!r}")


if __name__ == "__main__":
    main()
