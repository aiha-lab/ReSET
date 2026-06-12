#!/usr/bin/env python3
"""Compute the global entropy threshold tau_0 from an entropy JSONL.

Following the paper, tau_0 is the 80th percentile of the NVFP4 token-entropy
distribution (over the NuminaMath calibration subset). Input JSONL is produced
by dump_entropy.py:
  {"idx": int, "correct": bool|None, "n_tokens": int, "entropies": [float, ...]}

Usage:
  python calibrate.py --nvfp4 nvfp4.jsonl            # tau_0 = NVFP4 80th percentile
  python calibrate.py --nvfp4 nvfp4.jsonl --bf16 bf16.jsonl   # + BF16 comparison
"""
import argparse
import json
from pathlib import Path

import numpy as np


def load(path: Path):
    ents_per_prob = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            ents_per_prob.append(np.array(d["entropies"], dtype=np.float64))
    return ents_per_prob


def summarize(name: str, ents_per_prob):
    flat = np.concatenate(ents_per_prob) if ents_per_prob else np.array([])
    print("=" * 72)
    print(f"{name}")
    print("=" * 72)
    print(f"  problems     : {len(ents_per_prob)}")
    print(f"  total tokens : {flat.size:,}")
    if flat.size == 0:
        return None
    print()
    print(f"  Entropy summary (nats)")
    print(f"    mean   = {flat.mean():.4f}")
    print(f"    median = {np.median(flat):.4f}")
    for p in (50, 60, 70, 75, 80, 85, 90, 95, 99):
        print(f"    p{p:<3d}  = {np.percentile(flat, p):.4f}")
    print(f"    max    = {flat.max():.4f}")
    print()
    return dict(p80=float(np.percentile(flat, 80)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nvfp4", required=True, help="NVFP4 entropy JSONL (tau_0 source)")
    ap.add_argument("--bf16", default=None, help="optional BF16 entropy JSONL (comparison)")
    args = ap.parse_args()

    nvfp4 = summarize("NVFP4", load(Path(args.nvfp4)))
    if args.bf16:
        summarize("BF16", load(Path(args.bf16)))

    print("=" * 72)
    if nvfp4 is not None:
        print(f"tau_0 = NVFP4 80th-percentile token entropy = {nvfp4['p80']:.4f}")


if __name__ == "__main__":
    main()
