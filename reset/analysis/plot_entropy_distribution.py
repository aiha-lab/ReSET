#!/usr/bin/env python3
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_entropies(jsonl_path):
    all_ent = []
    n_correct, n_total = 0, 0
    with open(jsonl_path) as f:
        for line in f:
            d = json.loads(line)
            all_ent.extend(d["entropies"])
            n_correct += int(bool(d.get("correct")))
            n_total += 1
    return np.array(all_ent), n_correct, n_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-path", default="outputs/entropy/bf16-seed42-10/AIME-90.jsonl")
    parser.add_argument("--nvfp4-path", default="outputs/entropy/nvfp4-seed42-10/AIME-90.jsonl")
    parser.add_argument("--corrected-path", default=None,
                        help="Optional: NVFP4 + temperature correction entropy JSONL")
    parser.add_argument("--output-dir", default="outputs/entropy_analysis")
    parser.add_argument("--label", default="model, calibration set",
                        help="dataset/model label shown in the overlay figure title")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    bf16_ent, bf16_correct, bf16_total = load_entropies(args.bf16_path)
    nvfp4_ent, nvfp4_correct, nvfp4_total = load_entropies(args.nvfp4_path)

    corrected_ent, corrected_correct, corrected_total = None, 0, 0
    if args.corrected_path:
        corrected_ent, corrected_correct, corrected_total = load_entropies(args.corrected_path)

    bf16_p80 = np.percentile(bf16_ent, 80)
    nvfp4_p80 = np.percentile(nvfp4_ent, 80)

    print(f"BF16:  {len(bf16_ent):,} tokens from {bf16_total} problems ({bf16_correct}/{bf16_total} correct)")
    print(f"  mean={bf16_ent.mean():.4f}  median={np.median(bf16_ent):.4f}  p80={bf16_p80:.4f}  max={bf16_ent.max():.4f}")
    print(f"NVFP4: {len(nvfp4_ent):,} tokens from {nvfp4_total} problems ({nvfp4_correct}/{nvfp4_total} correct)")
    print(f"  mean={nvfp4_ent.mean():.4f}  median={np.median(nvfp4_ent):.4f}  p80={nvfp4_p80:.4f}  max={nvfp4_ent.max():.4f}")
    if corrected_ent is not None:
        corr_p80 = np.percentile(corrected_ent, 80)
        print(f"NVFP4+T: {len(corrected_ent):,} tokens from {corrected_total} problems ({corrected_correct}/{corrected_total} correct)")
        print(f"  mean={corrected_ent.mean():.4f}  median={np.median(corrected_ent):.4f}  p80={corr_p80:.4f}  max={corrected_ent.max():.4f}")

    # ---- Figure 1: Side-by-side BF16 vs NVFP4 entropy distributions ----
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    bins = np.linspace(0, 4.5, 80)

    for ax, ent, p80, label, color in [
        (axes[0], bf16_ent, bf16_p80, "BF16", "#4A90D9"),
        (axes[1], nvfp4_ent, nvfp4_p80, "NVFP4 (W4A4)", "#D94A4A"),
    ]:
        ax.hist(ent, bins=bins, color=color, alpha=0.85, edgecolor="white", linewidth=0.3)
        ax.axvline(p80, color="red", linestyle="--", linewidth=2.5,
                   label=f"80th percentile: {p80:.3f}")
        ax.set_yscale("log")
        ax.set_xlabel("Entropy (nats)", fontsize=18)
        ax.set_ylabel("Frequency (log scale)", fontsize=18)
        ax.set_title(f"{label} — Token Entropy Distribution", fontsize=20, fontweight="bold")
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=16, loc="upper right")
        ax.set_xlim(0, 4.5)
        ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout(w_pad=3)
    out1 = os.path.join(args.output_dir, "entropy_distribution_sidebyside.png")
    fig.savefig(out1, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {out1}")

    # ---- Figure 2: Overlaid histogram ----
    fig, ax = plt.subplots(figsize=(14, 9))

    ax.hist(bf16_ent, bins=bins, color="#4A90D9", alpha=0.7, edgecolor="white",
            linewidth=0.3, label=f"BF16 (p80={bf16_p80:.3f})")
    ax.hist(nvfp4_ent, bins=bins, color="#D94A4A", alpha=0.5, edgecolor="white",
            linewidth=0.3, label=f"NVFP4 (p80={nvfp4_p80:.3f})")
    ax.axvline(bf16_p80, color="#4A90D9", linestyle="--", linewidth=2.5)
    ax.axvline(nvfp4_p80, color="#D94A4A", linestyle="--", linewidth=2.5)

    ax.set_yscale("log")
    ax.set_xlabel("Entropy (nats)", fontsize=18)
    ax.set_ylabel("Frequency (log scale)", fontsize=18)
    ax.set_title("BF16 vs NVFP4 — Token Entropy Distribution\n"
                 f"({args.label}, {bf16_total} problems)",
                 fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=16, loc="upper right")
    ax.set_xlim(0, 4.5)
    ax.grid(True, alpha=0.2, which="both")

    plt.tight_layout()
    out2 = os.path.join(args.output_dir, "entropy_distribution_overlay.png")
    fig.savefig(out2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out2}")

    # ---- Figure 3: CDF comparison ----
    fig, ax = plt.subplots(figsize=(14, 9))

    cdf_entries = [
        (bf16_ent, "BF16", "#4A90D9", bf16_p80),
        (nvfp4_ent, "NVFP4", "#D94A4A", nvfp4_p80),
    ]
    if corrected_ent is not None:
        corr_p80 = np.percentile(corrected_ent, 80)
        cdf_entries.append(
            (corrected_ent, "NVFP4 + Temp Correction", "#2EA043", corr_p80))

    for ent, label, color, p80 in cdf_entries:
        sorted_ent = np.sort(ent)
        cdf = np.arange(1, len(sorted_ent) + 1) / len(sorted_ent)
        ax.plot(sorted_ent, cdf, linewidth=2.5, color=color,
                label=f"{label} (p80={p80:.3f})")
        ax.axvline(p80, color=color, linestyle="--", linewidth=2, alpha=0.7)

    ax.axhline(0.8, color="gray", linestyle=":", linewidth=1.5, alpha=0.5, label="80% line")
    ax.set_xlabel("Entropy (nats)", fontsize=18)
    ax.set_ylabel("Cumulative Proportion", fontsize=18)
    n_models = "BF16 vs NVFP4 vs Corrected" if corrected_ent is not None else "BF16 vs NVFP4"
    ax.set_title(f"CDF of Token Entropy — {n_models}", fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    ax.legend(fontsize=16, loc="lower right")
    ax.set_xlim(0, 4.5)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out3 = os.path.join(args.output_dir, "entropy_cdf.png")
    fig.savefig(out3, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out3}")

    # ---- Print detailed percentile table ----
    print("\n" + "=" * 70)
    print("Percentile Comparison")
    print("=" * 70)
    has_corr = corrected_ent is not None
    header = f"{'Percentile':>12s}  {'BF16':>10s}  {'NVFP4':>10s}  {'Shift':>10s}"
    if has_corr:
        header += f"  {'Corrected':>10s}  {'Corr Shift':>10s}"
    print(header)
    print("-" * (50 + 24 if has_corr else 50))
    for p in [10, 20, 30, 40, 50, 60, 70, 75, 80, 85, 90, 95, 99]:
        b = np.percentile(bf16_ent, p)
        n = np.percentile(nvfp4_ent, p)
        line = f"  p{p:>3d}        {b:10.4f}  {n:10.4f}  {n-b:+10.4f}"
        if has_corr:
            c = np.percentile(corrected_ent, p)
            line += f"  {c:10.4f}  {c-b:+10.4f}"
        print(line)

    # fraction below various thresholds
    print("\n" + "=" * 70)
    print("Fraction of tokens below threshold")
    print("=" * 70)
    for thresh in [0.1, 0.3, 0.5, bf16_p80, 0.672, nvfp4_p80, 1.0, 2.0]:
        fb = (bf16_ent < thresh).mean() * 100
        fn = (nvfp4_ent < thresh).mean() * 100
        line = f"  h < {thresh:.3f}:  BF16={fb:5.1f}%   NVFP4={fn:5.1f}%"
        if has_corr:
            fc = (corrected_ent < thresh).mean() * 100
            line += f"   Corrected={fc:5.1f}%"
        print(line)


if __name__ == "__main__":
    main()
