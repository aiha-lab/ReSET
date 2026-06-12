#!/usr/bin/env python3
"""Batch-size latency table + TPOT plot from e2e sweep results."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--data-file", default=None,
                help="Path to sweep .jsonl file. Defaults to sweep_20260503_053203.jsonl (8B).")
ap.add_argument("--model-label", default=None,
                help="Human-readable model name for plot title (default: inferred from filename).")
ap.add_argument("--out-prefix", default=None,
                help="Output file prefix (default: same dir as data file).")
args = ap.parse_args()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
RESULTS_DIR = Path(__file__).parent / "results" / "e2e_bs_sweep"

if args.data_file:
    DATA_FILE = Path(args.data_file)
else:
    DATA_FILE = RESULTS_DIR / "sweep_20260503_053203.jsonl"

# Infer model label from filename if not given
if args.model_label:
    MODEL_LABEL = args.model_label
elif "32b" in DATA_FILE.name.lower():
    MODEL_LABEL = "Qwen3-32B"
else:
    MODEL_LABEL = "Qwen3-8B"

OUT_DIR = DATA_FILE.parent if args.out_prefix is None else Path(args.out_prefix).parent
OUT_STEM = DATA_FILE.stem if args.out_prefix is None else Path(args.out_prefix).name

rows = [json.loads(l) for l in DATA_FILE.open() if l.strip()]
df = pd.DataFrame([r for r in rows if r.get("status", "ok") == "ok"])

BACKEND_LABEL = {
    "bf16":       "BF16",
    "cutlass":    "CUTLASS FP4",
    "flashinfer": "FlashInfer FP4",
    "nvfp4r":     "nvfp4r",
}
BACKEND_COLOR = {
    "bf16":       "#555555",
    "cutlass":    "#2196F3",
    "flashinfer": "#FF9800",
    "nvfp4r":     "#E53935",
}
BACKEND_MARKER = {
    "bf16":       "o",
    "cutlass":    "s",
    "flashinfer": "^",
    "nvfp4r":     "D",
}
BACKEND_ORDER = ["bf16", "cutlass", "flashinfer", "nvfp4r"]

batch_sizes  = sorted(df["batch_size"].unique())
output_lens  = sorted(df["output_len"].unique())
backends     = [b for b in BACKEND_ORDER if b in df["backend"].unique()]

# ---------------------------------------------------------------------------
# 1. Latency table  (ms_per_tok  ×  batch_size, one sub-table per output_len)
# ---------------------------------------------------------------------------
print("=" * 80)
print("TPOT (ms/token) by backend and batch size")
print("=" * 80)

pivot_frames = {}
for olen in output_lens:
    sub = df[df["output_len"] == olen]
    pivot = sub.pivot(index="batch_size", columns="backend", values="ms_per_tok")
    pivot = pivot[[b for b in backends if b in pivot.columns]]
    pivot.columns = [BACKEND_LABEL[b] for b in pivot.columns]
    pivot.index.name = "batch_size"
    pivot_frames[olen] = pivot

    print(f"\n  output_len = {olen:,} tokens")
    print(pivot.to_string(float_format=lambda x: f"{x:.2f}"))

    # also print relative increase vs BS=1
    base = pivot.loc[pivot.index[0]]
    rel = pivot.div(base)
    print(f"\n  Relative TPOT (BS=1 → 1.00x):")
    print(rel.to_string(float_format=lambda x: f"{x:.2f}x"))

# Average over output_lens for a single summary table
avg_pivot = sum(pivot_frames.values()) / len(pivot_frames)
print("\n" + "=" * 80)
print("Average TPOT (ms/token) across output lengths")
print("=" * 80)
print(avg_pivot.to_string(float_format=lambda x: f"{x:.2f}"))

# ---------------------------------------------------------------------------
# 2. TPOT plot  —  one panel per output_len, x=batch_size, y=ms_per_tok
# ---------------------------------------------------------------------------
n_panels = len(output_lens)
fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                         sharey=False, constrained_layout=True)

if n_panels == 1:
    axes = [axes]

for ax, olen in zip(axes, output_lens):
    sub = df[df["output_len"] == olen]
    for backend in backends:
        bdf = sub[sub["backend"] == backend].sort_values("batch_size")
        if bdf.empty:
            continue
        ax.plot(
            bdf["batch_size"], bdf["ms_per_tok"],
            label=BACKEND_LABEL[backend],
            color=BACKEND_COLOR[backend],
            marker=BACKEND_MARKER[backend],
            linewidth=2,
            markersize=7,
        )

    ax.set_title(f"output_len = {olen:,}", fontsize=11)
    ax.set_xlabel("Batch size", fontsize=10)
    ax.set_ylabel("TPOT (ms / token)", fontsize=10)
    ax.set_xticks(batch_sizes)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig.suptitle(f"Time Per Output Token (TPOT) vs Batch Size\n({MODEL_LABEL}, input_len=512)",
             fontsize=13, fontweight="bold")

out_plot = OUT_DIR / f"{OUT_STEM}_tpot_vs_batchsize.pdf"
fig.savefig(out_plot, bbox_inches="tight")
print(f"\nSaved plot → {out_plot}")

# ---------------------------------------------------------------------------
# 3. Throughput plot  (tok/s = batch_size * output_len / wall_s)
# ---------------------------------------------------------------------------
fig2, axes2 = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4.5),
                            sharey=False, constrained_layout=True)
if n_panels == 1:
    axes2 = [axes2]

for ax, olen in zip(axes2, output_lens):
    sub = df[df["output_len"] == olen]
    for backend in backends:
        bdf = sub[sub["backend"] == backend].sort_values("batch_size")
        if bdf.empty:
            continue
        ax.plot(
            bdf["batch_size"], bdf["tok_per_s"],
            label=BACKEND_LABEL[backend],
            color=BACKEND_COLOR[backend],
            marker=BACKEND_MARKER[backend],
            linewidth=2,
            markersize=7,
        )

    ax.set_title(f"output_len = {olen:,}", fontsize=11)
    ax.set_xlabel("Batch size", fontsize=10)
    ax.set_ylabel("Throughput (tok / s)", fontsize=10)
    ax.set_xticks(batch_sizes)
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")

fig2.suptitle(f"Throughput vs Batch Size\n({MODEL_LABEL}, input_len=512)",
              fontsize=13, fontweight="bold")

out_plot2 = OUT_DIR / f"{OUT_STEM}_throughput_vs_batchsize.pdf"
fig2.savefig(out_plot2, bbox_inches="tight")
print(f"Saved plot → {out_plot2}")
