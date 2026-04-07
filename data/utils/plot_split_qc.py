#!/usr/bin/env python3
"""
plot_split_qc.py

Generate QC plots for the clustered train/val/test split.

Usage:
    python scripts/plot_split_qc.py \
        --input_csv data/processed/s1a70/s1a70_clustered_dataset.csv \
        --outdir plots/split_qc/s1a70
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SPLIT_ORDER = ["train", "val", "test"]
CLASS_ORDER = ["S1A_trypsin_chymotrypsin", "background"]
MOTIF_ORDER = ["both_motifs", "histidine_only", "serine_only", "neither_motif"]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_csv",
        type=str,
        required=True,
        help="Path to clustered dataset CSV with split assignments.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory to save split QC plots.",
    )
    args, _ = parser.parse_known_args()
    return args


def ensure_outdir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def savefig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_total_counts(df: pd.DataFrame, outdir: str):
    counts = df["split"].value_counts().reindex(SPLIT_ORDER)

    plt.figure(figsize=(7, 5))
    bars = plt.bar(counts.index, counts.values)
    plt.xlabel("Dataset split")
    plt.ylabel("Number of sequences")
    plt.title("Total number of sequences in each split")

    for bar, val in zip(bars, counts.values):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:,}",
                 ha="center", va="bottom", fontsize=10)

    savefig(os.path.join(outdir, "01_total_counts_by_split.png"))


def plot_class_balance(df: pd.DataFrame, outdir: str):
    tab = pd.crosstab(df["split"], df["class_name"]).reindex(index=SPLIT_ORDER, columns=CLASS_ORDER)

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))

    for col in tab.columns:
        vals = tab[col].values
        plt.bar(x, vals, bottom=bottom, label=col)
        bottom += vals

    plt.xticks(x, tab.index)
    plt.xlabel("Dataset split")
    plt.ylabel("Number of sequences")
    plt.title("Class balance across train/val/test splits")
    plt.legend(title="Class")

    savefig(os.path.join(outdir, "02_class_balance_by_split.png"))


def plot_class_balance_fraction(df: pd.DataFrame, outdir: str):
    tab = pd.crosstab(df["split"], df["class_name"], normalize="index").reindex(index=SPLIT_ORDER, columns=CLASS_ORDER)

    plt.figure(figsize=(8, 5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))

    for col in tab.columns:
        vals = tab[col].values
        plt.bar(x, vals, bottom=bottom, label=col)
        bottom += vals

    plt.xticks(x, tab.index)
    plt.xlabel("Dataset split")
    plt.ylabel("Fraction of sequences")
    plt.title("Class fractions across train/val/test splits")
    plt.legend(title="Class")
    plt.ylim(0, 1)

    savefig(os.path.join(outdir, "03_class_fraction_by_split.png"))


def plot_positive_motif_composition(df: pd.DataFrame, outdir: str):
    pos = df[df["class_label"] == 1].copy()
    tab = pd.crosstab(pos["split"], pos["motif_class"]).reindex(index=SPLIT_ORDER, columns=MOTIF_ORDER)

    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))

    for col in tab.columns:
        vals = tab[col].values
        plt.bar(x, vals, bottom=bottom, label=col)
        bottom += vals

    plt.xticks(x, tab.index)
    plt.xlabel("Dataset split")
    plt.ylabel("Number of positive sequences")
    plt.title("Positive-class motif composition across train/val/test splits")
    plt.legend(title="Motif class")

    savefig(os.path.join(outdir, "04_positive_motif_counts_by_split.png"))


def plot_positive_motif_fraction(df: pd.DataFrame, outdir: str):
    pos = df[df["class_label"] == 1].copy()
    tab = pd.crosstab(pos["split"], pos["motif_class"], normalize="index").reindex(index=SPLIT_ORDER, columns=MOTIF_ORDER)

    plt.figure(figsize=(9, 5))
    bottom = np.zeros(len(tab))
    x = np.arange(len(tab.index))

    for col in tab.columns:
        vals = tab[col].values
        plt.bar(x, vals, bottom=bottom, label=col)
        bottom += vals

    plt.xticks(x, tab.index)
    plt.xlabel("Dataset split")
    plt.ylabel("Fraction of positive sequences")
    plt.title("Positive-class motif fractions across train/val/test splits")
    plt.legend(title="Motif class")
    plt.ylim(0, 1)

    savefig(os.path.join(outdir, "05_positive_motif_fraction_by_split.png"))


def plot_length_distributions(df: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, class_name in zip(axes, CLASS_ORDER):
        sub = df[df["class_name"] == class_name].copy()
        bins = np.arange(int(sub["length"].min()), int(sub["length"].max()) + 11, 10)

        for split in SPLIT_ORDER:
            vals = sub[sub["split"] == split]["length"]
            ax.hist(vals, bins=bins, alpha=0.45, label=split)

        ax.set_title(class_name)
        ax.set_xlabel("Sequence length (amino acids)")
        ax.set_ylabel("Number of sequences")
        ax.legend(title="Split")

    fig.suptitle("Sequence length distributions by split and class")
    savefig(os.path.join(outdir, "06_length_distributions_by_split_and_class.png"))


def plot_length_boxplot(df: pd.DataFrame, outdir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, class_name in zip(axes, CLASS_ORDER):
        sub = df[df["class_name"] == class_name].copy()
        data = [sub[sub["split"] == split]["length"].values for split in SPLIT_ORDER]

        ax.boxplot(data, tick_labels=SPLIT_ORDER, showfliers=False)
        ax.set_title(class_name)
        ax.set_xlabel("Dataset split")
        ax.set_ylabel("Sequence length (amino acids)")

    fig.suptitle("Sequence length boxplots by split and class")
    savefig(os.path.join(outdir, "07_length_boxplots_by_split_and_class.png"))


def plot_cluster_size_distribution(df: pd.DataFrame, outdir: str):
    cluster_df = df[["cluster_id", "cluster_size", "split", "class_name"]].drop_duplicates().copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    for ax, class_name in zip(axes, CLASS_ORDER):
        sub = cluster_df[cluster_df["class_name"] == class_name].copy()
        data = [sub[sub["split"] == split]["cluster_size"].values for split in SPLIT_ORDER]

        ax.boxplot(data, tick_labels=SPLIT_ORDER, showfliers=False)
        ax.set_title(class_name)
        ax.set_xlabel("Dataset split")
        ax.set_ylabel("Cluster size (number of sequences)")

    fig.suptitle("Cluster size distributions by split and class")
    savefig(os.path.join(outdir, "08_cluster_size_boxplots_by_split_and_class.png"))


def write_summary_tables(df: pd.DataFrame, outdir: str):
    total_counts = df["split"].value_counts().reindex(SPLIT_ORDER)
    class_counts = pd.crosstab(df["split"], df["class_name"]).reindex(index=SPLIT_ORDER, columns=CLASS_ORDER)
    class_fracs = pd.crosstab(df["split"], df["class_name"], normalize="index").reindex(index=SPLIT_ORDER, columns=CLASS_ORDER)

    pos = df[df["class_label"] == 1].copy()
    motif_counts = pd.crosstab(pos["split"], pos["motif_class"]).reindex(index=SPLIT_ORDER, columns=MOTIF_ORDER)
    motif_fracs = pd.crosstab(pos["split"], pos["motif_class"], normalize="index").reindex(index=SPLIT_ORDER, columns=MOTIF_ORDER)

    cluster_df = df[["cluster_id", "cluster_size", "split", "class_name"]].drop_duplicates().copy()
    cluster_summary = (
        cluster_df.groupby(["split", "class_name"])["cluster_size"]
        .describe()
        .reset_index()
    )

    with open(os.path.join(outdir, "split_qc_summary.txt"), "w") as f:
        f.write("=== Total counts by split ===\n")
        f.write(total_counts.to_string())
        f.write("\n\n=== Class counts by split ===\n")
        f.write(class_counts.to_string())
        f.write("\n\n=== Class fractions by split ===\n")
        f.write(class_fracs.to_string())
        f.write("\n\n=== Positive motif counts by split ===\n")
        f.write(motif_counts.to_string())
        f.write("\n\n=== Positive motif fractions by split ===\n")
        f.write(motif_fracs.to_string())
        f.write("\n\n=== Cluster size summary by split and class ===\n")
        f.write(cluster_summary.to_string(index=False))


def main():
    args = parse_args()
    ensure_outdir(args.outdir)

    df = pd.read_csv(args.input_csv)

    # Defensive typing
    df["class_label"] = pd.to_numeric(df["class_label"], errors="coerce").astype(int)
    df["length"] = pd.to_numeric(df["length"], errors="coerce")
    df["cluster_size"] = pd.to_numeric(df["cluster_size"], errors="coerce")

    plot_total_counts(df, args.outdir)
    plot_class_balance(df, args.outdir)
    plot_class_balance_fraction(df, args.outdir)
    plot_positive_motif_composition(df, args.outdir)
    plot_positive_motif_fraction(df, args.outdir)
    plot_length_distributions(df, args.outdir)
    plot_length_boxplot(df, args.outdir)
    plot_cluster_size_distribution(df, args.outdir)
    write_summary_tables(df, args.outdir)

    print(f"Wrote split QC plots and summary to: {args.outdir}")


if __name__ == "__main__":
    main()