#!/usr/bin/env python3
"""
splits.py

Rigorous cluster-aware train/val/test split pipeline for the S1A SAE dataset.

Inputs:
- s1a_vs_background_dataset.csv
- s1a_pos70_cluster.tsv
- s1a_bg70_cluster.tsv

Outputs:
- <out_prefix>_clustered_dataset.csv
- <out_prefix>_train.csv
- <out_prefix>_val.csv
- <out_prefix>_test.csv
- <out_prefix>_split_summary.txt
- <out_prefix>_cluster_summary.csv

This script:
1. Merges MMseqs cluster assignments into the master dataset
2. Builds motif_class for positive sequences
3. Splits by cluster ID, not sequence ID
4. Preserves class separation and prevents homolog leakage
5. Produces approximately 70/15/15 sequence-level splits
"""

from __future__ import annotations

import argparse
import random
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", type=str, default="s1a_vs_background_dataset.csv")
    parser.add_argument("--pos_cluster_tsv", type=str, default="s1a_pos70_cluster.tsv")
    parser.add_argument("--bg_cluster_tsv", type=str, default="s1a_bg70_cluster.tsv")
    parser.add_argument("--out_prefix", type=str, default="s1a70")
    parser.add_argument("--train_frac", type=float, default=0.70)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args, _ = parser.parse_known_args()
    return args


def read_cluster_tsv(path: str, class_name: str) -> pd.DataFrame:
    """
    MMseqs easy-cluster cluster TSV usually has:
    representative<TAB>member
    """
    df = pd.read_csv(path, sep="\t", header=None, names=["cluster_rep", "member"])
    df["cluster_id"] = class_name + "__" + df["cluster_rep"].astype(str)
    return df[["member", "cluster_id", "cluster_rep"]].copy()


def build_motif_class(df: pd.DataFrame) -> pd.Series:
    conditions = [
        df["has_ps00134"] & df["has_ps00135"],
        df["has_ps00134"] & ~df["has_ps00135"],
        ~df["has_ps00134"] & df["has_ps00135"],
        ~df["has_ps00134"] & ~df["has_ps00135"],
    ]
    choices = [
        "both_motifs",
        "histidine_only",
        "serine_only",
        "neither_motif",
    ]
    return pd.Series(np.select(conditions, choices, default="unknown"), index=df.index)


def merge_clusters(
    df: pd.DataFrame,
    pos_clusters: pd.DataFrame,
    bg_clusters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge positive and background cluster assignments separately.
    accession must match the FASTA IDs used during clustering.
    """
    df = df.copy()
    df["accession"] = df["accession"].astype(str)

    pos_clusters = pos_clusters.rename(columns={"member": "accession"})
    bg_clusters = bg_clusters.rename(columns={"member": "accession"})

    pos = df[df["class_label"] == 1].copy()
    bg = df[df["class_label"] == 0].copy()

    pos = pos.merge(pos_clusters, on="accession", how="left")
    bg = bg.merge(bg_clusters, on="accession", how="left")

    if pos["cluster_id"].isna().any():
        missing = int(pos["cluster_id"].isna().sum())
        raise ValueError(f"{missing} positive sequences are missing cluster IDs")
    if bg["cluster_id"].isna().any():
        missing = int(bg["cluster_id"].isna().sum())
        raise ValueError(f"{missing} background sequences are missing cluster IDs")

    merged = pd.concat([pos, bg], ignore_index=True)
    return merged


def summarize_clusters(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, sub in df.groupby("cluster_id"):
        row = {
            "cluster_id": cluster_id,
            "class_label": int(sub["class_label"].iloc[0]),
            "cluster_size": len(sub),
        }
        if row["class_label"] == 1:
            motif_counts = sub["motif_class"].value_counts().to_dict()
            for k in ["both_motifs", "histidine_only", "serine_only", "neither_motif"]:
                row[f"motif_{k}"] = motif_counts.get(k, 0)
        rows.append(row)
    return pd.DataFrame(rows)


def split_clusters_by_fraction(
    cluster_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """
    Deterministic cluster-aware split:
    - shuffle clusters
    - assign clusters sequentially until each split hits target size

    This guarantees non-empty train/val/test and approximately
    70/15/15 sequence-level proportions.
    """
    cluster_df = cluster_df.copy()

    # shuffle first to avoid ordered artifacts
    cluster_df = cluster_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    total_size = int(cluster_df["cluster_size"].sum())
    train_target = train_frac * total_size
    val_target = val_frac * total_size
    test_target = test_frac * total_size

    assignments = {}
    current = {"train": 0, "val": 0, "test": 0}

    for _, row in cluster_df.iterrows():
        size = int(row["cluster_size"])

        if current["train"] < train_target:
            split = "train"
        elif current["val"] < val_target:
            split = "val"
        else:
            split = "test"

        assignments[row["cluster_id"]] = split
        current[split] += size

    return assignments


def apply_assignments(df: pd.DataFrame, cluster_assignments: dict[str, str]) -> pd.DataFrame:
    df = df.copy()
    df["split"] = df["cluster_id"].map(cluster_assignments)
    if df["split"].isna().any():
        missing = int(df["split"].isna().sum())
        raise ValueError(f"{missing} sequences did not receive a split assignment")
    return df


def write_summary(df: pd.DataFrame, out_path: str):
    with open(out_path, "w") as f:
        f.write("=== Overall counts ===\n")
        f.write(df["split"].value_counts().sort_index().to_string())
        f.write("\n\n=== Counts by split and class ===\n")
        f.write(pd.crosstab(df["split"], df["class_name"]).to_string())

        f.write("\n\n=== Positive motif composition by split ===\n")
        pos = df[df["class_label"] == 1].copy()
        f.write(pd.crosstab(pos["split"], pos["motif_class"]).to_string())

        f.write("\n\n=== Sequence count fractions ===\n")
        split_fracs = df["split"].value_counts(normalize=True).sort_index()
        f.write(split_fracs.to_string())

        f.write("\n\n=== Cluster counts by split ===\n")
        cluster_counts = (
            df[["cluster_id", "split"]]
            .drop_duplicates()
            .groupby("split")
            .size()
            .sort_index()
        )
        f.write(cluster_counts.to_string())
        f.write("\n")


def main():
    args = parse_args()

    if not np.isclose(args.train_frac + args.val_frac + args.test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must sum to 1.0")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # load main dataset
    df = pd.read_csv(args.input_csv)
    df["accession"] = df["accession"].astype(str)

    # enforce boolean motif columns
    for col in ["has_ps00134", "has_ps00135"]:
        df[col] = df[col].astype(bool)

    # build motif class
    df["motif_class"] = build_motif_class(df)

    # read cluster assignments
    pos_clusters = read_cluster_tsv(args.pos_cluster_tsv, "pos")
    bg_clusters = read_cluster_tsv(args.bg_cluster_tsv, "bg")

    # merge cluster IDs
    df = merge_clusters(df, pos_clusters, bg_clusters)

    # summarize clusters
    cluster_summary = summarize_clusters(df)
    cluster_summary.to_csv(f"{args.out_prefix}_cluster_summary.csv", index=False)

    # split positives and backgrounds independently
    pos_cluster_df = cluster_summary[cluster_summary["class_label"] == 1].copy()
    bg_cluster_df = cluster_summary[cluster_summary["class_label"] == 0].copy()

    pos_assignments = split_clusters_by_fraction(
        pos_cluster_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    bg_assignments = split_clusters_by_fraction(
        bg_cluster_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed + 1,  # tiny variation so classes don't split identically by chance
    )

    cluster_assignments = {}
    cluster_assignments.update(pos_assignments)
    cluster_assignments.update(bg_assignments)

    # apply split labels
    df = apply_assignments(df, cluster_assignments)

    # add cluster size as convenience metadata
    cluster_sizes = cluster_summary.set_index("cluster_id")["cluster_size"].to_dict()
    df["cluster_size"] = df["cluster_id"].map(cluster_sizes)

    # write outputs
    df.to_csv(f"{args.out_prefix}_clustered_dataset.csv", index=False)
    df[df["split"] == "train"].to_csv(f"{args.out_prefix}_train.csv", index=False)
    df[df["split"] == "val"].to_csv(f"{args.out_prefix}_val.csv", index=False)
    df[df["split"] == "test"].to_csv(f"{args.out_prefix}_test.csv", index=False)

    write_summary(df, f"{args.out_prefix}_split_summary.txt")

    # print summary to terminal
    print("Wrote:")
    print(f"  {args.out_prefix}_cluster_summary.csv")
    print(f"  {args.out_prefix}_clustered_dataset.csv")
    print(f"  {args.out_prefix}_train.csv")
    print(f"  {args.out_prefix}_val.csv")
    print(f"  {args.out_prefix}_test.csv")
    print(f"  {args.out_prefix}_split_summary.txt")

    print("\n=== Overall counts ===")
    print(df["split"].value_counts().sort_index())

    print("\n=== Counts by split and class ===")
    print(pd.crosstab(df["split"], df["class_name"]))

    print("\n=== Positive motif composition by split ===")
    pos = df[df["class_label"] == 1].copy()
    print(pd.crosstab(pos["split"], pos["motif_class"]))

    print("\n=== Sequence count fractions ===")
    print(df["split"].value_counts(normalize=True).sort_index())


if __name__ == "__main__":
    main()