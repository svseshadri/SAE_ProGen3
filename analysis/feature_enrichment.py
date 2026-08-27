from __future__ import annotations

import argparse
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, pointbiserialr
from sklearn.metrics import roc_auc_score
from statsmodels.stats.multitest import multipletests

DEFAULT_METADATA_ID_COLUMNS = {"sequence_id", "accession", "entry_name", "id", "protein_id", "uniprot_id", "cluster_id", "cluster_rep"}


def infer_annotation_columns(seq_df: pd.DataFrame, excluded: Iterable[str] | None = None) -> list[str]:
    excluded = set(excluded or []) | DEFAULT_METADATA_ID_COLUMNS
    cols: list[str] = []
    for col in seq_df.columns:
        if col in excluded:
            continue
        s = seq_df[col].dropna()
        if s.empty:
            continue
        unique_vals = pd.Series(s).drop_duplicates()
        ok_as_binary = False
        if pd.api.types.is_numeric_dtype(s):
            ok_as_binary = unique_vals.shape[0] <= 2 and set(unique_vals.astype(int).tolist()).issubset({0, 1})
        elif pd.api.types.is_bool_dtype(s):
            ok_as_binary = True
        else:
            normalized = pd.Series(s).astype(str).str.lower()
            ok_as_binary = unique_vals.shape[0] <= 2 and normalized.isin({"true", "false", "0", "1", "y", "n", "yes", "no"}).all()
        if ok_as_binary:
            cols.append(col)
    return cols


def _coerce_binary_annotation(series: pd.Series) -> pd.Series:
    s = series.copy()
    normalized = s.astype(str).str.lower().str.strip()
    mapping = {"true": 1, "false": 0, "1": 1, "0": 0, "yes": 1, "no": 0, "y": 1, "n": 0}
    mapped = normalized.map(mapping)
    numeric = pd.to_numeric(mapped, errors="coerce")
    if numeric.notna().all():
        return numeric.astype(int)
    numeric2 = pd.to_numeric(normalized, errors="coerce")
    if numeric2.notna().all():
        return numeric2.astype(int)
    raise ValueError(f"Could not coerce annotation values to binary 0/1: {series.unique()[:10]}")


def _binary_annotation_table(seq_df: pd.DataFrame, annotation_cols: Iterable[str] | None = None) -> pd.DataFrame:
    out = seq_df.copy()
    annotation_cols = list(annotation_cols) if annotation_cols is not None else infer_annotation_columns(out)
    for col in annotation_cols:
        if col not in out.columns:
            raise KeyError(f"Annotation column '{col}' not found in sequence metadata.")
        out[col] = _coerce_binary_annotation(out[col])
    return out


def _per_seq_latent_features(token_df: pd.DataFrame, seq_df: pd.DataFrame, latent_ids: Iterable[int] | None = None) -> pd.DataFrame:
    required = ["sequence_id", "latent_idx", "activation"]
    missing = [c for c in required if c not in token_df.columns]
    if missing:
        raise ValueError(f"token_df must contain {required}. Missing: {missing}")

    latent_ids = list(latent_ids) if latent_ids is not None else sorted(token_df["latent_idx"].unique().tolist())
    token_df = token_df[token_df["latent_idx"].isin(latent_ids)].copy()

    seq_latent = (
        token_df.groupby(["sequence_id", "latent_idx"], as_index=False)
        .agg(
            any_active=("activation", lambda s: bool((s > 0).any())),
            fire_count=("activation", lambda s: int((s > 0).sum())),
            max_activation=("activation", "max"),
            mean_activation_all_tokens=("activation", "mean"),
            activation_sum=("activation", "sum"),
        )
    )

    seq_latent["mean_activation_active_only"] = np.where(
        seq_latent["fire_count"] > 0,
        seq_latent["activation_sum"] / seq_latent["fire_count"],
        0.0,
    )

    return seq_latent.merge(seq_df, on="sequence_id", how="left")


def _compute_annotation_statistics(seq_latent_df: pd.DataFrame, annotation_col: str) -> pd.DataFrame:
    rows: list[dict] = []
    for latent_idx, g in seq_latent_df.groupby("latent_idx"):
        g = g.sort_values("sequence_id").copy()
        binary = g[annotation_col].astype(int).to_numpy()
        any_active = g["any_active"].astype(int).to_numpy()
        fire_count = g["fire_count"].to_numpy(dtype=float)
        max_activation = g["max_activation"].to_numpy(dtype=float)

        if binary.size == 0 or np.all(np.unique(binary) == 0) or np.all(np.unique(binary) == 1):
            continue

        point_r = np.nan
        if np.unique(any_active).size > 1 and np.unique(binary).size > 1:
            try:
                point_r = float(pointbiserialr(binary, any_active).statistic)
            except Exception:
                point_r = np.nan

        auroc_any = np.nan
        auroc_fire = np.nan
        auroc_max = np.nan
        if np.unique(binary).size > 1:
            try:
                auroc_any = float(roc_auc_score(binary, any_active))
            except Exception:
                auroc_any = np.nan
            try:
                auroc_fire = float(roc_auc_score(binary, fire_count))
            except Exception:
                auroc_fire = np.nan
            try:
                auroc_max = float(roc_auc_score(binary, max_activation))
            except Exception:
                auroc_max = np.nan

        a = int(((binary == 1) & (any_active == 1)).sum())
        b = int(((binary == 0) & (any_active == 1)).sum())
        c = int(((binary == 1) & (any_active == 0)).sum())
        d = int(((binary == 0) & (any_active == 0)).sum())

        odds = np.nan
        fisher_p = np.nan
        if (a + b + c + d) > 0:
            try:
                odds, fisher_p = fisher_exact([[a, b], [c, d]], alternative="greater")
                odds = float(odds)
                fisher_p = float(fisher_p)
            except Exception:
                odds = np.nan
                fisher_p = np.nan

        rows.append(
            {
                "latent_idx": int(latent_idx),
                "annotation": annotation_col,
                "metric": "any_active",
                "pointbiserial_r": point_r,
                "auroc": auroc_any,
                "auroc_any_active": auroc_any,
                "auroc_fire_count": auroc_fire,
                "auroc_max_activation": auroc_max,
                "fisher_odds_ratio": odds,
                "fisher_pvalue": fisher_p,
                "n_pos": int((binary == 1).sum()),
                "n_neg": int((binary == 0).sum()),
                "hit_rate_pos": float(any_active[binary == 1].mean()) if (binary == 1).any() else np.nan,
                "hit_rate_neg": float(any_active[binary == 0].mean()) if (binary == 0).any() else np.nan,
                "fire_count_mean_pos": float(fire_count[binary == 1].mean()) if (binary == 1).any() else np.nan,
                "fire_count_mean_neg": float(fire_count[binary == 0].mean()) if (binary == 0).any() else np.nan,
                "max_activation_mean_pos": float(max_activation[binary == 1].mean()) if (binary == 1).any() else np.nan,
                "max_activation_mean_neg": float(max_activation[binary == 0].mean()) if (binary == 0).any() else np.nan,
            }
        )

    return pd.DataFrame(rows)


def compute_feature_enrichment(
    token_df: pd.DataFrame,
    seq_df: pd.DataFrame,
    latent_ids: Iterable[int] | None = None,
    annotation_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Compute latent-by-annotation enrichment statistics across all binary annotations."""
    annotation_cols = list(annotation_cols) if annotation_cols is not None else infer_annotation_columns(seq_df)
    seq_df = _binary_annotation_table(seq_df, annotation_cols=annotation_cols)
    seq_latent_df = _per_seq_latent_features(token_df, seq_df, latent_ids=latent_ids)

    results: list[pd.DataFrame] = []
    for annotation_col in annotation_cols:
        stats = _compute_annotation_statistics(seq_latent_df, annotation_col)
        if not stats.empty:
            results.append(stats)

    if not results:
        return pd.DataFrame(columns=[
            "latent_idx",
            "annotation",
            "metric",
            "pointbiserial_r",
            "auroc",
            "auroc_any_active",
            "auroc_fire_count",
            "auroc_max_activation",
            "fisher_odds_ratio",
            "fisher_pvalue",
            "q_value",
            "n_pos",
            "n_neg",
            "hit_rate_pos",
            "hit_rate_neg",
            "fire_count_mean_pos",
            "fire_count_mean_neg",
            "max_activation_mean_pos",
            "max_activation_mean_neg",
        ])

    combined = pd.concat(results, ignore_index=True)
    valid_mask = combined["fisher_pvalue"].notna()
    combined["q_value"] = np.nan
    if valid_mask.any():
        combined.loc[valid_mask, "q_value"] = multipletests(
            combined.loc[valid_mask, "fisher_pvalue"].to_numpy(),
            method="fdr_bh",
        )[1]

    combined = combined.sort_values(["annotation", "q_value", "fisher_pvalue"], na_position="last").reset_index(drop=True)
    return combined


def rank_feature_enrichment(results: pd.DataFrame) -> pd.DataFrame:
    df = results.copy()
    if df.empty:
        return df
    df["sort_key"] = df["q_value"].fillna(1.0)
    return df.sort_values(["annotation", "sort_key", "fisher_pvalue"], na_position="last").reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute latent feature enrichment against biological annotations.")
    parser.add_argument("--token-csv", type=str, required=True, help="CSV containing sequence_id, latent_idx, activation columns.")
    parser.add_argument("--metadata-csv", type=str, required=True, help="CSV containing sequence_id plus biological annotation columns.")
    parser.add_argument("--output-csv", type=str, required=True, help="Output CSV for ranked latent-annotation statistics.")
    parser.add_argument("--annotation-cols", nargs="*", default=None, help="Optional binary annotation columns to test.")
    parser.add_argument("--latent-ids", nargs="*", type=int, default=None, help="Optional subset of latent indices to evaluate.")
    args = parser.parse_args()

    token_df = pd.read_csv(args.token_csv)
    seq_df = pd.read_csv(args.metadata_csv)
    results = compute_feature_enrichment(
        token_df=token_df,
        seq_df=seq_df,
        latent_ids=args.latent_ids,
        annotation_cols=args.annotation_cols,
    )
    ranked = rank_feature_enrichment(results)
    ranked.to_csv(args.output_csv, index=False)
    print(f"Saved {len(ranked)} latent-annotation association rows to {args.output_csv}")


if __name__ == "__main__":
    main()
