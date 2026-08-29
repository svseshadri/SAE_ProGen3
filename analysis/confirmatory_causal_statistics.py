from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


DEFAULT_INPUT = "results/causal_feature_dose_response.csv"
DEFAULT_SUMMARY_OUTPUT = "results/confirmatory_causal_statistics.csv"
DEFAULT_BOOTSTRAP_OUTPUT = "results/confirmatory_causal_bootstrap.csv"
DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_BOOTSTRAP_SEED = 3256
DEFAULT_MATCHED_KL_TARGET = 0.025

REQUIRED_COLUMNS = {
    "sequence_id",
    "feature_id",
    "matched_concept",
    "concept_positive",
    "dose",
    "motif_specificity_score",
}

BIOLOGICAL_CANDIDATE_FEATURES = (3256, 2942, 1644, 727)
CONTROL_FEATURES = (1,)


@dataclass(frozen=True)
class FitResult:
    feature_id: int
    matched_concept: str
    n_rows: int
    n_sequences: int
    n_concept_positive_sequences: int
    n_concept_negative_sequences: int
    beta_dose: float
    beta_concept: float
    beta_dose_x_concept: float
    interaction_clustered_se: float
    interaction_ci_low: float
    interaction_ci_high: float
    interaction_p: float
    interaction_term: str
    estimable: bool
    non_estimable_reason: str


def load_causal_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    out = df.copy()
    out["feature_id"] = out["feature_id"].astype(int)
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["concept_positive"] = _coerce_bool(out["concept_positive"])
    out["concept_positive_int"] = out["concept_positive"].astype(int)
    out["dose_centered"] = pd.to_numeric(out["dose"], errors="raise")
    out["motif_specificity_score"] = pd.to_numeric(out["motif_specificity_score"], errors="raise")
    return out


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    coerced = normalized.map(mapping)
    if coerced.isna().any():
        bad = series[coerced.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Could not coerce concept_positive values to bool: {bad}")
    return coerced.astype(bool)


def inspect_dataset(df: pd.DataFrame) -> dict:
    support = support_by_feature(df)
    missing = {col: int(df[col].isna().sum()) for col in REQUIRED_COLUMNS}
    dose_support = (
        df.groupby(["feature_id", "concept_positive", "dose"], dropna=False)
        .size()
        .rename("n_rows")
        .reset_index()
        .to_dict(orient="records")
    )
    return {
        "columns": list(df.columns),
        "feature_ids": sorted(int(v) for v in df["feature_id"].dropna().unique()),
        "matched_concepts": sorted(str(v) for v in df["matched_concept"].dropna().unique()),
        "dose_values": sorted(float(v) for v in df["dose"].dropna().unique()),
        "n_rows": int(len(df)),
        "n_sequences": int(df["sequence_id"].nunique()),
        "required_column_missing_values": missing,
        "support_by_feature": support.to_dict(orient="records"),
        "dose_support": dose_support,
        "has_native_dose_zero": bool((df["dose"] == 0).any()),
        "has_signed_suppression_and_amplification": bool((df["dose"] < 0).any() and (df["dose"] > 0).any()),
    }


def support_by_feature(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for feature_id, g in df.groupby("feature_id", sort=True):
        pos_seq = g.loc[g["concept_positive"], "sequence_id"].nunique()
        neg_seq = g.loc[~g["concept_positive"], "sequence_id"].nunique()
        rows.append(
            {
                "feature_id": int(feature_id),
                "matched_concept": str(g["matched_concept"].iloc[0]),
                "n_rows": int(len(g)),
                "n_sequences": int(g["sequence_id"].nunique()),
                "n_concept_positive_sequences": int(pos_seq),
                "n_concept_negative_sequences": int(neg_seq),
                "dose_values": ";".join(str(float(v)) for v in sorted(g["dose"].unique())),
            }
        )
    return pd.DataFrame(rows)


def is_estimable_feature(feature_df: pd.DataFrame) -> tuple[bool, str]:
    if feature_df["concept_positive"].nunique() < 2:
        return False, "missing concept-positive or concept-negative cohort"
    if feature_df.loc[feature_df["concept_positive"], "sequence_id"].nunique() == 0:
        return False, "missing concept-positive sequence support"
    if feature_df.loc[~feature_df["concept_positive"], "sequence_id"].nunique() == 0:
        return False, "missing concept-negative sequence support"
    if feature_df["dose_centered"].nunique() < 2:
        return False, "fewer than two dose values"
    design = np.column_stack(
        [
            np.ones(len(feature_df)),
            feature_df["dose_centered"].to_numpy(dtype=float),
            feature_df["concept_positive_int"].to_numpy(dtype=float),
            (feature_df["dose_centered"] * feature_df["concept_positive_int"]).to_numpy(dtype=float),
        ]
    )
    if np.linalg.matrix_rank(design) < design.shape[1]:
        return False, "singular fixed-effects interaction design"
    return True, ""


def _series_from_result(values: Iterable[float], names: list[str]) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=names)


def _interaction_term(exog_names: list[str]) -> str:
    candidates = [name for name in exog_names if ":" in name and "dose_centered" in name and "concept_positive_int" in name]
    if len(candidates) != 1:
        raise ValueError(f"Could not identify a unique dose x concept interaction term from {exog_names}")
    return candidates[0]


def fit_clustered_interaction(feature_df: pd.DataFrame) -> FitResult:
    feature_id = int(feature_df["feature_id"].iloc[0])
    matched_concept = str(feature_df["matched_concept"].iloc[0])
    estimable, reason = is_estimable_feature(feature_df)
    n_pos = int(feature_df.loc[feature_df["concept_positive"], "sequence_id"].nunique())
    n_neg = int(feature_df.loc[~feature_df["concept_positive"], "sequence_id"].nunique())
    if not estimable:
        return FitResult(
            feature_id=feature_id,
            matched_concept=matched_concept,
            n_rows=int(len(feature_df)),
            n_sequences=int(feature_df["sequence_id"].nunique()),
            n_concept_positive_sequences=n_pos,
            n_concept_negative_sequences=n_neg,
            beta_dose=np.nan,
            beta_concept=np.nan,
            beta_dose_x_concept=np.nan,
            interaction_clustered_se=np.nan,
            interaction_ci_low=np.nan,
            interaction_ci_high=np.nan,
            interaction_p=np.nan,
            interaction_term="",
            estimable=False,
            non_estimable_reason=reason,
        )

    model = smf.ols("motif_specificity_score ~ dose_centered * concept_positive_int", data=feature_df).fit()
    robust = model.get_robustcov_results(cov_type="cluster", groups=feature_df["sequence_id"])
    names = list(model.model.exog_names)
    params = _series_from_result(robust.params, names)
    bse = _series_from_result(robust.bse, names)
    pvalues = _series_from_result(robust.pvalues, names)
    conf = pd.DataFrame(np.asarray(robust.conf_int(), dtype=float), index=names, columns=["low", "high"])
    interaction = _interaction_term(names)
    return FitResult(
        feature_id=feature_id,
        matched_concept=matched_concept,
        n_rows=int(len(feature_df)),
        n_sequences=int(feature_df["sequence_id"].nunique()),
        n_concept_positive_sequences=n_pos,
        n_concept_negative_sequences=n_neg,
        beta_dose=float(params["dose_centered"]),
        beta_concept=float(params["concept_positive_int"]),
        beta_dose_x_concept=float(params[interaction]),
        interaction_clustered_se=float(bse[interaction]),
        interaction_ci_low=float(conf.loc[interaction, "low"]),
        interaction_ci_high=float(conf.loc[interaction, "high"]),
        interaction_p=float(pvalues[interaction]),
        interaction_term=interaction,
        estimable=True,
        non_estimable_reason="",
    )


def bootstrap_sequence_interactions(
    feature_df: pd.DataFrame,
    n_replicates: int,
    seed: int,
) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    sequences = np.array(sorted(feature_df["sequence_id"].unique()))
    valid_rows: list[dict] = []
    failed = 0
    for iteration in range(n_replicates):
        sampled = rng.choice(sequences, size=len(sequences), replace=True)
        boot = resample_sequence_clusters(feature_df, sampled)
        estimable, _ = is_estimable_feature(boot)
        if not estimable:
            failed += 1
            continue
        try:
            model = smf.ols("motif_specificity_score ~ dose_centered * concept_positive_int", data=boot).fit()
            interaction = _interaction_term(list(model.model.exog_names))
            beta = float(model.params[interaction])
        except Exception:
            failed += 1
            continue
        if not np.isfinite(beta):
            failed += 1
            continue
        valid_rows.append(
            {
                "feature_id": int(feature_df["feature_id"].iloc[0]),
                "bootstrap_iteration": int(iteration),
                "beta_dose_x_concept": beta,
            }
        )

    values = np.array([row["beta_dose_x_concept"] for row in valid_rows], dtype=float)
    if values.size:
        summary = {
            "bootstrap_n_requested": int(n_replicates),
            "bootstrap_n_valid": int(values.size),
            "bootstrap_n_failed": int(failed),
            "bootstrap_mean": float(np.mean(values)),
            "bootstrap_median": float(np.median(values)),
            "bootstrap_ci_low": float(np.quantile(values, 0.025)),
            "bootstrap_ci_high": float(np.quantile(values, 0.975)),
            "bootstrap_fraction_positive": float(np.mean(values > 0)),
        }
    else:
        summary = {
            "bootstrap_n_requested": int(n_replicates),
            "bootstrap_n_valid": 0,
            "bootstrap_n_failed": int(failed),
            "bootstrap_mean": np.nan,
            "bootstrap_median": np.nan,
            "bootstrap_ci_low": np.nan,
            "bootstrap_ci_high": np.nan,
            "bootstrap_fraction_positive": np.nan,
        }
    return pd.DataFrame(valid_rows), summary


def resample_sequence_clusters(feature_df: pd.DataFrame, sampled_sequence_ids: Iterable[str]) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for draw_index, sequence_id in enumerate(sampled_sequence_ids):
        seq_rows = feature_df.loc[feature_df["sequence_id"] == str(sequence_id)].copy()
        seq_rows["bootstrap_sequence_id"] = f"{sequence_id}__draw_{draw_index}"
        pieces.append(seq_rows)
    if not pieces:
        return feature_df.iloc[0:0].copy()
    return pd.concat(pieces, ignore_index=True)


def apply_bh_fdr(summary: pd.DataFrame, candidate_features: Iterable[int] = BIOLOGICAL_CANDIDATE_FEATURES) -> pd.DataFrame:
    out = summary.copy()
    out["interaction_q"] = np.nan
    out["interaction_q_lt_0_05"] = False
    candidate_set = {int(v) for v in candidate_features}
    mask = (
        out["feature_id"].isin(candidate_set)
        & out["estimable"].astype(bool)
        & out["interaction_p"].notna()
        & (out["matched_concept"] != "matched_control")
    )
    if mask.any():
        q_values = multipletests(out.loc[mask, "interaction_p"].to_numpy(dtype=float), method="fdr_bh")[1]
        out.loc[mask, "interaction_q"] = q_values
        out.loc[mask, "interaction_q_lt_0_05"] = q_values < 0.05
    return out


def directionality_summary(feature_df: pd.DataFrame) -> dict:
    negative = feature_df.loc[feature_df["dose"] < 0, "motif_specificity_score"]
    zero = feature_df.loc[feature_df["dose"] == 0, "motif_specificity_score"]
    positive = feature_df.loc[feature_df["dose"] > 0, "motif_specificity_score"]
    return {
        "suppression_mean_specificity": float(negative.mean()) if not negative.empty else np.nan,
        "noop_mean_specificity": float(zero.mean()) if not zero.empty else np.nan,
        "amplification_mean_specificity": float(positive.mean()) if not positive.empty else np.nan,
        "suppression_minus_noop": float(negative.mean() - zero.mean()) if not negative.empty and not zero.empty else np.nan,
        "amplification_minus_noop": float(positive.mean() - zero.mean()) if not positive.empty and not zero.empty else np.nan,
    }


def matched_kl_summary(feature_df: pd.DataFrame, target_kl: float = DEFAULT_MATCHED_KL_TARGET) -> dict:
    if "kl" not in feature_df.columns:
        return {
            "matched_kl_target": target_kl,
            "matched_kl_positive_specificity": np.nan,
            "matched_kl_negative_specificity": np.nan,
            "matched_kl_gap": np.nan,
        }
    values = {}
    for concept_positive, label in [(True, "positive"), (False, "negative")]:
        g = feature_df.loc[feature_df["concept_positive"] == concept_positive]
        if g.empty:
            values[label] = np.nan
            continue
        by_dose = g.groupby("dose", as_index=False).agg(mean_kl=("kl", "mean"), mean_specificity=("motif_specificity_score", "mean"))
        chosen = by_dose.iloc[(by_dose["mean_kl"] - target_kl).abs().argsort().iloc[0]]
        values[label] = float(chosen["mean_specificity"])
    gap = values["positive"] - values["negative"] if np.isfinite(values["positive"]) and np.isfinite(values["negative"]) else np.nan
    return {
        "matched_kl_target": float(target_kl),
        "matched_kl_positive_specificity": values["positive"],
        "matched_kl_negative_specificity": values["negative"],
        "matched_kl_gap": float(gap) if np.isfinite(gap) else np.nan,
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def run_confirmatory_analysis(
    input_path: str | Path = DEFAULT_INPUT,
    summary_output: str | Path = DEFAULT_SUMMARY_OUTPUT,
    bootstrap_output: str | Path = DEFAULT_BOOTSTRAP_OUTPUT,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = load_causal_data(input_path)
    inspection = inspect_dataset(df)
    summary_rows: list[dict] = []
    bootstrap_frames: list[pd.DataFrame] = []
    timestamp = datetime.now(timezone.utc).isoformat()
    commit = git_commit()

    for feature_id in sorted(df["feature_id"].unique()):
        feature_df = df.loc[df["feature_id"] == feature_id].copy()
        fit = fit_clustered_interaction(feature_df)
        row = fit.__dict__.copy()
        if fit.estimable and bootstrap_replicates > 0:
            boot_df, boot_summary = bootstrap_sequence_interactions(
                feature_df=feature_df,
                n_replicates=bootstrap_replicates,
                seed=bootstrap_seed + int(feature_id),
            )
            if not boot_df.empty:
                bootstrap_frames.append(boot_df)
        else:
            boot_summary = {
                "bootstrap_n_requested": int(bootstrap_replicates) if fit.estimable else 0,
                "bootstrap_n_valid": 0,
                "bootstrap_n_failed": 0,
                "bootstrap_mean": np.nan,
                "bootstrap_median": np.nan,
                "bootstrap_ci_low": np.nan,
                "bootstrap_ci_high": np.nan,
                "bootstrap_fraction_positive": np.nan,
            }
        row.update(boot_summary)
        row.update(directionality_summary(feature_df))
        row.update(
            {
                "analysis_script": "analysis/confirmatory_causal_statistics.py",
                "input_path": str(input_path),
                "bootstrap_seed": int(bootstrap_seed),
                "model_specification": "motif_specificity_score ~ dose_centered * concept_positive_int",
                "git_commit": commit,
                "timestamp_utc": timestamp,
            }
        )
        summary_rows.append(row)

    summary = apply_bh_fdr(pd.DataFrame(summary_rows))
    preferred = [
        "feature_id",
        "matched_concept",
        "n_rows",
        "n_sequences",
        "n_concept_positive_sequences",
        "n_concept_negative_sequences",
        "estimable",
        "non_estimable_reason",
        "beta_dose",
        "beta_concept",
        "beta_dose_x_concept",
        "interaction_clustered_se",
        "interaction_ci_low",
        "interaction_ci_high",
        "interaction_p",
        "interaction_q",
        "interaction_q_lt_0_05",
        "bootstrap_n_requested",
        "bootstrap_n_valid",
        "bootstrap_n_failed",
        "bootstrap_mean",
        "bootstrap_median",
        "bootstrap_ci_low",
        "bootstrap_ci_high",
        "bootstrap_fraction_positive",
    ]
    remaining = [col for col in summary.columns if col not in preferred]
    summary = summary[preferred + remaining]
    bootstrap = pd.concat(bootstrap_frames, ignore_index=True) if bootstrap_frames else pd.DataFrame(columns=["feature_id", "bootstrap_iteration", "beta_dose_x_concept"])

    summary_path = Path(summary_output)
    bootstrap_path = Path(bootstrap_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    bootstrap_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    bootstrap.to_csv(bootstrap_path, index=False)
    return summary, bootstrap, inspection


def main() -> None:
    parser = argparse.ArgumentParser(description="Run confirmatory fixed-effects causal selectivity statistics.")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--bootstrap-output", default=DEFAULT_BOOTSTRAP_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        summary, bootstrap, inspection = run_confirmatory_analysis(
            input_path=args.input,
            summary_output=args.summary_output,
            bootstrap_output=args.bootstrap_output,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.bootstrap_seed,
        )

    print(json.dumps({"inspection": inspection, "summary": summary.to_dict(orient="records"), "n_bootstrap_rows": int(len(bootstrap))}, indent=2))


if __name__ == "__main__":
    main()
