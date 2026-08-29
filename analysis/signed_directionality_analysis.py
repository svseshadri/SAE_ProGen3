from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

try:
    from analysis.confirmatory_causal_statistics import resample_sequence_clusters
except ImportError:  # pragma: no cover - script fallback
    from confirmatory_causal_statistics import resample_sequence_clusters


DEFAULT_CAUSAL_INPUT = "results/causal_feature_dose_response.csv"
DEFAULT_CONFIRMATORY_INPUT = "results/confirmatory_causal_statistics.csv"
DEFAULT_RECONSTRUCTION_INPUT = "results/reconstruction_evaluation.csv"
DEFAULT_IDENTITY_INPUT = "results/identity_patch_evaluation.csv"
DEFAULT_RESIDUAL_INPUT = "results/residual_sensitivity_subset.csv"
DEFAULT_RANDOM_NOISE_INPUT = "results/random_noise_control.csv"
DEFAULT_OUTPUT_DIR = "results"
DEFAULT_FIGURE_DIR = "results/figures"
DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_BOOTSTRAP_SEED = 93256

PRIMARY_FEATURE = 3256
SECONDARY_FEATURE = 2942
FEATURE_LABELS = {
    1: "Control feature 1",
    727: "727 / IPR001314",
    1644: "1644 / PS00135",
    2942: "2942 / both motifs",
    3256: "3256 / PS00134",
}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def load_causal_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {
        "sequence_id",
        "feature_id",
        "matched_concept",
        "concept_positive",
        "dose",
        "motif_delta_logprob",
        "nonmotif_delta_logprob",
        "motif_specificity_score",
        "delta_nll",
        "kl",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    out = df.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["feature_id"] = out["feature_id"].astype(int)
    if not pd.api.types.is_bool_dtype(out["concept_positive"]):
        out["concept_positive"] = out["concept_positive"].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
    out["concept_positive"] = out["concept_positive"].astype(bool)
    out["concept_positive_int"] = out["concept_positive"].astype(int)
    for col in ["dose", "motif_delta_logprob", "nonmotif_delta_logprob", "motif_specificity_score", "delta_nll", "kl"]:
        out[col] = pd.to_numeric(out[col], errors="raise")
    out["global_signed_nll_effect"] = -out["delta_nll"]
    return out


def audit_metric_definition(source_path: str | Path = "scripts/causal_feature_dose_response.py") -> dict:
    text = Path(source_path).read_text()
    function_match = re.search(r"def compute_localized_logit_shift\([\s\S]*?\n\ndef ", text)
    source_excerpt = function_match.group(0).rsplit("\n\ndef ", 1)[0] if function_match else ""
    uses_abs = "abs(float(d))" in source_excerpt or "abs(" in source_excerpt
    return {
        "source_path": str(source_path),
        "source_function": "compute_localized_logit_shift",
        "motif_delta_logprob_definition": "mean absolute patched-minus-base log probability shift for the top-k positions with largest absolute true-token log-probability changes",
        "nonmotif_delta_logprob_definition": "mean absolute patched-minus-base log probability shift for the remaining valid positions outside the top-k largest absolute changes",
        "motif_specificity_score_definition": "motif_delta_logprob - nonmotif_delta_logprob, i.e. a difference of two unsigned magnitude summaries",
        "uses_absolute_value": bool(uses_abs),
        "preserves_underlying_sign": False,
        "positive_motif_delta_logprob_means": "larger absolute localized perturbation magnitude in the selected top-k positions; not increase in token probability",
        "positive_nonmotif_delta_logprob_means": "larger absolute perturbation magnitude outside the selected top-k positions; not increase in token probability",
        "positive_motif_specificity_score_means": "top-k positions changed by larger magnitude than non-top-k positions; not signed biological direction",
        "signed_localized_effect_available": False,
        "signed_directionality_testable_from_motif_specificity_score": False,
        "classification_options_considered": "A/B/C/D",
        "directionality_gate_classification": "INCONCLUSIVE",
        "classification_reason": "The only localized readouts stored in the canonical causal CSV were computed after abs(delta), so the sign of patched_logp - base_logp cannot be recovered from existing artifacts.",
    }


def cluster_bootstrap_mean(values: pd.DataFrame, metric: str, n_replicates: int, seed: int) -> tuple[float, float, int]:
    seqs = np.array(sorted(values["sequence_id"].unique()))
    if len(seqs) == 0:
        return np.nan, np.nan, 0
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_replicates):
        sampled = rng.choice(seqs, size=len(seqs), replace=True)
        pieces = [values.loc[values["sequence_id"] == seq] for seq in sampled]
        boot = pd.concat(pieces, ignore_index=True)
        boot_means.append(float(boot[metric].mean()))
    return float(np.quantile(boot_means, 0.025)), float(np.quantile(boot_means, 0.975)), int(n_replicates)


def summarize_by_dose(df: pd.DataFrame, features: Iterable[int], n_bootstrap: int = 1000, seed: int = DEFAULT_BOOTSTRAP_SEED) -> pd.DataFrame:
    rows = []
    for feature_id in features:
        feat = df.loc[df["feature_id"] == int(feature_id)]
        for (concept_positive, dose), g in feat.groupby(["concept_positive", "dose"], sort=True):
            row = {
                "feature_id": int(feature_id),
                "matched_concept": str(g["matched_concept"].iloc[0]),
                "concept_positive": bool(concept_positive),
                "dose": float(dose),
                "dose_class": "suppression" if dose < 0 else "no_op" if dose == 0 else "amplification",
                "n_rows": int(len(g)),
                "n_sequences": int(g["sequence_id"].nunique()),
                "motif_specificity_score_mean": float(g["motif_specificity_score"].mean()),
                "motif_specificity_score_median": float(g["motif_specificity_score"].median()),
                "motif_delta_logprob_mean": float(g["motif_delta_logprob"].mean()),
                "nonmotif_delta_logprob_mean": float(g["nonmotif_delta_logprob"].mean()),
                "global_signed_nll_effect_mean": float(g["global_signed_nll_effect"].mean()),
                "global_signed_nll_effect_median": float(g["global_signed_nll_effect"].median()),
                "delta_nll_mean": float(g["delta_nll"].mean()),
                "kl_mean": float(g["kl"].mean()),
                "localized_metric_preserves_sign": False,
            }
            ci_low, ci_high, boot_n = cluster_bootstrap_mean(g, "motif_specificity_score", n_bootstrap, seed + int(feature_id) + int(abs(dose) * 1000) + int(concept_positive))
            row.update({"motif_specificity_bootstrap_ci_low": ci_low, "motif_specificity_bootstrap_ci_high": ci_high, "bootstrap_n_requested": boot_n})
            rows.append(row)
    return pd.DataFrame(rows)


def fit_global_signed_model(feature_df: pd.DataFrame) -> dict:
    model = smf.ols("global_signed_nll_effect ~ dose * concept_positive_int", data=feature_df).fit()
    robust = model.get_robustcov_results(cov_type="cluster", groups=feature_df["sequence_id"])
    names = list(model.model.exog_names)
    params = pd.Series(np.asarray(robust.params), index=names)
    bse = pd.Series(np.asarray(robust.bse), index=names)
    pvalues = pd.Series(np.asarray(robust.pvalues), index=names)
    conf = pd.DataFrame(np.asarray(robust.conf_int()), index=names, columns=["low", "high"])
    interaction = [n for n in names if ":" in n and "dose" in n and "concept_positive_int" in n][0]
    return {
        "global_signed_beta_dose": float(params["dose"]),
        "global_signed_beta_concept": float(params["concept_positive_int"]),
        "global_signed_beta_dose_x_concept": float(params[interaction]),
        "global_signed_interaction_se": float(bse[interaction]),
        "global_signed_interaction_ci_low": float(conf.loc[interaction, "low"]),
        "global_signed_interaction_ci_high": float(conf.loc[interaction, "high"]),
        "global_signed_interaction_p": float(pvalues[interaction]),
    }


def bootstrap_global_signed_interaction(feature_df: pd.DataFrame, n_replicates: int, seed: int) -> tuple[pd.DataFrame, dict]:
    rng = np.random.default_rng(seed)
    seqs = np.array(sorted(feature_df["sequence_id"].unique()))
    rows = []
    failed = 0
    for iteration in range(n_replicates):
        sampled = rng.choice(seqs, size=len(seqs), replace=True)
        boot = resample_sequence_clusters(feature_df, sampled)
        if boot["concept_positive"].nunique() < 2 or boot["dose"].nunique() < 2:
            failed += 1
            continue
        try:
            model = smf.ols("global_signed_nll_effect ~ dose * concept_positive_int", data=boot).fit()
            term = [n for n in model.model.exog_names if ":" in n and "dose" in n and "concept_positive_int" in n][0]
            beta = float(model.params[term])
        except Exception:
            failed += 1
            continue
        if math.isfinite(beta):
            rows.append({"feature_id": int(feature_df["feature_id"].iloc[0]), "bootstrap_iteration": int(iteration), "global_signed_beta_dose_x_concept": beta})
        else:
            failed += 1
    values = np.array([r["global_signed_beta_dose_x_concept"] for r in rows], dtype=float)
    summary = {
        "global_signed_bootstrap_n_requested": int(n_replicates),
        "global_signed_bootstrap_n_valid": int(len(values)),
        "global_signed_bootstrap_n_failed": int(failed),
        "global_signed_bootstrap_mean": float(np.mean(values)) if len(values) else np.nan,
        "global_signed_bootstrap_median": float(np.median(values)) if len(values) else np.nan,
        "global_signed_bootstrap_ci_low": float(np.quantile(values, 0.025)) if len(values) else np.nan,
        "global_signed_bootstrap_ci_high": float(np.quantile(values, 0.975)) if len(values) else np.nan,
        "global_signed_bootstrap_fraction_positive": float(np.mean(values > 0)) if len(values) else np.nan,
    }
    return pd.DataFrame(rows), summary


def per_sequence_slopes(df: pd.DataFrame, feature_id: int = PRIMARY_FEATURE) -> pd.DataFrame:
    rows = []
    feat = df.loc[df["feature_id"] == feature_id]
    for sequence_id, g in feat.groupby("sequence_id"):
        if g["dose"].nunique() < 2:
            continue
        x = g["dose"].to_numpy(dtype=float)
        for metric, signed, localized in [
            ("motif_specificity_score", False, True),
            ("global_signed_nll_effect", True, False),
        ]:
            y = g[metric].to_numpy(dtype=float)
            slope = float(np.polyfit(x, y, 1)[0])
            rows.append({
                "feature_id": int(feature_id),
                "sequence_id": sequence_id,
                "concept_positive": bool(g["concept_positive"].iloc[0]),
                "matched_concept": str(g["matched_concept"].iloc[0]),
                "metric": metric,
                "metric_is_signed": bool(signed),
                "metric_is_localized": bool(localized),
                "n_dose_observations": int(len(g)),
                "n_unique_doses": int(g["dose"].nunique()),
                "slope": slope,
            })
    return pd.DataFrame(rows)


def sequence_slope_summary(slopes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (metric, concept_positive), g in slopes.groupby(["metric", "concept_positive"], sort=True):
        q25, q75 = np.quantile(g["slope"], [0.25, 0.75])
        rows.append({
            "metric": metric,
            "concept_positive": bool(concept_positive),
            "n_sequences": int(g["sequence_id"].nunique()),
            "median_sequence_slope": float(np.median(g["slope"])),
            "iqr_low": float(q25),
            "iqr_high": float(q75),
            "fraction_positive_slopes": float(np.mean(g["slope"] > 0)),
            "fraction_negative_slopes": float(np.mean(g["slope"] < 0)),
            "max_abs_slope_sequence_id": str(g.iloc[g["slope"].abs().argmax()]["sequence_id"]),
            "max_abs_slope": float(g["slope"].iloc[g["slope"].abs().argmax()]),
        })
    return pd.DataFrame(rows)


def run_analysis(
    causal_input: str | Path = DEFAULT_CAUSAL_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    by_dose_bootstrap_replicates: int = 1000,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict:
    df = load_causal_data(causal_input)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    audit = audit_metric_definition()
    audit.update({"analysis_timestamp_utc": timestamp, "git_commit": git_commit(), "input_path": str(causal_input)})
    pd.DataFrame([audit]).to_csv(out_dir / "signed_directionality_metric_audit.csv", index=False)

    by_dose = summarize_by_dose(df, [PRIMARY_FEATURE, SECONDARY_FEATURE], n_bootstrap=by_dose_bootstrap_replicates, seed=bootstrap_seed)
    by_dose.to_csv(out_dir / "signed_directionality_by_dose.csv", index=False)

    primary = df.loc[df["feature_id"] == PRIMARY_FEATURE].copy()
    stats = {
        "feature_id": PRIMARY_FEATURE,
        "matched_concept": str(primary["matched_concept"].iloc[0]),
        "n_rows": int(len(primary)),
        "n_sequences": int(primary["sequence_id"].nunique()),
        "n_concept_positive_sequences": int(primary.loc[primary["concept_positive"], "sequence_id"].nunique()),
        "n_concept_negative_sequences": int(primary.loc[~primary["concept_positive"], "sequence_id"].nunique()),
        "localized_signed_metric_available": False,
        "localized_signed_analysis_performed": False,
        "localized_signed_analysis_reason": "No signed localized readout is recoverable because motif_delta_logprob and nonmotif_delta_logprob are stored after abs(delta).",
        "directionality_gate_classification": "INCONCLUSIVE",
        "directionality_gate_reason": "Existing localized metrics are unsigned; signed global delta NLL is available but is not a motif-localized biological direction readout.",
    }
    stats.update(fit_global_signed_model(primary))
    boot, boot_summary = bootstrap_global_signed_interaction(primary, bootstrap_replicates, bootstrap_seed + PRIMARY_FEATURE)
    stats.update(boot_summary)
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(out_dir / "signed_directionality_statistics.csv", index=False)
    boot.to_csv(out_dir / "signed_directionality_bootstrap.csv", index=False)

    slopes = per_sequence_slopes(df, PRIMARY_FEATURE)
    slopes.to_csv(out_dir / "signed_directionality_sequence_slopes.csv", index=False)
    sequence_slope_summary(slopes).to_csv(out_dir / "signed_directionality_sequence_slope_summary.csv", index=False)
    return {
        "audit": audit,
        "statistics": stats,
        "by_dose_rows": int(len(by_dose)),
        "bootstrap_rows": int(len(boot)),
        "slope_rows": int(len(slopes)),
    }


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def save_figure(fig: plt.Figure, base_path: Path) -> list[str]:
    paths = []
    for ext in ["png", "pdf", "svg"]:
        out = base_path.with_suffix(f".{ext}")
        fig.savefig(out, bbox_inches="tight")
        paths.append(str(out))
    plt.close(fig)
    return paths


def plot_reconstruction_paradox(recon_path: str | Path, identity_path: str | Path, residual_path: str | Path, random_path: str | Path, figure_dir: str | Path) -> list[str]:
    setup_style()
    recon = pd.read_csv(recon_path)
    ident = pd.read_csv(identity_path)
    resid = pd.read_csv(residual_path)
    random = pd.read_csv(random_path)
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4))
    axes[0].bar([0, 1], [recon["nmse"].mean(), recon["explained_variance"].mean()], color=["#4C78A8", "#59A14F"], width=0.6)
    axes[0].set_xticks([0, 1], ["NMSE", "Explained\nvariance"])
    axes[0].set_title("Numerical reconstruction")
    axes[0].set_ylim(0, 1.05)
    nll_means = [recon["base_nll"].mean(), recon["patched_nll"].mean(), ident["patched_nll"].mean()]
    axes[1].bar(range(3), nll_means, color=["#777777", "#E15759", "#59A14F"], width=0.65)
    axes[1].set_xticks(range(3), ["Baseline", "SAE\npatch", "Identity"], rotation=0)
    axes[1].set_ylabel("Mean NLL")
    axes[1].set_title("Downstream behavior")
    r = resid.groupby("lambda", as_index=False).agg(mean_delta=("sae_delta_nll", "mean"), se=("sae_delta_nll", lambda s: s.std(ddof=1) / np.sqrt(s.nunique() if s.nunique() else len(s))))
    axes[2].plot(r["lambda"], r["mean_delta"], marker="o", color="#4C78A8", label="Residual restored")
    axes[2].axhline(float(random["random_noise_delta_nll"].mean()), color="#E15759", linestyle="--", linewidth=1.2, label="Random-noise control")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_xlabel("Fraction residual restored")
    axes[2].set_ylabel("Delta NLL")
    axes[2].set_title("Structured residual")
    axes[2].legend(frameon=False)
    fig.suptitle("Reconstruction fidelity does not imply functional fidelity", y=1.05, fontsize=11)
    return save_figure(fig, Path(figure_dir) / "figure1_reconstruction_fidelity_paradox")


def plot_causal_dose_response(causal_path: str | Path, figure_dir: str | Path) -> list[str]:
    setup_style()
    df = load_causal_data(causal_path)
    fig, ax = plt.subplots(figsize=(4.9, 3.1))
    colors = {3256: "#4C78A8", 2942: "#F28E2B", 1644: "#59A14F", 727: "#B07AA1", 1: "#777777"}
    for fid, g in df.groupby("feature_id", sort=True):
        dd = g.groupby("dose", as_index=False).agg(mean_kl=("kl", "mean"), se=("kl", lambda s: s.std(ddof=1) / np.sqrt(len(s))))
        ax.errorbar(dd["dose"], dd["mean_kl"], yerr=1.96 * dd["se"], marker="o", linewidth=1.4, capsize=2, label=FEATURE_LABELS.get(int(fid), str(fid)), color=colors.get(int(fid)))
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Signed steering dose")
    ax.set_ylabel("Mean KL divergence")
    ax.set_title("Causal sensitivity of additive SAE directions")
    ax.legend(frameon=False, ncol=1)
    return save_figure(fig, Path(figure_dir) / "figure2_causal_additive_dose_response")


def plot_confirmatory_forest(stats_path: str | Path, figure_dir: str | Path) -> list[str]:
    setup_style()
    stats = pd.read_csv(stats_path)
    stats = stats.loc[stats["estimable"] == True].copy()
    stats = stats.sort_values("beta_dose_x_concept")
    y = np.arange(len(stats))
    fig, ax = plt.subplots(figsize=(4.9, 2.4))
    ax.errorbar(stats["beta_dose_x_concept"], y, xerr=[stats["beta_dose_x_concept"] - stats["interaction_ci_low"], stats["interaction_ci_high"] - stats["beta_dose_x_concept"]], fmt="o", color="#4C78A8", capsize=3)
    labels = [f"{int(r.feature_id)} / {r.matched_concept}\nq={r.interaction_q:.3g}" for r in stats.itertuples()]
    ax.set_yticks(y, labels)
    ax.axvline(0, color="#333333", linewidth=0.9)
    ax.set_xlabel("Dose x concept interaction beta")
    ax.set_title("Confirmatory clustered interaction estimates")
    ax.text(0.98, 0.04, "727 not estimable: no concept-negative support", transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    return save_figure(fig, Path(figure_dir) / "figure4_confirmatory_interaction_forest")


def plot_directionality_audit(by_dose_path: str | Path, audit_path: str | Path, figure_dir: str | Path) -> list[str]:
    setup_style()
    by_dose = pd.read_csv(by_dose_path)
    audit = pd.read_csv(audit_path).iloc[0]
    data = by_dose.loc[by_dose["feature_id"] == PRIMARY_FEATURE]
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    colors = {True: "#4C78A8", False: "#E15759"}
    for concept, g in data.groupby("concept_positive"):
        g = g.sort_values("dose")
        label = "PS00134-positive" if concept else "PS00134-negative"
        yerr = [g["motif_specificity_score_mean"] - g["motif_specificity_bootstrap_ci_low"], g["motif_specificity_bootstrap_ci_high"] - g["motif_specificity_score_mean"]]
        ax.errorbar(g["dose"], g["motif_specificity_score_mean"], yerr=yerr, marker="o", linewidth=1.4, capsize=2, color=colors[bool(concept)], label=label)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("Signed steering dose")
    ax.set_ylabel("Unsigned motif-specificity magnitude")
    ax.set_title("3256 directionality audit")
    ax.text(0.02, 0.98, "Stored localized metric uses abs(delta);\nnot a signed biological effect", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.legend(frameon=False)
    assert bool(audit["preserves_underlying_sign"]) is False
    return save_figure(fig, Path(figure_dir) / "figure5_directionality_metric_audit")


def plot_sequence_slope_robustness(slopes_path: str | Path, figure_dir: str | Path) -> list[str]:
    setup_style()
    slopes = pd.read_csv(slopes_path)
    slopes = slopes.loc[slopes["metric"] == "motif_specificity_score"].copy()
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    rng = np.random.default_rng(4)
    for idx, concept in enumerate([False, True]):
        g = slopes.loc[slopes["concept_positive"] == concept]
        x = np.full(len(g), idx) + rng.normal(0, 0.04, len(g))
        ax.scatter(x, g["slope"], s=18, alpha=0.8, color="#E15759" if not concept else "#4C78A8")
        if len(g):
            ax.plot([idx - 0.18, idx + 0.18], [g["slope"].median(), g["slope"].median()], color="#333333", linewidth=1.4)
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_xticks([0, 1], ["PS00134-negative\nn=10", "PS00134-positive\nn=54"])
    ax.set_ylabel("Per-sequence slope\n(unsigned specificity per dose)")
    ax.set_title("3256 sequence-level slope distribution")
    ax.text(0.02, 0.98, "Robustness for unsigned metric;\nnot signed directionality", transform=ax.transAxes, ha="left", va="top", fontsize=8)
    return save_figure(fig, Path(figure_dir) / "figure6_sequence_slope_robustness_unsigned")


def write_figure_manifest(records: list[dict], figure_dir: str | Path) -> Path:
    path = Path(figure_dir) / "figure_manifest.csv"
    pd.DataFrame(records).to_csv(path, index=False)
    return path


def create_figures(
    causal_input: str | Path = DEFAULT_CAUSAL_INPUT,
    confirmatory_input: str | Path = DEFAULT_CONFIRMATORY_INPUT,
    reconstruction_input: str | Path = DEFAULT_RECONSTRUCTION_INPUT,
    identity_input: str | Path = DEFAULT_IDENTITY_INPUT,
    residual_input: str | Path = DEFAULT_RESIDUAL_INPUT,
    random_noise_input: str | Path = DEFAULT_RANDOM_NOISE_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    figure_dir: str | Path = DEFAULT_FIGURE_DIR,
) -> pd.DataFrame:
    fig_dir = Path(figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    records = []
    outputs = plot_reconstruction_paradox(reconstruction_input, identity_input, residual_input, random_noise_input, fig_dir)
    records.append({"figure": "figure1_reconstruction_fidelity_paradox", "source_artifacts": ";".join(map(str, [reconstruction_input, identity_input, residual_input, random_noise_input])), "outputs": ";".join(outputs)})
    outputs = plot_causal_dose_response(causal_input, fig_dir)
    records.append({"figure": "figure2_causal_additive_dose_response", "source_artifacts": str(causal_input), "outputs": ";".join(outputs)})
    records.append({"figure": "figure3_context_selectivity_at_matched_disturbance", "source_artifacts": str(causal_input), "outputs": "not_created", "reason": "The established KL~0.025 matched-disturbance values in PROJECT_STATUS.md are not directly reproducible from an available canonical matched-KL artifact without redefining the calculation."})
    outputs = plot_confirmatory_forest(confirmatory_input, fig_dir)
    records.append({"figure": "figure4_confirmatory_interaction_forest", "source_artifacts": str(confirmatory_input), "outputs": ";".join(outputs)})
    outputs = plot_directionality_audit(Path(output_dir) / "signed_directionality_by_dose.csv", Path(output_dir) / "signed_directionality_metric_audit.csv", fig_dir)
    records.append({"figure": "figure5_directionality_metric_audit", "source_artifacts": f"{output_dir}/signed_directionality_by_dose.csv;{output_dir}/signed_directionality_metric_audit.csv", "outputs": ";".join(outputs)})
    outputs = plot_sequence_slope_robustness(Path(output_dir) / "signed_directionality_sequence_slopes.csv", fig_dir)
    records.append({"figure": "figure6_sequence_slope_robustness_unsigned", "source_artifacts": f"{output_dir}/signed_directionality_sequence_slopes.csv", "outputs": ";".join(outputs)})
    manifest = write_figure_manifest(records, fig_dir)
    return pd.read_csv(manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit localized metric directionality and generate reproducible figures from existing artifacts.")
    parser.add_argument("--causal-input", default=DEFAULT_CAUSAL_INPUT)
    parser.add_argument("--confirmatory-input", default=DEFAULT_CONFIRMATORY_INPUT)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--figure-dir", default=DEFAULT_FIGURE_DIR)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--by-dose-bootstrap-replicates", type=int, default=1000)
    parser.add_argument("--bootstrap-seed", type=int, default=DEFAULT_BOOTSTRAP_SEED)
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args()

    summary = run_analysis(
        causal_input=args.causal_input,
        output_dir=args.output_dir,
        bootstrap_replicates=args.bootstrap_replicates,
        by_dose_bootstrap_replicates=args.by_dose_bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    if not args.skip_figures:
        manifest = create_figures(causal_input=args.causal_input, confirmatory_input=args.confirmatory_input, output_dir=args.output_dir, figure_dir=args.figure_dir)
        summary["figures"] = manifest.to_dict(orient="records")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
