from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROGEN3_SRC = REPO_ROOT / "progen3" / "src"
if str(PROGEN3_SRC) not in sys.path:
    sys.path.insert(0, str(PROGEN3_SRC))

from scripts.causal_feature_3256_signed_directionality import (  # noqa: E402
    CONTROL_FEATURE,
    FEATURE_3256,
    LEGACY_TOP_K,
    aggregate_signed_legacy_topk,
    residue_metadata,
    token_signed_deltas,
    write_outputs,
)
from scripts.causal_feature_dose_response import feature_concept_name  # noqa: E402

DEFAULT_MODEL_NAME = "Profluent-Bio/progen3-112m"
DEFAULT_SAE_CKPT = "results/topk_sae_layer6_d4096_k32_run1/best.pt"
DEFAULT_DATASET = "data/processed/s1a70/s1a70_test.csv"
DEFAULT_CANONICAL_CAUSAL = "results/causal_feature_dose_response.csv"
DEFAULT_NATIVE_SUMMARY_OUTPUT = "results/feature_3256_native_activation_summary.csv"
DEFAULT_SUMMARY_OUTPUT = "results/causal_feature_3256_identity_centered_directionality.csv"
DEFAULT_POSITION_OUTPUT = "results/causal_feature_3256_identity_centered_position_effects.csv"
DEFAULT_STATS_OUTPUT = "results/causal_feature_3256_identity_centered_statistics.csv"
DEFAULT_BOOTSTRAP_OUTPUT = "results/causal_feature_3256_identity_centered_bootstrap.csv"
DEFAULT_SEQUENCE_SLOPES_OUTPUT = "results/causal_feature_3256_identity_centered_sequence_slopes.csv"
DEFAULT_SEQUENCE_SLOPE_SUMMARY_OUTPUT = "results/causal_feature_3256_identity_centered_sequence_slope_summary.csv"
DEFAULT_METADATA_OUTPUT = "results/causal_feature_3256_identity_centered_metadata.json"
DEFAULT_FIGURE_STEM = "results/figures/figure_3256_identity_centered_directionality"
DEFAULT_BOOTSTRAP_REPLICATES = 5000
DEFAULT_BOOTSTRAP_SEED = 932560
LAYER_INDEX = 6


@dataclass(frozen=True)
class IdentityDiagnostics:
    native_min: float
    native_max: float
    requested_displacement: float
    realized_displacement_min: float
    realized_displacement_max: float
    max_abs_realized_minus_requested: float
    crosses_nonnegative_latent_domain: bool
    max_abs_hidden_delta: float


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def import_model_helpers():
    import torch
    from scripts.evaluate_reconstruction import (
        ProGen3BatchPreparer,
        ProGen3ForCausalLM,
        compute_mean_kl,
        forward_with_replaced_hidden_state,
        get_layer_hidden_state,
        load_sae_checkpoint,
        sequence_nll,
        top1_agreement,
    )

    return {
        "torch": torch,
        "ProGen3BatchPreparer": ProGen3BatchPreparer,
        "ProGen3ForCausalLM": ProGen3ForCausalLM,
        "compute_mean_kl": compute_mean_kl,
        "forward_with_replaced_hidden_state": forward_with_replaced_hidden_state,
        "get_layer_hidden_state": get_layer_hidden_state,
        "load_sae_checkpoint": load_sae_checkpoint,
        "sequence_nll": sequence_nll,
        "top1_agreement": top1_agreement,
    }


def _coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    return series.astype(str).str.strip().str.lower().map({"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}).astype(bool)


def load_feature_3256_cohort(
    canonical_causal_path: str | Path = DEFAULT_CANONICAL_CAUSAL,
    dataset_path: str | Path = DEFAULT_DATASET,
    include_control: bool = True,
    max_sequences: int | None = None,
) -> pd.DataFrame:
    causal = pd.read_csv(canonical_causal_path)
    feature_rows = causal.loc[causal["feature_id"] == FEATURE_3256].copy()
    sequence_ids = sorted(int(v) for v in feature_rows["sequence_id"].drop_duplicates())
    if max_sequences is not None:
        sequence_ids = sequence_ids[: int(max_sequences)]

    dataset = pd.read_csv(dataset_path).reset_index().rename(columns={"index": "sequence_id"})
    cohort = dataset.loc[dataset["sequence_id"].isin(sequence_ids)].copy()
    if cohort["sequence_id"].nunique() != len(sequence_ids):
        found = set(int(v) for v in cohort["sequence_id"])
        missing = [v for v in sequence_ids if v not in found]
        raise ValueError(f"Could not recover sequence_id values from dataset: {missing[:10]}")

    rows: list[dict[str, Any]] = []
    features = [FEATURE_3256, CONTROL_FEATURE] if include_control else [FEATURE_3256]
    for _, row in cohort.sort_values("sequence_id").iterrows():
        for feature_id in features:
            rows.append(
                {
                    "sequence_id": int(row["sequence_id"]),
                    "sequence": str(row["sequence"]),
                    "length": int(row.get("length", len(str(row["sequence"])))),
                    "feature_id": int(feature_id),
                    "matched_concept": "has_ps00134" if feature_id == FEATURE_3256 else "matched_control_feature_1_in_ps00134_cohort",
                    "concept_positive": bool(row.get("has_ps00134", False)),
                    "class_label": int(row.get("class_label", 0)),
                    "has_ps00134": bool(row.get("has_ps00134", False)),
                    "has_ps00135": bool(row.get("has_ps00135", False)),
                    "has_both_catalytic_motifs": bool(row.get("has_both_catalytic_motifs", False)),
                    "accession": row.get("accession", ""),
                }
            )
    return pd.DataFrame(rows)


def biological_ps00134_position_mask_available(dataset_path: str | Path = DEFAULT_DATASET) -> bool:
    columns = set(pd.read_csv(dataset_path, nrows=0).columns)
    coordinate_like = {c for c in columns if "ps00134" in c.lower() and any(term in c.lower() for term in ["start", "end", "pos", "coord"])}
    return bool(coordinate_like)


def make_displacement_grid(native_summary: pd.DataFrame) -> tuple[list[float], dict[str, Any]]:
    feature_summary = native_summary.loc[native_summary["feature_id"] == FEATURE_3256].iloc[0].to_dict()
    min_native = float(feature_summary["min"])
    q75 = float(feature_summary["percentile_75"])
    q95 = float(feature_summary["percentile_95"])
    nonzero_fraction = 1.0 - float(feature_summary["fraction_exactly_zero"])
    candidate_d = min(q95, 2.0 * max(q75, 0.0))
    feasible = candidate_d > 0.0 and min_native >= candidate_d
    metadata = {
        "scale_rule": "D = min(native p95, 2 * native p75), evaluated before outcome measurement",
        "candidate_D": float(candidate_d),
        "native_min": min_native,
        "native_q75": q75,
        "native_q95": q95,
        "native_nonzero_fraction": nonzero_fraction,
        "symmetric_grid_feasible_without_crossing_nonnegative_domain": bool(feasible),
        "feasibility_reason": "TopKSAE activations are ReLU/top-k nonnegative; a global negative displacement is only latent-domain-preserving when every native token activation is at least D.",
    }
    if not feasible:
        return [0.0], metadata
    d = float(candidate_d)
    return [-d, -d / 2.0, -d / 4.0, 0.0, d / 4.0, d / 2.0, d], metadata


def identity_centered_intervention(
    hidden_state,
    sae,
    feature_id: int,
    displacement: float,
    input_mean=None,
    input_std=None,
) -> tuple[Any, IdentityDiagnostics]:
    torch = import_model_helpers()["torch"]
    flat_hidden = hidden_state.float().reshape(-1, hidden_state.shape[-1])
    x = flat_hidden
    if input_mean is not None and input_std is not None:
        x = (x - input_mean.to(x.device)) / input_std.to(x.device)
    out = sae(x)
    z = out["z"]
    native = z[:, int(feature_id)]
    if isinstance(displacement, (float, int)):
        displacement_tensor = torch.full_like(native, float(displacement))
    else:
        displacement_tensor = displacement.to(native.device).to(native.dtype).expand_as(native)
    target = native + displacement_tensor
    delta_z = torch.zeros_like(z)
    delta_z[:, int(feature_id)] = target - native
    delta_x = (delta_z @ sae.decoder.weight.T).reshape_as(hidden_state)
    if input_mean is not None and input_std is not None:
        delta_x = delta_x * input_std.to(delta_x.device)
    patched = hidden_state + delta_x
    realized = (target - native).detach()
    hidden_delta = (patched.float() - hidden_state.float()).detach()
    diag = IdentityDiagnostics(
        native_min=float(native.detach().min().cpu().item()),
        native_max=float(native.detach().max().cpu().item()),
        requested_displacement=float(displacement_tensor.detach().mean().cpu().item()),
        realized_displacement_min=float(realized.min().cpu().item()),
        realized_displacement_max=float(realized.max().cpu().item()),
        max_abs_realized_minus_requested=float((realized - displacement_tensor.detach()).abs().max().cpu().item()),
        crosses_nonnegative_latent_domain=bool((target.detach() < 0).any().cpu().item()),
        max_abs_hidden_delta=float(hidden_delta.abs().max().cpu().item()),
    )
    return patched, diag


def absolute_target_equivalent_intervention(hidden_state, sae, feature_id: int, target_activation, input_mean=None, input_std=None):
    from scripts.evaluate_reconstruction import additive_sae_intervention

    return additive_sae_intervention(
        hidden_state=hidden_state,
        sae=sae,
        feature_id=int(feature_id),
        target_activation=target_activation,
        input_mean=input_mean,
        input_std=input_std,
    )


def summarize_native_activations(
    cohort: pd.DataFrame,
    model_name: str = DEFAULT_MODEL_NAME,
    sae_checkpoint: str | Path = DEFAULT_SAE_CKPT,
    device: str | None = None,
    layer_index: int = LAYER_INDEX,
) -> pd.DataFrame:
    helpers = import_model_helpers()
    torch = helpers["torch"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = helpers["ProGen3ForCausalLM"].from_pretrained(model_name, torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).to(device)
    model.eval()
    sae, sae_meta = helpers["load_sae_checkpoint"](sae_checkpoint, device)
    batch_preparer = helpers["ProGen3BatchPreparer"]()
    rows = []
    for feature_id, feature_cohort in cohort.groupby("feature_id", sort=True):
        values: list[float] = []
        for _, row in feature_cohort.drop_duplicates("sequence_id").iterrows():
            batch = batch_preparer.get_batch_kwargs([str(row["sequence"])], device=device, reverse=False)
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
                hidden = helpers["get_layer_hidden_state"](outputs, layer_index)
                x = hidden.float().reshape(-1, hidden.shape[-1])
                if sae_meta["input_mean"] is not None and sae_meta["input_std"] is not None:
                    x = (x - sae_meta["input_mean"].to(x.device)) / sae_meta["input_std"].to(x.device)
                z = sae(x)["z"][:, int(feature_id)].detach().cpu().numpy().astype(float)
            values.extend(z.tolist())
        arr = np.asarray(values, dtype=float)
        rows.append(
            {
                "feature_id": int(feature_id),
                "matched_concept": feature_concept_name(int(feature_id)) if int(feature_id) != CONTROL_FEATURE else "matched_control_feature_1_in_ps00134_cohort",
                "n_sequences": int(feature_cohort["sequence_id"].nunique()),
                "n_token_activations": int(arr.size),
                "min": float(np.min(arr)),
                "percentile_5": float(np.quantile(arr, 0.05)),
                "percentile_25": float(np.quantile(arr, 0.25)),
                "median": float(np.median(arr)),
                "mean": float(np.mean(arr)),
                "sd": float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0,
                "percentile_75": float(np.quantile(arr, 0.75)),
                "percentile_95": float(np.quantile(arr, 0.95)),
                "max": float(np.max(arr)),
                "fraction_exactly_zero": float(np.mean(arr == 0.0)),
                "sae_activation_domain": "nonnegative_ReLU_topk",
            }
        )
    return pd.DataFrame(rows)


def run_identity_centered_experiment(
    cohort: pd.DataFrame,
    displacements: Iterable[float],
    model_name: str = DEFAULT_MODEL_NAME,
    sae_checkpoint: str | Path = DEFAULT_SAE_CKPT,
    device: str | None = None,
    layer_index: int = LAYER_INDEX,
    top_k: int = LEGACY_TOP_K,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    helpers = import_model_helpers()
    torch = helpers["torch"]
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = helpers["ProGen3ForCausalLM"].from_pretrained(model_name, torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32).to(device)
    model.eval()
    sae, sae_meta = helpers["load_sae_checkpoint"](sae_checkpoint, device)
    batch_preparer = helpers["ProGen3BatchPreparer"]()

    summary_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for _, design_row in cohort.sort_values(["feature_id", "sequence_id"]).iterrows():
        seq = str(design_row["sequence"])
        batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
            base_hidden = helpers["get_layer_hidden_state"](outputs, layer_index)
            base_logits = outputs.logits
            base_nll = helpers["sequence_nll"](base_logits[0], batch["labels"][0], model.config.pad_token_id)
            for displacement in displacements:
                patched_hidden, diag = identity_centered_intervention(base_hidden, sae, int(design_row["feature_id"]), float(displacement), sae_meta["input_mean"], sae_meta["input_std"])
                patched_logits = helpers["forward_with_replaced_hidden_state"](model, batch, layer_index, patched_hidden)
                patched_nll = helpers["sequence_nll"](patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                raw_positions = token_signed_deltas(base_logits[0], patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                metrics = aggregate_signed_legacy_topk(raw_positions, top_k=top_k)
                max_abs_position_delta = float(max((abs(r["signed_delta_logprob"]) for r in raw_positions), default=0.0))
                base_record = {
                    "sequence_id": int(design_row["sequence_id"]),
                    "accession": str(design_row.get("accession", "")),
                    "feature_id": int(design_row["feature_id"]),
                    "matched_concept": str(design_row["matched_concept"]),
                    "concept_positive": bool(design_row["concept_positive"]),
                    "intervention_parameterization": "identity_centered_signed_displacement",
                    "requested_displacement": float(displacement),
                    "signed_displacement": float(displacement),
                    "realized_displacement_min": diag.realized_displacement_min,
                    "realized_displacement_max": diag.realized_displacement_max,
                    "max_abs_realized_minus_requested": diag.max_abs_realized_minus_requested,
                    "native_activation_min": diag.native_min,
                    "native_activation_max": diag.native_max,
                    "steered_activation_min": diag.native_min + float(displacement),
                    "steered_activation_max": diag.native_max + float(displacement),
                    "crosses_nonnegative_latent_domain": diag.crosses_nonnegative_latent_domain,
                    "clamping_applied": False,
                    "base_nll": float(base_nll),
                    "patched_nll": float(patched_nll),
                    "delta_nll": float(patched_nll - base_nll),
                    "kl": float(helpers["compute_mean_kl"](base_logits[0], patched_logits[0])),
                    "top1_agreement": float(helpers["top1_agreement"](base_logits[0], patched_logits[0])),
                    "max_abs_hidden_delta": diag.max_abs_hidden_delta,
                    "max_abs_signed_position_delta_logprob": max_abs_position_delta,
                    "top_k": int(top_k),
                    "position_set_definition": "legacy_topk_abs_delta_not_biological_ps00134_mask",
                    "biological_ps00134_position_mask_available": False,
                }
                summary_rows.append(
                    {
                        **base_record,
                        "signed_legacy_topk_delta_logprob": metrics.signed_motif_delta_logprob,
                        "signed_legacy_nontopk_delta_logprob": metrics.signed_nonmotif_delta_logprob,
                        "signed_legacy_topk_contrast": metrics.signed_motif_specificity,
                        "legacy_topk_abs_delta_logprob": metrics.motif_delta_logprob,
                        "legacy_nontopk_abs_delta_logprob": metrics.nonmotif_delta_logprob,
                        "legacy_unsigned_topk_contrast": metrics.motif_specificity_score,
                    }
                )
                for pos_row in metrics.position_rows:
                    residue_position, residue_identity = residue_metadata(seq, int(pos_row["token_position"]), int(pos_row["token_id"]))
                    position_rows.append(
                        {
                            **base_record,
                            "residue_position": residue_position,
                            "residue_identity": residue_identity,
                            "legacy_topk_position": bool(pos_row["legacy_topk_abs_delta_position"]),
                            "biological_ps00134_position": np.nan,
                            **{k: v for k, v in pos_row.items() if k not in {"motif_position", "motif_position_definition"}},
                        }
                    )
    metadata = {
        "script": "scripts/causal_feature_3256_identity_centered_directionality.py",
        "git_commit": git_commit(),
        "model_name": model_name,
        "sae_checkpoint": str(sae_checkpoint),
        "layer_index": int(layer_index),
        "top_k": int(top_k),
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features": sorted(int(v) for v in cohort["feature_id"].unique()),
        "intervention_definition": "z_steered[:, feature_id] = z_native[:, feature_id] + requested_displacement; h_steered = h + W_dec @ (z_steered - z_native)",
        "primary_metric": "signed_legacy_topk_contrast",
        "position_set_definition": "legacy_topk_abs_delta_not_biological_ps00134_mask",
        "biological_ps00134_position_mask_available": False,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(position_rows), metadata


def identity_smoke_check(summary: pd.DataFrame, tolerance: float = 1e-6) -> dict[str, Any]:
    zero = summary.loc[(summary["feature_id"] == FEATURE_3256) & np.isclose(summary["requested_displacement"], 0.0)]
    checks = {
        "checked_rows": int(len(zero)),
        "tolerance": float(tolerance),
        "max_abs_realized_minus_requested": float(zero["max_abs_realized_minus_requested"].abs().max()) if len(zero) else np.nan,
        "max_abs_hidden_delta": float(zero["max_abs_hidden_delta"].abs().max()) if len(zero) else np.nan,
        "max_abs_delta_nll": float(zero["delta_nll"].abs().max()) if len(zero) else np.nan,
        "max_abs_kl": float(zero["kl"].abs().max()) if len(zero) else np.nan,
        "max_abs_signed_position_delta_logprob": float(zero["max_abs_signed_position_delta_logprob"].abs().max()) if len(zero) else np.nan,
        "max_abs_signed_legacy_topk_contrast": float(zero["signed_legacy_topk_contrast"].abs().max()) if len(zero) else np.nan,
    }
    checks["passes"] = bool(
        len(zero)
        and checks["max_abs_realized_minus_requested"] <= tolerance
        and checks["max_abs_hidden_delta"] <= tolerance
        and checks["max_abs_delta_nll"] <= tolerance
        and checks["max_abs_kl"] <= tolerance
        and checks["max_abs_signed_position_delta_logprob"] <= tolerance
        and checks["max_abs_signed_legacy_topk_contrast"] <= tolerance
    )
    return checks


def steering_sign_check(summary: pd.DataFrame) -> dict[str, Any]:
    feat = summary.loc[summary["feature_id"] == FEATURE_3256]
    pos = feat.loc[feat["requested_displacement"] > 0]
    neg = feat.loc[feat["requested_displacement"] < 0]
    zero = feat.loc[np.isclose(feat["requested_displacement"], 0.0)]
    return {
        "positive_displacement_increases_activation": bool((pos["steered_activation_min"] > pos["native_activation_min"]).all()) if len(pos) else None,
        "negative_displacement_decreases_activation": bool((neg["steered_activation_max"] < neg["native_activation_max"]).all()) if len(neg) else None,
        "zero_displacement_preserves_activation": bool(np.isclose(zero["steered_activation_min"], zero["native_activation_min"]).all() and np.isclose(zero["steered_activation_max"], zero["native_activation_max"]).all()) if len(zero) else False,
        "any_negative_displacement_crosses_nonnegative_domain": bool(neg["crosses_nonnegative_latent_domain"].any()) if len(neg) else False,
    }


def load_identity_summary(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"sequence_id", "feature_id", "concept_positive", "requested_displacement", "signed_legacy_topk_contrast", "delta_nll", "kl"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns in {path}: {missing}")
    out = df.copy()
    out["sequence_id"] = out["sequence_id"].astype(str)
    out["feature_id"] = out["feature_id"].astype(int)
    out["concept_positive"] = _coerce_bool(out["concept_positive"])
    out["concept_positive_int"] = out["concept_positive"].astype(int)
    out["requested_displacement"] = pd.to_numeric(out["requested_displacement"], errors="raise")
    out["signed_legacy_topk_contrast"] = pd.to_numeric(out["signed_legacy_topk_contrast"], errors="raise")
    return out


def fit_identity_statistics(df: pd.DataFrame, n_bootstrap: int = DEFAULT_BOOTSTRAP_REPLICATES, seed: int = DEFAULT_BOOTSTRAP_SEED) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import statsmodels.formula.api as smf

    rows = []
    boot_rows = []
    slope_rows = []
    rng = np.random.default_rng(seed)
    for feature_id, feat in df.groupby("feature_id", sort=True):
        estimable = feat["concept_positive"].nunique() == 2 and feat["requested_displacement"].nunique() >= 2
        row = {
            "feature_id": int(feature_id),
            "matched_concept": str(feat["matched_concept"].iloc[0]),
            "response": "signed_legacy_topk_contrast",
            "n_rows": int(len(feat)),
            "n_sequences": int(feat["sequence_id"].nunique()),
            "n_concept_positive_sequences": int(feat.loc[feat["concept_positive"], "sequence_id"].nunique()),
            "n_concept_negative_sequences": int(feat.loc[~feat["concept_positive"], "sequence_id"].nunique()),
            "estimable": bool(estimable),
            "non_estimable_reason": "" if estimable else "requires both concept groups and at least two displacement values",
        }
        if estimable:
            model = smf.ols("signed_legacy_topk_contrast ~ requested_displacement * concept_positive_int", data=feat).fit()
            robust = model.get_robustcov_results(cov_type="cluster", groups=feat["sequence_id"])
            names = list(model.model.exog_names)
            params = pd.Series(np.asarray(robust.params), index=names)
            bse = pd.Series(np.asarray(robust.bse), index=names)
            pvalues = pd.Series(np.asarray(robust.pvalues), index=names)
            conf = pd.DataFrame(np.asarray(robust.conf_int()), index=names, columns=["low", "high"])
            term = [n for n in names if ":" in n and "requested_displacement" in n and "concept_positive_int" in n][0]
            row.update(
                {
                    "beta_displacement": float(params["requested_displacement"]),
                    "beta_concept": float(params["concept_positive_int"]),
                    "beta_displacement_x_concept": float(params[term]),
                    "interaction_clustered_se": float(bse[term]),
                    "interaction_ci_low": float(conf.loc[term, "low"]),
                    "interaction_ci_high": float(conf.loc[term, "high"]),
                    "interaction_p": float(pvalues[term]),
                    "interaction_term": term,
                }
            )
            seqs = np.asarray(sorted(feat["sequence_id"].unique()))
            failed = 0
            betas = []
            for iteration in range(int(n_bootstrap)):
                sampled = rng.choice(seqs, size=len(seqs), replace=True)
                boot = pd.concat([feat.loc[feat["sequence_id"] == seq] for seq in sampled], ignore_index=True)
                if boot["concept_positive"].nunique() < 2:
                    failed += 1
                    continue
                try:
                    bmod = smf.ols("signed_legacy_topk_contrast ~ requested_displacement * concept_positive_int", data=boot).fit()
                    bterm = [n for n in bmod.model.exog_names if ":" in n and "requested_displacement" in n and "concept_positive_int" in n][0]
                    beta = float(bmod.params[bterm])
                except Exception:
                    failed += 1
                    continue
                if np.isfinite(beta):
                    betas.append(beta)
                    boot_rows.append({"feature_id": int(feature_id), "bootstrap_iteration": int(iteration), "beta_displacement_x_concept": beta})
                else:
                    failed += 1
            values = np.asarray(betas, dtype=float)
            row.update(
                {
                    "bootstrap_n_requested": int(n_bootstrap),
                    "bootstrap_n_valid": int(values.size),
                    "bootstrap_n_failed": int(failed),
                    "bootstrap_median": float(np.median(values)) if values.size else np.nan,
                    "bootstrap_ci_low": float(np.quantile(values, 0.025)) if values.size else np.nan,
                    "bootstrap_ci_high": float(np.quantile(values, 0.975)) if values.size else np.nan,
                    "bootstrap_fraction_expected_sign": float(np.mean(values > 0.0)) if values.size else np.nan,
                }
            )
        rows.append(row)

        for sequence_id, seq_g in feat.groupby("sequence_id", sort=True):
            if seq_g["requested_displacement"].nunique() < 2:
                continue
            slope = float(np.polyfit(seq_g["requested_displacement"].to_numpy(float), seq_g["signed_legacy_topk_contrast"].to_numpy(float), 1)[0])
            slope_rows.append(
                {
                    "feature_id": int(feature_id),
                    "sequence_id": str(sequence_id),
                    "matched_concept": str(seq_g["matched_concept"].iloc[0]),
                    "concept_positive": bool(seq_g["concept_positive"].iloc[0]),
                    "metric": "signed_legacy_topk_contrast",
                    "slope": slope,
                    "n_displacements": int(seq_g["requested_displacement"].nunique()),
                }
            )
    slopes = pd.DataFrame(slope_rows)
    summaries = []
    if not slopes.empty:
        for (feature_id, concept_positive), g in slopes.groupby(["feature_id", "concept_positive"], sort=True):
            q25, q75 = np.quantile(g["slope"], [0.25, 0.75])
            summaries.append(
                {
                    "feature_id": int(feature_id),
                    "concept_positive": bool(concept_positive),
                    "n_sequences": int(g["sequence_id"].nunique()),
                    "mean_sequence_slope": float(g["slope"].mean()),
                    "median_sequence_slope": float(g["slope"].median()),
                    "iqr_low": float(q25),
                    "iqr_high": float(q75),
                    "fraction_expected_positive_slope": float(np.mean(g["slope"] > 0.0)),
                    "max_abs_slope_sequence_id": str(g.iloc[g["slope"].abs().argmax()]["sequence_id"]),
                    "max_abs_slope": float(g["slope"].iloc[g["slope"].abs().argmax()]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(boot_rows), slopes, pd.DataFrame(summaries)


def create_identity_directionality_figure(summary_csv: str | Path, stats_csv: str | Path, slopes_csv: str | Path, output_stem: str | Path = DEFAULT_FIGURE_STEM) -> list[str]:
    import matplotlib.pyplot as plt

    summary = load_identity_summary(summary_csv)
    stats = pd.read_csv(stats_csv)
    slopes = pd.read_csv(slopes_csv)
    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.7), constrained_layout=True)
    ax = axes[0, 0]
    ax.axis("off")
    ax.text(0.05, 0.75, "native z_3256", fontsize=11)
    ax.text(0.05, 0.52, "+ signed displacement", fontsize=11)
    ax.text(0.05, 0.29, "steered z_3256", fontsize=11)
    ax.annotate("", xy=(0.38, 0.34), xytext=(0.38, 0.73), arrowprops={"arrowstyle": "->", "lw": 1.2})
    ax.text(0.55, 0.58, "negative: suppression\nzero: identity\npositive: amplification", fontsize=9, va="center")
    ax.set_title("A. Native-Centered Intervention", loc="left", fontsize=12)

    ax = axes[0, 1]
    feature = summary.loc[summary["feature_id"] == FEATURE_3256]
    grouped = feature.groupby(["concept_positive", "requested_displacement"], as_index=False)["signed_legacy_topk_contrast"].agg(["mean", "sem"]).reset_index()
    for concept_positive, color, label in [(False, "#4c78a8", "PS00134-negative"), (True, "#f58518", "PS00134-positive")]:
        g = grouped.loc[grouped["concept_positive"] == concept_positive].sort_values("requested_displacement")
        if g.empty:
            continue
        ax.errorbar(g["requested_displacement"], g["mean"], yerr=g["sem"].fillna(0.0), marker="o", lw=1.4, color=color, label=label)
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("signed activation displacement")
    ax.set_ylabel("signed legacy top-k contrast")
    ax.set_title("B. Signed Dose Response", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 0]
    components = feature.groupby("requested_displacement", as_index=False)[["signed_legacy_topk_delta_logprob", "signed_legacy_nontopk_delta_logprob", "delta_nll"]].mean()
    ax.plot(components["requested_displacement"], components["signed_legacy_topk_delta_logprob"], marker="o", label="legacy top-k")
    ax.plot(components["requested_displacement"], components["signed_legacy_nontopk_delta_logprob"], marker="o", label="non-top-k")
    ax.axhline(0, color="black", lw=0.8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("signed activation displacement")
    ax.set_ylabel("signed true-token delta")
    ax.set_title("C. Perturbation-Selected Positions", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1, 1]
    for concept_positive, color, label in [(False, "#4c78a8", "negative"), (True, "#f58518", "positive")]:
        vals = slopes.loc[(slopes["feature_id"] == FEATURE_3256) & (_coerce_bool(slopes["concept_positive"]) == concept_positive), "slope"].to_numpy(float)
        if vals.size:
            ax.hist(vals, bins=min(12, max(4, vals.size // 2)), alpha=0.55, color=color, label=label)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("per-sequence slope")
    ax.set_ylabel("sequence count")
    ax.set_title("D. Sequence Robustness", loc="left", fontsize=12)
    ax.legend(frameon=False, fontsize=8)

    outputs = []
    for ext in ["png", "pdf", "svg"]:
        path = f"{output_stem}.{ext}"
        fig.savefig(path, dpi=350 if ext == "png" else None)
        outputs.append(path)
    plt.close(fig)
    return outputs


def append_figure_manifest(source_artifacts: list[str], outputs: list[str], manifest_path: str | Path = "results/figures/figure_manifest.csv") -> None:
    manifest = Path(manifest_path)
    existing = pd.read_csv(manifest) if manifest.exists() else pd.DataFrame(columns=["figure", "source_artifacts", "outputs", "reason"])
    existing = existing.loc[existing["figure"] != "figure_3256_identity_centered_directionality"].copy()
    row = pd.DataFrame(
        [
            {
                "figure": "figure_3256_identity_centered_directionality",
                "source_artifacts": ";".join(source_artifacts),
                "outputs": ";".join(outputs),
                "reason": "",
            }
        ]
    )
    pd.concat([existing, row], ignore_index=True).to_csv(manifest, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature 3256 identity-centered signed-displacement directionality experiment.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sae-checkpoint", default=DEFAULT_SAE_CKPT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--canonical-causal", default=DEFAULT_CANONICAL_CAUSAL)
    parser.add_argument("--native-summary-output", default=DEFAULT_NATIVE_SUMMARY_OUTPUT)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--position-output", default=DEFAULT_POSITION_OUTPUT)
    parser.add_argument("--statistics-output", default=DEFAULT_STATS_OUTPUT)
    parser.add_argument("--bootstrap-output", default=DEFAULT_BOOTSTRAP_OUTPUT)
    parser.add_argument("--sequence-slopes-output", default=DEFAULT_SEQUENCE_SLOPES_OUTPUT)
    parser.add_argument("--sequence-slope-summary-output", default=DEFAULT_SEQUENCE_SLOPE_SUMMARY_OUTPUT)
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--figure-stem", default=DEFAULT_FIGURE_STEM)
    parser.add_argument("--displacements", default=None, help="Comma-separated signed displacement grid. If omitted, derive from native activation summary.")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--top-k", type=int, default=LEGACY_TOP_K)
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--allow-domain-crossing", action="store_true")
    parser.add_argument("--make-figure", action="store_true")
    args = parser.parse_args()

    cohort = load_feature_3256_cohort(args.canonical_causal, args.dataset, include_control=True, max_sequences=args.max_sequences)
    native = summarize_native_activations(cohort, args.model_name, args.sae_checkpoint, args.device, args.layer_index)
    Path(args.native_summary_output).parent.mkdir(parents=True, exist_ok=True)
    native.to_csv(args.native_summary_output, index=False)
    mask_available = biological_ps00134_position_mask_available(args.dataset)
    if args.displacements:
        displacements = [float(v.strip()) for v in args.displacements.split(",") if v.strip()]
        grid_metadata = {"scale_rule": "user_provided", "symmetric_grid_feasible_without_crossing_nonnegative_domain": None}
    else:
        displacements, grid_metadata = make_displacement_grid(native)

    metadata = {
        "script": "scripts/causal_feature_3256_identity_centered_directionality.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "native_summary_output": args.native_summary_output,
        "biological_ps00134_position_mask_available": bool(mask_available),
        "displacement_grid": displacements,
        "grid_metadata": grid_metadata,
    }
    if args.prepare_only:
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return
    feature_native_min = float(native.loc[native["feature_id"] == FEATURE_3256, "min"].iloc[0])
    negative_grid_crosses_domain = any((feature_native_min + float(d)) < 0.0 for d in displacements)
    derived_infeasible = grid_metadata.get("symmetric_grid_feasible_without_crossing_nonnegative_domain") is False
    if not args.allow_domain_crossing and (derived_infeasible or negative_grid_crosses_domain):
        metadata["stop_condition"] = "negative_displacement_grid_crosses_nonnegative_topk_sae_latent_domain"
        metadata["feature_3256_native_min"] = feature_native_min
        metadata["negative_grid_crosses_domain"] = bool(negative_grid_crosses_domain)
        Path(args.metadata_output).write_text(json.dumps(metadata, indent=2, sort_keys=True))
        print(json.dumps(metadata, indent=2, sort_keys=True))
        raise SystemExit("Suppression displacement grid is not feasible without crossing the nonnegative TopKSAE latent domain; stopping before outcome measurement.")

    summary, positions, run_metadata = run_identity_centered_experiment(cohort, displacements, args.model_name, args.sae_checkpoint, args.device, args.layer_index, args.top_k)
    metadata.update(run_metadata)
    metadata["identity_smoke_check"] = identity_smoke_check(summary)
    metadata["steering_sign_check"] = steering_sign_check(summary)
    write_outputs(summary, positions, metadata, args.summary_output, args.position_output, args.metadata_output)
    if not metadata["identity_smoke_check"]["passes"]:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        raise SystemExit("Identity smoke check failed; stopping before full analysis/figure.")
    if args.smoke_only:
        print(json.dumps(metadata, indent=2, sort_keys=True))
        return

    stats, bootstrap, slopes, slope_summary = fit_identity_statistics(load_identity_summary(args.summary_output), args.bootstrap_replicates)
    stats.to_csv(args.statistics_output, index=False)
    bootstrap.to_csv(args.bootstrap_output, index=False)
    slopes.to_csv(args.sequence_slopes_output, index=False)
    slope_summary.to_csv(args.sequence_slope_summary_output, index=False)
    outputs: list[str] = []
    if args.make_figure:
        outputs = create_identity_directionality_figure(args.summary_output, args.statistics_output, args.sequence_slopes_output, args.figure_stem)
        append_figure_manifest(
            [args.summary_output, args.statistics_output, args.sequence_slopes_output, args.sequence_slope_summary_output],
            outputs,
        )
    result = {"metadata": metadata, "n_summary_rows": int(len(summary)), "n_position_rows": int(len(positions)), "figures": outputs}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
