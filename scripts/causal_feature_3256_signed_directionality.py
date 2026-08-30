from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PROGEN3_SRC = REPO_ROOT / "progen3" / "src"
if str(PROGEN3_SRC) not in sys.path:
    sys.path.insert(0, str(PROGEN3_SRC))

DEFAULT_MODEL_NAME = "Profluent-Bio/progen3-112m"
DEFAULT_SAE_CKPT = "results/topk_sae_layer6_d4096_k32_run1/best.pt"
DEFAULT_DATASET = "data/processed/s1a70/s1a70_test.csv"
DEFAULT_CANONICAL_CAUSAL = "results/causal_feature_dose_response.csv"
DEFAULT_SUMMARY_OUTPUT = "results/causal_feature_3256_signed_directionality.csv"
DEFAULT_POSITION_OUTPUT = "results/causal_feature_3256_signed_position_effects.csv"
DEFAULT_METADATA_OUTPUT = "results/causal_feature_3256_signed_metadata.json"
FEATURE_3256 = 3256
CONTROL_FEATURE = 1
LAYER_INDEX = 6
LEGACY_TOP_K = 16


@dataclass(frozen=True)
class SignedMetricResult:
    signed_motif_delta_logprob: float
    signed_nonmotif_delta_logprob: float
    signed_motif_specificity: float
    motif_delta_logprob: float
    nonmotif_delta_logprob: float
    motif_specificity_score: float
    position_rows: list[dict[str, Any]]


def token_signed_deltas(base_logits, patched_logits, labels, pad_token_id: int):
    import torch

    if base_logits.ndim == 3:
        base_logits = base_logits[0]
    if patched_logits.ndim == 3:
        patched_logits = patched_logits[0]
    if labels.ndim == 2:
        labels = labels[0]
    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    patched_logp = torch.log_softmax(patched_logits.float(), dim=-1)
    rows = []
    for pos, token in enumerate(labels):
        token_id = int(token.item())
        if token_id == int(pad_token_id) or token_id >= base_logp.shape[-1]:
            continue
        baseline = float(base_logp[pos, token_id].detach().cpu().item())
        steered = float(patched_logp[pos, token_id].detach().cpu().item())
        signed_delta = steered - baseline
        rows.append(
            {
                "token_position": int(pos),
                "token_id": token_id,
                "baseline_logprob": baseline,
                "steered_logprob": steered,
                "signed_delta_logprob": signed_delta,
                "abs_delta_logprob": abs(signed_delta),
            }
        )
    return rows


def aggregate_signed_legacy_topk(position_rows: list[dict[str, Any]], top_k: int = LEGACY_TOP_K) -> SignedMetricResult:
    if not position_rows:
        return SignedMetricResult(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, [])
    ordered = sorted(range(len(position_rows)), key=lambda i: position_rows[i]["abs_delta_logprob"])
    k = min(int(top_k), len(ordered))
    top_indices = set(ordered[-k:]) if k else set()
    annotated = []
    for idx, row in enumerate(position_rows):
        out = dict(row)
        out["legacy_topk_abs_delta_position"] = bool(idx in top_indices)
        out["motif_position"] = bool(idx in top_indices)
        out["motif_position_definition"] = "legacy_topk_abs_delta_not_biological_ps00134_mask"
        out["biological_motif_mask_available"] = False
        annotated.append(out)
    top = [r for r in annotated if r["legacy_topk_abs_delta_position"]]
    rest = [r for r in annotated if not r["legacy_topk_abs_delta_position"]]
    signed_top = float(np.mean([r["signed_delta_logprob"] for r in top])) if top else 0.0
    signed_rest = float(np.mean([r["signed_delta_logprob"] for r in rest])) if rest else 0.0
    unsigned_top = float(np.mean([r["abs_delta_logprob"] for r in top])) if top else 0.0
    unsigned_rest = float(np.mean([r["abs_delta_logprob"] for r in rest])) if rest else 0.0
    return SignedMetricResult(
        signed_motif_delta_logprob=signed_top,
        signed_nonmotif_delta_logprob=signed_rest,
        signed_motif_specificity=signed_top - signed_rest,
        motif_delta_logprob=unsigned_top,
        nonmotif_delta_logprob=unsigned_rest,
        motif_specificity_score=unsigned_top - unsigned_rest,
        position_rows=annotated,
    )


def residue_metadata(sequence: str, token_position: int, token_id: int) -> tuple[float, str]:
    residue_index = token_position - 2
    if 0 <= residue_index < len(sequence):
        return float(residue_index + 1), sequence[residue_index]
    return np.nan, f"token_id:{token_id}"


def load_target_design(canonical_causal_path: str | Path, dataset_path: str | Path, max_sequences: int | None = None) -> pd.DataFrame:
    causal = pd.read_csv(canonical_causal_path)
    dataset = pd.read_csv(dataset_path).reset_index().rename(columns={"index": "sequence_id_from_row"})
    causal = causal.loc[causal["feature_id"].isin([FEATURE_3256, CONTROL_FEATURE])].copy()
    if max_sequences is not None:
        allowed = set(sorted(causal["sequence_id"].unique())[: int(max_sequences)])
        causal = causal.loc[causal["sequence_id"].isin(allowed)].copy()
    needed = ["sequence_id", "feature_id", "matched_concept", "concept_positive", "baseline_feature_active", "dose", "target_value"]
    missing = [c for c in needed if c not in causal.columns]
    if missing:
        raise ValueError(f"Canonical causal CSV missing required columns: {missing}")
    design = causal[needed].drop_duplicates().copy()
    design["sequence_id"] = design["sequence_id"].astype(int)
    merged = design.merge(dataset, left_on="sequence_id", right_on="sequence_id_from_row", how="left", validate="many_to_one")
    if merged["sequence"].isna().any():
        missing_ids = merged.loc[merged["sequence"].isna(), "sequence_id"].drop_duplicates().tolist()
        raise ValueError(f"Could not recover sequences for sequence_id values: {missing_ids[:10]}")
    return merged.sort_values(["feature_id", "sequence_id", "dose"]).reset_index(drop=True)


def import_model_helpers():
    import torch
    from scripts.evaluate_reconstruction import (
        ProGen3BatchPreparer,
        ProGen3ForCausalLM,
        additive_sae_intervention,
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
        "additive_sae_intervention": additive_sae_intervention,
        "compute_mean_kl": compute_mean_kl,
        "forward_with_replaced_hidden_state": forward_with_replaced_hidden_state,
        "get_layer_hidden_state": get_layer_hidden_state,
        "load_sae_checkpoint": load_sae_checkpoint,
        "sequence_nll": sequence_nll,
        "top1_agreement": top1_agreement,
    }


def run_signed_experiment(
    design: pd.DataFrame,
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
    model = helpers["ProGen3ForCausalLM"].from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32,
    ).to(device)
    model.eval()
    sae, sae_meta = helpers["load_sae_checkpoint"](sae_checkpoint, device)
    batch_preparer = helpers["ProGen3BatchPreparer"]()

    summary_rows: list[dict[str, Any]] = []
    position_rows: list[dict[str, Any]] = []
    for (sequence_id, feature_id), seq_feature_design in design.groupby(["sequence_id", "feature_id"], sort=True):
        first = seq_feature_design.iloc[0]
        seq = str(first["sequence"])
        batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
            base_hidden = helpers["get_layer_hidden_state"](outputs, layer_index)
            base_logits = outputs.logits
            base_nll = helpers["sequence_nll"](base_logits[0], batch["labels"][0], model.config.pad_token_id)
            for row in seq_feature_design.itertuples(index=False):
                target_value = float(row.target_value)
                patched_hidden = helpers["additive_sae_intervention"](
                    hidden_state=base_hidden,
                    sae=sae,
                    feature_id=int(feature_id),
                    target_activation=target_value,
                    input_mean=sae_meta["input_mean"],
                    input_std=sae_meta["input_std"],
                )
                patched_logits = helpers["forward_with_replaced_hidden_state"](model, batch, layer_index, patched_hidden)
                patched_nll = helpers["sequence_nll"](patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                raw_positions = token_signed_deltas(base_logits[0], patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                metrics = aggregate_signed_legacy_topk(raw_positions, top_k=top_k)
                base_record = {
                    "sequence_id": int(sequence_id),
                    "concept_label": str(row.matched_concept),
                    "feature_id": int(feature_id),
                    "matched_concept": str(row.matched_concept),
                    "concept_positive": bool(row.concept_positive),
                    "baseline_feature_active": bool(row.baseline_feature_active),
                    "steering_direction": "suppression" if float(row.dose) < 0 else "target_zero" if float(row.dose) == 0 else "amplification",
                    "signed_dose": float(row.dose),
                    "dose_magnitude": abs(float(row.dose)),
                    "target_value": target_value,
                    "base_nll": float(base_nll),
                    "patched_nll": float(patched_nll),
                    "delta_nll": float(patched_nll - base_nll),
                    "kl": float(helpers["compute_mean_kl"](base_logits[0], patched_logits[0])),
                    "top1_agreement": float(helpers["top1_agreement"](base_logits[0], patched_logits[0])),
                    "top_k": int(top_k),
                    "position_set_definition": "legacy_topk_abs_delta_not_biological_ps00134_mask",
                    "biological_motif_mask_available": False,
                }
                summary_rows.append(
                    {
                        **base_record,
                        "signed_motif_delta_logprob": metrics.signed_motif_delta_logprob,
                        "signed_nonmotif_delta_logprob": metrics.signed_nonmotif_delta_logprob,
                        "signed_motif_specificity": metrics.signed_motif_specificity,
                        "motif_delta_logprob": metrics.motif_delta_logprob,
                        "nonmotif_delta_logprob": metrics.nonmotif_delta_logprob,
                        "motif_specificity_score": metrics.motif_specificity_score,
                    }
                )
                for pos_row in metrics.position_rows:
                    residue_position, residue_identity = residue_metadata(seq, int(pos_row["token_position"]), int(pos_row["token_id"]))
                    position_rows.append(
                        {
                            **base_record,
                            "residue_position": residue_position,
                            "residue_identity": residue_identity,
                            **pos_row,
                        }
                    )
    metadata = {
        "script": "scripts/causal_feature_3256_signed_directionality.py",
        "model_name": model_name,
        "sae_checkpoint": str(sae_checkpoint),
        "layer_index": int(layer_index),
        "top_k": int(top_k),
        "device": str(device),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "features": [FEATURE_3256, CONTROL_FEATURE],
        "signed_metric_definition": "mean signed true-token log-probability delta over the legacy top-k absolute-delta positions minus mean signed delta over all remaining positions",
        "position_set_definition": "legacy_topk_abs_delta_not_biological_ps00134_mask",
        "biological_ps00134_position_mask_available": False,
    }
    return pd.DataFrame(summary_rows), pd.DataFrame(position_rows), metadata


def validate_against_legacy_unsigned(new_summary: pd.DataFrame, canonical_causal_path: str | Path, tolerance: float = 5e-5) -> dict:
    canonical = pd.read_csv(canonical_causal_path)
    canonical = canonical.rename(columns={"dose": "signed_dose"})
    cols = ["sequence_id", "feature_id", "signed_dose"]
    merged = new_summary.merge(
        canonical[cols + ["motif_delta_logprob", "nonmotif_delta_logprob", "motif_specificity_score", "delta_nll", "kl"]],
        on=cols,
        suffixes=("_new", "_canonical"),
        how="left",
        validate="one_to_one",
    )
    result = {"n_compared_rows": int(len(merged)), "tolerance": float(tolerance)}
    for metric in ["motif_delta_logprob", "nonmotif_delta_logprob", "motif_specificity_score", "delta_nll", "kl"]:
        diff = (merged[f"{metric}_new"] - merged[f"{metric}_canonical"]).abs()
        result[f"max_abs_diff_{metric}"] = float(diff.max())
        result[f"{metric}_within_tolerance"] = bool((diff <= tolerance).all())
    result["all_legacy_unsigned_metrics_within_tolerance"] = all(v for k, v in result.items() if k.endswith("_within_tolerance"))
    return result


def target_zero_noop_check(summary: pd.DataFrame, tolerance: float = 1e-6) -> dict:
    zero = summary.loc[(summary["feature_id"] == FEATURE_3256) & (summary["signed_dose"] == 0.0)]
    max_abs_signed = float(zero["signed_motif_specificity"].abs().max()) if not zero.empty else np.nan
    max_abs_delta_nll = float(zero["delta_nll"].abs().max()) if not zero.empty else np.nan
    return {
        "checked_rows": int(len(zero)),
        "tolerance": float(tolerance),
        "max_abs_signed_motif_specificity_at_target_zero": max_abs_signed,
        "max_abs_delta_nll_at_target_zero": max_abs_delta_nll,
        "target_zero_is_noop_like": bool(max_abs_signed <= tolerance and max_abs_delta_nll <= tolerance) if len(zero) else False,
        "interpretation": "target_activation=0 is an identity/no-op condition" if len(zero) and max_abs_signed <= tolerance and max_abs_delta_nll <= tolerance else "target_activation=0 is not identity/no-op for the current feature-3256 cohort",
    }


def write_outputs(summary: pd.DataFrame, positions: pd.DataFrame, metadata: dict, summary_output: str | Path, position_output: str | Path, metadata_output: str | Path) -> None:
    summary_path = Path(summary_output)
    position_path = Path(position_output)
    metadata_path = Path(metadata_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    position_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    positions.to_csv(position_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Targeted signed localized directionality experiment for feature 3256 using the existing causal design.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--sae-checkpoint", default=DEFAULT_SAE_CKPT)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--canonical-causal", default=DEFAULT_CANONICAL_CAUSAL)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--position-output", default=DEFAULT_POSITION_OUTPUT)
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--layer-index", type=int, default=LAYER_INDEX)
    parser.add_argument("--top-k", type=int, default=LEGACY_TOP_K)
    parser.add_argument("--require-target-zero-noop", action="store_true")
    args = parser.parse_args()

    design = load_target_design(args.canonical_causal, args.dataset, max_sequences=args.max_sequences)
    summary, positions, metadata = run_signed_experiment(
        design=design,
        model_name=args.model_name,
        sae_checkpoint=args.sae_checkpoint,
        device=args.device,
        layer_index=args.layer_index,
        top_k=args.top_k,
    )
    legacy_check = validate_against_legacy_unsigned(summary, args.canonical_causal)
    zero_check = target_zero_noop_check(summary)
    metadata["legacy_unsigned_consistency_check"] = legacy_check
    metadata["target_zero_noop_check"] = zero_check
    write_outputs(summary, positions, metadata, args.summary_output, args.position_output, args.metadata_output)
    result = {"n_summary_rows": int(len(summary)), "n_position_rows": int(len(positions)), "legacy_check": legacy_check, "target_zero_noop_check": zero_check, "metadata": metadata}
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.require_target_zero_noop and not zero_check["target_zero_is_noop_like"]:
        raise SystemExit("target_activation=0 failed the requested no-op smoke invariant; full run should not proceed under the current task constraints")


if __name__ == "__main__":
    main()
