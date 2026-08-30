import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.causal_feature_3256_signed_directionality import (
    aggregate_signed_legacy_topk,
    load_target_design,
    residue_metadata,
    target_zero_noop_check,
    token_signed_deltas,
    validate_against_legacy_unsigned,
    write_outputs,
)


def test_signed_delta_preserves_sign_and_reverses_with_direction():
    base = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    up = torch.tensor([[0.0, 4.0], [0.0, 2.0]])
    down = torch.tensor([[0.0, 0.0], [4.0, 0.0]])
    labels = torch.tensor([1, 0])
    up_rows = token_signed_deltas(base, up, labels, pad_token_id=-1)
    down_rows = token_signed_deltas(base, down, labels, pad_token_id=-1)

    assert up_rows[0]["signed_delta_logprob"] > 0
    assert down_rows[0]["signed_delta_logprob"] < 0
    assert np.isclose(up_rows[0]["abs_delta_logprob"], abs(up_rows[0]["signed_delta_logprob"]))


def test_zero_logits_produce_zero_signed_effect():
    logits = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    labels = torch.tensor([1, 0])
    rows = token_signed_deltas(logits, logits.clone(), labels, pad_token_id=-1)
    metrics = aggregate_signed_legacy_topk(rows, top_k=1)

    assert all(abs(r["signed_delta_logprob"]) < 1e-12 for r in rows)
    assert abs(metrics.signed_motif_specificity) < 1e-12


def test_legacy_topk_mask_and_signed_aggregation_are_separate():
    rows = [
        {"token_position": 0, "token_id": 1, "baseline_logprob": 0.0, "steered_logprob": -3.0, "signed_delta_logprob": -3.0, "abs_delta_logprob": 3.0},
        {"token_position": 1, "token_id": 1, "baseline_logprob": 0.0, "steered_logprob": 1.0, "signed_delta_logprob": 1.0, "abs_delta_logprob": 1.0},
        {"token_position": 2, "token_id": 1, "baseline_logprob": 0.0, "steered_logprob": 0.5, "signed_delta_logprob": 0.5, "abs_delta_logprob": 0.5},
    ]
    metrics = aggregate_signed_legacy_topk(rows, top_k=1)

    assert metrics.position_rows[0]["legacy_topk_abs_delta_position"] is True
    assert metrics.position_rows[0]["motif_position_definition"] == "legacy_topk_abs_delta_not_biological_ps00134_mask"
    assert metrics.signed_motif_delta_logprob == -3.0
    assert metrics.motif_delta_logprob == 3.0
    assert metrics.signed_nonmotif_delta_logprob == 0.75
    assert metrics.signed_motif_specificity == -3.75


def test_signed_dose_grid_is_loaded_without_magnitude_conversion(tmp_path):
    causal = tmp_path / "causal.csv"
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sequence_id": 0, "feature_id": 3256, "matched_concept": "has_ps00134", "concept_positive": True, "baseline_feature_active": True, "dose": -2.0, "target_value": -2.0},
            {"sequence_id": 0, "feature_id": 3256, "matched_concept": "has_ps00134", "concept_positive": True, "baseline_feature_active": True, "dose": 0.0, "target_value": 0.0},
            {"sequence_id": 0, "feature_id": 3256, "matched_concept": "has_ps00134", "concept_positive": True, "baseline_feature_active": True, "dose": 2.0, "target_value": 2.0},
            {"sequence_id": 0, "feature_id": 1, "matched_concept": "matched_control", "concept_positive": True, "baseline_feature_active": False, "dose": 0.0, "target_value": 0.0},
        ]
    ).to_csv(causal, index=False)
    pd.DataFrame([{"sequence": "ACD", "has_ps00134": True}]).to_csv(dataset, index=False)
    design = load_target_design(causal, dataset)

    assert set(design["dose"]) == {-2.0, 0.0, 2.0}
    assert set(design["feature_id"]) == {1, 3256}


def test_position_metadata_marks_residues_and_special_tokens():
    assert residue_metadata("ACD", 2, 99) == (1.0, "A")
    pos, ident = residue_metadata("ACD", 0, 1)
    assert np.isnan(pos)
    assert ident == "token_id:1"


def test_target_zero_noop_check_detects_non_noop_zero_rows():
    summary = pd.DataFrame(
        [
            {"feature_id": 3256, "signed_dose": 0.0, "signed_motif_specificity": 0.01, "delta_nll": 0.0},
            {"feature_id": 3256, "signed_dose": 2.0, "signed_motif_specificity": 1.0, "delta_nll": 1.0},
        ]
    )
    check = target_zero_noop_check(summary, tolerance=1e-6)

    assert check["target_zero_is_noop_like"] is False
    assert "not identity" in check["interpretation"]


def test_validate_against_legacy_unsigned_metrics(tmp_path):
    canonical = tmp_path / "canonical.csv"
    summary = pd.DataFrame(
        [{"sequence_id": 0, "feature_id": 3256, "signed_dose": 1.0, "motif_delta_logprob": 0.3, "nonmotif_delta_logprob": 0.1, "motif_specificity_score": 0.2, "delta_nll": 0.4, "kl": 0.5}]
    )
    summary.to_csv(canonical, index=False)
    check = validate_against_legacy_unsigned(summary, canonical)

    assert check["all_legacy_unsigned_metrics_within_tolerance"] is True


def test_output_schema_contains_provenance_fields(tmp_path):
    summary = pd.DataFrame([{"sequence_id": 0, "feature_id": 3256, "signed_dose": 0.0}])
    positions = pd.DataFrame([{"sequence_id": 0, "feature_id": 3256, "token_position": 2, "signed_delta_logprob": 0.0}])
    metadata = {"script": "script.py", "model_name": "model", "sae_checkpoint": "sae.pt", "position_set_definition": "legacy_topk_abs_delta_not_biological_ps00134_mask"}
    write_outputs(summary, positions, metadata, tmp_path / "summary.csv", tmp_path / "positions.csv", tmp_path / "metadata.json")

    assert pd.read_csv(tmp_path / "summary.csv").columns.tolist() == ["sequence_id", "feature_id", "signed_dose"]
    assert "signed_delta_logprob" in pd.read_csv(tmp_path / "positions.csv").columns
    loaded = json.loads((tmp_path / "metadata.json").read_text())
    assert loaded["position_set_definition"] == "legacy_topk_abs_delta_not_biological_ps00134_mask"
