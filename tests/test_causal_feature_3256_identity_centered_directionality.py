import json

import numpy as np
import pandas as pd
import torch

from scripts.causal_feature_3256_identity_centered_directionality import (
    biological_ps00134_position_mask_available,
    create_identity_directionality_figure,
    fit_identity_statistics,
    identity_centered_intervention,
    identity_smoke_check,
    load_feature_3256_cohort,
    load_identity_summary,
    make_displacement_grid,
    steering_sign_check,
)
from scripts.causal_feature_3256_signed_directionality import aggregate_signed_legacy_topk, token_signed_deltas
from scripts.evaluate_reconstruction import additive_sae_intervention


class TinySAE(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.decoder.weight.copy_(torch.tensor([[1.0, 2.0, 0.0], [0.0, -1.0, 3.0]]))

    def forward(self, x):
        z = torch.relu(torch.tensor([[0.5, 0.0, 2.0], [1.0, 3.0, 0.0]], dtype=x.dtype, device=x.device))
        return {"z": z}


def test_displacement_zero_leaves_z_and_hidden_delta_unchanged():
    sae = TinySAE()
    hidden = torch.zeros(1, 2, 2)

    patched, diag = identity_centered_intervention(hidden, sae, feature_id=1, displacement=0.0)

    assert torch.equal(patched, hidden)
    assert diag.realized_displacement_min == 0.0
    assert diag.realized_displacement_max == 0.0
    assert diag.max_abs_hidden_delta == 0.0


def test_positive_and_negative_displacements_change_activation_direction_when_feasible():
    sae = TinySAE()
    hidden = torch.zeros(1, 2, 2)

    _, up = identity_centered_intervention(hidden, sae, feature_id=0, displacement=0.25)
    _, down = identity_centered_intervention(hidden, sae, feature_id=0, displacement=-0.25)

    assert up.realized_displacement_min > 0
    assert up.realized_displacement_max > 0
    assert down.realized_displacement_min < 0
    assert down.realized_displacement_max < 0
    assert down.crosses_nonnegative_latent_domain is False


def test_negative_displacement_records_domain_crossing_without_clamping():
    sae = TinySAE()
    hidden = torch.zeros(1, 2, 2)

    _, diag = identity_centered_intervention(hidden, sae, feature_id=1, displacement=-0.25)

    assert diag.crosses_nonnegative_latent_domain is True
    assert np.isclose(diag.realized_displacement_min, -0.25)
    assert np.isclose(diag.realized_displacement_max, -0.25)


def test_absolute_target_equivalence_for_matching_displacement():
    sae = TinySAE()
    hidden = torch.zeros(1, 2, 2)
    target = torch.tensor([1.25, 1.75])
    native = sae(hidden.reshape(-1, hidden.shape[-1]))["z"][:, 0]
    displacement = target - native

    patched_new, diag = identity_centered_intervention(hidden, sae, feature_id=0, displacement=displacement)
    patched_old = additive_sae_intervention(hidden, sae, feature_id=0, target_activation=target)

    assert torch.allclose(patched_new, patched_old)
    assert diag.max_abs_realized_minus_requested == 0.0


def test_signed_delta_retains_sign_and_zero_logits_are_identity():
    base = torch.tensor([[0.0, 2.0], [2.0, 0.0]])
    up = torch.tensor([[0.0, 4.0], [0.0, 2.0]])
    labels = torch.tensor([1, 0])
    rows = token_signed_deltas(base, up, labels, pad_token_id=-1)
    metrics = aggregate_signed_legacy_topk(rows, top_k=1)

    assert rows[0]["signed_delta_logprob"] > 0
    top_rows = [r for r in metrics.position_rows if r["legacy_topk_abs_delta_position"]]
    assert len(top_rows) == 1
    assert top_rows[0]["motif_position_definition"] == "legacy_topk_abs_delta_not_biological_ps00134_mask"

    zero_rows = token_signed_deltas(base, base.clone(), labels, pad_token_id=-1)
    assert all(abs(r["signed_delta_logprob"]) < 1e-12 for r in zero_rows)


def test_native_grid_stops_when_symmetric_negative_displacement_is_infeasible():
    native = pd.DataFrame(
        [
            {
                "feature_id": 3256,
                "min": 0.0,
                "percentile_75": 1.0,
                "percentile_95": 3.0,
                "fraction_exactly_zero": 0.8,
            }
        ]
    )

    grid, metadata = make_displacement_grid(native)

    assert grid == [0.0]
    assert metadata["symmetric_grid_feasible_without_crossing_nonnegative_domain"] is False


def test_identity_smoke_and_sign_checks_use_realized_values():
    summary = pd.DataFrame(
        [
            {
                "feature_id": 3256,
                "requested_displacement": 0.0,
                "max_abs_realized_minus_requested": 0.0,
                "max_abs_hidden_delta": 0.0,
                "delta_nll": 0.0,
                "kl": 0.0,
                "max_abs_signed_position_delta_logprob": 0.0,
                "signed_legacy_topk_contrast": 0.0,
                "native_activation_min": 0.5,
                "native_activation_max": 1.0,
                "steered_activation_min": 0.5,
                "steered_activation_max": 1.0,
                "crosses_nonnegative_latent_domain": False,
            },
            {
                "feature_id": 3256,
                "requested_displacement": 0.25,
                "max_abs_realized_minus_requested": 0.0,
                "max_abs_hidden_delta": 0.5,
                "delta_nll": 0.1,
                "kl": 0.1,
                "max_abs_signed_position_delta_logprob": 0.2,
                "signed_legacy_topk_contrast": 0.2,
                "native_activation_min": 0.5,
                "native_activation_max": 1.0,
                "steered_activation_min": 0.75,
                "steered_activation_max": 1.25,
                "crosses_nonnegative_latent_domain": False,
            },
            {
                "feature_id": 3256,
                "requested_displacement": -0.25,
                "max_abs_realized_minus_requested": 0.0,
                "max_abs_hidden_delta": 0.5,
                "delta_nll": 0.1,
                "kl": 0.1,
                "max_abs_signed_position_delta_logprob": 0.2,
                "signed_legacy_topk_contrast": -0.2,
                "native_activation_min": 0.5,
                "native_activation_max": 1.0,
                "steered_activation_min": 0.25,
                "steered_activation_max": 0.75,
                "crosses_nonnegative_latent_domain": False,
            },
        ]
    )

    assert identity_smoke_check(summary)["passes"] is True
    sign = steering_sign_check(summary)
    assert sign["positive_displacement_increases_activation"] is True
    assert sign["negative_displacement_decreases_activation"] is True
    assert sign["zero_displacement_preserves_activation"] is True


def test_load_feature_3256_cohort_uses_existing_ps00134_labels(tmp_path):
    causal = tmp_path / "causal.csv"
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sequence_id": 0, "feature_id": 3256, "dose": 0.0},
            {"sequence_id": 1, "feature_id": 3256, "dose": 0.0},
            {"sequence_id": 0, "feature_id": 1, "dose": 0.0},
        ]
    ).to_csv(causal, index=False)
    pd.DataFrame(
        [
            {"sequence": "ACD", "length": 3, "has_ps00134": True, "has_ps00135": False, "has_both_catalytic_motifs": False, "class_label": 1},
            {"sequence": "EFG", "length": 3, "has_ps00134": False, "has_ps00135": False, "has_both_catalytic_motifs": False, "class_label": 0},
        ]
    ).to_csv(dataset, index=False)

    cohort = load_feature_3256_cohort(causal, dataset, include_control=True)

    assert set(cohort["feature_id"]) == {1, 3256}
    assert cohort.loc[cohort["sequence_id"] == 0, "concept_positive"].all()
    assert not cohort.loc[cohort["sequence_id"] == 1, "concept_positive"].any()


def test_biological_mask_detection_requires_position_columns(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([{"sequence": "ACD", "has_ps00134": True, "prosite_ids": "PS00134;"}]).to_csv(dataset, index=False)

    assert biological_ps00134_position_mask_available(dataset) is False


def test_statistics_and_plot_consume_generated_csvs(tmp_path):
    summary = pd.DataFrame(
        [
            {
                "sequence_id": seq,
                "feature_id": 3256,
                "matched_concept": "has_ps00134",
                "concept_positive": concept,
                "requested_displacement": d,
                "signed_legacy_topk_contrast": (0.2 + 0.3 * concept) * d,
                "signed_legacy_topk_delta_logprob": 0.1 * d,
                "signed_legacy_nontopk_delta_logprob": 0.02 * d,
                "delta_nll": abs(d) * 0.01,
                "kl": abs(d) * 0.01,
            }
            for seq, concept in [("p1", True), ("p2", True), ("n1", False), ("n2", False)]
            for d in [-1.0, 0.0, 1.0]
        ]
    )
    summary_path = tmp_path / "summary.csv"
    summary.to_csv(summary_path, index=False)

    loaded = load_identity_summary(summary_path)
    stats, bootstrap, slopes, slope_summary = fit_identity_statistics(loaded, n_bootstrap=20, seed=1)
    stats_path = tmp_path / "stats.csv"
    slopes_path = tmp_path / "slopes.csv"
    stats.to_csv(stats_path, index=False)
    slopes.to_csv(slopes_path, index=False)

    outputs = create_identity_directionality_figure(summary_path, stats_path, slopes_path, tmp_path / "figure")

    assert stats.loc[0, "beta_displacement_x_concept"] > 0
    assert not bootstrap.empty
    assert not slope_summary.empty
    assert all((tmp_path / f"figure.{ext}").exists() for ext in ["png", "pdf", "svg"])
    assert len(outputs) == 3
