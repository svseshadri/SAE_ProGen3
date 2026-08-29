import hashlib

import numpy as np
import pandas as pd

from analysis.confirmatory_causal_statistics import (
    apply_bh_fdr,
    bootstrap_sequence_interactions,
    fit_clustered_interaction,
    is_estimable_feature,
    load_causal_data,
    resample_sequence_clusters,
    run_confirmatory_analysis,
)


def synthetic_feature_df(feature_id=3256, matched_concept="has_ps00134"):
    rows = []
    for seq_id in range(12):
        concept = seq_id >= 6
        seq_offset = 0.01 * seq_id
        for dose in [-1.0, 0.0, 1.0, 2.0]:
            specificity = 0.2 + 0.1 * dose + 0.3 * int(concept) + 0.7 * dose * int(concept) + seq_offset
            rows.append(
                {
                    "sequence_id": str(seq_id),
                    "feature_id": feature_id,
                    "matched_concept": matched_concept,
                    "concept_positive": concept,
                    "dose": dose,
                    "dose_centered": dose,
                    "concept_positive_int": int(concept),
                    "motif_specificity_score": specificity,
                }
            )
    return pd.DataFrame(rows)


def test_interaction_coefficient_extracted_from_known_synthetic_effect():
    result = fit_clustered_interaction(synthetic_feature_df())

    assert result.estimable
    assert np.isclose(result.beta_dose_x_concept, 0.7)
    assert result.interaction_term == "dose_centered:concept_positive_int"


def test_sequence_bootstrap_resamples_clusters_and_keeps_rows_grouped():
    df = synthetic_feature_df()
    boot = resample_sequence_clusters(df, ["0", "0", "7"])

    assert boot["bootstrap_sequence_id"].nunique() == 3
    assert set(boot["bootstrap_sequence_id"]) == {"0__draw_0", "0__draw_1", "7__draw_2"}
    assert len(boot.loc[boot["bootstrap_sequence_id"] == "0__draw_0"]) == 4
    assert len(boot.loc[boot["bootstrap_sequence_id"] == "0__draw_1"]) == 4
    assert boot.groupby("bootstrap_sequence_id")["dose"].nunique().eq(4).all()


def test_bootstrap_records_valid_cluster_level_interactions():
    boot, summary = bootstrap_sequence_interactions(synthetic_feature_df(), n_replicates=25, seed=13)

    assert summary["bootstrap_n_requested"] == 25
    assert summary["bootstrap_n_valid"] > 0
    assert summary["bootstrap_n_failed"] >= 0
    assert set(boot.columns) == {"feature_id", "bootstrap_iteration", "beta_dose_x_concept"}
    assert (boot["beta_dose_x_concept"] > 0).all()


def test_bh_fdr_applies_only_to_estimable_biological_candidates():
    summary = pd.DataFrame(
        [
            {"feature_id": 3256, "matched_concept": "has_ps00134", "estimable": True, "interaction_p": 0.01},
            {"feature_id": 2942, "matched_concept": "has_both_catalytic_motifs", "estimable": True, "interaction_p": 0.04},
            {"feature_id": 727, "matched_concept": "has_ipr001314", "estimable": False, "interaction_p": np.nan},
            {"feature_id": 1, "matched_concept": "matched_control", "estimable": True, "interaction_p": 0.001},
        ]
    )

    adjusted = apply_bh_fdr(summary)

    assert adjusted.loc[adjusted["feature_id"] == 3256, "interaction_q"].notna().item()
    assert adjusted.loc[adjusted["feature_id"] == 2942, "interaction_q"].notna().item()
    assert adjusted.loc[adjusted["feature_id"] == 727, "interaction_q"].isna().item()
    assert adjusted.loc[adjusted["feature_id"] == 1, "interaction_q"].isna().item()


def test_missing_concept_negative_support_is_non_estimable():
    df = synthetic_feature_df()
    only_positive = df.loc[df["concept_positive"]].copy()

    estimable, reason = is_estimable_feature(only_positive)
    result = fit_clustered_interaction(only_positive)

    assert not estimable
    assert "concept" in reason
    assert not result.estimable
    assert np.isnan(result.beta_dose_x_concept)


def test_run_analysis_does_not_mutate_input_csv(tmp_path):
    input_path = tmp_path / "causal.csv"
    df = synthetic_feature_df()
    df.drop(columns=["dose_centered", "concept_positive_int"]).to_csv(input_path, index=False)
    before = hashlib.sha256(input_path.read_bytes()).hexdigest()

    run_confirmatory_analysis(
        input_path=input_path,
        summary_output=tmp_path / "summary.csv",
        bootstrap_output=tmp_path / "bootstrap.csv",
        bootstrap_replicates=5,
        bootstrap_seed=7,
    )

    after = hashlib.sha256(input_path.read_bytes()).hexdigest()
    assert before == after
    loaded = load_causal_data(input_path)
    assert "dose_centered" in loaded.columns
    assert "dose_centered" not in pd.read_csv(input_path).columns
