import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.signed_directionality_analysis import (
    audit_metric_definition,
    create_figures,
    load_causal_data,
    per_sequence_slopes,
    resample_sequence_clusters,
    run_analysis,
    summarize_by_dose,
)


def synthetic_causal_df():
    rows = []
    for seq_id in range(8):
        concept = seq_id >= 4
        for dose in [-2.0, 0.0, 2.0]:
            signed = (1 if concept else 0.5) * dose
            rows.append({
                "sequence_id": seq_id,
                "feature_id": 3256,
                "matched_concept": "has_ps00134",
                "concept_positive": concept,
                "dose": dose,
                "motif_delta_logprob": abs(signed) + 0.2,
                "nonmotif_delta_logprob": abs(signed) * 0.2 + 0.1,
                "motif_specificity_score": abs(signed) * 0.8 + 0.1,
                "delta_nll": -signed,
                "kl": abs(signed) * 0.01,
            })
    for seq_id in range(8):
        concept = seq_id >= 4
        for dose in [-1.0, 0.0, 1.0]:
            signed = (0.8 if concept else 0.4) * dose
            rows.append({
                "sequence_id": seq_id,
                "feature_id": 2942,
                "matched_concept": "has_both_catalytic_motifs",
                "concept_positive": concept,
                "dose": dose,
                "motif_delta_logprob": abs(signed) + 0.1,
                "nonmotif_delta_logprob": abs(signed) * 0.2,
                "motif_specificity_score": abs(signed) * 0.8 + 0.1,
                "delta_nll": -signed,
                "kl": abs(signed) * 0.02,
            })
    return pd.DataFrame(rows)


def write_core_artifacts(tmp_path):
    causal = tmp_path / "causal.csv"
    synthetic_causal_df().to_csv(causal, index=False)
    confirmatory = tmp_path / "confirmatory.csv"
    pd.DataFrame([
        {"feature_id": 3256, "matched_concept": "has_ps00134", "estimable": True, "beta_dose_x_concept": 0.1, "interaction_ci_low": 0.02, "interaction_ci_high": 0.18, "interaction_q": 0.03},
        {"feature_id": 2942, "matched_concept": "has_both_catalytic_motifs", "estimable": True, "beta_dose_x_concept": 0.2, "interaction_ci_low": 0.1, "interaction_ci_high": 0.3, "interaction_q": 0.01},
        {"feature_id": 1644, "matched_concept": "has_ps00135", "estimable": True, "beta_dose_x_concept": 0.05, "interaction_ci_low": 0.01, "interaction_ci_high": 0.09, "interaction_q": 0.02},
    ]).to_csv(confirmatory, index=False)
    recon = tmp_path / "recon.csv"
    pd.DataFrame({"base_nll": [2.0, 2.1], "patched_nll": [2.5, 2.6], "nmse": [0.02, 0.03], "explained_variance": [0.98, 0.97]}).to_csv(recon, index=False)
    ident = tmp_path / "identity.csv"
    pd.DataFrame({"patched_nll": [2.0, 2.1]}).to_csv(ident, index=False)
    resid = tmp_path / "resid.csv"
    pd.DataFrame({"sequence_id": [0, 1, 0, 1], "lambda": [0.0, 0.0, 1.0, 1.0], "sae_delta_nll": [0.4, 0.5, 0.0, 0.0], "random_noise_delta_nll": [0.3, 0.3, 0.3, 0.3]}).to_csv(resid, index=False)
    random = tmp_path / "random.csv"
    pd.DataFrame({"random_noise_delta_nll": [0.3, 0.31]}).to_csv(random, index=False)
    return causal, confirmatory, recon, ident, resid, random


def test_audit_records_that_localized_metric_is_unsigned():
    audit = audit_metric_definition()

    assert audit["uses_absolute_value"] is True
    assert audit["preserves_underlying_sign"] is False
    assert audit["signed_directionality_testable_from_motif_specificity_score"] is False


def test_sign_is_preserved_only_for_global_delta_nll_not_localized_metric(tmp_path):
    path = tmp_path / "causal.csv"
    synthetic_causal_df().to_csv(path, index=False)
    df = load_causal_data(path)

    neg = df[(df.feature_id == 3256) & (df.dose < 0)].iloc[0]
    pos = df[(df.feature_id == 3256) & (df.dose > 0)].iloc[0]
    assert neg["global_signed_nll_effect"] < 0
    assert pos["global_signed_nll_effect"] > 0
    assert neg["motif_specificity_score"] > 0
    assert pos["motif_specificity_score"] > 0


def test_negative_and_positive_doses_are_not_converted_to_magnitude(tmp_path):
    path = tmp_path / "causal.csv"
    synthetic_causal_df().to_csv(path, index=False)
    df = load_causal_data(path)
    by_dose = summarize_by_dose(df, [3256], n_bootstrap=5)

    assert (by_dose["dose"] < 0).any()
    assert (by_dose["dose"] > 0).any()
    assert set(by_dose["dose_class"]) == {"suppression", "no_op", "amplification"}


def test_concept_groups_remain_separate_and_repeated_rows_are_clustered():
    df = load_causal_data_from_frame(synthetic_causal_df())
    by_dose = summarize_by_dose(df, [3256], n_bootstrap=5)

    assert set(by_dose["concept_positive"]) == {False, True}
    assert by_dose.groupby(["concept_positive", "dose"])["n_sequences"].first().eq(4).all()


def test_sequence_bootstrap_keeps_duplicate_sequence_draws_distinct():
    df = load_causal_data_from_frame(synthetic_causal_df())
    boot = resample_sequence_clusters(df[df.feature_id == 3256], ["0", "0", "5"])

    assert set(boot["bootstrap_sequence_id"]) == {"0__draw_0", "0__draw_1", "5__draw_2"}
    assert boot.groupby("bootstrap_sequence_id")["dose"].nunique().eq(3).all()


def test_sequence_slopes_require_sufficient_dose_support():
    df = load_causal_data_from_frame(synthetic_causal_df())
    slopes = per_sequence_slopes(df, 3256)
    sparse = df[df["dose"] == 0].copy()
    sparse_slopes = per_sequence_slopes(sparse, 3256)

    assert slopes["sequence_id"].nunique() == 8
    assert {"motif_specificity_score", "global_signed_nll_effect"}.issubset(set(slopes["metric"]))
    assert sparse_slopes.empty


def test_run_analysis_does_not_mutate_input_csv(tmp_path):
    causal, *_ = write_core_artifacts(tmp_path)
    before = hashlib.sha256(causal.read_bytes()).hexdigest()
    run_analysis(causal_input=causal, output_dir=tmp_path, bootstrap_replicates=5, by_dose_bootstrap_replicates=5)
    after = hashlib.sha256(causal.read_bytes()).hexdigest()

    assert before == after
    assert (tmp_path / "signed_directionality_metric_audit.csv").exists()
    stats = pd.read_csv(tmp_path / "signed_directionality_statistics.csv")
    assert stats.loc[0, "localized_signed_metric_available"] == False


def test_plotting_functions_read_result_files(tmp_path):
    causal, confirmatory, recon, ident, resid, random = write_core_artifacts(tmp_path)
    run_analysis(causal_input=causal, output_dir=tmp_path, bootstrap_replicates=5, by_dose_bootstrap_replicates=5)
    manifest = create_figures(
        causal_input=causal,
        confirmatory_input=confirmatory,
        reconstruction_input=recon,
        identity_input=ident,
        residual_input=resid,
        random_noise_input=random,
        output_dir=tmp_path,
        figure_dir=tmp_path / "figures",
    )

    assert "not_created" in set(manifest["outputs"])
    created = manifest[manifest["outputs"] != "not_created"]
    assert not created.empty
    for outputs in created["outputs"]:
        for output in outputs.split(";"):
            assert Path(output).exists()


def load_causal_data_from_frame(frame):
    path = Path("/tmp/synthetic_signed_directionality.csv")
    frame.to_csv(path, index=False)
    return load_causal_data(path)
