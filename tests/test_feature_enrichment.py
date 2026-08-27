import warnings

import pandas as pd

from analysis.feature_enrichment import compute_feature_enrichment


def test_compute_feature_enrichment_skips_constant_activation_warning():
    token_df = pd.DataFrame(
        [
            {"sequence_id": 0, "latent_idx": 0, "activation": 1.0},
            {"sequence_id": 0, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 1, "latent_idx": 0, "activation": 2.0},
            {"sequence_id": 1, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 2, "latent_idx": 0, "activation": 3.0},
            {"sequence_id": 2, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 3, "latent_idx": 0, "activation": 4.0},
            {"sequence_id": 3, "latent_idx": 1, "activation": 0.0},
        ]
    )
    seq_df = pd.DataFrame(
        {
            "sequence_id": [0, 1, 2, 3],
            "is_s1a": [1, 1, 0, 0],
        }
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = compute_feature_enrichment(token_df, seq_df, latent_ids=[0, 1], annotation_cols=["is_s1a"])

    assert result.empty is False
    assert not any("ConstantInputWarning" in str(w.message) for w in caught)


def test_compute_feature_enrichment_ranked_for_binary_annotations():
    token_df = pd.DataFrame(
        [
            {"sequence_id": 0, "latent_idx": 0, "activation": 5.0},
            {"sequence_id": 0, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 1, "latent_idx": 0, "activation": 6.0},
            {"sequence_id": 1, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 2, "latent_idx": 0, "activation": 7.0},
            {"sequence_id": 2, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 3, "latent_idx": 0, "activation": 4.0},
            {"sequence_id": 3, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 4, "latent_idx": 0, "activation": 3.0},
            {"sequence_id": 4, "latent_idx": 1, "activation": 0.0},
            {"sequence_id": 5, "latent_idx": 0, "activation": 0.0},
            {"sequence_id": 5, "latent_idx": 1, "activation": 2.5},
            {"sequence_id": 6, "latent_idx": 0, "activation": 0.0},
            {"sequence_id": 6, "latent_idx": 1, "activation": 2.0},
            {"sequence_id": 7, "latent_idx": 0, "activation": 0.0},
            {"sequence_id": 7, "latent_idx": 1, "activation": 1.5},
            {"sequence_id": 8, "latent_idx": 0, "activation": 0.0},
            {"sequence_id": 8, "latent_idx": 1, "activation": 1.0},
            {"sequence_id": 9, "latent_idx": 0, "activation": 0.0},
            {"sequence_id": 9, "latent_idx": 1, "activation": 3.0},
        ]
    )
    seq_df = pd.DataFrame(
        {
            "sequence_id": list(range(10)),
            "is_s1a": [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        }
    )

    result = compute_feature_enrichment(token_df, seq_df, latent_ids=[0, 1], annotation_cols=["is_s1a"])

    assert {"latent_idx", "annotation", "metric"}.issubset(set(result.columns))
    assert (result["latent_idx"] == 0).any()
    assert (result["annotation"] == "is_s1a").any()
    assert (result["auroc"] >= 0.9).any()
    assert (result["fisher_pvalue"] <= 0.05).any()
    assert (result["pointbiserial_r"] > 0).any()
