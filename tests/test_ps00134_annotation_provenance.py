import pandas as pd

from analysis.ps00134_annotation_provenance import (
    audit_dataset,
    coordinate_like_columns,
    ps00134_from_prosite_ids,
    run_audit,
)


def test_binary_ps00134_labels_reproduce_from_prosite_ids(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"has_ps00134": True, "prosite_ids": "PS50240;PS00134;PS00135;"},
            {"has_ps00134": False, "prosite_ids": "PS50240;"},
            {"has_ps00134": True, "prosite_ids": "PS00134;"},
        ]
    ).to_csv(dataset, index=False)

    audit = audit_dataset(dataset)

    assert audit["binary_labels_reproduced"] is True
    assert audit["n_binary_mismatches"] == 0
    assert audit["n_has_ps00134_true"] == 2
    assert audit["n_regenerated_true"] == 2


def test_audit_detects_binary_label_mismatches(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([{"has_ps00134": True, "prosite_ids": "PS50240;"}]).to_csv(dataset, index=False)

    audit = audit_dataset(dataset)

    assert audit["binary_labels_reproduced"] is False
    assert audit["n_binary_mismatches"] == 1


def test_coordinate_like_columns_require_actual_coordinate_schema(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame([{"has_ps00134": True, "prosite_ids": "PS00134;", "sequence": "ACD"}]).to_csv(dataset, index=False)

    audit = audit_dataset(dataset)

    assert audit["has_position_level_ps00134_provenance"] is False
    assert audit["ps00134_coordinate_like_columns"] == ""


def test_coordinate_like_columns_are_reported_when_present(tmp_path):
    dataset = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "has_ps00134": True,
                "prosite_ids": "PS00134;",
                "ps00134_match_start": 2,
                "ps00134_match_end": 7,
            }
        ]
    ).to_csv(dataset, index=False)

    audit = audit_dataset(dataset)

    assert audit["has_position_level_ps00134_provenance"] is True
    assert "ps00134_match_start" in audit["ps00134_coordinate_like_columns"]


def test_run_audit_gate_a_requires_binary_reproduction_and_positions(tmp_path):
    with_positions = tmp_path / "with_positions.csv"
    without_positions = tmp_path / "without_positions.csv"
    pd.DataFrame([{"has_ps00134": True, "prosite_ids": "PS00134;", "ps00134_match_start": 1}]).to_csv(with_positions, index=False)
    pd.DataFrame([{"has_ps00134": True, "prosite_ids": "PS00134;"}]).to_csv(without_positions, index=False)

    _, pass_meta = run_audit((str(with_positions),))
    _, fail_meta = run_audit((str(without_positions),))

    assert pass_meta["gate_a_passes"] is True
    assert fail_meta["gate_a_passes"] is False


def test_ps00134_from_prosite_ids_is_literal_crossref_match():
    values = pd.Series(["PS00134;", "PS0013;", "", None])

    assert ps00134_from_prosite_ids(values).tolist() == [True, False, False, False]
    assert coordinate_like_columns(["match_start", "sequence", "residue_position"]) == ["match_start", "residue_position"]
