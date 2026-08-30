import json

import pandas as pd

from analysis.ps00134_scanprosite_recovery import (
    PS00134_REGEX,
    expand_position_rows,
    fasta_for_cohort,
    load_exact_causal_cohort,
    normalize_scan_matches,
    sequence_audit,
    sha256_text,
)


def test_fasta_uses_stable_sequence_id_headers():
    cohort = pd.DataFrame(
        [
            {"sequence_id": 7, "sequence": "ACDE"},
            {"sequence_id": 9, "sequence": "FGHI"},
        ]
    )

    fasta = fasta_for_cohort(cohort)

    assert fasta == ">seq_7\nACDE\n>seq_9\nFGHI\n"
    assert sha256_text(fasta) == sha256_text(fasta)


def test_load_exact_causal_cohort_uses_feature_3256_sequence_ids(tmp_path):
    dataset = tmp_path / "dataset.csv"
    causal = tmp_path / "causal.csv"
    pd.DataFrame(
        [
            {"sequence": "ACDE", "has_ps00134": True, "length": 4},
            {"sequence": "FGHI", "has_ps00134": False, "length": 4},
            {"sequence": "KLMN", "has_ps00134": True, "length": 4},
        ]
    ).to_csv(dataset, index=False)
    pd.DataFrame(
        [
            {"sequence_id": 2, "feature_id": 3256},
            {"sequence_id": 0, "feature_id": 3256},
            {"sequence_id": 1, "feature_id": 1},
        ]
    ).to_csv(causal, index=False)

    cohort = load_exact_causal_cohort(dataset, causal)

    assert cohort["sequence_id"].tolist() == [0, 2]
    assert cohort["sequence"].tolist() == ["ACDE", "KLMN"]


def test_normalize_scan_matches_slices_exact_stored_sequence_coordinates():
    cohort = pd.DataFrame(
        [
            {
                "sequence_id": 0,
                "sequence": "XXLSASHCYY",
                "length": 10,
                "has_ps00134": True,
                "accession": "A0",
                "entry_name": "TEST",
                "prosite_ids": "PS00134;",
            }
        ]
    )
    raw = {
        "n_match": 1,
        "n_seq": 1,
        "matchset": [
            {
                "sequence_ac": "seq_0",
                "start": 3,
                "stop": 8,
                "signature_ac": "PS00134",
                "signature_id": "TRYPSIN_HIS",
                "level_tag": "(0)",
            }
        ],
    }

    matches = normalize_scan_matches(raw, cohort)

    assert matches.loc[0, "matched_sequence"] == "LSASHC"
    assert matches.loc[0, "ps00134_scan_confidence"] == "(0)"
    assert matches.loc[0, "match_length"] == 6


def test_position_rows_preserve_one_based_coordinates_and_act_site_offset():
    matches = pd.DataFrame(
        [
            {
                "sequence_id": 0,
                "sequence_length": 10,
                "has_ps00134": True,
                "accession": "A0",
                "entry_name": "TEST",
                "prosite_ids": "PS00134;",
                "ps00134_scan_confidence": "(0)",
                "match_start_1_based": 3,
                "match_end_1_based": 8,
                "matched_sequence": "LSASHC",
                "annotation_source": "ScanProsite",
                "annotation_method": "scan",
            }
        ]
    )

    positions = expand_position_rows(matches)

    assert positions["residue_position_1_based"].tolist() == [3, 4, 5, 6, 7, 8]
    assert positions["residue_index_0_based"].tolist() == [2, 3, 4, 5, 6, 7]
    assert positions.loc[positions["ps00134_prorule_act_site"], "residue_identity"].tolist() == ["H"]
    assert positions["match_index"].nunique() == 1


def test_sequence_audit_keeps_has_ps00134_distinct_from_recoverable_pattern():
    cohort = pd.DataFrame(
        [
            {"sequence_id": 0, "accession": "A0", "entry_name": "POS", "length": 10, "has_ps00134": True, "prosite_ids": "PS00134;"},
            {"sequence_id": 1, "accession": "A1", "entry_name": "NEG", "length": 10, "has_ps00134": False, "prosite_ids": ""},
            {"sequence_id": 2, "accession": "A2", "entry_name": "FNLIKE", "length": 10, "has_ps00134": True, "prosite_ids": "PS00134;"},
        ]
    )
    matches = pd.DataFrame([{"sequence_id": 0}, {"sequence_id": 0}])

    audit = sequence_audit(cohort, matches)

    assert audit.loc[audit["sequence_id"] == 0, "n_ps00134_scan_matches"].item() == 2
    assert audit.loc[audit["sequence_id"] == 2, "has_recoverable_ps00134_pattern_match"].item() is False
    assert audit.loc[audit["sequence_id"] == 2, "binary_label_matches_recoverable_pattern"].item() is False


def test_regex_constant_matches_documented_six_residue_pattern():
    assert pd.Series(["LSASHC", "MTAAGC", "AAAAAA"]).str.contains(PS00134_REGEX, regex=True).tolist() == [True, False, False]
