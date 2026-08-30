from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_DATASETS = (
    "data/raw/s1a_vs_background_dataset.csv",
    "data/processed/s1a_positives.csv",
    "data/processed/background_length_matched.csv",
    "data/processed/s1a70/s1a70_clustered_dataset.csv",
    "data/processed/s1a70/s1a70_test.csv",
)
DEFAULT_SUMMARY_OUTPUT = "results/ps00134_annotation_provenance_audit.csv"
DEFAULT_METADATA_OUTPUT = "results/ps00134_annotation_provenance_audit.json"


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def coerce_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
    out = normalized.map(mapping)
    if out.isna().any():
        bad = series[out.isna()].drop_duplicates().head(10).tolist()
        raise ValueError(f"Could not coerce boolean values: {bad}")
    return out.astype(bool)


def ps00134_from_prosite_ids(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.contains("PS00134", regex=False)


def coordinate_like_columns(columns: list[str]) -> list[str]:
    terms = ("start", "end", "span", "coord", "match", "position", "residue")
    return [col for col in columns if any(term in col.lower() for term in terms)]


def audit_dataset(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    df = pd.read_csv(path)
    required = {"has_ps00134", "prosite_ids"}
    missing = sorted(required - set(df.columns))
    row: dict[str, Any] = {
        "dataset_path": str(path),
        "n_rows": int(len(df)),
        "columns": ";".join(df.columns),
        "required_columns_present": not missing,
        "missing_required_columns": ";".join(missing),
        "coordinate_like_columns": ";".join(coordinate_like_columns(list(df.columns))),
        "has_position_level_ps00134_provenance": False,
        "binary_reproduction_source": "prosite_ids contains PS00134",
    }
    if missing:
        row.update(
            {
                "n_has_ps00134_true": pd.NA,
                "n_regenerated_true": pd.NA,
                "binary_labels_reproduced": False,
                "n_binary_mismatches": pd.NA,
            }
        )
        return row
    observed = coerce_bool(df["has_ps00134"])
    regenerated = ps00134_from_prosite_ids(df["prosite_ids"])
    mismatches = observed != regenerated
    ps_coord_cols = [
        col
        for col in coordinate_like_columns(list(df.columns))
        if "ps00134" in col.lower() or "prosite" in col.lower() or "motif" in col.lower()
    ]
    row.update(
        {
            "n_has_ps00134_true": int(observed.sum()),
            "n_regenerated_true": int(regenerated.sum()),
            "binary_labels_reproduced": bool(not mismatches.any()),
            "n_binary_mismatches": int(mismatches.sum()),
            "example_mismatch_indices": ";".join(str(int(v)) for v in df.index[mismatches].tolist()[:10]),
            "has_position_level_ps00134_provenance": bool(ps_coord_cols),
            "ps00134_coordinate_like_columns": ";".join(ps_coord_cols),
        }
    )
    return row


def run_audit(dataset_paths: tuple[str, ...] = DEFAULT_DATASETS) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary = pd.DataFrame([audit_dataset(path) for path in dataset_paths])
    metadata = {
        "script": "analysis/ps00134_annotation_provenance.py",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "label_origin": "data/utils/build_s1a_dataset.py normalizes UniProt TSV field PROSITE to prosite_ids and sets has_ps00134 = prosite_ids.str.contains('PS00134', regex=False)",
        "binary_reproduction_passes_all_checked_datasets": bool(summary["binary_labels_reproduced"].fillna(False).all()),
        "position_level_provenance_available": bool(summary["has_position_level_ps00134_provenance"].fillna(False).any()),
        "gate_a_passes": bool(
            summary["binary_labels_reproduced"].fillna(False).all()
            and summary["has_position_level_ps00134_provenance"].fillna(False).any()
        ),
        "gate_a_reason": "Binary labels reproduce from stored PROSITE cross-reference IDs, but no motif start/end, match span, matched residues, regex scan output, or equivalent coordinate-level PS00134 provenance exists in the audited project artifacts.",
    }
    return summary, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit PS00134 annotation provenance and Gate A positional-mask recoverability.")
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT)
    parser.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    args = parser.parse_args()

    summary, metadata = run_audit(tuple(args.datasets))
    summary_path = Path(args.summary_output)
    metadata_path = Path(args.metadata_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True))
    print(json.dumps({"summary_output": str(summary_path), "metadata_output": str(metadata_path), **metadata}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
