from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

DEFAULT_DATASET = "data/processed/s1a70/s1a70_test.csv"
DEFAULT_CAUSAL = "results/causal_feature_dose_response.csv"
DEFAULT_DEFINITION_OUTPUT = "results/ps00134_external_definition.json"
DEFAULT_FASTA_OUTPUT = "results/provenance/ps00134_causal_cohort_sequences.fasta"
DEFAULT_RAW_SCAN_OUTPUT = "results/provenance/scanprosite_ps00134_raw.json"
DEFAULT_POSITION_OUTPUT = "results/ps00134_position_annotations.csv"
DEFAULT_SEQUENCE_AUDIT_OUTPUT = "results/ps00134_scanprosite_sequence_audit.csv"
DEFAULT_METADATA_OUTPUT = "results/ps00134_scanprosite_recovery_metadata.json"

PS00134_URL = "https://prosite.expasy.org/PS00134"
PS00134_TEXT_URL = "https://prosite.expasy.org/PS00134.txt"
PRU10078_URL = "https://prosite.expasy.org/rule/PRU10078"
PRU10078_TEXT_URL = "https://prosite.expasy.org/rule/PRU10078.txt"
SCANPROSITE_URL = "https://prosite.expasy.org/cgi-bin/prosite/scanprosite/PSScan.cgi"

PS00134_PATTERN = "[LIVM]-[ST]-A-[STAG]-H-C"
PS00134_REGEX = r"[LIVM][ST]A[STAG]HC"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def causal_sequence_ids(causal_path: str | Path = DEFAULT_CAUSAL) -> list[int]:
    causal = pd.read_csv(causal_path)
    required = {"sequence_id", "feature_id"}
    missing = sorted(required - set(causal.columns))
    if missing:
        raise ValueError(f"Missing required columns in {causal_path}: {missing}")
    return sorted(int(v) for v in causal.loc[causal["feature_id"] == 3256, "sequence_id"].drop_duplicates())


def load_exact_causal_cohort(dataset_path: str | Path = DEFAULT_DATASET, causal_path: str | Path = DEFAULT_CAUSAL) -> pd.DataFrame:
    ids = causal_sequence_ids(causal_path)
    dataset = pd.read_csv(dataset_path).reset_index().rename(columns={"index": "sequence_id"})
    cohort = dataset.loc[dataset["sequence_id"].isin(ids)].copy()
    if cohort["sequence_id"].nunique() != len(ids):
        found = set(int(v) for v in cohort["sequence_id"])
        missing = [v for v in ids if v not in found]
        raise ValueError(f"Could not recover causal sequence_id values from dataset: {missing[:10]}")
    cohort["has_ps00134"] = cohort["has_ps00134"].astype(bool)
    return cohort.sort_values("sequence_id").reset_index(drop=True)


def fasta_for_cohort(cohort: pd.DataFrame) -> str:
    records = []
    for row in cohort.itertuples(index=False):
        records.append(f">seq_{int(row.sequence_id)}")
        seq = str(row.sequence)
        records.extend(seq[i : i + 80] for i in range(0, len(seq), 80))
    return "\n".join(records) + "\n"


def write_text(path: str | Path, text: str) -> str:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    return sha256_file(out)


def fetch_text(url: str, timeout: int = 60) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def freeze_external_definition(output_path: str | Path = DEFAULT_DEFINITION_OUTPUT, retrieval_time: str | None = None) -> dict[str, Any]:
    retrieval_time = retrieval_time or datetime.now(timezone.utc).isoformat()
    ps00134_text = fetch_text(PS00134_TEXT_URL)
    prorule_text = fetch_text(PRU10078_TEXT_URL)
    definition = {
        "prosite_accession": "PS00134",
        "name": "TRYPSIN_HIS",
        "entry_type": "PATTERN",
        "description": "Serine proteases, trypsin family, histidine active site.",
        "pattern": PS00134_PATTERN,
        "pattern_regex_used_for_local_validation_only": PS00134_REGEX,
        "pattern_version": "1",
        "entry_created": "01-APR-1990",
        "entry_data_update": "01-APR-1990",
        "entry_info_update": "29-MAY-2024",
        "prosite_release_reported_on_entry_page": "UniProtKB/Swiss-Prot release 2026_02 numerical results",
        "associated_prorule": "PRU10078",
        "active_site_offset_within_match_1_based": 5,
        "prorule_feature_key": "ACT_SITE",
        "prorule_feature_note": "Charge relay system",
        "retrieval_date": retrieval_time,
        "source_urls": {
            "prosite_entry": PS00134_URL,
            "prosite_entry_raw_text": PS00134_TEXT_URL,
            "prorule": PRU10078_URL,
            "prorule_raw_text": PRU10078_TEXT_URL,
            "scanprosite_endpoint": SCANPROSITE_URL,
        },
        "downloaded_sources": {
            "PS00134.txt": {
                "url": PS00134_TEXT_URL,
                "sha256": sha256_text(ps00134_text),
                "bytes": len(ps00134_text.encode("utf-8")),
            },
            "PRU10078.txt": {
                "url": PRU10078_TEXT_URL,
                "sha256": sha256_text(prorule_text),
                "bytes": len(prorule_text.encode("utf-8")),
            },
        },
        "source_text_excerpt": {
            "PS00134.txt": ps00134_text,
            "PRU10078.txt": prorule_text,
        },
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(definition, indent=2, sort_keys=True))
    return definition


def scanprosite_ps00134(fasta_text: str, timeout: int = 120) -> dict[str, Any]:
    response = requests.post(
        SCANPROSITE_URL,
        data={"seq": fasta_text, "sig": "PS00134", "output": "json"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def normalize_scan_matches(raw: dict[str, Any], cohort: pd.DataFrame) -> pd.DataFrame:
    lookup = cohort.set_index("sequence_id").to_dict(orient="index")
    rows: list[dict[str, Any]] = []
    for match in raw.get("matchset", []):
        sequence_ac = str(match["sequence_ac"])
        if not sequence_ac.startswith("seq_"):
            raise ValueError(f"Unexpected ScanProsite sequence_ac: {sequence_ac}")
        sequence_id = int(sequence_ac.removeprefix("seq_"))
        if sequence_id not in lookup:
            raise ValueError(f"ScanProsite returned unknown sequence_id: {sequence_id}")
        seq = str(lookup[sequence_id]["sequence"])
        start = int(match["start"])
        stop = int(match["stop"])
        if start < 1 or stop > len(seq) or stop < start:
            raise ValueError(f"Invalid ScanProsite coordinates for seq_{sequence_id}: {start}-{stop}")
        matched = seq[start - 1 : stop]
        if len(matched) != 6:
            raise ValueError(f"PS00134 match should span 6 residues for seq_{sequence_id}: {matched}")
        if re.fullmatch(PS00134_REGEX, matched) is None:
            raise ValueError(f"ScanProsite PS00134 match does not match frozen pattern for seq_{sequence_id}: {matched}")
        rows.append(
            {
                "sequence_id": sequence_id,
                "sequence_length": int(lookup[sequence_id]["length"]),
                "has_ps00134": bool(lookup[sequence_id]["has_ps00134"]),
                "accession": lookup[sequence_id].get("accession", ""),
                "entry_name": lookup[sequence_id].get("entry_name", ""),
                "prosite_ids": lookup[sequence_id].get("prosite_ids", ""),
                "ps00134_pattern_match": True,
                "ps00134_scan_confidence": match.get("level_tag", ""),
                "scanprosite_sequence_ac": sequence_ac,
                "signature_ac": match.get("signature_ac", ""),
                "signature_id": match.get("signature_id", ""),
                "match_start_1_based": start,
                "match_end_1_based": stop,
                "matched_sequence": matched,
                "scanner_matched_sequence": match.get("matched_region", match.get("sequence", "")),
                "match_length": stop - start + 1,
                "annotation_source": "ScanProsite",
                "annotation_method": "official ScanProsite endpoint with sig=PS00134 on exact stored FASTA sequence",
            }
        )
    return pd.DataFrame(rows)


def expand_position_rows(matches: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    ordered = matches.sort_values(["sequence_id", "match_start_1_based"]).copy()
    ordered["match_index"] = ordered.groupby("sequence_id").cumcount() + 1
    for match in ordered.itertuples(index=False):
        for offset, residue in enumerate(str(match.matched_sequence), start=1):
            residue_position = int(match.match_start_1_based) + offset - 1
            rows.append(
                {
                    "sequence_id": int(match.sequence_id),
                    "sequence_length": int(match.sequence_length),
                    "has_ps00134": bool(match.has_ps00134),
                    "accession": str(match.accession),
                    "entry_name": str(match.entry_name),
                    "prosite_ids": str(match.prosite_ids),
                    "ps00134_pattern_match": True,
                    "ps00134_scan_confidence": str(match.ps00134_scan_confidence),
                    "match_index": int(match.match_index),
                    "match_start_1_based": int(match.match_start_1_based),
                    "match_end_1_based": int(match.match_end_1_based),
                    "matched_sequence": str(match.matched_sequence),
                    "residue_position_1_based": residue_position,
                    "residue_index_0_based": residue_position - 1,
                    "residue_identity": residue,
                    "ps00134_position": True,
                    "ps00134_pattern_offset_1_based": int(offset),
                    "ps00134_prorule_act_site": bool(offset == 5),
                    "annotation_source": str(match.annotation_source),
                    "annotation_method": str(match.annotation_method),
                }
            )
    return pd.DataFrame(rows)


def sequence_audit(cohort: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    if matches.empty:
        counts = pd.DataFrame(columns=["sequence_id", "n_ps00134_scan_matches"])
    else:
        counts = matches.groupby("sequence_id").size().rename("n_ps00134_scan_matches").reset_index()
    audit = cohort[["sequence_id", "accession", "entry_name", "length", "has_ps00134", "prosite_ids"]].merge(counts, on="sequence_id", how="left")
    audit["n_ps00134_scan_matches"] = audit["n_ps00134_scan_matches"].fillna(0).astype(int)
    audit["has_recoverable_ps00134_pattern_match"] = audit["n_ps00134_scan_matches"] > 0
    audit["binary_label_matches_recoverable_pattern"] = audit["has_ps00134"].astype(bool) == audit["has_recoverable_ps00134_pattern_match"]
    return audit


def run_recovery(
    dataset_path: str | Path = DEFAULT_DATASET,
    causal_path: str | Path = DEFAULT_CAUSAL,
    definition_output: str | Path = DEFAULT_DEFINITION_OUTPUT,
    fasta_output: str | Path = DEFAULT_FASTA_OUTPUT,
    raw_scan_output: str | Path = DEFAULT_RAW_SCAN_OUTPUT,
    position_output: str | Path = DEFAULT_POSITION_OUTPUT,
    sequence_audit_output: str | Path = DEFAULT_SEQUENCE_AUDIT_OUTPUT,
    metadata_output: str | Path = DEFAULT_METADATA_OUTPUT,
) -> dict[str, Any]:
    retrieval_time = datetime.now(timezone.utc).isoformat()
    cohort = load_exact_causal_cohort(dataset_path, causal_path)
    definition = freeze_external_definition(definition_output, retrieval_time=retrieval_time)
    fasta_text = fasta_for_cohort(cohort)
    fasta_sha256 = write_text(fasta_output, fasta_text)
    raw = scanprosite_ps00134(fasta_text)
    raw_path = Path(raw_scan_output)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(json.dumps(raw, indent=2, sort_keys=True))
    raw_sha256 = sha256_file(raw_path)
    matches = normalize_scan_matches(raw, cohort)
    positions = expand_position_rows(matches)
    audit = sequence_audit(cohort, matches)
    Path(position_output).parent.mkdir(parents=True, exist_ok=True)
    positions.to_csv(position_output, index=False)
    audit.to_csv(sequence_audit_output, index=False)
    mismatches = int((~audit["binary_label_matches_recoverable_pattern"]).sum())
    metadata = {
        "script": "analysis/ps00134_scanprosite_recovery.py",
        "timestamp_utc": retrieval_time,
        "git_commit": git_commit(),
        "dataset_path": str(dataset_path),
        "causal_path": str(causal_path),
        "definition_output": str(definition_output),
        "fasta_output": str(fasta_output),
        "fasta_sha256": fasta_sha256,
        "raw_scan_output": str(raw_scan_output),
        "raw_scan_sha256": raw_sha256,
        "position_output": str(position_output),
        "sequence_audit_output": str(sequence_audit_output),
        "scanprosite_url": SCANPROSITE_URL,
        "scan_signature": "PS00134",
        "n_causal_sequences": int(cohort["sequence_id"].nunique()),
        "n_has_ps00134_true": int(cohort["has_ps00134"].sum()),
        "n_has_ps00134_false": int((~cohort["has_ps00134"]).sum()),
        "scanprosite_n_seq": int(raw.get("n_seq", -1)),
        "scanprosite_n_match": int(raw.get("n_match", len(raw.get("matchset", [])))),
        "n_sequences_with_recoverable_pattern_match": int(audit["has_recoverable_ps00134_pattern_match"].sum()),
        "n_position_rows": int(len(positions)),
        "n_binary_label_vs_recoverable_pattern_mismatches": mismatches,
        "recoverable_pattern_reproduces_has_ps00134": bool(mismatches == 0),
        "has_ps00134_definition": "stored sequence-level UniProt PROSITE cross-reference contains PS00134",
        "has_recoverable_ps00134_pattern_match_definition": "official ScanProsite PS00134 hit on exact stored project sequence",
        "caveat": "These are distinct variables; the stored UniProt cross-reference may include historical or curated statuses that are not equivalent to a current literal pattern hit.",
        "external_definition": {
            key: definition[key]
            for key in [
                "prosite_accession",
                "name",
                "entry_type",
                "pattern",
                "pattern_version",
                "associated_prorule",
                "active_site_offset_within_match_1_based",
                "prorule_feature_key",
                "prorule_feature_note",
                "retrieval_date",
                "source_urls",
                "downloaded_sources",
            ]
        },
    }
    Path(metadata_output).parent.mkdir(parents=True, exist_ok=True)
    Path(metadata_output).write_text(json.dumps(metadata, indent=2, sort_keys=True))
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover PS00134 token-level provenance for the exact 64-sequence causal cohort via ScanProsite.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--causal", default=DEFAULT_CAUSAL)
    parser.add_argument("--definition-output", default=DEFAULT_DEFINITION_OUTPUT)
    parser.add_argument("--fasta-output", default=DEFAULT_FASTA_OUTPUT)
    parser.add_argument("--raw-scan-output", default=DEFAULT_RAW_SCAN_OUTPUT)
    parser.add_argument("--position-output", default=DEFAULT_POSITION_OUTPUT)
    parser.add_argument("--sequence-audit-output", default=DEFAULT_SEQUENCE_AUDIT_OUTPUT)
    parser.add_argument("--metadata-output", default=DEFAULT_METADATA_OUTPUT)
    args = parser.parse_args()

    metadata = run_recovery(
        dataset_path=args.dataset,
        causal_path=args.causal,
        definition_output=args.definition_output,
        fasta_output=args.fasta_output,
        raw_scan_output=args.raw_scan_output,
        position_output=args.position_output,
        sequence_audit_output=args.sequence_audit_output,
        metadata_output=args.metadata_output,
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
