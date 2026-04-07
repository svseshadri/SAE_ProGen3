import math
import time
import requests
import pandas as pd
from io import StringIO
from typing import Optional

UNIPROT_SEARCH = "https://rest.uniprot.org/uniprotkb/search"

FIELDS = [
    "accession",
    "id",
    "reviewed",
    "organism_name",
    "organism_id",
    "length",
    "sequence",
    "xref_interpro",
    "xref_prosite",
]

# -------------------------
# Config
# -------------------------

POSITIVE_QUERY = 'xref:InterPro-IPR001314 AND length:[180 TO 320]'
BACKGROUND_QUERY = (
    'reviewed:true AND length:[180 TO 320] '
    'NOT xref:InterPro-IPR001314 '
    'NOT xref:PROSITE-PS00134 '
    'NOT xref:PROSITE-PS00135 '
    'NOT xref:Pfam-PF00089'
)

POSITIVE_MAX_ROWS = 20000
BACKGROUND_MAX_ROWS = 50000

PAGE_SIZE = 500
REQUEST_TIMEOUT = 60
MAX_RETRIES = 5
RETRY_SLEEP_BASE = 2.0

BACKGROUND_RATIO = 5.0
LENGTH_BIN_SIZE = 10
RANDOM_STATE = 42

TMP_POS_FILE = "pos_tmp.csv"
TMP_BG_FILE = "bg_tmp.csv"

FINAL_DATASET_FILE = "s1a_vs_background_dataset.csv"
FINAL_POS_FILE = "s1a_positives.csv"
FINAL_BG_FILE = "background_length_matched.csv"


# -------------------------
# Helpers
# -------------------------

def safe_get(url: str, params: Optional[dict] = None, timeout: int = 60) -> requests.Response:
    """
    GET with retries and exponential backoff.
    """
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_exc = e
            sleep_s = RETRY_SLEEP_BASE * attempt
            print(f"[WARN] Request failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            print(f"[INFO] Sleeping for {sleep_s:.1f}s before retrying...", flush=True)
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed request after {MAX_RETRIES} retries: {last_exc}")


def fetch_uniprot_tsv(query: str, size: int = 500, max_rows: Optional[int] = None, label: str = "query") -> pd.DataFrame:
    """
    Download UniProtKB search results as TSV with pagination, progress printing, retries, and optional max_rows cap.
    """
    params = {
        "query": query,
        "format": "tsv",
        "fields": ",".join(FIELDS),
        "size": size,
    }

    frames = []
    url = UNIPROT_SEARCH
    total_rows = 0
    page = 0

    print(f"[INFO] Starting download for {label}")
    print(f"[INFO] Query: {query}", flush=True)

    while url:
        page += 1

        resp = safe_get(
            url,
            params=params if url == UNIPROT_SEARCH else None,
            timeout=REQUEST_TIMEOUT,
        )

        df = pd.read_csv(StringIO(resp.text), sep="\t")
        frames.append(df)
        total_rows += len(df)

        print(f"[INFO] {label}: page {page}, got {len(df)} rows, total so far = {total_rows}", flush=True)

        if max_rows is not None and total_rows >= max_rows:
            print(f"[INFO] {label}: reached max_rows={max_rows}, stopping pagination.", flush=True)
            break

        next_url = None
        link_header = resp.headers.get("Link")
        if link_header and 'rel="next"' in link_header:
            next_url = link_header.split(";")[0].strip("<> ")

        url = next_url
        params = None
        time.sleep(0.15)  # gentle throttling

    if not frames:
        print(f"[WARN] {label}: no rows returned.")
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Cap to max_rows exactly after concat, if needed
    if max_rows is not None and len(out) > max_rows:
        out = out.iloc[:max_rows].copy()

    # UniProt TSV columns come back as Entry / Sequence / etc.
    dedup_subset = [c for c in ["Entry", "Sequence"] if c in out.columns]
    if dedup_subset:
        before = len(out)
        out = out.drop_duplicates(subset=dedup_subset)
        after = len(out)
        print(f"[INFO] {label}: deduplicated from {before} -> {after} rows", flush=True)

    return out


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {
        "Entry": "accession",
        "Entry Name": "entry_name",
        "Reviewed": "reviewed",
        "Organism": "organism",
        "Organism (ID)": "taxon_id",
        "Length": "length",
        "Sequence": "sequence",
        "InterPro": "interpro_ids",
        "PROSITE": "prosite_ids",
    }
    df = df.rename(columns=rename).copy()

    for col in ["interpro_ids", "prosite_ids"]:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("")

    df["has_ipr001314"] = df["interpro_ids"].str.contains("IPR001314", regex=False)
    df["has_ps00134"] = df["prosite_ids"].str.contains("PS00134", regex=False)
    df["has_ps00135"] = df["prosite_ids"].str.contains("PS00135", regex=False)
    df["has_both_catalytic_motifs"] = df["has_ps00134"] & df["has_ps00135"]

    if "reviewed" in df.columns:
        df["reviewed"] = df["reviewed"].astype(str).str.lower().eq("reviewed")
    else:
        df["reviewed"] = False

    if "length" in df.columns:
        df["length"] = pd.to_numeric(df["length"], errors="coerce")

    return df


def filter_valid_sequences(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only sequences with standard 20 amino acids and valid lengths.
    """
    valid_aas = set("ACDEFGHIKLMNPQRSTVWY")

    def is_valid(seq: str) -> bool:
        if not isinstance(seq, str) or len(seq) == 0:
            return False
        return set(seq).issubset(valid_aas)

    before = len(df)
    df = df[df["sequence"].apply(is_valid)].copy()
    after = len(df)
    print(f"[INFO] Sequence validity filter: {before} -> {after}", flush=True)
    return df


def length_match_background(
    positives: pd.DataFrame,
    background_pool: pd.DataFrame,
    bin_size: int = 10,
    ratio: float = 5.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample background sequences so their length histogram approximately matches positives.
    ratio=5.0 means sample ~5x as many backgrounds as positives overall.
    """
    pos = positives.copy()
    bg = background_pool.copy()

    pos["len_bin"] = (pos["length"] // bin_size) * bin_size
    bg["len_bin"] = (bg["length"] // bin_size) * bin_size

    samples = []
    pos_counts = pos["len_bin"].value_counts().sort_index()

    print("[INFO] Starting length-matched background sampling...", flush=True)

    for len_bin, n_pos in pos_counts.items():
        n_bg = int(math.ceil(n_pos * ratio))
        candidates = bg[bg["len_bin"] == len_bin]

        if len(candidates) == 0:
            print(f"[WARN] No background candidates for len_bin={len_bin}")
            continue

        n_take = min(n_bg, len(candidates))
        sampled = candidates.sample(n=n_take, random_state=random_state)
        samples.append(sampled)

        print(
            f"[INFO] len_bin={len_bin}: positives={n_pos}, candidates={len(candidates)}, sampled={n_take}",
            flush=True,
        )

    if not samples:
        return pd.DataFrame(columns=bg.columns)

    out = pd.concat(samples, ignore_index=True)
    out = out.drop_duplicates(subset=["sequence"]).copy()

    print(f"[INFO] Final length-matched background size: {len(out)}", flush=True)
    return out


def print_sanity_checks(df: pd.DataFrame, name: str) -> None:
    print(f"\n[INFO] ===== Sanity check: {name} =====", flush=True)
    print(df["class_name"].value_counts(dropna=False), flush=True)

    if "class_label" in df.columns:
        motif_summary = (
            df.groupby("class_label")[[
                "has_ipr001314",
                "has_ps00134",
                "has_ps00135",
                "has_both_catalytic_motifs",
                "length",
            ]]
            .agg({
                "has_ipr001314": "mean",
                "has_ps00134": "mean",
                "has_ps00135": "mean",
                "has_both_catalytic_motifs": "mean",
                "length": ["mean", "median", "min", "max"],
            })
        )
        print(motif_summary, flush=True)


# -------------------------
# Main
# -------------------------

def main():
    start_time = time.time()

    # 1. Download positives
    pos_raw = fetch_uniprot_tsv(
        POSITIVE_QUERY,
        size=PAGE_SIZE,
        max_rows=POSITIVE_MAX_ROWS,
        label="positives",
    )

    # 2. Download background pool
    bg_raw = fetch_uniprot_tsv(
        BACKGROUND_QUERY,
        size=PAGE_SIZE,
        max_rows=BACKGROUND_MAX_ROWS,
        label="background",
    )

    # 3. Normalize columns
    pos = normalize_columns(pos_raw)
    bg = normalize_columns(bg_raw)

    # 4. Filter invalid sequences
    pos = filter_valid_sequences(pos)
    bg = filter_valid_sequences(bg)

    # 5. Exact-sequence dedup
    pos_before = len(pos)
    bg_before = len(bg)

    pos = pos.drop_duplicates(subset=["sequence"]).copy()
    bg = bg.drop_duplicates(subset=["sequence"]).copy()

    print(f"[INFO] Pos dedup by sequence: {pos_before} -> {len(pos)}", flush=True)
    print(f"[INFO] Bg dedup by sequence:  {bg_before} -> {len(bg)}", flush=True)

    # 6. Annotate classes
    pos["class_label"] = 1
    pos["class_name"] = "S1A_trypsin_chymotrypsin"
    pos["background_designation"] = pd.NA

    bg["class_label"] = 0
    bg["class_name"] = "background"
    bg["background_designation"] = "length_matched_non_S1A"

    # 7. Save temp files before sampling, just in case
    pos.to_csv(TMP_POS_FILE, index=False)
    bg.to_csv(TMP_BG_FILE, index=False)
    print(f"[INFO] Wrote temp files: {TMP_POS_FILE}, {TMP_BG_FILE}", flush=True)

    # 8. Length-match backgrounds
    bg_matched = length_match_background(
        pos,
        bg,
        bin_size=LENGTH_BIN_SIZE,
        ratio=BACKGROUND_RATIO,
        random_state=RANDOM_STATE,
    )

    # 9. Build final dataset
    dataset = pd.concat([pos, bg_matched], ignore_index=True)

    keep_cols = [
        "accession",
        "entry_name",
        "reviewed",
        "organism",
        "taxon_id",
        "sequence",
        "length",
        "class_label",
        "class_name",
        "background_designation",
        "has_ipr001314",
        "has_ps00134",
        "has_ps00135",
        "has_both_catalytic_motifs",
        "interpro_ids",
        "prosite_ids",
    ]

    for col in keep_cols:
        if col not in dataset.columns:
            dataset[col] = pd.NA
        if col not in pos.columns:
            pos[col] = pd.NA
        if col not in bg_matched.columns:
            bg_matched[col] = pd.NA

    dataset = dataset[keep_cols].copy()
    pos = pos[keep_cols].copy()
    bg_matched = bg_matched[keep_cols].copy()

    # 10. Write final files
    dataset.to_csv(FINAL_DATASET_FILE, index=False)
    pos.to_csv(FINAL_POS_FILE, index=False)
    bg_matched.to_csv(FINAL_BG_FILE, index=False)

    print(f"[INFO] Wrote final dataset: {FINAL_DATASET_FILE}", flush=True)
    print(f"[INFO] Wrote positives:     {FINAL_POS_FILE}", flush=True)
    print(f"[INFO] Wrote background:    {FINAL_BG_FILE}", flush=True)

    # 11. Sanity checks
    print_sanity_checks(dataset, "final dataset")

    elapsed = time.time() - start_time
    print(f"\n[INFO] Total elapsed time: {elapsed/60:.2f} minutes", flush=True)


if __name__ == "__main__":
    main()