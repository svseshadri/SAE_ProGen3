from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from scripts.evaluate_reconstruction import (
    ProGen3BatchPreparer,
    ProGen3ForCausalLM,
    additive_sae_intervention,
    compute_mean_kl,
    get_layer_hidden_state,
    load_sae_checkpoint,
    sequence_nll,
    top1_agreement,
    forward_with_replaced_hidden_state,
)

DEFAULT_MODEL_NAME = "Profluent-Bio/progen3-112m"
DEFAULT_SAE_CKPT = "results/topk_sae_layer6_d4096_k32_run1/best.pt"
DEFAULT_DATASET = "data/processed/s1a70/s1a70_test.csv"
DEFAULT_ENRICHMENT = "results/global_feature_enrichment.csv"
DEFAULT_OUTPUT = "results/causal_feature_dose_response.csv"

FEATURE_CONCEPTS = {
    727: "has_ipr001314",
    3256: "has_ps00134",
    1644: "has_ps00135",
    2942: "has_both_catalytic_motifs",
    1: None,
}


def candidate_latents_from_enrichment(path: str | Path, feature_ids: list[int] | None = None) -> list[int]:
    df = pd.read_csv(path)
    if feature_ids is not None:
        return feature_ids
    # Start from the strongest broad discriminators in the frozen enrichment table.
    candidates = [727, 3256, 1644, 2942, 20, 1]
    present = [int(fid) for fid in candidates if int(fid) in set(df["latent_id"].tolist())]
    return present


def feature_quantiles_for_dataset(
    model: ProGen3ForCausalLM,
    sae,
    batch_preparer: ProGen3BatchPreparer,
    dataset: pd.DataFrame,
    feature_id: int,
    device: str,
    sae_meta: dict,
    max_sequences: int | None = None,
) -> list[float]:
    values: list[float] = []
    seqs = dataset.head(max_sequences) if max_sequences is not None else dataset
    for _, row in seqs.iterrows():
        seq = str(row["sequence"])
        batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
            hidden = get_layer_hidden_state(outputs, 6)
            x = hidden.float().reshape(-1, hidden.shape[-1])
            if sae_meta["input_mean"] is not None and sae_meta["input_std"] is not None:
                x = (x - sae_meta["input_mean"].to(x.device)) / sae_meta["input_std"].to(x.device)
            z = sae(x)["z"]
            values.extend(float(v) for v in z[:, feature_id].detach().cpu().tolist())
    if not values:
        return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    q = torch.tensor(values, dtype=torch.float32).quantile(torch.tensor([0.0, 0.5, 0.9, 0.95, 0.99, 1.0], dtype=torch.float32)).tolist()
    return [float(v) for v in q]


def sequence_feature_active(
    model: ProGen3ForCausalLM,
    sae,
    batch_preparer: ProGen3BatchPreparer,
    seq: str,
    feature_id: int,
    device: str,
    sae_meta: dict,
) -> bool:
    batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
        hidden = get_layer_hidden_state(outputs, 6)
        x = hidden.float().reshape(-1, hidden.shape[-1])
        if sae_meta["input_mean"] is not None and sae_meta["input_std"] is not None:
            x = (x - sae_meta["input_mean"].to(x.device)) / sae_meta["input_std"].to(x.device)
        z = sae(x)["z"]
        mean_active = z[:, feature_id].mean().detach().cpu().item()
    return float(mean_active) > 0.0


def concept_flag_for_feature(feature_id: int, row: pd.Series) -> bool:
    concept_col = FEATURE_CONCEPTS.get(feature_id)
    if concept_col is None:
        return bool(row.get("class_label", 0) == 1)
    return bool(row.get(concept_col, 0) == 1)


def feature_concept_name(feature_id: int) -> str:
    concept_col = FEATURE_CONCEPTS.get(feature_id)
    if concept_col is None:
        return "matched_control"
    return concept_col


def sequence_identifier(dataset: pd.DataFrame, row_index: int, row: pd.Series) -> int:
    if "sequence_id" in dataset.columns:
        value = row.get("sequence_id", row_index)
        return int(value)
    return int(row_index)


def compute_localized_logit_shift(
    base_logits: torch.Tensor,
    patched_logits: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    top_k: int = 16,
) -> tuple[float, float, float]:
    if base_logits.ndim != 2 and base_logits.ndim != 3:
        raise ValueError(f"Unexpected logits shape: {base_logits.shape}")
    if base_logits.ndim == 3:
        base_logits = base_logits[0]
        patched_logits = patched_logits[0]
    if labels.ndim == 2:
        labels = labels[0]

    base_logp = torch.log_softmax(base_logits.float(), dim=-1)
    patched_logp = torch.log_softmax(patched_logits.float(), dim=-1)
    valid_positions = []
    for pos, token in enumerate(labels):
        if token == pad_token_id:
            continue
        valid_positions.append(pos)
    if not valid_positions:
        return 0.0, 0.0, 0.0

    deltas = []
    for pos in valid_positions:
        token = int(labels[pos].item())
        if token >= base_logp.shape[-1]:
            continue
        delta = float((patched_logp[pos, token] - base_logp[pos, token]).detach().cpu().item())
        deltas.append(delta)
    if not deltas:
        return 0.0, 0.0, 0.0

    sorted_deltas = sorted(abs(float(d)) for d in deltas)
    k = min(top_k, len(sorted_deltas))
    motif_delta = float(sum(sorted_deltas[-k:]) / max(k, 1)) if k else 0.0
    nonmotif_delta = float(sum(sorted_deltas[:-k]) / max(len(sorted_deltas) - k, 1)) if len(sorted_deltas) > k else 0.0
    specificity = motif_delta - nonmotif_delta
    return motif_delta, nonmotif_delta, specificity


def evaluate_additive_feature_dose_response(
    model: ProGen3ForCausalLM,
    sae,
    sae_meta: dict,
    batch_preparer: ProGen3BatchPreparer,
    dataset: pd.DataFrame,
    feature_id: int,
    device: str,
    target_levels: list[float],
    max_sequences: int | None = None,
) -> list[dict]:
    rows: list[dict] = []
    subset = dataset.head(max_sequences) if max_sequences is not None else dataset
    for row_index, row in subset.iterrows():
        seq = str(row["sequence"])
        seq_id = sequence_identifier(subset, row_index, row)
        batch = batch_preparer.get_batch_kwargs([seq], device=device, reverse=False)
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
            base_hidden = get_layer_hidden_state(outputs, 6)
            base_logits = outputs.logits
            base_nll = sequence_nll(base_logits[0], batch["labels"][0], model.config.pad_token_id)

            for target in target_levels:
                patched_hidden = additive_sae_intervention(
                    hidden_state=base_hidden,
                    sae=sae,
                    feature_id=feature_id,
                    target_activation=target,
                    input_mean=sae_meta["input_mean"],
                    input_std=sae_meta["input_std"],
                )
                patched_logits = forward_with_replaced_hidden_state(model, batch, 6, patched_hidden)
                patched_nll = sequence_nll(patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                motif_delta, nonmotif_delta, specificity = compute_localized_logit_shift(
                    base_logits[0],
                    patched_logits[0],
                    batch["labels"][0],
                    model.config.pad_token_id,
                    top_k=16,
                )
                rows.append(
                    {
                        "sequence_id": int(seq_id),
                        "class_label": int(row.get("class_label", 0)),
                        "length": int(row.get("length", len(seq))),
                        "feature_id": int(feature_id),
                        "matched_concept": feature_concept_name(feature_id),
                        "concept_positive": bool(concept_flag_for_feature(feature_id, row)),
                        "baseline_feature_active": bool(sequence_feature_active(model, sae, batch_preparer, seq, feature_id, device, sae_meta)),
                        "target_level": str(target),
                        "dose": float(target),
                        "target_value": float(target),
                        "base_nll": float(base_nll),
                        "patched_nll": float(patched_nll),
                        "delta_nll": float(patched_nll - base_nll),
                        "kl": float(compute_mean_kl(base_logits[0], patched_logits[0])),
                        "motif_delta_logprob": float(motif_delta),
                        "nonmotif_delta_logprob": float(nonmotif_delta),
                        "motif_specificity_score": float(specificity),
                        "top1_agreement": float(top1_agreement(base_logits[0], patched_logits[0])),
                    }
                )
    return rows


def build_feature_specific_cohorts(
    model: ProGen3ForCausalLM,
    sae,
    sae_meta: dict,
    batch_preparer: ProGen3BatchPreparer,
    dataset: pd.DataFrame,
    feature_id: int,
    device: str,
    per_cell_target: int = 16,
) -> list[dict]:
    cells: dict[tuple[bool, bool], list[int]] = {key: [] for key in [(True, True), (True, False), (False, True), (False, False)]}
    row_lookup: dict[int, pd.Series] = {}
    for idx, row in dataset.iterrows():
        seq_id = sequence_identifier(dataset, idx, row)
        row_lookup[seq_id] = row
        seq = str(row["sequence"])
        concept_positive = bool(concept_flag_for_feature(feature_id, row))
        baseline_active = bool(sequence_feature_active(model, sae, batch_preparer, seq, feature_id, device, sae_meta))
        cells[(concept_positive, baseline_active)].append(int(seq_id))

    rows: list[dict] = []
    for (concept_positive, baseline_active), seq_ids in cells.items():
        limited = seq_ids[:per_cell_target]
        support_n = len(limited)
        for sequence_id in limited:
            row = row_lookup.get(int(sequence_id))
            if row is None:
                continue
            rows.append(
                {
                    "feature_id": int(feature_id),
                    "matched_concept": feature_concept_name(feature_id),
                    "concept_positive": bool(concept_positive),
                    "baseline_feature_active": bool(baseline_active),
                    "support_n": int(support_n),
                    "sequence_id": int(sequence_id),
                    "sequence": str(row["sequence"]),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run additive causal steering dose response for a small panel of enriched SAE latents.")
    parser.add_argument("--model-name", type=str, default=DEFAULT_MODEL_NAME)
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET)
    parser.add_argument("--sae-checkpoint", type=str, default=DEFAULT_SAE_CKPT)
    parser.add_argument("--enrichment", type=str, default=DEFAULT_ENRICHMENT)
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT)
    parser.add_argument("--feature-ids", type=str, default="727,3256,1644,2942,1", help="Comma-separated latent IDs to test.")
    parser.add_argument("--layer-index", type=int, default=6)
    parser.add_argument("--max-sequences", type=int, default=32, help="Set to a small subset for rapid validation.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = ProGen3ForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16 if args.device.startswith("cuda") else torch.float32).to(args.device)
    model.eval()
    sae, sae_meta = load_sae_checkpoint(args.sae_checkpoint, args.device)
    batch_preparer = ProGen3BatchPreparer()
    df = pd.read_csv(args.dataset)
    if args.max_sequences is not None:
        df = df.head(args.max_sequences)

    feature_ids = [int(v.strip()) for v in args.feature_ids.split(",") if v.strip()]
    all_rows: list[dict] = []
    for feature_id in feature_ids:
        quantiles = feature_quantiles_for_dataset(model, sae, batch_preparer, df, feature_id, args.device, sae_meta, max_sequences=args.max_sequences)
        targets = [0.0, -max(quantiles[1], 1e-6), -max(quantiles[2], 1e-6), -max(quantiles[3], 1e-6), quantiles[1], quantiles[2], quantiles[3], quantiles[4], 2.0 * max(quantiles[4], 1e-6)]
        unique_targets = []
        seen = set()
        for t in targets:
            key = round(float(t), 10)
            if key not in seen:
                unique_targets.append(float(t))
                seen.add(key)

        rows = evaluate_additive_feature_dose_response(
            model=model,
            sae=sae,
            sae_meta=sae_meta,
            batch_preparer=batch_preparer,
            dataset=df,
            feature_id=feature_id,
            device=args.device,
            target_levels=unique_targets,
            max_sequences=args.max_sequences,
        )
        for row in rows:
            row["quantile_0"] = quantiles[0]
            row["quantile_50"] = quantiles[1]
            row["quantile_90"] = quantiles[2]
            row["quantile_95"] = quantiles[3]
            row["quantile_99"] = quantiles[4]
            row["quantile_100"] = quantiles[5]
            row["target_level_name"] = row["target_level"]
        all_rows.extend(rows)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_rows).to_csv(out, index=False)

    summary = []
    for feature_id in feature_ids:
        feat_rows = [r for r in all_rows if r["feature_id"] == feature_id]
        if not feat_rows:
            continue
        grouped = pd.DataFrame(feat_rows).groupby("dose")["delta_nll"].mean().reset_index()
        summary.append({
            "feature_id": feature_id,
            "matched_concept": feature_concept_name(feature_id),
            "levels": grouped.to_dict(orient="records"),
        })

    stratified: list[dict] = []
    for feature_id in feature_ids:
        cohort_rows = build_feature_specific_cohorts(model, sae, sae_meta, batch_preparer, df, feature_id, args.device, per_cell_target=16)
        if not cohort_rows:
            continue
        cohort_selection = []
        for entry in cohort_rows:
            seq = entry["sequence"]
            batch = batch_preparer.get_batch_kwargs([seq], device=args.device, reverse=False)
            batch = {k: v.to(args.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch, return_dict=True, output_hidden_states=True, use_cache=False)
                base_hidden = get_layer_hidden_state(outputs, 6)
                base_logits = outputs.logits
                base_nll = sequence_nll(base_logits[0], batch["labels"][0], model.config.pad_token_id)
                for dose in [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0]:
                    patched_hidden = additive_sae_intervention(
                        hidden_state=base_hidden,
                        sae=sae,
                        feature_id=feature_id,
                        target_activation=dose,
                        input_mean=sae_meta["input_mean"],
                        input_std=sae_meta["input_std"],
                    )
                    patched_logits = forward_with_replaced_hidden_state(model, batch, 6, patched_hidden)
                    patched_nll = sequence_nll(patched_logits[0], batch["labels"][0], model.config.pad_token_id)
                    motif_delta, nonmotif_delta, specificity = compute_localized_logit_shift(
                        base_logits[0],
                        patched_logits[0],
                        batch["labels"][0],
                        model.config.pad_token_id,
                        top_k=16,
                    )
                    cohort_selection.append(
                        {
                            "feature_id": int(feature_id),
                            "matched_concept": feature_concept_name(feature_id),
                            "concept_positive": bool(entry["concept_positive"]),
                            "baseline_feature_active": bool(entry["baseline_feature_active"]),
                            "support_n": int(entry["support_n"]),
                            "sequence_id": int(entry["sequence_id"]),
                            "dose": float(dose),
                            "delta_nll": float(patched_nll - base_nll),
                            "kl": float(compute_mean_kl(base_logits[0], patched_logits[0])),
                            "motif_delta_logprob": float(motif_delta),
                            "nonmotif_delta_logprob": float(nonmotif_delta),
                            "motif_specificity_score": float(specificity),
                        }
                    )
        if cohort_selection:
            cohort_df = pd.DataFrame(cohort_selection)
            grouped = cohort_df.groupby(["concept_positive", "baseline_feature_active", "dose"], as_index=False)[["delta_nll", "motif_specificity_score"]].mean()
            stratified.append({
                "feature_id": int(feature_id),
                "matched_concept": feature_concept_name(feature_id),
                "grouped": grouped.to_dict(orient="records"),
            })

    print(json.dumps({"dose_response": summary, "stratified": stratified}, indent=2))


if __name__ == "__main__":
    main()
