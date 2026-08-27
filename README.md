# ProGen-3 Sparse Autoencoder — Model, Findings, and Feature Atlas

This repository contains the analyses, model outputs, and a presentation dashboard for a sparse autoencoder trained on protein sequences (ProGen-3). This README focuses on the model, experimental workflows, key findings, and the artifact outputs you can inspect to reproduce or evaluate results.

## Model summary

- Architecture: ProGen-3 sparse autoencoder variant with a Top‑K sparsity mechanism.
- Target representation: layer-6 dictionary (run-specific naming indicates dictionary size and Top‑K, e.g. `d4096_k32`).
- Training objective: learn a sparse dictionary so that individual latent dimensions encode compact biological features (motifs, enzymatic signatures) rather than broadly distributed signals.

## Datasets & preprocessing

- Positive and background FASTA sources, clusters, and rep sequences are stored under `data/fasta` and `data/mmseqs`.
- Preprocessed CSV splits, memmap embeddings, and QC outputs are under `data/processed`, `data/embeddings_memmap`, and `data/plots/qc_plots`.
- Scripts used for preprocessing are in `data/utils` (notable: `convert_pt_to_memmap.py`, `build_s1a_dataset.py`, `prepare_fastas.py`).

## Core analyses

1. Feature discovery & ranking

   - Latent dimensions were scored using reconstruction impact, sparsity measures, and biological enrichment signals. Per-run summaries appear under `results/*/analysis/latent_summary.json`.

2. Biological enrichment

   - Global enrichment tables and per-feature summaries are under `results/global_feature_enrichment.csv` and in per-run analysis folders.
   - Computed metrics include AUROC, point-biserial correlations, Fisher odds ratios, and q-values for multiple-testing control.

3. Causal dose-response interventions

   - For top candidates, controlled increases of latent activations were applied during generation to produce dose-response curves showing how likelihoods and motif frequencies change with activation magnitude.
   - Results are available in `results/causal_feature_dose_response.csv` and summarized visually in the `dashboard/` app.

4. Reconstruction and ablation studies

   - Ablations measure how removing or zeroing specific latents affects reconstruction error and downstream metrics. See `results/reconstruction_evaluation.summary.json`.

## Key findings (high level)

- Multiple layer-6 latents consistently enrich for proteolytic/enzymatic motifs (examples: trypsin/chymotrypsin-like patterns) across independent metrics.
- Dose-response experiments provide initial causal evidence that manipulating certain latents shifts generation outcomes in interpretable directions (likelihood and motif frequency), supporting the hypothesis that those latents control specific generative properties.

## Primary artifacts and where to find them

- `results/global_feature_enrichment.csv` — aggregated enrichment for candidate latents.
- `results/causal_feature_dose_response.csv` — per-feature dose-response measurement table.
- `results/reconstruction_evaluation.summary.json` — reconstruction and ablation metrics.
- `results/topk_sae_layer6_d4096_k32_run1/` — per-run artifacts, checkpoints, history (`history.json`), and analysis outputs.

## Interpretation guidance & caveats

- Enrichment is correlational; causal claims require careful controls. Our dose-response tests extend claims toward causality but do not fully rule out confounds.
- Validate top candidates by inspecting representative sequences, sequence logos, and by testing on held-out or orthogonal datasets.
- Review q-values, sample sizes, and effect sizes in per-feature outputs before making strong biological assertions.

## Dashboard (presentation layer)

The `dashboard/` folder contains a Next.js app used to present and explore results. It is a visualization and communication tool — the raw CSV/JSON artifacts in `results/` are the authoritative data sources for analysis and publication.

Notable implementation details:

- The dashboard uses a client-only `3dmol` viewer for the protein hero; the viewer is dynamically imported and configured with a transparent WebGL canvas (`backgroundAlpha: 0`) so the page background shows through.
- Main charts are inline SVGs tuned for editorial spacing; the explained-variance chart uses padded plotting margins and axis labels to avoid overlap between data and labels.

## Quick developer note (optional)

If you want to run the dashboard locally for inspection:

```bash
cd dashboard
npm install
npm run dev
```

This note is intentionally minimal — the README prioritizes the model, analyses, and artifacts.

## Suggested next steps for research

- Add per-feature sequence logos and motif alignments for human verification.
- Extend causal baselines (random-latent, counterfactuals) and include negative controls.
- Validate candidates on independent datasets or through orthogonal experimental assays where possible.

## Authors & contact

Sri Seshadri / Romero Lab
