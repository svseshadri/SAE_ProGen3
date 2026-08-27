# ProGen-3 Sparse Autoencoder — Feature Atlas & Dashboard

This repository contains analysis, results, and a Next.js dashboard for the ProGen-3 sparse autoencoder project. The dashboard surfaces layer-6 sparse latent features discovered by the model, biological enrichment evidence, and causal dose-response experiments used to evaluate whether those latents can steer generation.

This README documents the project state as of the latest work and explains how to view and build the dashboard locally.

## Key Goals

- Identify biologically meaningful sparse latent features from a ProGen-3 autoencoder trained on protein sequences.
- Summarize evidence (enrichment statistics, AUROC, odds ratios, point-biserial correlations).
- Visualize dose-response / causal steering experiments showing how feature activation affects generation.
- Present a clean, editorial dashboard (Apple-inspired aesthetic) to communicate findings.

## Where to look (important paths)

- **Dashboard app**: [dashboard](dashboard) — Next.js App Router, TypeScript, Tailwind CSS.
	- Main page: [dashboard/app/page.tsx](dashboard/app/page.tsx)
	- Client-side protein viewer: [dashboard/components/protein-viewer.tsx](dashboard/components/protein-viewer.tsx)
	- Dashboard data loader: [dashboard/app/data.ts](dashboard/app/data.ts)
	- Global styles: [dashboard/app/globals.css](dashboard/app/globals.css)

- **Model results & analysis**: [results](results)
	- Feature enrichment: `global_feature_enrichment.csv`
	- Causal dose-response: `causal_feature_dose_response.csv`
	- Model checkpoints and training history under `results/topk_sae_layer6_d4096_k32_run1/`

- **Data / processed**: [data/processed](data/processed) — CSVs, memmaps, and sequence FASTAs used for analyses.

- **Utilities**: `data/utils` and `topk_sae` contain dataset and model training utilities.

## Dashboard: running locally

Prereqs: Node.js (recommended >=18), npm, and a working shell. The dashboard expects the repository results to be present in `results/` and some data files under `data/`.

1. Open a terminal and change into the dashboard folder:

```bash
cd dashboard
```

2. Install dependencies (if not already installed):

```bash
npm install
```

3. Run the development server:

```bash
npm run dev
```

4. Build for production (static prerender):

```bash
npm run build
```

Notes:
- The hero protein visualization uses `3dmol` and is initialized client-side. The viewer is configured with a transparent WebGL canvas (`backgroundAlpha: 0`) so the dashboard background shows through.
- If you encounter SSR errors referencing `window` or `3dmol`, ensure the viewer is only imported/initialized in client-side code.

## Charting and visual conventions

- The dashboard follows a soft editorial palette (warm white / pink / orange) and is intentionally minimal.
- Charts are implemented as inline SVGs on the main page and are responsive to container width. Axis labels, margins, and spacing are tuned to avoid overlap with plotted data.

## Recent edits & important implementation notes

- Protein viewer: `dashboard/components/protein-viewer.tsx` — updated to import `3dmol` dynamically and set `backgroundAlpha: 0` and `viewer.setBackgroundColor(0x000000, 0)` to ensure canvas transparency.
- Chart layout: the explained-variance chart on the home page uses padded plotting margins so the orange line never overlaps axis labels; axis titles and tick labels are placed outside the plotting area.

## Next steps / improvements (ideas)

- Add interactive axis zoom/hover tooltips to charts.
- Add unit/integration tests for data parsing in `dashboard/app/data.ts`.
- Expand dashboard pages to include a feature detail view with sequence logos and example generations.

## Contact / attribution

Project: ProGen-3 sparse autoencoder feature atlas
Authors & lab: Sri Seshadri / Romero Lab

If you want edits to the README or the dashboard content, tell me which sections to emphasize or any additional artifacts to surface.

---

Generated and maintained alongside the dashboard code at `dashboard/`.

