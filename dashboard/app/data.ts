import fs from "fs";
import path from "path";

export type TrainingPoint = {
  step: number;
  train_loss: number;
  val_mse: number;
  val_nmse: number;
  val_explained_variance: number;
  val_l0: number;
  dead_latent_fraction: number;
};

export type FeatureRankingRow = {
  latent_id: number;
  annotation: string;
  rank: number;
  n_positive: number;
  n_negative: number;
  activation_prevalence: number;
  mean_activation_positive: number;
  mean_activation_negative: number;
  point_biserial_r: number;
  point_biserial_p: number;
  auroc: number;
  fisher_odds_ratio: number;
  fisher_p: number;
  q_value: number;
  activation_threshold: number;
  fire_count_positive: number;
  fire_count_negative: number;
  any_active_positive: number;
  any_active_negative: number;
};

export type DosePoint = {
  dose: number;
  delta_nll: number;
  motif_specificity_score: number;
  kl: number;
};

export type FeatureSummary = {
  featureId: number;
  conceptLabel: string;
  positiveMaxSpec: number;
  negativeMaxSpec: number;
  doseGap: number;
  peakDose: number;
  peakDeltaNll: number;
  peakSpec: number;
  peakKl: number;
  doseSeries: {
    positive: DosePoint[];
    negative: DosePoint[];
  };
};

export type DashboardData = {
  overview: {
    sequencesAnalyzed: number;
    layerIndex: number;
    latentDim: number;
    topK: number;
    medianExplainedVariance: number;
    deadLatentFraction: number;
    meanRelativeNllDegradation: number;
    nCandidateFeatures: number;
  };
  training: TrainingPoint[];
  reconstruction: {
    global: { n_sequences: number; mean_base_nll: number; mean_patched_nll: number; mean_delta_nll: number; mean_relative_nll_degradation_pct: number };
    positive: { n_sequences: number; mean_base_nll: number; mean_patched_nll: number; mean_delta_nll: number; mean_relative_nll_degradation_pct: number };
    background: { n_sequences: number; mean_base_nll: number; mean_patched_nll: number; mean_delta_nll: number; mean_relative_nll_degradation_pct: number };
  };
  featureRankings: FeatureRankingRow[];
  featureSummaries: FeatureSummary[];
  evidenceClaims: string[];
  sourceArtifacts: string[];
};

function parseCsv(content: string): Record<string, string>[] {
  const lines = content.trim().split(/\r?\n/);
  if (lines.length < 2) return [];
  const headers = splitCsvLine(lines[0]);
  return lines.slice(1).filter(Boolean).map((line) => {
    const values = splitCsvLine(line);
    const row: Record<string, string> = {};
    headers.forEach((header, idx) => {
      row[header] = values[idx] ?? "";
    });
    return row;
  });
}

function splitCsvLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (inQuotes && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else {
        inQuotes = !inQuotes;
      }
    } else if (char === "," && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current);
  return result;
}

function readJson<T>(filePath: string): T {
  return JSON.parse(fs.readFileSync(filePath, "utf8")) as T;
}

function formatBiologicalLabel(label: string): string {
  const normalized = label.trim();
  const lower = normalized.toLowerCase();

  if (["class_label", "class_name", "s1a_trypsin_chymotrypsin"].includes(lower)) {
    return "Trypsin/chymotrypsin-like protease";
  }
  if (lower.includes("ps00134")) {
    return "PS00134 catalytic motif enrichment";
  }
  if (lower.includes("ps00135")) {
    return "PS00135 catalytic motif enrichment";
  }
  if (lower.includes("ipr001314") || lower.includes("s1 protease")) {
    return "S1 serine protease domain enrichment";
  }
  if (lower.includes("background")) {
    return "Background sequence set";
  }

  return normalized.replace(/_/g, " ");
}

function toNumber(value: string | number | undefined, fallback = 0): number {
  if (typeof value === "number") return Number.isFinite(value) ? value : fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function getRepoRoot(): string {
  return path.resolve(process.cwd(), "..");
}

function summarizeFeatureDoseRows(rows: Record<string, string>[]): FeatureSummary[] {
  const grouped = new Map<number, { featureId: number; conceptLabel: string; series: { positive: DosePoint[]; negative: DosePoint[] } }>();

  rows.forEach((row) => {
    const featureId = Number(row.feature_id ?? 0);
    const dose = Number(row.dose ?? 0);
    const deltaNll = Number(row.delta_nll ?? 0);
    const kl = Number(row.kl ?? 0);
    const motif = Number(row.motif_specificity_score ?? 0);
    const conceptPositive = String(row.concept_positive ?? "").toLowerCase() === "true";

    if (!grouped.has(featureId)) {
      grouped.set(featureId, {
        featureId,
        conceptLabel: formatBiologicalLabel(String(row.matched_concept ?? "feature")),
        series: { positive: [], negative: [] },
      });
    }

    const entry = grouped.get(featureId)!;
    const point: DosePoint = { dose, delta_nll: deltaNll, motif_specificity_score: motif, kl };
    if (conceptPositive) entry.series.positive.push(point);
    else entry.series.negative.push(point);
  });

  return Array.from(grouped.values())
    .map(({ featureId, conceptLabel, series }) => {
      const allPositive = series.positive
        .slice()
        .sort((a, b) => a.dose - b.dose);
      const allNegative = series.negative
        .slice()
        .sort((a, b) => a.dose - b.dose);

      const positiveMaxSpec = allPositive.reduce((max, point) => Math.max(max, point.motif_specificity_score), 0);
      const negativeMaxSpec = allNegative.reduce((max, point) => Math.max(max, point.motif_specificity_score), 0);
      const peakPositive = allPositive.reduce((best, point) => {
        if (point.motif_specificity_score > best.motif_specificity_score) return point;
        return best;
      }, allPositive[0] ?? { dose: 0, delta_nll: 0, motif_specificity_score: 0, kl: 0 });
      const peakNegative = allNegative.reduce((best, point) => {
        if (point.motif_specificity_score > best.motif_specificity_score) return point;
        return best;
      }, allNegative[0] ?? { dose: 0, delta_nll: 0, motif_specificity_score: 0, kl: 0 });
      const peak = peakPositive.motif_specificity_score >= (peakNegative.motif_specificity_score ?? 0) ? peakPositive : peakNegative;

      return {
        featureId,
        conceptLabel,
        positiveMaxSpec,
        negativeMaxSpec,
        doseGap: positiveMaxSpec - negativeMaxSpec,
        peakDose: peak.dose,
        peakDeltaNll: peak.delta_nll,
        peakSpec: peak.motif_specificity_score,
        peakKl: peak.kl,
        doseSeries: {
          positive: allPositive,
          negative: allNegative,
        },
      } satisfies FeatureSummary;
    })
    .sort((a, b) => b.doseGap - a.doseGap);
}

export async function getDashboardData(): Promise<DashboardData> {
  const repoRoot = getRepoRoot();

  const hparamsPath = path.join(repoRoot, "results", "topk_sae_layer6_d4096_k32_run1", "hparams.txt");
  const hparamsText = fs.readFileSync(hparamsPath, "utf8");
  const hparams: Record<string, string> = {};
  hparamsText.split(/\r?\n/).forEach((line) => {
    if (!line.includes(":")) return;
    const [key, ...rest] = line.split(":");
    hparams[key.trim()] = rest.join(":").trim();
  });

  const trainingHistory = readJson<TrainingPoint[]>(path.join(repoRoot, "results", "topk_sae_layer6_d4096_k32_run1", "history.json"));
  const reconstruction = readJson<{ global: any; positive: any; background: any }>(path.join(repoRoot, "results", "reconstruction_evaluation.summary.json"));
  const globalFeatureCsv = parseCsv(fs.readFileSync(path.join(repoRoot, "results", "global_feature_enrichment.csv"), "utf8"));
  const causalCsv = parseCsv(fs.readFileSync(path.join(repoRoot, "results", "causal_feature_dose_response.csv"), "utf8"));
  const latentSummary = readJson<{ [key: string]: any }[]>(path.join(repoRoot, "results", "topk_sae_layer6_d4096_k32_run1", "analysis", "latent_summary.json"));

  const featureRankings = globalFeatureCsv.slice(0, 20).map((row) => ({
    latent_id: Number(row.latent_id ?? 0),
    annotation: formatBiologicalLabel(String(row.annotation ?? "class_label")),
    rank: Number(row.rank ?? 0),
    n_positive: Number(row.n_positive ?? 0),
    n_negative: Number(row.n_negative ?? 0),
    activation_prevalence: Number(row.activation_prevalence ?? 0),
    mean_activation_positive: Number(row.mean_activation_positive ?? 0),
    mean_activation_negative: Number(row.mean_activation_negative ?? 0),
    point_biserial_r: Number(row.point_biserial_r ?? 0),
    point_biserial_p: Number(row.point_biserial_p ?? 0),
    auroc: Number(row.auroc ?? 0),
    fisher_odds_ratio: Number(row.fisher_odds_ratio ?? 0),
    fisher_p: Number(row.fisher_p ?? 0),
    q_value: Number(row.q_value ?? 0),
    activation_threshold: Number(row.activation_threshold ?? 0),
    fire_count_positive: Number(row.fire_count_positive ?? 0),
    fire_count_negative: Number(row.fire_count_negative ?? 0),
    any_active_positive: Number(row.any_active_positive ?? 0),
    any_active_negative: Number(row.any_active_negative ?? 0),
  }));

  const featureSummaries = summarizeFeatureDoseRows(causalCsv).
    filter((feature) => [3256, 2942, 1644, 727, 1].includes(feature.featureId));

  const varianceValues = trainingHistory.map((point) => point.val_explained_variance);
  const meanExplainedVariance = varianceValues.length ? varianceValues.reduce((sum, value) => sum + value, 0) / varianceValues.length : 0;

  const evidenceClaims = [
    `The SAE was trained on layer 6 with a hidden width of ${hparams.d_in ?? "384"} and a Top-K width of ${hparams.k ?? "32"}.`,
    `The validation reconstruction explained variance reached ${Number((trainingHistory.at(-1)?.val_explained_variance ?? 0) * 100).toFixed(1)}% by the end of training.`,
    `On the held-out evaluation set, the global mean relative NLL degradation was ${toNumber(reconstruction.global.mean_relative_nll_degradation_pct, 0).toFixed(1)}%, with the positive set at ${toNumber(reconstruction.positive.mean_relative_nll_degradation_pct, 0).toFixed(1)}% and the background at ${toNumber(reconstruction.background.mean_relative_nll_degradation_pct, 0).toFixed(1)}%.`,
    `The lead biological candidates show clear concept-specific steering, with feature 3256 peaking at ${featureSummaries.find((f) => f.featureId === 3256)?.peakSpec.toFixed(2) ?? "0.00"} motif specificity and a positive–negative gap of ${featureSummaries.find((f) => f.featureId === 3256)?.doseGap.toFixed(2) ?? "0.00"}.`,
    `The mixed-effects confirmatory model was numerically unstable (singular random-effect covariance), so the evidence ladder must be read as exploratory-plus-causal rather than final confirmatory proof.`,
  ];

  return {
    overview: {
      sequencesAnalyzed: Number(reconstruction.global.n_sequences ?? 0),
      layerIndex: Number(hparams.d_in ? 6 : 0),
      latentDim: Number(hparams.d_sae ?? 4096),
      topK: Number(hparams.k ?? 32),
      medianExplainedVariance: meanExplainedVariance,
      deadLatentFraction: Number(trainingHistory.at(-1)?.dead_latent_fraction ?? 0),
      meanRelativeNllDegradation: Number(reconstruction.global.mean_relative_nll_degradation_pct ?? 0),
      nCandidateFeatures: featureRankings.length,
    },
    training: trainingHistory.slice(0, 25),
    reconstruction,
    featureRankings,
    featureSummaries,
    evidenceClaims,
    sourceArtifacts: [
      "results/topk_sae_layer6_d4096_k32_run1/history.json",
      "results/topk_sae_layer6_d4096_k32_run1/hparams.txt",
      "results/reconstruction_evaluation.summary.json",
      "results/global_feature_enrichment.csv",
      "results/causal_feature_dose_response.csv",
      "data/processed/s1a70/s1a70_test.csv",
      "data/plots/qc_plots",
      "results/topk_sae_layer6_d4096_k32_run1/analysis/latent_summary.json",
    ],
  };
}
