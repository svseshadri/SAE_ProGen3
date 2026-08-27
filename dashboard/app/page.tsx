import { getDashboardData } from "./data";

function pct(value: number) {
  return `${value.toFixed(1)}%`;
}

function traceLine(points: { x: number; y: number }[]) {
  if (points.length === 0) return "";
  const xMin = Math.min(...points.map((p) => p.x));
  const xMax = Math.max(...points.map((p) => p.x));
  const yMin = Math.min(...points.map((p) => p.y));
  const yMax = Math.max(...points.map((p) => p.y));

  const width = 520;
  const height = 200;
  const pad = 18;

  return points
    .map((point, index) => {
      const x = pad + ((point.x - xMin) / Math.max(xMax - xMin, 1)) * (width - pad * 2);
      const y = height - pad - ((point.y - yMin) / Math.max(yMax - yMin, 1)) * (height - pad * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function RingGauge({ value, label, color }: { value: number; label: string; color: string }) {
  const safeValue = Math.max(0, Math.min(100, value));
  return (
    <div className="flex items-center gap-4 rounded-[28px] border border-rose-200 bg-white/90 p-4 shadow-[0_18px_38px_-26px_rgba(251,146,60,0.55)]">
      <div
        className="relative flex h-20 w-20 items-center justify-center rounded-full"
        style={{
          background: `conic-gradient(${color} ${safeValue}%, rgba(251,146,60,0.12) 0)`,
        }}
      >
        <div className="flex h-12 w-12 items-center justify-center rounded-full bg-white text-sm font-semibold text-slate-800 shadow-[inset_0_0_0_1px_rgba(251,146,60,0.2)]">
          {safeValue.toFixed(0)}
        </div>
      </div>
      <div>
        <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">{label}</div>
        <div className="mt-1 text-lg font-medium text-slate-800">{pct(value)}</div>
      </div>
    </div>
  );
}

function RankBar({ label, value, accent }: { label: string; value: number; accent: string }) {
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-slate-600">
        <span>{label}</span>
        <span className="font-medium text-slate-800">{value.toFixed(3)}</span>
      </div>
      <div className="h-2.5 w-full overflow-hidden rounded-full bg-orange-100/80">
        <div
          className="h-full rounded-full"
          style={{ width: `${Math.min(100, value * 100)}%`, background: accent }}
        />
      </div>
    </div>
  );
}

export default async function Home() {
  const data = await getDashboardData();
  const trainingCurve = data.training.map((point) => ({ x: point.step, y: point.val_explained_variance }));
  const candidateFeatures = data.featureSummaries.slice(0, 5);
  const topFeatures = data.featureRankings.slice(0, 5);

  return (
    <main className="min-h-screen bg-[#fffaf7] text-slate-800">
      <div className="absolute inset-x-0 top-0 h-[520px] bg-[radial-gradient(circle_at_top,_rgba(251,146,60,0.18),_transparent_55%)]" />

      <div className="relative mx-auto max-w-7xl px-4 pb-24 pt-6 md:px-8">
        <header className="flex items-center justify-between rounded-full border border-rose-200/80 bg-white/80 px-5 py-3 shadow-[0_10px_40px_-15px_rgba(251,146,60,0.35)] backdrop-blur-md">
          <div className="flex items-center gap-3">
            <div className="relative flex h-8 w-8 items-center justify-center overflow-hidden rounded-full border border-rose-200 bg-[radial-gradient(circle_at_30%_30%,rgba(255,255,255,0.9),rgba(251,146,60,0.25),rgba(244,114,182,0.18)_50%,rgba(255,255,255,0.8))] shadow-[inset_0_0_18px_rgba(251,146,60,0.12)]">
              <div className="absolute inset-[4px] rounded-full border border-orange-200/80" />
              <div className="absolute left-1/2 top-1/2 h-5 w-5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[radial-gradient(circle_at_center,rgba(255,255,255,0.9),rgba(251,146,60,0.7),rgba(244,114,182,0.28)_65%,rgba(255,255,255,0.2))]" />
              <div className="absolute left-[3px] top-[3px] h-1.5 w-1.5 rounded-full bg-white/90" />
              <div className="absolute right-[3px] top-[5px] h-1.5 w-1.5 rounded-full bg-orange-200/90" />
              <div className="absolute left-[5px] bottom-[4px] h-1.5 w-1.5 rounded-full bg-pink-200/90" />
              <div className="absolute right-[5px] bottom-[4px] h-1.5 w-1.5 rounded-full bg-white/90" />
              <div className="absolute left-1/2 top-[1px] h-2 w-px bg-orange-200/70" />
              <div className="absolute bottom-[1px] left-1/2 h-2 w-px bg-orange-200/70" />
              <div className="absolute left-[1px] top-1/2 h-px w-2 bg-orange-200/60" />
              <div className="absolute right-[1px] top-1/2 h-px w-2 bg-orange-200/60" />
            </div>
            <div className="text-[11px] uppercase tracking-[0.28em] text-slate-600">ProGen-3</div>
          </div>
          <div className="hidden items-center gap-6 text-sm text-slate-600 md:flex">
            <span>Project</span>
            <span>Evidence</span>
            <span>Causality</span>
            <span>Atlas</span>
          </div>
          <div className="flex items-center gap-4 text-slate-700">
            <div className="text-xl">⌕</div>
            <div className="text-xl">◌</div>
            <div className="text-xl">☰</div>
          </div>
        </header>

        <section className="mt-8 overflow-hidden rounded-[38px] border border-rose-200 bg-[linear-gradient(135deg,#fffaf7_0%,#fff1ed_28%,#ffe7d6_100%)] px-6 py-8 shadow-[0_35px_100px_-35px_rgba(251,146,60,0.55)] md:px-10 md:py-12">
          <div className="mb-4 flex justify-center">
            <div className="inline-flex items-center gap-2 rounded-full border border-rose-200 bg-white/70 px-4 py-2 text-[11px] uppercase tracking-[0.25em] text-slate-700">
              Evidence-backed interpretability
            </div>
          </div>

          <div className="text-center">
            <div className="mb-4 flex justify-center">
              <div className="inline-flex items-center gap-2 rounded-full border border-orange-200 bg-white/80 px-3 py-1.5 text-[10px] uppercase tracking-[0.25em] text-slate-600">
                <span>Sri Seshadri</span>
                <span className="text-orange-400">•</span>
                <span>Romero Lab</span>
              </div>
            </div>
            <h1 className="text-4xl font-semibold tracking-[-0.06em] text-slate-800 md:text-7xl">
              ProGen3 Feature Atlas
            </h1>
            <p className="mx-auto mt-5 max-w-3xl text-base text-slate-600 md:text-xl">
              Layer 6 sparse features that causally identify trypsin/chymotrypsin-like motifs, track biological signal, and show dose-sensitive causal control over ProGen3 generation.
            </p>
          </div>

          <div className="relative mt-10 flex justify-center">
            <div className="relative h-[260px] w-full max-w-5xl overflow-hidden rounded-[36px] border border-orange-200/80 bg-[radial-gradient(circle_at_center,_rgba(251,146,60,0.16),_rgba(255,255,255,0.7)_38%,_rgba(255,255,255,0.9)_100%)]">
              <div className="absolute inset-x-10 bottom-6 top-14 rounded-[32px] border border-white/70 bg-[linear-gradient(135deg,#fff7f2_0%,#fee4d6_40%,#ffd2b7_100%)] shadow-[0_40px_80px_-30px_rgba(251,146,60,0.45)]" />
              <div className="absolute left-1/2 top-1/2 h-52 w-[68%] -translate-x-1/2 -translate-y-1/2 rounded-[44px] border border-orange-200/80 bg-[linear-gradient(135deg,rgba(255,255,255,0.78),rgba(255,210,183,0.45),rgba(253,164,175,0.22))] shadow-[inset_0_0_22px_rgba(251,146,60,0.14)]" />

              <div className="absolute inset-0">
                <div className="absolute left-[15%] top-[20%] h-2.5 w-2.5 rounded-full bg-orange-500/90 shadow-[0_0_22px_rgba(251,146,60,0.7)]" />
                <div className="absolute left-[22%] top-[33%] h-2.5 w-2.5 rounded-full bg-pink-500/90 shadow-[0_0_20px_rgba(244,114,182,0.65)]" />
                <div className="absolute left-[30%] top-[18%] h-2.5 w-2.5 rounded-full bg-orange-400/90 shadow-[0_0_20px_rgba(251,146,60,0.7)]" />
                <div className="absolute left-[39%] top-[27%] h-2.5 w-2.5 rounded-full bg-pink-400/90 shadow-[0_0_20px_rgba(244,114,182,0.7)]" />
                <div className="absolute left-[48%] top-[40%] h-3 w-3 rounded-full bg-amber-500/90 shadow-[0_0_22px_rgba(251,191,36,0.7)]" />
                <div className="absolute left-[58%] top-[21%] h-2.5 w-2.5 rounded-full bg-orange-500/90 shadow-[0_0_20px_rgba(251,146,60,0.7)]" />
                <div className="absolute left-[66%] top-[36%] h-2.5 w-2.5 rounded-full bg-pink-500/90 shadow-[0_0_20px_rgba(244,114,182,0.7)]" />
                <div className="absolute left-[73%] top-[52%] h-2.5 w-2.5 rounded-full bg-orange-400/90 shadow-[0_0_20px_rgba(251,146,60,0.7)]" />
                <div className="absolute left-[46%] top-[66%] h-2.5 w-2.5 rounded-full bg-pink-400/90 shadow-[0_0_20px_rgba(244,114,182,0.7)]" />
                <div className="absolute left-[28%] top-[62%] h-2.5 w-2.5 rounded-full bg-orange-500/90 shadow-[0_0_20px_rgba(251,146,60,0.65)]" />

                <div className="absolute left-[15%] top-[20%] h-24 w-px bg-orange-300/60" />
                <div className="absolute left-[22%] top-[33%] h-18 w-px bg-pink-300/60" />
                <div className="absolute left-[30%] top-[18%] h-24 w-px bg-orange-300/60" />
                <div className="absolute left-[39%] top-[27%] h-18 w-px bg-pink-300/60" />
                <div className="absolute left-[48%] top-[40%] h-22 w-px bg-amber-300/70" />
                <div className="absolute left-[58%] top-[21%] h-24 w-px bg-orange-300/60" />
                <div className="absolute left-[66%] top-[36%] h-18 w-px bg-pink-300/60" />
                <div className="absolute left-[46%] top-[66%] h-16 w-px bg-pink-300/60" />

                <div className="absolute left-[15%] top-[20%] h-px w-[17%] bg-orange-300/60" />
                <div className="absolute left-[30%] top-[18%] h-px w-[18%] bg-pink-300/60" />
                <div className="absolute left-[39%] top-[27%] h-px w-[17%] bg-orange-300/60" />
                <div className="absolute left-[48%] top-[40%] h-px w-[15%] bg-amber-300/60" />
                <div className="absolute left-[28%] top-[62%] h-px w-[18%] bg-pink-300/60" />
                <div className="absolute left-[46%] top-[66%] h-px w-[20%] bg-orange-300/60" />
              </div>
            </div>
          </div>

          <div className="mt-10 flex flex-wrap items-center justify-center gap-4 text-sm text-slate-600">
            <div className="rounded-full border border-orange-200 bg-white/75 px-4 py-2">Layer 6</div>
            <div className="rounded-full border border-pink-200 bg-white/75 px-4 py-2">Top-K 32</div>
            <div className="rounded-full border border-orange-200 bg-white/75 px-4 py-2">4096 dictionary</div>
            <div className="rounded-full border border-pink-200 bg-white/75 px-4 py-2">7963 sequences</div>
          </div>
        </section>

        <section className="mt-10 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <div className="rounded-[28px] border border-orange-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(251,146,60,0.5)]">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Explained variance</div>
            <div className="mt-3 text-4xl font-semibold text-slate-800">{data.overview.medianExplainedVariance.toFixed(2)}</div>
            <div className="mt-3 text-sm text-slate-600">Validation fit improves steadily throughout training.</div>
          </div>
          <div className="rounded-[28px] border border-pink-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(244,114,182,0.45)]">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Dead latents</div>
            <div className="mt-3 text-4xl font-semibold text-slate-800">{data.overview.deadLatentFraction.toFixed(3)}</div>
            <div className="mt-3 text-sm text-slate-600">Sparse coding remains healthy and non-collapsed.</div>
          </div>
          <div className="rounded-[28px] border border-orange-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(251,146,60,0.5)]">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">NLL degradation</div>
            <div className="mt-3 text-4xl font-semibold text-slate-800">{pct(data.overview.meanRelativeNllDegradation)}</div>
            <div className="mt-3 text-sm text-slate-600">Residual gap remains meaningful but bounded.</div>
          </div>
          <div className="rounded-[28px] border border-pink-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(244,114,182,0.45)]">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-500">Ranked features</div>
            <div className="mt-3 text-4xl font-semibold text-slate-800">{data.overview.nCandidateFeatures}</div>
            <div className="mt-3 text-sm text-slate-600">Candidate latents with strongest biological enrichment.</div>
          </div>
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[30px] border border-orange-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(251,146,60,0.5)]">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Training dynamics</div>
                <h2 className="mt-2 text-2xl font-semibold text-slate-800">Explained variance over time</h2>
              </div>
              <div className="rounded-full border border-orange-200 bg-orange-50 px-3 py-1 text-xs text-orange-700">Stable</div>
            </div>
            <svg viewBox="0 0 520 200" className="h-56 w-full overflow-visible rounded-[20px] bg-[#fff7f4] p-3">
              <path d={traceLine(trainingCurve)} fill="none" stroke="#fb923c" strokeWidth="3" strokeLinecap="round" />
            </svg>
          </div>

          <div className="grid gap-4">
            <RingGauge value={Math.max(0, 100 - data.overview.meanRelativeNllDegradation)} label="Functional retention" color="#fb923c" />
            <RingGauge value={Math.min(100, data.overview.medianExplainedVariance * 100)} label="Variance captured" color="#f472b6" />
          </div>
        </section>

        <section className="mt-10 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
          <div className="rounded-[30px] border border-pink-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(244,114,182,0.45)]">
            <div className="mb-5 flex items-center justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Feature atlas</div>
                <h2 className="mt-2 text-2xl font-semibold text-slate-800">Biological enrichment</h2>
              </div>
              <div className="text-sm text-slate-500">Top candidates</div>
            </div>
            <div className="space-y-5">
              {topFeatures.map((feature, index) => (
                <div key={`${feature.latent_id}-${feature.annotation}`} className="rounded-[22px] border border-orange-100 bg-[#fffaf7] p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">#{index + 1} latent {feature.latent_id}</div>
                      <div className="mt-1 text-lg font-medium text-slate-800">{feature.annotation}</div>
                    </div>
                    <div className="rounded-full border border-emerald-200 bg-emerald-50 px-2.5 py-1 text-xs text-emerald-700">q={feature.q_value.toFixed(3)}</div>
                  </div>
                  <div className="grid gap-3 md:grid-cols-3">
                    <RankBar label="AUROC" value={feature.auroc} accent="#fb923c" />
                    <RankBar label="r" value={Math.abs(feature.point_biserial_r)} accent="#f472b6" />
                    <RankBar label="Odds" value={Math.min(10, Math.log10(feature.fisher_odds_ratio + 1))} accent="#fbbf24" />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="space-y-6">
            <div className="rounded-[30px] border border-orange-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(251,146,60,0.5)]">
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Causal steering</div>
              <h2 className="mt-2 text-2xl font-semibold text-slate-800">Dose-response</h2>
              <div className="mt-5 space-y-4">
                {candidateFeatures.map((feature) => (
                  <div key={feature.featureId} className="rounded-[20px] border border-orange-100 bg-[#fffaf7] p-4">
                    <div className="flex items-center justify-between">
                      <div className="text-base font-medium text-slate-800">Feature {feature.featureId}</div>
                      <div className="rounded-full bg-orange-100 px-2.5 py-1 text-xs text-orange-700">Gap {feature.doseGap.toFixed(2)}</div>
                    </div>
                    <div className="mt-3 text-sm text-slate-600">{feature.conceptLabel}</div>
                    <div className="mt-4 grid grid-cols-3 gap-2 text-xs text-slate-600">
                      <div className="rounded-xl bg-white p-2 shadow-[inset_0_0_0_1px_rgba(251,146,60,0.12)]"><div>Peak</div><div className="mt-1 text-lg font-medium text-slate-800">{feature.peakSpec.toFixed(2)}</div></div>
                      <div className="rounded-xl bg-white p-2 shadow-[inset_0_0_0_1px_rgba(244,114,182,0.12)]"><div>Dose</div><div className="mt-1 text-lg font-medium text-slate-800">{feature.peakDose.toFixed(2)}</div></div>
                      <div className="rounded-xl bg-white p-2 shadow-[inset_0_0_0_1px_rgba(251,146,60,0.12)]"><div>ΔNLL</div><div className="mt-1 text-lg font-medium text-slate-800">{feature.peakDeltaNll.toFixed(3)}</div></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="rounded-[30px] border border-amber-200 bg-amber-50 p-5 text-sm leading-6 text-amber-900">
              The mixed-effects confirmatory model is still flagged as pending because the variance structure was numerically singular. The strongest evidence remains the intersection of biological enrichment, motif specificity, and targeted causal steering.
            </div>
          </div>
        </section>

        <section className="mt-10 rounded-[30px] border border-dashed border-orange-200 bg-white/90 p-5 shadow-[0_18px_40px_-28px_rgba(251,146,60,0.4)]">
          <div className="text-[11px] uppercase tracking-[0.22em] text-slate-500">Source artifacts</div>
          <div className="mt-4 flex flex-wrap gap-2">
            {data.sourceArtifacts.map((artifact) => (
              <span key={artifact} className="rounded-full border border-orange-200 bg-[#fffaf7] px-3 py-1.5 text-xs text-slate-700">{artifact}</span>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
