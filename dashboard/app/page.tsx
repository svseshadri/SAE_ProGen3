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
  const pad = 22;

  return points
    .map((point, index) => {
      const x = pad + ((point.x - xMin) / Math.max(xMax - xMin, 1)) * (width - pad * 2);
      const y = height - pad - ((point.y - yMin) / Math.max(yMax - yMin, 1)) * (height - pad * 2);
      return `${index === 0 ? "M" : "L"}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function StatCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-3xl border border-white/10 bg-white/5 p-5 shadow-[0_10px_40px_-15px_rgba(120,119,198,0.2)] backdrop-blur-sm">
      <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">{label}</div>
      <div className="mt-3 text-3xl font-semibold text-white">{value}</div>
      <div className="mt-2 text-sm text-slate-300">{detail}</div>
    </div>
  );
}

export default async function Home() {
  const data = await getDashboardData();
  const trainingCurve = data.training.map((point) => ({ x: point.step, y: point.val_explained_variance }));
  const candidateFeatures = data.featureSummaries.slice(0, 5);
  const topFeatures = data.featureRankings.slice(0, 8);

  return (
    <main className="min-h-screen bg-[#050816] text-slate-100">
      <div className="mx-auto max-w-7xl px-6 pb-16 pt-8 md:px-10">
        <header className="mb-8 flex items-center justify-between border-b border-white/10 pb-6">
          <div>
            <p className="text-xs uppercase tracking-[0.3em] text-cyan-300">ProGen-3 / SAE</p>
            <h1 className="mt-2 text-2xl font-semibold text-white md:text-3xl">Sparse Autoencoder Atlas</h1>
          </div>
          <div className="hidden items-center gap-3 md:flex">
            <span className="rounded-full border border-cyan-400/40 bg-cyan-400/10 px-3 py-1 text-xs text-cyan-200">Layer 6</span>
            <span className="rounded-full border border-violet-400/40 bg-violet-400/10 px-3 py-1 text-xs text-violet-200">Top-K 32</span>
          </div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.4fr_0.8fr]">
          <div className="rounded-[32px] border border-white/10 bg-[radial-gradient(circle_at_top_left,_rgba(34,211,238,0.22),_transparent_30%),linear-gradient(135deg,_rgba(15,23,42,0.94),_rgba(15,23,42,0.76))] p-7 shadow-[0_20px_80px_-30px_rgba(34,211,238,0.35)]">
            <div className="mb-5 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.22em] text-slate-300">
              <span className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1">Evidence-backed</span>
              <span className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-2.5 py-1 text-emerald-200">Causal steering</span>
            </div>
            <h2 className="max-w-2xl text-4xl font-semibold tracking-tight text-white md:text-5xl">
              What the sparse autoencoder learned about protein biology.
            </h2>
            <p className="mt-5 max-w-2xl text-base leading-7 text-slate-300">
              This dashboard links sparse feature discovery, biological enrichment, and directed intervention results from the ProGen-3 layer-6 SAE run. The strongest evidence is not just reconstruction quality, but whether a learned feature matches a real biological motif and can shift the model’s behavior in a controlled way.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <a href="#evidence" className="rounded-full bg-cyan-300 px-5 py-2.5 text-sm font-medium text-slate-950 transition hover:bg-cyan-200">View evidence</a>
              <a href="#features" className="rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-white/10">Feature atlas</a>
            </div>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">Model health</div>
                <div className="mt-2 text-3xl font-semibold text-white">{pct(data.overview.medianExplainedVariance)}</div>
              </div>
              <div className="rounded-full border border-emerald-400/30 bg-emerald-400/10 px-3 py-1 text-xs text-emerald-200">Stable</div>
            </div>
            <div className="space-y-4 text-sm text-slate-300">
              <div className="flex items-center justify-between rounded-2xl bg-white/5 p-3">
                <span>Dead latent fraction</span>
                <span className="font-medium text-white">{data.overview.deadLatentFraction.toFixed(3)}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/5 p-3">
                <span>Relative NLL degradation</span>
                <span className="font-medium text-white">{pct(data.overview.meanRelativeNllDegradation)}</span>
              </div>
              <div className="flex items-center justify-between rounded-2xl bg-white/5 p-3">
                <span>Candidate features</span>
                <span className="font-medium text-white">{data.overview.nCandidateFeatures}</span>
              </div>
            </div>
          </div>
        </section>

        <section className="mt-8 grid gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard label="Sequences analyzed" value={data.overview.sequencesAnalyzed.toLocaleString()} detail="Held-out evaluation set" />
          <StatCard label="Latent dimension" value={data.overview.latentDim.toLocaleString()} detail="Sparse dictionary width" />
          <StatCard label="Top-K" value={data.overview.topK.toString()} detail="Active latents per example" />
          <StatCard label="Layer" value={data.overview.layerIndex.toString()} detail="ProGen-3 representation stage" />
        </section>

        <section className="mt-10 grid gap-6 xl:grid-cols-[1.3fr_0.7fr]">
          <div className="rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
            <div className="mb-4 flex items-center justify-between">
              <div>
                <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Training</div>
                <h3 className="mt-2 text-xl font-semibold text-white">Validation explained variance</h3>
              </div>
              <div className="text-sm text-slate-300">{data.training.length} checkpoints</div>
            </div>
            <svg viewBox="0 0 520 200" className="h-56 w-full overflow-visible rounded-2xl bg-slate-950/60 p-2">
              <path d={traceLine(trainingCurve)} fill="none" stroke="#67e8f9" strokeWidth="3" strokeLinecap="round" />
            </svg>
            <div className="mt-3 flex justify-between text-[10px] uppercase tracking-[0.2em] text-slate-500">
              <span>Early</span>
              <span>Mid</span>
              <span>Late</span>
            </div>
          </div>

          <div id="evidence" className="rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Evidence ladder</div>
            <ul className="mt-5 space-y-4">
              {data.evidenceClaims.map((claim, index) => (
                <li key={claim} className="flex gap-3 rounded-2xl border border-white/10 bg-white/5 p-3">
                  <span className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-cyan-300/15 text-xs font-semibold text-cyan-200">{index + 1}</span>
                  <p className="text-sm leading-6 text-slate-200">{claim}</p>
                </li>
              ))}
            </ul>
          </div>
        </section>

        <section id="features" className="mt-10 rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
          <div className="mb-5 flex items-center justify-between">
            <div>
              <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Feature atlas</div>
              <h3 className="mt-2 text-xl font-semibold text-white">Top latent candidates by biological association</h3>
            </div>
            <div className="text-sm text-slate-400">Q-value controlled enrichment</div>
          </div>

          <div className="overflow-x-auto">
            <table className="min-w-full text-left text-sm text-slate-200">
              <thead className="border-b border-white/10 text-[11px] uppercase tracking-[0.18em] text-slate-400">
                <tr>
                  <th className="pb-3 pr-4">Rank</th>
                  <th className="pb-3 pr-4">Latent</th>
                  <th className="pb-3 pr-4">Annotation</th>
                  <th className="pb-3 pr-4">AUROC</th>
                  <th className="pb-3 pr-4">r</th>
                  <th className="pb-3 pr-4">Odds ratio</th>
                  <th className="pb-3 pr-4">Q-value</th>
                </tr>
              </thead>
              <tbody>
                {topFeatures.map((feature) => (
                  <tr key={`${feature.latent_id}-${feature.annotation}`} className="border-b border-white/5 align-top">
                    <td className="py-3 pr-4 text-slate-300">#{feature.rank}</td>
                    <td className="py-3 pr-4 font-medium text-white">{feature.latent_id}</td>
                    <td className="py-3 pr-4 text-cyan-200">{feature.annotation}</td>
                    <td className="py-3 pr-4">{feature.auroc.toFixed(3)}</td>
                    <td className="py-3 pr-4">{feature.point_biserial_r.toFixed(3)}</td>
                    <td className="py-3 pr-4">{feature.fisher_odds_ratio.toFixed(1)}</td>
                    <td className="py-3 pr-4">{feature.q_value.toFixed(3)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="mt-10 grid gap-6 lg:grid-cols-2">
          <div className="rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Causal validation</div>
            <h3 className="mt-2 text-xl font-semibold text-white">Dose-response steering candidates</h3>
            <div className="mt-5 space-y-4">
              {candidateFeatures.map((feature) => (
                <div key={feature.featureId} className="rounded-2xl border border-white/10 bg-white/5 p-4">
                  <div className="mb-3 flex items-center justify-between">
                    <div>
                      <div className="text-xs uppercase tracking-[0.18em] text-slate-400">Feature {feature.featureId}</div>
                      <div className="mt-1 text-lg font-medium text-white">{feature.conceptLabel}</div>
                    </div>
                    <div className="rounded-full bg-violet-400/10 px-2.5 py-1 text-xs text-violet-200">Gap {feature.doseGap.toFixed(2)}</div>
                  </div>
                  <div className="grid grid-cols-3 gap-3 text-sm text-slate-300">
                    <div className="rounded-xl bg-slate-950/70 p-3"><div className="text-slate-400">Peak specificity</div><div className="mt-1 text-lg font-medium text-white">{feature.peakSpec.toFixed(2)}</div></div>
                    <div className="rounded-xl bg-slate-950/70 p-3"><div className="text-slate-400">Peak dose</div><div className="mt-1 text-lg font-medium text-white">{feature.peakDose.toFixed(2)}</div></div>
                    <div className="rounded-xl bg-slate-950/70 p-3"><div className="text-slate-400">ΔNLL</div><div className="mt-1 text-lg font-medium text-white">{feature.peakDeltaNll.toFixed(3)}</div></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="rounded-[28px] border border-white/10 bg-slate-900/70 p-5">
            <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Scientific caveat</div>
            <div className="mt-4 rounded-2xl border border-amber-400/25 bg-amber-400/10 p-4 text-sm leading-6 text-amber-100">
              The confirmatory mixed-effects analysis is explicitly marked as pending because the random-effect covariance was numerically singular. In other words, the strongest validated story here is a combination of:<br /><br />
              1. biologically enriched latent structure,<br />
              2. target-matched causal dose responses, and<br />
              3. conservative interpretation of effect magnitude.
            </div>
            <div className="mt-5 space-y-3 text-sm text-slate-300">
              <div className="rounded-2xl bg-white/5 p-3">Global reconstruction degradation: <span className="font-medium text-white">{pct(data.reconstruction.global.mean_relative_nll_degradation_pct)}</span></div>
              <div className="rounded-2xl bg-white/5 p-3">Positive set degradation: <span className="font-medium text-white">{pct(data.reconstruction.positive.mean_relative_nll_degradation_pct)}</span></div>
              <div className="rounded-2xl bg-white/5 p-3">Background degradation: <span className="font-medium text-white">{pct(data.reconstruction.background.mean_relative_nll_degradation_pct)}</span></div>
            </div>
          </div>
        </section>

        <section className="mt-10 rounded-[28px] border border-dashed border-white/15 bg-slate-950/70 p-5">
          <div className="text-[11px] uppercase tracking-[0.22em] text-slate-400">Source artifacts</div>
          <div className="mt-4 flex flex-wrap gap-2">
            {data.sourceArtifacts.map((artifact) => (
              <span key={artifact} className="rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-xs text-slate-200">
                {artifact}
              </span>
            ))}
          </div>
        </section>
      </div>
    </main>
  );
}
