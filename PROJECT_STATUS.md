# PROJECT_STATUS.md

Last updated: 2026-08-29

## Current stage

PS00134 token-level provenance recovered for the exact 64-sequence causal
cohort using the frozen official ScanProsite PS00134 definition; multiplicative
native-scaling steering has not yet been implemented or run.

The SAE training, biological feature-enrichment, reconstruction diagnostics,
residual analysis, additive intervention machinery, and prespecified
fixed-effects confirmatory statistics are complete enough for the current
scientific question.

The immediate question is no longer whether the intervention code works or
whether the fixed-effects dose x concept interaction is positive. The
directionality audit established that the stored localized motif-specificity
readouts are unsigned magnitude summaries. A targeted signed-localized rerun
script showed that `target_activation=0` in the existing causal dose grid is not
an identity/no-op condition for feature 3256.

A new identity-centered signed-displacement runner now implements:

    z'_3256 = z_native,3256 + displacement

with the same residual-preserving additive hidden-state intervention. A
two-sequence CUDA identity smoke with `displacement=0` passed exactly at the
recorded tolerance: max hidden delta 0, max delta NLL 0, max KL 0, max signed
token log-probability delta 0, and max signed legacy top-k contrast 0.

However, the TopKSAE encoder is ReLU/top-k, so native latent activations are
nonnegative. On the exact 64-sequence feature-3256 cohort, native feature-3256
activations are sparse: min 0, median 0, mean 0.219558, SD 0.824249, 5th
percentile 0, 25th percentile 0, 75th percentile 0, 95th percentile 2.585158,
max 5.720966, and fraction exactly zero 0.929282. A globally symmetric negative
displacement would therefore push many token activations below the valid encoded
latent domain. Per the predefined stop condition, the full identity-centered
nonzero signed experiment was not run and no directionality figure was created.

A subsequent internal Gate A audit traced `has_ps00134` to the dataset builder's
literal UniProt PROSITE cross-reference check: `prosite_ids` contains `PS00134`.
This exactly reproduces the stored binary labels in all checked raw/processed/
split datasets, but no checked project artifact contained PS00134 motif start/end
coordinates, matched residues, regex spans, PROSITE scan output, or equivalent
position-level provenance.

An external definition recovery step then froze the official PROSITE definition
and scanned the exact stored 64-sequence causal cohort FASTA through the official
ScanProsite endpoint using only PS00134. The recovered current ScanProsite pattern
matches exactly reproduce the 64-sequence cohort binary labels: 54 stored
PS00134-positive sequences have one recoverable PS00134 pattern match each, 10
stored PS00134-negative sequences have no recoverable match, and there are 0
binary-label-versus-recoverable-pattern mismatches. All 54 recovered matches have
ScanProsite confidence `(0)` and span six residues, yielding 324 residue-level
position rows and 54 PRU10078 active-site rows at pattern offset 5.

The external recovery preserves `has_ps00134` and
`has_recoverable_ps00134_pattern_match` as distinct variables. For this exact
cohort they agree, but the original retained UniProt TSV metadata still contains
only `prosite_ids`; hit count, FALSE_NEG, PARTIAL, UNKNOWN, or richer PROSITE
status fields were not retained in the project datasets.

Per the current task scope, multiplicative native-scaling steering was not
implemented or run, no signed steering statistics were generated, and no
native-scaling directionality figure was created. Do not advance to
autoregressive generation until the new PS00134 position annotations are used in
a separately gated steering experiment.

---

## Current scientific objective

Determine whether an interpretable SAE feature identified in ProGen-3 can act
as a biologically selective causal control direction rather than merely:

- correlating with a biological annotation;
- perturbing ProGen-3 globally;
- or producing a large but nonselective steering effect.

The current lead hypothesis is that latent 3256 is selectively associated with
PS00134-related biology.

The fixed-effects confirmatory statistics support a positive 3256 interaction,
but full promotion is currently blocked because the existing canonical causal
artifact does not contain a signed localized biological readout, the targeted
signed rerun exposed a target-zero/no-op mismatch in the existing dose grid, and
the identity-centered signed-displacement follow-up cannot use a nonzero global
symmetric suppression grid without crossing the sparse nonnegative SAE latent
domain.

---

## Completed

- TopK SAE trained on ProGen-3 layer-6 activations.
- SAE latent width: 4,096 features.
- Global biological feature-enrichment pipeline implemented.
- Canonical global enrichment analysis completed.
- Feature-enrichment statistical edge cases and constant-array handling added.
- Full SAE reconstruction fidelity evaluated on held-out data.
- Identity patching validated.
- Layer-alignment / patching convention corrected.
- Full SAE reconstruction found to be behaviorally non-faithful despite very
  strong numerical reconstruction quality.
- Residual interpolation experiment completed.
- Matched random/noise residual control completed.
- High-leverage structure in the SAE reconstruction residual identified.
- Residual-preserving additive intervention implemented.
- Additive identity/no-op condition validated.
- Initial causal feature dose-response experiment completed.
- Feature-specific matched-concept causal experiment completed.
- Both steering directions evaluated where applicable.
- Motif-localized and nonmotif effects quantified.
- Matched-disturbance / matched-KL comparisons completed.
- Candidate hierarchy established descriptively.
- Full mixed-effects confirmatory model attempted and found numerically
  non-identifiable.
- Reduced random-intercept mixed-effects model attempted and also found to have
  singular/boundary random-effects behavior.
- Prespecified fixed-effects OLS fallback with sequence-clustered robust
  uncertainty completed.
- Whole-sequence bootstrap completed with 5,000 requested replicates for each
  estimable biological candidate.
- Benjamini-Hochberg FDR correction applied across the estimable biological
  candidate family.
- Directionality metric audit completed from existing causal artifacts.
- Publication-quality figure-generation script added for reproducible figures
  derived from canonical artifacts.
- Targeted signed-localized feature-3256 rerun script implemented and unit
  tested.
- Tiny signed-localized GPU smoke run completed, reproducing legacy unsigned
  metrics but failing the requested target-zero/no-op invariant.
- Identity-centered signed-displacement feature-3256 runner implemented and
  unit tested.
- Native feature-3256 activation distribution over the exact experimental
  cohort computed.
- Identity-centered `displacement=0` two-sequence CUDA smoke passed exactly, but
  the full nonzero signed-displacement experiment was stopped because a global
  symmetric negative displacement is not feasible without crossing the
  nonnegative TopKSAE latent domain.
- PS00134 annotation provenance audit completed. Binary `has_ps00134` labels
  reproduce exactly from stored `prosite_ids` cross-reference strings, but no
  residue-level PS00134 match coordinates were present in the checked canonical
  project artifacts.
- Official PS00134 external definition frozen and exact 64-sequence causal
  cohort scanned with ScanProsite using only PS00134. Current ScanProsite
  pattern matches reproduce the cohort binary labels exactly and provide
  residue-level coordinates for the 54 PS00134-positive sequences.

---

## Established findings

### 1. The SAE contains biologically structured features

The global feature-enrichment analysis identified SAE latents strongly
associated with predefined serine-protease-related biological annotations.

Important candidate mappings currently include:

- feature 3256 → PS00134
- feature 1644 → PS00135
- feature 2942 → both catalytic motifs
- feature 727 → IPR001314 / broad S1A context

This establishes biological association / detection.

It does not by itself establish causal control.

### 2. Numerical SAE reconstruction fidelity does not imply behavioral fidelity

The SAE reconstructs ProGen-3 layer-6 hidden states extremely well under
standard Euclidean reconstruction metrics, but replacing the true hidden state
with the SAE reconstruction substantially degrades downstream ProGen-3
behavior.

Therefore reconstruction quality and downstream causal faithfulness must be
treated as separate properties.

### 3. The SAE reconstruction residual is small but behaviorally high-leverage

Residual interpolation showed that restoring the missing residual rapidly
recovers downstream ProGen-3 behavior.

The reconstruction error is therefore not well modeled as arbitrary
low-importance noise.

It contains structured directions that have disproportionate downstream
importance.

### 4. Full reconstruction replacement is not the causal steering interface

The canonical causal intervention for the current project is the
residual-preserving additive displacement:

    delta_z = z_steered - z
    delta_h = W_dec @ delta_z
    h_steered = h + delta_h

This preserves the original ProGen-3 hidden state and changes only the decoded
SAE latent displacement.

Full SAE reconstruction remains useful as a diagnostic experiment but is not
the default steering mechanism.

### 5. Selected SAE directions exhibit causal sensitivity

Increasing or decreasing selected SAE features produces dose-dependent changes
in downstream ProGen-3 behavior.

The negative/control feature remains essentially flat.

This establishes causal sensitivity of selected SAE decoder directions.

It does not alone establish biological concept selectivity.

### 6. The strongest descriptive concept-selective candidate is feature 3256

Matched-concept and matched-disturbance analyses currently favor feature 3256
as the lead selective candidate.

The reason is not that 3256 produces the largest raw steering effect.

The reason is that its effect is more strongly separated between its matched
biological context and the corresponding negative context.

---

## Important negative / diagnostic findings

### Full SAE reconstruction is not sufficiently faithful as an intervention interface

Corrected held-out reconstruction evaluation showed approximately:

- base NLL: ~2.145
- SAE-patched NLL: ~2.667
- delta NLL: ~0.522
- relative NLL degradation: ~36%

The identity patch produces effectively zero behavioral change, confirming that
the corrected patching pathway itself is valid.

Residual interpolation approximately showed:

- 0% residual restored → delta NLL ~0.449
- 10% → ~0.383
- 25% → ~0.273
- 50% → ~0.109
- 75% → ~0.024
- 90% → ~0.003
- 95% → approximately 0
- 100% → 0

A matched random-noise perturbation produced substantially different behavior,
supporting the interpretation that the missing residual is structured and
high-leverage.

This negative result must be preserved even if later SAE variants improve
reconstruction fidelity.

---

## Current candidate hierarchy

### 1. Feature 3256 — lead candidate

Matched concept:

    PS00134

Current interpretation:

3256 has a statistically supported positive dose x concept interaction in the
prespecified fixed-effects fallback analysis, but it should not yet be promoted
to a formally validated causal controller because the current canonical causal
artifact does not preserve signed localized motif effects.

Its value as the current lead comes from positive-versus-negative selectivity,
not maximum raw steering magnitude.

### 2. Feature 2942 — predefined secondary comparator

Matched concept:

    both catalytic motifs

Current interpretation:

2942 is a strong steering direction and can produce larger absolute effects,
but it also produces a larger response in concept-negative sequences.

It currently appears broader and less context-selective than 3256.

### 3. Feature 1644 — weaker candidate

Matched concept:

    PS00135

Current interpretation:

1644 shows causal activity but weaker positive-versus-negative separation than
the leading candidates.

### 4. Feature 727 — directional but not currently concept-selective

Matched concept:

    IPR001314

Current interpretation:

727 is a potent direction but the current matched experiment does not contain
a sufficiently usable concept-negative comparison to support a formal
concept-selectivity claim.

Do not assign an artificial zero negative effect or force an interaction test
when the required biological support is absent.

### 5. Feature 1 — negative/control feature

Current interpretation:

The control remains essentially flat and does not reproduce the candidate
dose-response behavior.

Do not rank control features using unstable effect / KL or effect / delta-NLL
ratios near the no-op regime.

---

## Current descriptive evidence

### Matched-KL comparison at KL ~0.025

Feature 3256:

- concept-positive specificity: ~0.782
- concept-negative specificity: ~0.447
- positive-minus-negative gap: ~0.335

Feature 2942:

- concept-positive specificity: ~0.829
- concept-negative specificity: ~0.580
- positive-minus-negative gap: ~0.249

Interpretation:

2942 has slightly larger absolute specificity in the positive cohort at this
disturbance level, but 3256 has greater separation between matched
concept-positive and concept-negative contexts.

### Maximum observed specificity separation

Feature 3256:

- concept-positive maximum specificity: ~1.849
- concept-negative maximum specificity: ~0.985
- maximum gap: ~0.864

Feature 2942:

- concept-positive maximum specificity: ~2.085
- concept-negative maximum specificity: ~1.479
- maximum gap: ~0.606

Interpretation:

2942 can create the larger raw effect, but 3256 currently behaves as the more
selective biological direction.

### Screening dose-response slopes

Approximate descriptive slopes:

Feature 3256:

- concept-positive slope: ~0.0726
- concept-negative slope: ~0.0608
- gap: ~0.0118

Feature 2942:

- concept-positive slope: ~0.1149
- concept-negative slope: ~0.0833
- gap: ~0.0316

Feature 1644:

- concept-positive slope: ~0.1424
- concept-negative slope: ~0.1218
- gap: ~0.0207

Feature 727:

- concept-positive slope: ~0.1043
- matched concept-negative slope: not currently estimable

These slope summaries are descriptive screening evidence only.

They do not replace the prespecified confirmatory interaction analysis.

---

## Current statistical status

The initial confirmatory model attempted to estimate a richer fixed-effects /
random-effects structure and failed with a singular matrix.

A reduced mixed-effects model retaining the primary dose x concept term and a
sequence random intercept was then attempted.

The reduced model produced warnings including:

- singular random-effects covariance;
- MLE on the boundary of the parameter space;
- unstable / unidentified random-effect variance.

These mixed-effects fits remain numerically unstable and inconclusive.

The prespecified fallback fixed-effects analysis is now complete using:

    motif_specificity_score ~ dose_centered * concept_positive

with sequence-clustered robust uncertainty, whole-sequence bootstrap, and
Benjamini-Hochberg FDR across estimable biological candidates.

Canonical outputs:

- `results/confirmatory_causal_statistics.csv`
- `results/confirmatory_causal_bootstrap.csv`

Estimable biological candidate results:

Feature 3256 / PS00134:

- beta_dose_x_concept: 0.0118277
- clustered 95% CI: [0.0013365, 0.0223188]
- clustered p-value: 0.0277552
- BH-FDR q-value: 0.0277552
- bootstrap 95% interval: [0.002200, 0.023565]
- bootstrap fraction positive: 0.9922
- support: 54 concept-positive sequences, 10 concept-negative sequences

Feature 2942 / both catalytic motifs:

- beta_dose_x_concept: 0.0316049
- clustered 95% CI: [0.0153850, 0.0478249]
- clustered p-value: 0.0002411
- BH-FDR q-value: 0.0007233
- bootstrap 95% interval: [0.015005, 0.047627]
- bootstrap fraction positive: 0.9998
- support: 49 concept-positive sequences, 15 concept-negative sequences

Feature 1644 / PS00135:

- beta_dose_x_concept: 0.0206628
- clustered 95% CI: [0.0051296, 0.0361961]
- clustered p-value: 0.0099440
- BH-FDR q-value: 0.0149159
- bootstrap 95% interval: [0.005468, 0.036630]
- bootstrap fraction positive: 0.9960
- support: 49 concept-positive sequences, 15 concept-negative sequences

Feature 727 / IPR001314 was not estimable for the interaction because the
current matched dataset contains 64 concept-positive sequences and 0 genuine
concept-negative sequences for its matched concept.

The matched-control feature was flat and did not reproduce the interaction, but
it also lacks a concept-negative cohort and is not part of the biological
candidate FDR family.

The 3256 statistical interaction criteria passed for the unsigned
motif-specificity magnitude metric, but the full promotion gate remains
inconclusive because the canonical causal artifact does not preserve signed
localized motif effects. For 3256, concept-positive suppression-minus-no-op was
0.1301 and amplification-minus-no-op was 0.4861; concept-negative
suppression-minus-no-op was 0.1038 and amplification-minus-no-op was 0.4114.
The directionality audit showed that this is not a valid signed-directionality
failure: the stored localized metric uses absolute patched-minus-base
true-token log-probability shifts.

Canonical directionality-audit outputs:

- `results/signed_directionality_metric_audit.csv`
- `results/signed_directionality_statistics.csv`
- `results/signed_directionality_by_dose.csv`
- `results/signed_directionality_bootstrap.csv`
- `results/signed_directionality_sequence_slopes.csv`
- `results/signed_directionality_sequence_slope_summary.csv`

Audited metric definitions from `scripts/causal_feature_dose_response.py`:

- `motif_delta_logprob`: mean absolute patched-minus-base true-token
  log-probability shift for the top-k positions with largest absolute changes;
- `nonmotif_delta_logprob`: mean absolute patched-minus-base true-token
  log-probability shift for the remaining valid positions;
- `motif_specificity_score`: `motif_delta_logprob - nonmotif_delta_logprob`, a
  difference of unsigned magnitude summaries.

Therefore `motif_specificity_score` does not preserve the sign of the underlying
log-probability effect. Positive values mean the selected top-k positions changed
by larger magnitude than the remaining positions, not that motif probability
increased.

The existing signed global readout `global_signed_nll_effect = -delta_nll` was
recorded as an ancillary nonlocalized diagnostic only. For feature 3256, its
dose x concept interaction was 0.000791 with clustered 95% CI [-0.000201,
0.001783], p = 0.1162, and whole-sequence bootstrap 95% interval [-0.000220,
0.001750]. This does not provide localized biological directionality evidence.

Directionality gate classification: INCONCLUSIVE. The prior apparent
directionality failure was not scientifically meaningful as a signed biological
failure, but the current artifacts are insufficient to show coherent opposing
signed localized effects under suppression versus amplification.

A targeted signed-localized follow-up script was then added:

- `scripts/causal_feature_3256_signed_directionality.py`

The script preserves the existing additive intervention helper and target-value
grid, emits signed true-token log-probability deltas at each valid token
position, aggregates signed deltas over the same legacy top-k absolute-delta
position set used by the unsigned metric, and labels that position set as
`legacy_topk_abs_delta_not_biological_ps00134_mask`. Inspection confirmed that
the current code does not define a biological PS00134 position mask.

A two-sequence CUDA smoke run wrote temporary `/tmp` outputs and verified that
the new signed path exactly reproduces the legacy unsigned canonical metrics
within tolerance. Maximum absolute differences were approximately:

- motif_delta_logprob: 9.0e-17
- nonmotif_delta_logprob: 6.1e-14
- motif_specificity_score: 6.1e-14
- delta_nll: 8.3e-17
- KL: 9.4e-17

However, the requested precondition that zero dose be an identity/no-op failed:
for the two-sequence smoke, feature-3256 `target_activation=0` rows had maximum
absolute signed motif specificity 0.1057 and maximum absolute delta NLL 0.001813.
This reflects the existing script's use of `dose`/`target_value` as an SAE target
activation, not necessarily an identity-centered latent displacement.

Per the signed-rerun task stop condition, the full absolute-target signed
localized experiment, signed statistics, bootstrap, and signed-directionality
figure were not run.

The identity-centered follow-up uses distinct artifacts and terminology:

- `results/feature_3256_native_activation_summary.csv`
- `results/causal_feature_3256_identity_centered_metadata.json`

The old experiment remains the absolute target-activation design:

    z'_3256 = target_activation

The new implemented design is the identity-centered signed-displacement design:

    z'_3256 = z_native,3256 + displacement

At `displacement=0`, the two-sequence CUDA smoke produced exact no-op behavior
within the `1e-6` gate. The attempted predeclared symmetric grid used
`D = min(native p95, 2 * native p75)`, evaluated before outcome measurement.
Because p75 is 0 and native min is 0, the only globally feasible symmetric
latent-domain-preserving displacement is 0. Nonzero suppression would require
negative realized feature activations at many token positions; no clamping was
applied and no outcome-driven alternative grid was introduced.

No true biological PS00134 positional mask was found in the existing canonical
dataset/schema. The only currently available localized position set remains the
legacy top-k absolute perturbation set, explicitly labeled as not a biological
PS00134 mask.

Directionality gate classification remains INCONCLUSIVE. The project now needs
a prespecified suppression parameterization for sparse nonnegative latents
before it can generate canonical signed-localized evidence.

A PS00134 annotation provenance audit was then added:

- `analysis/ps00134_annotation_provenance.py`
- `results/ps00134_annotation_provenance_audit.csv`
- `results/ps00134_annotation_provenance_audit.json`

The audit inspected the raw master dataset, processed positives, matched
background, clustered dataset, and test split. In each checked dataset,
`has_ps00134` is exactly reproduced by the stored sequence-level rule:

    prosite_ids contains "PS00134"

Checked support:

- raw master dataset: 16,837 PS00134-positive rows, 0 binary mismatches
- processed positives: 16,837 PS00134-positive rows, 0 binary mismatches
- matched background: 0 PS00134-positive rows, 0 binary mismatches
- clustered dataset: 16,837 PS00134-positive rows, 0 binary mismatches
- test split: 2,269 PS00134-positive rows, 0 binary mismatches

However, none of the checked artifacts contains motif start/end coordinates,
matched residues, regex match spans, PROSITE scan output, or equivalent
position-level provenance. The label source is an external UniProt PROSITE
cross-reference string exported by `data/utils/build_s1a_dataset.py`, not a
project-local PS00134 scan whose spans can be replayed.

That internal-only Gate A failed, so no motif coordinates should be inferred
from the stored datasets alone. A follow-up external recovery step then froze
the official PROSITE/ScanProsite definition and scanned the exact stored causal
cohort sequences. New annotation artifacts are:

- `results/ps00134_external_definition.json`
- `results/provenance/ps00134_causal_cohort_sequences.fasta`
- `results/provenance/scanprosite_ps00134_raw.json`
- `results/ps00134_position_annotations.csv`
- `results/ps00134_scanprosite_sequence_audit.csv`
- `results/ps00134_scanprosite_recovery_metadata.json`

Frozen external definition:

- PROSITE accession: PS00134
- name: TRYPSIN_HIS
- entry type: PATTERN
- pattern: `[LIVM]-[ST]-A-[STAG]-H-C`
- pattern version: 1
- associated ProRule: PRU10078
- active-site offset: position 5 within the six-residue match
- ProRule feature: ACT_SITE, note `Charge relay system`
- PS00134 raw text SHA256:
  `3059cf69ad245b950c98c3848cb462634292295ab7c4f1418f00521390d6fe13`
- PRU10078 raw text SHA256:
  `565e81499ef18e5592100dd7c2809864715acd4242719ab582a886c9e93780b4`
- exact submitted 64-sequence FASTA SHA256:
  `eaba3617784664a0f402366f56450f80c32231e5dba7e2ba58ed4c275b39c5cf`
- raw ScanProsite JSON SHA256:
  `2b060925a53675634399739d2db97282c9fdd85605d88fafbc6b232f608271a6`

ScanProsite recovery on the exact causal cohort produced:

- 64 submitted project sequences
- 54 sequences with a recoverable PS00134 pattern match
- 54 total PS00134 matches
- 324 residue-level PS00134 position rows
- 54 PRU10078 active-site rows at pattern offset 5
- ScanProsite confidence `(0)` for all 54 matches
- 0 binary-label-versus-recoverable-pattern mismatches

For this exact 64-sequence cohort, Gate A now passes under the frozen current
ScanProsite PS00134 definition. `has_ps00134` remains the original stored
UniProt cross-reference label, while
`has_recoverable_ps00134_pattern_match` denotes a current official ScanProsite
pattern hit on the exact stored sequence. Native 3256 activation localization at
PS00134 versus non-PS00134 positions has not yet been computed because the
current task explicitly stopped before ProGen-3/SAE execution.

---

## Current confirmatory model

The prespecified fallback analysis is:

    motif_specificity_score ~ dose * concept_positive

The primary coefficient of interest is:

    beta_dose_x_concept

Conceptually:

    S_ij =
        beta_0
        + beta_1 D_ij
        + beta_2 C_i
        + beta_3 (D_ij * C_i)
        + epsilon_ij

where:

- S = motif specificity score
- D = steering dose
- C = matched concept-positive status
- beta_3 = dose × concept interaction

For feature 3256:

A positive beta_3 means that increasing the feature produces a stronger
motif-localized response in PS00134-positive proteins than in
PS00134-negative proteins.

Dose should be centered around the native / no-op condition before fitting.

If dose scales differ materially across independently fitted candidate
features, within-feature standardization may be used for coefficient
comparability, but it must not change the underlying intervention data or
scientific meaning.

---

## Immediate next task

Use recovered PS00134 coordinates in a separately gated native-scaling steering experiment.

The existing causal artifact cannot adjudicate signed localized directionality
because the localized readouts were stored after applying absolute value, and
its `dose`/`target_value` grid is an absolute target-activation design rather
than an identity-centered displacement design.

The identity-centered additive implementation now gives an exact mathematical
identity at displacement 0, but a nonzero globally symmetric negative
displacement is not feasible for feature 3256 without crossing the ReLU/top-k
SAE latent domain because most token activations are zero.

The next step should use `results/ps00134_position_annotations.csv` as the true
PS00134 coordinate mask for the exact 64-sequence causal cohort, then implement
the separately gated multiplicative native-scaling experiment:

    z'_3256 = alpha * z_native,3256

Do not advance to generation unless signed localized directionality is resolved
under this explicit design, or the limitation is explicitly accepted.

---

## Canonical artifacts

The following files are the current primary scientific artifacts when present
in the repository:

### Global feature enrichment

    results/global_feature_enrichment.csv

Contains the global SAE latent × biological annotation enrichment analysis.

### Reconstruction fidelity

    results/reconstruction_evaluation.csv

Contains the corrected held-out reconstruction fidelity evaluation.

### Matched causal dose response

    results/causal_feature_dose_response.csv

Contains sequence-level matched-concept steering observations including the
candidate feature, biological context, steering dose, model disturbance, and
motif-specificity measurements.

When updating a conclusion, inspect the corresponding canonical artifact rather
than relying only on this summary.

If a canonical artifact and this file disagree, investigate the discrepancy and
treat the generated artifact plus producing code/configuration as the primary
quantitative source of truth.

---

## Promotion criteria for feature 3256

Feature 3256 should be promoted from:

    leading concept-selective causal candidate

to something equivalent to:

    statistically supported concept-selective causal candidate

only if the confirmatory evidence collectively supports the claim.

The current gate is:

1. the dose × concept interaction coefficient is positive;
2. its valid 95% confidence interval excludes zero;
3. the interaction survives Benjamini-Hochberg FDR across the estimable
   candidate panel;
4. the sequence-level bootstrap supports the same positive interaction and is
   not driven by a very small number of proteins;
5. suppression and amplification show sensible opposing / directional behavior;
6. the concept-positive advantage remains under matched global disturbance,
   including the existing matched-KL analysis;
7. the negative/control direction does not reproduce the interaction.

Failure of a criterion must be reported explicitly.

Do not silently weaken or redefine the gate after seeing the result.

---

## Current scientific claim

The strongest defensible statement after the directionality audit, targeted
signed-rerun smoke check, and identity-centered displacement preflight is:

> Latent 3256 has a statistically supported positive dose x PS00134-context
> interaction under the prespecified fixed-effects fallback analysis. However,
> it should remain a leading statistically supported candidate rather than a
> formally validated causal controller, because the current canonical causal
> artifact does not preserve signed localized motif effects, the attempted
> targeted signed rerun exposed that the existing target-activation dose grid
> does not make zero dose an identity/no-op condition for feature 3256, and a
> new identity-centered signed-displacement preflight found that nonzero global
> symmetric suppression is not feasible without crossing the sparse nonnegative
> TopKSAE latent domain. A subsequent provenance audit showed that the exact
> current canonical project artifacts: binary labels reproduce from UniProt
> PROSITE cross-reference IDs, but position-level match coordinates were absent.
> A frozen external ScanProsite recovery has now supplied coordinates for the
> exact 64-sequence cohort and exactly reproduces the cohort labels, but no
> signed localized steering experiment has used those coordinates yet.

Do not currently describe 3256 as a formally validated causal controller of
PS00134 biology.

---

## If 3256 passes the confirmatory gate

Advance to the first controlled autoregressive ProGen-3 generation experiment.

Predefined primary comparison:

- unsteered ProGen-3
- feature-3256 steering
- feature-2942 steering
- unrelated/control SAE direction
- matched random direction where practical

Feature 3256 is the primary steering hypothesis.

Feature 2942 is the predefined secondary comparator.

Generation settings such as prompt, temperature, top-k/top-p, sequence length,
sampling seeds, and intervention dose should be controlled and recorded.

Initial biological evaluation should include:

- PS00134 motif recovery / enrichment
- PS00135 recovery
- both-motif recovery
- IPR001314 / S1A-family recovery
- ProGen-3 likelihood / plausibility
- sequence diversity
- duplication / degeneration checks
- similarity to known or training proteins where appropriate

Promising generated sequences can then progress to structural and mechanistic
evaluation, including catalytic-residue context and active-site geometry.

---

## If 3256 does not pass the confirmatory gate

Do not automatically discard the descriptive results.

Instead determine which part of the gate failed:

- weak interaction effect;
- broad uncertainty;
- failure after FDR;
- bootstrap instability;
- dependence on a small number of sequences;
- inconsistent steering direction;
- loss of positive-versus-negative advantage under matched disturbance.

Treat the result according to the failure mode.

Possible scientifically justified next steps may include:

- increasing biological cohort support;
- collecting additional concept-negative sequences;
- improving experimental balance;
- evaluating per-sequence dose-response heterogeneity;
- revisiting candidate selection;
- treating 3256 as an exploratory rather than confirmatory feature.

Do not redesign the intervention or statistical analysis solely to obtain a
significant result.

---

## Unresolved questions

- Does native multiplicative scaling of feature 3256, using the recovered
  ScanProsite PS00134 coordinate mask, show signed localized directionality?
- Which additional suppression parameterizations, if any, are scientifically
  valid for sparse nonnegative TopKSAE latents beyond native multiplicative
  scaling?
- Is a signed localized log-probability artifact required before generation,
  or is the unsigned motif-specificity limitation acceptable for the next gate?
- Is feature 3256's apparent selectivity distributed across many proteins or
  driven by a small responsive subset?
- Does the logit-level effect survive actual autoregressive generation once the
  directionality limitation is resolved or explicitly accepted?
- Does steering enrich the intended biological motif without merely degrading or
  globally perturbing ProGen-3?
- Do generated motif-positive sequences exhibit broader S1A-family consistency?
- Are catalytic residues positioned in structurally plausible contexts?
- Can multiple SAE features eventually provide more precise biological control
  than single-feature steering?

These questions should be addressed in approximately this order rather than
expanding the project laterally before the primary 3256 hypothesis is resolved.