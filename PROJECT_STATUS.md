# PROJECT_STATUS.md

Last updated: 2026-08-29

## Current stage

Directionality metric audit complete; signed localized evidence unavailable in
the current canonical causal artifact.

The SAE training, biological feature-enrichment, reconstruction diagnostics,
residual analysis, additive intervention machinery, and prespecified
fixed-effects confirmatory statistics are complete enough for the current
scientific question.

The immediate question is no longer whether the intervention code works or
whether the fixed-effects dose x concept interaction is positive. The
directionality audit established that the stored localized motif-specificity
readouts are unsigned magnitude summaries.

The current question is:

> Should the project collect or recover a signed localized log-probability
> readout before deciding whether latent 3256 is ready for autoregressive
> generation?

Do not advance to autoregressive generation on the basis of the current
localized directionality evidence alone.

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
artifact does not contain a signed localized biological readout.

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

Decide whether a signed localized log-probability readout is required before
autoregressive generation.

The existing causal artifact cannot adjudicate signed localized directionality
because the localized readouts were stored after applying absolute value. If the
directionality gate remains mandatory, the next scientifically appropriate step
is to recover or regenerate signed localized token-level summaries with clear
provenance, while preserving the frozen additive intervention semantics and
existing cohort definitions.

Do not advance to generation unless the directionality limitation is explicitly
accepted or resolved with a signed localized artifact.

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

The strongest defensible statement after the directionality audit is:

> Latent 3256 has a statistically supported positive dose x PS00134-context
> interaction under the prespecified fixed-effects fallback analysis. However,
> it should remain a leading statistically supported candidate rather than a
> formally validated causal controller, because the current canonical causal
> artifact does not preserve signed localized motif effects and therefore cannot
> establish coherent opposing biological direction under suppression versus
> amplification.

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

- Is a signed localized log-probability artifact required before generation,
  or is the unsigned motif-specificity limitation acceptable for the next gate?
- If required, can signed localized token-level summaries be recovered without
  changing the frozen additive intervention semantics or cohort definitions?
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