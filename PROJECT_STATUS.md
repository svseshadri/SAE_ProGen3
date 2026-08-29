# PROJECT_STATUS.md

Last updated: 2026-08-29

## Current stage

Confirmatory statistical validation of causal feature selectivity.

The SAE training, biological feature-enrichment, reconstruction diagnostics,
residual analysis, and additive intervention machinery are complete enough for
the current scientific question.

The immediate question is no longer whether the intervention code works.

The current question is:

> Does latent 3256 show a statistically robust dose-dependent effect that is
> stronger in the matched PS00134-positive biological context than in the
> PS00134-negative context?

Do not broaden or redesign the intervention experiment until this confirmatory
question is resolved.

---

## Current scientific objective

Determine whether an interpretable SAE feature identified in ProGen-3 can act
as a biologically selective causal control direction rather than merely:

- correlating with a biological annotation;
- perturbing ProGen-3 globally;
- or producing a large but nonselective steering effect.

The current lead hypothesis is that latent 3256 is selectively associated with
PS00134-related biology.

If this hypothesis survives confirmatory statistics, the next major experiment
is autoregressive ProGen-3 generation under additive SAE steering.

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

3256 is the strongest context-selective causal candidate among the tested
features.

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
random-effects structure.

That model failed with a singular matrix.

A reduced mixed-effects model retaining the primary dose × concept term and a
sequence random intercept was then attempted.

The reduced model produced warnings including:

- singular random-effects covariance;
- MLE on the boundary of the parameter space;
- unstable / unidentified random-effect variance.

These fits must be treated as numerically unstable and inconclusive.

They are not valid confirmatory evidence for or against the biological
interaction.

Do not continue simplifying or modifying the random-effects structure merely
to obtain a favorable result.

The mixed-effects pathway is considered exhausted for the current dataset
unless new data materially change identifiability.

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

Run the confirmatory analysis on the matched causal dataset.

Primary input:

    results/causal_feature_dose_response.csv

For each estimable biological candidate:

1. confirm cohort support and required columns;
2. center dose around the no-op / native intervention point;
3. fit:

       motif_specificity_score ~ dose * concept_positive

4. estimate uncertainty using sequence-clustered robust standard errors;
5. extract:
   - dose coefficient
   - concept coefficient
   - dose × concept coefficient
   - interaction standard error
   - interaction 95% confidence interval
   - interaction p-value
   - number of unique concept-positive sequences
   - number of unique concept-negative sequences
6. bootstrap whole sequence IDs with replacement;
7. retain all repeated dose observations belonging to each resampled sequence;
8. refit the same interaction model for each bootstrap replicate;
9. report:
   - bootstrap median interaction
   - bootstrap 2.5th percentile
   - bootstrap 97.5th percentile
   - fraction of bootstrap interactions > 0
10. apply Benjamini-Hochberg FDR across estimable candidate interaction tests;
11. report the FDR-adjusted q-value for each estimable feature.

Do not bootstrap individual dose rows as if they were independent proteins.

Do not force a concept-selectivity test for feature 727 if the required
concept-negative support is absent.

Do not alter the additive intervention implementation while performing this
analysis.

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

The strongest defensible statement before completion of the confirmatory
analysis is:

> Latent 3256 is the leading concept-selective causal candidate among the tested
> SAE features. It retains a clear positive-versus-negative motif-specificity
> advantage under matched model disturbance, while the control remains flat and
> latent 2942 behaves as a stronger but less selective steering direction.
> Formal statistical confirmation of the dose × concept interaction remains
> pending.

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

- Does feature 3256 survive sequence-aware confirmatory uncertainty estimation?
- Is its apparent selectivity distributed across many proteins or driven by a
  small responsive subset?
- Does the positive-versus-negative advantage remain after FDR correction?
- Does the logit-level effect survive actual autoregressive generation?
- Does steering enrich the intended biological motif without merely degrading or
  globally perturbing ProGen-3?
- Do generated motif-positive sequences exhibit broader S1A-family consistency?
- Are catalytic residues positioned in structurally plausible contexts?
- Can multiple SAE features eventually provide more precise biological control
  than single-feature steering?

These questions should be addressed in approximately this order rather than
expanding the project laterally before the primary 3256 hypothesis is resolved.