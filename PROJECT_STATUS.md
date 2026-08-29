# PROJECT_STATUS.md

Last updated: 2026-08-29

## Current stage

Causal feature validation / confirmatory statistics.

## Completed

- TopK SAE trained on ProGen-3 layer-6 activations.
- Global feature enrichment complete.
- Canonical enrichment output generated.
- Full SAE reconstruction evaluated.
- Identity patch validated.
- Full SAE reconstruction found behaviorally non-faithful.
- Residual interpolation and matched-noise controls completed.
- Residual-preserving additive intervention implemented and identity-validated.
- Additive causal dose-response sweep completed.
- Matched feature/concept causal sweep completed.

## Current candidate hierarchy

- 3256: lead candidate, matched to PS00134.
- 2942: secondary comparator, matched to both catalytic motifs.
- 1644: weaker PS00135 candidate.
- 727: strong directional IPR001314 feature, but current experiment lacks a
  usable matched concept-negative comparison.
- feature 1: negative/control feature.

## Current evidence

At matched KL ~0.025:

- feature 3256:
  - concept-positive specificity ~0.782
  - concept-negative specificity ~0.447
  - gap ~0.335

- feature 2942:
  - concept-positive specificity ~0.829
  - concept-negative specificity ~0.580
  - gap ~0.249

3256 is therefore the current lead based on context selectivity, not raw
steering magnitude.

## Important negative result

Full SAE reconstruction is not sufficiently faithful as an intervention
interface.

Corrected held-out reconstruction evaluation showed approximately:
- base NLL ~2.145
- patched NLL ~2.667
- delta NLL ~0.522
- relative degradation ~36%

Residual interpolation showed that restoring a small high-leverage residual
component rapidly restores downstream behavior.

Therefore causal steering uses the additive residual-preserving intervention,
not full reconstruction replacement.

## Current statistical status

Attempts to estimate the random-effects structure using mixed models produced
singular covariance / boundary behavior.

Do not interpret those failed fits as confirmatory evidence.

Next confirmatory analysis:
- fixed-effects motif-specificity ~ dose * concept interaction
- sequence-clustered robust standard errors
- sequence-level bootstrap
- BH-FDR across estimable candidate interactions

## Current gate

Do not promote 3256 to a validated concept-specific controller until the
confirmatory interaction analysis is complete.

If 3256 clears the statistical gate, the next major experiment is
autoregressive ProGen-3 generation using:
- 3256 as primary steering feature
- 2942 as predefined secondary comparator
- unsteered and matched control directions
