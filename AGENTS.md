# AGENTS.md

## Project

This repository investigates mechanistic interpretability and inference-time
control of ProGen-3 using sparse autoencoders (SAEs).

The current scientific workflow is approximately:

ProGen-3 hidden-state extraction
→ SAE training
→ biological feature enrichment
→ reconstruction-fidelity diagnostics
→ residual analysis
→ additive causal intervention
→ concept-specific causal validation
→ autoregressive feature steering
→ biological evaluation

The objective is not merely to produce working software. Changes must preserve
the scientific meaning, reproducibility, controls, and interpretability of the
experiments.

Before modifying analysis or experiment logic, inspect `PROJECT_STATUS.md` if
it exists. `AGENTS.md` contains stable project rules; `PROJECT_STATUS.md`
contains the current experimental state and should not be treated as a
permanent methodological specification.

## Repository structure

- `analysis/`: statistical analysis and feature interpretation.
- `scripts/`: executable reconstruction, intervention, steering, and evaluation workflows.
- `topk_sae/`: SAE implementation and related utilities.
- `tests/`: regression and scientific-invariant tests.
- `data/`: source datasets and intermediate analysis inputs.
- `results/`: generated scientific outputs and canonical result tables.
- `progen3/`: upstream ProGen-3 dependency / Git submodule.

Avoid modifying the `progen3/` submodule unless the task explicitly requires it.
Prefer implementing project-specific behavior outside the upstream dependency.

Do not accidentally commit large checkpoints, activation tensors, embedding
memmaps, caches, or temporary GPU outputs. Inspect file size and Git status
before adding generated artifacts.

## Scientific invariants

1. Do not silently alter the mathematical definition of an intervention,
   reconstruction, steering operation, normalization, biological label, or
   evaluation metric.

2. The canonical causal SAE intervention is conceptually:

       delta_z = z_steered - z
       delta_h = W_dec @ delta_z
       h_steered = h + delta_h

   Exact tensor orientation depends on the implementation.

   Preserve these semantics unless explicitly asked to investigate an
   alternative.

3. For additive steering:
   - preserve the original ProGen-3 hidden state;
   - decode only the latent displacement;
   - do not add decoder bias to a displacement;
   - do not reinsert an SAE normalization mean into a displacement;
   - do not replace the full hidden state with the SAE reconstruction.

4. Full SAE reconstruction patching is a diagnostic experiment, not the
   default steering mechanism.

       h_reconstructed = SAE(h)

   and

       h_steered = h + W_dec @ (z_steered - z)

   are scientifically different operations and must never be conflated.

5. Identity/no-op interventions must reproduce the original hidden state and
   model behavior within numerical tolerance:

       z_steered == z
       => delta_z == 0
       => delta_h == 0
       => h_steered == h

   A zero-dose/additive identity test should be treated as a required
   regression invariant.

6. Reconstruction experiments must clearly distinguish, when applicable:
   - untouched ProGen-3 baseline
   - identity patch
   - full SAE reconstruction
   - residual interpolation
   - matched random/noise controls
   - additive feature intervention

7. Never treat statistical association as causal evidence.

   Preserve the distinction among:
   - feature enrichment / detection
   - causal sensitivity
   - concept-selective causal effects
   - generative steering
   - biological/function validation

8. Preserve the distinction between biological concepts including:
   - IPR001314
   - PS00134
   - PS00135
   - both catalytic motifs
   - concept-positive vs concept-negative cohorts
   - baseline feature-active vs feature-inactive cohorts
   - matched/random controls

9. "Concept-positive" is feature-specific. For example, PS00134-positive is
   appropriate for a PS00134-matched feature and must not be silently replaced
   with generic S1A membership.

10. Global perturbation metrics such as delta NLL or KL do not by themselves
    establish biological specificity. Concept-specific claims require the
    relevant localized/context-matched analysis.

11. Do not fabricate, infer, or backfill missing measurements, annotations,
    statistical results, support counts, or dashboard values.

12. Missing biological annotation must not automatically be interpreted as a
    confirmed biological negative unless the data contract explicitly defines
    it that way.

## Statistical analysis

Prefer effect sizes and uncertainty alongside significance.

For multiple-feature hypothesis testing, preserve the defined multiple-testing
correction, normally Benjamini-Hochberg FDR.

When observations contain repeated doses from the same sequence, do not treat
rows as independent.

If a mixed-effects model:
- fails to converge,
- has singular random-effects covariance,
- places the MLE on the parameter-space boundary,
- or otherwise has an unidentified variance structure,

report the model as numerically unstable/inconclusive.

Do not interpret non-convergence as evidence for absence of a biological
effect, and do not force a more complex model to fit.

Use a simpler prespecified fallback when appropriate, such as:
- fixed-effects dose × concept interaction,
- sequence-clustered robust standard errors,
- sequence-level bootstrap,
- FDR across estimable candidate tests.

Bootstrap repeated-measures experiments by sequence, retaining all relevant
observations for each resampled sequence. Do not bootstrap individual rows as
if they were independent proteins.

Raw ratio metrics such as:

    specificity / KL
    specificity / abs(delta_NLL)

can become unstable near the no-op regime. Do not rank features solely by such
ratios without a denominator floor, matched-disturbance analysis, or another
explicitly justified procedure.

Clearly distinguish:
- exploratory evidence
- candidate-selection evidence
- descriptive causal evidence
- confirmatory statistics
- causal intervention evidence
- generative evidence
- biological validation

Do not strengthen the wording of a scientific conclusion unless the underlying
analysis supports it.

## Experimental reproducibility

- Preserve existing random seeds unless changing them is part of the experiment.
- Record important hyperparameters, model/checkpoint identifiers, feature IDs,
  intervention definitions, doses, dataset splits, and relevant software
  configuration in outputs or configs.
- Prefer deterministic cohort-selection logic where scientifically appropriate.
- Preserve support counts for sparse cohorts rather than silently filling or
  merging them.
- Do not silently substitute sequences when a requested experimental cell lacks
  sufficient support.
- Do not manually edit generated scientific result files to obtain or repair a
  scientific result.
- Regenerate outputs using the corresponding script whenever possible.
- Do not overwrite canonical/raw results unless explicitly requested.
- Prefer timestamped, versioned, or clearly named outputs for new experiments.
- Preserve sufficient provenance to determine which script/config/checkpoint
  produced each important result.

Canonical CSVs should remain machine-readable. Do not change column semantics
or units without explicitly documenting the change.

## GPU / environment

The project uses a CUDA-enabled ProGen-3 runtime.

Before a GPU experiment, verify the active Python environment rather than
assuming GPU availability from `nvidia-smi` alone.

Useful checks include:

    which python
    python --version

and:

    python - <<'PY'
    import torch
    print("torch:", torch.__version__)
    print("torch CUDA:", torch.version.cuda)
    print("CUDA available:", torch.cuda.is_available())
    print("device count:", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device:", torch.cuda.get_device_name(0))
    PY

The known working project environment has used CUDA-enabled PyTorch; do not
replace package versions casually when diagnosing an experiment.

Respect `CUDA_VISIBLE_DEVICES`. A process may correctly report one GPU when
the server contains several GPUs because only one physical device has been
exposed to that process.

Before launching an expensive job:

1. inspect the command, dataset, checkpoint, device, and output path;
2. verify imports and model loading;
3. run the smallest meaningful smoke test;
4. verify expected tensor shapes and scientific invariants;
5. verify that the output schema is correct;
6. only then run the full experiment.

Do not launch multiple large GPU jobs concurrently unless explicitly requested.

Do not rerun an expensive completed experiment merely to confirm that a file
exists when its saved output is sufficient for the requested task.

## Coding workflow

For a nontrivial task:

1. inspect the relevant implementation, tests, configs, and result schema;
2. identify the scientific invariant(s) affected;
3. form a short implementation plan;
4. make the smallest coherent change;
5. run the narrowest relevant test or smoke check;
6. inspect the resulting diff;
7. inspect generated output when applicable;
8. report exactly what changed and what was verified.

Unless the user explicitly asks only for a plan, do not stop after planning
when the requested implementation and verification can be performed.

Do not refactor unrelated code while implementing a scientific change.

Prefer extending an existing tested pathway over creating a second
inconsistent implementation of the same mathematical operation.

Avoid duplicating intervention or normalization math across scripts. Shared
scientific operations should use a common tested helper when practical.

## Testing

Run the narrowest relevant test first.

When changing:

- feature enrichment:
  test statistical edge cases, constant arrays, missing/degenerate labels,
  annotation matching, and FDR behavior.

- reconstruction:
  verify identity patch behavior and distinguish full SAE reconstruction from
  additive steering.

- additive steering:
  verify exact zero-delta/no-op behavior before any biological experiment.

- normalization:
  separately test full-state reconstruction transforms and displacement
  transforms; offsets/biases that belong to reconstruction must not leak into
  delta decoding.

- causal dose response:
  verify both steering directions, cohort assignment, support counts, matched
  concepts, controls, and output schema.

- statistical confirmation:
  verify repeated-measures handling, sequence-level resampling, coefficient
  extraction, and multiple-testing correction.

- autoregressive generation:
  verify that the steering hook is applied at the intended layer/token step,
  zero steering reproduces the unsteered sampling path when deterministic
  conditions permit, and sampling settings are recorded.

- plotting/dashboard code:
  verify that every plotted value comes from real result files and that axis
  labels/units match the underlying metrics.

If a focused test is blocked by an environment or optional dependency issue,
report the exact failure. Do not represent an unexecuted test as passing.

If a full test suite exists, run it after focused tests pass when practical.

## Results and interpretation

Treat generated scientific outputs as data, not UI decoration.

Before summarizing an experiment:
- inspect the actual output file;
- report sample/support sizes where relevant;
- distinguish means, medians, maxima, slopes, interaction coefficients, and
  matched-disturbance comparisons;
- do not replace a failed confirmatory analysis with a favorable descriptive
  statistic without clearly labeling it exploratory/descriptive.

If an analysis yields an unexpected negative result, preserve and report it.
Do not optimize code or parameters specifically to remove an inconvenient
result unless a new experiment is explicitly justified.

## Project status maintenance

`PROJECT_STATUS.md` is the canonical human-readable record of the current
scientific state of the project.

After completing a task that materially changes the scientific state of the
project, update `PROJECT_STATUS.md` before finishing the task.

Examples of status-changing events include:
- a new experiment is successfully completed;
- a scientific control passes or fails;
- a candidate feature is promoted or demoted;
- a statistical analysis changes the strength of a conclusion;
- a previously suspected implementation bug is confirmed or ruled out;
- a new canonical result artifact is generated;
- the next experimental gate changes.

A failed run may still be scientifically informative. Record failures that
change the interpretation, invalidate a planned analysis, reveal insufficient
support, expose an implementation issue, or alter the next experimental step. 

Do not record routine infrastructure failures such as transient CUDA OOM,
network interruption, or missing optional packages unless they affect the
scientific interpretation.

Do NOT update `PROJECT_STATUS.md` for:
- formatting-only changes;
- refactors with no scientific effect;
- failed runs that provide no new scientific information;
- speculative interpretations not supported by generated results.

When updating `PROJECT_STATUS.md`:

1. inspect the actual result artifact(s);
2. preserve prior important negative results;
3. update the "Last updated" date;
4. update the current stage if appropriate;
5. move newly completed work into "Completed";
6. update current evidence using actual generated values;
7. update candidate rankings only if supported by the experiment;
8. explicitly record failed or inconclusive confirmatory analyses;
9. update the next gate / next experiment;
10. never represent a planned experiment as completed.

Keep the file concise. It should describe the current scientific state, not
serve as a chronological laboratory notebook.

Before reporting task completion, state whether `PROJECT_STATUS.md` required an
update and whether it was updated.

## Scientific source-of-truth hierarchy

When sources disagree, use the following order of authority:

1. raw/canonical generated result artifacts;
2. the code/configuration that produced those artifacts;
3. PROJECT_STATUS.md;
4. EXPERIMENT_LOG.md, if present;
5. comments, README text, issue descriptions, or prior conversational context.

Do not preserve a conclusion in PROJECT_STATUS.md if newly generated canonical
results contradict it. Instead, inspect the discrepancy and update the status
accordingly.

Never infer a scientific result solely from filenames, comments, or prior
summaries when the underlying result artifact is available.


## Git / generated artifacts

The `progen3/` directory is a Git submodule. Changes inside it require their
own submodule commit and a parent-repository pointer update.

Some generated directories may be intentionally ignored by Git.

Before claiming that a result was committed or pushed, verify with:

    git status
    git ls-files <path>

Use `git add -f` for an ignored scientific artifact only when that artifact is
intentionally meant to be version-controlled.

Do not force-add large intermediate arrays, embeddings, model weights, caches,
or temporary results.

Do not rewrite Git history, force-push, delete branches, or modify remotes
unless explicitly requested.

## Final response after coding

Report:
- files changed
- behavior changed
- tests/checks actually run
- outputs actually generated
- scientific assumptions or invariants affected
- remaining uncertainties, warnings, or failures

If an experiment was not actually run, explicitly say so.

Do not claim that a scientific gate has passed unless the required analysis was
successfully executed and its result supports that conclusion.
