---
name: deep-learning-training-expert
description: Manage real deep-learning training through task-specific initialization, diagnosis, adjustment, experiments, stopping and checkpoint selection. Use this coordinating entry for an overall run; individual professional skills can handle narrower questions.
---

# Deep Learning Training Expert

Improve the deployed model using current evidence and executable controls. Use one scientific owner; initialization, diagnosis, adjustment, stopping and selection are capabilities, not eight mandatory reviews or extra calls. Defaults are [gpt-6.1-sol / medium](scientific_defaults.json), telemetry each epoch, evaluation/checkpoint every five and decisions every ten. Explicit job contracts and user review boundaries take precedence.

## Read the current brief, not the whole archive

Use the host's [expert brief and curves](references/expert-review.md): task/component state, current controls, current metrics, the whole observed trajectory, actual intervention markers, unresolved questions and relevant knowledge excerpts. The brief must stand alone after a fresh model session. Do not depend on remembered numbers or resend complete manuals. A graph path is not an image attachment; exact values and metric definitions accompany the curve.

Focus on the question that changes the next decision:

- Target quality: fixed-decoder precision/recall/F1, FP/FN and class support; changed labels or evaluation populations break comparability.
- Learning and retention: train losses with their actual definitions, source/class performance and the learning-rate schedule. Source quality is not target quality.
- Gradients and updates: classification/local/global adversarial norms, weighted contributions and directions when measured. Sampled gradients, actual optimizer displacement and clipping are distinct. Missing measurements are unknown, not zero.
- Spatial behavior: prediction area/fraction, prior support and their spatial agreement on stated masks. Inspect relevant auxiliary exposure and discriminator behavior.
- Response to previous actions: what actually changed, when, what followed and the strongest plausible alternative. A curve or scalar correlation alone does not establish a cause.

When focus-class false positives decrease, inspect their destination classes and
new false negatives. A former false positive becoming a different wrong class
improves the binary count without repairing that class confusion. Use paired
errors and full supported class confusion alongside the declared main objective.
Give trend statements their measured epoch range. Equal first/latest values
do not mean a metric stayed constant; check intermediate observations before
saying "always" or "never." Unmeasured epochs remain unknown.

For Cashew, inspect the approved prior before initialization: local support, regional evidence, confidence/unknowns and score scale. Form a provisional view of whether substantial cashew is plausible. High-support regions can legitimately have substantial predictions; a weak regional prior is neither calibrated crop coverage nor a quota. Same-mask comparison matters. Read the [project profile](task_profiles/cashew_fada/profile.md) for facts absent from the host's verified card; use [data roles](task_profiles/cashew_fada/data-labels-and-evidence-roles.md) when provenance is uncertain.

## Decide and explain consequential changes

KEEP or reference initialization is valid. Unexplained differences are not personalization. Prefer delta-only output: the worker copies unchanged values from verified controls. Each proposed change identifies current evidence, mechanism, direction/magnitude, expected response, alternative and reversal condition. Distinguish missing prerequisites from an unmeasured response: applicable mechanism knowledge and current task evidence can justify a bounded first intervention without a previous same-run success. Ordinary legal adjustments within an authorized training run do not require separate approval. Extra branches or methods outside that scope do. Do not claim an untested change is proven effective.

When a useful authorized supervision route remains inactive, or KEEP/STOP relies on lack of evidence from a treatment never tried, read [first use of auxiliary supervision](references/auxiliary-first-use.md). Check the pool's suitability and actual exposure; zero weight is an off state, not evidence of ineffectiveness. Consider a bounded, recoverable online trial and assess its response before treating the available remedy as exhausted. This does not mandate activating every pool or running to the epoch cap.

Review responses at a useful horizon. Do not repeatedly tune a proxy whose semantic hypothesis failed. Restoring a full state differs from reversing one control. STOP judges further benefit; SELECT chooses an existing eligible checkpoint. Neither defaults to the latest or a favorable small-panel fluctuation.

## Expand knowledge only when needed

Use the relevant professional Skill or one specific reference section: `training-initialization`, `training-diagnosis`, `training-adjustment`, `training-experiment`, `training-stop`, `training-checkpoint-selection`; use `training-fact-reconstruction` for conflicting records. The [knowledge map](references/knowledge-use-and-navigation.md) and [detailed guidance](references/scientific-judgment.md) are on-demand resources, not default prompt bodies. Missing consequential measurements should be requested, not invented. General recipes require matching prerequisites.

Retrieve a few conditional lessons through the authorized [project store](../training-experience-update/references/project-store.md). Preserve contrary outcomes, development-use counts and costs. Years/seeds of one tile are not independent replication. Use `training-experience-update` after real outcomes; project records do not belong inside the generic Skill.

At a new decision, match knowledge to the current uncertainty and training stage.
Do not keep an initialization excerpt as the only guidance for later opponent
dynamics, semantic regression or selection. A tools-disabled host must supply
the needed bodies or fulfill an explicit information request before a dependent
decision. A small initial knowledge selection is not a permanent reading limit.

## Execution boundaries

For new trainer integrations, use the [fixed execution template](references/execution-template.md)
and its reusable program. The executable part contains only action, parameter
changes and selected epoch; write scientific evidence and reasoning in the
separate free-form training record. The host records measured facts and binds
artifact identities. Do not pause because of prose formatting or an ordinary
future-observation request. Only an explicit HOLD blocks for scientific reasons.

Use authorized validation feedback for decisions, never its labels in loss; keep sealed tests outside tuning. Preserve support, state/data/decoder identities and timing. For Cashew, supplementary human samples are training data; do not manufacture a human holdout. Small-panel or fitted-threshold results are development evidence.

The worker owns execution and recovery. [Control-change checks](references/execution-and-recovery.md#control-change-evidence) bind values to evidence but do not prove scientific validity. Respect per-decision human approval when requested. Preserve rejected answers and costs; reconcile uncertain calls before retrying. Unsupported methods remain proposals. Read [trial execution](references/algorithm-trial-bench.md) or [recovery](references/execution-and-recovery.md) only when invoking or integrating those operations.

The caller's response schema governs the answer; the evidence brief is not an
output template. When the caller requires JSON, return that exact object without
markdown or extra fields. Use the supplied measured values for temporal claims.
For selection, distinguish the current review checkpoint from the selected
candidate. Missing candidate identity is a host evidence defect, not a reason
to prefer the latest checkpoint. Request the missing identity if it can affect
selection. Future observations do not block a supported current action; mark
only indispensable missing evidence as a prerequisite.
