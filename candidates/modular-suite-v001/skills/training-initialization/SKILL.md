---
name: training-initialization
description: Choose task-specific starting states and initial optimization settings for deep-learning training. Use for fresh starts, transfer learning or a diagnosed initialization problem; exact continuation requires restoring its existing state.
---

# Training Initialization

When a permitted starting-state comparison has real validation predictions and labels, use confusion, Precision/Recall/F1, support and class-pair errors to locate the existing weakness. Map density or a fitted threshold alone does not establish initialization failure. Respect finite development-use history, keep validation labels out of training loss and retrieve relevant explicit same-project experience. Record actual initialization tests and outcomes in the authorized [project store](../training-experience-update/references/project-store.md); unclear project identity/path permits no write.

Choose a useful complete starting state, rather than inventing different scalar values for every dataset. Read the supplied task, data permissions and executable controls. For exact continuation, identify and restore the declared full state; changing initializers is a different treatment.

## Establish what actually needs learning

Use [functional starts and transfer](references/functional-starts-and-transfer.md) for the starting function, head bias, inherited versus new components and component preparation. Use [normalization and domain mixtures](references/normalization-freezing-and-domain-mixtures.md) when freezing, extra forwards or buffers matter, and [model-family mechanisms](references/transformer-and-segmentation-transfer.md) for Transformer/dense-prediction transfer. Read the relevant body, not every topic at every review. The shorter [component-state reference](references/initialization.md) covers cross-cutting start identity and probe interpretation.

Identify inherited parameters, new components, active loss paths, optimizer groups, normalization buffers and schedules. A pretrained encoder does not imply a trained head; a loaded checkpoint does not imply the head is random. Inspect loading and dependency evidence when this distinction changes the choice.

Read [optimization mechanisms](references/optimization-practice.md) when choosing rates, warmup, regularization or freezing. Explain how component state and current data make the proposed direction plausible. Differential rates may preserve inherited behavior while allowing a new component to learn, but the direction and multiplier require task evidence. A trained decoder must not inherit a random-head recipe automatically.

## Choose the discriminating evidence

If initial loss or output is surprising, compare its expectation under the actual loss, sampling, weighting and inherited predictor. Uniform log(C) is not an invariant for a pretrained or weighted classifier. A prior-frequency bias can be reasonable for a genuinely new head under compatible sampling; it can overwrite useful learned calibration in an inherited head.

Check actual effective updates when gradient scale is being used to choose rates. Adaptive moments, clipping, decay and near-zero parameter scales complicate gradient-to-update comparisons. An LR-range test changes training state while changing rate; it is not a set of same-state trials. Count preparation and restore the complete start unless adoption is explicit.

A cached-batch fit can test implementation or component capacity; it does not establish generalization or a semantically useful adversarial signal. Where the executor permits preparation, distinguish an adopted warm start from a discarded probe. State which uncertainty the preparation can resolve and what adverse response would reject it.

For example, a genuinely new head failing to use useful fixed features can support head preparation or differentiated rates. An inherited trained head with new target false positives instead calls for preserving its useful behavior while locating the changed supervision/alignment path. The same tile-distance statistic does not make those starts equivalent. Choose the relevant component or exposure treatment, not just a different common LR.

## Return an executable initialization

Use the caller's actual output format and legal controls, including unchanged values where requested. Identify the complete starting artifact and proposed group/schedule settings, their task-specific rationale, expected early response and relevant uncertainty. Unsupported preparation remains a proposal. Completion means an actionable supported starting choice, not proof of optimality; the executor owns admission and adoption.

Check the returned parameter vector against the stated reference before submission. A changed initial value needs current evidence plus an applicable mechanism, direction/magnitude rationale, expected response and rejection condition. It need not have already won a same-tile trial, but a generic possibility alone is insufficient. If your own reasoning says the change lacks support, retain that reference value and separate any useful diagnostic proposal. Do not leave an unexplained numerical change inside an otherwise conservative initialization.

## Read the relevant professional knowledge

- New head, inherited predictor, output bias or residual outlet: [functional starts and transfer](references/functional-starts-and-transfer.md). Includes fan scaling, prior-aware bias, sampled/weighted class distributions, component preparation and concrete failure distinctions.
- A freeze or extra forward changes predictions: [normalization and domain mixtures](references/normalization-freezing-and-domain-mixtures.md). Distinguish parameters, buffers, modes and opponent-fitting exposure.
- Transformer, feature-pyramid or segmentation transfer: [model-family mechanisms](references/transformer-and-segmentation-transfer.md). Relate component readiness, actual input semantics, crop exposure and context to parameter groups; use the DAFormer/MIC examples with their specific assumptions.

These references contain operational knowledge and worked illustrations, not mandatory preparation steps. Read the topic that changes the current choice.
