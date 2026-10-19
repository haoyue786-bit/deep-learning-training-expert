---
name: training-diagnosis
description: Diagnose deep-learning instability, unexpected errors or adverse intervention responses from current evidence. Distinguish data, implementation, numerical, optimization and objective problems before selecting a remedy.
---

# Training Diagnosis

Identify the explanation that matters to the next useful decision. Start with [symptoms and differential diagnosis](references/symptoms-and-differential-diagnosis.md) for the observed poor fit, instability, regression or class error. For source/target divergence use [domain shift and negative transfer](references/domain-shift-and-negative-transfer.md); for competing losses or proposed balancing use [multiple losses and gradient conflict](references/multiple-losses-and-gradient-conflict.md). Read the relevant mechanism and counterexamples; do not load every topic or start from an automatic scalar rule.

The shorter [diagnosis and experiment reference](references/diagnosis-and-experiments.md) helps with comparison identity and contradicted interventions. [Optimization practice](references/optimization-practice.md) and [adversarial practice](references/adversarial-practice.md) provide cross-cutting context when those mechanisms are implicated.

Start from actual symptoms and what changed. Ask whether data, label mapping, sampling, loss definition, normalization mode or executed updates can explain the observation before attributing it to a scalar. A corrupt batch and excessive LR can both produce a spike but require different remedies. A finite run after clipping is not evidence of repaired semantics.

Distinguish fit, ranking, calibration and deployment decisions. A higher best threshold alone cannot prove undertraining; compare permitted discrimination, probability-scale and matched-duration evidence. Class-specific error support matters: a negative-only panel can show false positives but not retained recall. A changed map density is not measured target quality.

For multi-loss systems, locate shared parameter support and actual remaining paths. Raw loss sizes, gradient norms, weighted norm ratios and optimizer displacement are different quantities. Opposing gradients can motivate investigation without proving harmful target transfer. If pausing one objective leaves the problem, investigate the active losses and normalization rather than repeatedly adjusting the inactive mechanism.

For a useful leading explanation, name a consequential alternative and a feasible observation that distinguishes them. Do not list every conceivable cause. A simple reversible intervention can already be justified; a contrast is useful when its result can change the decision. If multiple settings change, attribute response to their joint policy.

Return the supported diagnosis, uncertainty and next useful action or measurement in the existing caller format. When contradicted, identify whether the premise, mechanism, numerical value or response timing failed. Change the explanation where necessary rather than treating every result as partial support. The executor, not this Skill, determines whether a proposed action can run.

## Diagnose the concrete symptom pattern

Read [symptoms and differential diagnosis](references/symptoms-and-differential-diagnosis.md) for poor fit, silent implementation faults, overfit versus distribution mismatch, rare-class exposure and forgetting. It connects observed combinations to discriminating observations and plausible remedies rather than mapping a single curve to a fixed scalar.

## Distribution and objective mechanisms

- Source/target divergence, dense target predictions or an inferred class prior: [domain shift and negative transfer](references/domain-shift-and-negative-transfer.md).
- Sensor, ground resolution, crop context, spatial error pattern or a published agricultural recipe: [remote-sensing shift and context](references/remote-sensing-shift-and-context.md). Distinguish appearance, physical support and class semantics; use Fourier and agricultural Swin findings with their specific assumptions.
- Competing losses, a slow task attracting weight, learned uncertainty weights or a negative cosine: [multiple losses and gradient conflict](references/multiple-losses-and-gradient-conflict.md), including GradNorm, PCGrad and likelihood-based uncertainty weighting with their different assumptions.

Use the actual assumptions and counterexamples before borrowing an algorithm or assigning a causal explanation.
