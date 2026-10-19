---
name: training-adjustment
description: Choose or reverse task-specific optimization and training-control changes using a supported diagnosis and observed response. Use for online tuning on the current trajectory; retain settings when a change lacks useful support.
---

# Training Adjustment

Choose a useful executable change or KEEP from current evidence. Use the mechanism-specific knowledge below for the actual control: optimizer state, schedules/numerics, sampling/loss, pseudo-label correction or adversarial dynamics. Read the relevant detailed body when it is not already present in the continuing context. The shorter [optimization practice](references/optimization-practice.md), [adversarial practice](references/adversarial-practice.md) and [diagnosis/response](references/diagnosis-and-experiments.md) references provide cross-cutting context, not replacements for those mechanisms.

## Relate the control to the suspected mechanism

An excessive update can support a lower rate, warmup treatment or clipping, with different implications. Weak train fit can instead reflect insufficient exposure, blocked gradients, bad inputs or excessive regularization. Lower LR can slow memorization without repairing a generalization problem. Weight decay, dropout, gradient penalties and normalization act on different objects; they are not interchangeable knobs.

Check actual effective rates, optimizer groups and schedule phases. Batch and accumulation change exposure, noise and update frequency; reconsider coupled settings when relevant rather than assuming universal linear scaling. Adaptive optimizer history and decay can make a raw gradient ratio a poor predictor of actual motion. Inherited components and new signal-producing components may need different treatment, but the direction requires current evidence.

For adversarial training, decide whether the issue concerns D learning speed, overfitting, class conditioning, model-side pressure or semantic mismatch. LR and update cadence differ in exposure and cost. A smaller adversarial coefficient can reduce motion while leaving errors unchanged; a stronger D can sharpen a wrong signal. Domain confusion alone does not establish good adaptation.

## Make an interpretable, responsive choice

State why this task's state favors the choice over a plausible alternative. Choose a numerical value as a supported bounded step or reference, not a discovered optimum unless actually measured. Do not make parameters different merely to appear personalized. No parameter grid or extra branch is required for an otherwise supported direct adjustment.

Give an observation horizon appropriate to the expected response and actual schedule. Preserve the previous effective intervention through intervening KEEP decisions. Inspect quality, relevant class errors and numerical/source safety separately. If the expected response fails, reconsider cause and timing; do not keep turning the same knob because it moves a proxy predictably.

Reversing a control, rolling back a complete state and continuing with a repaired setting preserve different progress and optimizer histories. Compare plausible recovery with restoration and count replayed or discarded work. A short adverse response does not prove irreversibility. Required safety exits remain binding.

Return only legal advertised controls and actions in the caller's format. An unexposed architecture or objective change is a proposal, not an executable scalar edit. Adoption requires actual execution feedback; an experiment request does not authorize promoting its branch.

## Read the mechanism behind the proposed control

- LR, betas, epsilon, weight decay, freezing or an unexplained gradient-to-motion relation: [optimizer state and interactions](references/optimizer-state-and-control-interactions.md).
- Warmup, decay, batch/accumulation, clipping, precision or a slow post-intervention response: [schedules, batches and stability](references/schedules-batches-and-numerical-stability.md).

Use the specific calculations, examples and competing explanations in these references. A control's name or a prior winning value is insufficient reason to change it.

## Supervision, exposure and generalization

- Class imbalance, sampling, CE reductions, regularization or augmentation: [regularization, sampling and loss design](references/regularization-sampling-and-loss-design.md).
- Teacher confidence, consistency, entropy or automatic weak labels: [pseudo-label confidence and feedback](references/pseudo-label-confidence-and-feedback.md).
- Dominant pseudo-classes, missing rare-class coverage or a proposed log-prior correction: [class bias and pseudo-label correction](references/class-bias-and-pseudo-label-correction.md). Distinguish assignment, student margin, sampling and final decoding; DebiasPL and CCA-LSC change different parts of the loop.

Trace the actual supervision route and exposure before choosing a weight or threshold. The knowledge here does not authorize changing fixed source weights or adding a new method.

For GAN or domain-adversarial changes, read [adversarial objectives and time scales](references/adversarial-objectives-and-timescales.md): objective derivatives, actual D/model update paths, rate versus cadence versus pressure, conditional coding, regularizer differences and counterexamples.
