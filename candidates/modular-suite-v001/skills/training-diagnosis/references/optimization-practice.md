# Conditional optimization practice

Read when choosing an optimization intervention. These are literature-supported
hypotheses, not mandatory trials or universal parameter directions. Use the
actual task's controls, optimizer, component state and permitted validation.

## Unstable or ineffective updates

A sudden loss spike or nonfinite value can reflect a corrupt input, invalid loss,
precision overflow, excessive update, or a schedule transition. Inspect the first
failing operation and affected batches before changing every learning rate.
Where the numerical evidence implicates update size, a lower LR, justified warmup
or clipping can be alternatives with different effects. Warmup addresses the
initial high-rate transition; clipping limits unusually large gradients. Neither
repairs wrong labels or an incorrectly constructed objective.

Check actual effective LR and completed optimizer steps, not just a base setting.
If available, compare parameter-group update displacement with parameter scale;
raw gradient norm does not include adaptive optimizer history or weight decay.
Near-zero initialized parameters make relative-update ratios ill-conditioned.
Repeated clipping can hide an overlarge rate; inspect clipping frequency and
learning response before treating a finite run as repaired.

A short LR-range run can identify an unstable region in that run. Since rate and
training progress change together, it is not a collection of same-state causal
trials. Restore its complete starting state unless adoption is explicitly chosen.
A stable first interval does not establish the best full-run rate.

## Training fit improves while validation worsens

Use like-for-like populations and loss definitions. Increasing held-out loss with
increasing accuracy can reflect a probability-scale tradeoff rather than fewer
correct decisions. Identify the actual primary objective before stopping or
selecting. Training-mode augmentation/dropout also changes measured train loss.

If there is supported generalization deterioration, candidate responses include
stronger appropriate regularization, less exposure, a smaller inherited-component
rate, or correcting sampling/label problems. They address different explanations.
A lower LR may merely slow further memorization; it need not correct insufficient
regularization. Weight decay and dropout are not interchangeable, and their
usefulness may require more than a few updates to measure. Do not diagnose
regularization from a single small-panel fluctuation.

Choose an interpretable allowed contrast when its result could change the next
action. Retain accuracy, loss and class-specific errors separately. Test the
predicted response at the declared horizon; do not select a favorable intermediate
endpoint afterward. A null response should restrict that hypothesis, not imply
that all regularization is useless.

## Inherited and new components

First verify actual loaded tensors, trainability and normalization state. New
components can need a different learning speed from an inherited predictor, but
a different head rate is a hypothesis, not an automatic requirement. The example
DAFormer configuration uses a head multiplier; it does not prove that multiplier
for a trained Cashew decoder or a text classifier.

If new-component fitting dominates useful inherited behavior, advertised options
can include component preparation, differential rates or temporary freezing.
Compare complete states and count preparation. A component that supplies an
auxiliary signal can improve its own fit while harming the task's semantics.
Freezing parameters does not necessarily freeze normalization buffers or dropout.
Inspect the actual mode and forward path. Never reset an inherited decoder
because a generic transfer-learning example assumes a random head.

## Interacting settings and schedules

Batch size changes update count and gradient noise; LR, momentum and regularization
may need reconsideration. Do not assume a linear LR scaling law applies across
optimizers and tasks. A shorter total schedule may change the entire prefix,
including warmup fraction and decay; equal epoch labels do not guarantee matched
exposure or effective rates. Inspect actual parameter groups and exclusions:
AdamW decay and an L2 term inside an adaptive optimizer are different operations.

For a local attribution, hold the relevant other factors fixed. For a whole-method
comparison, allow consequential nuisance settings to be retuned under comparable
budgets. State whether the observed effect belongs to one scalar or a joint policy.
No broad sweep is required simply because several controls exist.

## Sources and limits

- Google Research, [Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook): initial configuration, scientific/nuisance variables, instability, warmup, clipping and batch-size FAQ. Practical guidance, not a universal convergence proof.
- Stanford, [CS231n Part 3](https://cs231n.github.io/neural-networks-3/): sanity checks, learning curves and update scales. Teaching examples are not validated thresholds for adaptive multi-loss training.
- Smith (2018), [A Disciplined Approach](https://arxiv.org/abs/1803.09820): coupled LR/momentum/batch/regularization and range tests. Its SGD experiments do not supply an AdamW/FADA prescription.
- Microsoft, [Swin implementation](https://github.com/microsoft/Swin-Transformer/tree/f82860bfb5225915aca09c3227159ee9e1df874d): actual scaling, optimizer groups, warmup and resume paths depend on configuration and entry point.
- Hoyer et al., [DAFormer code](https://github.com/lhoyer/DAFormer/tree/e6f6e041e7477b642106c134c1715fc934ccd210): one conditional example of inherited backbone/new head treatment, with different data and architecture from Cashew.
