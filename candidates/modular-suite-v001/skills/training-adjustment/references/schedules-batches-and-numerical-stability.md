# Schedules, batch exposure and numerical stability

Read when choosing warmup/decay, changing batch or accumulation, interpreting
slow response, or separating numerical instability from bad task learning.

## Translate the schedule into actual updates

An epoch names data exposure, not a universal amount of optimization. Determine
microbatches, accumulation, distributed averaging, final partial windows and
whether the scheduler advances per microbatch, successful optimizer step or
epoch. A skipped precision-scaled step is not a completed parameter update.
Check the caller's actual loop rather than inferring every count from a formula.

Effective batch is often `microbatch * accumulation * data_parallel_workers`,
but this describes intended aggregation only when samples and reductions match.
Different source/target mixtures, ignored pixels, dynamic batches and uneven
tails change the effective objective. A larger aggregate does not guarantee
equivalence when stochastic operations or normalization differ.

### A worked schedule mistake

Suppose a loader supplies 125 microbatches per epoch and accumulation is four.
A loop that updates only on divisible counts performs 31 complete updates and
may discard the final microbatch; a loop that flushes the remainder performs
32 with a partial final window. A schedule still stepped 125 times has another
time scale. Inspect the actual loss normalization for that partial update.
Calling all three implementations "one epoch" hides consequential differences.

If a run was designed to decay linearly over 200 epochs, a separately configured
20-epoch decay is usually not its first 20-epoch prefix. Likewise, changing a
warmup fraction changes warmup length. Record actual schedule semantics when
judging short trials, continuation, intervention timing or a supposed early stop.

## Choose among constant, decaying and cyclic behavior

A constant phase makes local responses easier to interpret and may retain
useful optimization speed, but can continue fluctuating around a useful region.
Decay reduces movement for refinement; decaying too early can stall a model
whose representation or new head has not learned enough. Cosine, linear or
polynomial schedules encode different allocation of update sizes. Names do not
specify endpoints, minimum rates, warmup inclusion or restart behavior.

A plateau-triggered decay depends on a trustworthy monitored metric and its
evaluation cadence. Noisy tiny panels can trigger arbitrary reductions; a
healthy proxy cannot guarantee task improvement. A planned step decay allows
prediction of its timing but may be poorly aligned with the task's learning.
Choose on evidence and task budget rather than claiming one schedule family
is universally best.

Smith's LR-range and 1cycle material illustrates interactions with momentum
and regularization. A range test increases rate while the model learns; it
does not isolate same-state rate effects. A full cyclic policy changes the
trajectory and can change effective regularization. Its SGD examples do not
justify dropping an arbitrary cycle into pretrained AdamW domain adaptation.

## Warmup: a transition treatment with a cost

Warmup gradually admits a rate that can be disruptive at the initial state.
It can be relevant for a newly constructed component, abrupt group activation
or early instability, but it does not fix label mapping, incompatible loss
inputs or an incorrect objective. Inspect when instability occurs: a late spike
at a schedule transition is different from a failing first forward pass.

There is no task-independent correct warmup fraction. Too short may leave the
transition unstable; too long spends scarce useful learning at low rates.
Distinguish prepending warmup from consuming part of a fixed total horizon.
An author's experiment that extends the horizon is not permission to extend
the current registered cap. Check whether the relevant component already has
adapted weights and moments rather than treating every resume as a cold start.

Warmup chosen for a high peak rate should be judged with that peak actually
reached. A run that stays finite only because it has not left warmup has not
demonstrated stable training at its advertised base LR.

## Batch size is a statistical and systems choice

Changing batch changes gradient noise, update count per data pass and often
normalization statistics. Rates, momentum and regularization may need joint
reconsideration. Linear LR scaling is a conditional heuristic, not a theorem
for all architectures, adaptive methods and label mixtures.

Gradient accumulation is useful for a desired aggregate that cannot fit in
one microbatch. It need not improve throughput and does not reproduce a large
BatchNorm batch: separate microbatch forwards use different statistics. Define
whether a comparison fixes examples, updates, wall time or resource consumption;
each answers a different question.

When measured throughput stops improving with a larger batch, memory capacity
alone is not a reason to increase it. Conversely, a slower step can still yield
a faster useful model if fewer steps are needed. Keep systems throughput and
learning efficiency separate rather than diagnosing one from the other.

## Clipping and mixed precision: inspect the failing path

With norm clipping, the gradient's direction is retained while sufficiently
large norm is reduced. A few outlier updates and persistent oversized updates
imply different remedies. Inspect unclipped norms and when clipping happens.
Clipping a combined loss gradient does not independently bound each component's
semantic effect; conflicting contributions may cancel before clipping.

In scaled mixed-precision training, clipping should act on the intended unscaled
gradients. The referenced AMP examples unscale before clipping and manage scaling
at the actual optimizer-update granularity. Applying a physical norm threshold
to scaled gradients changes its meaning. With multiple optimizers, inspect which
parameters, scalers and successful steps the logs actually refer to.

A sudden NaN/Inf calls for locating the first invalid input/operation, loss
domain, scale or gradient before changing all rates. Lower LR acts after the
backward computation; it cannot repair a forward invalid logarithm. A lower
precision overflow or skipped update can mimic a learning stall. Compare the
relevant operation and value ranges to a supported higher-precision path when
that observation is genuinely needed. BF16 range and FP16 scaling are different
topics; do not assume the same scaler prescription applies to every dtype.

### Distinguish three superficially similar unstable runs

1. Inputs contain invalid targets and the first loss is nonfinite: fix the
   input/objective contract. Warmup has no chance to repair that forward pass.
2. Finite learning is interrupted by isolated large gradient updates: clipping
   may target those events while preserving ordinary updates; inspect outcome
   and avoid treating mere finiteness as restored quality.
3. Most updates are strongly clipped and progress is slow: the rate/scale may
   be mismatched. Continually tightening the cap can approximate a poor step-size
   policy; reconsider rate, scaling or objective rather than clipping forever.

These are explanatory examples, not automatic triggers or mandatory experiments.

## Sources

- [Google Tuning Playbook](https://github.com/google-research/tuning_playbook), batch/time/resource sections and instability/warmup/clipping FAQ: practical guidance with workload-dependent choices.
- Smith, [A Disciplined Approach, 2018](https://arxiv.org/abs/1803.09820), sections 3–5: coupled rate, momentum, batch and regularization, including curve-order disagreements.
- [Swin main.py, fixed f82860b](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/main.py): batch-scaled LR, accumulation, actual optimizer and schedule paths. The worked 125-microbatch example is an illustration, not a claim about every Swin loader.
- [PyTorch AMP examples, 2.14](https://docs.pytorch.org/docs/2.14/notes/amp_examples.html): unscaling, clipping, accumulation and multiple-optimizer examples. Apply release-specific details only after checking the real runtime.
