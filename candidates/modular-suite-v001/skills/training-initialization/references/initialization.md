# Reason about initialization from the learning system

Use when choosing a fresh start or investigating an unexpectedly poor start.
For an exact continuation, restore the declared full state instead of applying
fresh-start heuristics to it.

## Locate what is inherited and what has to be learned

Check component-level loading and the actual active forward/loss paths. A
classifier may already be trained even when an encoder is described as
pretrained. Conversely, freshly constructed legacy layers may be inactive.
A module can execute in a discarded branch or influence only an auxiliary loss.
When its role affects the proposed start, check the objective dependency;
inference hooks alone do not establish training activity. Optimizer membership
alone does not establish a parameter update.
Distinguish model parameters, running buffers, optimizer moments, EMA/teacher
state, schedules and RNG. A matching model tensor digest alone does not identify
the complete training start.

Understand functional initialization. Zeroing a residual branch's outlet may
preserve the inherited function while delaying gradients into its interior.
This differs from zeroing an entire model or a trainable classification head.
LayerNorm has no BatchNorm running-statistic calibration to transfer.

## Relate a proposed start to current evidence

Ask what presently limits useful learning. For example, a new discriminator
learning on cached features shows it can fit those features; it does not show
that its class-conditioned gradient is appropriate on current target examples.
A pretrained network may need smaller movement in some components while a new
component needs learning, but this direction is not universal. Feature distance
can represent meaningful domain shift, mixed inputs or unfamiliar background.
It does not identify an optimal LR or assign a class label.

Use joint evidence where the hypothesis is joint. A pooled target distance and
a pooled predicted class fraction do not establish that the predicted positive
pixels are those far from source positives. State the conditioning, sample
support and dependence of a statistic on the model's own predictions.

For each exposed initial choice, explain its role in this task. A justified
reference is valid. Unsupported precision, maximum-bound choices and assigning
every task a new vector are not evidence of expertise.

## Choose measurements and preparations selectively

A diagnostic should answer a decision-changing uncertainty. Distinguish raw
gradient norms, directions, actual optimizer updates and response to training.
Adam moments, clipping, decay, parameter groups and schedules can separate them.
Check the direction of a numeric transformation before citing it. A table keyed
by desired response may store the required control value, rather than a measured
response at that control. Reconstruct the forward calculation and distinguish a
clipped candidate from the nominal target. Keep held-out sample counts separate
from total fitted-plus-held-out counts.
If a cap is inactive, another cap scan may add little; if a new module is the
uncertainty, measuring an unrelated LR scale will not resolve it.

If supported and justified, compare a prepared start with the reference: a new
module warm start, limited freezing, group learning rates or a normalization
state treatment. Define exactly what is allowed to change and what is preserved.
Count all preparation, including D-only updates and inference. A prepared state
adopted into training is a different treatment from a probe whose weights are
discarded. Never silently promote the latter.

Keep comparisons informative. A normalization-only contrast isolates a buffer
effect only when learned parameters, data/augmentation and relevant execution
modes are matched. It may improve one error type and harm another. Success on a
negative-only panel cannot establish positive retention. A new-module warmup can
also make a wrong signal more confident; assess that alternative rather than
requiring warmup on every run.

## Interpret response at the declared horizon

Record what early benefit means before seeing outcomes. Do not select the better
of an early and a later endpoint. Distinguish a harmless initial delay, a later
benefit and a loss of earlier useful behavior. A short trial can falsify a proposed
mechanism without identifying a globally best initialization.

If state preparation or a needed measurement is unavailable, name the limitation
and make the best supported legal choice. Do not claim it ran or use a fictional
control to satisfy an output schema.
