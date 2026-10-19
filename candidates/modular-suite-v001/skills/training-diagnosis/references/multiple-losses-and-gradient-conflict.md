# Multiple losses, shared parameters and gradient conflict

Read when losses compete, a slow objective keeps receiving more weight, a raw
gradient ratio is treated as control truth, or gradient surgery is proposed as
a remedy. Separate semantic priorities, magnitude, direction and actual motion.

## Map each objective onto actual parameters and samples

For `L=sum_i w_i L_i`, record which parameters each loss reaches, how its samples
and reduction are defined, and whether its forward changes buffers. A feature
gradient measurement cannot describe an unmeasured decoder. Missing gradients
and zero tensors have different meanings; align parameter identities before
flattening vectors or comparing norms.

On a common parameter subset, define `g_i=grad(L_i)`. Weighted norms describe
component scale; dot products/cosines describe raw directional agreement.
Neither includes Adam moments, per-coordinate normalization, clipping or decay.
An actual displacement measurement answers another question. Do not report the
same ratio as all three kinds of evidence.

Raw loss magnitudes are even less comparable: sum versus mean, class weights,
target scale, ignored support and additive constants can alter them without
the intended semantic change. Establish definitions before claiming one task
dominates because its printed loss is larger.

## Negative cosine is a clue, not a diagnosis

For a sufficiently small plain-gradient step along a chosen combined direction,
the first-order change to task `i` depends on its dot product with that direction.
Two task gradients can oppose while their sum still improves the declared
weighted total. Priorities matter: preserving source semantics and adapting
domains are not necessarily symmetric objectives that should never interfere.

PCGrad identifies conflict by a negative inner product, but its motivation also
discusses magnitude imbalance and curvature. Its projected raw gradient does not
guarantee every task improves after an adaptive optimizer, nor does a single
sampled negative cosine establish harmful target transfer. Inspect whether the
observed error and proposed pathway actually correspond.

### A simple counterexample to "remove all conflict"

In one shared coordinate, let `g1=1` and `g2=-2`. Projecting both against the
other removes both directions, yielding zero, even though their sum is -1.
That may protect both tasks from first-order tradeoff while sacrificing progress
on a specified total objective. Near opposition can also slow learning. The
choice requires the task's priorities rather than a universal zero-conflict goal.

## What uncertainty weighting adjusts

Kendall et al. model task-dependent homoscedastic noise: a task shares a scale
across inputs. This is different from a pixel's predicted confidence,
heteroscedastic uncertainty or epistemic uncertainty. For a scalar Gaussian
residual sum `L` and `s=log(sigma^2)`, its negative log likelihood has the form
`0.5*exp(-s)*L + 0.5*s`, omitting constants. The residual coefficient and the log
term must agree with the likelihood, dimensionality and reduction. Summing many
valid pixels versus averaging them can change this correspondence.

The classification construction first introduces `softmax(logits/sigma^2)`.
The paper then makes an explicit approximation to obtain a weighted ordinary CE
plus a log-scale term. That approximation is not exact softmax NLL at arbitrary
temperature. The paper's practical depth/instance losses also use L1, so do not
copy a Gaussian squared-error coefficient without checking the intended
likelihood and scale parameterization. This training scale is not a temperature
fitted afterward on independent calibration labels.

For a fixed positive scalar residual, differentiating the Gaussian form gives
the stationary relation `exp(-s)*L=1`. For the paper's classification approximation
`exp(-s)*CE + 0.5*s`, the corresponding relation is `exp(-s)*CE=0.5`.
These relations explain why weights can rise as residuals fall; they are not
equalities at every update or measures of task value. At exactly zero residual,
the simplified scale objective can decrease without bound as s decreases. Log
parameterization avoids direct division by zero, not every numerical or modeling
degeneracy.

A large learned scale can reflect noisy labels, model bias, missing capacity or
a mismatched likelihood. It does not identify pure observation noise from the
training residual alone. Shared representations can make several heads agree
on the same wrong explanation; multiple agreeing outputs are not independent
validation evidence.

This method can help combine supported supervised tasks with different units
and noise under an appropriate model. It can also reduce the influence of a
hard task that the product values highly, or amplify easier tasks as their
residuals shrink. The original paper notes increasing weights and compensates
with a decaying SGD schedule. GradNorm reports difficulties in other optimizer,
data and weighting settings. Those results concern interacting training
recipes, rather than a universal verdict that either method always wins.

Before proposing uncertainty weighting, identify the likelihood, valid-support
normalization, learned-scale optimizer, schedule interaction and product
priority. A source segmentation loss, adversarial game and weak-negative loss
do not automatically form independent noisy observations under this model.
Task-level weighting also does not authorize changing fixed within-task class
weights. Introducing the method requires executable support; it is not a
hidden reinterpretation of an existing lambda field.

## What GradNorm adjusts

GradNorm measures weighted gradient norms on a selected shared subset, compares
each task's loss reduction relative to its initial loss, and adjusts task
weights toward a target involving relative inverse training progress. With
`l_i=L_i(t)/L_i(0)` and `r_i=l_i/mean(l)`, its target has form
`mean(G)*r_i^alpha`. Equal norms arise only in a particular setting; the method
does not require every task always to have identical gradients.

The target is held fixed for the weight-update calculation; weights are
renormalized to control total weight mass. Implementation still needs choices
of measurement layer, initial loss, weight-update optimizer and constraints.
The progress ratio is not validation learning speed or a derivative in time.
Tiny/negative/offset losses, changing masks and pretrained initial behavior can
make a naive baseline such as `log(C)` inappropriate.

An important published failure is a task stuck predicting its majority class:
its slow progress can keep attracting weight without fixing its pathology.
Before increasing a slow task's weight, investigate noisy supervision,
unlearnable distinctions, missing exposure, wrong implementation and capacity.
Automatic balancing is not automatic diagnosis.

## What PCGrad adjusts

For a conflicting pair, the Euclidean component of a task gradient along the
other is removed. In multiple tasks, sequential projection order matters;
a single pass is not a joint optimal projection satisfying every final pairwise
constraint. Shared parameters and task-specific heads need separate handling.
Zero-norm guards, aligned gradient coordinates and projection-order randomness
are real implementation concerns, not optional labels on a generic algorithm.

Passing corrected gradients into Adam changes how their geometry translates to
motion. Gradient storage, additional backwards and projection can increase wall
time and memory. Better sample efficiency is not automatically cheaper compute.
Introducing surgery changes the training method and requires advertised support.

## Choose the least misplaced intervention

| Main uncertainty | Useful evidence | Why a popular balancing method may fail |
|---|---|---|
| One loss has a larger numerical scale | Reduction, weighting and shared-gradient scale | Equalizing printed losses need not equalize gradients |
| A task improves slowly | Exposure, labels, capacity and actual initial loss | Progress weighting can amplify an impossible/noisy task |
| Tasks have different units or residual scales | Likelihood, masks/reduction, learned scales and product priorities | Uncertainty weighting estimates a model-dependent scale, not task importance or pointwise correctness |
| Two trusted tasks harm one another | Aligned directions, relevant error response and priorities | Removing opposition can stall a necessary tradeoff |
| Adversarial term opposes source task | Objective meaning, conditional signal and target response | Opposition may be an intended adaptation force rather than a bug |
| Paused adaptation still causes drift | Remaining supervised/auxiliary paths and buffers | Retuning inactive D or adversarial weights targets no active cause |

A measured mechanism can justify a modest reversible adjustment without adding
a new algorithm. If a new weighting/surgery method is proposed, preserve the
original objective, source-weight contract and comparison scope rather than
silently redefining task priorities. Inspect each important class response,
not just a lower combined loss or a tidier gradient plot.

## Sources and limits

- Chen et al., [GradNorm, ICML 2018](https://arxiv.org/abs/1711.02257), section3/Algorithm1 and majority-class failure discussion: adaptive magnitude/progress balancing under its multitask setting.
- Yu et al., [PCGrad, NeurIPS 2020](https://arxiv.org/abs/2001.06782), section2/Algorithm1, theory conditions and compute discussion: directional intervention with limited guarantees.
- Kendall et al., [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://arxiv.org/abs/1705.07115), Section3.2 and AppendixB: likelihood scaling, explicit classification approximation and changing effective weighting; experiments include task tradeoffs rather than a single universal optimum.
- [CS231n Part 3](https://cs231n.github.io/neural-networks-3/), gradient versus update monitoring; the actual optimizer implementation remains necessary.

The one-coordinate example is algebraic, not a newly executed experiment.
Adversarial games, fixed-priority supervision and arbitrary weighted losses
cannot automatically inherit cooperative multitask guarantees.
