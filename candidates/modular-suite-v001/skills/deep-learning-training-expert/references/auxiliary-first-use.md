# First use of an available auxiliary objective

Use when authorized training labels or another legal auxiliary signal are available
but receive no effective training exposure. This is a decision procedure for an
untested opportunity, not a universal recommendation to add supervision.

## Distinguish prerequisites from outcomes

Check label meaning, image year/domain, marked-pixel masks, permitted training
role and the implemented loss path. Use the current errors to judge whether the
signal addresses a plausible weakness. Nearby geography alone does not establish
domain equivalence; a neighboring-domain pool can still support a provisional
transfer hypothesis. Keep validation labels out of the loss.

A missing permission, broken input, label conflict or incompatible objective can
block use. Missing evidence of benefit after first activation cannot logically
block every first activation. Likewise, uncertainty about the optimum coefficient
does not prevent choosing a reversible initial magnitude. Separate “supported
reason to try” from “measured evidence to retain or scale.”

If the signal is unsuitable, explain the actual concern. If current learning is
adequate, retaining zero can be reasonable. Neither possession of labels nor a
zero-positive validation panel makes activation mandatory. Absence of positive
validation support limits positive-performance claims; it does not prohibit using
authorized positive training annotations.

## Choose a dose the implementation can execute

Inspect the reduction, sign/class weighting, sample cadence and repeated exposure.
A million marked pixels do not imply a million times the loss when the objective
uses a mean. A small pool can still dominate if oversampled or heavily weighted.
An unknown gradient contribution before activation remains unknown, not zero.

Use an applicable historical configuration, the objective scale or an available
preflight measurement as a starting reference. Adjust conservatively for changed
reduction, domain, annotation reliability or exposure. State why the selected
magnitude is a plausible first step; do not present it as optimal. A gradient
measurement can help but is not a universal prerequisite.

For the Cashew masked human objective, verify the host implementation before
using its semantics. In the current separate-sign-means implementation:
`L_human = w * (mean_negative[-log(1-p2)] + s * mean_positive[-log(p2)])`.
Only valid marked pixels contribute. The positive coefficient is `w*s`; zero
`w` disables both signs. Non-cashew annotations do not assign class zero. Counts,
cadence and occurrence limits are separate from these coefficients. Do not
silently apply this formula to a different adapter.

## Execute, observe and retain only what is supported

A scalar adjustment on the authorized main trajectory can be the trial. Preserve
a complete starting checkpoint, choose the next useful existing review interval
and avoid unrelated simultaneous changes when attribution matters. Additional
branches need only be used when their information justifies their cost and the
executor permits them. Do not request new approval for a trial already within
the user's authorized scope.

At the next review check actual nonzero exposure and loss first, then comparable
target errors, destination classes, source retention and supported positive
recall. Use a later horizon when the executed dose makes the first interval
inconclusive. Null or adverse responses can justify revising, reducing, disabling
or restoring; switching a scalar back does not restore optimizer/model history.
Loss reduction or fewer predicted cashew pixels alone is not a quality gain.

Before STOP, compare this concrete remaining opportunity with the observed
trajectory, cost and hard limits. Do not declare “nothing useful remains” solely
because the never-enabled route has no response history. Also do not demand
exhaustive interventions before stopping. Write the hypothesis, horizon and
reversal condition in the free training record; the executor should validate
legal numerical controls without parsing prose into extra approval gates.

## Basis and limits

This correction addresses a documented decision deadlock: an available auxiliary
pool remained at zero because its first-use response and optimum were unknown.
It makes no claim that activation has improved accuracy. Mechanism details and
conditional practice are in
[loss design](../../training-adjustment/references/regularization-sampling-and-loss-design.md)
and [diagnostic experiments](../../training-adjustment/references/diagnosis-and-experiments.md).
