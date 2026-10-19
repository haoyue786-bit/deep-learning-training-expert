# Diagnose the cause before repeating a treatment

Use for unexplained training behavior, model errors or interpretation of a trial.
The aim is a consequential choice, not an exhaustive list of every possible cause.

## Read an observation at its actual scope and time

Resolve class names and IDs, units, train/validation/deployment populations,
sampling and missing support before attribution. Distinguish a model-conditioned
slice from ground-truth conditioning. A reported zero after disabling a loss is
not a measurement of its gradient. A decision at epoch t acts after its observed
checkpoint; observations from the next interval are its response. Do not move
later evidence back into the decision's baseline.

Source preservation, unlabeled distribution summaries and target accuracy answer
different questions. Do not describe a proxy's unfavorable movement as measured
task harm, even after adding a disclaimer elsewhere. A trusted local error panel
supports that panel and its support, not an entire region. Absence of positives
makes retention unknown.

When permitted feedback includes sample identities, compare which errors were
corrected, introduced or merely changed type. Equal confusion counts need not
mean the same positives survived. The caller can use the paired-decisions mode
of [score diagnostics](../scripts/score_diagnostics.py) on identical labelled
populations; it rejects changed truth, duplicate IDs and silent intersections.
Error slices can narrow the next question but do not identify a parameter's
causal effect or make correlated samples independent. Aggregate-only feedback
does not support claims about individual error recovery.

For complaints about probability cutoffs or excessive predicted area, separate
score ranking, probability scale, class support and the actual delivery rule.
When labels are explicitly permitted, [score diagnostics](../scripts/score_diagnostics.py)
can compare the delivered decisions, average precision and a same-panel maximum
F1 partition. A high optimal cutoff alone cannot identify undertraining: a
strictly monotonic score transformation can move that cutoff without training
or changing ranking. Conversely, global rescaling cannot repair a ranking gap.
Pooled scores from different models/groups can also conceal group differences;
inspect support before averaging them. Thresholds fitted and scored on the same
points are descriptive, and groups without positives cannot establish retention.
Do not change decoding or a conditioning temperature to act as unadvertised
calibration. A selected checkpoint's epoch is not the stopping epoch or actual
executed work. To test training duration as a cause, seek a matched continuation
with the same state, data and evaluation rule, rather than inferring duration
from the cutoff. Missing original state limits that causal test; preserve the
unknown and describe a future controlled test without calling it a recovery.

## Distinguish explanations that imply different actions

Consider the actual causal paths: inputs/labels/sampling, loss construction,
normalization and execution mode, trainable components, optimizer updates and
the semantic appropriateness of the objective. Use the current response to narrow
this inquiry; do not conduct a compulsory checklist on every healthy run.

In multi-loss training, inspect shared and disjoint parameter support. A ratio
on feature parameters may omit a decoder's auxiliary pressure. A lower weighted
ratio can result from its numerator, denominator or multiplier; none alone says
whether the task errors improved. Distinguish immediate mechanical effects from
the proposed semantic effect.

If smaller LR repeatedly makes outputs quieter without explaining an error,
consider a different contrast: input/label identity, normalization-only changes,
component-isolated pressure or conditioning behavior, as permitted. Do not cycle
the same scalar because it predictably moves a proxy. Equally, do not abandon a
useful modest intervention merely because it is simple.

## Select a discriminating, affordable contrast

State a leading explanation, a consequential alternative, and an observable that
would change the next choice. Use an unchanged arm from the same state when it
is needed to attribute a response. Match inputs, RNG, exposure, schedule phase,
completion and evaluation scope; distinguish claimed matching from recorded or
verified matching. A finite result with no effective exposure is not a negative
result for the treatment.

When implementing paired continuations, keep the saved starting state independent
of mutable model and optimizer tensors in each branch. Loading a state dictionary
can retain shared tensor storage; running one branch can then change the state
used by the next. Restore from an independent copy or fresh deserialization and
verify the original state remains unchanged. Matching seeds cannot repair a
changed optimizer history.

When attributing improvement to an auxiliary objective, adaptation or a new
component, ask whether the comparison separates that contribution from continued
learning of the inherited predictor. Its untrained continuation checkpoint is
not a continued-training baseline. Where this distinction can change the choice,
consider a matched continuation retaining the main supervised objective while
omitting the questioned mechanism, under the advertised execution contract.
Disabling its coefficient need not disable its forwards, normalization exposure
or optimizer work. Treat a whole-policy contrast accordingly. A useful response
does not establish the best mechanism or setting, and early rankings can reverse;
retain the declared later endpoint rather than converting a short response into
a universal initialization rule. This is a conditional experiment, not a required
ablation or a default to remove auxiliary learning.

One-factor contrasts answer local questions. Interacting settings may warrant a
declared joint treatment or a method comparison with retuning. Credit the joint
treatment as joint; do not infer an individual cause from it. Avoid spending a
long horizon on a contrast whose result cannot affect a decision.

Interpret both arms relative to their start and each other, at every declared
endpoint. Keep null, adverse and incomplete results. A failed comparison says the
effect is unknown. A supported mechanical mechanism can justify a cautious change
without establishing target accuracy; state that limit and check the actual next
response. Apply or adopt only what the run contract permits.

## Follow a contradicted hypothesis somewhere useful

For an adverse intervention response, consider whether the current schedule and
remaining horizon permit recovery before discarding the trained state. Reversing
one control, restoring an earlier complete state and continuing the current state
preserve different learned progress and optimizer history. A short quality drop
does not establish irreversible damage or make full rollback optimal. Compare a
plausible recovery explanation with the evidence favoring restoration; keep both
artifacts when permitted. Count discarded updates and replay against the actual
remaining budget. This is not a reason to delay a required safety exit or demand
an extra experiment when the contract cannot support one.

Specify what failed: the measured premise, parameter effect, semantic explanation,
numerical choice, or expected persistence. Do not call every mixed result partly
supported indefinitely. Preserve a supported component while withdrawing an
unsupported inference. KEEP, a reversal, a different experiment or STOP can all
be reasonable depending on what remains useful and possible.
