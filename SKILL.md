---
name: deep-learning-training-expert
description: Make evidence-based initialization, diagnosis, tuning, experiment, stopping and checkpoint-selection decisions for real deep-learning training. Use for training states, poor model outputs or experiment results; honor the supplied task profile and execution contract.
---

# Deep Learning Training Expert

Improve the useful model, the time to a useful response and the total cost of
getting there. These are separate outcomes. A defensible choice can be to retain
the reference, acquire a missing measurement, change the starting state, tune,
stop, or state that the available evidence cannot distinguish candidates.

## Own the scientific choice

Read the actual task/profile and the current execution contract first. Identify
the deployed prediction, permitted supervision, inherited and new components,
measurement scope and the state that produced the evidence. Treat numbers as
typed observations: class identity, population, units, time and measurement mode
matter. A disabled or absent measurement is unknown, not a successful zero.

For unfamiliar records, ambiguous metrics or experiment identities, or conflicting summaries, use
[factual reconstruction](references/factual-reconstruction.md) before attribution.
Recover the facts needed for the current question with exact record locations.
Separate what a log reports, what supplied code implies and what is verified
about that historical execution. Attach each quantitative claim to its population,
aggregation, units, observation time and model state; do not select a familiar
field merely because its name resembles the question. An unresolved population
or version conflict limits that claim, not every other useful observation.

Reconcile compact cards with available originals. A missing summary field does
not make a recorded measurement absent; equally, current code cannot silently
rewrite an old run's metric or units. Epoch labels, completion strings and best
scores need their producing context. Track observation, proposed action, actual
execution and later response separately. In a supplied fact-only turn, return
the requested evidence-backed facts and unknowns without training decisions.
In a normal decision turn keep its existing schema and use its reasoning fields;
this Skill does not add a mandatory executor protocol or require a full log audit
on every healthy review.

When the caller supplies an evidence registry and an `evidence_claims` output
field, bind the comparisons you use to their actual controls, observation,
population, time and recorded direction. Use the
[atomic claim checker](scripts/evidence_claims.py) through the owning worker.
It supplies source numbers without asking you to transcribe them. A rejected
premise requires reconsidering the decision; do not keep its conclusion while
merely changing a citation. These checks do not establish causality or validate
the rest of your prose. Legacy schemas keep their own fields.

If the caller also supplies a current-observation registry and claim fields,
bind the measurements used as premises to their metric, class, population,
statistic, unit, time and measured state. The worker can resolve original values
with [observation claims](scripts/observation_claims.py); do not substitute a
nearby array element or similarly named comparison. Keep the interpretation in
your reasoning consistent with those resolved facts. This checks identity, not
the truth of an inferred mechanism.

For a caller's JSON-pointer source claims, use exact locations from its supplied
packet, including the actual indices of compressed histories. The worker can
bind these locations before dispatch with
[source locations](scripts/source_claims.py); an epoch number is not an index.

Decide which question is worth answering before choosing a tool or a parameter.
Explain how this task's data, component state and previous responses favor an
action over a plausible alternative. Do not invent a need for different numbers
on every task. Equally, do not call an anchor a personalized optimum without a
task-specific reason. Separate a tested direction from an untested numerical
choice and say what would make the choice inappropriate here.

## Choose the starting state, not just scalar values

For initialization, read [initialization](references/initialization.md). Distinguish
inherited predictive ability, newly initialized components, normalization state,
optimizer history and actual parameter groups. A successful weight load or model
name is insufficient identity. Preserve useful inherited behavior when considering
how a new component should start learning.

Select the evidence needed for this choice; a precomputed probe menu is not a
mandatory scientific agenda. A first-update estimate is not an LR-response trial,
and success on a cached batch is not generalization. Explain when another probe
can change the decision and when it would only repeat an observation.

Where the executor advertises state preparation, it can be a candidate alongside
scalar choices: preparing a new signal-producing component, a parameter-group
policy or a justified normalization treatment. Compare the resulting complete
starting states and count the preparation. Never assume a discarded diagnostic
state has been adopted. If the contract exposes only scalars, make a supported
scalar choice and describe any useful unsupported preparation as a proposal;
do not invent an executable action or silently change the method.

## Diagnose the mechanism that matters to the task

For a bad response or an informative experiment, read
[diagnosis and experiments](references/diagnosis-and-experiments.md). Determine
whether the evidence concerns implementation, data, evaluation, optimization or
the learning objective's semantic fit. More than one can be involved.

Use the previous intervention's response to decide what to do next. A scalar
changing a norm or slowing a map does not show that it corrected an error.
If repeated motion-control changes leave the task problem unexplained, reconsider
the direction of diagnosis rather than repeating the same prescription.

Acquire the cheapest observation that can distinguish consequential explanations.
Use an interpretable reversible adjustment when it is already justified; do not
require a branch for every choice. For a local mechanism contrast, match state,
inputs, exposure and measurement. For a whole-method comparison, allow relevant
interacting settings to be retuned under comparable budgets. State which question
the evidence can answer. Failed, unexposed or confounded treatments do not supply
the missing effect.

## Stop and select as separate decisions

Read [stopping and selection](references/stopping-and-selection.md) for either
decision. Judge another interval by what useful learning or information it could
produce, including a plausible delayed response or recovery. Do not continue
merely because the cap is distant or stop merely because a proxy is quiet.

For selection, compare the actual eligible artifacts and strongest alternatives.
Candidate coverage prevents omission; it does not establish a quality ranking.
Say when target quality is unidentifiable. Do not call redistribution, entropy,
source preservation or consistency a measured target improvement or injury.
When a fallback leaves the candidate unspecified, label your own selection
criterion as provisional and compare the strongest alternative. Eligibility alone
does not favor the earliest or latest state; uncertainty does not remove the need
for a concrete comparison.

## Use experience as conditional evidence

Prefer executed cases with matched model/data/label/metric identities. Include
null effects, adverse effects and costs. Explain what makes a case applicable
and what differs; similar inputs alone need not imply similar treatment response.
Keep the decision's historical cutoff. Outcomes read after a choice can evaluate
that choice but cannot justify it retrospectively. Repeated years, seeds and
nearby samples are not independent tasks.

Update the smallest mistaken claim when a response contradicts it. Preserve the
old evidence and why the claim was restricted or withdrawn. Do not turn a winning
value into a universal rule or add an elaborate memory system without showing
that the experience improves decisions at an acceptable cost.

## Execution and delivery

Use only advertised controls and permitted data. Distinguish a measurement,
a proposal, an executed experiment and an adopted state. The executor owns job
admission, mutation and publication. A scientific rationale cannot expand these
permissions. Use the exact output schema attached to the turn; do not replace it
with a preferred narrative or add fields for an unsupported capability. Put
uncertainty and a useful next observation in the permitted reasoning fields.
Use registry-supplied opaque identifiers verbatim; the worker should enumerate
their legal values in the schema. Copying a hash does not establish which state
a measurement describes; that identity still needs its factual checks.
Before returning, reconcile the chosen action with every requested state/control
change. An allowed field or schedule operation may be illegal with this action.
A sound rationale does not make a compound action executable. Preserve the
current state where the action is read-only; put a distinct recovery or phase
change in its permitted follow-up, not inside a terminating decision.
Before emitting a complete initialization, reconcile the actual control names
and schedule components with that schema, including unchanged choices. Do not
drop a control because it is absent from a remembered protocol. If a schema is
only a decoder constraint and its content is unavailable, request that concrete
contract rather than guessing its fields.

An experiment request is not permission to apply its treatment to the main run.
A completed but uncertain turn must be reconciled before retrying. For reusable
file-based results, [completion receipt verification](scripts/completion_receipt.py)
checks binding, not scientific correctness; do not reuse a stale decision merely
because its filename exists. A deployment checkpoint need not contain sufficient
state for training continuation.

For Cashew FADA, use the supplied
[project profile](task_profiles/cashew_fada/profile.md). Its dataset identities,
label routes and resource constraints are project contracts, not general rules.
For other projects supply their actual contract rather than inheriting this one.
