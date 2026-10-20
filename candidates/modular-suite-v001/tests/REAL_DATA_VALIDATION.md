# Verification of modular trial interfaces, 2026-10-20

The private experiment receipts, data identities, full predictions, development
usage ledger and actual experience belong to the authorized project directory.
They are not distributed as Skill memory. This file reports aggregate release
verification and the limits of what was tested.

## Deterministic tests

`python -m unittest discover -s tests -v`: 32 tests pass in the local
Python 3.8/PyTorch/NumPy environment. Coverage includes all five callable
kernels through the common registry, neutral loss/gradient behavior, detached
conditioning, real source-feature gradient paths, missing-class support,
forbidden data roles, invalid/unavailable production capabilities, versioned
cadence, per-epoch numerical handling, committed decision recovery, finite pool
exposure/resume, matched model/optimizer/RNG restoration, failed/discarded cost
accounting, project isolation, immutable evidence and concurrent validation caps.
Fixtures establish software behavior, not scientific efficacy.

## Actual data checks

The opt-in checks used a pinned existing A3/FADA implementation, its verified
source E173 checkpoint, one real source-training patch, one separate source
validation patch and real unlabelled target imagery. CPU only; no production
weights were overwritten and no optimizer state was adopted.

|Implemented operation|Actual observed execution|
|---|---|
|M01 separate model-side adversarial scales|Neutral reproduces the original sum; nonneutral loss gradients are 0.5 and 1.5|
|M02 whole-pixel class priority|Mean priority 1.0; class proportions preserved; nonzero discriminator gradient|
|M03 pair reliability and ecological-context input|Nonzero loss attenuation and gradient; no attenuation renormalization; ecology is uncertain context|
|M04 source-feature margin|Both classes supported; nonzero feature gradient, about 0.00295 norm|
|M08 known source hard negatives|131 of 524 eligible source negatives selected; nonzero logit gradient|
|Inherited automatic-negative loss|Four real protected masks; finite weighted loss about 0.00017767 and model gradient norm about 0.01022|
|Inherited controller with new cadence adapter|Four real cached protected candidates produce four new isolated masks; same-turn plan and resume use zero additional API calls|

The inherited pool/controller tests reuse actual external code rather than a
second implementation of its loss, exclusion, BN or microbatch semantics.
Controller verification uses an existing bounded protected catalog; it does
not claim a new full-country catalog scan. Four selected patches schedule once
each over ten epochs, expire at the next decision, and preserve the consumed
cursor. Original input artifacts and the full model/optimizer/mode/RNG state
were checked unchanged after the smoke.

One first attempt of the opt-in automatic smoke stopped before pool execution
because its harness supplied a duplicate keyword. That defect was fixed; the
original failed log and failure case were retained in the project store. No
optimizer update or validation evaluation occurred in that failed attempt.

## Small development trial

A deterministic, pre-prediction panel contained seven points in six patches
within one tile: three mixed-trees/grassland aggregate points and four Cashew
points. It is too small and spatially narrow for country accuracy claims.
The fixed four-class argmax start had Cashew TP=1, FP=2, FN=3: Precision=1/3,
Recall=1/4, F1=2/7. Three Cashew→aggregate and two aggregate→Cashew errors
motivated M04 and bounded M02 priority tests, rather than automatic extra
negative supervision. The aggregate labels do not identify pure forest or
wetland mistakes.

Each test compared two matched optimizer-update steps per arm from the same
full state and inputs. Both arms used one D update per branch per step for the
bounded trial; this is not a full production training reproduction.

|Treatment|Mechanism executed|Reference F1|Treatment F1|Outcome at declared endpoint|
|---|---|---:|---:|---|
|M04 source pair margin|Yes|0.285714|0.285714|Null|
|M02 class priority|Yes|0.285714|0.285714|Null|

Eight model updates and sixteen discriminator optimizer updates were executed
across both paired tests; all branches were discarded after measurement.
One initial diagnosis and four trial endpoints consumed five of six declared
development feedback uses. Exact cached feedback did not add uses. No
threshold search, independent final test or superiority claim was made. The
outcomes justify retaining these null cases and not automatically scaling up
an unproductive search; they do not prove the methods can never help.

## Model delivery and cost

The suite default is `gpt-6.1-sol`, reasoning `medium`. Callers read the same
central configuration. Compact packets carry measured class errors, relevant
module cards and bounded project experience; the kernels and local checks make
zero additional model/API calls. Serialized characters/bytes can be measured;
actual token savings and model reasoning quality were not benchmarked here.
Production dispatch remains the host's responsibility.

M05 remains design-only. M06 and dynamic M07 require an advertised pinned host
adapter. Installing the suite does not migrate live jobs, open sealed tests or
register untested production interventions.
