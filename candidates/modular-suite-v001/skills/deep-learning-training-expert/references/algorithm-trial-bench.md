# Callable methods and bounded development trials

Read this reference when the actual question concerns implementing, invoking or
testing a method. Ordinary reviews need the compact current facts and at most
the relevant module cards, not this entire document or eight module bodies.

## One owner, one configuration

The suite's [scientific_defaults.json](../scientific_defaults.json) is the central
default for scientific requests: `gpt-6.1-sol`, reasoning `medium`. The task may
explicitly override it. Credential, transport and proxy routing come from the
caller; this suite does not rewrite global Codex settings. `suite_packet.py`,
the decision packet and `ReviewDriver` expose the same defaults. The host's
`decide(packet)` must use those fields in its actual model request and report
actual usage. Merely naming a model in prose is not dispatch verification.

Install all nine sibling Skills. The executable entry is
`scripts/algorithm_trial_bench.py --config /absolute/project/trial.json`.
The repository includes a placeholder example configuration. The Cashew adapter
reuses the supplied job's pinned A3/FADA runtime and source checkpoint. Keep
imagery, weights, raw predictions, labels and empirical experience in the
authorized project. Other projects implement the small `TrialBench` adapter
protocol; they do not inherit Cashew paths, class order or source loss.

With no extra option, the command measures a fixed small development panel,
writes confusion/Precision/Recall/F1 and both class-pair error directions,
and builds a compact scientific packet. `--interface-smoke` calls all five
implemented kernels on real source/unlabelled-target tensors without optimizer
updates or validation feedback. After reading the measured diagnosis, request
one supported treatment, for example:

```sh
python scripts/algorithm_trial_bench.py --config /absolute/project/trial.json \
  --trial M04:source_pair_margin --controls '{"margin":0.3}' --strength 0.01 \
  --plan /absolute/project/reviewed_trial_plan.json
```

This example is a bounded mechanism test, not a recommended production recipe.
No command here makes an additional model/API call. A scientific owner chooses
from the evidence; deterministic kernels execute the choice.

## Actual capability boundary

`training_runtime.registry` is the single source for callable identity, control
schema, input roles and module cards. Use `invoke(module_id, variant, inputs=...,
controls=..., data_role=..., authorized_variants=...)`; no plugin compilation,
generated Python, separate Agent or new model request is needed per module.
The host still supplies the correct tensors and authorized training provenance.

|Module|Implemented trial operation|Boundary|
|---|---|---|
|M01|Separate local/global model adversarial scales; neutral 1/1|Does not change discriminator cadence or silently alter source CE|
|M02|Detached mean-one whole-pixel class priority; default all ones|Retains original clipped within-pixel class mass/proportions; no channel reweighting|
|M03|Detached pair-ambiguity attenuation; optional uncertain ecological context|No renormalization that cancels attenuation; no prior-as-truth labels|
|M04|Differentiable source-feature cosine-margin pair separation|Actual source training truth only; missing class support is unsupported|
|M08 source|Top-fraction known-negative source/human training loss|No target validation labels; negative provenance remains distinct|
|M08 automatic|Adapter over existing protected pool executor|Host supplies its actual controller/pool; reuse masks/loss/BN/probes|
|M05|Design only|Class-mass correction is unavailable|
|M06/M07|Existing host implementation required|No duplicate projection or unverified dynamic component policy|

The five bench kernels are not a complete MCC, CDAN or CCA-LSC reproduction.
Installing this suite does not inject any kernel into a live production job.
Disable prevents future contribution; undoing updates requires restoring the
complete model, optimizers, buffers, modes, RNG and pool state.

## Trial and validation contract

Use actual predictions and versioned labels to change the diagnosis. Error
counts provide advisory leads; the expert may choose any authorized implemented
variant, or no trial. Every trial requires an evidence-bound scientific plan,
including suggested variants. More negative supervision is not automatically appropriate when
recall is the larger problem. Preserve class1 aggregate semantics; a mixed
trees/grassland label is not proof that an error belongs to pure forest/wetland.

The plan is a JSON object with `hypothesis`, `alternative`, `why_this_action`,
`expected_response`, `reversal_condition`, the diagnosed `initial_state_sha256`
and exact `dataset`, `variant` as `module:variant`, actual `controls`, `strength`,
`steps_per_arm`, and hashed `evidence` file pointers. Read initial identity and
feedback from the bench output; do not invent these values. `run_pair(...,
scientific_plan=plan)` validates them before admitting work. A changed plan
does not expand permissions, mutate the reference or reset the trial budget.
Plan fields make the choice reviewable; schema validity does not prove a sound
scientific explanation.

The compact packet initially lists available variant names without loading all
research cards. Select at most two relevant cards explicitly when needed.
Ordinary host controls and KEEP remain legitimate alternatives. Contextual
experience can be queried with `experience_query` in the config; historical
replay additionally supplies `evidence_cutoff` and excluded case/unit identities.

Before a paired update, declare the endpoint, controls, reference, exact full
state and source/unlabelled images. Default two updates/arm, at most twenty;
maximum two treatments and a finite development-use ledger. Restore the exact
start for each arm and after the trial. Save failed and discarded work and
check protected input hashes. Both arms receive the same declared D cadence;
the small adapter's one D update/branch is not a production reproduction.

`source_safety()` currently measures one fixed source-validation patch. The
bounded target panel has explicit point/patch/tile support. Neither is a full
country result. A positive tiny-panel response supports review for a bounded
confirmation; a null/adverse/unsupported result is a reason to stop expanding
that search, not to fit another threshold until it wins.

Validation labels enter inference feedback, never `training_step`. Reserve a
development use before observing feedback; failed uses count. Reusing a verified
receipt requires the same checkpoint/data/label identities. Do not reset the
budget by renaming the dataset. Previously inspected development data cannot
be described as an independent test. Threshold optimization and formal method
comparison are separate prospective tasks; do not claim they already happened.

## Epoch bridge and automatic negatives

Use `ReviewDriver` at epoch boundaries with five host callbacks:
`evaluate`, `save_checkpoint`, `decide`, `apply_decision`, `refresh_automatic`.
Every epoch records losses and nonfinite measurements. Evaluation/checkpoint
work remains every five epochs; the scientific decision is every ten, plus a
terminal or explicit emergency review. Each decision receives the last two
compact observations. Stop on a numerical anomaly before further updates.
Supply Python scalar summaries, not tensors/full logs/raw prediction arrays.

One decision may include `automatic_negative_plan`; the adapter forwards this
same plan through the existing protected selection executor without a second
API request. It preserves original manifests and writes a cadence sidecar.
New pools expire at the next decision and spread **one occurrence per selected
patch over the actual intervening update horizon**. Longer validity does not
multiply exposure. Persist generation identity, hash and consumed cursor;
resuming cannot renew or extend a spent generation. Legacy jobs without an
explicit cadence migration retain their registered interval and 128-update
pools. Reconcile divergent checkpoint/pool state before resuming.

The bridge journals stages and commits a received model decision before host
mutation. Use its explicit `resume(epoch)` and host reconciliation hooks for a
failed request/apply/refresh. Completed stages are not repeated; uncertain
effects require host evidence. Read [execution and recovery](execution-and-recovery.md)
for state binding, native schema validation and production integration limits.
Wall budgets are
checked between blocking host calls, not by asynchronously killing them.

## Economical delivery

One continuing scientific owner uses the relevant professional Skill bodies,
at most two current module cards, a few actual errors, and at most three concise
project experience summaries. Full logs/data stay behind evidence pointers.
Packet bounds fail explicitly rather than silently dropping decisive facts.
Serialized bytes/characters are measured; token counts remain unknown until
actual model usage reports them. No automatic reviewer calls or claimed token
savings percentage is built into these deterministic interfaces.
