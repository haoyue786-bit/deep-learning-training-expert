# A compact expert review with external training history

The host computes and stores history; the model interprets it. Do not force the
model to remember a growing conversation, reconstruct curves from raw logs or
read the entire knowledge library at every review.

## What the expert sees

1. A task card: objective/decoder, inherited/new components, active supervision,
   current state/control vector, legal bounds and unresolved questions. Keep
   execution receipts and manifests outside the prompt with verified identities.
2. Whole-run curves ending at the current epoch and a compact exact table:
   target F1/precision/recall, FP/FN/support; source retention; separate losses;
   gradient norms/weighted ratios/directions; effective model/D learning rates
   and lambda; prediction area/fraction and prior context. D fit and auxiliary
   exposure belong in relevant active diagnostics.
3. Actual applied intervention markers, pending/rejected actions, recent response,
   baseline and historical extrema. Rejected proposals are never applied markers.
   Restored branches have separate identities.
4. Start with a few specific knowledge sections and compact conditional lessons.
   Supply the body if the model cannot read it. Add further knowledge only when
   it changes a decision, not by sending all professional skills.

At E0 say “one observation; no trend yet.” Do not invent gradient measurements,
smooth over spikes or bridge missing observations. Preserve sampling definitions:
raw gradient ratio is not AdamW displacement; classification gradients may include
auxiliary losses. Inspect the producer before calling them pure source gradients.
E0 can have a gradient diagnostic only if it was actually computed.

Lineage, label version, population/mask, units, decoder and metric definitions
must match within a trend. Split incompatible trajectories. Mark changes at
their actual effective epoch; compare adjacent observations without claiming
that temporal differences establish a causal effect.

## Prior before initialization

For Cashew, supply approved common-scale prior identity, region coverage,
confidence/unknown/conflict and available local agricultural evidence before
choosing adaptation or negative mining. Ask for a provisional prior-informed
view of local cashew abundance and competing evidence, not an invented number.

Keep these quantities distinct:

- Predicted area/fraction and its decoder, valid mask and scope.
- Prior score statistics and fractions above stated fixed common-scale cutoffs.
  Those are descriptive support, not calibrated crop coverage.
- Spatial overlap of predictions and prior on the same valid mask, if measured.
  Whole-tile prior cannot be subtracted from conditional monitor predictions.

Static prior context is a reference, not a learned improvement. A verified
regional crop-coverage estimate can be supplied separately with place/date and
uncertainty. Do not translate prior score0.9 into90% crop cover. High support
makes substantial cashew plausible and calls for preserving credible positives;
it neither labels every pixel cashew nor fixes a required area quota.

## Host interface

Run `scripts/build_expert_review.py --input /project/review_input.json --output
/project/reviews/E010`. It writes all observed points to PNG and CSV/JSON plus a
bounded brief with recent/baseline/extrema and event-adjacent exact rows.

Input keys: `run_id`, `branch_id`, `data_identity`, `cutoff_epoch`, `task`,
`controls`, `bounds`, `metric_specs`, `observations`, `events`, `prior_context`,
`open_questions`, `knowledge`, `experience`.

- Each observation binds the three identities, `epoch`, `checkpoint`, `metrics`.
- Metric specs name `key`, `label`, `unit`, `panel`, `definition`, `direction`
  (`max`, `min`, `none`). Effective controls can be observed metrics.
- Events bind the same identities, `epoch`, `status` (`applied`, `proposed`,
  `rejected`) and a short `summary`.
- Knowledge starts with relevant exact section excerpts, each with `source`,
  `sha256`, `heading`, `text`. The helper `section` extracts a verified heading.
   Experience contains explicit short lessons, not raw archives. Match these
   to the current mechanism; a fixed E0 lesson does not cover every later review.

`diagnostics` carries compact class confusion/support, paired prediction
transitions, opponent learning and active conditioning/feature evidence that
matter to the current question. The host must preserve computed relevant
evidence when compressing a packet; untransmitted values cannot support a
tools-disabled decision. `paired_errors.compare` checks same-point identities
and distinguishes reduced focus-class false positives from correct multiclass
reclassification.

Scope temporal claims explicitly. "Unchanged at E10 and E15" differs from
"constant across all observed epochs": an intermediate E5 exception invalidates
the latter even when E0 and E15 agree. For consequential numeric trend premises,
`temporal_claims.validate` checks the metric, interval, complete measured epoch
list and claimed relation against history. It distinguishes equal endpoints from
constant observations and does not infer unmeasured epochs. The host still
checks free prose for broader claims; schema validity is not semantic validity.

Definition strings and summary column names are shared in the serialized brief;
summary string aliases name another column of that row, preserving identical
exact endpoints without repeating them. Sparse anchors complement those values.
Full histories and unsmoothed plots remain intact. Budget-check the actual
observed history before dispatch, including end-of-interval reviews. A packaging
failure after checkpoint save is recoverable from that checkpoint, not a reason
to repeat completed training.

Use `knowledge_delivery` to state the selected topic, selection basis and actual
available read/supplement mechanism. If the model cannot open files, the caller
can fulfill a requested named section within the existing decision; record any
additional request and cost. Never claim a path was read. Optional supplement
requests need not delay a supported KEEP or ordinary direct adjustment.

Missing metrics stay missing. Required anchors/events that exceed the packet
budget cause an explicit refusal, not hidden truncation. Unknown fields in the
task card are not grounds to invent data.

`model_inputs(brief_path, vision_available=...)` returns compact text and a real
`localImage` item after checking the PNG hash. The host must transmit these items
to the model. Text-only mode explicitly reports no image delivered; exact anchors
and summaries remain usable. A path alone is not proof that the model saw a plot.
Count actual runtime and image tokens; JSON length is not a token-cost estimate.

ReviewDriver accepts `expert_brief` in `review_context` alongside the validator's
control baseline/evidence. It sends the brief instead of duplicating logs, static
context and recent observations. The brief itself must include current controls,
history identity, missing observations and pending questions. Frozen hosts keep
their original input contracts until explicitly migrated.

## Sources and limits

[Google's tuning playbook](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/scientific)
recommends simple starting configurations, scoped questions and analysis of
training curves before adopting complexity. Its [pipeline guidance](https://developers.google.com/machine-learning/guides/deep-learning-tuning-playbook/additional-guidance)
discusses training observations. These motivate the workflow, not a universal
Cashew recipe or a demonstrated model-quality gain from shorter prompts.
