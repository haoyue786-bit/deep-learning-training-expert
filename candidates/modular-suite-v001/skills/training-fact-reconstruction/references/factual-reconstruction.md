# Recover usable facts before attributing a training response

Use for first contact with a training task, uncertain metric meaning, or conflicting
records. Resolve the observations that can change this decision; do not inventory
every file before doing useful work. For routine reviews retain verified definitions
and inspect new evidence or changed identities.

## Find meaning and state, not just a matching number

Start from the run contract and the records available at this turn's cutoff. Locate
the actual metric-producing evaluation/configuration when names are insufficient.
Record the relevant model/component, data and label identity and whether evidence
is from training, validation, a probe, deployment prediction or an external reference.
For a numerical claim identify the statistic and denominator, class or conditioning,
population/sample support, aggregation, unit, observation time and measured state.
Unknown parts stay unknown. A zero emitted for undefined support or disabled work
does not establish measured failure or measured absence.

A mean over batches need not be a pixel/sample-weighted mean. A predicted-class
slice is not a true-class slice; a cumulative best is not the current model score.
Read the generating context rather than searching for the first similar key.

## Bind an effect to its actual experiment

A task, dataset or year can contain several different interventions. Before using
a historical response, identify the exact run/comparison, reference and actual
control vectors, and observed endpoint/exposure. Do not join an outcome to a
treatment merely because their task names match or their scores look similar.
When supplied, use the comparison ID and explicit control changes, then check
them against the record. A result for another learning rate or loss weight is
not evidence for an update-cadence change. If the binding is unsupported, withdraw
that historical claim; do not silently repair its provenance or discard the
whole remaining evidence. A short probe cannot establish a longer-horizon
expectation that was never observed.

When a supplied registry separates outcomes, use the observation's population
and metric identity together: a source-validation change cannot become a target
evaluation change by reusing a task name or an experiment ID. Recorded increase,
decrease or equality describes numeric direction, not significance or usefulness.
Use the advertised structured claim fields when present; the worker validates
their binding and renders the original values. Do not invent fields for another
schema, and do not assume a matching structured claim makes contradictory prose
correct. If the check rejects a consequential premise, reassess its implication.

Separate state lineage from resource accounting. Extra exploratory work is not
extra exposure of the final model if that state was discarded and the main was
restarted from its declared initial state. If adoption or restoration is absent
from the evidence, keep it unknown rather than declaring either comparability
or an exposure confound. Equal epoch counts alone do not resolve the question.

## Reconcile without inventing a winner

Follow a summary back to its original measurement when fields are missing or
inconsistent. Preserve both values and their scope when they answer different
questions; restore information actually present in originals. Check a derived flag
against its declared inputs rather than repeating it as independent verification.
Equal scope names and counts alone do not verify data bytes or pixel order.

Distinguish a historical log's report from a deduction about supplied code. The code
stored beside a log may be a different executable version. A changed metric label,
unit suffix or missing entrypoint is evidence of uncertainty; do not silently correct
the historical score using present code. State which identity or definition would
resolve a consequential conflict, using a targeted read when permitted.

## Restore the actual sequence

Find where counters advance and where measurement occurs. Epoch zero may already
follow updates; a resumed local counter need not equal total training exposure.
Separate requested configuration, successful construction, executed updates and
adopted/checkpointed state. A completion marker can follow an exception; one trial
ID can contain failed and retried attempts. Configured epochs are not observed work.
Scheduled changes do not imply a new Agent action. A post-observation action cannot
explain its baseline, and later evidence cannot justify an earlier decision.

Inspect a supplied checkpoint's actual container before claiming its contents.
A bare model-state mapping does not contain optimizer/RNG history or a saved epoch;
a filename or registry epoch is external provenance, not an internal saved counter.
Inventory unmatched state keys before calling them new active components. Compatibility
modules can exist in the model and optimizer without being traversed by its current
forward path. Distinguish loaded tensors, active computation and actual updates.

Keep a prior's original sampling population separate from the current model monitor.
Reusing a context helper does not establish that its sample count, spatial footprint
or whole-population wording matches this run. Derive those descriptions from the
current pinned frame; do not silently carry them over from the helper's first task.

## Deliver facts that can be checked

Cite the original file plus exact JSON pointer or line span for each decisive claim.
In the supplied fact-stage contract, give file and line bounds; the program extracts
the verbatim span. Numerical observations use their raw numeric value; nonnumeric
facts use a null value and their statement, so prose is not mistaken for a literal.
Separate recorded values, code-based deductions, calculations and unknowns. Give a
calculation's operands and units; compare compatible populations/statistics only.
In the fact-stage schema, each calculated operand needs its own recorded numeric
fact. Use its exact `facts[].id` in `left_fact` or `right_fact`; these fields are
references, not descriptions or paths. A second number mentioned only in a
statement is not a separately declared operand.
Do not resolve a version or semantic ambiguity just to fill a required field.

When a caller supplies a fact-only schema, obey it and stop before recommending
training actions. The [fact-stage helper](../scripts/fact_stage.py) can prepare an
explicit read-only source bundle, extract verbatim spans, and check source binding,
line bounds, direct numeric lineage and arithmetic. Its successful result is
mechanical verification only; a human or separate semantic review must still judge
population meaning, chronology, contradictions and omissions. A correct citation
to the wrong population remains a wrong answer.

Population/statistic labels in this interface are descriptions, not canonical
identifiers. The helper does not treat equal text as proof of comparability or
reject a valid comparison merely for different wording; it exposes both operands
for semantic review. A raw numeric difference alone does not validate its unit.


For a supplied comparison registry, the caller can pass its exact nonempty ID set to `scripts/evidence_claims.py:claim_schema(comparison_ids=...)` before dispatch. An opaque comparison ID stays identical across E5/E10 observations; the observation ID carries the endpoint. Constrained IDs prevent invented identities, not wrong metric/control attribution; semantic validation still applies. With no comparisons, the caller should constrain the claim array to empty.

For current measurements, a task adapter may supply a compact registry of typed
observations with exact source JSON pointers. Use its claim fields when advertised;
the worker renders values from the original packet with
`scripts/observation_claims.py`. Its definitions must be checked against the
metric producer: a hash cannot establish that an adapter chose the correct class
or denominator. Explicit named metrics reduce array-position mistakes; they do
not replace the originals or prove comprehension. The consumer checks packet
identity and declared semantics, while a semantic review still checks the
interpretation, omissions and contradictions in prose. A successful check of
structured claims is not permission to ignore contradictory reasoning.

For an existing `source_claims[].pointer` interface, the worker can use
[source locations](../scripts/source_claims.py) to enumerate the actual packet's
JSON pointers in its schema before dispatch (`bind_pointer_schema`), and resolve
returned claims to their original values (`resolve_claims`). Use these exact
locations; a logical epoch is not an array index, and a compressed history may
omit earlier rows. Binding prevents nonexistent locations, including off-by-one
indices; it does not establish that a real location supports the interpretation.
Preserve rejected original answers and their costs. Correct numeric prose does
not make an invalid citation executable, and the helper does not rewrite it.
