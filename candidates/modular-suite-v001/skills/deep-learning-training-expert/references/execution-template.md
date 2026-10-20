# Fixed execution template and separate scientific record

Use `scripts/training_runtime/execution_template.py` when integrating a trainer.
The host supplies current controls, legal bounds, eligible actions/checkpoints
and integer parameters. It owns measured facts and artifact hashes.

The model returns one envelope:

```json
{
  "execution": {
    "action": "KEEP",
    "changes": [],
    "selected_epoch": 0
  },
  "training_record": "Free-form evidence, reasoning, uncertainty and next observations."
}
```

`changes` contains only `{"parameter": "name", "value": 0.001}` entries.
The trainer advertises the actual names and ranges. Never copy this illustrative
value as a recipe. KEEP retains current controls; ADJUST changes at least one
legal control; INITIALIZE uses the host's verified reference plus any justified
deltas. STOP ends further updates and selects a real eligible state. HOLD is an
explicit scientific block; ordinary questions in the record do not pause a run.
The host decides which actions exist at each training stage.

Call `response_schema` and send `instructions(schema)` alongside the measured
brief. Validate with `validate_response` before any change. Then `save_response`
writes `execution.json`, `training_record.md`, `host_facts.json`, and provenance.
The host must bind run/epoch/data/decoder/checkpoint identities itself and apply
controls only once after recording that transaction.

The explanation has no machine-enforced subfields. Include the evidence and
mechanism needed to understand the action, a useful expected response, and its
limitations. The program must not parse doubts, evidence requests, thresholds,
or parameter suggestions from this prose into executable changes.

This removes fragile metric copying and prose-field checks, not scientific
responsibility. An accepted template does not prove a good decision. Preserve
the record for audit, including the strongest continuation or selection
alternative. Missing narrative is recorded as an audit warning rather than a
second action channel. A host may request scientific review of a documented
defect; it must not silently rewrite the model's scientific answer.

For malformed executable output, preserve raw response, cost and error. Use a
bounded correction only after the previous call is confirmed terminal. Preserve
genuine HOLD; fulfill its substantive prerequisite before reconsideration.
Parameter limits, immutable state identity, data roles, update cap and atomic
application remain deterministic host responsibilities.

Minimal host integration:

```python
schema = response_schema(current_controls, allowed_actions, eligible_epochs)
# Send instructions(schema), the measured brief, and transport outputSchema.
validated = validate_response(
    response, before=current_controls, bounds=parameter_bounds,
    actions=allowed_actions, candidates=eligible_epochs,
    integer_parameters=integer_parameters,
)
save_response(review_dir, validated, measured_host_facts, provider_provenance)
# Apply validated["execution"] exactly once through the trainer's state journal.
```
