# Host execution and recovery

Use when binding the suite to a real trainer or recovering a partial review.
Installing knowledge and kernels does not migrate a production worker. Existing
jobs retain their pinned protocol. Normal scientific use does not require
reading this implementation reference.

## One scientific owner, explicit host capabilities

The ordinary action space comes from the host: KEEP/continue, existing parameter
adjustments, data/implementation inspection, recovery, STOP and separate
checkpoint selection. Research kernels are optional trials, not default
production controls. Advertise only supported operations and exact schemas.
The expert chooses a cause and useful next action; the worker validates bounds,
data roles, artifact identities and budget.

At each review supply the actual current configuration, complete state identity,
source and development errors/support, executed updates and schedule phase,
recent intervention/response, eligible checkpoint identities and evidence
cutoff. Include relevant conditional lessons and selected knowledge bodies.
A static description of the original job is insufficient after tuning.
Keep independent final evaluation outside this packet.

`ReviewDriver(..., run_identity={...}, scientific_context={...})` binds the
output directory to the run/branch, cadence and static contract. Use a fresh
directory for a new branch or migrating an older unbound output. Static context
can identify the exact schema and knowledge; `hooks.review_context(epoch,
evaluation)` optionally supplies a journaled compact dynamic context with
current controls, experience and legal capabilities. The host's `decide(packet)`
must actually deliver that content and dispatch the supplied scientific
model/effort, recording provider request ID and usage. Merely creating a packet
does not demonstrate model comprehension.

The existing five callbacks remain `evaluate`, `save_checkpoint`, `decide`,
`apply_decision`, `refresh_automatic`. A production integration must also supply
native `validate_decision(decision)` and `commit_review(epoch, receipt)`.
Set `production=True` to require the integration hooks and explicit contract/
knowledge identities at construction. This is an integration guard, not a
certification that the host implements the hooks correctly.
Validate action-dependent constraints before mutation, including STOP semantics,
checkpoint eligibility and any job-specific minimum epoch or hard cap. Commit
the resulting configuration, model/optimizers/scheduler/scaler, modes/buffers,
RNG/sampler and automatic-pool cursor as one recoverable state. The callback
journal alone is not that checkpoint.

For a same-turn automatic-negative plan the host supplies the actual current
catalog identity/support before deciding. Do not ask for a count against an
unknown catalog or make a hidden second scientific request inside a module.
Development use is reserved and recorded by the host before label feedback;
the trial bench's ledger does not automatically cover an unrelated trainer.

## Recovery without duplicate effects

Every stage is journaled before execution. `begin_operation(stage, operation_id)`
optionally passes a stable idempotency key to the host/client. A completed stage
is reused. An interrupted stage has an uncertain outcome, even if its Python
exception is called TimeoutError.

Call `resume(epoch)` after the host restores/reconciles the actual trainer.
`verify_resume_state(epoch, run_identity)` must return True only after checking
the complete state and its committed effects. Then
`reconcile_operation(stage, operation_id)` returns one of:

- `{"status":"completed","result":...,"evidence":...}`: consume the actual saved
  result without repeating its request or mutation.
- `{"status":"not_started","evidence":...}`: retry the stage within the configured
  finite attempt limit, using the same operation identity.
- Unknown/pending: leave it unresolved; no automatic retry or later epoch.

For a model request, a transport error alone cannot prove non-submission.
For mutation, the host checks its durable action receipt rather than guessing
from the desired configuration. Completed apply followed by failed refresh
must recover the refresh only. A genuinely new training trajectory uses a new
branch identity instead of editing earlier completed observations.

An invalid completed model answer is preserved. Before any apply attempt,
`replace_rejected_decision(epoch, decision, response_evidence=...)` can consume
a separately obtained corrected answer only if the native validator rejects
the original and accepts the replacement. The hashed JSON response receipt
must contain its request ID and exact normalized `decision`; retain the raw
provider answer and all usage outside the compact journal. This API does not
make a model call or authorize manually rewriting a scientific answer.

Automatic-pool restore refuses cursor divergence in either direction. Losing
the in-memory config does not authorize borrowing a later pool cursor; the host
must explicitly verify matching complete-state recovery before using
`resume_state_verified=True`. Do not renew spent samples to repair a mismatch.

## Evidence required before describing an integration as production-ready

Fixture tests exercise journal invariants and host contracts. They do not
prove a real provider call, complete checkpoint restoration or scientific
quality. Validate the actual chosen trainer with one isolated end-to-end
review and injected recoverable failure, then STOP and separate SELECT.
Keep the test's identity, original response, applied controls and recovered
state evidence. A narrow supervised rollout can follow successful integration;
superiority over fixed training or human experts requires separate fair outcome
evaluation and is not inferred from schema or unit tests.
