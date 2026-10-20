# Project-owned experience and development-use history

The generic Skill owns rules and executable helpers, never a project's actual
experiment database. Require an explicit `project_id`, absolute existing
`project_root`, dedicated `experience_dir` inside that root, and authorization
`{"write_experience": true, "source": "actual user/task authorization"}`.
Do not invent that authorization or default to Cashew, the current directory,
a Codex installation, another project's store, or a shared global memory.
Missing or conflicting identity/path means no write; continue read-only work
with already supplied facts and identify the missing context.

`scripts/project_experience.py` provides `ProjectStore(context)`,
`record(case)`, `retrieve(tags=...,context=...,cutoff=...,excluded_case_ids=...,excluded_units=...,limit=...)`, `reserve_validation(...)` and
`finish_validation(...)`. A context file contains those four fields. CLI:

```sh
python scripts/project_experience.py --project-context /project/context.json \
  retrieve --context '{"mechanism":"adversarial_pressure","source_profile":"source2019_benin_e173"}' --limit 3
python scripts/project_experience.py --project-context /project/context.json \
  record --case /project/completed_case.json
```

The dedicated store has a checked project identity, append-only
`experience.jsonl` and `validation_usage.jsonl`; concurrent reservations/writes
use a file lock. Supply actual evidence files with SHA256 using `file_evidence`.
The same case ID/content is idempotent; changed content needs a new correction
ID. Never erase an adverse outcome to repair its narrative.

Each case requires `case_id`, `project_id`, actual `action`, `result`, `outcome`,
hashed `evidence`, applicable `conditions`, unverified `hypotheses`, actual `cost`
and `evaluation_role`. Outcomes include positive, null, adverse, failed,
incomplete and observation. Preserve attempted/discarded work and measured
exposure, not just successful optimizer updates. An import is an observation
of an existing record, not a recomputed result or newly verified causal effect.

Before authorized development feedback, reserve its stable dataset identity,
label version/hash, checkpoint identity/hash, purpose and finite maximum uses.
Then finish with the immutable result or failure artifact. Failed exposure
still counts. An exact cached receipt does not count twice. Raising a later
caller limit or relabeling the same dataset does not erase past exposure;
do not create aliases to evade the record. This ledger does not authorize
threshold sweeps, sealed-test inspection or validation-label backpropagation.

Retrieve only a few matching same-project records and summarize mechanism,
actual outcome, conditions, contradictions and artifact pointers. A generic
reasoning principle may be proposed separately after evidence review; empirical
cases remain in their owning project. Cross-project reuse requires separately
authorized access and explicit attribution, never automatic store mixing.

## Distill first, retrieve second

Use `scripts/curated_lessons.py --project-context /project/context.json
--lessons /project/reviewed_lessons.json` to index an explicitly reviewed collection.
The document contains `project_id` and `lessons`. Each lesson has `lesson_id`,
`mechanism`, `conditions`, `observation`, `interpretation`, `use_when`,
`do_not_infer`, `counterexamples` and hashed original `evidence`. Optional `tags`,
`unresolved` and `evaluation_role` retain scientific scope. The importer adds
the reviewed collection's hash and records synthesis as observation, not a new
successful experiment. Revisions use a new ID and explicit correction pointers.

Prefer a handful of mechanism lessons over one entry per raw run. Keep exact
strata separate inside their evidence and do not add overlapping counts.
Separate mechanical effects, development quality and independent evaluation.
An unavailable target label cannot become an accuracy claim through repetition.
Do not embed private empirical records in the generic installed or public Skill.

The retriever prioritizes matched caller-supplied conditions over tag recency,
returns known differences and missing fields, and retains a compatible contrary
case when available. It does not establish causal equivalence. Query using the
actual mechanism and relevant source/model/data/horizon identities; do not
fabricate context just to obtain more matches. Unrelated recent receipts do not
fill a scientific query. At most a few lesson summaries enter the next packet.

For historical replay supply a timezone-qualified `cutoff` on knowledge
availability and exclude held-out case IDs/task units. A newly distilled lesson
was not available to an earlier historical decision even if its raw run is old.
Without a cutoff retrieval means current knowledge, not a sealed replay.
Original evidence pointers are opened only when needed to resolve a live issue.
