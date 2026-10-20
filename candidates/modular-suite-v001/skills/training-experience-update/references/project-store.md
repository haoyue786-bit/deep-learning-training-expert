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
`record(case)`, `retrieve(tags=...,limit=...)`, `reserve_validation(...)` and
`finish_validation(...)`. A context file contains those four fields. CLI:

```sh
python scripts/project_experience.py --project-context /project/context.json \
  retrieve --tag M04 --limit 3
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
