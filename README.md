# Deep Learning Training Expert

A Codex Skill for evidence-based initialization, training diagnosis, tuning, bounded experiments, stopping, and independent checkpoint selection.

## Install

```bash
git clone https://github.com/haoyue786-bit/deep-learning-training-expert.git /path/to/training-expert-source
cd /path/to/training-expert-source/candidates/modular-suite-v001
python scripts/install_suite.py --skills-root /path/to/codex-home/skills \
  --backup-dir /path/to/installation-backups/modular-suite
```

This installs all nine maintained sibling Skills. Keep the repository checkout
outside the discovered Skill directory: its root v028 is historical. Archive
obsolete Cashew routing/decision Skills outside discovery when replacing them,
while preserving pinned old experiment snapshots. Invoke
`$deep-learning-training-expert` with your task profile, training evidence,
permitted actions and output schema.

## Core behavior

Understand metric populations, units, timestamps and model state before attribution. Distinguish inherited and new components during initialization. Diagnose data, implementation, optimization and objective mismatch. Use actual responses to retain settings, adjust, request an optional bounded comparison or reconsider a hypothesis. Decide STOP independently from SELECT. Preserve uncertainty, failures and costs.

There is no mandatory probe menu, fixed optimal parameter set, universal stopping patience or guaranteed improvement. The worker owns execution, permissions and resource admission. Installation does not launch training.

## Contents

`SKILL.md` is the entry point; `references/` supplies focused guidance and `scripts/` supplies deterministic binding checks. `task_profiles/cashew_fada/` is a project-specific example. Its historical local account paths, proxy and hardware settings are examples requiring replacement by the actual task contract, not generic defaults.

## Version and evidence

v028 contains the exact 18 files activated locally. `MANIFEST.json` records each SHA-256. Python helpers were parsed and the original package was hash-verified before upload. This repository is the instruction package, not the full training executor.

Research included fact-understanding tests, 16 initialization runs, 192 frozen Cashew training runs and a bounded continuation/selection case. The 192-run study averaged +0.3987 F1 percentage points relative to fixed training, but an interval accounting for related observations included zero. The later bounded case lost to its fixed baseline on final evaluation. Stable superiority, reliable natural STOP and general-purpose accuracy gains are not established. Public-task comparisons are currently held while professional knowledge construction is completed.

No datasets, checkpoints, raw private logs or credentials are included. The interactive explanation is https://training-skill-in-motion.bravebow12.chatgpt.site .

## Verify

```python
import hashlib, json
from pathlib import Path
manifest = json.loads(Path('MANIFEST.json').read_text())
for name, expected in manifest['files_sha256'].items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
print('Verified', manifest['file_count'], 'Skill files')
```

## New modular suite

The existing modular suite is updated in place on 2026-10-21 with a fixed
execution envelope, separate free-form scientific record and first-use auxiliary
supervision guidance. The [current status](candidates/modular-suite-v001/README.md#status-and-evidence)
states the production integration boundary: automatic experiment requests and
paired-continuation adoption are not connected by the scalar-control template.

The [maintained modular suite](candidates/modular-suite-v001/README.md) contains nine cooperating Skills, callable real-data trial interfaces, bounded validation feedback and project-owned experience. Its default scientific model is `gpt-6.1-sol` with `medium` reasoning; new reviews decide every ten epochs while retaining five-epoch evaluation/checkpoints. The installation above selects this suite. See [actual test scope](candidates/modular-suite-v001/tests/REAL_DATA_VALIDATION.md) and [module design/status](DESIGN.md). The earlier [Chinese review](reviews/modular-suite-v001/README.zh-CN.md) is historical and retains its original construction-stage scope.
