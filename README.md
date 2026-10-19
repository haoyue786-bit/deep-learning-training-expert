# Deep Learning Training Expert

A Codex Skill for evidence-based initialization, training diagnosis, tuning, bounded experiments, stopping, and independent checkpoint selection.

## Install

```bash
git clone https://github.com/haoyue786-bit/deep-learning-training-expert.git "${CODEX_HOME:-$HOME/.codex}/skills/deep-learning-training-expert"
```

Back up an existing installation first. Restart your Codex session, then invoke `$deep-learning-training-expert` with your task profile, training evidence, permitted actions and output schema.

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

## Modular professional knowledge candidate

The [English modular candidate](candidates/modular-suite-v001/README.md) contains the expanded professional references and nine cooperating Skill entries. Its [Chinese review index](reviews/modular/README.md) links the content review, per-file guide and coverage checklist. The [full review](reviews/modular-suite-v001/README.zh-CN.md) explains the actual mechanisms, concrete cases and use of knowledge. This nested review package does not replace the v028 files installed by the command above.
