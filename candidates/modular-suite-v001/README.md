# Modular Training Expert: callable trial suite

This suite contains English professional knowledge, nine cooperating Skills and small executable trial interfaces. Repository-root v028 remains separately versioned. Install the nine siblings explicitly; a Git clone alone does not activate this nested suite.

## Professional capabilities

|Entry|Decision it supports|
|---|---|
|[deep-learning-training-expert](skills/deep-learning-training-expert/SKILL.md)|Maintain the scientific thread and use relevant knowledge across a real run.|
|[training-fact-reconstruction](skills/training-fact-reconstruction/SKILL.md)|Recover measurement definitions, populations, state and event timing.|
|[training-initialization](skills/training-initialization/SKILL.md)|Choose functional starts, component preparation and initial optimization.|
|[training-diagnosis](skills/training-diagnosis/SKILL.md)|Distinguish data, implementation, numerical, optimization and semantic failures.|
|[training-adjustment](skills/training-adjustment/SKILL.md)|Choose an applicable control change, reversal or KEEP on the current trajectory.|
|[training-experiment](skills/training-experiment/SKILL.md)|Design and interpret a decision-changing mechanism contrast when useful.|
|[training-stop](skills/training-stop/SKILL.md)|Judge sufficient training, useful continuation, pause or stopping.|
|[training-checkpoint-selection](skills/training-checkpoint-selection/SKILL.md)|Compare existing model-policy artifacts for the declared deployment objective.|
|[training-experience-update](skills/training-experience-update/SKILL.md)|Retrieve, transfer, narrow or withdraw conditional lessons from real feedback.|

The detailed [problem-to-knowledge navigation](skills/deep-learning-training-expert/references/knowledge-use-and-navigation.md) points to mechanisms, procedures, worked examples, interactions and counterexamples. Read the relevant body when it changes the current choice; one Agent can use several capabilities within its continuing context. Separate Skill directories do not require nine model calls, a parameter grid or a fixed nine-step review.

## Knowledge substance

The references cover component readiness and initialization; SGD/Adam/AdamW state and actual displacement; schedules, batch exposure and numerical stability; regularization, sampling and pseudo-label feedback; class-bias correction; Transformer and dense-prediction transfer; adversarial objectives and timescales; domain/label/support shift; gradient balancing, surgery and uncertainty weighting; physical remote-sensing context; concrete intervention designs; sufficient training and recovery; ranking, calibration and deployment; and mechanism-matched experience.

The Cashew profile separately preserves the source checkpoint/class objective, audited paths, AEF semantics, sample/prior roles, corrected-year identity, weak negatives, regional reasoning and actual historical counterexamples. Its local paths and execution settings belong to that project and the specified cohorts. Another task supplies its own profile and permissions.

Actual historical cases and numerical evidence are kept in the explicitly authorized owning project experience directory. The profile explains how to retrieve and interpret them; another installation does not inherit a private case database. Earlier bundled historical files were preserved as hashed project-owned imports.

Sources are cited in the relevant references. `PROFESSIONAL_KNOWLEDGE_SOURCE_MAP.json` retains hashes and interpretation scope for the archived originals and project evidence; local research paths in it identify provenance, not files supplied in this public repository. Original papers, private data, weights and account credentials are not bundled.

## Use and delivery

The intended suite layout is nine sibling Skill directories under a host's Skill root. A nested directory in this repository is not automatically installed or discovered. A tools-disabled caller must actually supply the selected instruction/reference bodies. The included `suite_packet.py` packages explicitly selected Markdown and deduplicates identical bodies within one request; it performs no scientific routing or model call.

Deterministic helpers calculate or bind facts. They do not establish a method's assumptions, infer target accuracy from unlabeled proxies, or prove that an Agent understood a reference. The actual worker contract defines executable controls, restoration, adoption and resource admission.

## Status and evidence

The 2026-10-20 upgrade includes actual real-data CPU kernel, gradient, matched two-update trial and inherited automatic-controller checks. See [validation scope and results](tests/REAL_DATA_VALIDATION.md). The trials supply narrow development feedback, not independent evaluation or established method superiority. Historical v028 studies retain their original scope. No large GPU run or production checkpoint migration is part of this installation.

## Install, configure and call

```sh
python scripts/install_suite.py --skills-root /absolute/codex-home/skills \
  --backup-dir /absolute/installation-backups/modular-suite
```

The installer verifies the build manifest and copies the nine Skill directories,
preserving existing versions in the specified backup directory. Restart the
Codex session to refresh discovered Skill text. The actual scientific request
defaults are centralized in
[scientific_defaults.json](skills/deep-learning-training-expert/scientific_defaults.json):
`gpt-6.1-sol`, reasoning `medium`. Credentials and transport remain caller-owned.

Copy [the example config](examples/cashew_trial_config.json) into your project
and replace its paths/identity/actual authorization. One config selects the
adapter, data, output, project experience and finite trial/validation budget.
Call `skills/deep-learning-training-expert/scripts/algorithm_trial_bench.py`
with `--config /absolute/project/trial.json`. Omit `--trial` for fixed-decoder
feedback only; use `--interface-smoke` to check all implemented kernels.
Read the [complete calling contract](skills/deep-learning-training-expert/references/algorithm-trial-bench.md)
for a bounded matched trial and the five-hook training bridge.

New bridge defaults: epoch telemetry every 1, evaluation/checkpoint every 5,
Agent decisions and automatic-negative refresh every 10. The same owner returns
the pool plan; no extra model call per algorithm or pool. Each selected automatic
patch occurs once over its finite generation horizon; persisted exposure is not
renewed on resume. Frozen legacy jobs require explicit migration.

Actual cases, predictions and validation-use history belong in the configured
project experience directory. Missing or mixed project identity/path fails
closed. Validation labels guide development decisions and never enter these
training losses. Threshold selection remains a separate task.

Run deterministic tests with a PyTorch/NumPy environment:

```sh
python -m unittest discover -s tests -v
```

The opt-in `tests/real_cashew_smoke.py` and `tests/real_controller_smoke.py`
require authorized real data and a pinned external runtime. Data, weights and
that production runtime are not bundled. Generic kernels, cadence, validation,
registry, project store and unit fixtures are supplied here.

The organization draws on [Scientific Agent Skills](https://github.com/K-Dense-AI/scientific-agent-skills) and [Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents](https://arxiv.org/abs/2609.00065): professional scope, progressive disclosure, useful procedures, supporting references and tools. Our entry's scientific-continuity role is a project design choice. The paper is not evidence that this training suite improves accuracy.
