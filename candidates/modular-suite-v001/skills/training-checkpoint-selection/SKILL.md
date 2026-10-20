---
name: training-checkpoint-selection
description: Select an existing eligible deep-learning checkpoint for the stated deployment objective. Use after or during training when choosing an artifact; distinguish model quality, calibration and continuation state, and never use sealed test data to choose.
---

# Training Checkpoint Selection

For eligible checkpoints with authorized real development predictions/labels, compare the declared decoder's confusion, Precision/Recall/F1, support and both confusing-class error directions. Reconcile dataset/label versions and point identity before ranking. Record feedback-use history; it remains development evidence. Threshold fitting is separate and must not be silently added. Retrieve relevant same-project evidence and append the selection and later outcome only to the explicit authorized [project store](../training-experience-update/references/project-store.md); sealed-test outcomes do not select the artifact.

Choose among actual eligible artifacts using permitted evidence and the stated deployment objective. Training termination does not automatically select its last checkpoint. Use the caller's candidate identities and output contract verbatim.

For a real comparison, high thresholds, probability correction or missing target labels, read [checkpoint ranking, calibration and deployment](references/checkpoint-ranking-calibration-and-deployment.md). It explains attainable operating points, multiclass temperature effects, label-shift correction, abstention coverage and the complete predictor identity. Apply the method whose assumptions fit the evidence rather than treating all score changes as learning progress.

If checkpoint metrics, populations or times conflict, use [fact reconstruction](references/factual-reconstruction.md) before ranking. The shorter [selection principles](references/stopping-and-selection.md) supply cross-cutting context; do not automatically load every reference when the relevant comparison is already clear.

## Establish the comparison

Identify what every candidate actually contains and which state produced each observation. Missing or failed evaluation is unknown, not zero. Compare the strongest eligible alternative, including an earlier useful phase or a recovery state; do not narrow eligibility to whichever candidates the summary highlighted.

Where direct permitted validation exists, use the specified metric and relevant class errors, uncertainty and dependence. Accuracy, F1, cross-entropy and calibration ask different questions. A checkpoint with higher cross-entropy may still have higher decision accuracy. Do not invent a composite score to erase this tradeoff.

Where direct target quality is unmeasured, source preservation, distribution stability, entropy and domain confusion are limited evidence. Explain which candidate behavior they support and what remains unidentifiable. A map matching a weak prior is not a measured accuracy gain. Follow an explicit fallback criterion where supplied; otherwise state and justify a provisional criterion against the strongest alternative.

Use authorized validation feedback to rank eligible checkpoints. Record its label version, sample support, decoding and evaluation time, and distinguish per-class errors from an aggregate score. Validation may also fit a threshold when that comparison is authorized; record the fitted checkpoint/threshold pair and its development-score status. For Cashew, supplementary human annotations remain training samples rather than being split into an evaluation/test panel; the separately designated validation set supplies selection feedback.

## Separate weights from decision thresholds

For thresholded classifiers, record whether candidates were compared under a fixed threshold or a legally fitted threshold policy. A higher fitted threshold alone does not prove insufficient epochs, poorer ranking or better generalization. Inspect permitted score distributions, discrimination and calibration before attributing it to training duration. Fit thresholds only on authorized selection data and preserve an untouched final evaluation. Count that fitting as selection work.

If the deployment objective includes a constraint such as recall or false positives, compare its actual supported evidence. A negative-only panel cannot establish positive retention. Sparse, overlapping or repeatedly inspected samples limit confidence, even when their point estimate looks best.

## Commit a concrete artifact

Return the exact existing artifact, criterion, strongest alternative and evidence limitations in the caller's format. An inference checkpoint may be sufficient for deployment but insufficient for continuation; preserve optimizer, schedule, component, normalization and RNG state separately when continuation is needed. Selection is read-only unless the execution contract advertises a distinct mutation.

Freeze the choice before opening final test outcomes. A later oracle best checkpoint evaluates selection regret but is not evidence available to the original selector. Completion means a concrete defensible selection, not a claim that target quality is identifiable when it is not.
