# Checkpoint ranking, calibration and deployment

Read when selecting a model, interpreting a high class threshold, comparing score curves, or deciding whether a selection proxy reflects useful task quality. A deployable predictor consists of the learned state, preprocessing, output/EMA branch, decoding and any legally fitted calibration or threshold. Compare those objects explicitly.

## Match the comparison to the actual question

|Question|Relevant evidence|Different question it does not answer|
|Which checkpoint performs best under the declared decision rule?|Permitted metrics and error counts under identical populations, preprocessing and decoding.|Which model could perform best after separately tuning every threshold.|
|Which scores rank the positive class better?|PR/ROC or supported operating points for that score on the relevant labeled population.|Whether scores are calibrated or map area is correct.|
|Which probabilities better express uncertainty?|NLL, reliability information, Brier-type criteria when available, evaluated on appropriate populations.|Whether fixed-threshold F1 is higher or a new domain is calibrated.|
|Which model can meet a deployment constraint?|Supported false-positive/recall/cost evidence under the proposed rule.|A guarantee beyond the sample's support or at another prevalence.|
|Which model should be resumed?|A suitable complete training state and the rationale for future learning.|Which existing inference artifact should be delivered now.|

Start with the declared objective. Do not quietly replace it with training loss, the smoothest curve, or a composite invented after seeing results. When several objectives matter, show the actual tradeoff and explicit constraints. A null evaluation is not a zero score. An earlier checkpoint remains an alternative unless the task makes it ineligible.

## Threshold, ranking and calibration can disagree

For binary score s and decision s >= t, strictly increasing transformations preserve ranking and, with transformed thresholds, the attainable sets of positive predictions. They can move the numerically optimal threshold substantially without changing those sets. Consequently a shift from an optimum near 0.5 to one near 0.9 does not itself prove poorer learning, fewer epochs or worse discrimination.

The type of transformation matters:

- **Binary positive-temperature logit scaling:** sigmoid(z/T), T > 0, is monotone in z. It preserves ranking; fitting T changes probability calibration.
- **Multiclass common-temperature scaling:** softmax(z/T) preserves the argmax class at each example. A particular class probability can nevertheless change its ranking across examples because its denominator includes all competing logits. Do not claim multiclass p2 PR curves must be invariant.
- **Class-specific bias or vector scaling:** these can change class decisions as well as confidence. Treat their fitted parameters as part of the selected predictor.
- **Argmax versus a class threshold:** “class 2 wins” means p2 exceeds every competitor, not that p2 exceeds 0.5. In multiclass problems these rules need not agree.

A simple binary illustration, not a measured result: score sets [0.2, 0.4, 0.6, 0.8] and their squares [0.04, 0.16, 0.36, 0.64] have identical ordering. A corresponding boundary can move numerically while selecting the same examples. This cannot diagnose why an actual model's scores changed; inspect its paired predictions and errors.

A multiclass counterexample makes the distinction concrete. Let class 0 logits be A=[0,-1,-1] and B=[0,-0.1,-10]. At T=1, p0(A) is about 0.5761 and p0(B) 0.5250; at T=2 they become about 0.4519 and 0.5107. Both argmax decisions remain class 0, but their p0 ordering reverses. Common-temperature argmax preservation therefore does not imply invariance of a class-specific score ranking.

When a threshold rises, separate these possibilities:

1. Similar ranking and attainable operating points, shifted score scale: calibration or class-mixture effects are plausible.
2. Similar recall requires substantially more false positives: discrimination or the evaluated negative population has changed.
3. Fixed-rule F1 improves while peak permitted F1 barely changes: the score/decision interface may have improved more than the attainable discrimination.
4. Every apparent gain is in a repeatedly tuned small panel: selection overfitting is plausible.
5. Scores and errors change after additional epochs on the same population: training duration is relevant evidence, but compare the actual response rather than inferring it from the threshold alone.

Use the host's actual score diagnostics when available. A plot filename, score column or “confidence” label does not identify its mathematical meaning. Confirm softmax temperature, class index, margin/entropy, weighting and decoding from code when necessary.

## Fit and compare decision policies legitimately

Keep training, selection/calibration and final evaluation roles distinct according to the task's permissions. A permitted selection panel can support threshold fitting; a sealed test cannot be opened to choose. If data are too sparse to support separate fitting and evaluation, state that limitation rather than describing the in-sample maximum as generalization.

- For a **fixed-rule comparison**, use the same declared rule for all eligible checkpoints.
- For a **policy-fitted comparison**, give each model the same allowed fitting procedure and include that work in the comparison. Model-specific fitted thresholds produce different model-policy pairs.
- Preserve **ties, undefined metrics and support counts**. A negative-only sample can estimate false positives on its covered negatives. It cannot establish recall or identify the best F1 policy for a population with unknown positives.
- If choosing a threshold to meet a false-positive constraint, inspect the uncertainty around that constraint. Zero observed false positives on a few highly related points does not demonstrate zero deployment risk.
- For spatial data, distinguish number of pixels from independent places. Grouped or spatially separated evidence can be more informative than many adjacent pixels; use a supported uncertainty method rather than assuming IID samples.

F1 equals 2TP/(2TP+FP+FN) when its denominator is positive. If the evaluated ground truth has no positives and FP > 0, F1 is zero and recall has no support. If TP=FP=FN=0, F1 is undefined mathematically; libraries may fill a value. That filled value cannot identify a useful positive detector. State the metric convention when it affects ranking.

Selection across many epochs and thresholds is itself fitting. Keeping a final evaluation untouched helps assess the selected procedure. Checkpoint comparisons should use matched examples and a stable protocol; gains smaller than evaluation variability warrant modest claims rather than a fabricated certainty or arbitrary improvement cutoff.

## Calibration and label-shift adjustment are different operations

Calibration asks whether predicted probabilities match outcome frequencies on an evaluated distribution. A model calibrated on source is not automatically calibrated on target. Label-shift correction makes an additional assumption: class proportions change while class-conditional input distributions remain the same.

Under that assumption and accurate source posterior p_S(y|x), the target posterior is proportional to

    p_T(y|x) ∝ [pi_T(y) / pi_S(y)] * p_S(y|x)

with normalization over classes. This is a prior-ratio adjustment, not a new feature representation. The denominator requires supported source priors and a class with positive source support. Do not insert oversampled training frequencies in place of the intended source population without accounting for the sampling and weighted-loss mechanism.

The bias-corrected maximum-likelihood study first calibrates on held-out labeled source data, estimates source priors consistently, and fits target priors using unlabeled target probabilities. Its temperature-plus-class-bias and vector-scaling variants address systematic class bias that temperature alone may leave. Its empirical wins occur under the studied label-shift conditions; lower source ECE alone does not prove better target correction.

Important limits affect the actual action:

- If source predictions are nearly constant at the source priors, target data provide little information about target proportions. A concave likelihood need not have a unique informative optimum.
- If mixed forest and wetland produce new cashew-like features, class-conditional stability is doubtful. A prior adjustment may change the area without resolving semantic confusion.
- If the target contains an unsupported class or the source classifier cannot distinguish relevant classes, reweighting cannot manufacture missing semantic information.
- Stable EM estimates mean the optimization settled under its model. They do not demonstrate that label shift is true.
- A weak spatial prior is neither a calibrated target class prior nor a pixel label. A visually plausible adjusted map is a hypothesis, not an accuracy measurement.

Use prior correction when its assumptions and available evidence make it defensible and the task permits it. Otherwise it can still be an explicitly labeled sensitivity analysis, rather than an adopted accuracy claim. Reconstruct the exact scores and class semantics before applying any correction.

## Abstention changes coverage

Rejecting uncertain examples can improve performance on the retained subset. That is useful when the deployment process can handle rejection. It does not mean the original complete map became more accurate.

Report coverage, the retained population and the rejected classes/regions alongside the retained-set metric. Rejected pixels are unknown, not automatically background or non-cashew. The abstention study considers different metrics and label-shift assumptions; a policy optimal for retained-set accuracy need not optimize F1 or a geographic area estimate. A threshold optimized with labels is not an unsupervised selection method just because it is applied to unlabeled inputs later.

## Choose when direct target labels are unavailable

Source retention measures source retention. Target entropy measures concentration, not correctness. Domain indistinguishability can result from useful alignment, class-mixture matching or destructive collapse. Prior agreement can reject some implausible hypotheses without verifying the rest. These proxies must retain their distinct meanings.

Use a task-declared fallback criterion when one exists. Otherwise choose a provisional criterion that reflects the actual evidence and strongest alternative: for example, preserving a supported source predictor while declining an unsupported target expansion. Explain why that is preferable under the information available. Do not present a weighted sum of arbitrary proxies as identified target accuracy, and do not claim the selected model is objectively best on target.

An independent negative panel can make false expansion observable on its covered cases, especially if those cases resemble a known confusion. It does not establish positive retention or regional class proportions. Its training, selection and evaluation roles must remain clear.

## Commit the complete inference identity

Select the exact checkpoint and prediction branch, including EMA/raw identity and relevant buffers. Pair it with the preprocessing, class mapping, decoder and authorized calibration/threshold parameters that produced the supporting observation. A metric computed with EMA weights does not establish the raw weights' performance. Saving the last optimizer state does not prove that an exported best inference model is resumable from that point.

Preserve the strongest alternative and the reason for preferring the selected artifact. A later oracle comparison can reveal selection regret; future outcomes cannot be inserted into the original rationale. If the evidence supports only a provisional choice, make that choice concrete and retain the limitation.

## Source support

- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), evaluating models, checkpoint selection and metric/noise considerations.
- [Maximum Likelihood with Bias-Corrected Calibration is Hard-To-Beat at Label Shift Adaptation](https://arxiv.org/abs/1901.06852), Sections 2–4, calibration variants, prior adjustment and experiments. The local archive's legacy filename is `S039/abstention_paper.text.txt`; use the audited title, not the filename.
- [A General Framework for Abstention Under Label Shift](https://arxiv.org/abs/1802.07024), metric-specific abstention and coverage. Its local legacy filename is `S039/calibration_paper.text.txt`.
- [Reviewed timm checkpoint saver](https://github.com/huggingface/pytorch-image-models/blob/c709c9a4a26af45d01221caadfdc6b00a9622630/timm/utils/checkpoint_saver.py), concrete model/EMA/state-saving mechanics.

The threshold/ranking distinctions and numeric illustration are direct mathematical reasoning. They do not identify the cause of any historical Nigeria result without its actual paired evidence.
