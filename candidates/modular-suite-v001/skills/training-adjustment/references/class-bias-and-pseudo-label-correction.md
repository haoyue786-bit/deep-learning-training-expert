# Class bias and pseudo-label correction

Read when one pseudo-class becomes dominant, rare classes vanish from the accepted pool, class balancing is proposed, or a prior-correction method is being considered. Separate the distribution being estimated from the operation being changed: sampling, pseudo-label assignment, student loss and deployment decoding have different effects.

## Why predicted frequency is ambiguous

The target prediction distribution mixes true class proportions with the model's confusions. Even a balanced population can produce imbalanced pseudo-labels. Conversely, a naturally rare target class should not necessarily receive a large prediction fraction. Neither a balanced prediction histogram nor a histogram matching the source demonstrates correct target semantics.

A confidence filter adds another selection mechanism. The accepted frequency of class c depends on its prevalence, recognition errors and probability of passing the filter. A difficult but present class may disappear from the accepted pool. Raising a common threshold can increase precision on retained cases while amplifying differences in class coverage. High confidence alone cannot identify those missing classes.

Use available class-specific error evidence, accepted support and provenance to distinguish these explanations. Without adequate information, state which interpretation is unresolved and choose a provisional reversible treatment rather than claiming to have estimated the true population prior.

## Methods change different parts of the loop

|Approach|What it uses and changes|Useful condition|Main failure to consider|
|Improve valid class exposure|Changes which trusted labels/crops contribute gradients.|Relevant classes are missing from actual exposure rather than just poorly classified.|Oversampling noisy or redundant cases magnifies their problems; archive counts do not equal valid crop exposure.|
|Rebalance pseudo-class sampling|Resamples examples according to model-assigned classes.|Pseudo-class identity is sufficiently informative and useful minority examples lack exposure.|A mislabeled rare pseudo-class gets amplified; a uniform prediction target can conflict with genuine class imbalance.|
|Selective consistency/entropy, as in SENTRY|Uses transformation agreement to choose confidence-sharpening or confidence-reducing pressure.|The transformations preserve task semantics and agreement is informative.|A stable wrong prediction can agree across views; entropy maximization does not label it as a different class.|
|DebiasPL|Uses an EMA of model responses to alter pseudo-label assignment and a separate student margin.|A model-induced pseudo-label bias is plausible and the feedback statistic can track it.|Mean response also contains real prevalence; correcting a truly absent class upward can create errors.|
|CCA-LSC|Uses estimated target/source class ratios to reweight pseudo-label decisions, combined with conditional attraction/separation.|The target ratio estimate and pseudo-class semantics are informative.|An already biased estimate reinforces the wrong class; pseudo-label-based alignment can attract the wrong features.|
|Calibrated label-shift correction|Uses supported source calibration and a target prior estimate under stable class-conditionals.|The label-shift model is appropriate and identifiable from the available predictor.|Conditional or support changes violate the model; converged EM need not be correct.|

Read the suite's sampling/regularization and pseudo-label-feedback references for the first three routes. Label-shift posterior correction is developed in checkpoint selection. The procedures below explain the less interchangeable correction mechanisms; they are not automatic new capabilities.

## DebiasPL: distinguish assignment from the learning margin

Let q_hat be an EMA of the average weak-view class probability vector, initialized uniformly in the described method. It estimates a model-response statistic. It is not an independently identified true target prior.

The two main operations have opposite signs:

    pseudo-label logits: f_weak - lambda * log(q_hat)
    student-loss logits: z_strong + lambda * log(q_hat)

The first relatively boosts infrequently predicted classes when assigning pseudo-labels. The second imposes a larger required raw margin for those classes when fitting the student: its adjusted logit is depressed, so the student must learn a stronger raw separation to fit a selected label. Applying the same subtraction at both places is not the same method.

Illustration, not an experiment: suppose q_hat=[0.8,0.2], lambda=1 and raw two-class logits are equal. Pseudo-label adjustment gives the second class a relative advantage of log(4). In the student loss, that class instead suffers a relative log(4) penalty, requiring a larger raw logit to fit it. This shows why the sign and stage must be recorded; “subtract the majority prior” is an incomplete implementation description.

The paper motivates its correction through a counterfactual interpretation approximated by average model responses. The approximation assumes enough stability for the statistic to track bias. An abrupt objective change, real target imbalance or unsupported classes can undermine it. An EMA following the predictions smoothly is not proof that it has isolated model bias from semantic signal.

The reviewed implementation updates q_hat from original weak-view probabilities, matching the main-text equation rather than an appendix pseudocode line that uses corrected probabilities. It then adds the log-q_hat term to the strong-view logits. These differences affect the feedback loop. Inspect the active implementation, including cross-rank aggregation, masks and initialization, instead of combining prose and pseudocode into a third recipe.

An important state detail is that the inspected driver reconstructs a uniform q_hat after loading its model/optimizer checkpoint and does not store q_hat in that checkpoint. This is evidence of incomplete feedback-state continuation in that snapshot, not a measurement of its numerical impact. An adapted implementation would need an explicit state policy.

When useful, inspect the distribution and classes changed by the correction, accepted support, trusted task errors and contribution of the corrected objective. A larger or more balanced pool is only an intermediate response. The paper's ordinary FixMatch adaptation can harm a CLIP baseline while its corrected method helps under its studied protocol. Its gains with additional views or CLIP knowledge do not isolate the correction's effect or cost.

## CCA-LSC: bounded ratio correction and conditional alignment

In the paper, a preliminary model estimates target class proportions from selected pseudo-labels. With source proportions pi_S and estimated target proportions pi_hat_T, define M_c=pi_hat_T(c)/pi_S(c), then

    W_c = 1 / (h + exp(-sqrt(M_c)))
    pseudo_class = argmax_c p_c * W_c

The paper's h=1.5 is a reported setting. For M>=0 it places W in [0.4, 2/3), so the relative class multiplier is less than 5/3. This bounds which probability odds can reverse. It does not directly identify a reliable semantic boundary, produce an independently calibrated posterior or zero out an absent class: M=0 still receives weight 0.4.

If an assignment changes from A to B, the described confidence weighting uses B's original probability, not a newly certified confidence. A flipped pseudo-label can remain uncertain or wrong. Source proportions must have positive support for a ratio to exist. Estimating target proportions only above a threshold can undercount difficult classes in a class-dependent way.

The conditional-alignment component pulls same-pseudo-class source/target examples or centroids together and separates different ones. This can improve useful class structure when labels are informative; systematic pseudo-class errors can make it align the wrong populations more confidently. Confidence weighting does not guarantee removal of confidently wrong examples.

The inspected public snapshot has consequential differences from the paper: its distribution-estimation function counts `target[i]` from the dataset labels after a confidence gate, rather than the predicted class; the main path retains the final call with threshold 0.4 and uses a different source-ratio expression. It also reconstructs the classifier for the second phase rather than continuing the first classifier's full state. These are version-specific code observations, not proof that every reported paper experiment used that code. That snapshot must not be treated as a ready strict-unlabeled implementation.

The method's reported initial three epochs and confidence threshold are contextual recipe choices. They do not establish when a new Cashew tile's prior estimate becomes reliable or justify a mandatory high-probability gate for the user-authorized automatic-negative mechanism.

## Which response would support a correction?

For the chosen route, distinguish:

1. Whether the intended class-assignment or exposure change happened.
2. Whether the legal labeled evidence shows the targeted confusion improving without unacceptable loss of useful classes.
3. Whether the effect persists beyond the immediate altered pool and relevant schedule phase.

If only the histogram becomes more uniform, the mechanical operation worked but semantic improvement remains unknown. If supported minority recall improves while precision collapses, report the tradeoff rather than calling it solved. If corrected labels keep flipping between classes, inspect statistic lag, feedback strength, representation mismatch and actual exposure before adding another balancing mechanism.

Do not stack sampling correction, class weights, logit adjustment and prior correction without understanding their combined effective target. Several individually reasonable corrections can compensate for the same imbalance multiple times. Fixed source-class weights remain fixed where contracted. A pseudo-label procedure is also distinct from fitting a final map threshold.

For Cashew, the project allows broad Agent assessment of source-positive AEF discrepancy, rising p2 and prior context for weak negatives. These papers do not replace that permission with a required non-cashew prototype, confidence cutoff, geographic allocation or sample-count ceiling. They supply mechanisms to reason about feedback, not a new rigid selection rule.

## Sources and scope

- [Debiased Learning from Naturally Imbalanced Pseudo-Labels](https://arxiv.org/abs/2201.01490), Sections3–4 and appendix; [reviewed code](https://github.com/frank-xwang/debiased-pseudo-labeling/tree/3d68ccb52e73627252d6d761501dbe9185f50a39), `main_DebiasPL.py`, pseudo-label and q_hat updates, margin construction and resume path. Full driver and dependencies were not all audited.
- [Contrastive Conditional Alignment based on Label Shift Calibration for Imbalanced Domain Adaptation](https://arxiv.org/abs/2412.20337), conditional alignment and label-shift calibration; [reviewed code](https://github.com/ysxcj-hub/CCA-LSC/tree/4c78f079026cfce7b55b0bf36d45251f3d134ed7), `CCA-LSC.py` distribution estimation and phase transition, plus reviewed dataset helpers. Paper and snapshot differ; neither was rerun here.
- SENTRY and calibrated label-shift correction are developed with their sources in the suite's pseudo-label and checkpoint-selection references.

The sign example and multiplier bounds are direct mathematical reasoning. No new correction method was installed, trained or claimed to improve this project.
