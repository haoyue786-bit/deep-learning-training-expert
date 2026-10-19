# Pseudo-label confidence and feedback

Read for self-training, consistency/entropy objectives, weak automatic labels or
a confidently expanding predicted class. These routes depend on the current
model and can reinforce its existing mistakes.

## Separate the source of a label from its certainty score

A human label, source-domain label, model argmax, EMA-teacher prediction,
transformation agreement and weak environmental prior have different error
mechanisms. High probability means the model's scores are concentrated under
its own representation; it does not independently establish the true class.
Teacher and student can share a systematic bias even when weights differ.

Trace how supervision enters the objective: hard versus soft target, mask versus
weight, per-pixel versus per-image/batch confidence, detach path, refresh cadence
and sample exposure. A threshold number alone does not define the mechanism.

### A concrete implementation distinction

In the inspected DAFormer/DACS implementation, the proportion of target pixels
above a confidence threshold becomes a common target weight, broadcast across
the target batch before specific spatial exclusions and mixing. It is not a
simple rule deleting every individually low-confidence pixel. Thus raising the
threshold can weaken target supervision broadly without necessarily selecting
a purer set of trained pixels. Inspect `dacs.py` rather than borrowing the
usual interpretation of a confidence threshold.

The same implementation initializes teacher parameters from the student and
uses an early effective EMA coefficient capped by the configured coefficient.
The update iterates over parameters, and teacher prediction handles dropout/
drop-path explicitly. Do not assume an EMA name proves all running buffers,
modes and resume counters are automatically restored or averaged.

## Confidence, consistency and entropy answer different questions

Entropy minimization sharpens current predictions, including wrong ones.
Consistency can identify sensitivity to valid transformations, but stable
systematic errors also agree across transforms. A committee formed by several
views of one model is not independent expert testimony.

SENTRY uses agreement under transformations to choose entropy minimization or
maximization. Disagreement does not label a sample as the negative class; making
an uncertain prediction less confident does not identify its correct class.
Its additional pseudo-class balancing and information-entropy treatment change
the sampled learning behavior, not the natural population's true proportions.

Inspect whether a transformation preserves this task's label. A geometry change
must preserve spatial supervision; RGB intensity/colour changes cannot simply
be assumed meaningful for a learned feature embedding. Stochastic layers and
training-mode normalization can also affect agreement, so determine what the
consistency measurement actually varies.

## Track the feedback loop, not just acceptance rate

A typical failure loop is: an early confusing sample receives a wrong label;
that label is sampled or weighted; the next model becomes more confident on
similar inputs; a confidence filter accepts more of them. Increasing acceptance
can reflect reinforced error, not improved coverage. Raising a threshold can
remove difficult true positives while leaving confidently wrong negatives.

Useful diagnostics include class-specific accepted support, the concentration
of samples/regions contributing gradients, label refresh timing, and any legally
available error slices. Separate changes caused by the model from independent
context or human evidence. Without permitted truth, these observations can
motivate intervention but cannot prove target accuracy.

| Suspected mechanism | A relevant permitted response | What would remain unresolved |
|---|---|---|
| Unreliable weak labels dominate useful inherited behavior | Reduce their independent weight/exposure, delay their use or withdraw a suspect route | Whether another pseudo-labeler would be correct |
| Class has too little reliable exposure | Improve valid sampling or obtain allowed information rather than merely sharpen all logits | Whether apparent rare examples are truly that class |
| Teacher follows transient student noise | Examine EMA time scale, initialization, mode and update cadence | A slower teacher can retain an old systematic error |
| Model is consistently wrong on a confusing background | Reconsider conditioning, information and weak-label provenance | More transformation agreement alone cannot fix semantics |

These are candidates, not new executable capabilities. Use the current contract;
do not silently add a teacher or change architecture to satisfy this guide.

## Automatic negatives need their own reliability accounting

A sample far from a source-positive feature distribution can be a plausible
negative candidate, but may also be a positive under real domain shift. Rising
positive probability reveals a changing model response, not its true sign.
An external prior supplies context and can be wrong or uncalibrated. Combining
these signals is expert evidence assessment, not a proof of the inferred label.

Preserve a separate low-weight route, sample provenance, revisions and actual
exposure when the project authorizes it. Low per-sample weight does not imply
low total influence if an enormous repeated pool contributes most gradients.
Inspect loss normalization and contribution relative to trusted supervision.
Human/evaluation samples must not be silently repurposed as automatic training
labels. Reversal of a weak-label decision should not erase why it was made.

Do not impose a universal probability gate, geographic quota or fixed number
of accepted samples. Reliability and useful diversity depend on this task and
the observed feedback; preserve the supplied project's explicit choices.

## Worked illustrative reasoning

A model becomes increasingly confident that a forest-like region is the target
crop. Its augmentation predictions agree. One explanation is new true crop
coverage; another is shared feature confusion reinforced by pseudo-supervision.
Agreement strengthens the invariance claim but does not discriminate those
semantic explanations. Check permitted source-class support, contextual prior,
actual weak-label path and any independent authorized error evidence. A change
should target the suspected path, with positive retention considered alongside
false-positive reduction. A smaller final map alone does not establish success.

## Sources and boundaries

- Prabhu et al., [SENTRY, ICCV 2021](https://arxiv.org/abs/2012.11460), selective entropy and committee consistency; [inspected solver, 7d594a5](https://github.com/virajprabhu/SENTRY/blob/7d594a54b5d96d317f8c7be5296d8819a7f1644a/adapt/solvers/adapt.py). Its classification and transformation assumptions are not an AEF segmentation recipe.
- [DAFormer/DACS implementation, e6f6e04](https://github.com/lhoyer/DAFormer/blob/e6f6e041e7477b642106c134c1715fc934ccd210/mmseg/models/uda/dacs.py), `_update_ema` and pseudo-label/weight construction; [DAFormer paper](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html), rare-class learning and inherited-feature preservation.

The feedback-loop and automatic-negative examples are conditional syntheses,
not claims that either paper implements the supplied project's weak negatives
or that the examples have been run in this reconstruction.
