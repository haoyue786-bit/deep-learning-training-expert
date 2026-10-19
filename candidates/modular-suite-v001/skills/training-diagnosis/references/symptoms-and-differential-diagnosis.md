# Symptoms and differential diagnosis

Read when training runs but does not learn the intended task, errors change
unexpectedly, or familiar curves suggest incompatible remedies. This guide
provides specific distinctions; it is not a compulsory debug sweep.

## Establish which learning problem the curves describe

A curve combines data, sampling, objective, mode and model state. Before calling
it underfitting or overfitting, identify whether the training and validation
numbers measure the same quantity. Augmentation, dropout, weighted loss,
ignored examples and batch reductions can make their scales differ even for
the same weights. A training loss above validation loss need not be a bug;
subtracting incomparable reductions does not estimate a generalization gap.

Keep source training fit, source held-out behavior, target permitted feedback
and unlabeled target proxies distinct. Source fit improving while target errors
increase suggests a different question from poor fit on both domains. Neither
pattern identifies a unique hyperparameter without further evidence.

## Poor fit: optimization, capacity or implementation?

First inspect an actual post-transform input and target entering the loss.
Shape errors can be silent when broadcasting is legal. Check class mapping,
ignored labels, alignment after spatial transforms, and logits versus
probabilities expected by the loss. Increasing capacity cannot repair a
shifted segmentation mask or a loss applied to the wrong tensor.

A small-subset fit can isolate whether this implementation can represent and
optimize those examples. Remove unnecessary stochastic augmentation and strong
regularization when the question requires a deterministic fit. Failure can
reflect too small a rate, saturated activations, blocked gradients, missing
optimizer parameters, a wrong objective or incompatible capacity. A random
label memorization test has a different interpretation and can require more
capacity/exposure; it is not mandatory on every run.

Success is narrower evidence: a sufficiently flexible model can memorize
corrupted labels and still pass a small-subset check. Return to independent
data and the intended pipeline before claiming correct generalization.

### Useful symptom combinations

| Observed combination | Important alternatives | Observation or action that distinguishes them |
|---|---|---|
| Loss stays constant and every parameter is unchanged | No optimizer step, detached loss, parameters absent from groups, or deliberately inactive branch | Trace the active loss dependency and a real successful update before increasing LR |
| Parameters change but labels are not learned | Wrong inputs/labels, unsuitable representation, excessive/insufficient updates, regularization | Inspect actual examples and component responses; a small matched subset can narrow the question |
| Loss spikes at a particular batch | Data corruption, rare but valid hard cases, reduction/empty-mask issue, numeric overflow | Reproduce the operation and sample identity; do not delete hard examples solely because loss is high |
| Loss oscillates with valid inputs | Update overshoot, stochastic batches/augmentations, changing objective or opponent | Examine timing, effective rate and batch composition; reduced LR is one hypothesis, not the diagnosis |
| Rare class never improves while common classes fit | Missing effective exposure, label mapping, crop exclusion, noisy class labels, representation | Count actual labeled class exposure and inspect examples before raising every loss weight |

## Overfitting versus distribution mismatch

If comparable training performance improves while independent same-distribution
validation degrades, memorization or a generalization tradeoff is plausible.
More appropriate data, regularization, augmentation or a different trajectory
may help. Reducing LR alone can merely slow further memorization.

If same-distribution validation remains healthy but deployment-domain evidence
degrades, investigate distribution and labeling policy. Two distinct validation
populations can separate these questions when both are permitted. This does not
license opening a sealed final test or manufacturing a precise additive
decomposition of bias, variance and domain shift from a few losses.

Repeatedly selecting against a small validation panel can overfit the selection
procedure even when network training never uses those labels. Geographic,
subject or temporal dependence changes the effective independent support.
Re-shuffling an already inspected test into development does not restore its
independence; it changes the study and must be disclosed.

## Diagnose forgetting and negative transfer separately

An inherited model can lose useful behavior while fitting source labels better.
Potential paths include an inappropriate adaptation objective, loss competition,
sampling bias, normalization drift or excessive motion in inherited components.
These imply different interventions: protecting a useful representation is
different from fitting a new head faster or reducing an unreliable target loss.

DAFormer's class-time examples motivate distinguishing late rare-class exposure
from early inherited-feature forgetting. Rare-class sampling targets the first;
feature-distance protection targets the second under an assumption that the
reference knowledge is useful. Their mechanisms are not interchangeable, and
preserving every reference feature equally can be counterproductive.

When a suspected adverse mechanism is disabled, inspect what remains active.
If target errors continue while adversarial updates are off, changing inactive
D controls cannot explain a remedy. Examine supervised/auxiliary updates,
buffers and the evaluation rule. A quiet map can still be wrong.

## Work through the strongest alternative

**Illustrative case:** validation Cashew precision falls, recall rises and
predicted area expands. Lowering LR might slow expansion; increasing negative
weight might shift the decision boundary; fixing an erroneous class mapping
would address another mechanism entirely. First ask whether errors are diffuse,
concentrated in one confusing class or caused by a changed decoding rule. If
forest-like samples are repeatedly accepted by an erroneous pseudo-label route,
investigate that route rather than call the whole pattern optimization noise.

**Counterexample:** area rises because previously missed true orchards are
recovered while precision is retained. The same area change now supports useful
learning. The observable error support, not the area direction, differentiates
the two cases. Without permitted target truth, retain both interpretations and
use contextual evidence to prioritize investigation rather than invent accuracy.

Use the cheapest consequential next observation. Some states justify a direct
adjustment; others require checking an input or contrast. An elaborate experiment
is wasted when its outcome cannot distinguish the explanations or change an
action. When the premise fails, revise that premise rather than perpetually
adding cautious language to an unchanged prescription.

## Sources and scope

- Karpathy, [A Recipe for Training Neural Networks](https://karpathy.github.io/2019/04/25/recipe/), end-to-end skeleton, overfit and regularize: actual input inspection and dependency reasoning. Its historical optimizer and pretraining opinions are not universal rules.
- [FSDL Training and Debugging](https://fall2019.fullstackdeeplearning.com/course-content/training-and-debugging), Debug/Evaluate/Improve: symptom alternatives and distribution-aware evaluation. Its teaching decompositions are not exact identification methods.
- [CS231n Part 3](https://cs231n.github.io/neural-networks-3/), sanity checks and learning diagnostics: inspect gradients, updates and layer behavior with architecture-appropriate interpretation.
- Hoyer et al., [DAFormer, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Hoyer_DAFormer_Improving_Network_Architectures_and_Training_Strategies_for_Domain-Adaptive_Semantic_CVPR_2022_paper.html), sections 3–4 and class-time/feature-distance analysis: concrete mechanisms and limited-scope examples, not a validated Cashew remedy.
