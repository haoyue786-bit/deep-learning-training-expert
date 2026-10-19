# Regularization, sampling and loss design

Read when class errors, poor generalization or changing training exposure make
"adjust the loss" ambiguous. Different mechanisms change different parts of
the learning problem; do not treat them as equivalent strength controls.

## Distinguish the objectives before selecting a remedy

Write the effective risk at the level that matters: which examples are sampled,
which labels and masks are valid, which per-example losses are weighted, and
how the batch is reduced. The natural deployment class distribution, sampled
training distribution and evaluation weighting can all differ.

For sampled class proportions `q_c` and constant class weights `w_c`, an
unnormalized expected log-loss scales class contributions with `q_c*w_c`.
Oversampling and loss weighting together can therefore amplify the same class
twice. This is an objective calculation, not a claim that the resulting model
will predict that exact frequency or that balancing is always inappropriate.
Changing exposure also changes gradient noise and repeated-example frequency.

### Reduction changes the meaning of a class weight

An explicitly computed mean of weighted per-pixel losses divides by valid
pixel count. A class-index weighted CE mean can instead divide by the sum of
valid label weights. The resulting gradient scale depends on class mixture.
Do not infer the reduction from a loss's printed name; inspect the actual call.

**Worked arithmetic:** two valid examples have unweighted losses `a` and `b`
and class weights 1 and 10. The pixel-count mean is `(a+10b)/2`; the
weight-normalized mean is `(a+10b)/11`. If the batch contains only the second
class, its class factor can cancel in the weight-normalized mean. Thus "weight
10" does not imply ten times every optimizer update, and multiplying all
weights by a common constant may cancel under one reduction but not another.
Empty masks, ignored targets and soft labels need their actual reduction rules.

Batch normalization of a loss can change more than its instantaneous scale.
For a constant classifier and random class counts n_c, a batch-weight-normalized
mean targets the expected normalized masses `E[n_c*w_c/sum_j n_j*w_j]`.
This is not generally the normalized population product `q_c*w_c`. With one
valid example per batch, positive finite class weights cancel entirely. Thus a
prior-aware head bias or a claim of doubled class emphasis needs the actual
sampling and reduction, not only archive class proportions and a weight vector.

This distinction also prevents false train/validation gap diagnoses when two
producers report different reductions. Preserve declared source weights in a
project that does not authorize changing them.

## Match the intervention to the suspected bottleneck

| Mechanism | What it changes | Evidence favoring it | A failure mode to inspect |
|---|---|---|---|
| Weight decay | Parameter shrinkage under the actual optimizer | Comparable fit/generalization suggests excess unconstrained complexity or drift | Can weaken useful inherited behavior; LR and update count affect cumulative shrinkage |
| Dropout / stochastic depth | Training computation and reliance on paths | Co-adaptation or overfit with sufficient capacity and compatible architecture | Can suppress a small/new component, affect train/eval comparison or interact with normalization |
| Label-preserving augmentation | Training invariance and effective diversity | Real deployment variation represented by the transformation | Invalid transforms alter labels, boundaries or feature meaning |
| Rare-class sampling | Frequency and timing of exposure | Classes are genuinely underexposed or excluded by crops | Repeats noisy examples; class co-occurrence prevents desired pixel balance |
| Class-weighted loss | Relative error cost and training gradient contributions | Objective should emphasize a supported class error under fixed exposure | Can reinforce label noise or combine with oversampling to overemphasize a class |
| Label smoothing / softer supervision | Confidence targets and penalties | Excessive certainty is relevant and labels/teacher warrant softer targets | Does not identify wrong classes; can alter probability calibration and threshold behavior |

This table is a set of mechanisms, not a menu that must be executed. Pick an
advertised legal intervention supported by current evidence. Keep architecture
changes separate from scalar edits and do not install a new method silently.

## Sampling: count what reaches the model

Archive class frequency is not effective exposure. A rare class may be present
in many images but absent from the selected crop, ignored at a boundary, removed
by a confidence mask or paired with many more common pixels. Inspect actual
valid labels and crop support before calling its loss coefficient insufficient.

DAFormer's RCS first emphasizes classes and then selects source images/crops
containing them. The sampled image still contains other classes. Its temperature
therefore controls a co-occurrence-dependent exposure policy, not the natural
target-area prior. Excessive emphasis can crowd out medium-frequency classes.
The policy is a useful example of targeting a mechanism, not a requirement to
force every tile to contain equal proportions.

Repeatedly presenting one rare example increases exposure, not independent
information. If that example is mislabeled or atypical, greater weighting and
oversampling amplify its influence. Consider label provenance and diversity
before escalating either. Missing true target classes cannot be created by
resampling existing wrong pseudo-labels.

## Regularization should preserve useful task information

More appropriate data can improve coverage, but more mislabeled or shifted data
can hurt. Augmentation needs a domain argument: a spatial transform for a
segmentation input must transform supervision consistently; an RGB color
operation need not preserve a learned multichannel embedding's semantics.
Normalization, resolution and object scale interact with these choices.

Increasing regularization can improve held-out behavior while worsening training
fit. When training already cannot fit a trusted subset, adding regularization
may target the wrong bottleneck. A smaller LR is not a substitute for every
regularizer; it can only delay further memorization. Conversely, an increasingly
confident model can improve decision accuracy while its CE worsens, so use the
actual objective rather than diagnosing from one curve alone.

Dropout probability conventions deserve inspection: the common PyTorch `p`
denotes dropping, while some teaching examples use a keep probability. Check
the implementation and whether inference disables the randomness. A config
change to an unused layer is not an executed regularization change.

## A worked imbalance diagnosis

Suppose positives are heavily resampled, also have larger class weights, and
recall is high while false positives concentrate in a semantically similar
negative class. Three explanations lead to different choices:

1. Relative positive exposure/cost is excessive: inspect the sampled objective
   and legal sampling or auxiliary-weight controls.
2. The confusing negative type is absent or mislabeled: seek appropriate allowed
   supervision or revise an unreliable weak-label route.
3. Features cannot distinguish the classes at the available input scale:
   merely increasing negative repetition may not supply the missing information.

Do not infer the winner from the class count alone. If only global probability
scale changed but ranking is retained, calibration is another question, not
evidence that the network must forget all adapted features. Explain the chosen
mechanism, expected class response and plausible adverse tradeoff.

## Sources and implementation limits

- [CS231n Part 2](https://cs231n.github.io/neural-networks-2/), regularization: mechanisms and teaching conventions; older claims do not settle adaptive-optimizer behavior.
- [Karpathy Recipe](https://karpathy.github.io/2019/04/25/recipe/), regularize; [FSDL](https://fall2019.fullstackdeeplearning.com/course-content/training-and-debugging), Improve: contextual intervention priorities, not guaranteed monotonic benefits.
- [PyTorch CrossEntropyLoss, 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.CrossEntropyLoss.html), class-index/soft-label reductions; [Dropout](https://docs.pytorch.org/docs/2.14/generated/torch.nn.Dropout.html), probability convention. Inspect the actual runtime version before inferring historical execution.
- [DAFormer source sampler, e6f6e04](https://github.com/lhoyer/DAFormer/blob/e6f6e041e7477b642106c134c1715fc934ccd210/mmseg/datasets/uda_dataset.py), `get_rcs_class_probs` and `get_rare_class_sample`.

Arithmetic examples are explicit derivations. They do not authorize changing a
project's fixed class weights or claim that a newly proposed policy has run.
