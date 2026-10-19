# Metrics, populations and error slices

Read when a metric changes meaning across records, aggregate improvements hide
important failures, or a threshold/area observation is being used to diagnose
training. Recover the measurement before deciding what it means.

## Confusion counts expose what changed

For a declared positive class, precision is `TP/(TP+FP)`, recall is
`TP/(TP+FN)`, and F1 is `2TP/(2TP+FP+FN)` when defined. Retain the counts and
population with the score. A different class ID, ignored-label policy, matched
sample intersection or aggregation can change the question while leaving the
field name unchanged.

A panel with no ground-truth positive examples cannot measure positive retention or recall.
Its false-positive rate can still be informative. If false positives are present,
the count formula gives F1 zero, but that does not measure performance on unseen
positive examples; with no positive examples or predictions the F1 denominator
is zero. Software may return zero,
NaN or another configured value for undefined cases; report that convention
instead of interpreting an arbitrary zero as demonstrated poor positive
recognition. Likewise a predictor emitting no positives does not supply a
measured precision estimate merely because a library fills a value.

Pooling counts and averaging group F1 answer different questions. Micro-style
pooling gives larger populations more influence; an unweighted mean gives each
eligible group equal weight. A mean excluding undefined groups also changes
support. Do not mix them or allow groups with no evaluation points to disappear
without explanation.

### Equal totals do not identify equal errors

Two checkpoints can have equal TP/FP/FN but recover different positive points.
If paired identities are permitted, compare corrected, introduced and persistent
errors. Keep labels fixed or explicitly record revisions. Repeated geographic
points, neighboring pixels and the same location across years are dependent;
hundreds of such measurements do not imply hundreds of independent regions.

## Loss, ranking and decisions differ

Log-loss penalizes probability assigned to the true class, so becoming very
confident on a few mistakes can increase it while the number of correct argmax
decisions improves. Conversely, improving loss can leave decision counts
unchanged. Determine which objective matters, retaining the other evidence.

A strictly increasing transformation of a scalar score preserves ranking and
its attainable score-order partitions while moving numeric thresholds. This
explains why a higher fitted cutoff alone does not prove undertraining. Ties,
finite precision and a restricted threshold grid can affect observed searches.
Calibration, class-prior changes and representation quality require their own
evidence; a threshold cannot repair every ranking error.

Four-class argmax selecting class2 is not the same rule as `p2>=0.5`. Class2 can
win with probability below 0.5, and other delivery policies may add masks or
postprocessing. Resolve the actual output, raster sampling and ignored areas
before comparing probabilities, exported maps and point metrics.

## Select error slices that can change a decision

Useful slices often follow a suspected mechanism: true class, confusing class,
domain, annotation source, sensor/time, object scale, boundary distance, valid
crop support or a known preprocessing route. Separate a ground-truth slice
from a slice selected by the model's own prediction. The latter can change
membership as the model changes.

| Pattern | Question suggested, not automatically answered |
|---|---|
| Errors cluster at shifted boundaries | Are image/label transforms aligned and annotation policies comparable? |
| Rare positives are absent from actual crops | Is exposure rather than global model capacity limiting learning? |
| One background type supplies most false positives | Is its representation/support or pseudo-label treatment inadequate? |
| Source-domain holdout is healthy but target-domain errors rise | Is adaptation, label policy or target distribution the relevant mechanism? |
| Scores rescale but ordering is stable | Does deployment calibration explain more than a new training schedule? |

Slices chosen after inspecting many outcomes are exploratory. A small favorable
slice is not a new independent confirmation. Preserve total and slice support
and avoid claiming a complete causal error decomposition from their scores.

## Loss reductions and area measurements

Record whether a loss averages examples, valid pixels, class-weight sums or
already-averaged batch scalars. Unequal batch sizes make an unweighted
mean-of-batch-means different from a pooled per-example mean. Soft-label and
ignored-target reductions also need their implementation. Comparable numbers
require comparable definitions, not identical names.

Hard predicted class fraction, mean class probability, polygon area and sampled
point error rate are different statistics. Their units, denominator, coverage,
spatial resolution and state must be attached. Increasing predicted area can
mean improved recall or more false positives. An environmental prior can flag
an implausible pattern without supplying ground truth or a measured accuracy.

## Practical reconstruction order

For the current question, locate the producing record and model state; identify
class, population, decoding and reduction; recover counts and timing; then
interpret the relevant change. If only part is available, use that part with
its limits. Do not conduct every possible calculation on every healthy review.
Exact source pointers protect identity, not the correctness of an explanation.

## Sources and derivations

- [FSDL Training and Debugging](https://fall2019.fullstackdeeplearning.com/course-content/training-and-debugging), Evaluate and error-analysis material; separate training-domain and deployment-domain questions.
- [CS231n Part 3](https://cs231n.github.io/neural-networks-3/), learning curves and actual updates; [Karpathy Recipe](https://karpathy.github.io/2019/04/25/recipe/), trustworthy evaluation skeleton and prediction dynamics.
- Smith, [A Disciplined Approach](https://arxiv.org/abs/1803.09820), sections4.2/5.2: concrete disagreements between validation loss and accuracy ordering.

Confusion identities and monotonic-score reasoning are explicit mathematical
explanations. Project class indices and raster semantics apply only when the
current supplied contract confirms them.
