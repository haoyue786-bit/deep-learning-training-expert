# Domain shift, class proportions and negative transfer

Read when adaptation harms a useful predictor, a target class becomes implausibly
common, source and target responses diverge, or a feature-distance argument is
being used as a class label. Different distribution shifts suggest different
diagnoses; do not assume every gap is an LR problem.

## Identify the shifted relationship

Changes can concern input appearance, class proportions, class-conditional
features, annotation definitions, support coverage or several together. Standard
covariate-shift reasoning assumes a stable labeling relation; label-shift
reasoning assumes stable class-conditional distributions under the relevant
representation. A new sensor, changed object appearance and an annotation
policy difference need not satisfy either simple model.

Useful comparisons include source training versus source holdout, source holdout
versus permitted target feedback, class-specific errors and actual component
responses. Do not open final labels to complete a desired diagnosis. A source
score can remain healthy while target semantics are harmed; it can fall while
an authorized target objective improves. State the actual objective and tradeoff.

## Why marginal alignment can conflict with class structure

For feature `z`, the target mixture is
`p_T(z) = sum_c pi_T(c) * p_T(z|c)`. Even if corresponding class-conditional
features already match the source, different class proportions produce different
marginals. Demanding identical unweighted marginals can therefore create
pressure that conflicts with keeping classes correctly separated.

**Illustration:** source data are half crop and half background, while the target
has much less crop. Perfectly separable class features can still have very
different domain densities. Making the densities match is not the same problem
as preserving a useful crop boundary. This explains a possible pathway to
overprediction; it does not prove that a particular adversarial continuation
must increase crop area or that its true target prior is known.

Conditional methods improve the information available for alignment, but
predicted conditions can be wrong. CDAN's feature/prediction interaction and
FADA's domain-by-class coding do not automatically correct prior shift. A
class-conditioned cluster can be compact and confidently assigned to the wrong
class. Stable domain confusion or prettier embedding plots are not enough.

## Distinguish class mass from conditional support

Matching class-conditional densities, matching their supports and matching
unconditional densities impose different demands. A support-based method can
tolerate density differences but still relies on identifying corresponding
classes and estimating useful finite-sample neighborhoods. High-dimensional
nearest-neighbor distance depends on representation, sample coverage, outliers
and rare-class support.

A distant sample can be background, a shifted true positive, an uncommon source
mode or a preprocessing failure. A distance is meaningful only with its metric,
normalization and conditioning. Cosine distance, squared Euclidean distance and
a ratio of between/within-class dispersions do not share the same numeric range.
Never inherit a universal "maximum one" or negative-label cutoff from a different
statistic.

CASUAL provides a useful class-support perspective while retaining unknown risk
terms and assumptions. Its finite proxy and pseudo-conditions do not identify
every true target class. Some results require all class probabilities positive;
a truly absent target class is a distinct boundary, not a guaranteed solved
case simply because the method tolerates label shift.

## What prior estimation can and cannot recover

Under appropriate stable class-conditional prediction assumptions, let
`C_ij = P_S(pred=i, true=j)` be a source joint confusion matrix and
`w_j = pi_T(j)/pi_S(j)`. Then the target prediction proportions satisfy
`mu_T = C*w`. If instead using a column-normalized conditional matrix
`A_ij=P_S(pred=i|true=j)`, the corresponding equation is
`mu_T = A*pi_T`. Matrix definition and unknown vector must agree.

When classes produce nearly identical prediction patterns, these equations are
ill-conditioned or unidentifiable. Constraints such as nonnegativity and total
probability can yield plausible estimates without creating missing information.
Changing representations during an estimation window, missing source classes
or violating class-conditional stability can invalidate the interpretation.

Estimated proportions are not ground truth, and smoothing a stable estimate
does not make it correct. An environmental prior is another kind of evidence,
not automatically `pi_T/pi_S`, a source CE weight or a supervision confidence.
Do not change a project's fixed source class weights because a prior estimator
was described in a paper.

## Select an intervention according to the suspected path

| Evidence pattern | More relevant question | A potentially misplaced response |
|---|---|---|
| Healthy source classes, target confusion between similar vegetation | Conditional semantics, target support, weak-label reinforcement | Repeatedly optimizing D accuracy alone |
| Source classes deteriorate after adaptation starts | Shared loss pressure, normalization, parameter motion or labels | Assuming target benefit solely because domains look mixed |
| Errors follow a changed annotation boundary/policy | Label definition and spatial alignment | Increasing capacity or training duration without correcting the task |
| A class is naturally rare or absent in the target | Prior/support assumptions and effective supervision | Forcing balanced predicted area or sampling imagined positives |
| Extra weak negatives reduce area but hurt true positives | Weak-label reliability and aggregate contribution | Declaring success from smaller coverage alone |

The table nominates mechanisms; it does not decide their truth from the pattern.
Look for the observation or allowed intervention that changes the choice.
Protect useful inherited behavior selectively rather than assuming all original
features should remain fixed or all target motion is beneficial.

## Preserve domain knowledge without making it a hard label

Contextual information can make a region-wide dense prediction suspicious.
Its role is to direct attention toward inconsistent patterns and plausible
confusions. Rare genuine exceptions remain possible. A high suitability value
is not a calibrated crop probability; a low value is not an absence annotation.
Use the actual task's ecological and sensor knowledge alongside model trends,
source support and permitted independent evidence.

In a weak-negative route, retain provenance and reversibility. Investigate whether
the chosen negatives represent diverse confusing backgrounds or merely repeat
one model-dependent mistake. A low per-sample weight can still dominate through
large exposure. None of these cautions creates a mandatory probability threshold
or geographical quota; the expert must weigh the actual evidence.

## Sources

- Ganin et al., [DANN](https://jmlr.org/papers/v17/15-239.html): domain-adversarial representation learning, with task loss retained.
- Wang et al., [FADA](https://arxiv.org/abs/2007.09222), and Long et al., [CDAN](https://arxiv.org/abs/1705.10667): different conditional mechanisms and prediction-dependent information.
- Tachet des Combes et al., [Conditional Distribution Matching and Generalized Label Shift](https://arxiv.org/abs/2003.04475): mixture/prior conflict, identifiable estimation assumptions and limits. Its joint-error results are not an online monotonic-response guarantee.
- [CASUAL, AAAI 2025](https://ojs.aaai.org/index.php/AAAI/article/view/34166): conditional-support perspective, unknown bound terms and class-support conditions.

Mixture examples and the diagnostic mapping are conditional syntheses. Source
proportions, true labels and the best target checkpoint must not be inferred
from the existence of these theories.
