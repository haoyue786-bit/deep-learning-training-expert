# Adversarial objectives, signal quality and time scales

Read for an image GAN, a domain-adversarial predictor, or a proposed change to
D rate, D update cadence, model-side adversarial weight or conditioning. Start
with the actual objectives: similar names do not make these systems equivalent.

## Write down who minimizes what and which path receives gradients

An image GAN changes generated samples to match a real-image distribution.
A domain-adaptive classifier retains a supervised semantic task while making
source/target representations harder to distinguish. Its useful output is a
class prediction, not an image that fools D. Domain confusion can accompany
semantic collapse or class mixing, so GAN quality metrics and a domain loss
cannot substitute for classification evidence.

For a logistic image discriminator, let `a` be its realness logit on a generated
sample. A non-saturating generator loss has derivative `sigmoid(a)-1` with
respect to `a`; the original saturating minimax term has derivative
`-sigmoid(a)`. When D confidently rejects fake inputs, their scalar logit
derivatives differ substantially. Nevertheless the useful generator gradient
also depends on D's input derivative and the generator Jacobian: a strong scalar
coefficient alone cannot guarantee informative parameter updates.

Trace detach/freeze/step boundaries. A D parameter may receive a gradient during
a model backward pass and have it cleared before its own update; that is not
the same as D being optimized by the model loss. An advertised pause can stop
adversarial optimization while leaving additional forwards and buffers active.
Inspect the actual policy rather than infer execution from one coefficient.

## D strength is multidimensional

D learning depends on architecture, the current distinction, sample diversity,
objective, regularization, optimizer history, rate and actual update count.
High training accuracy with poor out-of-fit discrimination suggests a different
problem from high accuracy on independent domain samples. Near-chance accuracy
can reflect genuinely similar domains, insufficient D learning, mislabeled
domains, mixed classes or an uninformative measurement.

| Observation | Mechanism to distinguish | Why the obvious scalar can fail |
|---|---|---|
| D cannot fit fixed current features | Inadequate optimization/capacity versus invalid labels or inputs | Raising LR cannot repair a domain-label mismatch |
| D fits cached features but lags during joint training | Moving representation versus cache-specific overfit | More D work can track motion or memorize an irrelevant distinction |
| D is accurate but model task quality degrades | Useful domain signal versus wrong conditional alignment or excessive pressure | Making D even stronger can sharpen the harmful signal |
| D loss is quiet and model stops improving | Stable balance versus weak gradient, collapse or an inactive path | A flat loss does not choose between more training and objective revision |

These are questions to answer, not a rule that D must be near a particular
accuracy. Count data and work in a fixed-feature diagnostic; its result applies
to that representation and exposure, not to future semantic correctness.

## Rate, cadence and model-side weight control different mechanisms

Increasing D cadence adds optimizer updates and exposure for each model update.
Increasing D LR changes the step policy under its existing moments. Increasing
the model adversarial coefficient changes how its gradient combines with task
gradients; it does not directly train D better. One can compensate mechanically
for another in a small interval while producing different longer trajectories.

Use the suspected bottleneck to choose a direct adjustment or an informative
contrast. If local attribution matters, vary rate or cadence separately. If a
joint policy is justified, name and interpret it as joint. Retain G/model and D
update counts, preparation, inference cost and wall time rather than comparing
outer epochs alone.

TTUR motivates different coupled time scales, not a fixed numerical LR ordering.
Its image-GAN examples include both a larger G LR in one DCGAN setting and a
larger D LR in WGAN-GP. A BEGAN application did not show clear benefit. The
asymptotic assumptions of the theoretical results do not automatically hold for
finite constant-rate AdamW runs. Late ranking reversals also limit conclusions
from a short apparently successful prefix.

## FADA and CDAN condition on semantics differently

FADA's inspected discriminator emits domain-by-class channels and is trained
using soft class information from the predictor. For model adaptation, target
features are encouraged toward source-domain channels with target predicted
class conditioning. Wrong target class assignments can therefore supply a
systematically wrong alignment target. Source segmentation truth remains a
separate supervised path; it does not make every D class target ground truth.

CDAN instead feeds a feature/prediction interaction to a domain discriminator.
Its outer product explicitly exposes class-feature combinations. This differs
from FADA's output encoding. Their detach and entropy-weight paths also differ;
do not prescribe a temperature or gradient change without locating its effect.

In inspected FADA code, clipped soft class probabilities are not renormalized
after clipping. Their sum can change the effective example mass in a soft-target
loss. In that particular public trainer, temperature also touches source logits.
Another executor may expose conditioning temperature only. A prediction cutoff,
a soft-label clipping cap and a conditioning temperature are distinct controls.

## Regularization can affect the useful opponent, not just stability

Spectral normalization constrains a weight parameterization; a gradient penalty
constrains a particular derivative under sampled inputs; weight decay shrinks
parameters under its optimizer. Their scalar values and loss magnitudes are
not comparable measures of the same force. Weakening a regularizer can recover
D learning or amplify overfit. Replacing the mechanism changes the method and
requires an actual supported capability.

Before interpreting a penalty, identify its parameter/input support, its actual
gradient contribution, when it is evaluated and any lazy-evaluation scaling.
A large penalty scalar does not alone establish that it dominates an update.
Likewise, stable D/G losses do not establish a Nash equilibrium or useful
generalization, even when the mathematical game is correctly specified.

### Spectral normalization: sensitivity rather than rate

For an ordinary linear map, the largest singular value bounds its L2 input
amplification. Spectral normalization uses `W_bar=W/sigma_max(W)`. In a sequential
composition with appropriately bounded activations, the product of layer bounds
controls a network bound. This is a useful mechanism for limiting a D's
sensitivity; it is not evidence that its semantic alignment is correct.

It normalizes the largest value to one, including when that amplifies a matrix
whose original norm was below one. It preserves ratios among a matrix's singular
values, rather than forcing all of them to one. Differentiating through the
normalization adds a direction-dependent term; reducing D LR cannot reproduce
the method merely by matching a scalar scale.

The paper estimates the norm cheaply through power iteration, reusing the
previous singular-vector estimate as weights change. One iteration is an
approximation, especially at initialization, with a small spectral gap or under
rapid weight movement. Its convolution implementation reshapes a kernel matrix;
the full image convolution operator also depends on stride, padding and overlap.
Residual sums and pooling require their own bounds. Do not call an arbitrary
spectrally normalized D a certified exact 1-Lipschitz network.

The retained power-iteration state and its update timing matter to continuation
and eval behavior; inspect the chosen library implementation. The paper's
algorithm alone does not establish a current library's buffer/mode behavior.
Gradient penalties on sampled interpolated inputs constrain another property
and may be complementary. A version with a learned multiplier on normalized
weights can deliberately relax the fixed bound; it is a different parameterization.

Use this candidate when excessive sensitivity/unstable signals are supported,
while checking capacity and out-of-fit D behavior. It does not fix wrong class
conditioning, a label mismatch or a harmful source/target mixture objective.
The paper includes an interaction where high-rate SN with one D update performs
poorly and more D updates help, as well as comparisons affected by loss,
architecture and horizon. This prevents both “SN removes tuning” and “one D
update is always enough” prescriptions. A new normalization mechanism is a
method change, not an ordinary review's silently available scalar adjustment.

## Worked policy choice

Suppose inherited source recognition remains healthy, target recall erodes and
an expanding predicted class is not independently verified as correct. A
model-side pressure reduction or pause can investigate the adversarial pathway;
D-rate tuning investigates a different explanation, namely the signal provider's
learning. Compare them conceptually before spending work. If errors continue
under an actual pause, inspect remaining supervision and buffers rather than
retuning inactive D parameters. If a pause helps, do not claim every source of
harm was identified or that permanent removal is optimal.

## Sources and boundaries

- Heusel et al., [TTUR](https://arxiv.org/abs/1706.08500), sections2–4 and appendicesA4–A5: coupled dynamics, task-dependent rates, late responses and a null application.
- Wang et al., [FADA](https://arxiv.org/abs/2007.09222); [train_adv.py, a1c6403](https://github.com/JDAI-CV/FADA/blob/a1c6403963184a3427eda68cc94b03ff6143368a/train_adv.py): soft-label and update paths. This is the public street-scene implementation, not a substitute for the supplied project executor.
- Long et al., [CDAN](https://arxiv.org/abs/1705.10667); [loss.py, c904cd3](https://github.com/thuml/CDAN/blob/c904cd3f2fe092bb5175be1c5b6e42fa2036dece/pytorch/loss.py): conditional input and entropy weighting.
- Miyato et al., [Spectral Normalization for Generative Adversarial Networks](https://arxiv.org/abs/1802.05957), Sections2–4 and AppendicesA,C,E,F of the reviewed v1: parameterization, power iteration, cadence interactions and relaxed-bound combinations. Author code was not inspected for this source; do not infer a tested integration.

Logit derivatives and the diagnostic example are mechanistic explanations, not
newly executed experiments. No paper's default rate, accuracy target or epoch
budget is made universal here.
