# Functional initialization and transfer starts

Read for a new model, new head or residual component, transfer initialization,
or a start that unexpectedly destroys inherited behavior. This is substantive
initialization knowledge, not a requirement to rerun all diagnostics.

## Identify the function being initialized

An initializer changes more than a tensor histogram. For a composition
`prediction = head(backbone(x))`, a new head can initially supply an unreliable
gradient to an already useful backbone. For `y = x + W_out h(x)`, setting only
`W_out = 0` preserves the inherited function `y = x`. Initially the loss gradient
to `h` through this branch can be zero while the outlet receives a nonzero
gradient. This is different from initializing an entire hidden network to zero,
which can leave symmetric units unable to specialize.

Before prescribing an initializer, identify the active path, inherited tensors,
new tensors, class semantics and trainability. A discarded auxiliary branch can
execute without affecting the trained objective. Conversely, an apparently
legacy normalization layer can affect an adversarial path. Parameter names,
forward hooks and optimizer membership each answer only part of this question.

### Initialization scales: when the textbook calculation helps

For independent zero-mean weights and suitably controlled inputs, the variance
of a linear sum grows with fan-in. Xavier's compromise scales variance with
`2/(fan_in + fan_out)`; the familiar He/ReLU fan-in construction uses `2/fan_in`.
These are assumptions about propagation and nonlinearities, not universal
replacement rules for pretrained attention weights or arbitrary residual blocks.
Convolution groups, matrix orientation and an initializer's fan mode matter.

Use the architecture's established initialization for genuinely new components
unless there is a reason to change it. Inspect per-stage activation and gradient
behavior when saturation, vanishing or an outsized new branch is suspected.
Extremely small random weights can weaken useful gradient propagation; smaller
is not automatically safer. A near-zero parameter makes update/weight ratios
ill-conditioned, so do not force every new outlet to meet a common ratio.

## Classifier bias and the training distribution

For a genuinely new binary logistic head and an intended unweighted training
prior `pi`, the constant-predictor bias is `log(pi/(1-pi))`. For a new multiclass
softmax head with negligible feature-dependent logits, biases proportional to
`log(pi_c)` produce the desired constant probabilities. A common additive
constant does not change softmax. Priors must be positive or treated explicitly
rather than producing accidental infinite logits.

Sampling and loss weighting change what constant prediction minimizes training
risk. For fixed sampled class proportions `q_c` and constant weights `w_c`, an
expected weighted log-loss with a fixed normalization has a constant optimum
proportional to `q_c*w_c`. That need not be the deployment prior. Merely saying
the denominator is independent of the prediction is insufficient: if every
finite batch is divided by its own sum of valid label weights, its random
class mixture also enters the denominator.

For class counts `n_c` in a batch and `S=sum_c n_c*w_c>0`, the constant optimum
for the expectation of that batch-normalized loss is
`p_c=E[n_c*w_c/S]`, not generally `q_c*w_c/sum_j q_j*w_j`.
The expectation of a ratio need not equal the ratio of expectations. For a
single-valid-example batch, its class weight cancels and the optimum is the
sampled `q`, regardless of positive finite weights. Larger representative
batches can make the population weighted-risk expression a useful
approximation, but correlated patches or varying class mixtures can matter.

For example, 10% sampled positives with positive weight10 and negative weight1
give a fixed-normalizer weighted-risk optimum about0.5263. With one valid
example per independently averaged weight-normalized batch, the optimum is
instead0.1. This is a mathematical illustration, not a Cashew head-bias recipe.

**Worked illustration, not a recommended Cashew initializer:** a binary training
stream with 10% positives and no class weighting gives bias about -2.20. If
positives are resampled to 50%, copying -2.20 from the raw archive no longer
matches the current stream. If the existing head is already trained, replacing
its bias also discards learned behavior. Check which of these situations holds
before using the formula.

Uniform-output CE near `log(C)` is a useful sanity reference only when logits,
loss and reduction imply it. With uniform logits and a nonempty class-index
weighted mean normalized by label-weight sum, CE is exactly `log(C)`; a
pixel-count mean of weighted CE instead scales it by the mean valid class
weight. Ignored pixels, sigmoid versus softmax, inherited predictions and a
nonuniform head also affect the relevant calculation.
An unexpected loss should prompt a calculation under the actual definition,
not an immediate optimizer change.

## Preserve useful inherited structure while allowing adaptation

Different component rates, a short head preparation or temporary backbone
freezing can reduce disruption from a new signal. Their rationale is strongest
when inherited task behavior is useful and the new component has not yet learned
an appropriate interface. It is weaker when the inherited representation is
inadequate or the old class semantics do not match the new task.

A frozen-backbone probe can reveal whether a simple head can exploit existing
features, but poor probe performance need not prove the backbone is useless:
head capacity, normalization, sampling and optimization also matter. Successful
head fitting on cached samples is not evidence of target generalization.

Use complete states when comparing prepared starts. A preparation whose weights
are discarded supplies evidence; an adopted preparation changes the actual
training method and exposure. A new discriminator can become more accurate at
the wrong distinction, so its fitting loss alone cannot justify adopting it.

### Transfer configuration is evidence, not a universal recipe

The inspected DAFormer configuration uses AdamW with a head LR multiplier and
no-decay custom keys for normalization/positional components. It supplies an
example of treating component roles differently, not a rule that all heads need
ten times the rate. Its backbone, objective, RGB input and supervision differ
from another trained segmentation decoder.

The inspected Swin code has distinct ordinary and SimMIM optimizer branches.
A layer-decay field does not make layerwise LR active in an entry that calls
only the ordinary grouping function. Verify actual parameter names, groups and
effective rates after configuration overrides and optimizer restoration.

## If the first interval is disappointing

| Observation | Useful distinction | Consequence for the next choice |
|---|---|---|
| New head fits, inherited validation deteriorates | Excessive backbone movement versus changing buffers or data | Group rates/freezing may help; compare modes and actual component motion |
| New head does not move | Missing optimizer membership/gradient versus intentionally delayed residual gradient | Fix execution if absent; do not inflate all rates for a structural zero |
| Frozen features fit cached samples but fail held-out inputs | Capacity, overfit or domain mismatch | More head warmup may strengthen the wrong signal |
| Initial outputs change after a supposed exact resume | Weights-only load, mode, buffers, input preprocessing or optimizer reset | Restore identity before applying fresh-start heuristics |

These are diagnostic hypotheses. A particular action follows from the supplied
task and actual evidence, not from the row alone.

## Sources and derivation status

- CS231n [Part 2, Weight Initialization](https://cs231n.github.io/neural-networks-2/#init): fan scaling and symmetry; teaching assumptions do not cover all pretrained architectures.
- Karpathy, [A Recipe](https://karpathy.github.io/2019/04/25/recipe/), end-to-end skeleton and initialization: functional checks and output-scale examples. The weighted-prior calculation above is an explicit algebraic extension, not a quoted author recommendation.
- [Swin optimizer.py at f82860b](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/optimizer.py), `build_optimizer` and grouping branches.
- [DAFormer configuration at e6f6e04](https://github.com/lhoyer/DAFormer/blob/e6f6e041e7477b642106c134c1715fc934ccd210/configs/daformer/gta2cs_uda_warm_fdthings_rcs_croppl_a999_daformer_mitb5_s0.py), `optimizer.paramwise_cfg`.

Project component identities belong in the supplied project profile. These
mechanisms do not establish an optimal numerical initializer for a new tile.
