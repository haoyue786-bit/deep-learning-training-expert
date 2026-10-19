# Optimizer state and control interactions

Read when an LR, gradient norm, loss coefficient or optimizer change is being
interpreted as a change in actual learning. Optimizer names alone do not define
the effective update; inspect the implementation, parameter groups and state.

## Gradient size is not parameter motion

In plain SGD without momentum or decay, `delta(theta) = -lr*g`. With momentum,
an accumulated direction can continue moving parameters after the current
gradient changes sign. An approximate history scale `1/(1-momentum)` is useful
for intuition about a fixed coefficient, not a promise of that many useful
steps or a formula for choosing a new coefficient.

Adam keeps first- and second-moment histories. In a common bias-corrected form,
the adaptive direction is `m_hat/(sqrt(v_hat)+eps)`. Therefore two states with
the same current gradient and LR can have different next updates. Increasing
`beta1` retains directional history longer; increasing `beta2` makes the scale
estimate adapt more slowly. Whether either helps depends on noise and on how
quickly the objective is changing. A moving adversarial opponent can make old
statistics less representative; noisy small batches can make overly short
statistics unstable. Do not prescribe either direction from the task name.

`eps` can materially influence coordinates whose estimated scale is small.
It is not merely a harmless numerical constant, but increasing it is also not
a general cure for unstable training. Its position inside/outside a square root,
bias correction, RMSProp initialization and fused variants must match the actual
optimizer. Preserve these distinctions when borrowing a paper's configuration.

### Why loss scaling and LR scaling are not interchangeable

Scaling an entire loss by a positive constant scales SGD gradients, but adaptive
normalization can partly cancel that scaling after its histories adapt. Epsilon,
transients, clipping and decay prevent a universal equivalence. Scaling only an
auxiliary loss also changes the total gradient's direction and component support;
there is even less reason to treat it as a global LR multiplier.

For multiple losses, a weighted raw gradient norm ratio describes the measured
parameter subset at that instant. It does not include optimizer history or
unmeasured heads. A smaller ratio can arise from a changed numerator,
denominator or coefficient. Check which changed before calling it reduced
harmful influence.

## Weight decay depends on the update definition

Adding an L2 term inside an adaptive optimizer changes its gradient histories;
decoupled decay applies a separate parameter shrinkage. They need not produce
the same trajectory. Even for decoupled decay, LR and step count affect cumulative
shrinkage: under a simple AdamW-style implementation the decay-only multiplier
over updates is `product_t(1-lr_t*wd_t)` for parameters to which it is applied.
This identity is not a claim about total model behavior when loss gradients act.

**Arithmetic illustration:** with constant `lr=0.001`, `wd=0.1`, 1000 decay-only
updates give a multiplier near 0.905. Doubling update count strengthens this
shrinkage even with unchanged displayed WD. The numbers illustrate exposure;
they are not recommended rates, decay or a prescribed amount of forgetting.

Decay exclusions deserve their own check. The inspected ordinary Swin grouping
excludes one-dimensional parameters, biases and declared skip names/keywords.
The DAFormer custom-key example is a different grouping mechanism. Inspect the
actual resulting groups; a config key that matches nothing changes nothing.

## Frozen parameters, zero gradients and stale gradients

Freezing autograd does not by itself define optimizer behavior. Parameters may
be omitted from groups, retain a previous gradient, receive a zero tensor or
have `grad=None`. Existing momentum and decay treatment can distinguish these
cases. In the referenced PyTorch interface, a zero tensor and None can lead to
different step behavior. Do not assume that multiplying one loss by zero means
every associated parameter is frozen or that no optimizer work remains.

For unfreezing, determine whether the optimizer already owns the parameters,
whether state is old or new, and how the scheduler handles newly active groups.
Rebuilding the optimizer may discard useful state for other parameters. Loading
an old optimizer state can overwrite intended group settings or map state to
the wrong groups if ordering/identity changed. Inspect post-load groups.

Changing optimizer family is a method change, not a pure scalar adjustment.
Moment histories do not have an automatic equivalent in the new method. State
which histories are preserved or reset and which other rates/schedules require
reconsideration before attributing a response to the optimizer name.

## Measurements that distinguish common explanations

| Symptom | Evidence worth checking | Possible response when supported |
|---|---|---|
| Gradient norm falls but model still drifts | Actual per-group displacement, moments, decay, active losses | Address the active motion rather than repeatedly reducing an inactive loss |
| Scalar LR edit has no apparent effect | Effective post-load groups, schedule overwrites, actual optimizer steps | Fix or explain the effective setting before judging the scalar ineffective |
| Unfreezing abruptly harms inherited behavior | New gradients, buffers, group rate and old optimizer history | Consider gentler transition or selective groups; do not reset every component automatically |
| Very small gradients produce unexpectedly large movement | Adaptive denominator, epsilon and parameter scale | Examine actual update and implementation rather than a universal gradient threshold |
| New component stalls while another learns | Group membership, gradients, reduction scale and effective rate | Correct the specific path; a global LR increase can harm the healthy component |

When affordable observations already justify a modest adjustment, make it
directly. Full per-step tensor logging is not required on every healthy run.
Raw loss, gradient and displacement answer different questions; request the
one that can change the decision.

## Sources and implementation scope

- Google [Tuning Playbook](https://github.com/google-research/tuning_playbook), optimization-update FAQ and scientific/nuisance parameter distinction.
- [CS231n Part 3](https://cs231n.github.io/neural-networks-3/), updates, adaptive methods and update-scale monitoring; its older algorithm recommendations are contextual.
- PyTorch [AdamW, documented 2.14 form](https://docs.pytorch.org/docs/2.14/generated/torch.optim.AdamW.html) and [zero_grad semantics](https://docs.pytorch.org/docs/2.14/generated/torch.optim.Optimizer.zero_grad.html). These specify that release; inspect the run's installed version before asserting historical behavior.
- [Swin optimizer.py, fixed f82860b](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/optimizer.py).

The cumulative-shrinkage example and constant-loss-scaling discussion are
mechanistic derivations under the stated simplified updates. They are not
new training results or direct evidence for a particular Cashew intervention.
