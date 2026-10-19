# Normalization, freezing and domain mixtures

Read when initialization, a freeze/unfreeze, extra discriminator fitting or a
domain mixture unexpectedly changes predictions despite stable learned weights.

## Four different objects can change

Separate learned parameters, optimizer history, non-parameter buffers and
execution mode. Freezing parameter gradients does not freeze every buffer or
disable dropout. Changing to evaluation mode changes some operations without
turning autograd off. A no-grad forward can still use training-mode statistics
and update buffers. Identify what the actual model and loop do.

BatchNorm uses batch statistics in its usual training mode and tracks running
statistics for usual inference. Its affine parameters and running buffers are
different objects. LayerNorm normally uses current per-example feature statistics
in both modes and has no analogous running population calibration to transfer.
Architectures and optional flags matter: do not generalize default BatchNorm
behavior to every custom norm or to disabled running-stat tracking.

## Why extra forwards can be an intervention

If source and target forwards share BatchNorm buffers, their order, frequency
and mixture can change inference behavior. More D steps may involve extra
predictor forwards even when predictor parameters are not stepped. A coefficient
of zero may remove a loss gradient without removing those forwards. Therefore
"turn off adversarial loss" and "omit the adversarial branch entirely" can be
different policies.

Cached-feature discriminator fitting avoids some new predictor forwards, but
the cached representation may become stale as the predictor changes. It can
answer whether D can fit that representation; it cannot alone show that its
signal is useful on the current target stream.

## Choosing a normalization policy

| Choice | Why it can help | Why it can fail |
|---|---|---|
| Keep inherited running statistics | Avoid noisy or biased target updates when inherited inference is useful | Real domain shift can make those statistics inappropriate |
| Update statistics on a permitted representative stream | Adapt to a changed activation distribution without an intended parameter update | Wrong class mixture/order or unreliable samples can move predictions in the wrong direction |
| Freeze affine parameters but update buffers | Retain learned scale/offset while adapting population moments | This is not a fully frozen predictor and can still alter class decisions |
| Evaluate with current batch statistics | May reveal sensitivity to stale running statistics | Predictions depend on batch composition; it is a different delivery policy |

Do not adopt one policy by default for all transfers. Class mixture shift and
feature shift can both affect moments; calibration cannot identify which
semantic mapping is correct. Extra target exposure may improve background
errors and harm rare positives, so a negative-only panel cannot establish the
whole tradeoff.

For a normalization-only comparison, keep parameters, input ordering,
augmentations and the intended evaluation mode matched. Report exactly which
buffers and modes changed. If the executor cannot isolate this treatment,
describe a whole-policy contrast rather than claiming a pure buffer effect.
Any new calibration must use permitted inputs and is not automatically allowed
on final evaluation samples.

## Worked transfer diagnosis

An inherited predictor's tensors match the source checkpoint, but predictions
drift after extra target forwards while its optimizer is paused. A first useful
question is whether shared running buffers changed, not which model LR to lower.
If restoring only the verified original buffers restores the original outputs
under matched inputs and modes, that supports a buffer pathway. It does not
prove which buffer state is better for the deployment population. If outputs
do not restore, inspect modes, random transforms and other active state.

This is a diagnostic example, not a report of a newly executed experiment.

## Sources and limits

- [PyTorch BatchNorm2d, 2.14](https://docs.pytorch.org/docs/2.14/generated/torch.nn.BatchNorm2d.html): affine state, tracked statistics, momentum convention and mode behavior. The relevant runtime may differ.
- [Google Tuning Playbook](https://github.com/google-research/tuning_playbook), BatchNorm implementation details: batch/microbatch and moving-stat considerations.
- [Swin optimizer.py, fixed f82860b](https://github.com/microsoft/Swin-Transformer/blob/f82860bfb5225915aca09c3227159ee9e1df874d/optimizer.py): gradient-frozen parameter exclusion is separate from forward modes and buffers.

The domain-mixture contrasts above are mechanism-based reasoning under the
supplied execution contract, not a universal recommendation to recalibrate a
pretrained model or replace its normalization architecture.
