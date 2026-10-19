# Mechanism-directed training procedures

Read when deciding which concrete observation or intervention can separate explanations of a training problem. These are useful designs, not a requirement to run a parameter sweep. A supported direct action on the existing trajectory may be preferable. Execution, permitted controls and adoption come from the actual worker contract.

## Choose the question before the branch

A local control contrast asks what changes if one control changes from this state under these conditions. A whole-method comparison asks whether a method works well with reasonable interacting settings. Holding every optimizer setting fixed can help the first question and bias the second. Google's tuning playbook calls these scientific, nuisance and fixed hyperparameters; the roles depend on the question, not the parameter name.

For example, “does a smaller model LR reduce the present update problem?” can use a local matched continuation. “Is AdamW better than SGD for this task?” needs appropriate rates and momentum/betas for each and a declared resource protocol. One optimizer losing under the other's recipe is a limited result. A factorial interaction contrast may be useful when two controls are suspected to interact, but does not justify opening all combinations automatically.

Match the complete starting state, effective exposure and observations sufficiently for the intended attribution. In adversarial work this includes both model and D states, their optimizers, modes/buffers, sampled data, schedule phase and what “one step” counts. Seeds do not compensate for shared mutable checkpoint tensors or different input lists. Choose a response horizon that can expose the mechanism; a regularization or sampling effect can be invisible in a tiny branch.

## Nine retained adversarial/optimization procedures

These retain the scientific content of the project's earlier experiment library while leaving its packet-specific bounds and branch budgets to the owning executor. No fixed treatment value or canned winning arm is implied.

### 1. Model-side adversarial pressure

**Use when:** an auxiliary/adversarial path may be driving harmful movement and the coefficient is active.

**Contrast:** change the model-side coefficient while preserving main supervision, D optimization, starting state and evaluation. Inspect the actual gradient path: the coefficient may weight only one part of a multi-head adversary.

**Measure:** weighted and unweighted gradient diagnostics on the named coordinates, actual model changes, relevant class errors where permitted, and source/target observations separately.

**Interpret:** lower weighted auxiliary pressure verifies a mechanical effect. Improved task errors support the proposed intervention at that state. Quieter outputs alone do not establish accuracy. If the ratio decreases while false expansion persists or worsens, the coefficient is not a reliable map-area control; investigate objective semantics, other paths, calibration or class support rather than merely reducing it again.

**Inactive contrast:** a coefficient cannot identify a path effect if that path is disabled, detached from the measured parameters or never executed.

### 2. Main-model step size

**Use when:** update magnitude/noise, inherited-feature damage or inadequate adaptation is a plausible cause.

**Contrast:** change the actual model rate or a justified group rate while retaining other active objectives. Confirm scheduler behavior; a base-rate change that the next scheduler step overwrites is not the intended treatment.

**Measure:** optimizer updates and task response by component, effective rates and exposure, not only raw gradient norms.

**Interpret:** a smaller rate reducing all movement supports motion control. It does not distinguish useful preservation from failure to learn. A larger rate improving head fit while damaging a useful backbone supports differentiated readiness rather than a larger rate everywhere. A weighted gradient ratio may increase under a smaller rate because the ratio is not the Adam update.

### 3. Discriminator learning rate

**Use when:** D's tracking or update stability may limit the signal given to the model.

**Contrast:** change D rate while preserving D cadence and the model's control settings. Inspect D's actual optimizer/betas and effective updates.

**Measure:** D fit on training features and an appropriate out-of-fit domain sample, logit/gradient behavior, model response and permitted semantic quality.

**Interpret:** better D training fit with poor out-of-fit performance suggests memorization rather than a better semantic signal. Better held-out domain discrimination can still encourage harmful alignment under class-mixture mismatch. An unchanged model response may mean short exposure, saturation, a weak coefficient or an irrelevant D path. Neither “D wins” nor “D stays near chance” has a universal quality interpretation.

### 4. Discriminator cadence

**Use when:** opponent tracking versus model movement, or unnecessary D work, is the specific uncertainty.

**Contrast:** vary actual D updates per model update while holding D LR fixed for the local attribution. State whether cadence also changes batches, normalization exposure or total compute.

**Measure:** completed model/D updates, D fit/generalization and model semantic response. Equal model steps and equal total runtime are different comparison protocols.

**Interpret:** extra D steps may improve tracking of moving features, overfit a narrow sample or add no useful information. More D steps are not equivalent to a proportionally larger D LR: intermediate gradients and optimizer history differ. A joint rate/cadence intervention is legitimate if that is the intended policy, but cannot identify their individual effects.

### 5. Conditioning temperature

**Use when:** an adversary conditions on model class probabilities and the sharpness of that signal is plausibly consequential.

**Contrast:** vary the conditioning temperature under the actual softmax, detach, clipping and normalization path. Determine whether it also affects source classification in this producer.

**Measure:** conditioning-vector distribution, active classes, clip exposure, relevant D/model gradients and task response. Temperature is not automatically an inference calibration control.

**Interpret:** softening can reduce domination by a confidently wrong class, but can also erase meaningful class structure. A tiny short response is inconclusive, especially if clipping or weak logits made the conditioning almost unchanged. Class-mixture bias can remain under softer conditioning. Do not infer a universal direction from a near-zero area difference.

### 6. Conditioning confidence cap

**Use when:** clipping is active on a meaningful portion of conditioning values and that distortion may affect the signal.

**Contrast:** change the cap while preserving the rest of conditioning and inspect whether the vector is renormalized afterward.

**Measure:** fraction and classes affected by clipping, vector mass and actual downstream change, followed by permitted task evidence.

**Interpret:** an inactive cap cannot explain an effect. Elementwise clipping without renormalization changes total mass as well as peak size; it is not ordinary calibrated uncertainty or the same operation as temperature. A mechanical signal change alone does not establish useful adaptation.

### 7. Model weight decay

**Use when:** supported generalization behavior or excessive parameter drift makes regularization a plausible mechanism.

**Contrast:** change the actual decay in the relevant groups under the known optimizer. Keep track of LR and update count because AdamW shrinkage accumulates through their product.

**Measure:** parameter norms/displacements, fit and permitted generalization errors over a horizon where decay can matter. Confirm no-decay groups and whether a parameter has no gradient or a zero gradient.

**Interpret:** a small short-branch null result can mean insufficient accumulated effect. More decay can improve a supported gap, worsen underfitting or erase useful transferred structure. Adam's coupled L2 and AdamW's decoupled decay do not define the same treatment by sharing a numeric value.

### 8. Discriminator weight decay

**Use when:** D memorization or excessive complexity, rather than simple tracking speed, is supported.

**Contrast:** change D decay while preserving D LR/cadence for this local question.

**Measure:** out-of-fit domain discrimination and signal behavior as well as training D loss. Check relevant D norms and the cumulative exposure.

**Interpret:** worse training fit with better generalization can be a useful regularization response. Worse fit on both may simply weaken an already inadequate D. A regularized but semantically inappropriate alignment objective remains inappropriate. Spectral normalization and input-gradient penalties constrain different properties; they are not synonyms for D decay or silently legal scalar substitutes.

### 9. Fixed-feature discriminator fitting

**Use when:** the Agent needs to separate “D cannot learn this available signal” from “D cannot track the moving model.”

**Contrast:** fix the model features and train D under declared rates/exposure, comparing an appropriate out-of-fit feature sample. This is a diagnostic/preparation operation only if the executor supports it.

**Measure:** D learning progression and held-out behavior, plus whether model buffers/mode changed during feature production. A cached feature set removes its augmentation/normalization variation and can make the problem easier than the live stream.

**Interpret:** successful fixed-feature fit shows learnability in that setting; poor live tracking may remain. Failure narrows attention toward representation/conditioning, optimization, labels or D capacity. Neither outcome establishes target semantic correctness. Adopting the prepared D is a separate treatment whose complete resulting state must be supported; some executors discard branch weights and adopt only scalar controls.

## Other informative contrasts

|Question|Useful design|Crucial interpretation|
|Does useful source learning explain the apparent gain without adaptation?|Continue a matched state with main supervision retained and the questioned auxiliary path omitted or weakened.|The initial checkpoint alone is not a continued-learning baseline. Disabled coefficients may leave forwards, buffers or other work active.|
|Can the model fit a clean small subset at all?|Use a controlled valid subset with a suitable fit objective and inspect predictions/gradients.|Success demonstrates local capacity/execution, not domain generalization. Failure may be label alignment or signal ambiguity rather than model capacity.|
|Is normalization drift causing regression?|Compare relevant parameter-frozen/buffer/mode policies or a buffer restoration when available.|A “frozen encoder” can still change through buffers or stochastic mode. Identify what was actually held fixed.|
|Do weak negatives reduce a particular confusion?|Compare actual weak-label contribution while preserving human positives and the main objective.|An empty pool or a zero-weight path cannot identify a negative-label effect. Raw count does not equal total weighted exposure.|
|Does more training explain a high threshold?|Continue a matched state and compare ranking, calibration and the declared decision rule on the same permitted population.|The threshold number alone does not identify duration. An exhausted schedule can make more epochs uninformative.|
|Is a new head ready to use inherited features?|If supported, compare head-only preparation with the intended joint start.|Source fit is not target quality; count preparation and preserve complete states.|

## Interpret the response at the right level

First establish that the intended treatment executed with meaningful exposure and a comparable observation. Then separate:

1. **Mechanical response:** did the control change its intended quantity or path?
2. **Task response:** did supported class errors or the declared objective improve, worsen or remain inconclusive?
3. **Persistence/transfer:** did the effect survive the relevant later phase or another independent setting?

An unchanged scalar metric can hide corrected and newly introduced errors. Preserve paired identities when permitted. A tiny difference, high variability or a confound should reduce the strength of the conclusion; do not convert it into a winning parameter by tie-breaking. A later endpoint can reverse an early ranking, so do not discard declared later observations after an attractive early result.

After a contradiction, specify what failed: exposure, mechanical premise, proposed semantic mechanism, chosen amplitude or persistence. Preserve a supported part while withdrawing the unsupported claim. KEEP, a direct change, a different observation, recovery and STOP can all be defensible; another branch is useful only if its result can change a decision.

## Source support and limits

- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), scientific/nuisance/fixed parameters, informative affordable comparisons and short-horizon transfer.
- Earlier Cashew Training Expert `experiment-library.md`, retained locally as `research/acquired/S078/experiment-library.md`: the nine named procedures, actual executor distinctions and historical evidence statuses. The earlier 32/64/128-step and later 5/10-epoch capabilities belong to their packet versions, not to this general reference.
- Mechanisms are developed in the suite's optimizer, numerical-stability, adversarial, domain-shift and pseudo-label references. The designs above are conditional synthesis, not newly executed experiments.

The earlier archive reports repeated paired artifacts for lambda, model/D LR, cadence and temperature; cap, decay and fixed-feature support did not imply demonstrated benefit in the sampled history. Keep that distinction when using the retained methods.
