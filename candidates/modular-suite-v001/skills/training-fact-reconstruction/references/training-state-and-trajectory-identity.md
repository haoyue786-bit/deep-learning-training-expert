# Training state and trajectory identity

Read when interpreting initialization, a resumed job, a recovery branch, an adopted experiment or a metric that may belong to another model state. Recover the state facts needed for the current scientific question; do not perform a full systems audit on every healthy review.

## Four starts that can look similar in a directory

|Operation|What is inherited|How to interpret the next observations|
|New initialization|Declared random/pretrained components and the selected fresh optimization state.|An initialization treatment; its randomness and loaded coverage matter.|
|Weights-only warm start|Selected predictor parameters/buffers, with new optimization and possibly new auxiliary state.|A new trajectory from learned weights; early changes can reflect restarted moments or schedule as well as chosen rates.|
|Declared full continuation|The participating training state saved at the cutoff, restored under its continuation contract.|Continuation to the extent supported by the actual state and loader/runtime semantics.|
|Adopted prepared or donor state|The explicitly selected branch/components and its declared optimization/state policy.|An adopted treatment with a changed lineage, not merely scalar tuning of the untouched main run.|

An epoch label or filename cannot distinguish these. A bare source checkpoint can preserve a useful trained head without containing the optimizer history that produced it. Conversely, a checkpoint with an optimizer key can still omit an active teacher, prior estimate or sampler state.

## What state affects the next update?

|State|Why it matters|A useful check when relevant|
|Predictor parameters and registered buffers|Define the inference function and parts of train-time behavior.|Loaded/missing/extra tensors, component mapping, normalization statistics and selected prediction branch.|
|Other trainable networks|A discriminator or second task model defines the coupled signal.|Which network states were saved/restored, their modes and active paths.|
|Optimizer groups and history|Moments, step counters, rates and decay affect the next displacement.|Whether the optimizer state maps to the intended components and groups, not merely whether a load returned.|
|Schedule and progress counters|Determine the actual next effective rate/phase.|Step/epoch interpretation, update order and whether configuration or restored state wins.|
|AMP scaler and update status|Can change scale, skipped updates and effective exposure.|Saved scaler, finite-step history and which optimizers actually stepped.|
|EMA/teacher or feedback statistics|Influence pseudo-labels and how quickly a changed student affects supervision.|Teacher parameters/buffers/mode, EMA counter, prior/confidence estimates and refresh state.|
|Random and data state|Control future batches, augmentation and stochastic computation.|Actual RNG states, loader/sampler policy, dataset identity, order and supported resume position.|
|Mode, freezing and gradient accumulation|Change forwards, participating coordinates and the next optimizer step.|Train/eval policies, requires_grad, partial accumulated gradients and microbatch progress if interruption occurred mid-update.|

Not every method has every state above. Identify the actual participating system rather than declaring a fixed key list universally sufficient. Some workers guarantee resume only at completed update/epoch boundaries, deliberately avoiding partial-accumulation restoration. State that boundary instead of claiming arbitrary exact recovery.

The PyTorch general-checkpoint tutorial explains why model weights alone are insufficient for continuation and how to save multiple model/optimizer pairs. Its illustrative dictionary is not a proof of complete RNG, scheduler, teacher, scaler or data-stream replay. The reviewed timm implementation provides a concrete richer example; the supplied worker may implement another policy.

## Loading success is narrower than functional inheritance

Inspect loading coverage when a missing or mismatched component would change the decision. A permissive key-loading policy can intentionally leave a new component random. Name changes, wrappers, changed channels and shape incompatibilities need their actual loader semantics; `strict=False` is not a general promise that every shape mismatch is harmless.

Separate these claims:

- A tensor is present in a checkpoint.
- That tensor loaded into the current component.
- The current objective traverses it and has a gradient dependency on it.
- The current optimizer updates it.
- It was historically trained under the claimed source objective.

One does not establish all the others. Forward hooks can establish a traversal in the observed path, not past optimization of every tensor. An unused legacy layer may be present and in an optimizer while receiving no relevant gradients. A root normalization layer may participate only in the adversarial extraction path. These facts can matter to an initializer without requiring a reset.

## A saved best must be the actual best state

The PyTorch tutorial warns that `best_state = model.state_dict()` retains references to the current state. Without serialization or a suitable independent copy, later updates can change the supposedly retained best. The metric from epoch 8 and an eventual export labeled “best” need an actual artifact correspondence.

The same issue can confound paired branches if one mutates tensors retained in the common in-memory start. Fresh deserialization or independent copies and a preserved original identity can prevent that aliasing. Do not infer independent starting state merely because both arms print the same epoch or seed.

For deployment, bind the selected weights, buffers, raw/EMA branch, preprocessing and decoder to the observation that supports them. A last full training checkpoint and a best inference artifact can legitimately differ; keep their roles explicit. A tracking model ID or manifest hash identifies its recorded object, not an unrecorded causal interpretation.

## Reproducibility has levels

Same integer seeds initialize some RNGs; they do not restore all random generators at the same future position. Python, NumPy global and Generator objects, framework/device generators, workers, augmentations and sampling can have distinct state. The data files behind a list or job path can also change. A restored shuffle seed does not repair changed labels.

Even identical intended inputs and random states do not ensure bitwise equality across framework versions, hardware, different kernels or numerical backends. Algorithm selection and floating-point accumulation order matter. Deterministic settings can reduce variability at a cost and can have version-specific behavior; an archived modern document does not describe an older installed runtime automatically.

Use the level needed for the question and report what is supported:

- **Recorded comparable setup:** configuration/provenance says the intended conditions match.
- **Matched state/input evidence:** relevant actual state, data and exposure identities are checked.
- **Numerically reproduced response:** a permitted rerun measured agreement under a specified environment and tolerance.

Do not promote the first level to the third. Lack of bitwise proof does not erase every useful observation; it limits precision and attribution. A clean within-state local contrast can be informative without establishing that an old multi-month experiment was exactly reproduced.

## Scalar adoption and weight adoption are different

A branch may be discarded after comparing two parameter settings. If its scalar is then applied to the original main state, the main model did not inherit the branch's useful fitting or its optimizer history. The result informs the choice locally; it does not become the main trajectory's next metric.

If a complete branch is adopted, bind the actual resulting state and describe what it contains. If only selected components are copied, record the mix and state policy. Population-based training illustrates why a successful final parameter vector can depend on a lineage of copied states and earlier settings; copying just the final rates to a fresh model need not reproduce the result.

Metrics from discarded, donor and main trajectories must not be concatenated into a fictitious smooth curve. Continued supervised learning, new auxiliary preparation and model selection can contribute separately to a gain. Preserve that distinction in the scientific interpretation and cost account.

## Compact recovery facts for the decision model

The worker should carry the complete resumable artifact; the Agent normally needs its identity, participating components, actual restoration boundary and consequential exceptions. It does not need every optimizer tensor in the prompt. Deterministic helpers can expose loaded coverage, effective controls, completed updates and artifact lineage.

If a call or worker observation is interrupted, reconcile whether its action completed before repeating a state mutation. An observation timeout is not a failed update. A returned proposal is not an executed intervention. A rejected output is not an adopted decision. These distinctions protect the scientific timeline without imposing an extra review model on every step.

## Worked state interpretation

Illustration: two runs begin with the same encoder/head weights at nominal epoch 100. One restores AdamW moments, scheduler and D; the other creates fresh optimizers and D and restarts warmup. Their epoch-101 difference does not isolate the Agent's chosen model LR. The update system and signal provider changed together. Treat the second as a warm-started treatment and preserve the inherited predictor's identity; do not call the first hundred epochs untrained or claim a complete continuation.

For Cashew, an E173 bare mapping and a full epoch-100 fixed-run checkpoint have different roles. Use the current project contract for fresh initialization versus fixed-run continuation, and retain source class weights. This general reference does not authorize reconstructing a missing optimizer state or switching a production run's start.

## Sources and scope

- [PyTorch Saving and Loading Models](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), state_dict, best-state reference behavior, general/multiple-model checkpoints and warm starts; archived S062 document.
- [PyTorch Reproducibility](https://docs.pytorch.org/docs/2.14/notes/randomness.html), independent RNGs, worker seeding, algorithm/numerical variability and cross-environment limits; archived S008 version2.14, not proof of a local runtime version.
- [Reviewed timm checkpoint saver](https://github.com/huggingface/pytorch-image-models/blob/c709c9a4a26af45d01221caadfdc6b00a9622630/timm/utils/checkpoint_saver.py), actual model/EMA/optimizer saving; related train-loop behavior is versioned separately in the archive.
- [Population Based Training](https://arxiv.org/abs/1711.09846), lineage, state copying and dynamic hyperparameter paths; no population training is required or launched by this Skill.

The state interpretation and worker/Agent division are synthesis for real training. No checkpoint was loaded into a framework, no GPU operation ran and no historical trajectory was rerun in constructing this reference.
