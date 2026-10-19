# Training sufficiency, delayed response and recovery

Read when deciding whether a flat, noisy, damaged or apparently good run merits another interval. Stopping is an allocation decision about future training; selection chooses an existing model. A useful earlier checkpoint can coexist with a reason to continue.

## Establish the learning exposure

An epoch is a data-loader traversal, not a universal unit of learning. Relate the trajectory to actual optimizer steps, valid examples and schedule phase. Gradient accumulation, dropped tails, skipped AMP updates, D-only updates, oversampling and component freezing can all change exposure. A warmup epoch with tiny updates is different from a mature epoch at the working rate.

Construct the relevant timeline: full starting state, actual model updates, LR/phase transitions, changes of loss or sampling, observation times, and which state produced each metric. Use the declared primary objective and population. Keep raw noisy observations; smoothing can help visualize a trend but can hide a short damaging phase or invent apparent response lag.

The question is not whether loss could ever fall further. Ask what another legal interval can plausibly improve, whether that improvement matters for this task, and which observation would distinguish it from noise or damage. Quality can be the priority even when training is more expensive; do not substitute a cost preference for that objective.

## What kind of plateau is it?

|Observed pattern|Mechanism worth considering|Useful next action and interpretation|
|Training loss oscillates at a working LR while useful validation errors change slowly.|Updates may be too noisy for refinement; a planned decay may unlock progress.|Observe the scheduled transition, or consider an authorized rate adjustment. A gain after decay supports phase dependence, not proof that every plateau needs a smaller LR.|
|The run spends most of its exposure near the schedule floor; training remains poorly fit.|Updates may be too small or a short-run schedule may have been stretched incorrectly.|Inspect effective rates and update sizes. Extending at a negligible rate can repeat the same failure. Replanning the schedule is a treatment, not exact continuation.|
|A newly activated head, objective or sampler has received little relevant exposure.|Apparent delay can reflect component readiness or rare-class opportunity.|Observe enough real exposure to test the stated mechanism. If gradients or valid examples are missing, waiting does not solve that defect.|
|Training fit improves; permitted same-domain validation repeatedly worsens in the relevant errors.|Overfitting is plausible under comparable populations and evaluation.|Preserve a useful earlier model; regularization or stopping may help. A changing validation sample or preprocessing difference weakens this diagnosis.|
|Source quality is retained; target proxy confidence or map density rises.|The target could improve, become miscalibrated or suffer negative transfer.|Inspect available semantic/error evidence and the adaptation mechanism. Proxy saturation alone establishes neither convergence nor target damage.|
|Both training and validation remain poor with little parameter movement.|Wrong optimization, frozen/missing paths, excessive regularization or insufficient signal may dominate.|Diagnose the missing progress before granting more epochs. “Still underfit” is not an explanation of why further identical training will help.|
|Successive useful checkpoints improve through the latest observed phase.|The run may remain compute-limited or simply have unfinished training.|Continue when permitted and worthwhile; an arbitrary patience rule would discard observed opportunity. Quantify improvement in the task metric rather than the smoother loss alone.|

Loss and decision metrics need not plateau together. Cross-entropy can rise while accuracy improves if a few wrong predictions become more confident. F1 can remain unchanged while scores and margins move toward a later decision change. A high fitted threshold can arise from score calibration, class mix or cost preferences; it is not by itself evidence of too few epochs.

## Choose an observation horizon from the mechanism

Do not replace one universal patience with a universal minimum epoch. The relevant horizon depends on what must occur:

- **Scheduled refinement:** include the transition and meaningful updates after it, not only the last high-rate checkpoint or the first post-decay batch.
- **Rare-class learning:** inspect valid exposure and class errors. Thousands of majority-only updates are not thousands of opportunities for the missing class.
- **New head or opponent:** distinguish its own adaptation from movement of the shared representation. A fixed-feature opponent fit can inform signal quality but cannot forecast final target accuracy.
- **EMA teacher:** account for averaging lag. With a fixed smoothing factor alpha, past-state weight decays as alpha to the power k. The approximate e-folding horizon is -1/log(alpha) teacher updates, subject to the actual update schedule and implementation. This is a response timescale, not a required wait or proof that the teacher improves.
- **Small LR change:** inspect meaningful cumulative displacement and task response. A smaller rate can need more updates; it can also be so small that continued training is unproductive.
- **Stable harmful exposure:** repeated supported class damage can justify acting before a long observation window. Keep useful states recoverable rather than waiting for a cosmetic patience count.

Choose the next interval around an informative event or adequate exposure and name the expected observation. A continuation rationale should expire or be revised if that observation does not occur. Healthy progress does not require a separate Agent call per epoch.

## Schedules and horizons are coupled

For a horizon-normalized cosine or polynomial schedule, changing total steps changes rates before the old endpoint as well as adding steps. A run trained with a short schedule is not generally the prefix of the corresponding long-schedule run. Conversely, extending a finished short run at its final tiny LR may provide little new learning.

Distinguish three operations:

1. Continue a saved state under its already declared schedule and horizon.
2. Keep the current learned state but adopt an explicitly revised future schedule.
3. Restart a different trajectory from a common earlier state with a different schedule.

They answer different scientific questions and incur different costs. A retrospective longer-run advantage does not tell which operation was available at the original decision. A task cap, worker interruption and scientific convergence should have different labels.

Google's tuning playbook favors adequate training with retrospective checkpoint selection in the relevant supervised setting and warns about short-horizon transfer. Retaining earlier candidates means a longer run need not discard an earlier observed best. This depends on usable checkpoints, permitted selection measurements and an appropriate objective. It does not guarantee better true generalization, make repeated validation free of selection bias, or make unlabeled target quality observable.

## Recovery: reversing a value does not reverse training

|Operation|What it preserves|When it can be useful|What it cannot establish|
|Change a scalar back in the current state|All learned parameters, optimizer history and accumulated buffers remain.|The intervention was reversible enough, damage is not severe, and preserving progress is valuable.|That the original trajectory or representation has been restored.|
|Restore model weights only|The selected predictor, subject to which buffers/branches were saved.|Inference or a deliberately new optimization start.|Exact continuation with the old moments, schedule, RNG or data position.|
|Restore the declared full training state|The saved model/components, buffers, optimizers, schedule, scaler and recorded random/data state to the extent supported.|An adopted recovery branch or exact continuation.|Bitwise replay across unsupported loader/kernel/environment differences.|
|Pause with a complete resumable state|The option to continue later.|Account, resource or scientific allocation interruptions.|A scientific verdict that the model cannot improve.|

After recovery, compare observations from the restored/adopted trajectory. Do not concatenate metrics from discarded states into one continuous curve. Preserve intervention and artifact identities even in a compact history. If the same harmful mechanism recurs after a correctly restored state, that changes the evidence for the intervention; a failed restore should not be counted as a failed scientific treatment.

## Curve extrapolation: useful tools with assumptions

Domhan et al. combine saturating learning-curve models and posterior uncertainty to estimate whether a partial run can exceed an incumbent by a target horizon. This is richer than a flat patience rule. Its probability is conditional on the curve family, fitted observations and noise assumptions. It is not an empirically guaranteed probability that stopping is safe in a new task.

Freeze-thaw Bayesian optimization can pause and later resume runs rather than discard them permanently. Its temporal GP encodes decaying behavior toward an asymptote and allocates among multiple configurations. Those assumptions and the multi-run setting do not turn it into a requirement that a single-trajectory Agent open many trials.

Both ideas become less directly applicable across a newly changed objective, schedule transition, rare-class onset or adversarial regime change. Preserve phase information; do not extrapolate a pre-intervention segment as though the future update rule were unchanged. Use such a tool only if it is implemented, permitted and its uncertainty is credible for the present task. Do not fabricate posterior numbers from prose judgment.

## Worked decisions

These are mechanism illustrations, not reported experiment results.

**Flat before decay.** A model has stable but imperfect same-domain validation decisions; training loss still oscillates; its first planned rate reduction is imminent. A reasonable decision is to continue through enough post-decay updates to test refinement. If permitted decision errors improve, the old plateau was not final. If only confidence improves and the task metric does not, reconsider the value of further identical training. The mere presence of a future decay does not justify unlimited waiting.

**Flat after accidental schedule exhaustion.** A resumed run has a nominally larger epoch cap, but restored rates remain at a floor and update magnitudes are negligible. “Continue to the new cap” is poorly targeted. Establish the continuation contract and, if allowed, adopt a stated schedule intervention. Otherwise stop on the actual resource/contract basis without calling the result convergence.

**Target expansion under adaptation.** A rare-class map expands while source F1 falls. Negative transfer is plausible but the source decline and expansion do not identify its cause. Inspect actual errors, score changes and which auxiliary gradient path changed. A supported harmful path can justify weakening or suspending that path while retaining supervised learning. With only unlabeled target proxies, record the uncertainty and preserve a recovery state; do not claim target accuracy was measured.

## Source support

- [Google Deep Learning Tuning Playbook](https://github.com/google-research/tuning_playbook), sections on training steps, compute-bound workloads, short-run transfer, evaluation and checkpoint selection. Longer-run advice has the supervised retrospective-selection conditions explained above.
- [Speeding Up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves](https://www.ijcai.org/Proceedings/15/Papers/487.pdf), model combination, predictive termination and experiments.
- [Freeze-Thaw Bayesian Optimization](https://arxiv.org/abs/1406.3896), temporal model and pause/resume allocation.
- [Reviewed timm checkpoint saver](https://github.com/huggingface/pytorch-image-models/blob/c709c9a4a26af45d01221caadfdc6b00a9622630/timm/utils/checkpoint_saver.py), save/recovery mechanics; actual local workers may implement a different state contract.

The phase-specific decision table, EMA timescale calculation and recovery comparisons are conditional synthesis and direct mechanism reasoning. They are not universal early-stopping rules or measured quality gains.
