---
name: training-stop
description: Judge whether another interval of an existing deep-learning run is worthwhile using its actual learning response, schedule and objective. Use for continue, pause or stop decisions; checkpoint selection is a separate question.
---

# Training Continuation and Stopping

Judge the value of another training interval against the actual task objective. Read current observations, previous interventions, completed optimizer exposure, effective schedule and remaining budget. Use the caller's allowed actions and response schema.

For an actual plateau, delayed response, exhausted schedule or damaged state, read [training sufficiency and recovery](references/training-sufficiency-and-recovery.md). Use its phase/exposure distinctions, concrete continuation opportunities, recovery choices and curve-extrapolation assumptions to judge this run. An epoch count alone is not that judgment. The shorter [continuation/selection principles](references/stopping-and-selection.md) and [optimization context](references/optimization-practice.md) are available for cross-cutting interpretation; do not load all three automatically.

## Determine what the apparent plateau means

First compare the same metric and population over time. A rising validation cross-entropy with improving accuracy may concern confidence rather than fewer correct decisions. Conversely a better aggregate can hide an important class deterioration. Use the stated primary objective and relevant error slices; do not silently replace it with a smoother proxy.

Interpret epochs through actual updates and schedule phases. Warmup, a recent LR reduction, new-component adaptation or an objective transition can make a short flat interval inconclusive. Reducing update size also changes the exposure needed to observe a response. Explain a plausible benefit and an informative observation horizon when choosing to continue; merely naming delayed improvement does not justify indefinite training.

When evidence indicates overfitting or semantic damage, identify the measurement and population supporting that conclusion. Data or evaluation defects require their own diagnosis; more epochs cannot repair a wrong label mapping. Unlabeled entropy, map density or domain confusion alone cannot establish target accuracy harm or recovery impossibility.

## Compare the next real opportunities

Ask whether another legal interval, an already justified adjustment or a cheap discriminating observation can change the useful result. Continuing need not require an extra branch. Stopping need not require every ideal measurement. Respect the task's hard limits without treating budget exhaustion as convergence.

Do not impose a universal patience, minimum epoch or run-to-cap rule. Karpathy's recipe includes early stopping as a regularization option; FSDL teaching material gives it lower priority. These contextual recommendations do not settle an unfamiliar run. Compare the present learning evidence, schedule and task value.

## State the decision and its limits

Return CONTINUE, STOP or another advertised action with the concrete evidence, strongest alternative and useful response horizon. If continuing, say what the next interval could improve or distinguish. If stopping, state whether the basis is scientific progress, measured harm, safety or resources. Do not choose a deployment checkpoint implicitly: preserve a recoverable state and leave artifact selection as a distinct decision.

Retrospective audit may inspect later fixed-baseline improvement after decisions freeze. Such future evidence can reveal an early-stop mistake; it cannot be injected into the original stopping rationale.

## Source interpretation

The accompanying references preserve conditional guidance. Additional contextual sources: Karpathy, A Recipe for Training Neural Networks (2019), https://karpathy.github.io/2019/04/25/recipe/ ; FSDL training/debugging material, https://fullstackdeeplearning.com/ . Their disagreement about prioritizing early stopping is not resolved by majority vote. No fixed patience or guaranteed optimum is derived here.
