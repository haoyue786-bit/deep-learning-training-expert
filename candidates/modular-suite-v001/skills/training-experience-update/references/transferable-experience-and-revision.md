# Transferable experience, retrieval and revision

Read when using a past training case to choose a new action, extracting a lesson from feedback, or revising a rule after a contradiction. The aim is a useful conditional relationship between the current training state and an action, not a catalogue of winning numbers.

## Separate three things that can be learned

|Knowledge|Example|What it supports|
|Mechanism|AdamW decay depends on effective rates and update count; domain marginals mix class proportions and conditional features.|A reason to inspect or act under matching assumptions; it does not supply a universal best amplitude.|
|Observed case|At a named saved state, a completed contrast changed a stated metric under a particular population and horizon.|An association or local intervention result with that scope.|
|Conditional lesson|When useful inherited features are damaged before a new head learns, differentiated readiness is a candidate explanation and intervention.|A starting hypothesis/action for a sufficiently similar mechanism, revised by current evidence.|

Store the case facts separately from its interpretation. Updating an explanation should not rewrite the original trajectory. Keep the policy/state that produced an observation distinct from the policy selected afterward. If a label correction changes the meaning of an old score, mark the affected result and lesson accordingly; preserve the original artifact for traceability.

## What the research contributes

**TNP: learn response behavior, not raw dataset resemblance alone.** The paper *Meta-learning Hyperparameter Performance Prediction with Neural Processes* transfers historical configuration/outcome observations, surrogate initialization and initial configurations, with dataset-aware attention adapted to current observations. Its similarity is learned in relation to hyperparameter-performance behavior. This supports treating history as a revisable prior about a response surface. It does not establish that an AEF distance, geography or source-feature statistic directly determines the best LR. Implementing TNP would require compatible trial data and a real surrogate; prose inspired by it is not an implementation.

**Reflexion: feedback can guide another attempt, but the evaluator matters.** Its actor, evaluator and reflection memory help revise actions without updating the language model's weights. Much of the evidence concerns repeated attempts on the same task. Incorrect self-generated tests can give false success and prematurely stop useful revision; the paper reports adverse as well as positive task outcomes. A fluent reflection is an interpretation of feedback, not an independently verified cause.

**ExpeL: retain cases and extract cross-task lessons.** It contrasts success/failure trajectories and summarizes common patterns from successes, then combines insights with retrieved demonstrations for unseen tasks. Its editable/voted insights illustrate revision rather than immutable rules. Importance votes are not calibrated confidence or independent replication counts. Its paper also reports that adding reflection text can hurt and that combined memory is not best in every task subtype. In the reviewed implementation, rule extraction and the retrieved case pool have different filtering paths; do not assume one split automatically isolates every information channel.

**A-Mem: organize relationships while preserving content identity.** Original content, generated context, tags and links can be represented separately; later observations can change a memory's interpretation and neighbors. A semantic link is a retrieval aid, not proof of a causal relationship. Its conversation-memory evaluations do not establish training-control expertise or that an updated note became more scientifically accurate.

The practical synthesis is modest and useful: retrieve actual relevant cases; use original evidence to extract conditional claims; keep interpretation editable; retain contradictions; and check current response. No vector database, embedding model or extra reflection call is required for this Skill.

## Match a case by the mechanism that matters

Before reusing a historical action, compare the aspects that can change its effect:

- **Objective and observation:** source/target/deployment population, legal labels, class support, primary metric and decoder.
- **Starting components:** inherited/new head and backbone, optimizer/mode/buffer state, teacher and opponent readiness.
- **Update regime:** optimizer, group membership, active paths, rate/schedule phase, batch/exposure and recent changes.
- **Error mechanism:** the actual confusion, loss/gradient support, sampling opportunity and type of shift.
- **Treatment and horizon:** what really changed, whether weights or only scalars were adopted, how much exposure occurred and when results were measured.

Country, year, tile embedding or a shared low F1 can help locate candidates. They do not replace this comparison. Two tiles can be near in input-feature space but differ in class mixtures or initialized errors; the same tile across years can change labels or land cover. Conversely different countries can share a useful component-readiness problem.

Do not require a unique value for each tile to demonstrate personalization. Identical supported conditions can justify the same starting setting. A different state may instead call for a different component, objective path, sampling change or observation, rather than a cosmetic LR change. The relationship to learn can be “rare class lacks valid crop exposure → fix exposure before increasing weight,” not a fixed function from feature distance to one scalar.

## Extract a usable lesson from an executed case

Use the existing caller's record format. A compact lesson should preserve enough to answer:

- What was known at the decision time, and what mechanism did the action target?
- Which complete state and actual treatment produced the response?
- What changed mechanically and what changed in the permitted task metric?
- What important alternative remains, including data or schedule changes?
- In which new conditions would the lesson be relevant or misleading?
- Which original artifacts support it, and what did acquiring it cost?

The lesson's strength should follow evidence, not a vote count. A completed local contrast supports that observation, not a reusable empirical response. Require comparable repeated actions across meaningful task units before promoting a relationship. For spatial projects, count distinct tiles and collapse correlated years/seeds/branches within a tile. No count alone proves generality; geography and shared source models can create dependence. Report the compatible set, agreements, opposite responses and small effects. Missing or failed arms leave the intended contrast unknown.

Keep two comparisons separate: matching all recorded background controls and data procedures, and matching a procedure whose tile-specific sample pools intentionally differ. Preserve those pool differences as conditions. Do not merge different starting values, active/paused objectives or response horizons into one directional rule. Searching many strata and selecting an all-positive one is exploratory; compare its neighboring strata and retain the mixed result. A uniform mechanical response can be reusable even when a task-quality prescription remains unsupported.

## Use positive, null and adverse cases differently

**Positive response.** If a group-rate change improves supported task errors while preserving a useful component, retain the state and conditions that made it work. Do not automatically copy its final rate into every fresh start or credit the whole improvement to that scalar if the treatment also changed the schedule.

**Mechanical success, semantic null.** If reducing a loss weight reduces the weighted gradient but does not improve the relevant errors, retain the control-path knowledge. Withdraw the claim that the gradient ratio was a reliable quality target. The next case should examine a consequential alternative, not repeat the same easy proxy adjustment indefinitely.

**Adverse response.** If the intended treatment executes and worsens a supported error, record which error and exposure. It may contradict the proposed mechanism, indicate an unsuitable amplitude, or expose an interaction. State which interpretation the evidence supports. Do not use “more tuning needed” to protect the original rule from every contradiction.

**Null with weak exposure.** A confidence cap never reached, an empty negative pool, or a tiny decay horizon does not test the intended mechanism. Preserve the inactive/null execution fact; do not conclude the mechanism never matters.

**Confounded or corrected case.** If a job path now points to a different supplementary list, or updated labels invalidate a comparison, restrict the claim. A different label version cannot silently become evidence for the old intervention. This can invalidate a quality ranking while leaving a mathematical update identity intact.

These are lesson forms, not claims of improvement on a particular project. Actual cases and original evidence belong in the explicitly authorized corresponding project store; the installed Skill holds general rules only. See [project ownership and validation-use history](project-store.md).

## Retrieve enough to decide

Prefer a case that exposes the decisive interaction or a strong counterexample over many superficially similar winners. Include the relevant negative/null case when its conditions match. A short case can point to the original record and detailed mechanism reference; do not repeatedly paste the full archive.

When a historical lesson and current evidence disagree, current directly relevant evidence should change its applicability. Distinguish an inapplicable case from a refuted mechanism: an LR recipe failing after a different schedule does not refute gradient descent, but does restrict that recipe's scope. Keep the strongest live alternative rather than allowing the latest retrieved narrative to overwrite it.

In a sealed comparison, knowledge extraction and retrieval both follow the declared training/evaluation boundary. A history pool can leak future outcomes or held-out cases even if the prompt never says “test.” A later oracle winner may audit the old decision; it cannot become knowledge that was supposedly available at that time.

## Revise without accumulating rules

Correct the smallest unsupported claim: narrow a condition, revise the proposed mechanism, split two different outcomes, or withdraw a prescription. Merge genuinely duplicate lessons while retaining their sources and contrary evidence. Do not turn each one-off failure into another global prohibition.

Separate stable professional knowledge from volatile run state. A new case can update a local experience record when authorized; it does not automatically modify a frozen published Skill or production caller. Important changes to the public knowledge should explain what evidence changed, which old claim is affected and the remaining scope. A successful release or a valid JSON record is not scientific confirmation.

## Source support

- [Meta-learning Hyperparameter Performance Prediction with Neural Processes](https://proceedings.mlr.press/v139/wei21c.html), configuration/outcome representation, dataset-aware attention and transfer; local source S010.
- [Reflexion](https://arxiv.org/abs/2303.11366), actor/evaluator/reflection mechanisms, programming feedback analysis and limitations; local S066 paper and S067 reviewed code.
- [ExpeL](https://arxiv.org/abs/2308.10144), Sections 4–5, insight operations, retrieval and ablations; [reviewed implementation](https://github.com/LeapLabTHU/ExpeL/tree/e41ec9a24823e7b560c561ab191441b56d9bcefc), especially extraction, evaluation and case-pool paths; local S068.
- [A-Mem](https://arxiv.org/abs/2502.12110), Sections 3–4 and appendix, original content versus generated context, links, evolution and retrieval; local S069.

The conditional-case procedure is a synthesis for training work, not a claim that any of these papers validated Cashew-specific parameter rules or this suite's effectiveness.
