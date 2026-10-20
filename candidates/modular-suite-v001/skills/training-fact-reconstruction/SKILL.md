---
name: training-fact-reconstruction
description: Recover training facts, metric semantics and observation/action timing from real logs, code and checkpoints. Use for unfamiliar or conflicting records or fact-only requests; do not automatically make training decisions.
---

# Training Fact Reconstruction

For authorized point feedback, reconstruct actual prediction/label pairs, confusion, Precision/Recall/F1, support and both confusing-class error directions under the declared decoder. Preserve aggregate label semantics and distinguish development exposure from independent evaluation. Report whether threshold selection actually occurred; prior Agent feedback alone is not a threshold study. Read/reconcile the corresponding project's evidence; save corrections only through its explicit authorized [project store](../training-experience-update/references/project-store.md), never an assumed default project.

Recover the facts that can change the present question. Read [factual reconstruction](references/factual-reconstruction.md). Honor the caller's output format and permitted evidence. A healthy run does not need a complete historical audit at every review.

For unfamiliar measurements use [metrics, populations and error slices](references/metrics-populations-and-error-slices.md). For a start, resume, adopted branch or potentially mismatched checkpoint use [training state and trajectory identity](references/training-state-and-trajectory-identity.md). These references supply the concrete definitions and implementation distinctions; choose according to the unresolved fact rather than reading the full archive.

Identify task, data and label versions, model state and actual executed exposure. Bind each used metric to its population, class identity, statistic, units, evaluation mode and observation time. Separate what an original record reports, what supplied code implies and what is verified about the historical execution. Missing measurements are unknown; disabled loss values are not measured zero gradients.

Track observation, proposed change, actual execution, adoption and response separately. An epoch number is not a JSON array index. A selected epoch does not identify the last executed epoch. Current code cannot silently redefine an old metric. Resolve disagreements from original records where possible; constrain only the unresolved claim rather than declaring everything unknowable.

Use the supplied registry or exact source pointers when available. The bundled claim helpers can bind literal values and identities; they cannot establish a causal interpretation. Preserve each requested metric's actual identity rather than choosing the nearest familiar field. A schema-valid statement can still misunderstand the metric.

Return the needed evidence-backed facts, conflicts, timeline and unknowns in the requested contract. For a fact-only request, stop there. For a scientific decision, pass a compact fact basis into reasoning without inventing a new mandatory protocol. Completion means the relevant facts are recoverable and uncertainty explicit, not that the resulting model is accurate.

## Interpret the measurement, not just its field name

Read [metrics, populations and error slices](references/metrics-populations-and-error-slices.md) for confusion support, undefined classes, aggregation, loss versus accuracy, ranking versus score scale, decoding rules and spatial statistics. Use its concrete distinctions to restore the meaning required by the current question.
