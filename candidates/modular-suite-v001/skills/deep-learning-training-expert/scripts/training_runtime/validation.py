"""Bounded development feedback from real predictions and versioned point labels.

This module computes fixed-decoder metrics. It never fits thresholds, trains on
validation labels or quietly treats development feedback as an independent test.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Iterable


def _ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_points(rows: Iterable[dict], *, class_names: list[str], dataset: dict,
                    checkpoint: dict, focus_class: int, confusing_class: int,
                    maximum_error_examples: int = 12) -> dict:
    if dataset.get("role") not in ("development", "independent_evaluation"):
        raise PermissionError("Declare the actual evaluation role")
    if dataset.get("labels_for_gradient", False):
        raise PermissionError("Validation feedback is not an auxiliary training-label route")
    for key in ("dataset_id", "label_version", "sha256"):
        if not dataset.get(key): raise ValueError(f"Missing dataset identity: {key}")
    if not checkpoint.get("identity") or not checkpoint.get("sha256"):
        raise ValueError("Bind metrics to a concrete checkpoint")
    classes = len(class_names)
    if classes < 2 or focus_class == confusing_class or not all(0 <= c < classes for c in (focus_class, confusing_class)):
        raise ValueError("Distinct supported class IDs required")
    if not 0 <= maximum_error_examples <= 100:
        raise ValueError("Keep feedback examples bounded")
    matrix = [[0]*classes for _ in range(classes)]
    examples = []; seen = set(); identities = []
    pair_errors = {"confusing_as_focus": [], "focus_as_confusing": []}
    for row in rows:
        point_id = str(row["point_id"])
        if point_id in seen: raise ValueError("Duplicate point identity; resolve ownership before evaluation")
        seen.add(point_id)
        actual = int(row["class_id"])
        if not 0 <= actual < classes: raise ValueError("Evaluation class outside the declared class order")
        if "probabilities" in row:
            values = row["probabilities"]
            if len(values) != classes or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError("Finite per-class probabilities required")
            if abs(sum(values)-1) > 1e-4: raise ValueError("Expected normalized class probabilities")
            predicted = max(range(classes), key=lambda c: values[c])
            if "predicted_class" in row and int(row["predicted_class"]) != predicted:
                raise ValueError("Recorded decoder disagrees with four-class argmax")
        else:
            predicted = int(row["predicted_class"])
        if not 0 <= predicted < classes: raise ValueError("Prediction class outside declared class order")
        matrix[actual][predicted] += 1
        identities.append((point_id, actual))
        if actual != predicted:
            example = {"point_id": point_id, "actual": actual, "predicted": predicted}
            for key in ("tile_id", "group", "year", "label_name"):
                if key in row: example[key] = row[key]
            if "probabilities" in row:
                example.update(focus_probability=row["probabilities"][focus_class],
                               confusing_probability=row["probabilities"][confusing_class])
            if len(examples) < maximum_error_examples: examples.append(example)
            key = ("confusing_as_focus" if (actual, predicted) == (confusing_class, focus_class)
                   else "focus_as_confusing" if (actual, predicted) == (focus_class, confusing_class) else None)
            if key and len(pair_errors[key]) < maximum_error_examples: pair_errors[key].append(example)
    if not seen: raise ValueError("Empty validation support is unknown, not zero accuracy")
    total = len(seen); per_class = []
    for c, name in enumerate(class_names):
        tp = matrix[c][c]; fp = sum(matrix[k][c] for k in range(classes))-tp
        fn = sum(matrix[c])-tp; tn = total-tp-fp-fn
        per_class.append(dict(class_id=c, name=name, support=tp+fn, tp=tp, fp=fp, fn=fn, tn=tn,
                              precision=_ratio(tp, tp+fp), recall=_ratio(tp, tp+fn), f1=_ratio(2*tp, 2*tp+fp+fn)))
    pair = dict(focus_class=focus_class, confusing_class=confusing_class,
                confusing_as_focus=matrix[confusing_class][focus_class],
                focus_as_confusing=matrix[focus_class][confusing_class], examples=pair_errors)
    return dict(version="point_validation_feedback_v1", dataset=dataset, checkpoint=checkpoint,
                evaluation_role=dataset["role"], decoder="multiclass_argmax", samples=total,
                class_order=class_names, confusion_matrix=matrix,
                confusion_axes="rows=true class, columns=predicted class", per_class=per_class,
                focus_metrics=per_class[focus_class], confusable_pair=pair, error_examples=examples,
                point_identity_sha256=hashlib.sha256(json.dumps(sorted(identities)).encode()).hexdigest(),
                threshold_fitted=False, labels_enter_backpropagation=False,
                independent_evaluation=(dataset["role"] == "independent_evaluation"))


def screen_candidates(feedback: dict, *, maximum_candidates: int = 2) -> dict:
    """Offer optional error-based leads, never a causal diagnosis or action gate.

The owning expert must assess the hypotheses and actual module availability.
This is an evidence router, not an automatic claim of algorithm superiority.
"""
    if feedback["evaluation_role"] != "development":
        raise PermissionError("Final/independent evaluation cannot select a new treatment")
    if not 0 <= maximum_candidates <= 2:
        raise ValueError("Initial bench screens at most two treatments; no unbounded search")
    pair = feedback["confusable_pair"]; fp = pair["confusing_as_focus"]; fn = pair["focus_as_confusing"]
    support = feedback["focus_metrics"]["support"]
    if not fp and not fn:
        candidates = []
        diagnosis = "This panel supplies no measured focus/confusing-class error to justify a pair-specific treatment."
    elif fp >= fn:
        diagnosis = (f"Observed {fp} confusing-class points predicted as focus and {fn} reverse errors. "
                     "Unreliable target conditioning and insufficient source class separation are competing hypotheses.")
        candidates = [dict(module_id="M03", variant="pair_ambiguity_reliability",
                           hypothesis="Attenuating ambiguous conditional alignment may reduce confusing-class false positives.",
                           check="Compare both error directions, focus recall and source safety; confidence/prior is not a label."),
                      dict(module_id="M04", variant="source_pair_margin",
                           hypothesis="A separation constraint on trainable source-class features may improve the transferable boundary.",
                           check="Require source support for both classes and real nonzero gradients; do not train on these validation labels.")]
    else:
        diagnosis = (f"Observed {fn} focus points predicted as confusing class and {fp} reverse errors. "
                     "Over-suppression and insufficient feature separation are competing hypotheses.")
        candidates = [dict(module_id="M04", variant="source_pair_margin",
                           hypothesis="Source-supervised class separation may recover missed focus examples without new target labels.",
                           check="Compare recall, false positives, pair errors and source safety at the declared endpoint."),
                      dict(module_id="M02", variant="class_pixel_priority",
                           hypothesis="A bounded focus-class alignment priority may change unequal class contribution.",
                           check="Record pseudo-class support and wrong-class amplification; do not infer target class proportions.")]
    return dict(version="validation_grounded_candidate_screen_v2", diagnosis=diagnosis,
                evidence_checkpoint=feedback["checkpoint"], evidence_dataset=feedback["dataset"],
                measured_pair_errors={"confusing_as_focus": fp, "focus_as_confusing": fn},
                focus_support=support, candidates=candidates[:maximum_candidates],
                advisory_only=True,
                alternatives=["KEEP with an informative observation horizon", "check labels/data/implementation",
                              "adjust an existing supported control", "inspect schedule/source retention/calibration",
                              "one mechanism-directed trial if it can change the decision"],
                owner="same scientific expert; worker checks registry, permissions and budget",
                quality_claim="Development hypotheses only; no causal attribution or efficacy inferred from routing.",
                threshold_search_authorized=False)


def compare_feedback(before: dict, after: dict) -> dict:
    for key in ("point_identity_sha256", "class_order", "decoder", "evaluation_role"):
        if before[key] != after[key]: raise ValueError(f"Incomparable feedback: {key}")
    if before["dataset"] != after["dataset"]:
        raise ValueError("Labels/population changed between endpoints")
    def difference(a, b): return None if a is None or b is None else b-a
    return dict(focus_f1_delta=difference(before["focus_metrics"]["f1"], after["focus_metrics"]["f1"]),
                focus_precision_delta=difference(before["focus_metrics"]["precision"], after["focus_metrics"]["precision"]),
                focus_recall_delta=difference(before["focus_metrics"]["recall"], after["focus_metrics"]["recall"]),
                confusing_as_focus_delta=after["confusable_pair"]["confusing_as_focus"]-before["confusable_pair"]["confusing_as_focus"],
                focus_as_confusing_delta=after["confusable_pair"]["focus_as_confusing"]-before["confusable_pair"]["focus_as_confusing"],
                evaluation_role=before["evaluation_role"], threshold_fitted=False)
