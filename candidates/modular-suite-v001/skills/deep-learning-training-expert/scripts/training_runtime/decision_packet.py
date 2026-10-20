"""Compact validation-grounded decision input; no model calls or token estimates."""
from __future__ import annotations

import json
from pathlib import Path

from .registry import cards
from .validation import screen_candidates


def defaults() -> dict:
    return json.loads((Path(__file__).resolve().parents[2] / "scientific_defaults.json").read_text())


def build(feedback: dict, *, available_variants: list[str], previous_response=None,
          experience_summaries=(), maximum_characters=16000, selected_variants=()) -> dict:
    screen = screen_candidates(feedback)
    if len(selected_variants)>2: raise ValueError('Select at most two relevant cards, not the whole catalog')
    if any(v not in available_variants for v in selected_variants):raise ValueError('Selected variant unavailable')
    selected = set(selected_variants)
    relevant_cards = [c for c in cards(available_variants=available_variants)
                      if c['callable'] and f"{c['module_id']}:{c['variant']}" in selected]
    screen={k:v for k,v in screen.items() if k not in ('evidence_checkpoint','evidence_dataset')}
    screen['evidence_pointer']='feedback.dataset and feedback.checkpoint in this same packet'
    compact_feedback = {k: feedback[k] for k in ("dataset", "checkpoint", "evaluation_role", "decoder", "samples",
                         "class_order", "confusion_matrix", "per_class", "focus_metrics", "threshold_fitted")}
    pair = feedback["confusable_pair"]
    compact_feedback["confusable_pair"] = {
        k: pair[k] for k in ("focus_class", "confusing_class", "confusing_as_focus", "focus_as_confusing")}
    compact_feedback["error_examples"] = feedback["error_examples"][:6]
    settings = defaults()
    result = dict(version="compact_scientific_decision_v1", model=settings["model"],
                  reasoning_effort=settings["reasoning_effort"], feedback=compact_feedback,
                  candidate_hypotheses=screen, available_relevant_modules=relevant_cards,
                  available_trial_variants=list(available_variants),
                  previous_response=previous_response, project_experience=list(experience_summaries)[:3],
                  decision_contract={
                      "owner": "one continuing scientific expert",
                      "required_reasoning": "Use measured confusion, Precision/Recall/F1 and both pair-error directions to revise the diagnosis and justify KEEP or one bounded trial.",
                      "control_changes": "Audit actual old/new values. Each change needs current evidence, an applicable mechanism, magnitude rationale, expected response, alternative and reversal condition. Unsupported hypotheses remain separate proposals; a caveat cannot authorize a contradictory numeric change.",
                      "prohibited": ["validation-label backpropagation", "threshold search", "unlimited candidate or parameter search", "production adoption without host authorization"],
                      "next_action": "Choose KEEP, inspect data/implementation, adjust an existing host control, or propose one useful trial. Error-count leads do not diagnose the cause or restrict the choice."
                  }, extra_model_calls=0)
    # Refuse oversized facts rather than silently truncating consequential evidence.
    data = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(data) > maximum_characters:
        raise ValueError("Decision packet exceeds the declared size budget; supply a smaller explicit history summary")
    result["size"] = dict(characters=len(data), actual_tokens=None,
                          interpretation="Measured serialization size, not a token count or savings claim")
    return result
