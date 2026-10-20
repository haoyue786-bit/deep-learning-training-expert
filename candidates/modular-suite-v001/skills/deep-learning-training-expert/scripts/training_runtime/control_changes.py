"""Bind actual control deltas to supplied evidence, without claiming scientific proof."""
from __future__ import annotations

import math
import copy


def resolve_control_changes(*, before, change_basis, evidence_catalog):
    """For a declared delta-only response schema, construct the complete vector.

    Do not use this to discard contradictory controls from an already rejected
    full-vector answer. Schema validation must reject such extra output first.
    """
    if not isinstance(before, dict) or not isinstance(change_basis, list):
        raise ValueError("Host baseline and explicit change list required")
    after = copy.deepcopy(before)
    for row in change_basis:
        if not isinstance(row, dict) or row.get("control") not in before or "after" not in row:
            raise ValueError("Unknown or incomplete proposed control delta")
        after[row["control"]] = row["after"]
    validate_control_changes(before=before, after=after, change_basis=change_basis,
                             evidence_catalog=evidence_catalog)
    return after


def _scalar(value):
    if type(value) not in (bool, int, float, str) and value is not None:
        raise ValueError("Control maps must contain flat JSON scalars")
    if type(value) is float and not math.isfinite(value):
        raise ValueError("Nonfinite control")
    return type(value).__name__, value


def validate_control_changes(*, before, after, change_basis, evidence_catalog):
    """Check a host-normalized complete control vector and per-change explanation.

    Evidence IDs are issued by the host for this state/packet, never by the model.
    The host still checks bounds, artifact identities, scientific applicability,
    native-action semantics and authorization. Nonempty prose cannot prove causality.
    """
    if not isinstance(before, dict) or not isinstance(after, dict) or set(before) != set(after):
        raise ValueError("Complete before/after control vectors with identical keys required")
    changed = {k for k in before if _scalar(before[k]) != _scalar(after[k])}
    if not isinstance(change_basis, list) or not isinstance(evidence_catalog, dict):
        raise ValueError("Explicit change_basis list and host evidence catalog required")
    addressed = set()
    for row in change_basis:
        if not isinstance(row, dict):
            raise ValueError("Each control change needs its own evidence-bound explanation")
        control = row.get("control")
        if not isinstance(control, str) or control not in changed or control in addressed:
            raise ValueError("Explanation must cover each actual changed control exactly once")
        addressed.add(control)
        if "before" not in row or "after" not in row or _scalar(row["before"]) != _scalar(before[control]) or _scalar(row["after"]) != _scalar(after[control]):
            raise ValueError("Explained values differ from the actual proposed control delta")
        if row.get("support_status") != "supported":
            raise ValueError("Unresolved support is a hypothesis, not an executable control change")
        for field in ("mechanism", "why_now", "why_this_magnitude", "expected_response", "alternative", "reversal_condition"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError("Missing control-change explanation: " + field)
        refs = row.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(not isinstance(r, str) or r not in evidence_catalog for r in refs):
            raise ValueError("Changed control must cite existing host-issued evidence IDs")
        if len(set(refs)) != len(refs):
            raise ValueError("Duplicate evidence IDs do not add support")
        if not any(evidence_catalog[r].get("kind") == "observation" for r in refs):
            raise ValueError("A generic recipe alone does not bind a change to the current task")
    if addressed != changed:
        raise ValueError("Actual changed controls lack an evidence-bound explanation: " + ", ".join(sorted(changed - addressed)))
    return dict(changed_controls=sorted(changed), status="structural_evidence_binding_passed",
                scientific_validity_verified=False)
