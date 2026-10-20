"""Transport and validation helpers for a host-declared scientific response.

These helpers never infer controls, fix scientific facts, or turn HOLD into KEEP.
"""
import json
import math


def validate_schema(value, schema, path="$"):
    kind = schema["type"]
    if kind == "object":
        if type(value) is not dict:
            raise ValueError("%s must be an object" % path)
        required = set(schema.get("required", []))
        missing = required - set(value)
        extra = set(value) - set(schema["properties"])
        if missing or (extra and schema.get("additionalProperties") is False):
            raise ValueError("%s missing=%s unexpected=%s" %
                             (path, sorted(missing), sorted(extra)))
        for key, item in value.items():
            if key in schema["properties"]:
                validate_schema(item, schema["properties"][key], path + "." + key)
    elif kind == "array":
        if type(value) is not list:
            raise ValueError("%s must be an array" % path)
        for index, item in enumerate(value):
            validate_schema(item, schema["items"], "%s[%d]" % (path, index))
    elif kind == "number":
        if type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError("%s must be a finite number" % path)
    elif kind == "null":
        if value is not None:
            raise ValueError("%s must be null" % path)
    elif type(value) is not {"string": str, "integer": int, "boolean": bool}[kind]:
        raise ValueError("%s must be %s" % (path, kind))
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError("%s is outside the declared enum" % path)


def schema_instruction(schema):
    return (
        "\nAUTHORITATIVE RESPONSE CONTRACT: Return exactly one JSON object "
        "matching this schema, without markdown fences, introductions, or extra "
        "keys. The input brief's format is not an output template. Do not invent "
        "a generic expert_review output. Retain null for unmeasured values. "
        "The transport outputSchema and this visible schema are identical:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def information_state(answer):
    """Accept harmless empty follow-up metadata without rewriting the answer."""
    action = answer["action"]
    dependency = answer["information_dependency"]
    requests = answer["knowledge_requests"]
    if action == "HOLD" or dependency == "required_before_action":
        return "blocked"
    if dependency == "none" and requests:
        raise ValueError("Nonempty requests must explicitly be follow_up_only "
                         "or required_before_action")
    if dependency not in ("none", "follow_up_only"):
        raise ValueError("Unknown information_dependency")
    return "follow_up" if requests else "ready"


def measured_temporal_table(observations, keys):
    """Supply exact measured ranges; model claims remain independently checked."""
    table = []
    for key in keys:
        rows = sorted((r["epoch"], r["metrics"][key]) for r in observations
                      if r["metrics"].get(key) is not None)
        if len(rows) < 2:
            continue
        values = [v for _, v in rows]
        table.append(dict(
            metric=key, measured_epochs=[e for e, _ in rows], values=values,
            first=values[0], last=values[-1],
            all_equal=all(v == values[0] for v in values),
            endpoints_relation=("increase_endpoints" if values[-1] > values[0]
                                else "decrease_endpoints" if values[-1] < values[0]
                                else "unchanged_endpoints"),
            scope="Listed measured epochs only; no causal inference"))
    return table
