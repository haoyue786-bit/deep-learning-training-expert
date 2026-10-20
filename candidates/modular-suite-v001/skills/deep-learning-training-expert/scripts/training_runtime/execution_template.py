"""Small executable response plus a separate, freely written training record.

The host provides facts and allowed operations. Scientific prose is stored, not
parsed into actions. No model-supplied metric, hash or temporal claim is needed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
from .response_contract import validate_schema


def object_schema(properties):
    return dict(type="object", properties=properties, required=list(properties),
                additionalProperties=False)


def response_schema(parameters, actions, candidates):
    execution = object_schema(dict(
        action=dict(type="string", enum=list(actions)),
        changes=dict(type="array", items=object_schema(dict(
            parameter=dict(type="string", enum=list(parameters)),
            value=dict(type="number")))),
        selected_epoch=dict(type="integer", enum=sorted(set([0] + list(candidates))))))
    return object_schema(dict(
        execution=execution,
        training_record=dict(type="string")))


def instructions(schema):
    return (
        "Return the following fixed envelope. Only execution is interpreted by "
        "the worker. Write your scientific explanation as free text in "
        "training_record: current evidence, continuation versus selection, "
        "reasons for changes, expected response, uncertainty and useful next "
        "observations. Do not create an additional schema for this prose. "
        "Questions or desired future observations in the record do NOT request "
        "a pause. Use action HOLD only if missing evidence makes the current "
        "decision impossible. A supported KEEP needs no perfect information. "
        "Do not repeat facts or checkpoint hashes in execution; the host binds "
        "them. changes lists only parameters to change; [] retains all current "
        "values. selected_epoch is 0 except for STOP. STOP chooses an actual "
        "eligible checkpoint and cannot change parameters. The input brief is "
        "not an output template. No markdown fences around the envelope.\n"
        'Example shape: {"execution":{"action":"KEEP","changes":[],'
        '"selected_epoch":0},"training_record":"Your freely written analysis"}\n'
        "Use only actions and values allowed by this exact schema:\n"
        + json.dumps(schema, ensure_ascii=False, separators=(",", ":"))
    )


def validate_response(response, *, before, bounds, actions, candidates,
                      integer_parameters=()):
    """Validate executable intent, never parse scientific prose as controls."""
    schema = response_schema(before, actions, candidates)
    validate_schema(response, schema)
    execution = copy.deepcopy(response["execution"])
    action = execution["action"]
    changes = execution["changes"]
    if action in ("KEEP", "STOP", "HOLD") and changes:
        raise ValueError("%s cannot change parameters" % action)
    if action == "ADJUST" and not changes:
        raise ValueError("ADJUST requires an actual change; use KEEP otherwise")
    if action != "STOP" and execution["selected_epoch"] != 0:
        raise ValueError("Only STOP may select a checkpoint")
    if action == "STOP" and execution["selected_epoch"] not in candidates:
        raise ValueError("STOP must select an existing eligible checkpoint")
    controls = copy.deepcopy(before)
    seen = set()
    for delta in changes:
        key, value = delta["parameter"], delta["value"]
        if key in seen:
            raise ValueError("Duplicate parameter: " + key)
        seen.add(key)
        if key in integer_parameters:
            if int(value) != value:
                raise ValueError("Integral value required for " + key)
            value = int(value)
            delta["value"] = value
        lower, upper = bounds[key]
        if not lower <= value <= upper:
            raise ValueError("Out of bounds: " + key)
        if value == before[key]:
            raise ValueError("Unchanged parameter in changes: " + key)
        controls[key] = value
    return dict(execution=execution, resolved_controls=controls,
                training_record=response["training_record"],
                record_present=bool(response["training_record"].strip()))


def save_response(directory, validated, host_facts, provenance):
    """Store explanation separately from executable intent and measured facts."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    values = {
        "execution.json": validated["execution"],
        "host_facts.json": host_facts,
        "response_provenance.json": dict(
            provenance, record_present=validated["record_present"],
            record_controls_execution=False,
            scientific_reasoning_automatically_verified=False),
    }
    for name, value in values.items():
        path = directory / name
        tmp = path.with_name(name + ".tmp." + str(os.getpid()))
        tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                  allow_nan=False) + "\n")
        tmp.replace(path)
    path = directory / "training_record.md"
    tmp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    tmp.write_text(validated["training_record"] + "\n")
    tmp.replace(path)
    return hashlib.sha256((directory / "execution.json").read_bytes()).hexdigest()
