"""Read-only contract/arithmetic projection; never select training controls."""
import argparse
import copy
import json
import math
from pathlib import Path


def specialize_output_schema(request, schema):
    """Match a selection-only transport to the executor's existing semantics.

    A shared multi-action schema permits null epochs for KEEP/STOP. Once SELECT
    is the sole action, that alternative is not executable. This does not rank
    checkpoints or change eligibility.
    """
    result = copy.deepcopy(schema)
    if request.get('allowed_actions') != ['SELECT_CHECKPOINT']:
        return result
    eligible = request.get('eligible_candidates')
    if (not isinstance(eligible, list) or not eligible or
            any(type(x) is not int or x < 0 for x in eligible) or
            len(set(eligible)) != len(eligible)):
        raise ValueError('Selection-only dispatch requires explicit unique eligible integer epochs')
    result['properties']['action'] = {'type': 'string', 'enum': ['SELECT_CHECKPOINT']}
    result['properties']['selected_epoch'] = {'type': 'integer', 'enum': eligible}
    result['properties']['continue_training'] = {'type': 'boolean', 'enum': [False]}
    return result


def derive_contract(request, schema):
    bounds = request.get("allowed_parameter_bounds", {})
    definitions = schema["$defs"]
    schema_controls = definitions["parameter_update"]["properties"]["parameter"]["enum"]
    if set(bounds) != set(schema_controls):
        raise ValueError("Packet controls and output schema disagree")
    schedules = definitions["schedule_choice"]["properties"]["component"]["enum"]
    properties = definitions["intervention"]["properties"]
    result = {
        "role": "contract and arithmetic only; not a decision or measured training response",
        "decision_kind": request["decision_kind"],
        "allowed_actions": request["allowed_actions"],
        "controls": [{"name": name, "allowed_bounds": bounds[name]} for name in schema_controls],
        "parameter_update_minimum": properties["parameter_updates"].get("minItems", 0),
        "parameter_update_maximum": properties["parameter_updates"].get("maxItems"),
        "schedule_components": schedules,
        "schedule_choice_minimum": properties["schedule_choices"].get("minItems", 0),
        "initialization_requires_all_controls": request["decision_kind"] == "initialization",
    }
    if request.get('allowed_actions') == ['SELECT_CHECKPOINT']:
        narrowed = specialize_output_schema(request, schema)
        result['selection_artifact_contract'] = {
            'selected_epoch_required': True,
            'nullable': False,
            'eligible_epochs': narrowed['properties']['selected_epoch']['enum'],
            'continue_training': False,
            'uncertain_quality': 'Make a provisional eligible selection with a concrete alternative; do not claim measured quality.',
            'schema_specialized': schema['properties']['selected_epoch'] == narrowed['properties']['selected_epoch'],
        }
    probe = request.get("pre_run", {}).get("gradient_probe", {})
    raw = probe.get("raw_adv_to_segmentation_ratio")
    if raw is not None:
        raw = float(raw)
        if not math.isfinite(raw) or raw < 0:
            raise ValueError("Raw gradient ratio is nonfinite or negative")
        rows = []
        for key, value in probe.get("lambda_candidates", {}).items():
            if not key.startswith("effective_ratio_"):
                raise ValueError("Unknown lambda table key semantics")
            nominal = float(key[len("effective_ratio_"):])
            amplitude = float(value)
            if not all(math.isfinite(x) and x >= 0 for x in (nominal, amplitude)):
                raise ValueError("Invalid candidate ratio or amplitude")
            rows.append({"nominal_desired_ratio": nominal, "lambda_amplitude": amplitude,
                         "forward_estimated_weighted_ratio": amplitude * raw,
                         "meaning": "forward probe estimate; clipping may prevent nominal target"})
        result["gradient_probe_projection"] = {"raw_ratio": raw, "rows": rows,
               "formula": "estimated weighted ratio = lambda amplitude * raw gradient ratio",
               "actual_optimizer_update_measured": False}
    domain = request.get("baseline_review", {}).get("independent_frozen_feature_probe")
    if domain:
        result["frozen_feature_probe_split"] = {
            "accuracy": domain.get("accuracy"), "fit_samples": domain.get("train_n"),
            "validation_samples": domain.get("validation_n"),
            "accuracy_population": "validation split, not fit plus validation"}
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--output-schema", type=Path,
                        help="Write specialized schema to a new file before model dispatch")
    arguments = parser.parse_args()
    request = json.loads(arguments.request.read_text())
    schema = json.loads(arguments.schema.read_text())
    if arguments.output_schema:
        schema = specialize_output_schema(request, schema)
        with arguments.output_schema.open('x') as stream:
            stream.write(json.dumps(schema, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(derive_contract(request, schema), ensure_ascii=False, indent=2))
