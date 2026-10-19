"""Check atomic comparison claims against the supplied evidence registry.

Identity/role/time/direction checks do not establish scientific inference or
free-text correctness. No model, data discovery, or training actions occur.
"""
import math


class EvidenceClaimError(ValueError):
    pass


def claim_schema(comparison_ids=None):
    properties = {
        'comparison_id': {'type': 'string'},
        'changed_controls': {'type': 'array', 'items': {'type': 'string'}},
        'observation_id': {'type': 'string'},
        'metric_id': {'type': 'string'},
        'population_role': {'type': 'string'},
        'time_label': {'type': 'string'},
        'effect': {'type': 'string', 'enum': ['increase', 'decrease', 'unchanged', 'unavailable']}}
    if comparison_ids is not None:
        ids = list(comparison_ids)
        if not ids or any(not isinstance(x, str) or not x for x in ids) or len(ids) != len(set(ids)):
            raise ValueError('A nonempty unique set of exact comparison IDs is required')
        properties['comparison_id']['enum'] = ids
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def effect_from_difference(value):
    if value is None:
        return 'unavailable'
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError('A difference must be a finite number or unavailable')
    return 'increase' if value > 0 else 'decrease' if value < 0 else 'unchanged'


def validate_claims(registry, claims):
    if not isinstance(claims, list):
        raise EvidenceClaimError('Evidence claims must be an array')
    checked = []
    seen = set()
    required = set(claim_schema()['properties'])
    for i, claim in enumerate(claims):
        if not isinstance(claim, dict) or set(claim) != required:
            raise EvidenceClaimError('Incomplete claim at index ' + str(i))
        comparison = registry.get('comparisons', {}).get(claim['comparison_id'])
        if comparison is None:
            raise EvidenceClaimError('Unknown comparison: ' + str(claim['comparison_id']))
        controls = claim['changed_controls']
        if (not isinstance(controls, list) or any(not isinstance(s, str) for s in controls) or
                len(controls) != len(set(controls)) or
                set(controls) != set(comparison['changed_controls'])):
            raise EvidenceClaimError('Wrong changed controls for ' + claim['comparison_id'])
        observation = comparison['observations'].get(claim['observation_id'])
        if observation is None:
            raise EvidenceClaimError('Observation does not belong to this comparison')
        for key in ('metric_id', 'population_role', 'time_label'):
            if claim[key] != observation[key]:
                raise EvidenceClaimError('Wrong ' + key + ' for ' + claim['observation_id'])
        actual_effect = effect_from_difference(observation['difference'])
        if claim['effect'] != actual_effect:
            raise EvidenceClaimError('Wrong recorded difference direction for ' + claim['observation_id'])
        identity = (claim['comparison_id'], claim['observation_id'])
        if identity in seen:
            raise EvidenceClaimError('Duplicate evidence claim')
        seen.add(identity)
        checked.append({'comparison_id': claim['comparison_id'],
                        'changed_controls': comparison['changed_controls'],
                        'observation_id': claim['observation_id'], **observation,
                        'effect': actual_effect})
    return {'state': 'atomic_comparison_claims_checked', 'claims': checked,
            'scientific_inference_verified': False, 'free_text_semantics_verified': False,
            'evidence_completeness_verified': False, 'training_action_executed': False}
