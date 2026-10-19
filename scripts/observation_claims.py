"""Bind current-observation premises to typed, source-backed facts.

The owning task adapter supplies metric semantics. This module checks identity
and renders source values; it never infers metric meaning, tunes, or validates
free prose. IDs alone are insufficient when the claimed class/scope differs.
"""
import copy
import hashlib
import json


IDENTITY = ('metric_id', 'population_role', 'statistic', 'unit', 'time_label', 'state_id', 'class_id')


def packet_digest(packet):
    return hashlib.sha256(json.dumps(packet, sort_keys=True, ensure_ascii=False,
                                    allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def at_pointer(packet, pointer):
    if not isinstance(pointer, str) or not pointer.startswith('/'):
        raise ValueError('A nonroot JSON pointer is required')
    value = packet
    for part in pointer[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def build_registry(packet, definitions):
    """Definitions are worker-owned, verified against the metric producer.

    A wrong worker definition is not detectable from a numeric value. Keep that
    adapter's independent producer checks and original packet available.
    """
    facts = {}
    for definition in definitions:
        if set(definition) != {'fact_id', 'pointer'} | set(IDENTITY):
            raise ValueError('Every observation requires explicit metric semantics')
        fid = definition['fact_id']
        if not isinstance(fid, str) or not fid or fid in facts:
            raise ValueError('Unique nonempty fact IDs required')
        if any(not isinstance(definition[k], str) or not definition[k] for k in IDENTITY if k != 'class_id'):
            raise ValueError('Unknown metric meaning must be resolved or omitted, not silently guessed')
        if definition['class_id'] is not None and type(definition['class_id']) is not int:
            raise ValueError('Class ID must be an integer or explicitly inapplicable')
        facts[fid] = dict(copy.deepcopy(definition), value=copy.deepcopy(at_pointer(packet, definition['pointer'])))
    if not facts:
        raise ValueError('No current observations supplied')
    return dict(packet_sha256=packet_digest(packet), observations=facts)


def claim_schema(fact_ids):
    ids = list(fact_ids)
    if not ids or len(ids) != len(set(ids)) or any(not isinstance(x, str) or not x for x in ids):
        raise ValueError('Exact nonempty observation ID domain required')
    properties = {'fact_id': {'type': 'string', 'enum': ids}}
    properties.update({k: {'type': 'string'} for k in IDENTITY if k != 'class_id'})
    # A producer-owned registry can prevent opaque-hash copying errors at the
    # decoder. This does not decide which state belongs to a particular fact:
    # validate_claims still checks that exact pairing and all scientific types.
    if isinstance(fact_ids, dict) and all(isinstance(value, dict) for value in fact_ids.values()):
        states = sorted({value['state_id'] for value in fact_ids.values()})
        if not states or any(not isinstance(value, str) or not value for value in states):
            raise ValueError('Exact registered state-ID domain required')
        properties['state_id'] = {'type': 'string', 'enum': states}
    properties['class_id'] = {'anyOf': [{'type': 'integer'}, {'type': 'null'}]}
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def validate_claims(packet, registry, claims):
    if packet_digest(packet) != registry['packet_sha256']:
        raise ValueError('Observation registry belongs to a different packet/state')
    if not isinstance(claims, list):
        raise ValueError('Observation claims must be an array')
    checked, seen = [], set()
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {'fact_id'} | set(IDENTITY):
            raise ValueError('Current-observation claim fields are incomplete')
        fid = claim['fact_id']
        fact = registry['observations'].get(fid)
        if fact is None or fid in seen:
            raise ValueError('Unknown or duplicated current-observation ID')
        for key in IDENTITY:
            if type(claim[key]) is not type(fact[key]) or claim[key] != fact[key]:
                raise ValueError('Wrong ' + key + ' for observation ' + fid)
        actual = at_pointer(packet, fact['pointer'])
        if type(actual) is not type(fact['value']) or actual != fact['value']:
            raise ValueError('Registry value differs from original source pointer')
        checked.append(copy.deepcopy(fact)); seen.add(fid)
    return dict(state='current_observation_identities_checked', observations=checked,
                values_rendered_from_source=True, free_text_semantics_verified=False,
                scientific_inference_verified=False, completeness_verified=False,
                training_action_executed=False)
