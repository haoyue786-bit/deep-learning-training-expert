"""Constrain JSON-pointer citations to the actual supplied source packet.

This prevents invented locations, not a wrong interpretation of a real value.
The caller keeps its existing action/claim schema and owns semantic review.
"""
import argparse
import copy
import json
from pathlib import Path


def source_pointers(bundle):
    result = []
    def walk(value, prefix):
        children = value.items() if isinstance(value, dict) else enumerate(value) if isinstance(value, list) else []
        for key, child in children:
            escaped = str(key).replace('~', '~0').replace('/', '~1')
            pointer = prefix + '/' + escaped
            result.append(pointer)
            walk(child, pointer)
    walk(bundle, '')
    return result


def bind_pointer_schema(bundle, output_schema, *, field='source_claims', pointer_field='pointer', allowed_pointers=None):
    """Return a copy with exact legal locations enumerated before dispatch."""
    available = source_pointers(bundle)
    selected = available if allowed_pointers is None else list(allowed_pointers)
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(available):
        raise ValueError('Require a nonempty unique subset of actual source locations')
    output = copy.deepcopy(output_schema)
    claim = output['properties'][field]['items']
    if claim['type'] != 'object' or claim['properties'][pointer_field]['type'] != 'string':
        raise ValueError('Expected the caller existing object claims and string pointer field')
    claim['properties'][pointer_field]['enum'] = selected
    return output


def resolve_claims(bundle, claims, *, pointer_field='pointer'):
    """Resolve exact sources without treating interpretation as established fact."""
    available = set(source_pointers(bundle)); resolved = []
    for claim in claims:
        pointer = claim[pointer_field]
        if pointer not in available:
            raise ValueError('Citation is absent from the actual source packet: ' + str(pointer))
        value = bundle
        for token in pointer[1:].split('/'):
            key = token.replace('~1', '/').replace('~0', '~')
            value = value[int(key)] if isinstance(value, list) else value[key]
        resolved.append(dict(pointer=pointer, source_value=copy.deepcopy(value)))
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['bind', 'resolve'])
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='Caller schema for bind; answer for resolve')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    bundle = json.loads(args.bundle.read_text()); supplied = json.loads(args.input.read_text())
    result = bind_pointer_schema(bundle, supplied) if args.operation == 'bind' else resolve_claims(bundle, supplied['source_claims'])
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
