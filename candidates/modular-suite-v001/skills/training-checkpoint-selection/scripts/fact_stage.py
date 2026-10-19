"""Build a read-only fact packet and check evidence, never training decisions.

Schema-valid and mechanically checked does not mean semantically correct.
Requires jsonschema for validation; it performs no model calls or code execution.
"""
import argparse
import hashlib
import json
import math
import re
from pathlib import Path


def obj(properties):
    return dict(type='object', properties=properties, required=list(properties), additionalProperties=False)


def schema():
    string = dict(type='string')
    strings = dict(type='array', items=string)
    citation = obj(dict(file=string, start_line=dict(type='integer', minimum=1),
                        end_line=dict(type='integer', minimum=1)))
    citations = dict(type='array', items=citation, minItems=1)
    fact = obj(dict(id=string, statement=string,
        basis=dict(type='string', enum=['recorded', 'code_inferred', 'unknown']),
        value=dict(type=['number', 'null'], description='Use the raw numeric value for a directly reported scalar; use null for a nonnumeric fact or unavailable scalar. Put narrative descriptions in statement, never encode a paraphrase as a recorded literal.'), population=string, statistic=string,
        unit=string, time=string, state=string, evidence=citations, uncertainty=string))
    event = obj(dict(order=dict(type='integer', minimum=0), time=string,
        kind=dict(type='string', enum=['observation', 'proposed', 'executed', 'failed', 'adopted', 'unknown']),
        statement=string, evidence=citations))
    conflict = obj(dict(statement=string, evidence=citations, resolution=string))
    operand = dict(type='string', description='Exact facts[].id of a separately declared recorded numeric fact. No descriptions, numeric strings, field paths or appended values.')
    calculation = obj(dict(id=string, operation=dict(type='string', enum=['difference', 'ratio']),
        left_fact=operand, right_fact=operand, value=dict(type='number'), unit=string))
    return obj(dict(mode=dict(type='string', enum=['facts_only']), task=string,
        facts=dict(type='array', items=fact, minItems=1), timeline=dict(type='array', items=event),
        conflicts=dict(type='array', items=conflict), unknowns=strings,
        calculations=dict(type='array', items=calculation)))


def sha(data):
    return hashlib.sha256(data).hexdigest()


def recorded_numeric_literals(text):
    """Read numbers including conventional adjacent units, without conversion.

    The old word boundary rejected the real author-log token `5.376s`.
    Recognizing a coefficient still does not verify a claim's units/meaning.
    Identifier/version fragments such as E100, v1.2 and 12abc stay excluded.
    """
    number = r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?'
    units = r'(?:ns|us|µs|μs|ms|s|sec|secs|seconds?|min|mins|minutes?|h|hr|hrs|hours?|B|KB|MB|GB|TB|KiB|MiB|GiB|TiB|Hz|kHz|MHz|GHz|W|kW)'
    boundary = r'(?=$|[^\w.]|' + units + r'(?![\w.]))'
    return [float(value) for value in re.findall(r'(?<![\w.])' + number + boundary, text)]


def make_bundle(records):
    root = Path(records).resolve(strict=True)
    files = {}
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        resolved = path.resolve(strict=True)
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError('Record symlink escapes bundle root')
        data = path.read_bytes()
        content = data.decode('utf-8')
        files[str(path.relative_to(root))] = dict(sha256=sha(data), lines=len(content.splitlines()))
    if not files:
        raise ValueError('No explicit records supplied')
    return dict(records_root=str(root), files=files)


def read_bound(bundle):
    root = Path(bundle['records_root']).resolve(strict=True)
    loaded = {}
    for name, identity in bundle['files'].items():
        path = Path(name)
        if path.is_absolute() or '..' in path.parts:
            raise ValueError('Invalid record name')
        path = (root / path).resolve(strict=True)
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError('Record escapes bundle root')
        data = path.read_bytes()
        if sha(data) != identity['sha256']:
            raise ValueError('Record hash mismatch: ' + name)
        lines = data.decode('utf-8').splitlines()
        if len(lines) != identity['lines']:
            raise ValueError('Record line count mismatch: ' + name)
        loaded[name] = lines
    return loaded


def prompt(bundle, question):
    parts = ['Fact reconstruction only. Use the supplied records and your assigned Skill. '
        'Return the exact attached schema. Do not recommend or execute training actions. '
        'Use original record evidence; state unresolved information. No tools, network, '
        'filesystem reads or extra model calls; all permitted source text follows. '
        'For evidence supply file and exact line bounds; Python extracts the original span. '
        'Do not copy, paraphrase or abbreviate quotations in output. For numeric facts use '
        'the unconverted raw number in value; for nonnumeric facts use value=null and explain '
        'in statement. Label implementation deductions separately from logged historical observations. '
        'For a calculation, declare each operand as its own recorded numeric fact with a distinct id and value. '
        'Set left_fact and right_fact to those exact ids; a description such as "fact_x: 0.5 score" is not an id. '
        'Do not compute against a second number mentioned only in another fact statement.',
        'TASK\n' + question, 'OUTPUT SCHEMA\n' + json.dumps(schema(), separators=(',', ':'))]
    for name, lines in read_bound(bundle).items():
        parts.append('FILE ' + name + '\n' + '\n'.join('%d: %s' % (i, line) for i, line in enumerate(lines, 1)))
    return '\n\n'.join(parts)


def validate(report, bundle):
    import jsonschema
    jsonschema.Draft202012Validator(schema()).validate(report)
    files = read_bound(bundle)
    checked_citations = 0
    resolved_citations = []

    def cite(citation):
        nonlocal checked_citations
        if citation['file'] not in files:
            raise ValueError('Citation is not in explicit bundle')
        lines = files[citation['file']]
        start, end = citation['start_line'], citation['end_line']
        if start > end or end > len(lines):
            raise ValueError('Citation line span invalid')
        text = '\n'.join(lines[start - 1:end])
        resolved_citations.append(dict(file=citation['file'], start_line=start, end_line=end,
                                       source_sha256=bundle['files'][citation['file']]['sha256'], quote=text))
        checked_citations += 1
        return text

    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value)

    facts = {}
    for fact in report['facts']:
        if not fact['id'] or fact['id'] in facts:
            raise ValueError('Duplicate or empty fact ID')
        facts[fact['id']] = fact
        quotes = [cite(c) for c in fact['evidence']]
        value = fact['value']
        if type(value) in (int, float) and not numeric(value):
            raise ValueError('Nonfinite fact value')
        if fact['basis'] == 'unknown' and value is not None:
            raise ValueError('Unknown observation cannot have an invented value')
        if fact['basis'] == 'recorded':
            if numeric(value):
                tokens = [token for q in quotes for token in recorded_numeric_literals(q)]
                if not any(math.isclose(value, token, rel_tol=1e-12, abs_tol=1e-14) for token in tokens):
                    raise ValueError('Recorded numeric value absent from quoted evidence')
    for group in ('timeline', 'conflicts'):
        for item in report[group]:
            for citation in item['evidence']:
                cite(citation)
    orders = [e['order'] for e in report['timeline']]
    if orders != sorted(set(orders)):
        raise ValueError('Timeline order keys must be unique and increasing')
    calculation_ids = set()
    calculation_review = []
    for item in report['calculations']:
        if not item['id'] or item['id'] in calculation_ids or item['id'] in facts:
            raise ValueError('Calculation ID duplicates another result')
        calculation_ids.add(item['id'])
        missing = [item[k] for k in ('left_fact', 'right_fact') if item[k] not in facts]
        if missing:
            raise ValueError('Calculation operand must be an exact declared numeric fact ID; unresolved: ' + ', '.join(missing))
        left, right = facts[item['left_fact']], facts[item['right_fact']]
        if not numeric(left['value']) or not numeric(right['value']):
            raise ValueError('Calculation operands are not measured numeric facts')
        if left['basis'] != 'recorded' or right['basis'] != 'recorded':
            raise ValueError('Calculation operands must reference recorded observations')
        # These fields are natural-language descriptions, not canonical dataset
        # or metric IDs. Text equality cannot establish semantic comparability,
        # and paraphrases must not become a mechanical rejection criterion.
        review = dict(id=item['id'], population_comparability_verified=False,
            statistic_comparability_verified=False, unit_interpretation_verified=False,
            left_description={k:left[k] for k in ('population','statistic','unit','state','time')},
            right_description={k:right[k] for k in ('population','statistic','unit','state','time')},
            requirement='Semantic review must establish compatible populations/statistics and declared units, including best-versus-current comparisons.')
        if item['operation'] == 'difference':
            expected = left['value'] - right['value']
            if left['unit'] == 'fraction' and item['unit'] == 'percentage_points':
                expected *= 100
                review['unit_interpretation_verified'] = right['unit'] == 'fraction'
            elif left['unit'] == 'percent' and item['unit'] == 'percentage_points':
                review['unit_interpretation_verified'] = right['unit'] == 'percent'
            elif item['unit'] == 'percentage_points':
                raise ValueError('Percentage-point conversion requires explicit fraction or percent operands')
            elif item['unit'] == left['unit'] == right['unit']:
                review['unit_interpretation_verified'] = left['unit'] in ('fraction','percent','count','seconds','dimensionless')
        else:
            if right['value'] == 0:
                raise ValueError('Undefined ratio')
            expected = left['value'] / right['value']
            if item['unit'] == 'percent':
                expected *= 100
            elif item['unit'] != 'ratio':
                raise ValueError('Ratio must use ratio or percent unit')
        if not numeric(item['value']) or not math.isclose(item['value'], expected, rel_tol=1e-9, abs_tol=1e-12):
            raise ValueError('Calculation mismatch')
        review['arithmetic_verified'] = True
        calculation_review.append(review)
    return dict(state='mechanically_checked_semantic_review_required',
        facts=len(facts), citations=checked_citations, calculations=len(report['calculations']),
        resolved_citations=resolved_citations,
        calculation_review=calculation_review,
        semantic_correctness_verified=False, training_decision_authorized=False,
        limitations='Exact evidence and arithmetic do not verify population meaning, chronology, omissions or interpretation.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    prep.add_argument('--records', type=Path, required=True)
    prep.add_argument('--output', type=Path, required=True)
    prep.add_argument('--question', type=Path, required=True)
    prep.add_argument('--prompt', type=Path, required=True)
    check = sub.add_parser('validate')
    check.add_argument('--bundle', type=Path, required=True)
    check.add_argument('--report', type=Path, required=True)
    sub.add_parser('schema')
    args = parser.parse_args()
    if args.command == 'schema':
        print(json.dumps(schema(), indent=2))
    elif args.command == 'prepare':
        bundle = make_bundle(args.records)
        for destination in (args.output, args.prompt):
            try:
                destination.resolve().relative_to(args.records.resolve())
            except ValueError:
                pass
            else:
                raise ValueError('Generated files must stay outside original records')
        args.output.write_text(json.dumps(bundle, indent=2) + '\n')
        args.prompt.write_text(prompt(bundle, args.question.read_text()))
        print(json.dumps(dict(state='prepared', files=len(bundle['files']), model_calls=0)))
    else:
        result = validate(json.loads(args.report.read_text()), json.loads(args.bundle.read_text()))
        print(json.dumps(result))


if __name__ == '__main__':
    main()
