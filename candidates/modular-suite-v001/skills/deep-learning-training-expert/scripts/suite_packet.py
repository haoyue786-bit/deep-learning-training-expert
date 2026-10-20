"""Package explicitly selected suite knowledge for a single model request.

No scientific routing, model call or token-saving guarantee is performed here.
Identical bodies are included once with all their source paths retained.
"""
import argparse
import hashlib
import json
from pathlib import Path


def build(suite_skills, skills, references=()):
    root = Path(suite_skills).resolve()
    if not skills or len(set(skills)) != len(skills):
        raise ValueError('Choose a nonempty unique skill list')
    if len(set(references)) != len(references):
        raise ValueError('Reference selections must be unique')
    files = []
    for name in skills:
        if not name or Path(name).name != name or name in ('.', '..'):
            raise ValueError('Skill name must be one directory component')
        files.append(name + '/SKILL.md')
    for name in references:
        parts = Path(name).parts
        if len(parts) < 3 or parts[0] not in skills or parts[1] not in ('references', 'task_profiles'):
            raise ValueError('Reference must belong to a selected skill')
        files.append(name)
    bodies = []
    by_digest = {}
    sources = []
    raw_bytes = 0
    for name in files:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('No absolute paths or traversal')
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            raise ValueError('Resolved file escapes suite')
        if path.suffix != '.md' or not path.is_file():
            raise ValueError('Existing Markdown knowledge file required')
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        raw_bytes += len(data)
        if digest not in by_digest:
            by_digest[digest] = len(bodies)
            bodies.append({'sha256': digest, 'text': data.decode('utf-8')})
        sources.append({'path': name, 'sha256': digest, 'body_index': by_digest[digest]})
    unique_bytes = sum(len(item['text'].encode('utf-8')) for item in bodies)
    settings=json.loads((root/'deep-learning-training-expert/scientific_defaults.json').read_text())
    return {
        'format': 'training_suite_single_request_v001',
        'skills': list(skills), 'sources': sources, 'bodies': bodies,
        'model':settings['model'],'reasoning_effort':settings['reasoning_effort'],
        'scientific_owner':'one continuing owner; no extra call per Skill or module',
        'size': {'raw_body_bytes': raw_bytes, 'unique_body_bytes': unique_bytes,
                 'duplicate_body_bytes_omitted': raw_bytes - unique_bytes,
                 'actual_token_count': None},
        'limits': 'Explicit packaging only. Caller must deliver the bodies; selection, comprehension and efficacy remain unverified. No across-request omission or assumed caching.'
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite-skills', required=True, type=Path)
    parser.add_argument('--skill', required=True, action='append')
    parser.add_argument('--reference', action='append', default=[])
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    packet = build(args.suite_skills, args.skill, args.reference)
    with args.output.open('x') as stream:
        json.dump(packet, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
