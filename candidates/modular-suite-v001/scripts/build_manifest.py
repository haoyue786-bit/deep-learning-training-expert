#!/usr/bin/env python3
"""Build/check hashes of the modular suite without modifying legacy v028."""
import argparse
import hashlib
import json
from pathlib import Path
from sync_shared_references import sync


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1];target=root/'BUILD_MANIFEST.json'
    sync(check=True)
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(root.rglob('*')) if p.is_file() and p!=target
           and '__pycache__' not in p.parts and p.suffix!='.pyc' and '.tmp.' not in p.name}
    payload=dict(status='callable trial suite; bounded real-data CPU validation; no production migration or superiority claim',
                 version='modular_expert_revision_20261020',file_count=len(files),files=files,
                 scientific_defaults='skills/deep-learning-training-expert/scientific_defaults.json',
                 empirical_records='explicit authorized owning project only; not bundled Skill memory',
                 validation='tests/REAL_DATA_VALIDATION.md',legacy_root_unchanged=True)
    if args.check:
        if json.loads(target.read_text())!=payload:raise ValueError('Modular build manifest is stale')
    else:target.write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(dict(verified=args.check,files=len(files))))


if __name__=='__main__':main()
