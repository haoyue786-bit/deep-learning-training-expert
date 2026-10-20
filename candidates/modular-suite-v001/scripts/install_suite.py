#!/usr/bin/env python3
"""Verify and install the nine sibling Skills; back up existing installations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def install(skills_root,backup_dir):
    suite=Path(__file__).resolve().parents[1]
    manifest=json.loads((suite/'BUILD_MANIFEST.json').read_text())
    for name,expected in manifest['files'].items():
        if hashlib.sha256((suite/name).read_bytes()).hexdigest()!=expected:
            raise ValueError('Source build manifest mismatch: '+name)
    root=Path(skills_root).expanduser().resolve();backup=Path(backup_dir).expanduser().resolve()
    if root==backup or root in backup.parents or backup in root.parents:
        raise ValueError('Use a backup directory separate from the Skill root')
    sources=sorted((suite/'skills').iterdir())
    sources=[p for p in sources if (p/'SKILL.md').is_file()]
    if len(sources)!=9:raise ValueError('Expected the complete nine-Skill suite')
    targets=[root/p.name for p in sources]
    if any(p.is_symlink() for p in targets):raise ValueError('Resolve existing Skill symlink ownership before replacement')
    existing=[p for p in targets if p.exists()]
    if any((backup/p.name).exists() for p in existing):raise ValueError('Choose an unused backup directory')
    root.mkdir(parents=True,exist_ok=True);backup.mkdir(parents=True,exist_ok=True)
    # Back up every existing target before replacing any target.
    for path in existing:shutil.copytree(path,backup/path.name,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for source,target in zip(sources,targets):
        staged=root/('.'+source.name+'.installing')
        if staged.exists():raise ValueError('An incomplete install needs reconciliation: '+str(staged))
        shutil.copytree(source,staged,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        if target.exists():shutil.rmtree(target)
        staged.rename(target)
    for source,target in zip(sources,targets):
        for path in source.rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc':
                if path.read_bytes()!=(target/path.relative_to(source)).read_bytes():raise ValueError('Installed bytes differ')
    return dict(installed=[p.name for p in sources],skills_root=str(root),backup_dir=str(backup),verified=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skills-root',required=True,type=Path)
    parser.add_argument('--backup-dir',required=True,type=Path)
    args=parser.parse_args();print(json.dumps(install(args.skills_root,args.backup_dir),indent=2))


if __name__=='__main__':main()
