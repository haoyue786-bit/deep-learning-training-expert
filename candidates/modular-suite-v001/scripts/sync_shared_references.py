"""Maintain independent Skill copies from one canonical source per shared reference."""
import argparse
from pathlib import Path

GROUPS = {
    'optimization-practice.md': ('training-adjustment', ['training-initialization','training-stop','training-diagnosis']),
    'stopping-and-selection.md': ('training-stop', ['training-checkpoint-selection']),
    'factual-reconstruction.md': ('training-fact-reconstruction', ['training-checkpoint-selection']),
    'diagnosis-and-experiments.md': ('training-diagnosis', ['training-adjustment','training-experiment']),
    'adversarial-practice.md': ('training-diagnosis', ['training-adjustment']),
}


def sync(check=False):
    skills=Path(__file__).resolve().parents[1]/'skills'
    for name,(owner,copies) in GROUPS.items():
        body=(skills/owner/'references'/name).read_bytes()
        for copy in copies:
            target=skills/copy/'references'/name
            if check:
                if target.read_bytes()!=body:raise ValueError('Shared reference drift: '+str(target))
            else:target.write_bytes(body)
    return len(GROUPS)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true')
    print('Verified shared reference groups:',sync(parser.parse_args().check))
