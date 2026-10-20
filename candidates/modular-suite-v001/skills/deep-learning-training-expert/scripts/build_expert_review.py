"""Build the compact decision brief and full-trajectory curve without training."""
import argparse
import json
from pathlib import Path
from training_runtime.expert_review import build


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",required=True,type=Path)
    parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    brief=build(json.loads(args.input.read_text()),args.output)
    print(json.dumps(dict(brief=str(args.output/"brief.json"),image=brief["image"],size=brief["size"])))
