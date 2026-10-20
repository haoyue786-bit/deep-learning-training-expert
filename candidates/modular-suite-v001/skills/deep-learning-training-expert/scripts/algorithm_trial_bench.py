#!/usr/bin/env python3
"""One project config: inspect real validation feedback, then run one bounded trial."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--trial',help='One measured-error-supported module:variant; omitted = feedback only')
    parser.add_argument('--controls',default='{}',help='One explicit JSON control object, never a parameter grid')
    parser.add_argument('--strength',type=float,default=.01)
    parser.add_argument('--interface-smoke',action='store_true',help='Call all implemented kernels on real training data; zero optimizer/validation updates')
    args=parser.parse_args();config=json.loads(args.config.read_text())
    if args.interface_smoke and args.trial:parser.error('Choose one interface check or one paired trial')
    skill=Path(__file__).resolve().parents[1];suite=skill.parent
    sys.path.insert(0,str(suite/'training-experience-update/scripts'))
    from project_experience import ProjectStore
    from training_runtime.trial_bench import TrialBench
    store=ProjectStore(config['project'])
    if config.get('adapter')!='cashew_fada':raise ValueError('Choose an implemented project adapter explicitly')
    # Import the existing project adapter only after validating project write identity.
    spec=importlib.util.spec_from_file_location('cashew_trial_adapter',skill/'task_profiles/cashew_fada/trial_adapter.py')
    adapter_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(adapter_module)
    bench=TrialBench(store,config);adapter=adapter_module.CashewTrialAdapter(config)
    from training_runtime.trial_bench import dump
    dump(bench.root/'data_manifest.json',adapter.data_manifest)
    if args.interface_smoke:
        result=adapter.interface_smoke()
        dump(bench.root/'interface_smoke.json',result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    elif args.trial:
        module_id,variant=args.trial.split(':',1)
        result=bench.run_pair(adapter,module_id=module_id,variant=variant,controls=json.loads(args.controls),strength=args.strength)
        print(json.dumps({k:v for k,v in result.items() if k!='results'},ensure_ascii=False,indent=2))
    else:
        packet=bench.prepare(adapter)
        print(json.dumps(packet,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
