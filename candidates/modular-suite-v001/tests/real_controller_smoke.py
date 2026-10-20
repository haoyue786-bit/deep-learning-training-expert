#!/usr/bin/env python3
"""Opt-in real legacy controller integration on four cached protected candidates."""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--catalog-directory',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    skill=Path(__file__).resolve().parents[1]/'skills/deep-learning-training-expert'
    sys.path.insert(0,str(skill.parent/'training-experience-update/scripts'))
    from project_experience import ProjectStore,file_evidence
    store=ProjectStore(config['project'])
    job=json.loads(Path(config['job_file']).read_text());job=copy.deepcopy(job)
    out=Path(config['run_dir']).resolve()/'automatic_controller_smoke'
    try:out.relative_to(store.project_root)
    except ValueError:raise PermissionError('Smoke output must stay in authorized project')
    out.mkdir(parents=True,exist_ok=True);job['output_root']=str(out)
    runtime=Path(job['runtime_provenance']['root']);sys.path.insert(0,str(runtime/'scripts'))
    from automatic_negative_production_v4 import Controller
    from automatic_negative_pool_uncapped_v3 import AutomaticNegativePool
    spec=importlib.util.spec_from_file_location('cadence_smoke',skill/'task_profiles/cashew_fada/automatic_cadence_adapter.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    import torch
    controller=Controller(job,None,None,None,torch.device('cpu'))
    catalog=args.catalog_directory/'candidate_catalog.json';identity_path=args.catalog_directory/'catalog_identity.json'
    identity=json.loads(identity_path.read_text());rows=json.loads(catalog.read_text())[:4]
    protected=[file_evidence(p) for p in [catalog,identity_path,args.catalog_directory/'manifest.json']]
    for row in rows:
        if control.sha(row['cache'])!=row['cache_sha256']:raise ValueError('Cached candidate identity changed')
        protected.extend(file_evidence(row[k]) for k in ('cache','image_path'))
    if control.sha(identity['source_checkpoint'])!=identity['source_checkpoint_sha256']:raise ValueError('Checkpoint identity changed')
    hidden=[]
    def forbidden(*args):hidden.append(True);raise AssertionError('Hidden second API call')
    controller.select=forbidden
    controller.catalog=lambda *args:(rows,identity)  # Bounded real protected cache; no full-country scan.
    adapter=control.AutomaticCadenceAdapter(controller,AutomaticNegativePool)
    plan=dict(action='ACTIVATE',selected_count=len(rows),automatic_weight=.001,
              ranking_weights=dict(source_cashew_distance=.25,p2_increase=.25,regional_prior=.5),
              rationale='Isolated execution check on existing protected candidates',expected_response='Actual mask and finite pool creation',
              reversal_condition='Any identity/exposure inconsistency')
    generation=dict(generation_epoch=10,expiry_epoch=20,exposure_horizon_updates=70,maximum_age_epochs=10,occurrence_cap_per_generation=1)
    config_state={}
    state=adapter.refresh(plan,generation,checkpoint=identity['source_checkpoint'],review={},config=config_state)
    resumed=adapter.refresh(plan,generation,checkpoint=identity['source_checkpoint'],review={},config={})
    assert state==resumed and not hidden
    assert all(file_evidence(p['path'])==p for p in protected)
    result=dict(status='completed',actual_existing_controller=True,actual_cached_candidates=len(rows),
                new_mask_count=state['selected_patch_count'],generation_horizon_updates=70,
                resume_cursor=state['cursor'],additional_api_calls=len(hidden),original_inputs_unchanged=True,
                optimizer_updates=0,scope='same-turn plan, actual protected-mask generation and resume interface; no efficacy claim')
    result_path=out/'result.json';control.dump(result_path,result)
    store.record(dict(case_id='real-controller-cadence-smoke-20261020',project_id=store.project_id,
        action={'operation':'reuse actual controller with bounded cached catalog'},result=result,outcome='observation',
        evidence=[file_evidence(result_path)],conditions={'cached_protected_candidates':True,'new_catalog_generation_tested':False},
        hypotheses=['Full production adoption still requires a versioned host job.'],cost={'optimizer_updates':0,'model_calls':0},
        evaluation_role='training_diagnostic',tags=['interface_smoke','M08']))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
