#!/usr/bin/env python3
"""Opt-in real-data kernel and inherited automatic-pool loss/gradient checks.

Needs an authorized private config and existing protected pool manifest. This
does not join unittest discovery, call an API, step an optimizer or edit a job.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys

import torch

SKILL=Path(__file__).resolve().parents[1]/'skills/deep-learning-training-expert'
sys.path.insert(0,str(SKILL/'scripts'))
sys.path.insert(0,str(SKILL.parent/'training-experience-update/scripts'))
from project_experience import ProjectStore,file_evidence
from training_runtime.trial_bench import TrialBench,dump,state_digest
from training_runtime.cadence import Cadence


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--automatic-manifest',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    store=ProjectStore(config['project']);bench=TrialBench(store,config)
    adapter=load('real_cashew_trial',SKILL/'task_profiles/cashew_fada/trial_adapter.py').CashewTrialAdapter(config)
    smoke=adapter.interface_smoke()
    control=load('real_cashew_cadence',SKILL/'task_profiles/cashew_fada/automatic_cadence_adapter.py')
    from automatic_negative_pool_uncapped_v3 import AutomaticNegativePool
    original=copy.deepcopy(adapter.snapshot());identity=state_digest(original)
    source=file_evidence(args.automatic_manifest)
    manifest=json.loads(args.automatic_manifest.read_text())
    # A separate test generation with at most four real protected rows. This
    # leaves the inherited committed generation/cursor/manifest untouched.
    manifest['selected']=manifest['selected'][:4]
    if not manifest['selected']:raise ValueError('Choose an existing nonempty protected generation')
    manifest.update(selected_patch_count=len(manifest['selected']),
                    **Cadence().generation(0,steps_per_epoch=2,maximum_epoch=10))
    manifest['policy']=dict(manifest['policy'],maximum_age_epochs=10)
    manifest['source_smoke_manifest']=source
    path=bench.root/'automatic_interface_smoke_manifest.json';dump(path,manifest)
    Pool=control.pool_class(AutomaticNegativePool)
    cursor=0;seen=[];loss_value=None;gradient_norm=None
    try:
        for age in range(10):
            pool=Pool(path,control.sha(path),.001,cursor)
            pool.begin_epoch(2,{'_negative_epoch_start':age},16)
            seen.extend(i for indices in pool.schedule.values() for i in indices)
            if loss_value is None and pool.schedule:
                adapter.model_opt.zero_grad(set_to_none=True)
                step=min(pool.schedule)
                loss=pool.step(step,adapter.gan,adapter.common,adapter.model,adapter.device,16)
                if loss is None or not torch.isfinite(loss):raise ValueError('Actual inherited pool loss did not execute finitely')
                loss.backward();loss_value=float(loss.detach())
                gradients=[p.grad for p in adapter.model.parameters() if p.grad is not None]
                if not gradients or not all(torch.isfinite(g).all() for g in gradients):raise ValueError('Invalid inherited automatic-loss gradients')
                gradient_norm=float(torch.sqrt(sum(g.square().sum() for g in gradients)))
                if gradient_norm<=0:raise ValueError('No actual automatic-loss gradient')
            cursor=pool.summary()['generation_end_cursor']
        expired=Pool(path,control.sha(path),.001,cursor)
        expired.begin_epoch(2,{'_negative_epoch_start':10},16)
    finally:adapter.restore(original)
    if sorted(seen)!=list(range(len(manifest['selected']))):raise AssertionError('Duplicate or absent exposure in generation')
    if expired.schedule or expired.active_weight:raise AssertionError('Expired generation remained active')
    if state_digest(adapter.snapshot())!=identity:raise AssertionError('Full reference not restored')
    if file_evidence(args.automatic_manifest)!=source:raise AssertionError('Original manifest changed')
    result=dict(status='completed',kernels=smoke,automatic=dict(actual_inherited_loss=loss_value,
        actual_model_gradient_norm=gradient_norm,actual_selected_patches=len(manifest['selected']),
        scheduled_unique_patches=len(set(seen)),occurrences=len(seen),expired_after_epoch=10,cursor=cursor,
        original_manifest_unchanged=True,model_optimizer_rng_restored=True),optimizer_updates=0,model_calls=0)
    result_path=bench.root/'real_interface_smoke.json';dump(result_path,result)
    store.record(dict(case_id=bench.root.name+':real_interface_smoke',project_id=store.project_id,
        action=dict(operation='real-data kernel and inherited automatic-loss smoke'),result=result,outcome='observation',
        evidence=[file_evidence(result_path),source],conditions=dict(scope='execution/gradient only; not recognition efficacy'),
        hypotheses=['Callable mechanisms may still yield no classification benefit.'],cost=dict(optimizer_updates=0,model_calls=0),
        evaluation_role='training_diagnostic',tags=['interface_smoke','M08']))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
