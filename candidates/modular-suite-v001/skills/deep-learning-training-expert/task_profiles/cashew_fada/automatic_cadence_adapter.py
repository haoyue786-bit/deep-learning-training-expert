"""Reuse an existing protected automatic-negative executor with ten-epoch windows.

No second scientific model call: selection is provided by the same owner turn.
Do not patch a running/frozen legacy executor or renew an inherited spent pool.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    import os
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def validate_plan(plan, candidate_count):
    if not isinstance(plan,dict):raise ValueError('An explicit same-turn automatic-negative plan required')
    action=plan.get('action');count=plan.get('selected_count');weight=plan.get('automatic_weight')
    if isinstance(count,bool) or not isinstance(count,int) or not 0<=count<=candidate_count:
        raise ValueError('Pool selection must refer to the actual candidate population')
    if isinstance(weight,bool) or not isinstance(weight,(float,int)) or not math.isfinite(weight) or not 0<=weight<=.005:
        raise ValueError('Preserve the advertised independent low-dose bound')
    if (action=='ABSTAIN' and (count or weight)) or (action=='ACTIVATE' and (not count or not weight)) or action not in ('ABSTAIN','ACTIVATE'):
        raise ValueError('Activation/abstention controls disagree')
    ranks=plan.get('ranking_weights',{})
    if set(ranks)!={'source_cashew_distance','p2_increase','regional_prior'}:
        raise ValueError('Preserve the registered three evidence channels')
    if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) and 0<=v<=1 for v in ranks.values()):
        raise ValueError('Finite nonnegative ranking weights required')
    if abs(sum(ranks.values())-1)>1e-6 or ranks['regional_prior']<.25:
        raise ValueError('Preserve this Cashew cohort prior-ranking contract')
    for key in ['rationale','expected_response','reversal_condition']:
        if not isinstance(plan.get(key),str) or not plan[key].strip():raise ValueError('Scientific pool plan rationale/response/reversal required')
    return dict(plan)


def pool_class(existing_uncapped_pool):
    """Reuse its actual mask loading, auxiliary BN, loss, microbatching and probes."""
    class CadencedPool(existing_uncapped_pool):
        def __init__(self,manifest,expected,weight,cursor):
            super().__init__(manifest,expected,weight)
            self.cursor=int(cursor)

        def begin_epoch(self,steps,config,seed):
            age=float(config.get('_negative_epoch_start',0))-self.manifest['generation_epoch']
            horizon=int(self.manifest['exposure_horizon_updates'])
            max_age=int(self.manifest['policy']['maximum_age_epochs'])
            if steps<=0 or horizon<=0 or max_age<=0:raise ValueError('Positive finite generation schedule required')
            if age<0 or not 0<=self.cursor<=horizon:raise ValueError('Invalid generation age/cursor')
            self.exposed=[];self.pixels=0;self.losses=[];self.probes=[];self.age=age
            self.active_weight=self.weight if age<max_age and self.cursor<horizon and not config.get('_automatic_source_only_repair') else 0.
            rng=np.random.RandomState((int(self.manifest['schedule_seed'])+700001)%(2**32-1))
            order=rng.permutation(len(self.rows))
            slots=list(range(0,horizon,4))
            if len(order)>len(slots)*4:slots=list(range(horizon))
            entire={}
            if len(order):
                batches=[list(a) for a in np.array_split(order,min(len(slots),max(1,int(np.ceil(len(order)/4)))))]
                selected=np.rint(np.linspace(0,len(slots)-1,len(batches))).astype(int)
                entire={slots[int(slot)]:batch for slot,batch in zip(selected,batches)}
            self.next_cursor=min(horizon,self.cursor+steps)
            self.schedule=({k-self.cursor:v for k,v in entire.items() if self.cursor<=k<self.next_cursor} if self.active_weight else {})
            self.cumulative_expected_patches=len({i for k,v in entire.items() if k<self.next_cursor for i in v})
            # A complete generation contains every selected patch once, with no cap.
            flattened=[i for values in entire.values() for i in values]
            if len(flattened)!=len(self.rows) or len(set(flattened))!=len(self.rows):raise RuntimeError('Generation schedule lost or duplicated selected patches')

        def summary(self):
            value=super().summary()
            value.update(generation_start_cursor=self.cursor,generation_end_cursor=self.next_cursor,
                         generation_horizon_updates=int(self.manifest['exposure_horizon_updates']),
                         generation_expected_cumulative_exposed_patches=self.cumulative_expected_patches,
                         all_selected_patches_scheduled=self.next_cursor>=int(self.manifest['exposure_horizon_updates']),
                         occurrence_cap_per_generation=1,cadence_version='separate_observation_decision_v1')
            return value
    return CadencedPool


class AutomaticCadenceAdapter:
    def __init__(self,existing_controller,existing_uncapped_pool):
        self.controller=existing_controller;self.Pool=pool_class(existing_uncapped_pool)

    def refresh(self,plan,generation,*,checkpoint,review,config,resume_state_verified=False):
        epoch=generation['generation_epoch'];previous=config.get('_automatic_negative_state',{})
        persisted=self.controller.out/'state.json'
        if persisted.exists():
            disk=json.loads(persisted.read_text())
            if disk.get('generation_epoch')==epoch:
                if previous.get('generation_epoch')==epoch and (
                        previous.get('cursor',0)!=disk.get('cursor',0)
                        or any(previous.get(k)!=disk.get(k) for k in ('manifest','manifest_sha256'))):
                    raise ValueError('Checkpoint and pool exposure differ; reconcile complete training state before resuming')
                if previous.get('generation_epoch')!=epoch and disk.get('cursor',0) and not resume_state_verified:
                    raise ValueError('Consumed pool without matching checkpoint state requires explicit host resume verification')
                previous=disk;config['_automatic_negative_state']=disk
        if previous.get('generation_epoch')==epoch and previous.get('manifest'):
            p=Path(previous['manifest'])
            if not p.exists() or sha(p)!=previous['manifest_sha256']:raise ValueError('Consumed pool identity changed')
            committed=json.loads(p.read_text())
            if committed.get('cadence_version')=='separate_observation_decision_v1':
                if any(committed.get(k)!=generation[k] for k in ('exposure_horizon_updates','expiry_epoch')):
                    raise ValueError('A committed generation cannot be extended')
                decision_path=Path(committed['selection_decision'])
                if plan is not None and json.loads(decision_path.read_text())!=plan:
                    raise ValueError('A committed generation cannot be rebound to a different plan')
                return previous  # Exact continuation never resets its cursor or exposure.
            if previous.get('cursor',0):raise ValueError('Cannot renew an already consumed legacy generation')
        if plan is None:return None  # Abstain from a hidden second model turn.
        if generation['occurrence_cap_per_generation']!=1:raise ValueError('Do not silently increase automatic-label exposure')
        if generation['maximum_age_epochs']<=0 or generation['exposure_horizon_updates']<=0:
            raise ValueError('Positive finite generation budget required')
        destination=self.controller.out/('epoch_%03d'%epoch)
        decision_path=destination/'selection_decision.json'
        if decision_path.exists() and json.loads(decision_path.read_text())!=plan:
            raise ValueError('Cached scientific selection differs; do not silently reuse it')
        if (destination/'cadence10_manifest.json').exists():
            raise ValueError('Existing cadence sidecar has no matching committed state; reconcile exposure before resume')
        original_select=self.controller.select
        def same_turn(epoch,destination,rows,identity,review,controls):
            decision=validate_plan(plan,len(rows))
            dump(destination/'selection_decision.json',decision)
            dump(destination/'selection_provenance.json',dict(mode='same_scientific_owner_turn',
                 additional_model_calls=0,decision_sha256=sha(destination/'selection_decision.json')))
            return decision
        self.controller.select=same_turn
        try:self.controller.refresh(epoch,checkpoint,review,config)
        finally:self.controller.select=original_select
        state=config['_automatic_negative_state'];legacy=Path(state['manifest']);manifest=json.loads(legacy.read_text())
        validate_plan(plan,max(manifest.get('candidate_patch_count',0),manifest['requested_patch_count']))
        if (manifest['requested_patch_count']!=plan['selected_count'] or manifest['agent_selected_weight']!=plan['automatic_weight']
                or manifest['ranking_weights']!=plan['ranking_weights'] or sha(decision_path)!=manifest['selection_decision_sha256']):
            raise ValueError('Host generation does not match the same-turn plan')
        # A new sidecar preserves the original committed generation and its hash.
        # It is not safe to retrofit this onto a consumed old generation.
        if state.get('cursor',0)!=0:raise ValueError('Cannot enlarge or renew an already exposed legacy generation')
        manifest.update(exposure_horizon_updates=generation['exposure_horizon_updates'],
                        cadence_version='separate_observation_decision_v1',
                        original_manifest=dict(path=str(legacy),sha256=sha(legacy)),
                        expiry_epoch=generation['expiry_epoch'])
        manifest['policy']=dict(manifest.get('policy',{}),maximum_age_epochs=generation['maximum_age_epochs'])
        sidecar=legacy.with_name('cadence10_manifest.json')
        if sidecar.exists() and json.loads(sidecar.read_text())!=manifest:raise ValueError('Refusing to rewrite a generation sidecar')
        dump(sidecar,manifest)
        state.update(manifest=str(sidecar),manifest_sha256=sha(sidecar),cursor=0,
                     exposure_horizon_updates=generation['exposure_horizon_updates'])
        config['_automatic_negative_state']=state
        dump(self.controller.out/'state.json',state)
        return state

    def for_epoch(self,config):
        state=config.get('_automatic_negative_state')
        if not state:return None
        if sha(state['manifest'])!=state['manifest_sha256']:raise ValueError('Committed pool manifest identity changed')
        manifest=json.loads(Path(state['manifest']).read_text())
        if manifest.get('cadence_version')!='separate_observation_decision_v1':
            return self.controller.for_epoch(config)  # Preserve inherited 128-update pools verbatim.
        return self.Pool(state['manifest'],state['manifest_sha256'],state['weight'],state['cursor'])

    def after_epoch(self,config,metrics):return self.controller.after_epoch(config,metrics)
