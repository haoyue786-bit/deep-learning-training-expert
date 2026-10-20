"""Journaled epoch boundary; host owns scientific dispatch and complete train state."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
from pathlib import Path

from .cadence import Cadence, numerical_anomalies
from .decision_packet import defaults
from .control_changes import validate_control_changes
from .trial_bench import dump


def identity(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,allow_nan=False).encode()).hexdigest()


class ReviewDriver:
    """Fresh callbacks retain the five-hook API. Recovery is explicit and journaled.

    Recovery hooks: verify_resume_state(epoch, run_identity) -> True;
    reconcile_operation(stage, operation_id) -> {status: completed, result: ...}
    or {status: not_started, evidence: ...}. Unknown effects never auto-retry.
    Optional begin_operation(stage, operation_id) binds external idempotency keys.
    Production hosts also validate_decision(decision), commit_review(epoch, receipt),
    and deliver the supplied knowledge/contract in their real model request.
    """
    def __init__(self, *, output, maximum_epoch, steps_per_epoch, hooks, cadence=None,
                 run_identity=None, scientific_context=None, maximum_stage_attempts=3,
                 production=False):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.maximum_epoch=maximum_epoch;self.steps_per_epoch=steps_per_epoch;self.hooks=hooks
        self.cadence=cadence or Cadence.from_job({'review_cadence':defaults()['review_cadence']})
        if isinstance(maximum_epoch,bool) or not isinstance(maximum_epoch,int) or maximum_epoch<=0:
            raise ValueError('A positive training epoch cap required')
        if isinstance(maximum_stage_attempts,bool) or not isinstance(maximum_stage_attempts,int) or not 1<=maximum_stage_attempts<=5:
            raise ValueError('Finite per-stage retry cap required')
        self.maximum_stage_attempts=maximum_stage_attempts
        self.cadence.generation(0,steps_per_epoch=steps_per_epoch,maximum_epoch=maximum_epoch)
        self.context=json.loads(json.dumps(scientific_context or {},allow_nan=False))
        if production:
            required=('validate_decision','commit_review','verify_resume_state','reconcile_operation',
                      'begin_operation','review_context')
            if not run_identity or any(not callable(getattr(hooks,k,None)) for k in required):
                raise ValueError('Production mode requires explicit run identity and complete host integration hooks')
            if not self.context.get('contract_identity') or not self.context.get('knowledge_identity'):
                raise ValueError('Production mode requires exact host contract and knowledge identities')
        self.run_identity=json.loads(json.dumps(run_identity or {
            'local_output':str(self.output.resolve()),'scope':'callback_integration_only'},allow_nan=False))
        self.binding=dict(run_identity=self.run_identity,maximum_epoch=maximum_epoch,
                          steps_per_epoch=steps_per_epoch,cadence=self.cadence.policy(),
                          scientific_context_sha256=identity(self.context),maximum_stage_attempts=maximum_stage_attempts)
        self.binding['production']=production
        with self._lock():
            path=self.output/'review_identity.json'
            if path.exists():
                if json.loads(path.read_text())!=self.binding:raise ValueError('Review directory belongs to a different run/branch/contract')
            elif any(p.name!='.review.lock' for p in self.output.iterdir()):
                raise ValueError('Existing unbound output requires explicit migration to a new directory')
            else:dump(path,self.binding)

    @contextmanager
    def _lock(self):
        with (self.output/'.review.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);yield

    def _path(self,epoch,suffix):return self.output/f'epoch_{epoch:04d}_{suffix}.json'

    def _save(self,journal):dump(self._path(journal['epoch'],'journal'),journal)

    def _validate(self, decision, journal):
        if not isinstance(decision,dict) or not isinstance(decision.get('continue_training'),bool):
            raise ValueError('Explicit continue_training required')
        dynamic=journal['stages'].get('context',{}).get('result',{})
        if self.binding['production'] or 'control_baseline' in dynamic:
            if not isinstance(dynamic.get('control_baseline'),dict) or not isinstance(dynamic.get('control_evidence'),dict):
                raise ValueError('Host must supply current control_baseline and control_evidence before deciding')
            validate_control_changes(before=dynamic['control_baseline'],after=decision.get('controls'),
                change_basis=decision.get('change_basis'),evidence_catalog=dynamic['control_evidence'])
        validator=getattr(self.hooks,'validate_decision',None)
        if validator and validator(decision) is False:raise ValueError('Native validator rejected the answer')

    def _stage(self,journal,name,function):
        stages=journal['stages'];record=stages.get(name)
        operation_id=identity([self.binding,journal['epoch'],name])
        if record and record['status']=='completed':return record['result']
        if record:
            reconcile=getattr(self.hooks,'reconcile_operation',None)
            if not reconcile:raise RuntimeError('Host must reconcile uncertain operation: '+name)
            resolution=reconcile(name,operation_id)
            if not isinstance(resolution,dict) or not resolution.get('evidence'):
                raise RuntimeError('Reconciliation requires an evidence receipt')
            record.setdefault('reconciliations',[]).append(resolution);self._save(journal)
            if resolution.get('status')=='completed':
                if 'result' not in resolution:raise ValueError('Completed reconciliation needs the actual result')
                record.update(status='completed',result=resolution['result']);self._save(journal)
                return record['result']
            if resolution.get('status')!='not_started':raise RuntimeError('Operation outcome remains unknown: '+name)
        else:
            record=dict(operation_id=operation_id,attempts=0);stages[name]=record
        if record['attempts']>=self.maximum_stage_attempts:raise RuntimeError('Per-stage retry budget exhausted: '+name)
        record.update(status='started',attempts=record['attempts']+1);self._save(journal)
        try:
            bind=getattr(self.hooks,'begin_operation',None)
            if bind:bind(name,operation_id)
            result=function()
            record.update(status='completed',result=result);self._save(journal)
            return result
        except Exception as exc:
            record.update(status='uncertain',error={'type':type(exc).__name__,'message':str(exc)})
            self._save(journal);raise

    def after_epoch(self,epoch,metrics,*,emergency=False):
        with self._lock():
            if self._path(epoch,'telemetry').exists():raise ValueError('Epoch already observed; use resume(epoch)')
            unfinished=[p for p in self.output.glob('epoch_*_journal.json') if json.loads(p.read_text()).get('status')!='complete']
            if unfinished:raise RuntimeError('Reconcile the pending epoch before advancing')
            prior=[json.loads(p.read_text())['epoch'] for p in self.output.glob('epoch_*_telemetry.json')]
            if prior and epoch<=max(prior):raise ValueError('Epochs must advance within this run/branch')
            events=self.cadence.events(epoch,maximum_epoch=self.maximum_epoch,emergency=emergency)
            issues=numerical_anomalies(metrics)
            def safe(value):
                if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [safe(v) for v in value]
                if isinstance(value,(float,int)):
                    import math
                    if not math.isfinite(value):return str(value)
                return value
            journal=dict(epoch=epoch,events=events,metrics=safe(metrics),numerical_anomalies=issues,
                         run_identity=self.run_identity,stages={},status='pending')
            # Journal first; resume also repairs a missing telemetry write after a crash.
            self._save(journal)
            dump(self._path(epoch,'telemetry'),dict(epoch=epoch,metrics=safe(metrics),numerical_anomalies=issues))
            return self._advance(journal)

    def resume(self,epoch):
        with self._lock():
            journal=json.loads(self._path(epoch,'journal').read_text())
            if journal['run_identity']!=self.run_identity:raise ValueError('Run identity mismatch')
            if journal['status']=='complete':return journal['result']
            verify=getattr(self.hooks,'verify_resume_state',None)
            if not verify or verify(epoch,self.run_identity) is not True:
                raise RuntimeError('Host must verify model/optimizer/schedule/RNG/pool state before recovery')
            dump(self._path(epoch,'telemetry'),{k:journal[k] for k in ('epoch','metrics','numerical_anomalies')})
            return self._advance(journal)

    def replace_rejected_decision(self,epoch,decision,*,response_evidence):
        """Consume a separately obtained new model answer, preserving the original.

        Allowed only after the native validator rejects the saved answer and
        before any apply attempt. This does not manufacture/submit a model turn.
        """
        from project_experience import file_evidence
        if file_evidence(response_evidence['path'])['sha256']!=response_evidence['sha256']:
            raise ValueError('New model response evidence changed')
        # Evidence must contain the exact normalized answer, not an unrelated log.
        evidence=json.loads(Path(response_evidence['path']).read_text())
        if evidence.get('decision')!=decision or not evidence.get('request_id'):
            raise ValueError('Response receipt needs request_id and exact decision')
        with self._lock():
            journal=json.loads(self._path(epoch,'journal').read_text())
            if journal['status']=='complete' or 'apply' in journal['stages']:
                raise ValueError('Cannot replace a decision after mutation may have started')
            old=journal['stages'].get('decide',{})
            if old.get('status')!='completed':raise ValueError('Reconcile the original request first')
            validator=getattr(self.hooks,'validate_decision',None)
            if not validator:raise ValueError('Native host validator required for a replacement')
            def validate(value):
                self._validate(value,journal)
            try:validate(old['result'])
            except (ValueError,PermissionError):pass
            else:raise ValueError('A valid completed scientific decision cannot be resampled here')
            validate(decision)
            journal.setdefault('rejected_decisions',[]).append(dict(decision=old['result'],
                replacement_evidence=response_evidence,request_id=evidence['request_id']))
            old['result']=decision
            self._save(journal)

    def _advance(self,journal):
        epoch=journal['epoch'];events=journal['events']
        if journal['numerical_anomalies']:raise FloatingPointError('Nonfinite measurements: restore a sound state in a new branch')
        evaluation=self._stage(journal,'evaluate',lambda:self.hooks.evaluate(epoch)) if events['evaluate'] else None
        if events['evaluate']:
            dump(self._path(epoch,'evaluation'),dict(epoch=epoch,evaluation=evaluation,run_identity=self.run_identity))
        if events['checkpoint']:self._stage(journal,'checkpoint',lambda:self.hooks.save_checkpoint(epoch,evaluation))
        result=dict(events=events,decision=None)
        if events['agent']:
            observations=[]
            for p in sorted(self.output.glob('epoch_*_evaluation.json')):
                row=json.loads(p.read_text())
                if row.get('run_identity')==self.run_identity and row['epoch']<=epoch:observations.append(row)
            settings=defaults()
            context_hook=getattr(self.hooks,'review_context',None)
            dynamic=self._stage(journal,'context',lambda:context_hook(epoch,evaluation)) if context_hook else {}
            if not isinstance(dynamic,dict):raise ValueError('Host review context must be an object')
            if self.binding['production'] and (not isinstance(dynamic.get('control_baseline'),dict)
                    or not isinstance(dynamic.get('control_evidence'),dict)):
                raise ValueError('Production context requires current control_baseline and host-issued control_evidence')
            packet=dict(epoch=epoch,evaluation=evaluation,current_metrics=journal['metrics'],
                        cadence=self.cadence.policy(),recent_observations=observations[-2:],
                        run_identity=self.run_identity,evidence_cutoff_epoch=epoch,
                        scientific_context=self.context,model=settings['model'],reasoning_effort=settings['reasoning_effort'],
                        current_context=dynamic,
                        control_change_contract='Return the complete controls vector and change_basis for every actual delta. Each basis binds control/before/after, support_status, existing evidence_ids, mechanism, why_now, why_this_magnitude, expected_response, alternative and reversal_condition. Unsupported ideas remain proposals, not control mutations. KEEP uses unchanged controls and an empty change_basis.',
                        automatic_negative_contract='Optional explicit plan in this same decision; current catalog identity/support must be supplied by the host before deciding.')
            if 'expert_brief' in dynamic:
                brief=dynamic['expert_brief']
                if (not isinstance(brief,dict) or brief.get('format')!='expert_review_v1'
                        or brief.get('epoch')!=epoch or not brief.get('history')
                        or brief.get('controls')!=dynamic.get('control_baseline')):
                    raise ValueError('Compact brief must bind this epoch, history and actual baseline')
                packet={k:packet[k] for k in ('model','reasoning_effort','control_change_contract')}
                packet.update(expert_brief=brief,run_identity=self.run_identity,
                    delivery='Host must send an actual verified curve attachment or declare text-only; no assumed conversational memory')
            if len(json.dumps(packet,allow_nan=False))>24000:raise ValueError('Supply compact facts/knowledge; full logs stay in artifacts')
            decision=self._stage(journal,'decide',lambda:self.hooks.decide(packet))
            result=dict(status='received',events=events,decision=decision,epoch=epoch,
                        model=settings['model'],reasoning_effort=settings['reasoning_effort'])
            receipt=self._path(epoch,'decision');dump(receipt,result)
            try:
                self._validate(decision,journal)
                self._stage(journal,'apply',lambda:self.hooks.apply_decision(decision))
                if events['automatic_refresh'] and decision['continue_training']:
                    generation=self.cadence.generation(epoch,steps_per_epoch=self.steps_per_epoch,maximum_epoch=self.maximum_epoch)
                    self._stage(journal,'refresh',lambda:self.hooks.refresh_automatic(decision.get('automatic_negative_plan'),generation))
                result['status']='applied'
                commit=getattr(self.hooks,'commit_review',None)
                if commit:self._stage(journal,'commit',lambda:commit(epoch,result))
            except Exception as exc:
                result.update(status='requires_reconciliation',error=dict(type=type(exc).__name__,message=str(exc)))
                dump(receipt,result);raise
            dump(receipt,result)
        journal.update(status='complete',result=result);self._save(journal)
        return result
