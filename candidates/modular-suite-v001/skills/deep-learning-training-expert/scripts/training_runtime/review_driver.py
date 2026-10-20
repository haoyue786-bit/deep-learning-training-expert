"""Small callback bridge: observation every five epochs, one decision every ten."""
from __future__ import annotations

import json
from pathlib import Path

from .cadence import Cadence, numerical_anomalies
from .decision_packet import defaults
from .trial_bench import dump


class ReviewDriver:
    """Integrate at an epoch boundary; the host continues to own training/state.

    Hooks: evaluate(epoch), save_checkpoint(epoch, evaluation),
    decide(packet), apply_decision(decision), refresh_automatic(plan, generation).
    A scientific decision may include an automatic-negative plan in that same
    answer. This bridge never makes a second model call for an algorithm module.
    """

    def __init__(self, *, output: Path, maximum_epoch: int, steps_per_epoch: int,
                 hooks, cadence=None):
        self.output=Path(output);self.output.mkdir(parents=True,exist_ok=True)
        self.maximum_epoch=maximum_epoch;self.steps_per_epoch=steps_per_epoch
        self.hooks=hooks;self.cadence=cadence or Cadence.from_job({'review_cadence':defaults()['review_cadence']})
        if isinstance(maximum_epoch,bool) or not isinstance(maximum_epoch,int) or maximum_epoch<=0:
            raise ValueError('A positive training epoch cap required')
        self.cadence.generation(0,steps_per_epoch=steps_per_epoch,maximum_epoch=maximum_epoch)

    def after_epoch(self, epoch: int, metrics: dict, *, emergency=False):
        events=self.cadence.events(epoch,maximum_epoch=self.maximum_epoch,emergency=emergency)
        issues=numerical_anomalies(metrics)
        def safe(value):
            if isinstance(value,dict):return {k:safe(v) for k,v in value.items()}
            if isinstance(value,(list,tuple)):return [safe(v) for v in value]
            if isinstance(value,(float,int)):
                import math
                if not math.isfinite(value):return str(value)
            return value
        row=dict(epoch=epoch,metrics=safe(metrics),numerical_anomalies=issues)
        path=self.output/f'epoch_{epoch:04d}_telemetry.json'
        if path.exists():
            if json.loads(path.read_text())!=row:raise ValueError('Completed epoch telemetry cannot be overwritten')
            # Exact retries must be reconciled by the caller; never repeat a model turn here.
            raise ValueError('Epoch already observed; reconcile its saved decision instead of replaying')
        dump(path,row)
        if issues:raise FloatingPointError('Nonfinite measured values recorded; stop before further optimizer work')
        evaluation=self.hooks.evaluate(epoch) if events['evaluate'] else None
        if events['evaluate']:
            dump(self.output/f'epoch_{epoch:04d}_evaluation.json',dict(epoch=epoch,evaluation=evaluation))
        if events['checkpoint']:self.hooks.save_checkpoint(epoch,evaluation)
        if not events['agent']:return dict(events=events,decision=None)
        settings=defaults()
        observations=[json.loads(p.read_text()) for p in sorted(self.output.glob('epoch_*_evaluation.json'))[-2:]]
        packet=dict(epoch=epoch,evaluation=evaluation,current_metrics=metrics,cadence=self.cadence.policy(),
                    recent_observations=observations,
                    model=settings['model'],reasoning_effort=settings['reasoning_effort'],
                    automatic_negative_contract='Return an optional explicit automatic_negative_plan in this same scientific decision; no hidden extra model turn.')
        if len(json.dumps(packet,allow_nan=False))>24000:
            raise ValueError('Supply compact evaluation/epoch summaries; full predictions and logs stay in artifacts')
        decision=self.hooks.decide(packet)
        result=dict(status='received',events=events,decision=decision,epoch=epoch,
                    model=settings['model'],reasoning_effort=settings['reasoning_effort'])
        receipt=self.output/f'epoch_{epoch:04d}_decision.json'
        dump(receipt,result)  # Commit a completed model turn before any host mutation.
        try:
            if not isinstance(decision,dict) or not isinstance(decision.get('continue_training'),bool):
                raise ValueError('Explicit continue_training decision required')
            # Host action validation remains authoritative, including STOP semantics.
            self.hooks.apply_decision(decision)
            if events['automatic_refresh'] and decision['continue_training']:
                generation=self.cadence.generation(epoch,steps_per_epoch=self.steps_per_epoch,maximum_epoch=self.maximum_epoch)
                self.hooks.refresh_automatic(decision.get('automatic_negative_plan'),generation)
            result['status']='applied'
        except Exception as exc:
            result.update(status='requires_reconciliation',error=dict(type=type(exc).__name__,message=str(exc)))
            dump(receipt,result)
            raise
        dump(receipt,result)
        return result
