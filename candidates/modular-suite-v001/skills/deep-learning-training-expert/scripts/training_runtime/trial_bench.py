"""Small, isolated paired trials using the host's existing model/optimizer adapter."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
from pathlib import Path
import time

from .decision_packet import build as decision_packet
from .registry import get as capability, resolve
from .validation import compare_feedback, evaluate_points, screen_candidates


def state_digest(value, *, legacy_numpy=False) -> str:
    """Semantic tensor/container identity, independent of pickle object addresses."""
    import torch
    import numpy as np
    h = hashlib.sha256()
    def add(x):
        if torch.is_tensor(x):
            a = x.detach().cpu().contiguous()
            h.update(str((str(a.dtype), tuple(a.shape))).encode()); h.update(a.numpy().tobytes())
        elif isinstance(x, np.ndarray) and not legacy_numpy:
            h.update(str((str(x.dtype), tuple(x.shape))).encode()); h.update(x.tobytes(order='C'))
        elif isinstance(x, dict):
            for k in sorted(x, key=str): h.update(repr(k).encode()); add(x[k])
        elif isinstance(x, (tuple, list)):
            h.update(type(x).__name__.encode())
            for child in x: add(child)
        else: h.update(repr(x).encode())
    add(value)
    return h.hexdigest()


def dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    import os
    temporary=path.with_name(path.name+f'.tmp.{os.getpid()}')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


class TrialBench:
    """The adapter owns source/unlabelled loss computation; labels enter evaluate only.

    Required adapter API: snapshot(), restore(state), evaluate_points(),
    training_step(module, step), source_safety(), dataset, class_names,
    focus_class, confusing_class, available_variants and protected_artifacts().
    """

    def __init__(self, store, config: dict):
        self.store = store
        self.config = config
        self.root = Path(config["run_dir"]).resolve()
        try: relative = self.root.relative_to(store.project_root)
        except ValueError: raise PermissionError("Trial outputs must stay in the explicitly authorized project")
        if not relative.parts: raise PermissionError("Trial artifacts need their own directory")
        self.steps = config.get("trial_steps", 2)
        self.max_trials = config.get("maximum_trials", 2)
        self.max_uses = config.get("maximum_validation_uses", 6)
        self.max_seconds = config.get("maximum_seconds_per_arm", 300)
        for key, value, limit in [("trial_steps", self.steps, 20), ("maximum_trials", self.max_trials, 2),
                                  ("maximum_validation_uses", self.max_uses, 20),
                                  ("maximum_seconds_per_arm", self.max_seconds, 900)]:
            if isinstance(value, bool) or not isinstance(value, int) or not 0 < value <= limit:
                raise ValueError(f"Finite low-cost trial budget required: {key}")
        self.root.mkdir(parents=True, exist_ok=True)
        marker = self.root / ".algorithm_trial_bench.json"
        identity = dict(project_id=store.project_id, format="algorithm_trial_bench_v1")
        if marker.exists():
            actual=json.loads(marker.read_text())
            if {k:actual[k] for k in identity} != identity: raise PermissionError("Trial directory belongs to another project")
            self.digest_version=actual.get('state_digest_version',1)
            if self.digest_version not in (1,2):raise ValueError('Unknown trial state identity version')
        elif any(self.root.iterdir()):
            raise PermissionError("Refusing to write into a nonempty unregistered run directory")
        else:
            self.digest_version=2
            dump(marker,dict(identity,state_digest_version=2))

    def _digest(self, state):
        return state_digest(state,legacy_numpy=self.digest_version==1)

    def _observe(self, adapter, *, usage_id: str, checkpoint_id: str, purpose: str, destination: Path):
        checkpoint = dict(identity=checkpoint_id, sha256=self._digest(adapter.snapshot()))
        self.store.reserve_validation(usage_id=usage_id, dataset=adapter.dataset, checkpoint=checkpoint,
                                      purpose=purpose, maximum_uses=self.max_uses)
        if destination.exists():
            old = json.loads(destination.read_text())
            if old["checkpoint"] != checkpoint or old["dataset"] != adapter.dataset:
                raise ValueError("Completed feedback cannot be rebound to a different model or label version")
            self.store.finish_validation(usage_id,destination)
            return old
        try:
            feedback = evaluate_points(adapter.evaluate_points(), class_names=adapter.class_names,
                                       dataset=adapter.dataset, checkpoint=checkpoint,
                                       focus_class=adapter.focus_class, confusing_class=adapter.confusing_class)
            dump(destination, feedback)
        except Exception as exc:
            error_path = destination.with_suffix(".failed.json")
            dump(error_path, dict(status="failed", error_type=type(exc).__name__, message=str(exc)))
            self.store.finish_validation(usage_id, error_path, status="failed")
            raise
        self.store.finish_validation(usage_id, destination)
        return feedback

    def prepare(self, adapter) -> dict:
        if adapter.dataset.get("role") != "development":
            raise PermissionError("Candidate screening needs explicitly designated development feedback")
        original = copy.deepcopy(adapter.snapshot()); initial_hash = self._digest(original)
        try:
            feedback = self._observe(adapter, usage_id=self.root.name+":initial", checkpoint_id=self.root.name+":initial",
                                     purpose="baseline_diagnosis", destination=self.root/"initial_feedback.json")
            screen = screen_candidates(feedback)
            experience=[]
            tags=[c['module_id'] for c in screen['candidates']]
            context=dict(getattr(adapter,'experience_context',{}))
            context.update(self.config.get('experience_query', {}))
            relevant=self.store.retrieve(tags=tags,context=context,limit=3,
                                         cutoff=self.config.get('evidence_cutoff'),
                                         excluded_case_ids=self.config.get('excluded_case_ids', []),
                                         excluded_units=self.config.get('excluded_units', []))
            for case in relevant[:3]:
                dataset=case.get('conditions',{}).get('dataset',{})
                result=case.get('result',{})
                experience.append(dict(case_id=case['case_id'],outcome=case['outcome'],action=case['action'],
                    applicability=dict(dataset_id=dataset.get('dataset_id'),label_version=dataset.get('label_version'),
                                       evaluation_role=case['evaluation_role']),
                    measured_comparison=result.get('comparison'),hypotheses=case['hypotheses'][:2],
                    lesson=result.get('lesson'),counterexamples=result.get('counterexamples',[])[:2],
                    matching=case.get('retrieval_match'),
                    evidence=case['evidence'][:2],transfer_claim='Conditional context only; inspect evidence for task matching'))
            packet = decision_packet(feedback, available_variants=adapter.available_variants,experience_summaries=experience)
            dump(self.root/"candidate_screen.json", screen)
            dump(self.root/"decision_packet.json", packet)
            dump(self.root/"initial_identity.json", dict(state_sha256=initial_hash, dataset=adapter.dataset,
                                                        source_artifacts=adapter.protected_artifacts()))
        finally:
            adapter.restore(original)
        if self._digest(adapter.snapshot()) != initial_hash:
            raise RuntimeError("Read-only development observation changed the starting state")
        from project_experience import file_evidence
        self.store.record(dict(case_id=self.root.name+':initial_observation',project_id=self.store.project_id,
            action=dict(operation='fixed-decoder development diagnosis'),
            result=dict(focus_metrics=feedback['focus_metrics'],measured_pair_errors=screen['measured_pair_errors'],
                        candidates=screen['candidates']),outcome='observation',
            evidence=[file_evidence(self.root/'initial_feedback.json')],
            conditions=dict(dataset=adapter.dataset,source_start=initial_hash),
            hypotheses=[c['hypothesis'] for c in screen['candidates']],
            cost=dict(training_updates=0,model_calls=0,feedback_uses='one reserved or exact cached receipt'),
            evaluation_role='development',tags=['baseline_diagnosis']))
        return packet

    def run_pair(self, adapter, *, module_id: str, variant: str, controls=None, strength=0.01,
                 scientific_plan=None) -> dict:
        from .cadence import numerical_anomalies
        if not isinstance(strength, (float, int)) or isinstance(strength, bool) or not 0 <= strength <= 1:
            raise ValueError("A bounded separate mechanism coefficient is required")
        cap = capability(module_id, variant)
        name = f"{module_id}:{variant}"
        if name not in adapter.available_variants or cap.implementation is None:
            raise PermissionError("Host has not implemented/authorized this variant")
        resolve(module_id,variant,controls=controls,data_role=cap.input_roles[0],
                authorized_variants=adapter.available_variants)
        if not (self.root/"initial_feedback.json").exists(): self.prepare(adapter)
        original = copy.deepcopy(adapter.snapshot()); identity = self._digest(original)
        expected = json.loads((self.root/"initial_identity.json").read_text())
        if expected["state_sha256"] != identity or expected["dataset"] != adapter.dataset:
            raise ValueError("Trial start/data differ from the diagnosed baseline")
        # Error-count suggestions never authorize or restrict a scientific choice.
        # One plan is required for every variant, including suggested variants.
        plan=copy.deepcopy(scientific_plan)
        required=('hypothesis','alternative','why_this_action','expected_response','reversal_condition')
        if not isinstance(plan,dict) or any(not isinstance(plan.get(k),str) or not plan[k].strip() for k in required):
            raise ValueError('Supply a scientific plan with hypothesis, alternative, action rationale and response/reversal conditions')
        if plan.get('initial_state_sha256')!=identity or plan.get('dataset')!=adapter.dataset:
            raise ValueError('Scientific plan must bind the diagnosed state and development dataset')
        if plan.get('variant')!=name or plan.get('controls',{})!=(controls or {}) or plan.get('strength')!=strength:
            raise ValueError('Scientific plan does not match the requested treatment')
        if plan.get('steps_per_arm')!=self.steps:
            raise ValueError('Scientific plan must declare the actual response horizon')
        from project_experience import file_evidence
        if not isinstance(plan.get('evidence'),list) or not plan['evidence']:
            raise ValueError('Scientific plan needs actual evidence pointers')
        for item in plan['evidence']:
            if file_evidence(item['path'])['sha256']!=item['sha256']:
                raise ValueError('Scientific plan evidence changed')
        with (self.root/'.trial_admission.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            existing=list((self.root/'trials').glob('trial_*'))
            if len(existing)>=self.max_trials:
                raise PermissionError('Declared trial budget exhausted; no automatic candidate/parameter sweep')
            trial_id=f'trial_{len(existing)+1:02d}'
            root=self.root/'trials'/trial_id;root.mkdir(parents=True,exist_ok=False)
        module = dict(module_id=module_id, variant=variant, controls=controls or {}, strength=strength)
        protocol = dict(project_id=self.store.project_id, trial_id=trial_id, module=module, steps=self.steps,
                        scientific_plan=plan,
                        initial_state_sha256=identity, dataset=adapter.dataset,
                        endpoint="after the declared number of matched optimizer updates; never best-of-checkpoints",
                        validation_labels_in_loss=False, threshold_search=False, production_adoption=False,
                        maximum_seconds_per_arm=self.max_seconds, source_artifacts=adapter.protected_artifacts())
        dump(root/"protocol.json", protocol)
        started = time.monotonic(); results = {}; status = "incomplete"; error = None
        attempted=completed=0;partial={}
        try:
            for arm in ("reference", "treatment"):
                adapter.restore(copy.deepcopy(original))
                if self._digest(adapter.snapshot()) != identity: raise RuntimeError("Matched-arm restore failed")
                arm_start = time.monotonic(); history = []
                partial[arm]=history
                for step in range(self.steps):
                    if time.monotonic()-arm_start > self.max_seconds:
                        raise TimeoutError("Bounded trial arm exceeded its prospectively declared wall budget")
                    attempted+=1
                    metric = adapter.training_step(None if arm == "reference" else module, step)
                    completed+=1
                    issues=numerical_anomalies(metric)
                    history.append(dict(step=step+1,numerical_anomalies=issues) if issues else metric)
                    dump(root/f"{arm}_updates.json",history)
                    if issues: raise FloatingPointError("Nonfinite trial update; no scaling up")
                dump(root/f"{arm}_updates.json", history)
                feedback = self._observe(adapter, usage_id=self.root.name+":"+trial_id+":"+arm,
                                         checkpoint_id=self.root.name+":"+trial_id+":"+arm,
                                         purpose="trial_evaluation", destination=root/f"{arm}_feedback.json")
                results[arm] = dict(feedback=feedback, source_safety=adapter.source_safety(), updates=history,
                                    seconds=round(time.monotonic()-arm_start, 3))
                if time.monotonic()-arm_start>self.max_seconds:
                    raise TimeoutError('Trial arm exceeded wall budget; record endpoint without further work')
            delta = compare_feedback(results["reference"]["feedback"], results["treatment"]["feedback"])
            measured = [r.get("module_effect", {}) for r in results["treatment"]["updates"]]
            executed = any(r.get("supported", False) and r.get("nonzero_effect", False) for r in measured)
            if not executed:
                outcome = "null"; continuation = "No supported nonzero mechanism effect; do not scale up from this trial."
            elif delta["focus_f1_delta"] is not None and delta["focus_f1_delta"] > 0:
                outcome = "positive"; continuation = "Promising on this small development panel; expert must review source safety and spatial support before one further bounded confirmation."
            elif delta["focus_f1_delta"] is not None and delta["focus_f1_delta"] < 0:
                outcome = "adverse"; continuation = "Measured development regression; do not expand this treatment without a revised supported diagnosis."
            else:
                outcome = "null"; continuation = "No measured fixed-decoder F1 gain at this endpoint; avoid expanding an unproductive search."
            status = "completed"
            result = dict(protocol=protocol, status=status, outcome=outcome, comparison=delta,
                          mechanism_executed=executed, results=results, continuation=continuation,
                          independent_test_performed=False, quality_scope="tiny fixed development panel, not country-wide efficacy")
        except Exception as exc:
            status = "failed"; error = dict(type=type(exc).__name__, message=str(exc))
            result = dict(protocol=protocol, status=status, outcome="failed", results=results, error=error,
                          continuation="Do not repeat automatically; reconcile the failure before another trial.")
        finally:
            restore_error=None
            try:
                adapter.restore(original)
                restored=self._digest(adapter.snapshot())==identity
            except Exception as exc:
                restored=False;restore_error=str(exc)
        try:
            unchanged=adapter.protected_artifacts()==protocol['source_artifacts']
        except Exception as exc:
            unchanged=False;restore_error=str(exc)
        if not restored or not unchanged:
            error=dict(type='ReferenceIntegrityError',message=restore_error or 'Reference state or protected input changed')
            result.update(status='failed',outcome='failed',error=error,
                          continuation='Stop; restore/reference integrity must be reconciled by the host.')
        result.update(reference_restored=restored, source_artifacts_unchanged=unchanged,
                      update_cost=dict(attempted_steps=attempted,completed_steps=completed,
                                       internal_optimizer_step_counts='reported in per-step adapter metrics'),
                      total_seconds=round(time.monotonic()-started, 3), model_calls=0)
        result_path = root/"result.json"; dump(result_path, result)
        from project_experience import file_evidence
        self.store.record(dict(case_id=self.root.name+":"+trial_id, project_id=self.store.project_id,
                               action=module, result={k:v for k,v in result.items() if k != "results"}, outcome=result["outcome"],
                               evidence=[file_evidence(result_path), file_evidence(root/"protocol.json")],
                               conditions=dict(dataset=adapter.dataset, source_start=identity, matched_updates=self.steps),
                               hypotheses=[plan['hypothesis'],plan['alternative']],
                               cost=dict(attempted_steps=attempted,completed_steps=completed,
                                         seconds=result["total_seconds"],model_calls=0),
                               evaluation_role="development", tags=[module_id, variant, "paired_trial"]))
        if error is not None: raise RuntimeError(f"Trial failed and was recorded: {error}")
        return result
