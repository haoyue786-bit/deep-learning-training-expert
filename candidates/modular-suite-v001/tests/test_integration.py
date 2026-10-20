"""Host-boundary tests; fixtures are not scientific efficacy evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

SKILLS=Path(__file__).resolve().parents[1]/'skills'
sys.path.insert(0,str(SKILLS/'deep-learning-training-expert/scripts'))
sys.path.insert(0,str(SKILLS/'training-experience-update/scripts'))
from project_experience import ProjectStore
from training_runtime import registry
from training_runtime.cadence import Cadence
from training_runtime.review_driver import ReviewDriver
from training_runtime.trial_bench import TrialBench,state_digest,dump

spec=importlib.util.spec_from_file_location('cadence_adapter',SKILLS/'deep-learning-training-expert/task_profiles/cashew_fada/automatic_cadence_adapter.py')
auto=importlib.util.module_from_spec(spec);spec.loader.exec_module(auto)


class Interfaces(unittest.TestCase):
    def test_every_advertised_kernel_uses_the_same_invocation_boundary(self):
        q=torch.softmax(torch.randn(1,4,2,2),1)
        labels=torch.tensor([[[1,2],[1,2]]])
        cases=[('M01','adversarial_path_balance',dict(local=torch.tensor(2.),global_loss=torch.tensor(3.)),{},'unlabelled_training'),
               ('M02','class_pixel_priority',dict(knowledge=q),{},'unlabelled_training'),
               ('M03','pair_ambiguity_reliability',dict(knowledge=q,class_pair=(2,1)),{},'unlabelled_training'),
               ('M04','source_pair_margin',dict(features=torch.randn(1,8,2,2,requires_grad=True),labels=labels,class_pair=(2,1)),{},'source_training'),
               ('M08','source_hard_negative',dict(logits=torch.randn(1,4,2,2,requires_grad=True),labels=labels,focus_class=2,negative_class=1),{},'source_training')]
        allowed=[f'{a}:{b}' for a,b,*_ in cases]
        for mid,variant,inputs,controls,role in cases:
            result=registry.invoke(mid,variant,inputs=inputs,controls=controls,data_role=role,authorized_variants=allowed)
            self.assertTrue(torch.isfinite(result[0] if isinstance(result,tuple) else result).all())
        priority,_=registry.invoke('M02','class_pixel_priority',inputs=dict(knowledge=q),data_role='unlabelled_training',authorized_variants=allowed)
        torch.testing.assert_close(priority,torch.ones_like(priority))
        with self.assertRaises(ValueError):
            registry.invoke('M02','class_pixel_priority',inputs=dict(knowledge=q),controls={'multipliers':[0,0,0,0]},data_role='unlabelled_training',authorized_variants=allowed)

    def test_state_digest_detects_hidden_numpy_mutation(self):
        a=np.arange(2000);b=a.copy();b[1000]=-1
        self.assertNotEqual(state_digest(a),state_digest(b))


class Hooks:
    def __init__(self,fail=False):self.evaluated=[];self.saved=[];self.decisions=[];self.refreshes=[];self.fail=fail
    def evaluate(self,e):self.evaluated.append(e);return dict(f1=.5,evaluation_role='development',epoch=e)
    def save_checkpoint(self,e,evaluation):self.saved.append(e)
    def decide(self,packet):self.decisions.append(packet);return dict(continue_training=True,automatic_negative_plan={'action':'ABSTAIN'})
    def apply_decision(self,decision):
        if self.fail:raise RuntimeError('host action failed')
    def refresh_automatic(self,plan,generation):self.refreshes.append((plan,generation))


class ReviewTests(unittest.TestCase):
    def test_five_epoch_observations_ten_epoch_decisions_and_no_extra_pool_turn(self):
        with tempfile.TemporaryDirectory() as d:
            hooks=Hooks();driver=ReviewDriver(output=Path(d),maximum_epoch=23,steps_per_epoch=7,hooks=hooks)
            for e in range(1,24):driver.after_epoch(e,{'loss':1/e})
            self.assertEqual(hooks.evaluated,[5,10,15,20,23]);self.assertEqual(hooks.saved,hooks.evaluated)
            self.assertEqual([p['epoch'] for p in hooks.decisions],[10,20,23])
            self.assertEqual([o['epoch'] for o in hooks.decisions[0]['recent_observations']],[5,10])
            self.assertEqual([x[1]['expiry_epoch'] for x in hooks.refreshes],[20,23])
            self.assertEqual(hooks.decisions[0]['model'],'gpt-6.1-sol');self.assertEqual(hooks.decisions[0]['reasoning_effort'],'medium')
            with self.assertRaises(ValueError):driver.after_epoch(10,{'loss':.1})
            self.assertEqual(len(hooks.decisions),3)

    def test_nonfinite_is_recorded_and_halts_before_any_decision(self):
        with tempfile.TemporaryDirectory() as d:
            hooks=Hooks();driver=ReviewDriver(output=Path(d),maximum_epoch=20,steps_per_epoch=7,hooks=hooks)
            with self.assertRaises(FloatingPointError):driver.after_epoch(3,{'loss':float('nan')})
            self.assertEqual(hooks.decisions,[])
            self.assertEqual(json.loads((Path(d)/'epoch_0003_telemetry.json').read_text())['numerical_anomalies'],['loss'])

    def test_received_decision_survives_host_failure_without_replay(self):
        with tempfile.TemporaryDirectory() as d:
            hooks=Hooks(fail=True);driver=ReviewDriver(output=Path(d),maximum_epoch=20,steps_per_epoch=7,hooks=hooks)
            with self.assertRaises(RuntimeError):driver.after_epoch(10,{'loss':1})
            receipt=json.loads((Path(d)/'epoch_0010_decision.json').read_text())
            self.assertEqual(receipt['status'],'requires_reconciliation')
            self.assertTrue(receipt['decision']['continue_training'])
            with self.assertRaises(ValueError):driver.after_epoch(10,{'loss':1})
            self.assertEqual(len(hooks.decisions),1)


class FakePool:
    def __init__(self,path,expected,weight):
        if auto.sha(path)!=expected:raise ValueError('wrong manifest')
        self.manifest=json.loads(Path(path).read_text());self.rows=self.manifest['selected'];self.weight=weight
    def summary(self):return dict(effective_weight=self.active_weight)


def plan():return dict(action='ACTIVATE',selected_count=11,automatic_weight=.001,
    ranking_weights=dict(source_cashew_distance=.25,p2_increase=.25,regional_prior=.5),
    rationale='Measured false positives justify testing',expected_response='fewer false positives with recall retained',
    reversal_condition='development recall or source safety worsens')


class FakeController:
    def __init__(self,out):self.out=Path(out);self.out.mkdir(exist_ok=True);self.hidden_calls=0;self.refresh_count=0
    def select(self,*args):self.hidden_calls+=1;raise AssertionError('second model call forbidden')
    def refresh(self,epoch,checkpoint,review,config):
        self.refresh_count+=1;d=self.out/('epoch_%03d'%epoch);d.mkdir(exist_ok=True)
        decision=self.select(epoch,d,list(range(20)),{},review,config)
        manifest=dict(generation_epoch=epoch,selected=list(range(11)),schedule_seed=16,
                      policy={'maximum_age_epochs':5},requested_patch_count=decision['selected_count'],
                      agent_selected_weight=decision['automatic_weight'],ranking_weights=decision['ranking_weights'],
                      selection_decision=str(d/'selection_decision.json'),selection_decision_sha256=auto.sha(d/'selection_decision.json'),
                      exposure_horizon_updates=128)
        auto.dump(d/'manifest.json',manifest)
        config['_automatic_negative_state']=dict(generation_epoch=epoch,manifest=str(d/'manifest.json'),
              manifest_sha256=auto.sha(d/'manifest.json'),cursor=0,weight=.001)
        auto.dump(self.out/'state.json',config['_automatic_negative_state'])
    def for_epoch(self,config):return 'legacy executor'
    def after_epoch(self,config,metrics):
        config['_automatic_negative_state']['cursor']=metrics['automatic_negative']['generation_end_cursor']
        auto.dump(self.out/'state.json',config['_automatic_negative_state'])


class AutomaticTests(unittest.TestCase):
    def test_one_occurrence_spread_to_next_decision_and_resume_does_not_renew(self):
        with tempfile.TemporaryDirectory() as d:
            controller=FakeController(d);adapter=auto.AutomaticCadenceAdapter(controller,FakePool);config={}
            generation=Cadence().generation(10,steps_per_epoch=7,maximum_epoch=200)
            state=adapter.refresh(plan(),generation,checkpoint='unused',review={},config=config)
            original_hash=auto.sha(Path(state['manifest']).parent/'manifest.json')
            seen=[]
            for age in range(10):
                config['_negative_epoch_start']=10+age
                pool=adapter.for_epoch(config);pool.begin_epoch(7,config,16)
                seen.extend(i for indices in pool.schedule.values() for i in indices)
                adapter.after_epoch(config,{'automatic_negative':pool.summary()})
                if age==3:
                    # Losing in-memory config uses committed exposure, not a renewed pool.
                    resumed={};restored=adapter.refresh(plan(),generation,checkpoint='unused',review={},config=resumed,resume_state_verified=True)
                    self.assertEqual(restored['cursor'],28);config=resumed
            self.assertEqual(sorted(seen),list(range(11)));self.assertEqual(len(seen),len(set(seen)))
            self.assertEqual(controller.hidden_calls,0);self.assertEqual(controller.refresh_count,1)
            self.assertEqual(auto.sha(Path(state['manifest']).parent/'manifest.json'),original_hash)
            config['_negative_epoch_start']=20;pool=adapter.for_epoch(config);pool.begin_epoch(7,config,16)
            self.assertEqual(pool.schedule,{});self.assertEqual(pool.active_weight,0)
            with self.assertRaises(ValueError):adapter.refresh(dict(plan(),automatic_weight=.002),generation,checkpoint='unused',review={},config=config)
            with self.assertRaises(ValueError):adapter.refresh(plan(),dict(generation,exposure_horizon_updates=140),checkpoint='unused',review={},config=config)

    def test_legacy_pool_and_no_plan_preserve_existing_contract(self):
        with tempfile.TemporaryDirectory() as d:
            controller=FakeController(d);adapter=auto.AutomaticCadenceAdapter(controller,FakePool)
            self.assertIsNone(adapter.refresh(None,Cadence().generation(10,steps_per_epoch=7,maximum_epoch=200),checkpoint='',review={},config={}))
            path=Path(d)/'old.json';auto.dump(path,{'generation_epoch':5})
            config={'_automatic_negative_state':{'manifest':str(path),'manifest_sha256':auto.sha(path)}}
            self.assertEqual(adapter.for_epoch(config),'legacy executor');self.assertEqual(controller.hidden_calls,0)


class TinyAdapter:
    class_names=['other','mixed','cashew','built'];focus_class=2;confusing_class=1
    available_variants=['M04:source_pair_margin','M02:class_pixel_priority']
    def __init__(self,fail=None):
        self.state={'weight':torch.tensor(1.),'rng':np.arange(2000)};self.fail=fail;self.starts=[]
        self.dataset={'dataset_id':'fixture-dev','label_version':'v1','sha256':'labels','role':'development','labels_for_gradient':False}
    def snapshot(self):return copy.deepcopy(self.state)
    def restore(self,state):self.state=copy.deepcopy(state)
    def protected_artifacts(self):return []
    def evaluate_points(self):return [dict(point_id=str(i),class_id=a,predicted_class=p) for i,(a,p) in enumerate([(2,1),(2,1),(1,2),(2,2)])]
    def training_step(self,module,step):
        if step==0:self.starts.append(state_digest(self.state))
        self.state['weight']+=1
        if self.fail=='update':raise RuntimeError('fixture interrupted after mutation')
        return dict(loss=float('nan') if self.fail=='nonfinite' else 1.,model_effect=True,
                    module_effect={'supported':True,'nonzero_effect':bool(module)})
    def source_safety(self):return {'f1':.5}


def run_test_trial(bench,adapter,**kwargs):
    bench.prepare(adapter)
    from project_experience import file_evidence
    plan=dict(hypothesis='Fixture mechanism may change the response',alternative='Insufficient exposure',
              why_this_action='Controlled fixture exercises allowed execution',expected_response='Measured feedback',
              reversal_condition='Source or target regression',initial_state_sha256=bench._digest(adapter.snapshot()),
              dataset=adapter.dataset,variant=kwargs['module_id']+':'+kwargs['variant'],
              controls=kwargs.get('controls',{}),strength=kwargs.get('strength',.01),steps_per_arm=bench.steps,
              evidence=[file_evidence(bench.root/'initial_feedback.json')])
    return bench.run_pair(adapter,scientific_plan=plan,**kwargs)


class BenchTests(unittest.TestCase):
    def run_fixture(self,root,fail=None):
        context=dict(project_id='fixture',project_root=str(root),experience_dir=str(root/'experience'),
                     authorization={'write_experience':True,'source':'test'})
        store=ProjectStore(context);bench=TrialBench(store,dict(run_dir=str(root/'bench'),trial_steps=2))
        return store,bench,TinyAdapter(fail)

    def test_real_state_restore_matched_exposure_and_null_experience(self):
        with tempfile.TemporaryDirectory() as d:
            store,bench,adapter=self.run_fixture(Path(d));original=state_digest(adapter.snapshot())
            bench.prepare(adapter)
            # Cached metrics also reconcile a previously committed reservation without a second label use.
            bench.prepare(adapter)
            result=run_test_trial(bench,adapter,module_id='M04',variant='source_pair_margin')
            self.assertEqual(result['outcome'],'null');self.assertEqual(result['update_cost']['completed_steps'],4)
            self.assertEqual(adapter.starts,[original,original]);self.assertEqual(state_digest(adapter.snapshot()),original)
            self.assertEqual(store.retrieve()[0]['outcome'],'null')
            self.assertEqual(result['model_calls'],0);self.assertFalse(result['protocol']['threshold_search'])

    def test_failed_or_nonfinite_updates_restore_and_count_discarded_work(self):
        for failure in ['update','nonfinite']:
            with tempfile.TemporaryDirectory() as d:
                store,bench,adapter=self.run_fixture(Path(d),failure);original=state_digest(adapter.snapshot())
                with self.assertRaises(RuntimeError):run_test_trial(bench,adapter,module_id='M04',variant='source_pair_margin')
                result=json.loads((bench.root/'trials/trial_01/result.json').read_text())
                self.assertEqual(state_digest(adapter.snapshot()),original);self.assertEqual(result['outcome'],'failed')
                self.assertEqual(result['update_cost']['attempted_steps'],1)
                self.assertEqual(result['update_cost']['completed_steps'],0 if failure=='update' else 1)
                self.assertEqual(store.retrieve()[0]['outcome'],'failed')

    def test_shortlist_and_trial_cap_are_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            store,bench,adapter=self.run_fixture(Path(d))
            with self.assertRaises(PermissionError):run_test_trial(bench,adapter,module_id='M03',variant='pair_ambiguity_reliability')
            run_test_trial(bench,adapter,module_id='M04',variant='source_pair_margin')
            run_test_trial(bench,adapter,module_id='M02',variant='class_pixel_priority',controls={'multipliers':[1,1,1.5,1]})
            with self.assertRaises(PermissionError):run_test_trial(bench,adapter,module_id='M04',variant='source_pair_margin')


if __name__=='__main__':unittest.main()
