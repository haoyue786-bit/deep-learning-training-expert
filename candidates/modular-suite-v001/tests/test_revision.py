"""Recovery, scientific freedom and conditional-memory invariants; no efficacy claims."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_integration import Hooks, ReviewDriver, run_test_trial, auto, FakeController, FakePool, plan, Cadence
import test_integration as integration
import test_runtime as runtime


class RecoverableHooks(Hooks):
    def __init__(self):
        super().__init__()
        self.effects=[];self.receipts={};self.known={};self.fail_stage='decide'
    def verify_resume_state(self,epoch,run_identity):return True
    def begin_operation(self,stage,operation_id):self.operation_id=operation_id
    def reconcile_operation(self,stage,operation_id):
        return self.known.get(stage,{'status':'unknown','evidence':'fixture remote status unresolved'})
    def decide(self,packet):
        if self.fail_stage=='decide':
            self.fail_stage=None
            self.known['decide']={'status':'not_started','evidence':'fixture server confirms request not accepted'}
            raise TimeoutError('fixture not submitted')
        return super().decide(packet)
    def apply_decision(self,decision):
        self.effects.append(decision)
        self.known['apply']={'status':'completed','result':None,'evidence':'fixture committed action receipt'}
        if self.fail_stage=='apply':
            self.fail_stage=None
            raise RuntimeError('fixture lost connection after committed mutation')


class RecoveryTests(unittest.TestCase):
    def driver(self,d,h,**kwargs):
        return ReviewDriver(output=Path(d),maximum_epoch=200,steps_per_epoch=7,hooks=h,
                            run_identity={'run':'a','branch':'main'},**kwargs)

    def test_known_unsubmitted_timeout_resumes_without_repeating_evaluation(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();driver=self.driver(d,h)
            with self.assertRaises(TimeoutError):driver.after_epoch(10,{'loss':1})
            with self.assertRaises(RuntimeError):driver.after_epoch(11,{'loss':1})
            self.assertEqual(driver.resume(10)['status'],'applied')
            self.assertEqual(h.evaluated,[10]);self.assertEqual(len(h.effects),1)
            driver.resume(10);self.assertEqual(len(h.effects),1)

    def test_committed_mutation_is_reconciled_without_reapplying_or_redeciding(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();h.fail_stage='apply';driver=self.driver(d,h)
            with self.assertRaises(RuntimeError):driver.after_epoch(10,{'loss':1})
            driver=self.driver(d,h)  # Simulated process restart, same binding.
            driver.resume(10)
            self.assertEqual(len(h.effects),1);self.assertEqual(len(h.decisions),1)
            self.assertEqual(len(h.refreshes),1)

    def test_unknown_remote_outcome_never_auto_retries(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();driver=self.driver(d,h)
            with self.assertRaises(TimeoutError):driver.after_epoch(10,{'loss':1})
            h.known={}
            with self.assertRaises(RuntimeError):driver.resume(10)
            self.assertEqual(len(h.decisions),0);self.assertEqual(len(h.effects),0)

    def test_future_and_foreign_observations_excluded_and_run_binding_enforced(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();h.fail_stage=None;driver=self.driver(d,h)
            Path(d,'epoch_0020_evaluation.json').write_text(json.dumps(
                dict(epoch=20,evaluation={},run_identity=driver.run_identity)))
            Path(d,'epoch_0005_evaluation.json').write_text(json.dumps(
                dict(epoch=5,evaluation={},run_identity={'run':'other'})))
            driver.after_epoch(10,{'loss':1})
            self.assertEqual([r['epoch'] for r in h.decisions[0]['recent_observations']],[10])
            with self.assertRaises(ValueError):
                ReviewDriver(output=Path(d),maximum_epoch=200,steps_per_epoch=7,hooks=h,
                             run_identity={'run':'a','branch':'new'})

    def test_incomplete_old_directory_is_not_silently_adopted(self):
        with tempfile.TemporaryDirectory() as d:
            Path(d,'epoch_0020_evaluation.json').write_text('{}')
            with self.assertRaises(ValueError):self.driver(d,Hooks())

    def test_retry_budget_does_not_allow_infinite_not_started_responses(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();driver=self.driver(d,h,maximum_stage_attempts=1)
            with self.assertRaises(TimeoutError):driver.after_epoch(10,{'loss':1})
            with self.assertRaisesRegex(RuntimeError,'budget exhausted'):driver.resume(10)

    def test_production_requires_real_integration_contract(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'integration hooks'):
                self.driver(d,Hooks(),production=True)

    def test_dynamic_context_is_saved_and_not_recomputed_after_timeout(self):
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();calls=[]
            def context(epoch,evaluation):
                calls.append(epoch)
                return {'controls':{'lambda_adv':.002},'source_f1':.8}
            h.review_context=context;driver=self.driver(d,h)
            with self.assertRaises(TimeoutError):driver.after_epoch(10,{'loss':1})
            driver.resume(10)
            self.assertEqual(calls,[10])
            self.assertEqual(h.decisions[0]['current_context']['controls']['lambda_adv'],.002)

    def test_invalid_answer_replacement_preserves_original_and_requires_validator(self):
        from project_experience import file_evidence
        with tempfile.TemporaryDirectory() as d:
            h=RecoverableHooks();h.fail_stage=None
            h.decide=lambda packet: {'continue_training':True,'action':'INVALID'}
            def validate(decision):
                if decision.get('action')!='KEEP':raise ValueError('native action rejected')
            h.validate_decision=validate;driver=self.driver(d,h)
            with self.assertRaises(ValueError):driver.after_epoch(10,{'loss':1})
            answer={'continue_training':True,'action':'KEEP'}
            evidence=Path(d)/'corrected_response.json'
            evidence.write_text(json.dumps({'request_id':'fixture-request-2','decision':answer}))
            driver.replace_rejected_decision(10,answer,response_evidence=file_evidence(evidence))
            driver.resume(10)
            journal=json.loads(Path(d,'epoch_0010_journal.json').read_text())
            self.assertEqual(journal['rejected_decisions'][0]['decision']['action'],'INVALID')
            self.assertEqual(len(h.effects),1)

    def test_pool_cursor_divergence_both_directions_and_unverified_missing_state(self):
        with tempfile.TemporaryDirectory() as d:
            controller=FakeController(d);adapter=auto.AutomaticCadenceAdapter(controller,FakePool);config={}
            generation=Cadence().generation(10,steps_per_epoch=7,maximum_epoch=200)
            adapter.refresh(plan(),generation,checkpoint='unused',review={},config=config)
            config['_automatic_negative_state']['cursor']=14
            auto.dump(controller.out/'state.json',config['_automatic_negative_state'])
            for cursor in (7,21):
                stale=copy.deepcopy(config);stale['_automatic_negative_state']['cursor']=cursor
                with self.assertRaises(ValueError):
                    adapter.refresh(plan(),generation,checkpoint='unused',review={},config=stale)
            with self.assertRaises(ValueError):
                adapter.refresh(plan(),generation,checkpoint='unused',review={},config={})


class ScientificChoiceTests(unittest.TestCase):
    def test_authorized_variant_outside_count_shortlist_can_execute(self):
        with tempfile.TemporaryDirectory() as d:
            _,bench,adapter=integration.BenchTests().run_fixture(Path(d))
            adapter.available_variants+=['M01:adversarial_path_balance']
            result=run_test_trial(bench,adapter,module_id='M01',variant='adversarial_path_balance',
                                  controls={'local_scale':.5})
            self.assertEqual(result['status'],'completed')
            self.assertEqual(result['protocol']['scientific_plan']['variant'],'M01:adversarial_path_balance')

    def test_suggested_variant_still_requires_evidence_bound_plan(self):
        with tempfile.TemporaryDirectory() as d:
            _,bench,adapter=integration.BenchTests().run_fixture(Path(d))
            with self.assertRaises(ValueError):
                bench.run_pair(adapter,module_id='M04',variant='source_pair_margin')
            self.assertEqual(list((bench.root/'trials').glob('*')),[])


class ConditionalMemoryTests(unittest.TestCase):
    def setUp(self):
        self.fixture=runtime.ExperienceTests();self.fixture.setUp()
        self.store=self.fixture.store;self.case=self.fixture.case
    def tearDown(self):self.fixture.tearDown()

    def test_context_match_beats_recent_tag_and_retains_contradiction(self):
        for name,source,outcome in [('good','Benin','positive'),('bad','Benin','adverse'),('recent','Ghana','positive')]:
            item=self.case();item.update(case_id=name,outcome=outcome,tags=['lambda'])
            item['conditions']={'source_profile':source,'mechanism':'adversarial_pressure'}
            self.store.record(item)
        selected=self.store.retrieve(tags=['lambda'],context={'source_profile':'Benin'},limit=2)
        self.assertEqual({c['case_id'] for c in selected},{'good','bad'})
        self.assertEqual(self.store.retrieve(context={'mechanism':'unrelated'}),[])
        self.assertEqual(self.store.retrieve(tags=['lambda'],cutoff='2000-01-01T00:00:00Z'),[])

    def test_holdout_case_and_task_unit_exclusions(self):
        item=self.case();item['conditions']['task_units']=['tile-A'];self.store.record(item)
        self.assertEqual(self.store.retrieve(excluded_units=['tile-A']),[])
        self.assertEqual(self.store.retrieve(excluded_case_ids=['case-1']),[])

    def test_curated_import_keeps_lesson_not_archive_and_is_idempotent(self):
        from curated_lessons import import_lessons
        from project_experience import file_evidence
        lesson=dict(lesson_id='distilled-1',mechanism='adversarial_pressure',
                    conditions={'source_profile':'Benin'},observation='Ratio fell, F1 unchanged',
                    interpretation='Mechanical effect only',use_when='Inspect harmful adaptation',
                    do_not_infer=['Quality gain'],counterexamples=['No F1 improvement'],
                    evidence=[file_evidence(self.fixture.evidence)],tags=['M01'])
        path=self.fixture.root/'lessons.json'
        path.write_text(json.dumps({'project_id':'project-A','lessons':[lesson]}))
        self.assertEqual(import_lessons(self.store,path),['distilled-1'])
        self.assertEqual(import_lessons(self.store,path),['distilled-1'])
        cases=self.store.retrieve(context={'mechanism':'adversarial_pressure'})
        self.assertEqual(len(cases),1)
        self.assertEqual(cases[0]['result']['lesson']['interpretation'],'Mechanical effect only')


if __name__=='__main__':unittest.main()
