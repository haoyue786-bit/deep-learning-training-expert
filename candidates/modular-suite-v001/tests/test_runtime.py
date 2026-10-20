"""Behavioral invariants of the callable algorithms and scientific data boundary."""
import concurrent.futures
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

SKILLS = Path(__file__).resolve().parents[1] / "skills"
sys.path.insert(0, str(SKILLS / "deep-learning-training-expert/scripts"))
sys.path.insert(0, str(SKILLS / "training-experience-update/scripts"))

import torch
from project_experience import ProjectStore, file_evidence
from training_runtime.cadence import Cadence, FinitePoolCursor, numerical_anomalies
from training_runtime.validation import evaluate_points, screen_candidates, compare_feedback
from training_runtime.decision_packet import build, defaults
from training_runtime import algorithms as algo
from training_runtime import registry


class CadenceTests(unittest.TestCase):
    def test_new_schedule_decouples_saved_candidates_and_agent_calls(self):
        cadence = Cadence()
        events = {e: cadence.events(e, maximum_epoch=23) for e in range(1, 24)}
        self.assertEqual([e for e, v in events.items() if v['evaluate']], [5, 10, 15, 20, 23])
        self.assertEqual([e for e, v in events.items() if v['agent']], [10, 20, 23])
        self.assertEqual([e for e, v in events.items() if v['automatic_refresh']], [10, 20])
        self.assertTrue(all(v['telemetry'] for v in events.values()))
        self.assertTrue(cadence.events(3, maximum_epoch=23, emergency=True)['agent'])

    def test_old_job_is_not_silently_migrated(self):
        self.assertEqual(Cadence.from_job({'review_interval': 5}).agent_every, 5)
        self.assertEqual(Cadence.from_job({'review_cadence': Cadence().policy()}).agent_every, 10)

    def test_invalid_cadences_rejected(self):
        for kwargs in [{'agent_every': 7}, {'telemetry_every': 2}, {'automatic_refresh_every': 5}]:
            with self.assertRaises(ValueError): Cadence(**kwargs)

    def test_pool_lasts_to_next_decision_without_increasing_occurrences(self):
        policy = Cadence().generation(50, steps_per_epoch=63, maximum_epoch=200)
        self.assertEqual(policy['exposure_horizon_updates'], 630)
        self.assertEqual(policy['expiry_epoch'], 60)
        self.assertEqual(policy['occurrence_cap_per_generation'], 1)
        cursor = FinitePoolCursor('poolA', 630)
        self.assertEqual(cursor.take(63), (0, 63))
        restored = FinitePoolCursor.restore(cursor.state_dict(), generation_id='poolA', horizon_updates=630)
        self.assertEqual(restored.take(63), (63, 126))
        with self.assertRaises(ValueError): FinitePoolCursor.restore(cursor.state_dict(), generation_id='poolB', horizon_updates=630)

    def test_nonfinite_measured_values_not_missing_diagnostics(self):
        self.assertEqual(numerical_anomalies({'loss': float('nan'), 'not_measured': None, 'x': [1., float('inf')]}), ['loss', 'x[1]'])


class AlgorithmTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(16)
        self.q = torch.softmax(torch.randn(1, 4, 3, 3), 1).clamp(max=.4).requires_grad_()

    def test_M01_neutral_matches_loss_and_gradients(self):
        local = torch.tensor(2., requires_grad=True); global_ = torch.tensor(3., requires_grad=True)
        loss = algo.combine_adversarial(local, global_)
        self.assertEqual(loss.item(), 5.)
        loss.backward(); self.assertEqual((local.grad.item(), global_.grad.item()), (1., 1.))
        self.assertEqual(algo.combine_adversarial(local, global_, local_scale=0.).item(), 3.)

    def test_M02_neutral_clipped_mass_and_gradient_detach(self):
        priority, stats = algo.class_pixel_priority(self.q, [1., 1., 1., 1.])
        torch.testing.assert_close(priority, torch.ones_like(priority))
        self.assertFalse(priority.requires_grad)
        self.assertFalse(stats['within_pixel_class_proportions_changed'])
        priority, _ = algo.class_pixel_priority(self.q, [1., 1., 3., 1.])
        self.assertAlmostEqual(priority.mean().item(), 1., places=6)
        self.assertGreater(priority.std().item(), 0.)

    def test_M03_soft_attenuation_is_not_renormalized(self):
        neutral, _ = algo.pair_ambiguity_reliability(self.q, class_pair=(2, 1), strength=0.)
        torch.testing.assert_close(neutral, torch.ones_like(neutral))
        reliability, stats = algo.pair_ambiguity_reliability(self.q, class_pair=(2, 1))
        self.assertLess(stats['mean_reliability'], 1.)
        self.assertFalse(reliability.requires_grad)
        self.assertFalse(stats['generated_target_labels'])

    def test_M03_missing_ecology_is_unknown_and_not_negative_label(self):
        unknown = torch.full((1, 3, 3), float('nan'))
        a, _ = algo.pair_ambiguity_reliability(self.q, class_pair=(2, 1))
        b, stats = algo.pair_ambiguity_reliability(self.q, class_pair=(2, 1), ecological_support=unknown,
                                                ecological_confidence=unknown, ecological_strength=1.)
        torch.testing.assert_close(a, b); self.assertEqual(stats['ecological_supported_pixels'], 0)

    def test_M02_M03_joint_loss_neutral_and_real_gradient(self):
        output = torch.randn(1, 8, 3, 3, requires_grad=True)
        encoded = torch.cat([self.q.detach(), torch.zeros_like(self.q)], 1)
        reference = -(encoded*torch.log_softmax(output, 1)).sum(1).mean()
        priority, _ = algo.class_pixel_priority(self.q, [1., 1., 1., 1.])
        neutral, _ = algo.pair_ambiguity_reliability(self.q, class_pair=(2, 1), strength=0.)
        loss, _ = algo.conditional_domain_loss(output, self.q, domain='source', priority=priority, reliability=neutral)
        torch.testing.assert_close(loss, reference)
        loss.backward(); self.assertGreater(output.grad.abs().sum().item(), 0.); self.assertIsNone(self.q.grad)
        weaker, _ = algo.conditional_domain_loss(output, self.q, domain='source', reliability=torch.zeros_like(neutral))
        self.assertEqual(weaker.item(), 0.)

    def test_M04_has_trainable_feature_gradient_and_missing_support(self):
        features = torch.randn(1, 5, 3, 3, requires_grad=True)
        labels = torch.tensor([[[1, 2, 1], [2, 1, 2], [1, 2, 10]]])
        loss, stats = algo.source_pair_margin(features, labels, class_pair=(2, 1), margin=2.)
        self.assertTrue(stats['supported']); loss.backward()
        self.assertGreater(features.grad.abs().sum().item(), 0.)
        loss, stats = algo.source_pair_margin(features, torch.ones_like(labels), class_pair=(2, 1))
        self.assertFalse(stats['supported']); self.assertEqual(loss.item(), 0.)
        with self.assertRaises(PermissionError): algo.source_pair_margin(features, labels, class_pair=(2, 1), data_role='development')

    def test_M08_real_hard_negative_gradient_and_role_guard(self):
        logits = torch.randn(1, 4, 3, 3, requires_grad=True); labels = torch.ones(1, 3, 3, dtype=torch.long)
        labels[0, 0, 0] = 10
        loss, stats = algo.source_hard_negative_loss(logits, labels, focus_class=2, negative_class=1, fraction=.5)
        self.assertEqual(stats['eligible_pixels'], 8); self.assertEqual(stats['selected_pixels'], 4)
        loss.backward(); self.assertGreater(logits.grad.abs().sum().item(), 0.)
        with self.assertRaises(PermissionError): algo.source_hard_negative_loss(logits, labels, focus_class=2, negative_class=1, data_role='development')

    def test_registry_rejects_unknown_unimplemented_or_production_calls(self):
        func, controls = registry.resolve('M03', 'pair_ambiguity_reliability', data_role='unlabelled_training',
                                         authorized_variants=['M03:pair_ambiguity_reliability'])
        self.assertIs(func, algo.pair_ambiguity_reliability); self.assertEqual(controls['strength'], .5)
        for kwargs in [{'controls': {'typo': 1}}, {'controls': {'strength': float('nan')}}, {'execution_scope': 'production'}]:
            with self.assertRaises((ValueError, PermissionError)):
                registry.resolve('M03', 'pair_ambiguity_reliability', data_role='unlabelled_training',
                                 authorized_variants=['M03:pair_ambiguity_reliability'], **kwargs)
        with self.assertRaises(ValueError): registry.resolve('M05', 'class_mass_correction', data_role='unlabelled_training', authorized_variants=['M05:class_mass_correction'])


class ValidationTests(unittest.TestCase):
    def feedback(self):
        return evaluate_points([{'point_id': str(i), 'class_id': actual, 'predicted_class': predicted}
                                for i, (actual, predicted) in enumerate([(1, 2), (1, 2), (2, 1), (2, 2), (0, 0)])],
                               class_names=['other', 'mixed', 'focus', 'built'],
                               dataset={'dataset_id':'dev1', 'label_version':'v1', 'sha256':'abc', 'role':'development'},
                               checkpoint={'identity':'E10', 'sha256':'def'}, focus_class=2, confusing_class=1)

    def test_confusion_orientation_precision_recall_and_absent_support(self):
        f = self.feedback(); self.assertEqual(f['confusion_matrix'][1][2], 2)
        self.assertEqual(f['confusable_pair']['focus_as_confusing'], 1)
        self.assertAlmostEqual(f['focus_metrics']['precision'], 1/3)
        self.assertAlmostEqual(f['focus_metrics']['recall'], 1/2)
        self.assertIsNone(f['per_class'][3]['f1'])

    def test_measured_feedback_changes_candidate_shortlist(self):
        f = self.feedback(); self.assertEqual(screen_candidates(f)['candidates'][0]['module_id'], 'M03')
        other = copy.deepcopy(f); other['confusable_pair'].update(confusing_as_focus=1, focus_as_confusing=3)
        self.assertEqual(screen_candidates(other)['candidates'][0]['module_id'], 'M04')
        other['evaluation_role'] = 'independent_evaluation'
        with self.assertRaises(PermissionError): screen_candidates(other)

    def test_compact_packet_has_requested_model_and_no_extra_calls(self):
        f = self.feedback(); packet = build(f, available_variants=['M03:pair_ambiguity_reliability', 'M04:source_pair_margin'])
        self.assertEqual(defaults()['model'], 'gpt-6.1-sol')
        self.assertEqual(packet['reasoning_effort'], 'medium'); self.assertEqual(packet['extra_model_calls'], 0)
        self.assertEqual(len(packet['available_relevant_modules']), 0)
        self.assertEqual(len(packet['available_trial_variants']), 2)
        self.assertTrue(packet['candidate_hypotheses']['advisory_only'])
        self.assertLess(len(json.dumps(packet)), 16000)
        self.assertNotIn('bodies', packet)

    def test_comparison_rejects_changed_labels_or_population(self):
        f = self.feedback(); changed = copy.deepcopy(f); changed['dataset']['label_version'] = 'v2'
        with self.assertRaises(ValueError): compare_feedback(f, changed)


class ExperienceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.root = Path(self.temp.name)
        self.context = dict(project_id='project-A', project_root=str(self.root), experience_dir=str(self.root/'experience'),
                            authorization={'write_experience':True, 'source':'unit-test authorization'})
        self.store = ProjectStore(self.context); self.evidence = self.root/'result.json'; self.evidence.write_text('{"result":"null"}')

    def tearDown(self): self.temp.cleanup()

    def case(self):
        return dict(case_id='case-1', project_id='project-A', action={'module':'M03'}, result={'f1_delta':0}, outcome='null',
                    evidence=[file_evidence(self.evidence)], conditions={'tile':'tile-A'}, hypotheses=['May need other mechanism'],
                    cost={'updates':2}, evaluation_role='development')

    def test_missing_identity_and_unauthorized_path_cannot_write(self):
        for bad in [{}, dict(self.context, project_id=''), dict(self.context, authorization={}),
                    dict(self.context, experience_dir=str(self.root.parent/'elsewhere'))]:
            with self.assertRaises((ValueError, PermissionError)): ProjectStore(bad)
        self.assertFalse((self.root/'experience').exists())

    def test_project_separation_idempotence_and_immutable_records(self):
        case = self.case(); a = self.store.record(case); self.assertEqual(self.store.record(case), a)
        case['result']['f1_delta'] = 1
        with self.assertRaises(ValueError): self.store.record(case)
        foreign = dict(self.context, project_id='project-B')
        with self.assertRaises(PermissionError): ProjectStore(foreign).retrieve()

    def test_evidence_hash_checked(self):
        case = self.case(); self.evidence.write_text('changed')
        with self.assertRaises(ValueError): self.store.record(case)

    def test_validation_budget_cannot_reset_or_become_threshold_search(self):
        args = dict(dataset={'dataset_id':'d', 'label_version':'v1', 'sha256':'a', 'role':'development'},
                    checkpoint={'identity':'e1', 'sha256':'b'}, purpose='trial_evaluation', maximum_uses=1)
        one = self.store.reserve_validation(usage_id='u1', **args)
        self.assertEqual(self.store.reserve_validation(usage_id='u1', **args), one)
        self.store.finish_validation('u1', self.evidence, status='failed')
        args['maximum_uses'] = 100
        with self.assertRaises(PermissionError): self.store.reserve_validation(usage_id='u2', **args)
        args['purpose'] = 'threshold_search'
        with self.assertRaises(PermissionError): self.store.reserve_validation(usage_id='u3', **args)

    def test_validation_reservations_are_atomic(self):
        def reserve(i):
            try:
                self.store.reserve_validation(usage_id=str(i), dataset={'dataset_id':'d', 'label_version':'v1','sha256':'x','role':'development'},
                                              checkpoint={'identity':str(i),'sha256':str(i)}, purpose='trial_evaluation', maximum_uses=3)
                return True
            except PermissionError: return False
        with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
            results = list(pool.map(reserve, range(8)))
        self.assertEqual(sum(results), 3)


if __name__ == '__main__': unittest.main()
