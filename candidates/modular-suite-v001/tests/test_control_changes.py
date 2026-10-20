"""Evidence binding rejects unexplained mutations; it is not an efficacy test."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from test_integration import Hooks, ReviewDriver
from training_runtime.control_changes import validate_control_changes, resolve_control_changes


BASE = {"confidence_cap": 0.9, "main_lr": 2e-6}
EVIDENCE = {
    "current": {"kind": "observation", "value": "fixture current measured response"},
    "knowledge": {"kind": "knowledge", "value": "fixture applicable mechanism"},
}


def basis():
    return dict(control="confidence_cap", before=0.9, after=0.93,
        support_status="supported", evidence_ids=["current", "knowledge"],
        mechanism="Fixture mechanism affecting an active path",
        why_now="Fixture current observation supports this direction",
        why_this_magnitude="Small bounded diagnostic step within host limits",
        expected_response="Inspect the affected path and paired semantic errors",
        alternative="Retain the reference and collect more observations",
        reversal_condition="Reject if predicted response fails or useful positives are lost")


class ControlChangeTests(unittest.TestCase):
    def check(self, after=None, rows=None, catalog=None):
        return validate_control_changes(before=BASE, after=after if after is not None else dict(BASE, confidence_cap=.93),
            change_basis=[basis()] if rows is None else rows, evidence_catalog=EVIDENCE if catalog is None else catalog)

    def test_unchanged_controls_need_no_justification_boilerplate(self):
        self.assertEqual(self.check(after=BASE, rows=[])["changed_controls"], [])

    def test_delta_only_keep_uses_exact_host_values_without_model_retyping(self):
        actual=resolve_control_changes(before=BASE,change_basis=[],evidence_catalog=EVIDENCE)
        self.assertEqual(actual,BASE)
        actual["confidence_cap"]=.5
        self.assertEqual(BASE["confidence_cap"],.9)

    def test_delta_only_supported_change_and_stale_or_unresolved_change(self):
        actual=resolve_control_changes(before=BASE,change_basis=[basis()],evidence_catalog=EVIDENCE)
        self.assertEqual(actual,dict(BASE,confidence_cap=.93))
        for row in (dict(basis(),before=.8),dict(basis(),support_status="unresolved")):
            with self.assertRaises(ValueError):
                resolve_control_changes(before=BASE,change_basis=[row],evidence_catalog=EVIDENCE)

    def test_second_real_failure_pattern_prose_cannot_override_actual_numeric_delta(self):
        with self.assertRaisesRegex(ValueError,"lack an evidence-bound"):
            self.check(after=dict(BASE,confidence_cap=.9107),rows=[])

    def test_actual_unexplained_delta_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "lack an evidence-bound"):
            self.check(rows=[])

    def test_declared_insufficient_support_cannot_be_executed(self):
        row=basis();row["support_status"]="unresolved"
        with self.assertRaisesRegex(ValueError, "hypothesis"):
            self.check(rows=[row])

    def test_stale_or_misdescribed_value_rejected(self):
        row=basis();row["after"]=.92
        with self.assertRaisesRegex(ValueError, "actual proposed"):
            self.check(rows=[row])

    def test_invented_evidence_and_recipe_without_current_observation_rejected(self):
        for refs in (["invented"], ["knowledge"]):
            row=basis();row["evidence_ids"]=refs
            with self.assertRaises(ValueError):self.check(rows=[row])

    def test_missing_mechanism_magnitude_or_reversal_is_rejected(self):
        for field in ("mechanism", "why_now", "why_this_magnitude", "expected_response", "alternative", "reversal_condition"):
            row=basis();row[field]=" "
            with self.assertRaises(ValueError):self.check(rows=[row])

    def test_duplicate_or_unchanged_control_explanation_rejected(self):
        with self.assertRaises(ValueError):self.check(rows=[basis(),basis()])
        with self.assertRaises(ValueError):self.check(after=BASE)

    def test_supported_bounded_new_hypothesis_can_pass_without_previous_trial_win(self):
        receipt=self.check()
        self.assertEqual(receipt["changed_controls"], ["confidence_cap"])
        self.assertFalse(receipt["scientific_validity_verified"])

    def test_nonfinite_and_missing_controls_rejected(self):
        for after in ({"confidence_cap": .93}, dict(BASE, main_lr=float("nan"))):
            with self.assertRaises(ValueError):self.check(after=after)


class BindingHooks(Hooks):
    def __init__(self):
        super().__init__()
        self.effects=[]
        self.answer=dict(continue_training=True,controls=dict(BASE,confidence_cap=.93),change_basis=[])
    def review_context(self, epoch, evaluation):
        return dict(control_baseline=BASE,control_evidence=EVIDENCE)
    def decide(self, packet):
        self.decisions.append(packet)
        return copy.deepcopy(self.answer)
    def apply_decision(self, decision):
        self.effects.append(decision)
    def validate_decision(self, decision):
        return True
    def verify_resume_state(self, epoch, run_identity):
        return True


class ReviewBindingTests(unittest.TestCase):
    def test_production_refuses_missing_control_context_before_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            hooks=BindingHooks()
            hooks.review_context=lambda epoch,evaluation: {}
            hooks.commit_review=lambda epoch,receipt: None
            hooks.begin_operation=lambda stage,operation_id: None
            hooks.reconcile_operation=lambda stage,operation_id: {"status":"unknown","evidence":"fixture"}
            driver=ReviewDriver(output=directory,maximum_epoch=200,steps_per_epoch=7,hooks=hooks,
                production=True,run_identity={"run":"fixture","branch":"main"},
                scientific_context={"contract_identity":"fixture-contract","knowledge_identity":"fixture-knowledge"})
            with self.assertRaisesRegex(ValueError,"Production context"):
                driver.after_epoch(10,{"loss":1})
            self.assertEqual(hooks.decisions,[])
            self.assertEqual(hooks.effects,[])

    def test_schema_valid_unexplained_mutation_never_reaches_apply_or_refresh(self):
        with tempfile.TemporaryDirectory() as directory:
            hooks=BindingHooks()
            driver=ReviewDriver(output=directory,maximum_epoch=200,steps_per_epoch=7,hooks=hooks)
            with self.assertRaisesRegex(ValueError,"lack an evidence-bound"):
                driver.after_epoch(10,{"loss":1})
            self.assertEqual(hooks.effects,[])
            self.assertEqual(hooks.refreshes,[])
            journal=json.loads(Path(directory,"epoch_0010_journal.json").read_text())
            self.assertNotIn("apply",journal["stages"])
            self.assertEqual(journal["stages"]["decide"]["result"],hooks.answer)

    def test_corrected_original_is_preserved_and_valid_keep_can_apply_once(self):
        from project_experience import file_evidence
        with tempfile.TemporaryDirectory() as directory:
            hooks=BindingHooks()
            driver=ReviewDriver(output=directory,maximum_epoch=200,steps_per_epoch=7,hooks=hooks)
            with self.assertRaises(ValueError):driver.after_epoch(10,{"loss":1})
            answer=dict(continue_training=True,controls=BASE,change_basis=[])
            evidence=Path(directory,"new_response.json")
            evidence.write_text(json.dumps(dict(request_id="new-scientific-request",decision=answer)))
            driver.replace_rejected_decision(10,answer,response_evidence=file_evidence(evidence))
            driver.resume(10);driver.resume(10)
            self.assertEqual(len(hooks.effects),1)
            self.assertEqual(len(hooks.decisions),1)
            journal=json.loads(Path(directory,"epoch_0010_journal.json").read_text())
            self.assertEqual(journal["rejected_decisions"][0]["decision"],hooks.answer)

    def test_replacement_cannot_hide_new_unexplained_control(self):
        from project_experience import file_evidence
        with tempfile.TemporaryDirectory() as directory:
            hooks=BindingHooks()
            driver=ReviewDriver(output=directory,maximum_epoch=200,steps_per_epoch=7,hooks=hooks)
            with self.assertRaises(ValueError):driver.after_epoch(10,{"loss":1})
            answer=dict(continue_training=True,controls=dict(BASE,main_lr=1e-6),change_basis=[])
            evidence=Path(directory,"new_response.json")
            evidence.write_text(json.dumps(dict(request_id="new-scientific-request",decision=answer)))
            with self.assertRaises(ValueError):
                driver.replace_rejected_decision(10,answer,response_evidence=file_evidence(evidence))
            self.assertEqual(hooks.effects,[])


if __name__ == "__main__":
    unittest.main()
