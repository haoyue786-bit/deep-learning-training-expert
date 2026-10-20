"""Full-history fidelity and compact transport, not model-quality assertions."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_integration import Hooks, ReviewDriver
from training_runtime.expert_review import build, model_inputs, normalize, compact_history, section
from training_runtime.paired_errors import compare


def fixture():
    ids=dict(run_id="fixture",branch_id="main",data_identity="labels-mask-decoder-v1")
    specs=[dict(key=k,label=k,unit=unit,panel=panel,definition="fixture "+k,direction=direction)
           for k,unit,panel,direction in [("f1","fraction","Quality","max"),
               ("gradient","L2","Gradient","none"),("prior","score","Prior","none")]]
    rows=[dict(**ids,epoch=e,checkpoint="fixture-"+str(e),
               metrics=dict(f1=(.1 if e==20 else .8+e*.0001),gradient=(99 if e==17 else None),prior=.9))
          for e in range(41)]
    events=[dict(**ids,epoch=25,status="applied",summary="fixture LR changed"),
            dict(**ids,epoch=30,status="rejected",summary="fixture unsupported cap")]
    return dict(**ids,cutoff_epoch=40,task={"objective":"fixture"},controls={"lr":1e-6},bounds={"lr":[1e-7,1e-5]},
        metric_specs=specs,observations=rows,events=events,prior_context={"scope":"fixture","calibrated":False},
        open_questions=["fixture"],knowledge=[],experience=[])


class ExpertReviewTests(unittest.TestCase):
    def test_class_suppression_is_not_correct_class_recovery(self):
        before = [dict(point_id=1,class_id=1,predicted_class=2),
                  dict(point_id=2,class_id=2,predicted_class=2)]
        after = [dict(point_id=1,class_id=1,predicted_class=3),
                 dict(point_id=2,class_id=2,predicted_class=1)]
        result = compare(before,after,focus_class=2,classes=[0,1,2,3])
        self.assertEqual(result["removed_focus_false_positives"],1)
        self.assertEqual(result["removed_false_positives_correctly_reclassified"],0)
        self.assertEqual(result["newly_missed_focus_positives"],1)
        after[0]["class_id"] = 0
        with self.assertRaisesRegex(ValueError,"Label change"):
            compare(before,after,focus_class=2,classes=[0,1,2,3])

    def test_compaction_keeps_exact_extrema_and_additional_diagnostics(self):
        with tempfile.TemporaryDirectory() as temp:
            data=fixture();data["diagnostics"]={"D_accuracy":.95,"pair_changes":[[1,2,3]]}
            b=build(data,Path(temp)/"review")
            table=b["trajectory_summary"];row=dict(zip(table["columns"],table["rows"]["f1"]))
            self.assertEqual(row["min"],[20,.1])
            self.assertEqual(b["diagnostics"],data["diagnostics"])
            self.assertEqual(json.loads((Path(temp)/"review/history.json").read_text())["observations"],data["observations"])

    def test_entire_history_retained_and_extrema_and_interventions_anchored(self):
        data=fixture()
        _,specs,rows,events,_=normalize(data)
        summary,missing,anchors=compact_history(specs,rows,events)
        self.assertEqual(len(rows),41)
        self.assertTrue({0,17,20,24,25,40} <= {r["epoch"] for r in anchors})
        self.assertEqual(summary["gradient"]["n"],1)
        self.assertIsNone(summary["gradient"]["recent_endpoint_change"])
        self.assertEqual(summary["f1"]["min"],[20,.1])

    def test_foreign_lineage_labels_mask_or_decoder_cannot_be_merged(self):
        for key in ("run_id","branch_id","data_identity"):
            data=fixture();data["observations"][2][key]="different"
            with self.assertRaisesRegex(ValueError,"Mixed"):normalize(data)

    def test_future_excluded_duplicate_or_missing_current_epoch_rejected(self):
        data=fixture();data["cutoff_epoch"]=10
        self.assertEqual(normalize(data)[-1],30)
        data=fixture();data["observations"].append(data["observations"][0])
        with self.assertRaises(ValueError):normalize(data)
        data=fixture();data["cutoff_epoch"]=41
        with self.assertRaises(ValueError):normalize(data)

    def test_nonfinite_is_not_hidden_or_converted_to_zero(self):
        data=fixture();data["observations"][0]["metrics"]["f1"]=float("nan")
        with self.assertRaisesRegex(ValueError,"Nonfinite"):normalize(data)

    def test_missing_gradient_e0_does_not_become_measured_zero(self):
        data=fixture();data["observations"]=data["observations"][:1];data["events"]=[];data["cutoff_epoch"]=0
        _,specs,rows,events,_=normalize(data)
        summary,missing,_=compact_history(specs,rows,events)
        self.assertIn("gradient",missing);self.assertNotIn("gradient",summary)

    def test_budget_failure_does_not_write_partial_or_truncated_brief(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"review"
            with self.assertRaisesRegex(ValueError,"budget"):build(fixture(),out,maximum_characters=10)
            self.assertFalse(out.exists())

    def test_actual_plot_csv_and_delivery_identities(self):
        with tempfile.TemporaryDirectory() as temp:
            out=Path(temp)/"review";brief=build(fixture(),out)
            self.assertEqual(len((out/"history.csv").read_text().splitlines()),42)
            self.assertGreater((out/"training_curves.png").stat().st_size,1000)
            inputs=model_inputs(out/"brief.json",vision_available=True)
            self.assertEqual(inputs[1]["type"],"localImage")
            self.assertEqual(json.loads(inputs[0]["text"])["image"]["delivery"],"attached_in_this_request")
            text_only=model_inputs(out/"brief.json",vision_available=False)
            self.assertEqual(len(text_only),1)
            self.assertEqual(json.loads(text_only[0]["text"])["image"]["delivery"],"not_delivered_text_only")
            (out/"training_curves.png").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"artifact changed"):model_inputs(out/"brief.json",vision_available=True)

    def test_knowledge_must_be_exact_relevant_section_not_entire_manual(self):
        with tempfile.TemporaryDirectory() as temp:
            p=Path(temp)/"knowledge.md";p.write_text("# Manual\n\n## One\nUseful mechanism.\n\n## Two\nUnrelated.\n")
            item=section(p,"One")
            self.assertNotIn("Unrelated",item["text"])
            data=fixture();data["knowledge"]=[dict(item,text="fabricated")]
            with self.assertRaisesRegex(ValueError,"differs"):build(data,Path(temp)/"review")

    def test_actual_chart_marks_only_applied_events_and_preserves_gaps(self):
        from training_runtime.expert_review import plot_history
        from matplotlib.axes import Axes
        calls=[];original=Axes.axvline
        def line(ax,x=0,*a,**kw):calls.append(x);return original(ax,x,*a,**kw)
        _,specs,rows,events,_=normalize(fixture())
        with tempfile.TemporaryDirectory() as temp,patch.object(Axes,"axvline",line):
            plot_history(Path(temp)/"plot.png",specs,rows,events,"Fixture")
        self.assertTrue(calls);self.assertEqual(set(calls),{25})

    def test_driver_sends_brief_without_duplicate_archive_context(self):
        with tempfile.TemporaryDirectory() as temp:
            h=Hooks();sent=[]
            brief=dict(format="expert_review_v1",epoch=10,history={"sha256":"fixture"},
                       controls={"lr":1e-6},open_questions=["fixture"])
            h.review_context=lambda e,v:dict(expert_brief=brief,control_baseline=brief["controls"],control_evidence={},
                                             unnecessary_archive="do not send this")
            h.decide=lambda packet:(sent.append(packet) or dict(continue_training=True,controls=brief["controls"],change_basis=[]))
            d=ReviewDriver(output=temp,maximum_epoch=200,steps_per_epoch=7,hooks=h,
                scientific_context={"legacy_knowledge":"do not duplicate"})
            d.after_epoch(10,{"loss":1})
            self.assertEqual(sent[0]["expert_brief"],brief)
            for key in ("evaluation","recent_observations","scientific_context","current_context"):
                self.assertNotIn(key,sent[0])


if __name__=="__main__":unittest.main()
