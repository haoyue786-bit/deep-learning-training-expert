"""Execution follows the bounded numeric envelope, never scientific prose."""
import sys,unittest,tempfile,json
from pathlib import Path
SKILLS=Path(__file__).resolve().parents[1]/'skills'
sys.path.insert(0,str(SKILLS/'deep-learning-training-expert/scripts'))
from training_runtime.execution_template import validate_response,save_response

class ExecutionTemplateTests(unittest.TestCase):
    def setUp(self):
        self.before={'lambda_adv':.004,'supplementary_loss_weight':0.,'D_steps_per_G_step':2}
        self.bounds={'lambda_adv':[.0001,.05],'supplementary_loss_weight':[0.,1.],'D_steps_per_G_step':[1,4]}
    def validate(self,action='KEEP',changes=None,selected=0,record='No change.'):
        return validate_response({'execution':{'action':action,'changes':changes or [],'selected_epoch':selected},'training_record':record},before=self.before,bounds=self.bounds,actions=['INITIALIZE','KEEP','ADJUST','STOP','HOLD'],candidates=[0,5,10],integer_parameters=['D_steps_per_G_step'])
    def test_prose_cannot_edit_controls_or_pause(self):
        r=self.validate(record='Consider increasing lambda_adv to 0.1. Future observations would help.')
        self.assertEqual(r['resolved_controls'],self.before)
        self.assertEqual(r['execution']['action'],'KEEP')
    def test_bounded_first_use_changes_only_requested_weight(self):
        r=self.validate('ADJUST',[{'parameter':'supplementary_loss_weight','value':.25}],record='Provisional first use; inspect the next response, not an optimum.')
        self.assertEqual(r['resolved_controls'],dict(self.before,supplementary_loss_weight=.25))
    def test_existing_baseline_can_be_selected(self):
        self.assertEqual(self.validate('STOP',selected=0)['execution']['selected_epoch'],0)
    def test_invalid_edits_do_not_execute(self):
        bad=[('KEEP',[{'parameter':'lambda_adv','value':.002}],0),('ADJUST',[],0),('ADJUST',[{'parameter':'lambda_adv','value':.2}],0),('ADJUST',[{'parameter':'D_steps_per_G_step','value':2.5}],0),('ADJUST',[{'parameter':'lambda_adv','value':.002},{'parameter':'lambda_adv','value':.003}],0),('KEEP',[],5),('STOP',[],20)]
        for action,changes,selected in bad:
            with self.subTest(action=action,changes=changes,selected=selected):
                with self.assertRaises((ValueError,AssertionError)):self.validate(action,changes,selected)
    def test_record_and_program_owned_facts_are_saved_separately(self):
        r=self.validate(record='Evidence is incomplete; KEEP remains justified.')
        with tempfile.TemporaryDirectory() as d:
            save_response(d,r,{'epoch':10,'source_f1':.8},{'model':'test'})
            p=Path(d)
            self.assertEqual((p/'training_record.md').read_text().strip(),r['training_record'])
            self.assertEqual(json.loads((p/'host_facts.json').read_text())['epoch'],10)
            self.assertNotIn('source_f1',json.loads((p/'execution.json').read_text()))

if __name__=='__main__':unittest.main()
