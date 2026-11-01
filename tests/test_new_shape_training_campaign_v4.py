import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import new_shape_training_campaign_v4 as target
from tools.predictor_validation_20261027.native_collect import receipt


class NewShapePreparationTests(unittest.TestCase):
    def fixture(self,root):
        row=dict(pool_id='new-training',task_type='text',sample=dict(id='new-training',context='ctx',question='q'))
        pool=root/'runtime-inputs.jsonl';pool.write_text(json.dumps(row)+'\n')
        selected=root/'selected.json';selected.write_text(json.dumps(row))
        split=root/'split.json';split.write_text('{}')
        report=dict(purpose='ADDITIONAL_TRAINING_NOT_HELDOUT_VALIDATION',selected_input=receipt(selected),
            catalog=receipt(pool),prior_split_selection=receipt(split),selected=dict(pool_id='new-training',input_tokens=963),excluded_pool_ids=['heldout'])
        selection=root/'selection.json';selection.write_text(json.dumps(report))
        reference=root/'reference';reference.mkdir();(reference/'campaign.json').write_text('{}')
        primer=[dict(task='text',pool_id='old-training',input_size=1)]
        source=dict(original_path=str(pool),frozen=receipt(pool))
        for domain in ('metadata','light-kernel','natural'):
            run=reference/'modalities'/domain;run.mkdir(parents=True)
            (run/'context.json').write_text(json.dumps(dict(execution_sources=[source],sources=[source['frozen']],measurement_domain=domain)))
            (run/'plan.json').write_text(json.dumps(dict(domain=domain,inputs=primer,split='training',jobs=[])))
        return reference,selection,primer

    def test_one_new_job_preserves_priming_and_binds_actual_frozen_catalog(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(target,'source_audit',return_value={}):
            root=Path(tmp);reference,selection,primers=self.fixture(root);output=root/'prepared'
            result=target.prepare(reference,selection,output)
            self.assertEqual(len(result['runs']),3)
            for domain in result['runs']:
                plan=json.loads((output/domain/'plan.json').read_text())
                self.assertEqual(plan['inputs'],primers)
                self.assertEqual(len(plan['jobs']),1)
                self.assertEqual(plan['jobs'][0]['inputs'],{'text#0':'new-training'})
                self.assertNotIn('new-training',{p['pool_id'] for p in plan['inputs']})
                self.assertEqual(target.audit(output/domain),{})

    def test_excluded_input_cannot_enter_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);reference,selection,_=self.fixture(root)
            report=json.loads(selection.read_text());report['excluded_pool_ids'].append('new-training');selection.write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError,'Eligible exact text training'):
                target.prepare(reference,selection,root/'prepared')
            self.assertFalse((root/'prepared').exists())


if __name__=='__main__':unittest.main()
