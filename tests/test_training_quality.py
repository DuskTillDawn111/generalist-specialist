"""Training-only boundary and error-unit aggregation; CPU fixtures only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.predictor_validation_20261027.training_quality import fit
from tools.predictor_validation_20261027.probe_validation import receipt


class TrainingQualityTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        context=self.write('context.json',{'residents':{'o':'ocr'}})
        self.catalog=self.write('catalog.json',{'quality':{'references':{'ocr':{
            'short':{'task':'ocr','label':'a'},'long':{'task':'ocr','label':'abcdefghi'}}}}})
        self.raw={'kind':'measured','backend':'native','status':'completed','context':receipt(context),
            'execution_context_sha256':'cpu-fixture','request_records':[
                {'request_id':'1','split':'training','status':'completed','model':'o','task':'ocr','pool_id':'short',
                 'worker_response':{'task':'ocr','prediction':'x','success':True}},
                {'request_id':'2','split':'training','status':'completed','model':'o','task':'ocr','pool_id':'long',
                 'worker_response':{'task':'ocr','prediction':'abcdefghi','success':True}}]}

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def run_fit(self):return fit(self.write('raw.json',self.raw),self.catalog,self.root/'quality.json')

    def test_total_units_ratio_and_source_binding(self):
        result=self.run_fit()
        self.assertEqual(result['loss_by_family_task']['ocr']['ocr'],.1) # 1/10, not mean(.0,1)=.5
        self.assertEqual(result['request_count'],2)
        self.assertFalse(result['heldout_observations_used'])
        for source in result['source_receipts']:self.assertEqual(receipt(source['path']),source)

    def test_validation_result_rejected(self):
        self.raw['request_records'][0]['split']='validation'
        with self.assertRaisesRegex(ValueError,'Training-only'):self.run_fit()

    def test_incomplete_cohort_and_duplicate_refused(self):
        self.raw['request_records'][0]['status']='error'
        with self.assertRaisesRegex(ValueError,'Training-only'):self.run_fit()
        self.raw['request_records'][0]['status']='completed';self.raw['request_records'][1]['request_id']='1'
        with self.assertRaisesRegex(ValueError,'Unique'):self.run_fit()

    def test_existing_output_never_overwritten(self):
        self.run_fit()
        with self.assertRaises(FileExistsError):self.run_fit()

if __name__=='__main__':unittest.main()
