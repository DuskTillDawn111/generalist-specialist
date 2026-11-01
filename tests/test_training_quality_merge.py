"""Independent training-run merge boundaries, CPU only."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from tools.predictor_validation_20261027.training_quality_merge import merge
from tools.predictor_validation_20261027.probe_validation import receipt

class MergeTests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        self.paths=[]
        for i,(errors,units) in enumerate([(1,1),(0,9)]):
            raw=self.root/('raw'+str(i));raw.write_text(str(i))
            doc={'schema':'gsdeploy.training_family_task_quality.v1','split':'training','heldout_observations_used':False,
                'execution_context_sha256':'cpu','request_count':1,'source_receipts':[receipt(raw)],
                'totals_by_request':{'same-local-id':{'family':'ocr','task':'ocr','pool_id':'p'+str(i),'numerator':errors,'denominator':units}}}
            self.paths.append(self.write(str(i)+'.json',doc))
    def write(self,name,value):
        p=self.root/name;p.write_text(json.dumps(value));return p
    def test_total_units_and_namespaced_duplicate_local_request_ids(self):
        result=merge(self.paths,self.root/'out.json')
        self.assertEqual(result['loss_by_family_task']['ocr']['ocr'],.1)
        self.assertEqual(result['request_count'],2)
        self.assertEqual(result['totals_by_family_task']['ocr']['ocr']['unique_input_count'],2)
    def test_duplicate_document_and_context_mismatch_refused(self):
        with self.assertRaisesRegex(ValueError,'Duplicate'):merge([self.paths[0]]*2,self.root/'out.json')
        doc=json.loads(self.paths[1].read_text());doc['execution_context_sha256']='other';self.write('1.json',doc)
        with self.assertRaisesRegex(ValueError,'contexts'):merge(self.paths,self.root/'out.json')
    def test_validation_or_changed_source_refused(self):
        doc=json.loads(self.paths[1].read_text());doc['split']='validation';self.write('1.json',doc)
        with self.assertRaisesRegex(ValueError,'Training-only'):merge(self.paths,self.root/'out.json')
        doc['split']='training';self.write('1.json',doc);(self.root/'raw1').write_text('changed')
        with self.assertRaisesRegex(ValueError,'source receipt'):merge(self.paths,self.root/'out.json')

if __name__=='__main__':unittest.main()
