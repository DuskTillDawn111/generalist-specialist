"""Independent CPU lineage checks, including actual historical priming format."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.native_collect import context_identity,receipt
from tools.predictor_validation_20261027.training_source_guard_v11 import availability_training_sources

ROOT=Path(__file__).resolve().parents[1]


class TrainingGuardIndependentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.context=dict(training_only=True,execution_sources=[])
        context_id=context_identity(self.context)
        self.plan=dict(split='training',training_only=True,jobs=[dict(split='training')],execution_context_sha256=context_id)
        self.raw=dict(kind='measured',status='completed',errors=[],execution_context_sha256=context_id,
            request_records=[dict(request_id='training:0',split='training',status='completed')],
            priming_request_records=[dict(request_id='priming:0',phase='priming',status='completed')])

    def refs(self,raw=None,plan=None,context=None,label=False):
        result=[]
        for i in range(2):
            directory=self.root/('capture'+str(i));directory.mkdir(exist_ok=True)
            docs={'raw.json':dict(copy.deepcopy(self.raw if raw is None else raw),capture=i),
                  'plan.json':self.plan if plan is None else plan,'context.json':self.context if context is None else context}
            sources=[]
            for name,doc in docs.items():
                path=directory/name;path.write_text(json.dumps(doc));sources.append(receipt(path))
            report=dict(schema='gsdeploy.recorded_DAG_pending_frontier.v8',status='SOURCE_DAG_PENDING_FRONTIER_EXTRACTED',validation_labels_used=label,sources=sources)
            path=directory/'report.json';path.write_text(json.dumps(report));result.append(receipt(path))
        return result

    def test_actual_two_training_captures_accept_legacy_unsplit_completed_priming(self):
        model=json.loads((ROOT/'artifacts/predictor-dag-availability-complete-fit-v8-20261028-training02/model.json').read_text())
        audits=availability_training_sources(model['source_DAG_reports'])
        self.assertEqual(len(audits),2);self.assertEqual([a['measurement_requests'] for a in audits],[22,22])
        self.assertNotEqual(audits[0]['raw']['sha256'],audits[1]['raw']['sha256'])

    def test_public_synthetic_lineage_rejects_validation_measurements_priming_and_report(self):
        self.assertEqual(len(availability_training_sources(self.refs())),2)
        for field in ('request_records','priming_request_records'):
            raw=copy.deepcopy(self.raw);raw[field][0]['split']='validation'
            with self.subTest(field=field),self.assertRaises(ValueError):availability_training_sources(self.refs(raw=raw))
        with self.assertRaises(ValueError):availability_training_sources(self.refs(label=True))

    def test_changed_context_incomplete_rows_or_duplicate_capture_cannot_add_training_support(self):
        for change in ('context','incomplete','priming-phase'):
            raw=copy.deepcopy(self.raw)
            if change=='context':raw['execution_context_sha256']='foreign'
            elif change=='incomplete':raw['request_records'][0]['status']='incomplete'
            else:raw['priming_request_records'][0]['phase']='measurement'
            with self.subTest(change=change),self.assertRaises(ValueError):availability_training_sources(self.refs(raw=raw))
        refs=self.refs()
        with self.assertRaises(ValueError):availability_training_sources([refs[0],refs[0]])


if __name__=='__main__':unittest.main()
