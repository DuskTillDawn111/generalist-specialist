"""CPU-only native handoff fit integration; temporary fixtures are not GPU evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import natural_lifecycle_fit_v2 as tool


class NaturalLifecycleFitTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.loaded={};self.reviews=[];self.sources=[]
        for i in range(2):
            context=self.write('context'+str(i)+'.json',{'execution':{'admission':'default_shared_no_permit'}})
            review=self.write('review'+str(i)+'.json',{'fixture':i});self.reviews.append(review)
            row=dict(request_id='r',split='training',status='completed',model='text#0',task='text',pool_id='pool',
                worker_response={'end_perf_ns':10_000_000},response_reader_event_enqueue_time_ns=12_000_000+i*1_000_000,
                completion_time_ns=12_000_000+i*1_000_000,previous_handler_next_request_ready=dict(request_id='r',
                    clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready',
                    scope='after_response_flush_before_next_read',pid=100,perf_ns=11_000_000+i*1_000_000,
                    received_time_ns=12_000_000+i*1_000_000))
            raw=dict(kind='cpu_fixture',execution_context_sha256='same-context',request_records=[row])
            rawpath=self.write('raw'+str(i)+'.json',raw)
            report={'raw':tool.receipt(rawpath),'context':tool.receipt(context)}
            self.sources.append(report['raw'])
            self.loaded[str(review)]=(report,{'execution_context_sha256':'same-context'},raw)
        self.clock=self.write('clock.json',dict(schema=tool.CLOCK_SCHEMA,sources=self.sources,
            clock_contract=dict(handler='host_perf_counter_ns',runtime='host_perf_counter_ns',
                worker_next_ready='host_perf_counter_ns',same_host_verified=True,source='will-be-bound')))

    def write(self,name,value):
        path=self.root/name;path.write_text(json.dumps(value));return path

    def run_fit(self,reviews=None,name='result'):
        with patch.object(tool,'load_report',side_effect=lambda path,domain:copy.deepcopy(self.loaded[str(path)])):
            return tool.fit(self.reviews if reviews is None else reviews,self.clock,self.root/name)

    def test_two_real_capture_identities_fit_distinct_receipt_and_ready_means(self):
        report=self.run_fit()
        self.assertEqual(report['status'],'FITTED')
        model=json.loads(Path(report['model']['path']).read_text())
        delays=model['profiles'][0]['scenarios']['mean']
        self.assertEqual(delays,dict(parent_receipt_ms=2.5,worker_next_ready_ms=1.5,permit_release_ms=None))
        self.assertEqual(len(model['profiles'][0]['joint_observations']),2)
        index=json.loads(Path(report['training_index']['path']).read_text())
        self.assertEqual(len({r['run_id'] for r in index['records']}),2)
        self.assertFalse(report['physical_accuracy_validated'])

    def test_single_capture_is_unknown_and_duplicate_does_not_satisfy_count(self):
        report=self.run_fit([self.reviews[0]])
        self.assertEqual(report['status'],'UNKNOWN');self.assertNotIn('model',report)
        with self.assertRaisesRegex(ValueError,'independent capture'):
            self.run_fit([self.reviews[0],self.reviews[0]],name='duplicates')

    def test_missing_readiness_remains_unknown_instead_of_zero(self):
        self.loaded[str(self.reviews[0])][2]['request_records'][0].pop('previous_handler_next_request_ready')
        report=self.run_fit()
        self.assertEqual(report['status'],'FITTED_PARTIAL')
        model=json.loads(Path(report['model']['path']).read_text())
        self.assertIsNone(model['profiles'][0]['scenarios']['mean']['worker_next_ready_ms'])

    def test_context_mismatch_and_clock_source_drift_rejected(self):
        self.loaded[str(self.reviews[1])][1]['execution_context_sha256']='other-context'
        with self.assertRaisesRegex(ValueError,'contexts differ'):self.run_fit()
        Path(self.sources[0]['path']).write_text('{}')
        with self.assertRaisesRegex(ValueError,'clock evidence'):self.run_fit(name='changed-clock')


if __name__=='__main__':unittest.main()
