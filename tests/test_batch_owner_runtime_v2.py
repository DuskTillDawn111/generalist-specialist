"""Public arrival list -> real prepare/planner/single-generate/delivery on CPU."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gsdeploy.backends.shared_text_batch import SharedTextBatchExecutor
from gsdeploy.runtime.batch_owner import BatchOwnerFailure, run_arrivals, run_retained_arrivals
from gsdeploy.runtime.batching import BatchConfiguration
from tests.test_shared_text_batch_v2 import FakeExecutor, sample


class Clock:
    def __init__(self):self.time=0;self.waits=[]
    def now(self):return self.time
    def advance(self,n):self.time+=n
    def wait(self,n):self.waits.append(n);self.advance(n)


def timed_factory(clock, *, fail_prepare=None, fail_generate_call=None):
    calls=[]
    def factory():
        calls.append('load');clock.advance(100)
        executor=FakeExecutor()
        original_prepare=executor.prepare
        def prepare(request):
            clock.advance(2)
            if request['sample_id']==fail_prepare:raise RuntimeError('prepare failed original')
            return original_prepare(request)
        executor.prepare=prepare
        original_generate=executor.model.generate
        def generate(**kwargs):
            clock.advance(5)
            if len(executor.model.calls)+1==fail_generate_call:
                raise RuntimeError('generate failed original')
            return original_generate(**kwargs)
        executor.model.generate=generate
        owner=SharedTextBatchExecutor(executor,tensor_factory=lambda rows:np.array(rows,dtype=np.int64),
            clock=clock.now,identity={'backend':'CPU_FAKE_TEST_ONLY'})
        calls.append(owner)
        return owner
    return factory,calls


class BatchOwnerRuntimeV2Tests(unittest.TestCase):
    def run_schedule(self,samples,arrivals,config=None,**kwargs):
        clock=Clock();factory,calls=timed_factory(clock,**kwargs)
        report=run_arrivals(samples,arrivals,config or BatchConfiguration(2,100,10,(4,12)),
                            factory,clock=clock.now,wait=clock.wait)
        return report,clock,calls

    def test_public_arrivals_actual_prepare_ready_full_and_timeout_delivery(self):
        samples=[sample('a',[1,2]),sample('b',[1,40,3]),sample('c',[1,4])]
        arrivals=[dict(request_id='r1',sample_id='a',arrival_ns=0,alias='text#0'),
                  dict(request_id='r2',sample_id='b',arrival_ns=5,alias='text#1'),
                  dict(request_id='r3',sample_id='c',arrival_ns=8,alias='text#0')]
        report,clock,calls=self.run_schedule(samples,arrivals)
        self.assertEqual(report['status'],'COMPLETED')
        self.assertEqual(report['delivered'],3)
        self.assertEqual(report['factory_calls'],1)
        self.assertEqual(calls[0],'load');self.assertEqual(len(calls),2)
        self.assertEqual(report['epoch_ns'],100)
        self.assertEqual([r['request_id'] for r in report['results']],['r1','r2','r3'])
        self.assertEqual([r['ready_ns'] for r in report['results']],[2,7,14])
        self.assertEqual([r['batch_dispatch_ns'] for r in report['results']],[7,7,24])
        self.assertEqual([r['completed_ns'] for r in report['results']],[12,12,29])
        self.assertEqual([r['formation_wait_ns'] for r in report['results']],[5,0,10])
        self.assertEqual([r['end_to_end_ns'] for r in report['results']],[12,7,21])
        self.assertEqual(report['results'][2]['pre_prepare_wait_ns'],4)
        self.assertEqual([b['selection']['dispatch_reason'] for b in report['batches']],['full','timeout'])
        self.assertEqual([b['backend_metrics']['generate_calls'] for b in report['batches']],[1,1])
        self.assertEqual(len(calls[1].executor.preparations),3)
        self.assertEqual(len(calls[1].executor.model.calls),2)
        self.assertTrue(report['not_for_training'])
        self.assertFalse(report['final_gain_eligible'])

    def test_bucket_barrier_no_identity_skip_and_true_padding_cost(self):
        samples=[sample('s',[1,2]),sample('l',[1,40,40,40,40,3])]
        arrivals=[dict(request_id='long',sample_id='l',arrival_ns=0),
                  dict(request_id='short',sample_id='s',arrival_ns=0)]
        report,_,calls=self.run_schedule(samples,arrivals)
        self.assertEqual([r['request_id'] for r in report['results']],['long','short'])
        self.assertEqual(report['batches'][0]['selection']['dispatch_reason'],'fifo_barrier')
        self.assertEqual([b['backend_metrics']['padding_tokens'] for b in report['batches']],[0,0])
        self.assertEqual(len(calls[1].executor.model.calls),2)

    def test_unsorted_schedule_ties_keep_public_order_and_no_future_lookahead(self):
        samples=[sample('a',[1,2]),sample('b',[1,3])]
        arrivals=[dict(request_id='later',sample_id='a',arrival_ns=20),
                  dict(request_id='first',sample_id='b',arrival_ns=0),
                  dict(request_id='tie',sample_id='a',arrival_ns=20)]
        report,_,_=self.run_schedule(samples,arrivals)
        self.assertEqual([r['request_id'] for r in report['results']],['first','later','tie'])
        self.assertEqual(report['batches'][0]['selection']['dispatch_reason'],'timeout')
        self.assertLess(report['results'][0]['completed_ns'],20)
        self.assertEqual([r['arrival_index'] for r in report['results']],[1,0,2])

    def test_timeout_during_nonpreemptible_prepare_records_actual_lateness(self):
        samples=[sample('a',[1,2])]
        arrivals=[dict(request_id='r1',sample_id='a',arrival_ns=0),
                  dict(request_id='r2',sample_id='a',arrival_ns=0)]
        report,_,_=self.run_schedule(samples,arrivals,config=BatchConfiguration(3,100,1))
        self.assertEqual(report['results'][0]['ready_ns'],2)
        self.assertEqual(report['results'][1]['ready_ns'],4)
        self.assertEqual(report['batches'][0]['dispatch_ns'],4)
        self.assertEqual(report['batches'][0]['timeout_lateness_ns'],1)

    def test_prepare_failure_keeps_active_pending_and_future_ids(self):
        clock=Clock();factory,calls=timed_factory(clock,fail_prepare='b')
        with self.assertRaisesRegex(BatchOwnerFailure,'prepare failed original') as caught:
            run_arrivals([sample('a',[1,2]),sample('b',[1,3])],
                [dict(request_id='r1',sample_id='a',arrival_ns=0),
                 dict(request_id='r2',sample_id='b',arrival_ns=0),
                 dict(request_id='r3',sample_id='a',arrival_ns=30)],
                BatchConfiguration(2,100,10),factory,clock=clock.now,wait=clock.wait)
        report=caught.exception.report
        self.assertEqual(report['active_prepare_request_id'],'r2')
        self.assertEqual(report['pending_request_ids'],['r1'])
        self.assertEqual(report['next_arrival_request_ids'],['r3'])
        self.assertEqual(report['delivered'],0)
        self.assertEqual(report['factory_calls'],1)
        self.assertFalse(report['retry_permitted'])

    def test_generate_failure_preserves_completed_prefix_and_never_reloads(self):
        clock=Clock();factory,calls=timed_factory(clock,fail_generate_call=2)
        with self.assertRaisesRegex(BatchOwnerFailure,'generate failed original') as caught:
            run_arrivals([sample('a',[1,2])],
                [dict(request_id='r1',sample_id='a',arrival_ns=0),dict(request_id='r2',sample_id='a',arrival_ns=20)],
                BatchConfiguration(1,100,0),factory,clock=clock.now,wait=clock.wait)
        report=caught.exception.report
        self.assertEqual(report['delivered'],1)
        self.assertEqual(report['results'][0]['request_id'],'r1')
        self.assertEqual(report['pending_request_ids'],['r2'])
        self.assertEqual(report['active_batch_request_ids'],['r2'])
        self.assertEqual(report['factory_calls'],1)

    def test_parent_receipt_output_exclusive_and_failure_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);parent=path/'parent.json';parent.write_text('{}')
            receipt={'path':str(parent),'sha256':hashlib.sha256(parent.read_bytes()).hexdigest()}
            clock=Clock();factory,calls=timed_factory(clock)
            args=([sample('a',[1,2])],[dict(request_id='r',sample_id='a',arrival_ns=0)],BatchConfiguration(1,100,0),factory)
            report=run_arrivals(*args,parent_receipt=receipt,output=path/'out.json',clock=clock.now,wait=clock.wait)
            self.assertEqual(json.loads((path/'out.json').read_text()),report)
            with self.assertRaises(FileExistsError):run_arrivals(*args,output=path/'out.json')
            self.assertEqual(len(calls),2)
            parent.write_text('changed')
            with self.assertRaisesRegex(ValueError,'Parent receipt changed'):
                run_arrivals(*args,parent_receipt=receipt)

    def test_unsupported_budget_stops_and_public_validation_prevents_model_load(self):
        clock=Clock();factory,calls=timed_factory(clock)
        with self.assertRaisesRegex(BatchOwnerFailure,'PREFILL_MICROBATCH_EXECUTOR_REQUIRED'):
            run_arrivals([sample('a',[1,2,3])],[dict(request_id='r',sample_id='a',arrival_ns=0)],
                BatchConfiguration(2,2,10),factory,clock=clock.now,wait=clock.wait)
        before=len(calls)
        for rows in ([dict(request_id='r',sample_id='missing',arrival_ns=0)],
                     [dict(request_id='r',sample_id='a',arrival_ns=-1)],
                     [dict(request_id='r',sample_id='a',arrival_ns=0)]*2):
            with self.assertRaises(ValueError):
                run_arrivals([sample('a',[1,2])],rows,BatchConfiguration(1,100,0),factory)
        self.assertEqual(len(calls),before)

    def test_parent_drift_stops_before_next_prepare_and_saves_completed_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory);parent=path/'parent.json';parent.write_text('{}')
            receipt={'path':str(parent),'sha256':hashlib.sha256(parent.read_bytes()).hexdigest()}
            clock=Clock();factory,calls=timed_factory(clock)
            def mutate(event):
                if event['event']=='BATCH_COMPLETE':parent.write_text('changed')
            with self.assertRaisesRegex(BatchOwnerFailure,'Parent receipt changed') as caught:
                run_arrivals([sample('a',[1,2])],
                    [dict(request_id='first',sample_id='a',arrival_ns=0),
                     dict(request_id='second',sample_id='a',arrival_ns=20)],
                    BatchConfiguration(1,100,0),factory,parent_receipt=receipt,output=path/'failure.json',
                    event_sink=mutate,clock=clock.now,wait=clock.wait)
            report=caught.exception.report
            self.assertEqual(report['delivered'],1)
            self.assertEqual(report['active_prepare_request_id'],'second')
            self.assertEqual(len(calls[1].executor.preparations),1)
            stored=json.loads((path/'failure.json').read_text())
            self.assertEqual(stored['status'],'FAILED')
            self.assertEqual(stored['requested'],2)
            self.assertEqual(stored['delivered'],1)

    def test_explicit_retained_entry_defaults_to_no_load(self):
        with self.assertRaisesRegex(RuntimeError,'execute=True'):
            run_retained_arrivals([],[],BatchConfiguration(1,10,0),'/missing','/missing')


if __name__=='__main__':unittest.main()
