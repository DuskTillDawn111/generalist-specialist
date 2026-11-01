"""Host queue snapshots are causal, complete and independent of label fields."""
from copy import deepcopy
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import patch

from tools.queue_aware_20261023.runtime_window import queue_snapshot,choose_and_enqueue


class QueueSnapshotTests(unittest.TestCase):
    def test_worker_pipe_backlog_included_beyond_host_queue(self):
        rows={
            'a':dict(request_id='a',model='g#0',task='image',pool_id='i',status='submitted',
                     enqueued_time_ns=100,actual_submit_time_ns=110),
            'b':dict(request_id='b',model='g#0',task='text',pool_id='t',status='queued',
                     enqueued_time_ns=130,actual_submit_time_ns=None),
            'c':dict(model='g#0',status='completed',completion_time_ns=90),
            'd':dict(model='g#0',status='planned',scheduled_time_ns=2000)}
        runtime=SimpleNamespace(pending=rows)
        result=queue_snapshot(runtime,['g#0'],200)
        self.assertEqual([r['request_id'] for r in result['g#0']['outstanding']],['a','b'])
        self.assertEqual(result['g#0']['last_completed_ns'],90)
        altered=deepcopy(rows);altered['a'].update(worker_latency_ms=9000,reference='secret',worker_response={'future':'label'})
        self.assertEqual(result,queue_snapshot(SimpleNamespace(pending=altered),['g#0'],200))

    def test_future_snapshot_timestamp_rejected(self):
        r=dict(model='g#0',status='completed',completion_time_ns=201)
        with self.assertRaisesRegex(ValueError,'future'):
            queue_snapshot(SimpleNamespace(pending={'a':r}),['g#0'],200)

    def test_decision_and_enqueue_happen_under_same_runtime_lock(self):
        lock=threading.RLock();runtime=SimpleNamespace(lock=lock,pending={})
        seen=[]
        class Policy:
            aliases={'g#0':{}}
            def choose(self,task,pool_id,now,snapshot):
                self.assert_owned=lock._is_owned();seen.append(snapshot)
                return dict(alias='g#0',audit={})
        policy=Policy()
        def enqueue(row,sample):
            self.assertTrue(lock._is_owned());self.assertEqual(row['model'],'g#0')
            row.update(status='queued',enqueued_time_ns=200,actual_submit_time_ns=None)
            runtime.pending[row['request_id']]=row
        runtime.enqueue=enqueue
        row=dict(request_id='new',task='text',pool_id='t')
        with patch('tools.queue_aware_20261023.runtime_window.time.perf_counter_ns',return_value=200):
            choose_and_enqueue(runtime,policy,row,{'text':'x'})
        self.assertTrue(policy.assert_owned);self.assertEqual(len(seen),1)
        self.assertEqual(row['routing_decision']['decision_time_ns'],200)
        self.assertEqual(seen[0]['g#0']['outstanding'],[])

    def test_unknown_decision_does_not_silently_fallback_or_enqueue(self):
        class Policy:
            aliases={}
            def choose(self,*args):return dict(alias=None,status='UNKNOWN')
        called=[]
        runtime=SimpleNamespace(lock=threading.RLock(),pending={},enqueue=lambda *a:called.append(a))
        row=dict(task='text',pool_id='x')
        with self.assertRaisesRegex(RuntimeError,'NO_KNOWN'):
            choose_and_enqueue(runtime,Policy(),row,{})
        self.assertEqual(called,[])


if __name__=='__main__':unittest.main()
