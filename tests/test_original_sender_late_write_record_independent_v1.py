"""Exercise original sender/collector concurrently; no worker/GPU startup."""
import queue
import threading
import time
import unittest
from types import SimpleNamespace

from gsdeploy.runtime.deployment import DeploymentRuntime


def source_counterexample():
    complete=threading.Event();runtime=DeploymentRuntime(SimpleNamespace(),{})
    owner='text#0';runtime.queues[owner]=queue.Queue();runtime.send_locks[owner]=threading.Lock()
    class Record(dict):
        def update(self,*args,**kwargs):
            super().update(*args,**kwargs)
            if self.get('status')=='completed':complete.set()
    record=Record(request_id='public:fast-response',model=owner,task='text',pool_id='exact-input',
        sample_id='source-control',arrival_index=0,scheduled_time_ns=time.perf_counter_ns(),status='planned')
    class CPUTransport:
        def send(self,payload):
            start=time.perf_counter_ns()
            runtime.events.put((owner,'START',dict(request_id=record['request_id'],perf_ns=start)))
            end=time.perf_counter_ns()
            runtime.events.put((owner,'RESULT',dict(request_id=record['request_id'],task_type='text',
                start_perf_ns=start,end_perf_ns=end,success=True,worker_latency_ms=0,queue_wait_ms=0)))
            # A legal send/record scheduling delay. Original sender writes its
            # completion record only after send returns and acquiring its lock.
            if not complete.wait(2):raise RuntimeError('Original collector did not process source-controlled CPU RESULT')
            runtime.stopping.set();runtime.collect_stop.set()
    runtime.workers[owner]=CPUTransport()
    runtime.enqueue(record,dict(context='public',question='q'))
    collector=threading.Thread(target=runtime._collect,daemon=True)
    sender=threading.Thread(target=runtime._send,args=(owner,),daemon=True)
    collector.start();sender.start()
    sender.join(3);runtime.stopping.set();runtime.collect_stop.set();collector.join(3)
    if sender.is_alive() or collector.is_alive():raise RuntimeError('CPU fixture threads did not terminate')
    return dict(record),runtime.errors


class OriginalLateWriteRecordTests(unittest.TestCase):
    def test_original_runtime_allows_RESULT_before_write_completion_record(self):
        row,errors=source_counterexample()
        self.assertEqual(errors,[]);self.assertEqual(row['status'],'completed')
        self.assertLessEqual(row['actual_submit_time_ns'],row['observed_worker_start_time_ns'])
        self.assertLessEqual(row['observed_worker_start_time_ns'],row['worker_response']['end_perf_ns'])
        self.assertLessEqual(row['worker_response']['end_perf_ns'],row['completion_time_ns'])
        self.assertLessEqual(row['completion_time_ns'],row['response_collector_dequeued_time_ns'])
        self.assertLessEqual(row['response_collector_dequeued_time_ns'],row['response_collector_processing_time_ns'])
        self.assertLessEqual(row['actual_submit_time_ns'],row['submit_write_completed_time_ns'])
        self.assertLess(row['completion_time_ns'],row['submit_write_completed_time_ns'])
        self.assertLess(row['response_collector_processing_time_ns'],row['submit_write_completed_time_ns'])


if __name__=='__main__':unittest.main()
