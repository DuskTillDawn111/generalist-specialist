"""CPU adversarial review using the original collector's real queue thread."""
import copy
import threading
import time
import unittest
from unittest.mock import patch

from gsdeploy.runtime.deployment import DeploymentRuntime
from tests.test_twoFIFO_owned_collect_v2 import FakeRuntime,FakeMonitor
from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v2 as old
from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v3 as new

RESULTS={}


def exercise(module):
    class Clock:
        clock=staticmethod(time.perf_counter_ns)
    clock=Clock();runtime=FakeRuntime(clock);monitor=FakeMonitor(runtime)
    runtime.collect_stop=threading.Event();runtime.admission=None
    probe=dict(request_id='cpu-review-probe',model='text#0',status='submitted')
    runtime.pending[probe['request_id']]=probe
    base,_=new.strict.load_frozen();case=copy.deepcopy(base['capture_blocks'][0]['cases'][1])
    for i,a in enumerate(case['arrivals']):a['arrival_ns']=i*2_000_000
    case['public_arrivals_sha256']=new.digest(case['arrivals'])
    plan=dict(capture_id='training-block-1',attempt_id='CPU_ONLY',cases=[case])
    pool={a['pool_id']:dict(pool_id=a['pool_id'],sample={}) for a in case['arrivals']}
    raw=dict(request_records=new.initial_rows(plan,pool),priming_request_records=[],measurement_windows=[],
        cases=[dict(case_id=case['case_id'],status='NOT_STARTED')],case_histories=[])
    stop=threading.Event();samples=[];spans=[]
    def producer():
        while not stop.is_set():
            now=clock.clock()
            runtime.events.put(('text#0','PHASE',dict(request_id=probe['request_id'],perf_ns=now),now,time.time_ns()))
            time.sleep(.002)
    def sampler():
        while not stop.is_set():
            with monitor._sample_lock:samples.append(clock.clock())
            time.sleep(.002)
    def checkpoint():
        begin=clock.clock()
        with runtime.lock,monitor._sample_lock:
            time.sleep(.10)
            copy.deepcopy(raw)
        # Delayed serialization outside the runtime lock also belongs to the
        # protected full span, since it consumes CPU/GIL in the old path.
        time.sleep(.04)
        spans.append((begin,clock.clock()))
    def wait(target):
        while clock.clock()<target:time.sleep(min(.001,(target-clock.clock())/1e9))
    threads=[threading.Thread(target=DeploymentRuntime._collect,args=(runtime,)),
        threading.Thread(target=producer),threading.Thread(target=sampler)]
    for t in threads:t.start()
    try:
        with patch.object(module,'MEASUREMENT_NS',120_000_000),patch.object(module,'DRAIN_NS',220_000_000):
            case_result=module.dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock.clock,
                wait_until_ns=wait,checkpoint=checkpoint,monitor=monitor)
    finally:
        stop.set()
        for t in threads[1:]:t.join(timeout=2)
        runtime.collect_stop.set();threads[0].join(timeout=2)
    assert not any(t.is_alive() for t in threads)
    assert not runtime.errors
    a=case_result['case_epoch_ns'];middle=case_result['measurement_end_ns'];b=case_result['fixed_drain_deadline_ns']
    measured=[e for e in probe.get('execution_events',[]) if a<=e['received_time_ns']<=b]
    gaps={}
    for phase,left,right in [('measurement',a,middle),('drain',middle,b)]:
        points=[left]+[x for x in samples if left<=x<=right]+[right]
        gaps[phase]=max(y-x for x,y in zip(points,points[1:]))
    return dict(case_status=case_result['status'],measurement_start_ns=a,measurement_end_ns=middle,
        fixed_drain_deadline_ns=b,checkpoint_spans=spans,protected_messages=len(measured),
        max_protected_reader_to_processing_ns=max(e['collector_processing_time_ns']-e['received_time_ns'] for e in measured),
        max_sampler_gap_ns=gaps,protected_fullspan_overlaps=sum(not(end<a or begin>b) for begin,end in spans),
        all_current_rows=len(raw['request_records']),queued_current_rows=len(runtime.enqueued),
        source='original_DeploymentRuntime._collect_actual_queue_thread_CPU_ONLY_NOT_NATIVE')


class IndependentConcurrencyReview(unittest.TestCase):
    def test_real_collector_detects_old_failure_and_new_protected_path(self):
        prior=exercise(old);current=exercise(new);RESULTS.update(v2=prior,v3=current)
        self.assertEqual(prior['protected_fullspan_overlaps'],1)
        self.assertGreater(prior['max_protected_reader_to_processing_ns'],70_000_000)
        self.assertGreater(prior['max_sampler_gap_ns']['measurement'],70_000_000)
        self.assertEqual(current['protected_fullspan_overlaps'],0)
        self.assertEqual(current['case_status'],'completed')
        self.assertEqual(current['all_current_rows'],current['queued_current_rows'])
        self.assertGreater(current['protected_messages'],70)
        self.assertLess(current['max_protected_reader_to_processing_ns'],70_000_000)
        self.assertLess(max(current['max_sampler_gap_ns'].values()),70_000_000)
        self.assertEqual(current['fixed_drain_deadline_ns']-current['measurement_start_ns'],340_000_000)
