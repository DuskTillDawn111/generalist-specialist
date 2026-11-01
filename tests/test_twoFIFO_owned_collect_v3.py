import copy
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v3 as m
from tools.predictor_validation_20261027 import twoFIFO_training_semantic_preflight_v2 as strict
from tests.test_twoFIFO_owned_collect_v2 import Clock, FakeRuntime, FakeMonitor


class CollectorV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base,cls.entry=strict.load_frozen()

    def test_slow_full_checkpoint_complete_path_has_no_protected_copy_or_write(self):
        clock=Clock();runtime=FakeRuntime(clock);context=m.read(strict.CONTEXT)
        plan=dict(attempt_id='training-block-1-shard-1',capture_id='training-block-1',execution_id='0'*32,
            execution_context_sha256='1'*64,context=m.receipt(strict.CONTEXT),sources=[],source_freeze=[],
            priming=m.priming_catalog(context),cases=copy.deepcopy(self.base['capture_blocks'][0]['cases'][:2]))
        writes=[];original_save=m.atomic_save
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);pool=directory/'pool.jsonl'
            pool.write_text(''.join(json.dumps(row)+'\n' for row in strict.source_catalog(context).values()))
            plan['prepared_pool']=m.receipt(pool);path=directory/'plan.json';path.write_text(json.dumps(plan))
            ownership=lambda values,*a,**k:dict(owned_aliases=[m.OWNERS[next(iter(values))-1]],
                unresolved_nvml_pids=[],foreign_pids=[],ownership_evidence=[{'live':True}])
            def slow_save(target,value):
                start=clock.clock();clock.now+=500_000_000
                original_save(target,value);writes.append((Path(target).name,start,clock.clock()))
            def start():
                saved=m.read(directory/'raw.json')
                self.assertTrue(saved['all_offered_registered_before_startup'])
                self.assertEqual(saved['public_case_calendars'],plan['cases'])
                self.assertEqual(saved['priming_public_calendar'],plan['priming'])
                self.assertTrue(all(r['status']=='not_started' for r in saved['request_records']))
            runtime.start=start
            with patch.object(m.supervisor,'validate_lease'),patch.object(m,'audit_plan',return_value=plan),patch.object(m,'atomic_save',side_effect=slow_save):
                raw=m.collect(path,directory,lease_fd=-1,runtime_factory=lambda *a:runtime,
                    monitor_factory=FakeMonitor,clock_ns=clock.clock,wait_until_ns=clock.wait,ownership_check=ownership)
            saved=m.read(directory/'raw.json');ledger=m.read(directory/m.ACTIVITY_FILE)
        self.assertEqual(raw['status'],'completed');self.assertEqual(raw,saved)
        self.assertEqual(raw['kind'],'CPU_TEST_ONLY');self.assertEqual(raw['backend'],'CPU_TEST_ONLY')
        self.assertEqual(ledger['execution_id'],raw['execution_id']);self.assertTrue(ledger['all_spans_complete'])
        self.assertEqual([s['span_id'] for s in ledger['checkpoint_spans']],list(range(len(ledger['checkpoint_spans']))))
        self.assertEqual(len(raw['priming_request_records']),18);self.assertEqual(len(raw['request_records']),9)
        self.assertEqual(len(raw['case_histories'][0]['prior_history_snapshot']['request_records']),18)
        self.assertEqual(len(raw['case_histories'][1]['prior_history_snapshot']['request_records']),21)
        self.assertEqual([p['public_arrivals'] for p in raw['case_preparations']],[c['arrivals'] for c in plan['cases']])
        # Snapshots are detached from rows subsequently changed by runtime.
        history=copy.deepcopy(raw['case_histories']);raw['priming_request_records'][0]['status']='changed'
        self.assertEqual(raw['case_histories'],history)
        for window in raw['measurement_windows']:
            a,b=window['start_ns'],window['fixed_drain_deadline_ns']
            self.assertEqual(b-a,m.MEASUREMENT_NS+m.DRAIN_NS)
            for span in ledger['checkpoint_spans']:
                self.assertTrue(span['end_ns']<a or span['start_ns']>b,(window,span))
            for name,start,end in writes:
                self.assertTrue(end<a or start>b,(window,name,start,end))
            rows=[r for r in saved['request_records'] if r['case_id']==window['case_id']]
            self.assertEqual([r['actual_submit_time_ns']-a for r in rows],[r['arrival_ns'] for r in rows])
        # Seven full snapshots: before startup, after all priming, pre/post x2,
        # and shutdown. There is none between eighteen priming requests.
        stages=[s['stage'] for s in ledger['checkpoint_spans'] if s['kind']=='checkpoint']
        self.assertEqual(stages,['before_startup','postpriming','precase','postdrain','precase','postdrain','shutdown'])
        self.assertTrue(runtime.closed);self.assertEqual(saved['board_samples'][-1]['phase'],'final_drain')

    def test_real_threads_slow_locked_checkpoint_cannot_block_notifications_or_monitor_during_measurement_or_drain(self):
        class RealClock:
            clock=staticmethod(time.perf_counter_ns)
        clock=RealClock();runtime=FakeRuntime(clock);monitor=FakeMonitor(runtime)
        case=copy.deepcopy(self.base['capture_blocks'][0]['cases'][1])
        # Preserve every public request and the original per-owner order while
        # compressing time for a CPU-only concurrency test.
        for index,a in enumerate(case['arrivals']):a['arrival_ns']=index*2_000_000
        case['public_arrivals_sha256']=m.digest(case['arrivals'])
        plan=dict(capture_id='training-block-1',attempt_id='attempt',cases=[case])
        pool={r['pool_id']:dict(pool_id=r['pool_id'],sample={}) for r in case['arrivals']}
        raw=dict(request_records=m.initial_rows(plan,pool),priming_request_records=[],measurement_windows=[],
            cases=[dict(case_id=case['case_id'],status='NOT_STARTED')],case_histories=[])
        stop=threading.Event();notifications=[];samples=[];checkpoints=[];drains=[]
        def notification_loop():
            while not stop.is_set():
                with runtime.lock:notifications.append(clock.clock())
                time.sleep(.002)
        def monitor_loop():
            while not stop.is_set():
                with monitor._sample_lock:samples.append(clock.clock())
                time.sleep(.002)
        def checkpoint():
            # Reproduce the old problematic lock structure with a slow copy.
            # It must run exclusively outside both protected phases.
            start=clock.clock()
            with runtime.lock,monitor._sample_lock:time.sleep(.10)
            checkpoints.append((start,clock.clock(),copy.deepcopy(raw)))
        def wait(target):
            while clock.clock()<target:time.sleep(min(.001,(target-clock.clock())/1e9))
        original_drain=runtime.drain
        def drain(rows,timeout):
            drains.append((clock.clock(),timeout));time.sleep(.03);original_drain(rows,timeout)
        runtime.drain=drain
        threads=[threading.Thread(target=notification_loop),threading.Thread(target=monitor_loop)]
        for t in threads:t.start()
        try:
            with patch.object(m,'MEASUREMENT_NS',100_000_000),patch.object(m,'DRAIN_NS',200_000_000):
                result=m.dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock.clock,wait_until_ns=wait,
                    checkpoint=checkpoint,monitor=monitor)
        finally:
            stop.set()
            for t in threads:t.join(timeout=2)
        self.assertEqual(result['status'],'completed');self.assertEqual(len(runtime.enqueued),len(case['arrivals']))
        a=result['case_epoch_ns'];middle=result['measurement_end_ns'];b=result['fixed_drain_deadline_ns']
        self.assertEqual(len(checkpoints),2)
        self.assertLess(checkpoints[0][1],a);self.assertGreater(checkpoints[1][0],b)
        self.assertEqual(len(checkpoints[0][2]['request_records']),len(case['arrivals']))
        self.assertTrue(all(r['status']=='not_started' for r in checkpoints[0][2]['request_records']))
        self.assertLessEqual(middle,drains[0][0]);self.assertGreater(drains[0][1],.15)
        self.assertGreaterEqual(result['observation_finalized_ns'],b)
        for stamps in (notifications,samples):
            for left,right in ((a,middle),(middle,b)):
                inside=[left]+[x for x in stamps if left<=x<=right]+[right]
                self.assertGreater(len(inside),8)
                self.assertLess(max(y-x for x,y in zip(inside,inside[1:])),70_000_000)
        for row in raw['request_records']:
            self.assertLess(row['actual_submit_time_ns']-row['scheduled_time_ns'],50_000_000)

    def test_prepare_freezes_new_schema_with_unchanged_budget_and_exact_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            output=Path(directory)/'campaign';campaign=m.prepare(output)
            self.assertEqual(campaign['schema'],m.SCHEMA);self.assertEqual(m.audit(output/'campaign.json'),campaign)
            plans=[m.audit_plan(r['path']) for r in campaign['attempts']]
            self.assertEqual([len(p['cases']) for p in plans],[13,13,13,11]*3)
            self.assertEqual(sum(c['offered'] for p in plans for c in p['cases']),546)
            self.assertEqual(sum(len(p['priming']) for p in plans),216)
            self.assertTrue(all(p['supervisor_budget']==m.supervisor.BUDGET and p['preflight_free_bytes_required']==m.supervisor.PREFLIGHT for p in plans))
            self.assertTrue(all(p['collector_timing_policy']['no_mutating_checkpoint_during_case'] for p in plans))
            self.assertEqual(len({p['execution_id'] for p in plans}),12)

    def test_activity_overlap_and_incomplete_span_fail_closed(self):
        ledger=m.ActivityLedger('/unused','execution',lambda:0)
        raw=dict(measurement_windows=[dict(start_ns=100,fixed_drain_deadline_ns=200)])
        for start,end in ((99,100),(200,201),(101,102),(1,None)):
            ledger.spans=[dict(span_id=0,start_ns=start,end_ns=end)]
            with self.assertRaises(ValueError):m.audit_activity(raw,ledger)
        ledger.spans=[dict(span_id=0,start_ns=1,end_ns=99),dict(span_id=1,start_ns=201,end_ns=202)]
        m.audit_activity(raw,ledger)


if __name__=='__main__':unittest.main()
