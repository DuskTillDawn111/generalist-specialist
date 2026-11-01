import copy
import json
from pathlib import Path
import queue
import tempfile
import threading
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import twoFIFO_owned_collect_v2 as m
from tools.predictor_validation_20261027 import twoFIFO_training_semantic_preflight_v2 as strict


class Clock:
    now=100
    def clock(self):return self.now
    def wait(self,deadline):self.now=max(self.now,deadline)


class FakeRuntime:
    def __init__(self,clock,fail_case=False):
        self.clock=clock;self.lock=threading.RLock();self.events=queue.Queue();self.pending={};self.errors=[]
        self.ready={alias:dict(pid=i+1) for i,alias in enumerate(m.OWNERS)};self.shutdown_report={}
        self.enqueued=[];self.drain_counts=[];self.fail_case=fail_case;self.closed=False
    def start(self):pass
    def enqueue(self,row,sample):
        self.pending[row['request_id']]=row;self.enqueued.append(row['request_id'])
        row.update(status='submitted',actual_submit_time_ns=self.clock.clock())
    def drain(self,rows,timeout):
        self.drain_counts.append(len(self.enqueued))
        for row in rows:
            if row['status']!='submitted':continue
            if self.fail_case and row['phase']=='profile':row.update(status='incomplete');continue
            row.update(status='completed',completion_time_ns=self.clock.clock())
            row['previous_handler_next_request_ready']=dict(perf_ns=self.clock.clock()+1)
    def close(self):self.closed=True


class FakeMonitor:
    def __init__(self,runtime):
        self.runtime=runtime;self.samples=[];self.errors=[];self._sample_lock=threading.RLock()
        self.collector_pid=1;self.process_group=1;self.ownership=object();self.phase='startup'
    def sample(self):
        assert not self.runtime.closed
        self.samples.append(dict(phase=self.phase))
    def check(self):pass
    def start(self):pass
    def set_phase(self,phase):self.phase=phase
    def close(self):self.set_phase('final_drain');self.sample()


class CollectorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.base,cls.entry=strict.load_frozen()

    def test_shards_preserve_every_case_once_and_balance_positions(self):
        blocks=m.shard_cases(self.base,self.entry)
        positions=[]
        for block in blocks:
            rows=[c for shard in block['shards'] for c in shard]
            self.assertEqual(len(block['shards']),4);self.assertLessEqual(max(map(len,block['shards'])),16)
            self.assertEqual(len(rows),50);self.assertEqual(len({r['case_id'] for r in rows}),50)
            self.assertEqual(sum(r['offered'] for r in rows),182)
            positions.append([r['case_id'] for r in rows])
        self.assertNotEqual(positions[0],positions[1]);self.assertNotEqual(positions[1],positions[2])
        self.assertEqual(len(m.priming_catalog(m.read(strict.CONTEXT))),18)

    def collect_fake(self,fail_case=False):
        clock=Clock();runtime=FakeRuntime(clock,fail_case);context=m.read(strict.CONTEXT)
        plan=dict(attempt_id='training-block-1-shard-1',capture_id='training-block-1',execution_id='0'*32,
            execution_context_sha256='1'*64,context=m.receipt(strict.CONTEXT),
            prepared_pool=self.base['prepared_pool'],sources=[],source_freeze=[],
            priming=m.priming_catalog(context),cases=copy.deepcopy(self.base['capture_blocks'][0]['cases'][:2]))
        # Original priming includes image/asr, so use actual source catalog.
        full_pool=strict.source_catalog(context)
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);pool=directory/'pool.jsonl'
            pool.write_text(''.join(json.dumps(row)+'\n' for row in full_pool.values()))
            plan['prepared_pool']=m.receipt(pool);path=directory/'plan.json';path.write_text(json.dumps(plan))
            ownership=lambda values,*a,**k:dict(owned_aliases=[m.OWNERS[next(iter(values))-1]],
                unresolved_nvml_pids=[],foreign_pids=[],ownership_evidence=[{'live':True}])
            with patch.object(m.supervisor,'validate_lease'),patch.object(m,'audit_plan',return_value=plan):
                raw=m.collect(path,directory,lease_fd=-1,runtime_factory=lambda *a:runtime,
                    monitor_factory=FakeMonitor,clock_ns=clock.clock,wait_until_ns=clock.wait,ownership_check=ownership)
            saved=m.read(directory/'raw.json')
        return raw,saved,runtime,clock

    def test_complete_control_path_primes18_independent_enqueue_fixed_cutoffs_history_and_cleanup(self):
        raw,saved,runtime,clock=self.collect_fake()
        self.assertEqual(raw['status'],'completed');self.assertEqual(saved,raw)
        self.assertEqual(len(raw['priming_request_records']),18)
        self.assertEqual(runtime.drain_counts[18],21) # all three offered, then first case drain
        self.assertEqual(clock.now,140_000_000_100)
        self.assertEqual(len(raw['case_histories'][0]['prior_history_snapshot']['request_records']),18)
        self.assertEqual(len(raw['case_histories'][1]['prior_history_snapshot']['request_records']),21)
        self.assertTrue(raw['case_histories'][0]['prior_history_snapshot']['request_records'][-1]['previous_handler_next_request_ready'])
        self.assertTrue(runtime.closed);self.assertEqual(raw['board_samples'][-1]['phase'],'final_drain')
        self.assertEqual(raw['lifecycle_boundaries'][-2]['boundary'],'shutdown_begin')

    def test_incomplete_keeps_all_future_offered_and_cleans_up(self):
        raw,_,runtime,_=self.collect_fake(fail_case=True)
        self.assertEqual(raw['status'],'failed');self.assertTrue(runtime.closed)
        self.assertEqual(raw['cases'][1]['status'],'NOT_STARTED')
        self.assertEqual(len(raw['request_records']),9)
        self.assertEqual(sum(r['status']=='not_started' for r in raw['request_records']),6)
        self.assertFalse(raw['training_feature_extraction_eligible'])

    def test_late_dispatch_keeps_original_deadline_and_all_unsubmitted(self):
        clock=Clock();runtime=FakeRuntime(clock);monitor=FakeMonitor(runtime)
        case=copy.deepcopy(self.base['capture_blocks'][0]['cases'][1])
        plan=dict(capture_id='training-block-1',attempt_id='attempt',cases=[case])
        pool={r['pool_id']:dict(pool_id=r['pool_id'],sample={}) for r in case['arrivals']}
        raw=dict(request_records=m.initial_rows(plan,pool),priming_request_records=[],measurement_windows=[],
            cases=[dict(case_id=case['case_id'],status='NOT_STARTED')],case_histories=[])
        original_enqueue=runtime.enqueue
        def enqueue(row,sample):
            original_enqueue(row,sample);clock.now=80_000_000_100
        runtime.enqueue=enqueue;snapshots=[]
        def checkpoint():
            with runtime.lock:snapshots.append(copy.deepcopy(raw))
        result=m.dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock.clock,wait_until_ns=clock.wait,
            checkpoint=checkpoint,monitor=monitor)
        self.assertEqual(result['fixed_drain_deadline_ns'],70_000_000_100)
        self.assertEqual(len(runtime.enqueued),1)
        self.assertEqual(len(raw['request_records']),6)
        self.assertTrue(all(r['status']=='incomplete' for r in raw['request_records'][1:]))
        self.assertEqual(len(snapshots[0]['request_records']),6)
        self.assertTrue(all(r['status']=='not_started' for r in snapshots[0]['request_records']))
        self.assertFalse(raw['request_records'][0]['RESULT_reader_within_fixed_drain'])

    def test_rehashed_semantic_mutants_rejected(self):
        mutations=[lambda b,e:b['capture_blocks'][1].update(capture_id='training-block-1'),
            lambda b,e:b['inputs']['text']['long'].update(exact_input_sha256='0'*64),
            lambda b,e:e.update(measurement_ns=20_000_000_000),
            lambda b,e:e['prepared_pool'].update(sha256='0'*64),
            lambda b,e:e['capture_blocks'][0]['cases'][1]['arrivals'][0].update(owner='unresident',task='image',pool_id='nonpublic',arrival_index=99),
            lambda b,e:b['capture_blocks'][1]['cases'][0]['arrivals'][0].update(arrival_ns=123),
            lambda b,e:b['capture_blocks'][0]['cases'].pop(),
            lambda b,e:b['capture_blocks'][0]['cases'].__setitem__(0,copy.deepcopy(b['capture_blocks'][0]['cases'][1]))]
        for mutation in mutations:
            b,e=copy.deepcopy(self.base),copy.deepcopy(self.entry);mutation(b,e)
            for design in (b,e):
                for block in design['capture_blocks']:
                    for case in block['cases']:case['public_arrivals_sha256']=m.digest(case['arrivals'])
            with self.assertRaises(ValueError):strict.validate(b,e)

    def test_unplanned_shard_aliases_rejected_even_without_campaign_audit(self):
        plan=dict(schema=m.SCHEMA+'.attempt',split='training',capture_id='training-block-1',
            supervisor_budget=m.supervisor.BUDGET,preflight_free_bytes_required=m.supervisor.PREFLIGHT,
            runtime_config=m.runtime_config().as_dict(),runtime_policy=self.base['runtime_policy'],
            base_manifest=m.receipt(strict.BASE),OCR_entry_manifest=m.receipt(strict.ENTRY))
        for index in (0,-1,True,5):
            with patch.object(m,'read',return_value=dict(plan,shard_index=index)):
                with self.assertRaisesRegex(ValueError,'one-based shard index'):m.audit_plan('unused')

    def test_monitor_failure_still_closes_original_runtime(self):
        clock=Clock();runtime=FakeRuntime(clock);monitor=FakeMonitor(runtime)
        def close():raise RuntimeError('injected monitor failure')
        monitor.close=close;raw=dict(status='completed',errors=[],lifecycle_boundaries=[]);checkpoints=[]
        m.finish_collection(runtime,monitor,raw,lambda:checkpoints.append(copy.deepcopy(raw)),clock.clock)
        self.assertTrue(runtime.closed);self.assertEqual(raw['status'],'failed')
        self.assertEqual(checkpoints[-1]['lifecycle_boundaries'][-1]['boundary'],'shutdown_end')

    def test_default_wait_handles_monitor_crossing_deadline(self):
        ticks=iter((99,101));sleeps=[]
        monitor=FakeMonitor(FakeRuntime(Clock()))
        m.wait_until(100,lambda:next(ticks),monitor,sleeps.append)
        self.assertEqual(sleeps,[])

    def test_slow_disk_checkpoint_is_outside_public_arrival_sequence(self):
        clock=Clock();runtime=FakeRuntime(clock);monitor=FakeMonitor(runtime)
        case=copy.deepcopy(self.base['capture_blocks'][0]['cases'][1])
        plan=dict(capture_id='training-block-1',attempt_id='attempt',cases=[case])
        pool={r['pool_id']:dict(pool_id=r['pool_id'],sample={}) for r in case['arrivals']}
        raw=dict(request_records=m.initial_rows(plan,pool),priming_request_records=[],measurement_windows=[],
            cases=[dict(case_id=case['case_id'],status='NOT_STARTED')],case_histories=[])
        def checkpoint():clock.now+=500_000_000
        case_record=m.dispatch_case(plan,case,runtime,pool,raw,clock_ns=clock.clock,
            wait_until_ns=clock.wait,checkpoint=checkpoint,monitor=monitor)
        self.assertEqual(case_record['case_epoch_ns'],500_000_100)
        self.assertEqual([r['actual_submit_time_ns']-case_record['case_epoch_ns'] for r in raw['request_records']],
            [a['arrival_ns'] for a in case['arrivals']])


if __name__=='__main__':unittest.main()
