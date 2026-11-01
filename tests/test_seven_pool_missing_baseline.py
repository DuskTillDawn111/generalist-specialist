"""CPU behavioral tests for bounded native discovery transport and provenance."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import threading
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'tools/seven_pool_baseline_20261022'


def load(name):
    spec=importlib.util.spec_from_file_location('_test_seven_'+name,DIRECTORY/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


run=load('run');prepare=load('prepare')


class Clock:
    def __init__(self):self.now=1_000_000_000;self.runtime=None
    def perf_counter_ns(self):return self.now
    def sleep(self,seconds):
        self.now+=round(seconds*1e9)
        for row,due in list(self.runtime.pending):
            if due<=self.now:
                row.update(status='error' if row['request_id'] in self.runtime.fail else 'completed',completion_time_ns=due)
                self.runtime.pending.remove((row,due))


class FakeRuntime:
    def __init__(self,clock,durations=None,fail=()):
        self.lock=threading.RLock();self.clock=clock;clock.runtime=self;self.pending=[];self.log=[]
        self.durations=durations or {};self.fail=set(fail)
    def enqueue(self,row,sample):
        if any(r['model']==row['model'] for r,_ in self.pending):raise AssertionError('More than one outstanding per alias')
        row.update(status='submitted',actual_submit_time_ns=self.clock.now)
        duration=self.durations.get((row['model'],row.get('role')),self.durations.get(row['model'],.002))
        self.pending.append((row,self.clock.now+round(duration*1e9)));self.log.append((row['request_id'],self.clock.now))


class Monitor:
    def check(self):pass


def fixture(a=1,b=3,support_count=20):
    primary={};support={};samples={}
    for alias,count in [('a',a),('b',b)]:
        primary[alias]=[dict(request_id=f'p:{alias}:{i}',task='text',model=alias,pool_id=f'p{i}',role='primary',arrival_index=i) for i in range(count)]
        support[alias]=[dict(request_id=f's:{alias}:{i}',task='text',model=alias,pool_id=f'p{i}',role='support',arrival_index=100+i) for i in range(support_count)]
        samples.update({r['pool_id']:{'sample':{}} for r in primary[alias]+support[alias]})
    phase=dict(task='text',primary_requests=primary,support_requests=support,measurement_timeout_seconds=600,
        warmup_requests={alias:[dict(task='text',model=alias,pool_id='p0')] for alias in primary},warmup_seconds=.006,warmup_timeout_seconds=.05)
    records=[run.record(r,'measurement') for rows in primary.values() for r in rows]
    return phase,records,samples


class ReleaseContracts(unittest.TestCase):
    def invoke(self,phase,records,samples,durations=None,**kwargs):
        clock=Clock();runtime=FakeRuntime(clock,durations)
        with patch.object(run.time,'perf_counter_ns',clock.perf_counter_ns),patch.object(run.time,'sleep',clock.sleep):
            meta=run.finite_with_support(runtime,phase,samples,records,Monitor(),**kwargs)
        return runtime,meta

    def test_support_starts_after_own_primary_and_stops_at_last_primary(self):
        phase,records,samples=fixture();runtime,meta=self.invoke(phase,records,samples)
        self.assertEqual(meta['primary_completed_by_alias'],{'a':1,'b':3})
        self.assertEqual(meta['support_released_by_alias'],{'a':2,'b':0})
        self.assertTrue(all(r['task']=='text' for r in records))
        self.assertTrue(all(r['status']=='completed' for r in records))
        self.assertLess(max(t for name,t in runtime.log if name.startswith('s:')),meta['support_admission_stop_time_ns'])

    def test_support_cap_invalidates_instead_of_accepting_tail(self):
        phase,records,samples=fixture(b=20,support_count=1)
        with self.assertRaisesRegex(RuntimeError,'Support cap exhausted'):self.invoke(phase,records,samples)
        self.assertTrue(any(r['status']=='planned' for r in records))

    def test_support_drain_inside_same_hard_wall(self):
        phase,records,samples=fixture(b=1);phase['measurement_timeout_seconds']=.01
        meta={}
        with self.assertRaisesRegex(TimeoutError,'wall timeout'):
            self.invoke(phase,records,samples,{('a','primary'):.001,('b','primary'):.005,('a','support'):.02},metadata=meta)
        self.assertIsNotNone(meta['primary_completion_time_ns'])
        self.assertEqual(meta['primary_completed_by_alias'],{'a':1,'b':1})
        self.assertTrue(any(r['role']=='support' and r['status']=='submitted' for r in records))

    def test_snapshot_work_is_charged_to_hard_wall(self):
        phase,records,samples=fixture();clock=Clock();runtime=FakeRuntime(clock)
        def progress():clock.now+=601_000_000_000
        with patch.object(run.time,'perf_counter_ns',clock.perf_counter_ns),patch.object(run.time,'sleep',clock.sleep),self.assertRaisesRegex(TimeoutError,'snapshots'):
            run.finite_with_support(runtime,phase,samples,records,Monitor(),progress=progress)

    def test_native_error_does_not_release_more_or_hide_planned_rows(self):
        phase,records,samples=fixture();clock=Clock();runtime=FakeRuntime(clock,fail=['p:a:0'])
        with patch.object(run.time,'perf_counter_ns',clock.perf_counter_ns),patch.object(run.time,'sleep',clock.sleep),self.assertRaisesRegex(RuntimeError,'Native request failure'):
            run.finite_with_support(runtime,phase,samples,records,Monitor())
        self.assertEqual(len(runtime.log),2)
        self.assertTrue(all(r['scheduled_time_ns'] is None for r in records if r['status']=='planned'))

    def test_unregistered_primary_rejected_before_release(self):
        phase,records,samples=fixture();records.pop()
        with self.assertRaisesRegex(ValueError,'preregistration'):self.invoke(phase,records,samples)

    def test_warmup_uses_separate_cursor_and_drains(self):
        phase,_,samples=fixture();records=[];clock=Clock();runtime=FakeRuntime(clock,{'a':.004,'b':.004})
        with patch.object(run.time,'perf_counter_ns',clock.perf_counter_ns),patch.object(run.time,'sleep',clock.sleep):
            meta=run.warmup(runtime,phase,samples,records,Monitor())
        self.assertTrue(all(r['status']=='completed' for r in records));self.assertEqual(len(records),4)
        self.assertGreaterEqual(meta['duration_seconds'],.008)
        self.assertTrue(all(r['request_id'].startswith('warmup:') for r in records))

    def test_completed_request_over_limit_still_fails(self):
        row=dict(request_id='slow',status='completed',scheduled_time_ns=0,completion_time_ns=301_000_000_000)
        with self.assertRaisesRegex(TimeoutError,'Request timeout'):run.validate_response(row,302_000_000_000,300)


class SafeMemoryContracts(unittest.TestCase):
    def monitor(self):
        class Process:pid=11
        class Worker:process=Process()
        class Runtime:workers={'a':Worker()};errors=[]
        return run.SafeOwnedMemory(Runtime(),{'device':'0','gpu_memory_mib':100,'safe_factor':1.1})

    def sample(self,monitor,stdout):
        with patch.object(run.subprocess,'run',return_value=subprocess.CompletedProcess([],0,stdout,'')):monitor.sample()

    def test_safe_factor_enforced_and_peak_failure_latches(self):
        monitor=self.monitor();self.sample(monitor,'11, 95\n');self.sample(monitor,'11, 50\n')
        with self.assertRaisesRegex(RuntimeError,'safe_factor'):monitor.check()
        self.assertEqual(len(monitor.samples),2)

    def test_missing_startup_pid_recorded_but_missing_ready_pid_fails(self):
        monitor=self.monitor();self.sample(monitor,'');monitor.check()
        self.assertEqual(monitor.samples[-1]['missing_owned_pids'],[11])
        monitor.ready=True;self.sample(monitor,'')
        with self.assertRaisesRegex(RuntimeError,'missing after READY'):monitor.check()

    def test_foreign_pid_invalidates_even_below_budget(self):
        monitor=self.monitor();self.sample(monitor,'11, 1\n12, 1\n')
        with self.assertRaisesRegex(RuntimeError,'Foreign'):monitor.check()

    def test_nan_observation_is_failure(self):
        monitor=self.monitor();self.sample(monitor,'11, nan\n')
        with self.assertRaisesRegex(RuntimeError,'Invalid process memory'):monitor.check()
        self.assertEqual(monitor.samples,[])


class FrozenPreparationContracts(unittest.TestCase):
    def test_original_pool_global_three_observations_and_full_priming(self):
        original=json.loads(prepare.ORIGINAL.read_text());samples=Path(original['inputs'])/'runtime-inputs.jsonl'
        with patch.object(subprocess,'run',side_effect=AssertionError('CPU roster generation must not launch/query')):
            jobs,pools=prepare.make_jobs(original['config'],samples)
        self.assertEqual(len(jobs),5);self.assertEqual(sum(len(j['phases']) for j in jobs),9)
        self.assertEqual(sum(len(j['priming_requests']) for j in jobs),240)
        self.assertEqual(sum(p['global_primary_count'] for j in jobs for p in j['phases']),1056)
        for job in jobs:
            self.assertEqual(len(job['priming_requests']),job['replicas']*12)
            for p in job['phases']:
                rows=[r for v in p['primary_requests'].values() for r in v]
                counts={pid:sum(r['pool_id']==pid for r in rows) for pid in pools[p['task']]}
                self.assertEqual(set(counts.values()),{3})
                self.assertTrue(any(v<len(pools[p['task']]) for v in p['planned_unique_primary_inputs_per_alias'].values()))

    def test_packet_check_cpu_only_and_rejects_rehashed_roster_tamper(self):
        path=prepare.DESTINATION/'experiment-packet.json'
        packet=json.loads(path.read_text())
        with patch.object(subprocess,'run',side_effect=AssertionError('CPU checker must not launch/query')):
            self.assertTrue(run.checker.validate_packet(packet)['valid'])
            changed=copy.deepcopy(packet);job=changed['jobs'][0];phase=job['phases'][0]
            row=next(iter(phase['primary_requests'].values()))[0];row['pool_id']='tampered'
            phase['phase_identity_sha256']=prepare.digest({k:v for k,v in phase.items() if k!='phase_identity_sha256'})
            job['context_identity_sha256']=prepare.digest({k:v for k,v in job.items() if k!='context_identity_sha256'})
            with self.assertRaisesRegex(ValueError,'roster'):run.checker.validate_packet(changed)

    def test_execute_needs_explicit_single_context_selector(self):
        with self.assertRaises(SystemExit):run.main(['--packet',str(prepare.DESTINATION/'experiment-packet.json'),'--execute'])


if __name__=='__main__':unittest.main()
