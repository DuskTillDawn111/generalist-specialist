"""CPU contracts for the diagnostic release loop and immutable attempts."""
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

PATH=Path(__file__).resolve().parents[1]/'tools/spillover_capacity_20261022/run.py'
spec=importlib.util.spec_from_file_location('capacity_runner',PATH)
runner=importlib.util.module_from_spec(spec);spec.loader.exec_module(runner)


class Monitor:
    def check(self): pass


class FakeRuntime:
    def __init__(self,fail=False):
        self.lock=threading.RLock();self.active={};self.peak={};self.submitted=[];self.threads=[];self.fail=fail
    def enqueue(self,row,sample):
        alias=row['model'];self.active[alias]=self.active.get(alias,0)+1
        self.peak[alias]=max(self.peak.get(alias,0),self.active[alias]);self.submitted.append(row)
        row.update(status='submitted',actual_submit_time_ns=time.perf_counter_ns())
        def complete():
            time.sleep(.006)
            with self.lock:
                self.active[alias]-=1
                row.update(status='error' if self.fail else 'completed',completion_time_ns=time.perf_counter_ns())
        thread=threading.Thread(target=complete);thread.start();self.threads.append(thread)
    def join(self):
        for t in self.threads:t.join()


def cohorts(n=4):
    return {a:[{'request_id':f'measurement:{a}:{i}','model':a,'task':'text','pool_id':str(i),'arrival_index':i,'sequence':i,'repetition':0} for i in range(n)] for a in ['text#0','text#1','generalist_2#0']}


class LoopTests(unittest.TestCase):
    def test_one_inflight_and_synchronized_release(self):
        runtime=FakeRuntime();rows=[];meta={};c=cohorts()
        runner.closed_loop(runtime,c,{str(i):{'sample':{}} for i in range(4)},rows,Monitor(),phase='measurement',timeout_seconds=2,request_timeout_seconds=1,metadata=meta)
        runtime.join()
        self.assertEqual(runtime.peak,{a:1 for a in c})
        self.assertEqual(len(rows),12)
        first=[r for r in rows if r['arrival_index']==0]
        self.assertEqual({r['scheduled_time_ns'] for r in first},{meta['start_time_ns']})
        self.assertEqual(set(meta['alias_completion_time_ns']),set(c))
        self.assertTrue(all(r['status']=='completed' for r in rows))
        for alias in c:
            rs=[r for r in rows if r['model']==alias]
            for previous,current in zip(rs,rs[1:]):
                self.assertGreaterEqual(current['actual_submit_time_ns'],previous['completion_time_ns'])

    def test_failure_stops_without_releasing_next_and_preserves_unknowns(self):
        runtime=FakeRuntime(fail=True);rows=[];c=cohorts()
        with self.assertRaisesRegex(RuntimeError,'request failure'):
            runner.closed_loop(runtime,c,{str(i):{'sample':{}} for i in range(4)},rows,Monitor(),phase='measurement',timeout_seconds=2,request_timeout_seconds=1)
        runtime.join()
        self.assertEqual(len(runtime.submitted),3)
        self.assertEqual(sum(r['status']=='planned' for r in rows),9)
        self.assertTrue(all(r['actual_submit_time_ns'] is None and r['completion_time_ns'] is None for r in rows if r['status']=='planned'))

    def test_warmup_stops_admitting_and_waits_for_last_response(self):
        runtime=FakeRuntime();rows=[];meta={};c=cohorts(1)
        runner.closed_loop(runtime,c,{'0':{'sample':{}}},rows,Monitor(),phase='warmup',warmup_seconds=.025,timeout_seconds=1,request_timeout_seconds=.5,metadata=meta)
        runtime.join()
        self.assertTrue(all(r['status']=='completed' for r in rows))
        self.assertGreaterEqual(meta['duration_seconds'],.025)
        self.assertTrue(all(r['scheduled_time_ns']-meta['start_time_ns']<25_000_000 for r in rows))
        self.assertEqual(len({r['request_id'] for r in rows}),len(rows))

    def test_deadline_is_finite_and_no_timestamp_fabrication(self):
        runtime=FakeRuntime()
        def enqueue(row,sample):row.update(status='submitted',actual_submit_time_ns=time.perf_counter_ns())
        runtime.enqueue=enqueue;rows=[]
        with self.assertRaisesRegex(TimeoutError,'Request timeout'):
            runner.closed_loop(runtime,cohorts(1),{'0':{'sample':{}}},rows,Monitor(),phase='measurement',timeout_seconds=1,request_timeout_seconds=.01)
        self.assertTrue(all(r['completion_time_ns'] is None for r in rows))


class PersistenceTests(unittest.TestCase):
    def test_failed_preflight_is_terminal_saved_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'attempt1';job={'job_id':'job1'}
            with patch.object(runner,'validate_packet',side_effect=ValueError('binding changed')):
                raw=runner.run_job({'aliases':{},'resident_deployment':{}},job,output,'packet-sha')
            self.assertEqual(raw['status'],'failed')
            self.assertEqual(raw['evaluation']['utility']['U'],None)
            saved=json.loads((output/'run.json').read_text())
            self.assertEqual(saved,raw)
            self.assertEqual(saved['errors'],['ValueError: binding changed'])
            before=(output/'run.json').read_bytes()
            with self.assertRaises(FileExistsError):runner.run_job({'aliases':{},'resident_deployment':{}},job,output,'packet-sha')
            self.assertEqual((output/'run.json').read_bytes(),before)

    def test_job_identity_binds_actual_content_not_self_field(self):
        job={'job_id':'x','planned_requests':{'a':[{'pool_id':'sample1'}]}}
        first=runner.job_digest(job)
        self.assertEqual(runner.job_digest({**job,'job_identity_sha256':first}),first)
        changed={'job_id':'x','planned_requests':{'a':[{'pool_id':'sample2'}]}}
        self.assertNotEqual(runner.job_digest(changed),first)

    def test_power_startup_is_bracketed_by_actual_samples(self):
        energy_spec=importlib.util.spec_from_file_location('frozen_energy',PATH.parents[1].parent/'artifacts/static-study-gpu-20261022/execution-source-v5/gsdeploy/energy.py')
        energy=importlib.util.module_from_spec(energy_spec);energy_spec.loader.exec_module(energy)
        class Power:
            def __init__(self):self.samples=[];self.started=False
            def sample(self):self.samples.append({'time_ns':time.perf_counter_ns(),'watts':100.})
            def start(self):self.started=True
        power=Power();startup=runner.begin_power_capture(power)
        time.sleep(.001);measurement_start=time.perf_counter_ns();time.sleep(.001);power.sample()
        self.assertTrue(power.started)
        self.assertLessEqual(power.samples[0]['time_ns'],startup)
        self.assertGreater(energy.integrate_power(power.samples,startup,measurement_start),0)

    def test_binding_tamper_detected_before_any_gpu_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'source';p.write_text('before')
            packet={'protocol':runner.PROTOCOL,'priming_policy':'largest_three_all_resident_supported_tasks',
                    'resident_deployment':{'text':2,'generalist_2':1,'ocr':1},
                    'aliases':{'text#0':'text','text#1':'text','generalist_2#0':'generalist_2','ocr#0':'ocr'},
                    **{k:str(p) for k in ['frozen_source','system_config','samples','workers','admission']},'bindings':{str(p):runner.digest(p)}}
            packet['bindings_sha256']=runner.hashlib.sha256(runner.canonical(packet['bindings'])).hexdigest()
            p.write_text('after')
            with patch.object(runner,'gpu_idle') as gpu:
                with self.assertRaisesRegex(ValueError,'Binding changed'):runner.validate_packet(packet)
                gpu.assert_not_called()

    def test_interrupted_attempt_raw_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(runner,'validate_packet',side_effect=KeyboardInterrupt('pause')):
                raw=runner.run_job({'aliases':{},'resident_deployment':{}}, {'job_id':'job'},Path(tmp)/'attempt1','sha')
            self.assertEqual(raw['status'],'failed')
            self.assertEqual(raw['errors'],['KeyboardInterrupt: pause'])
            self.assertTrue((Path(tmp)/'attempt1/run.json').is_file())


if __name__=='__main__':unittest.main()
