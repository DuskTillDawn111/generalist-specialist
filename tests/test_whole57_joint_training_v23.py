"""CPU-only collector tests, including real FIFO pipe backpressure and cleanup."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from gsdeploy.runtime.deployment import DeploymentRuntime
from gsdeploy.runtime.schema import MeasurementConfig
from tools.predictor_validation_20261027 import whole57_joint_training_v23 as collect
from tools.predictor_validation_20261027 import whole57_joint_review_v23 as review


class Whole57TrainingTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.inputs=[dict(task=t,pool_id=t+str(i),input_size=i+1) for t in ('text','image','asr','ocr') for i in range(2)]
        self.trace=collect.make_trace(self.inputs,'whole57-training-fixture')
        self.routes,self.weights=collect.routes_for(self.trace,'balanced')
        self.plan=dict(training_id='whole57-training-fixture',trace=self.trace,routes=self.routes,
            inputs=self.inputs,resource_configuration=collect.resource_configuration())
        self.pool={r['pool_id']:dict(pool_id=r['pool_id'],task_type=r['task'],sample=dict(task=r['task'],id=r['pool_id'])) for r in self.inputs}

    def test_recipe_keeps57_new_training_ids_four_routes_and_sustained_arrivals(self):
        self.assertEqual(len(self.trace),57);self.assertEqual(sum(r['task']=='text' for r in self.trace),56)
        self.assertTrue(all(r['request_id'].startswith('whole57-training-fixture:req') for r in self.trace))
        self.assertEqual({r['pool_id'] for r in self.trace if r['task']=='text'},{'text0','text1'})
        self.assertEqual(collect.RECIPE['route_source'],'PREDECLARED_TRAINING_RECIPE_NOT_SELECTED_FROM_FAILED_VALIDATION')
        for route in collect.RECIPE['routes']:
            routes,_=collect.routes_for(self.trace,route['id']);self.assertEqual(set(routes),{r['request_id'] for r in self.trace})
        with self.assertRaises(ValueError):collect.make_trace(self.inputs,'fresh-availability-v11-r01')

    def worker_script(self):
        path=self.root/'cpu_worker.py'
        path.write_text('''import json,os,pathlib,sys,time
def emit(k,p):print(k+'|'+json.dumps(p),flush=True)
emit('READY',dict(pid=os.getpid(),tasks=['text','image','asr','ocr']))
barrier=pathlib.Path(sys.argv[1])
while not barrier.exists():time.sleep(.005)
for line in sys.stdin:
 r=json.loads(line)
 if r.get('op')=='shutdown':emit('BYE',dict(pid=os.getpid()));break
 rid=r['request_id'];start=time.perf_counter_ns();emit('START',dict(request_id=rid,pid=os.getpid(),perf_ns=start))
 a=time.perf_counter_ns();emit('PHASE',dict(request_id=rid,event='start',kind='gpu',segment_id='native_inference',perf_ns=a))
 b=time.perf_counter_ns();emit('PHASE',dict(request_id=rid,event='end',kind='gpu',segment_id='native_inference',perf_ns=b))
 end=time.perf_counter_ns();emit('RESULT',dict(request_id=rid,task_type=r['task_type'],success=True,start_perf_ns=start,end_perf_ns=end,worker_latency_ms=(end-start)/1e6,service_ms=(end-start)/1e6,semantic_forward_observations=dict(records=[dict(start_ns=a,end_ns=b)],instrumentation_errors=[])))
 emit('WORKER_READY',dict(request_id=rid,pid=os.getpid(),perf_ns=time.perf_counter_ns(),clock='host_perf_counter_ns',boundary='previous_handler_next_request_ready'))
''')
        return path

    def test_real_blocked_dispatch_keeps_all57_registered_and_clock_chains(self):
        script=self.worker_script();barrier=self.root/'release';raw={};runtime=None
        registry={a:dict(command=[sys.executable,'-u',str(script),str(barrier)]) for a in collect.RESIDENTS}
        config=MeasurementConfig(deployment=dict.fromkeys(collect.RESIDENTS,1),workload_rps=dict(text=1.,ocr=1.),
            routing_rps=dict(text={'text#0':.5,'text#1':.5},ocr={'ocr#0':1.}),startup_timeout_seconds=5,shutdown_timeout_seconds=5)
        runtime=DeploymentRuntime(config,registry);runtime.start()
        try:
            for item in self.pool.values():item['sample']['large_payload']='x'*262144
            start=time.perf_counter_ns()-31_000_000_000;began=time.monotonic()
            collect.register_and_offer(raw,runtime,self.plan,self.pool,SimpleNamespace(check=lambda:None),start_ns=start,wait=lambda target:None)
            self.assertLess(time.monotonic()-began,2);self.assertEqual(len(raw['request_records']),57)
            self.assertTrue(all(r['status'] in ('queued','submitted') for r in raw['request_records']))
            # A worker has not read stdin: the real sender is blocked in pipe
            # write/flush, but the independent arrival loop retained every row.
            deadline=time.monotonic()+2
            while time.monotonic()<deadline and not any(r.get('actual_submit_time_ns') is not None for r in raw['request_records']):time.sleep(.005)
            self.assertTrue(any(r.get('actual_submit_time_ns') is not None and 'submit_write_completed_time_ns' not in r for r in raw['request_records']))
            self.assertTrue(any(r['status']=='queued' for r in raw['request_records']))
            barrier.touch();runtime.drain(raw['request_records'],15)
        finally:
            barrier.touch(exist_ok=True);runtime.close()
        self.assertEqual(runtime.errors,[]);self.assertTrue(all(r['status']=='completed' for r in raw['request_records']))
        raw.update(planned_public_cohort=self.trace,ready=runtime.ready,priming_request_records=[],observation_end_ns=time.perf_counter_ns())
        issues,rows,anchors=review.host_lifecycle_rows(self.plan,raw)
        self.assertEqual(issues,[]);self.assertEqual(len(rows),57);self.assertEqual(len(anchors),57)
        self.assertTrue(all(r['offsets_ns']['worker_next_ready_ns']>=r['offsets_ns']['handler_end_ns'] for r in rows))
        self.assertTrue(all(r['scheduled_time_ns']-raw['phase_start_ns']==d['offset_ns'] for r,d in zip(raw['request_records'],self.trace)))
        broken=copy.deepcopy(raw);broken['request_records'][-1]['previous_handler_next_request_ready']['clock']='guessed'
        self.assertTrue(any('WORKER_NEXT_READY' in s for s in review.host_lifecycle_rows(self.plan,broken)[0]))
        broken=copy.deepcopy(raw);broken['request_records'][0]['submit_write_completed_time_ns']=broken['request_records'][0]['actual_submit_time_ns']-1
        self.assertTrue(any('PIPE_WRITE_COMPLETION_PRECEDES_DISPATCH' in s for s in review.host_lifecycle_rows(self.plan,broken)[0]))
        for field,value in [('deadline_ns',raw['deadline_ns']-1),('arrival_loop_end_ns',raw['deadline_ns']-1),('observation_end_ns',raw['arrival_loop_end_ns']-1)]:
            broken=copy.deepcopy(raw);broken[field]=value
            self.assertIn('EXACT_ORIGINAL_WINDOW_OFFER_AND_TAIL_DRAIN_BOUNDARIES_REQUIRED',review.host_lifecycle_rows(self.plan,broken)[0])

    def test_all57_registered_before_arrival_wait_failure(self):
        raw={};calls=[]
        def fail(target):
            self.assertEqual(len(raw['request_records']),57);raise RuntimeError('CPU wait failure')
        with self.assertRaisesRegex(RuntimeError,'wait failure'):
            collect.register_and_offer(raw,SimpleNamespace(enqueue=lambda *args:calls.append(args)),self.plan,self.pool,
                SimpleNamespace(),start_ns=100,wait=fail)
        self.assertEqual(len(raw['request_records']),57);self.assertEqual(calls,[])
        self.assertTrue(all(r['status']=='planned' for r in raw['request_records']))

    def test_clock_anchor_records_causal_bracket_without_exact_alignment_claim(self):
        anchor=collect.clock_anchor('fixture')
        self.assertLessEqual(anchor['perf_before_ns'],anchor['perf_after_ns']);self.assertFalse(anchor['simultaneous_sample_claimed'])
        self.assertEqual(anchor['host_clock'],'time.perf_counter_ns')

    def test_prime_failure_saves_full_planned57_without_inventing_measurement(self):
        run=self.root/'prime-fail';run.mkdir();plan=copy.deepcopy(self.plan);plan.update(domain=collect.NATURAL,
            route_id='balanced',execution_context_sha256='ctx')
        context=dict(execution_nonce='fixture',execution_sources=[dict(original_path=str(self.root/'runtime-inputs.jsonl'))],collection_budget={})
        (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(context))
        (self.root/'runtime-inputs.jsonl').write_text('\n'.join(json.dumps(v) for v in self.pool.values()))
        class Monitor:
            def __init__(self,*args):self.samples=[];self.errors=[];self.identity_events=[];self._sample_lock=threading.RLock()
            def sample(self):pass
            def check(self):pass
            def start(self):pass
            def close(self):pass
        class Runtime:
            def __init__(self,*args):self.workers={};self.ready={};self.shutdown_report={};self.errors=[]
            def start(self):pass
            def enqueue(self,row,sample):row['status']='error'
            def drain(self,*args):pass
            def close(self):pass
        with patch.object(collect,'audit',return_value={}),patch.object(collect,'DeploymentRuntime',Runtime),\
             patch.object(collect,'prepare_registry',return_value={}),patch.object(collect.existing,'IdentityMonitor',Monitor),\
             patch.object(collect.existing,'build_registry',return_value={}),patch.object(collect.existing,'boundary',return_value=None),\
             patch.dict(os.environ,{collect.existing.NONCE_ENV:'fixture'}):
            result=collect.cohort(run)
        self.assertEqual(result['status'],'failed');self.assertIn('priming failed',result['errors'][0])
        self.assertEqual(len(result['planned_public_cohort']),57);self.assertEqual(result['request_records'],[])
        self.assertNotIn('phase_start_ns',result);self.assertTrue((run/'raw.json').exists())

    def test_single_session_command_has_no_child_profiler_or_request_gate(self):
        context=dict(controller_python=sys.executable,nsys='/fixture/nsys')
        command=collect.outer_command(self.root,context,collect.LIGHT)
        self.assertIn('--wait=all',command);self.assertIn('--capture-range=none',command)
        self.assertIn('--cuda-trace-scope=process-tree',command);self.assertEqual(command.count('profile'),1)
        natural=collect.outer_command(self.root,context,collect.NATURAL)
        self.assertNotIn('/fixture/nsys',natural);self.assertNotIn('profile_start',natural)

    def test_real_watchdog_timeout_cleans_owned_descendant_group(self):
        # Existing collector watchdog must also remove a child spawned by a
        # still-running controller, using its explicitly owned new session.
        pidfile=self.root/'descendant.pid';script=self.root/'tree.py'
        script.write_text('import pathlib,subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time;time.sleep(30)"])\npathlib.Path(sys.argv[1]).write_text(str(p.pid))\ntime.sleep(30)\n')
        p=subprocess.Popen([sys.executable,str(script),str(pidfile)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        budget=dict(free_disk_reserve_bytes=0,disk_abort_margin_bytes=0,incremental_storage_scenario_bytes=2**40,
            total_timeout_seconds=.3,watchdog_interval_seconds=.02,cleanup_grace_seconds=.1)
        try:
            result=collect.existing.wait_bounded(p,self.root,budget)
            self.assertEqual(result['status'],'TIMEOUT');self.assertEqual(result['cleanup']['live_survivors'],[])
            self.assertIsNotNone(p.poll());self.assertTrue(pidfile.exists())
            self.assertEqual(collect.existing.group_members(p.pid),[])
        finally:
            if collect.existing.group_members(p.pid):collect.existing.cleanup_group(p,.1)


if __name__=='__main__':unittest.main()
