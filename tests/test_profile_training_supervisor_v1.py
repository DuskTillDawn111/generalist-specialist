"""Real CPU process ownership and strict training plan/mode regressions."""
import copy
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import profile_training_supervisor_v1 as s
from tools.predictor_validation_20261027 import paired_profile_campaign_v4 as p
from tools.predictor_validation_20261027 import kernel_profile_campaign_v2 as old


class PairedPlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)/'campaign'
        p.prepare(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def copy_run(self, root):
        target = root/'run'; target.mkdir()
        for name in ('context.json','plan.json'):
            (target/name).write_bytes((self.root/'smoke/natural'/name).read_bytes())
        return target

    def test_matching_public_job_pair_and_required_training_controls(self):
        a=p.audit(self.root/'training/natural'); b=p.audit(self.root/'training/kernel')
        self.assertEqual(a['jobs'],b['jobs']); self.assertEqual(a['pair'],b['pair'])
        self.assertEqual(len(a['jobs']),26)
        self.assertTrue({'S0','S1','G','O','SS','SG','S1G','SSG','SSGO'} <= {j['context'] for j in a['jobs']})
        shifts=[j for j in a['jobs'] if j['context']=='SG']
        self.assertEqual({max(j['start_offsets_ms'].values()) for j in shifts},{0,50,200})
        self.assertTrue(any(j['start_offsets_ms']['text#0']>j['start_offsets_ms']['generalist_2#0'] for j in shifts))

    def test_old_profiler_plan_refused_and_mode_switch_cannot_be_resealed(self):
        with self.assertRaisesRegex(ValueError,'v4'):
            p.audit(self.root/'source-freeze/smoke')
        with tempfile.TemporaryDirectory() as d:
            run=self.copy_run(Path(d)); plan=p.read(run/'plan.json')
            plan['mode']='kernel';plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            (run/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'mode/deployment'):
                p.audit(run)

    def test_job_pair_change_rejected_even_with_new_plan_digest(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.copy_run(Path(d)); plan=p.read(run/'plan.json')
            plan['jobs'][0]['start_offsets_ms']['text#0']=1
            plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            (run/'plan.json').write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError,'pairing'):
                p.audit(run)

    def test_training_cannot_skip_both_smoke_and_new_modes_bind_supervisor(self):
        with self.assertRaises(FileNotFoundError):
            p.execution_gate(self.root/'training/natural')
        with tempfile.TemporaryDirectory() as d:
            manifest=Path(d)/'supervisor.json'
            s.prepare('paired',self.root/'smoke/natural',manifest)
            actual=s.audit(manifest)
            self.assertEqual(actual['binding']['mode'],'natural')
            self.assertEqual(actual['budget'],s.BUDGET)
            tampered=p.read(manifest);tampered['budget']['total_timeout_seconds']=9999
            manifest.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError,'schema/budget'):
                s.audit(manifest)

    def test_missing_supervisor_closure_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.copy_run(Path(d)); context=p.read(run/'context.json');plan=p.read(run/'plan.json')
            context['execution_sources']=[r for r in context['execution_sources'] if not r['original_path'].endswith('profile_training_supervisor_v1.py')]
            context['sources']=[r['frozen'] for r in context['execution_sources']]
            plan['execution_context_sha256']=p.context_identity(context)
            plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError,'implementation closure'):
                p.audit(run)

    def test_owned_collector_cannot_switch_declared_mode(self):
        from tools.predictor_validation_20261027 import profile_owned_collect_v1 as owned
        with patch.object(owned,'validate_lease'),patch.object(owned,'DeploymentRuntime') as runtime:
            with self.assertRaisesRegex(ValueError,'Collector mode'):
                owned.collect(self.root/'smoke/kernel',trace=False,lease_fd=42)
            runtime.assert_not_called()

    def test_dynamic_worker_observer_cannot_be_removed_then_resealed(self):
        with tempfile.TemporaryDirectory() as d:
            run=self.copy_run(Path(d));context=p.read(run/'context.json');plan=p.read(run/'plan.json')
            context['execution_sources']=[r for r in context['execution_sources'] if not r['original_path'].endswith(('semantic_worker.py','semantic_observer.py'))]
            context['sources']=[r['frozen'] for r in context['execution_sources']]
            plan['execution_context_sha256']=p.context_identity(context)
            plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(context))
            with self.assertRaisesRegex(ValueError,'implementation closure'):
                p.audit(run)

    def test_completed_execution_requires_exact_manifest_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);manifest_path=root/'manifest.json'
            manifest=s.prepare('paired',self.root/'smoke/natural',manifest_path)
            execution=dict(status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',live_group_pids=[],
                child_reaped=True,experiment_lock_held_through_group_exit=True,final_source_audit='PASS',
                all_owned_descendants_exited_and_reaped=True,
                watchdog=dict(status='EXITED',return_code=0),profile_execution=dict(
                    schema=s.SCHEMA+'.execution_binding',manifest=s.receipt(manifest_path),binding=manifest['binding']))
            record=root/'execution.json';s.save(record,execution)
            s.completed_execution(s.receipt(record),self.root/'smoke/natural')
            with self.assertRaisesRegex(ValueError,'completed target'):
                s.completed_execution(s.receipt(record),self.root/'smoke/kernel')
            execution['watchdog']['return_code']=1;record.write_text(json.dumps(execution))
            with self.assertRaisesRegex(ValueError,'completed target'):
                s.completed_execution(s.receipt(record),self.root/'smoke/natural')

    def test_natural_smoke_checks_actual_start_job_input_and_shutdown(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for mutation,want in [(None,'PASS'),('start','FAIL'),('input','FAIL'),('shutdown','FAIL')]:
                run=root/str(mutation);run.mkdir()
                for name in ('context.json','plan.json'):
                    (run/name).write_bytes((self.root/'smoke/natural'/name).read_bytes())
                plan=p.read(run/'plan.json');job=plan['jobs'][0]
                rows=[];prime=[];ready={}
                for i,(alias,family) in enumerate(p.RESIDENTS.items()):
                    pid=100+i;ready[alias]=dict(pid=pid)
                    rid='profile:'+str(i)
                    row=dict(request_id=rid,model=alias,task=job['tasks'][alias],pool_id=job['inputs'][alias],
                        profile_job_id=job['id'],scheduled_time_ns=1000,status='completed',
                        worker_response=dict(start_perf_ns=1100,end_perf_ns=1200),
                        response_reader_event_enqueue_time_ns=1300,
                        previous_handler_next_request_ready=dict(perf_ns=1250,received_time_ns=1350),
                        execution_events=[dict(request_id=rid,perf_ns=1100,pid=pid),
                            dict(request_id=rid,event='start',kind='gpu',segment_id='native',perf_ns=1110),
                            dict(request_id=rid,event='end',kind='gpu',segment_id='native',perf_ns=1190)])
                    rows.append(row)
                    tasks=['text','image','asr'] if family=='generalist_2' else ['ocr' if family=='ocr' else 'text']
                    for task in tasks:
                        prime.extend(dict(model=alias,task=task,pool_id=x['pool_id'],status='completed') for x in sorted(
                            (r for r in plan['inputs'] if r['task']==task),key=lambda x:x['input_size'],reverse=True)[:3])
                p.save(run/'executed-jobs.json',dict(split=None,contexts=None,limit=None,trace=False,jobs=plan['jobs']))
                raw=dict(status='completed',plan=p.receipt(run/'plan.json'),context=p.receipt(run/'context.json'),
                    executed_jobs=p.receipt(run/'executed-jobs.json'),performance_scope='unprofiled_with_phase_markers',
                    ready=ready,request_records=rows,priming_request_records=prime,batches=[dict(job_id=job['id'],start_ns=1000)],
                    shutdown={a:dict(exit_code=0,forced=False,bye_received=True) for a in p.RESIDENTS},
                    board_samples=[dict(foreign_pids=[],owned_aliases=list(p.RESIDENTS))])
                if mutation=='start':rows[0]['execution_events'].pop(0)
                if mutation=='input':rows[0]['pool_id']='wrong-input'
                if mutation=='shutdown':raw['shutdown']['text#0']['forced']=True
                p.save(run/'raw.json',raw);p.save(run/'supervision.json',dict(status='CPU_FIXTURE'))
                with patch.object(s,'completed_execution',return_value=dict(status='CHILD_EXITED_REQUIRES_PROFILE_REVIEW',live_group_pids=[],child_reaped=True)):
                    result=p.verify_smoke(run,run/'supervision.json')
                self.assertEqual(result['status'],want,result['issues'])


class ProcessBudgetTests(unittest.TestCase):
    def budget(self, timeout=.2):
        return dict(s.BUDGET,total_timeout_seconds=timeout,watchdog_interval_seconds=.01,
                    cleanup_grace_seconds=.01,free_disk_reserve_bytes=0,disk_abort_margin_bytes=0)

    def free(self,n):
        return type('Disk',(),dict(free=n))()

    def test_preflight_and_busy_lock_refuse_before_process_launch(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); lock=root/'gpu.lock'
            with patch.object(s.shutil,'disk_usage',return_value=self.free(s.PREFLIGHT-1)),patch.object(s.subprocess,'Popen') as launch:
                with self.assertRaisesRegex(ValueError,'18GiB'):
                    s.supervise([sys.executable,'-c','raise RuntimeError()'],root/'low',environment={},lock_path=lock)
                launch.assert_not_called();self.assertFalse((root/'low').exists())
            with lock.open('a') as holder:
                fcntl.flock(holder,fcntl.LOCK_EX|fcntl.LOCK_NB)
                with patch.object(s.subprocess,'Popen') as launch:
                    with self.assertRaises(BlockingIOError):
                        s.supervise([sys.executable,'-c','pass'],root/'busy',environment={},lock_path=lock,preflight=0)
                    launch.assert_not_called()

    def test_live_parent_lease_and_lock_retained_to_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=root/'gpu.lock'
            code='from pathlib import Path; import os,sys; from tools.predictor_validation_20261027 import profile_training_supervisor_v1 as s; s.LOCK=Path(sys.argv[1]); s.validate_lease(int(os.environ[s.LEASE_FD])); print("lease-valid")'
            result=s.supervise([sys.executable,'-c',code,str(lock)],root/'run',environment={},lock_path=lock,preflight=0,budget=self.budget(5))
            self.assertEqual(result['status'],'CHILD_EXITED_REQUIRES_PROFILE_REVIEW')
            self.assertEqual(result['live_group_pids'],[]);self.assertTrue(result['child_reaped'])
            self.assertIn('lease-valid',(root/'run/stdout.log').read_text())
            with lock.open('a') as lease:
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_unlocked_parent_cannot_create_lease_during_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=root/'gpu.lock'
            code='from pathlib import Path; import os,sys; from tools.predictor_validation_20261027 import profile_training_supervisor_v1 as s; s.LOCK=Path(sys.argv[1]); s.validate_lease(int(os.environ[s.LEASE_FD]))'
            with lock.open('a') as lease:
                proc=subprocess.run([sys.executable,'-c',code,str(lock)],env={**os.environ,
                    s.LEASE_FD:str(lease.fileno()),s.LEASE_PARENT:str(os.getpid())},
                    pass_fds=(lease.fileno(),),capture_output=True,text=True,timeout=10)
                self.assertNotEqual(proc.returncode,0)
                self.assertIn('Parent must already hold',proc.stderr)
                fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_timeout_kills_real_descendant_and_preserves_unrelated_group(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=root/'gpu.lock'; pidfile=root/'descendant.pid'
            code='import subprocess,sys,signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); p=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"]); open(sys.argv[1],"w").write(str(p.pid)); time.sleep(30)'
            try:
                result=s.supervise([sys.executable,'-c',code,str(pidfile)],root/'run',environment={},lock_path=lock,preflight=0,budget=self.budget(.35))
                self.assertEqual(result['status'],'FAILED_NO_RETRY')
                self.assertEqual(result['watchdog']['status'],'TIMEOUT');self.assertEqual(result['live_group_pids'],[])
                self.assertTrue(result['experiment_lock_held_through_group_exit'])
                self.assertTrue(pidfile.exists())
                self.assertFalse(int(pidfile.read_text()) in s.group_members(result['process_group']))
            finally:
                pass

    def test_preexisting_unrelated_tree_is_preserved_and_launch_refused(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            other=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            try:
                result=s.supervise([sys.executable,'-c','raise RuntimeError()'],root/'run',environment={},lock_path=root/'gpu.lock',preflight=0,budget=self.budget(5))
                self.assertEqual(result['status'],'FAILED_NO_RETRY')
                self.assertFalse(result['collection_subprocess_launched'])
                self.assertIn('no pre-existing child trees',result['error'])
                self.assertIsNone(other.poll())
            finally:
                other.kill();other.wait()

    def test_exit_with_abandoned_child_and_final_source_drift_fail(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            code='import subprocess,sys; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"])'
            result=s.supervise([sys.executable,'-c',code],root/'orphan',environment={},lock_path=root/'gpu.lock',preflight=0,budget=self.budget(5))
            self.assertEqual(result['status'],'FAILED_NO_RETRY')
            self.assertEqual(result['watchdog']['status'],'LIVE_DESCENDANTS_AFTER_EXIT')
            self.assertEqual(result['live_group_pids'],[])
            def drift():raise ValueError('changed implementation')
            result=s.supervise([sys.executable,'-c','pass'],root/'drift',environment={},lock_path=root/'gpu.lock',preflight=0,budget=self.budget(5),after_run=drift)
            self.assertEqual(result['status'],'FAILED_NO_RETRY');self.assertEqual(result['final_source_audit'],'FAIL')

    def test_setsid_descendant_with_cleared_environment_is_reaped_before_unlock(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);lock=root/'gpu.lock';pidfile=root/'detached.pid'
            code='import subprocess,sys; p=subprocess.Popen([sys.executable,"-c","import signal,time; signal.signal(signal.SIGINT,signal.SIG_IGN); signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(30)"],start_new_session=True,env={}); open(sys.argv[1],"w").write(str(p.pid))'
            result=s.supervise([sys.executable,'-c',code,str(pidfile)],root/'run',environment={},lock_path=lock,preflight=0,budget=self.budget(5))
            detached=int(pidfile.read_text())
            self.assertEqual(result['status'],'FAILED_NO_RETRY')
            self.assertEqual(result['watchdog']['status'],'EXITED')
            exit=result['owned_descendant_exit']
            self.assertTrue(any(x['pid']==detached and x['process_group']!=result['process_group'] for x in exit['observed_identities']))
            self.assertTrue(any(x['pid']==detached for x in exit['adopted_children_reaped']))
            self.assertFalse(Path(f'/proc/{detached}').exists())
            with lock.open('a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_current_13GiB_and_6GiB_boundaries_use_reviewed_wait_helper(self):
        class Exited:
            pid=999999999
            def poll(self):return 0
        for start,now,want in [(19*2**30,13*2**30,'EXITED'),(19*2**30,13*2**30-1,'DISK_MARGIN'),(20*2**30,14*2**30,'EXITED'),(20*2**30,14*2**30-1,'STORAGE_SCENARIO_EXCEEDED')]:
            with self.subTest(now=now),patch('tools.predictor_validation_20261027.unified_corun_collect_v8_1.shutil.disk_usage',side_effect=[self.free(start),self.free(now)]),patch('tools.predictor_validation_20261027.unified_corun_collect_v8_1.cleanup_group',return_value=dict(live_survivors=[])):
                result=s.wait_bounded(Exited(),'.',s.BUDGET)
                self.assertEqual(result['status'],want)

    def test_launch_growth_over_6GiB_cannot_reset_watchdog_baseline(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            # Preflight20, original20, after-launch13.5GiB:6.5GiB growth
            # already exceeds the increment while remaining above13GiB floor.
            values=iter([20*2**30,20*2**30,int(13.5*2**30)])
            def disk(_):
                return self.free(next(values,int(13.5*2**30)))
            with patch.object(s.shutil,'disk_usage',side_effect=disk),patch.object(s,'wait_bounded') as wait:
                result=s.supervise([sys.executable,'-c','import time;time.sleep(5)'],root/'run',environment={},lock_path=root/'gpu.lock',budget=self.budget(5))
                self.assertEqual(result['status'],'FAILED_NO_RETRY')
                self.assertEqual(result['watchdog']['status'],'STORAGE_SCENARIO_EXCEEDED')
                wait.assert_not_called()

    def test_second_sample_growth_uses_one_original_baseline(self):
        class Running:
            pid=999999999
            def poll(self):return None
        with patch.object(s.shutil,'disk_usage',side_effect=[self.free(int(14.1*2**30)),self.free(int(13.9*2**30))]),patch.object(s.time,'sleep'),patch.object(s,'cleanup_group',return_value=dict(live_survivors=[])):
            result=s.wait_from_baseline(Running(),'.',s.BUDGET,initial_free=20*2**30,begin=s.time.monotonic())
            self.assertEqual(result['status'],'STORAGE_SCENARIO_EXCEEDED')
            self.assertEqual(result['initial_free_bytes'],20*2**30)
            self.assertEqual(len(result['samples']),2)

    def test_incomplete_batch_child_has_nonzero_exit(self):
        with patch.object(sys,'argv',['supervisor','_child','--manifest','unused','--execute']),patch.object(s,'child',return_value=dict(status='INCOMPLETE_OR_FAILED_NOT_QUALIFIED')):
            self.assertEqual(s.main(),1)

    def test_mps_and_other_device_inheritance_blocked(self):
        with patch.dict(os.environ,{'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':'50'}):
            with self.assertRaisesRegex(ValueError,'MPS'):s.public_environment_gate()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'1'},clear=True):
            with self.assertRaisesRegex(ValueError,'device0'):s.public_environment_gate()

    def test_startup_owned_gpu_pid_can_be_descendant_before_ready(self):
        from tools.predictor_validation_20261027.profile_owned_collect_v1 import memory_ownership
        code='import subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); print(p.pid,flush=True); time.sleep(30)'
        launcher=subprocess.Popen([sys.executable,'-c',code],stdout=subprocess.PIPE,text=True)
        foreign=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        gpu_pid=None
        try:
            gpu_pid=int(launcher.stdout.readline())
            runtime=SimpleNamespace(workers={'text#0':SimpleNamespace(process=launcher)},ready={})
            result=memory_ownership({gpu_pid:16,foreign.pid:9,999999999:11},runtime,os.getpid(),os.getpgrp())
            self.assertEqual(result['owned_used_mib'],16)
            self.assertEqual(result['owned_aliases'],['text#0'])
            self.assertEqual(result['foreign_pids'],[foreign.pid])
            self.assertEqual(result['unresolved_nvml_pids'],[999999999])
            runtime.ready={'text#0':dict(pid=gpu_pid)}
            after=memory_ownership({gpu_pid:16},runtime,os.getpid(),os.getpgrp())
            self.assertEqual(after['owned_used_mib'],16)
        finally:
            if gpu_pid is not None:
                try:os.kill(gpu_pid,9)
                except ProcessLookupError:pass
            launcher.kill();launcher.wait();launcher.stdout.close()
            foreign.kill();foreign.wait()

    def test_overlapping_monitor_samples_serialize_query_phase_and_append(self):
        from tools.predictor_validation_20261027.profile_owned_collect_v1 import StartupBoardMonitor
        monitor=object.__new__(StartupBoardMonitor);monitor._sample_lock=threading.RLock()
        entered=threading.Event();release=threading.Event();finished=threading.Event();order=[]
        def query():
            order.append(threading.current_thread().name)
            if len(order)==1:
                entered.set();release.wait(2)
            return len(order)
        monitor._sample=query
        first=threading.Thread(target=monitor.sample,name='first')
        second=threading.Thread(target=lambda:(monitor.sample(),finished.set()),name='second')
        first.start();self.assertTrue(entered.wait(2));second.start()
        self.assertFalse(finished.wait(.05));self.assertEqual(order,['first'])
        release.set();first.join(2);second.join(2)
        self.assertEqual(order,['first','second'])


if __name__=='__main__':unittest.main()
