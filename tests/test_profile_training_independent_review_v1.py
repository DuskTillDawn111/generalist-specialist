"""Independent process/closure regressions derived from concrete CPU counterexamples."""
import copy
import fcntl
import ctypes as C
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import paired_profile_campaign_v4 as p
from tools.predictor_validation_20261027 import profile_training_supervisor_v1 as s
from tools.predictor_validation_20261027 import profile_owned_collect_v1 as owned


class IndependentLeaseAndEnvironmentTests(unittest.TestCase):
    def test_inherited_open_description_must_already_carry_parent_lock(self):
        with tempfile.TemporaryDirectory() as directory:
            lock=Path(directory)/'gpu.lock'
            code='from pathlib import Path;import os,sys;from tools.predictor_validation_20261027 import profile_training_supervisor_v1 as s;s.LOCK=Path(sys.argv[1]);s.validate_lease(int(os.environ[s.LEASE_FD]))'
            with lock.open('a') as locked,lock.open('a') as independent:
                def run(fd):
                    env=dict(os.environ,**{s.LEASE_FD:str(fd.fileno()),s.LEASE_PARENT:str(os.getpid())})
                    return subprocess.run([sys.executable,'-c',code,str(lock)],env=env,
                        pass_fds=(fd.fileno(),),capture_output=True,text=True)
                self.assertNotEqual(run(locked).returncode,0)
                fcntl.flock(locked,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertEqual(run(locked).returncode,0)
                # Same inode and parent's fd number exist, but this open-file
                # description is independent of the parent's locked handle.
                self.assertNotEqual(run(independent).returncode,0)

    def test_hidden_mps_or_other_device_cannot_enter_public_execution(self):
        for env in ({'CUDA_MPS_ACTIVE_THREAD_PERCENTAGE':'10'}, {'CUDA_MPS_PIPE_DIRECTORY':'/tmp/hidden'},
                    {'CUDA_VISIBLE_DEVICES':'1'}):
            with self.subTest(env=env),patch.dict(os.environ,env,clear=True),self.assertRaises(ValueError):
                s.public_environment_gate()
        with patch.dict(os.environ,{'CUDA_VISIBLE_DEVICES':'0'},clear=True):s.public_environment_gate()


class IndependentDynamicClosureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory();cls.root=Path(cls.temp.name)
        p.prepare(cls.root/'campaign')

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_each_dynamic_worker_observer_and_catalog_cannot_be_deleted_then_rehashed(self):
        for name in ('semantic_worker.py','semantic_observer.py','runtime-inputs.jsonl'):
            with self.subTest(name=name):
                run=self.root/name.replace('.','_');run.mkdir()
                original=self.root/'campaign/smoke/natural'
                plan=p.read(original/'plan.json');context=p.read(original/'context.json')
                removed=[src for src in context['execution_sources'] if Path(src['original_path']).name==name]
                self.assertTrue(removed)
                context['execution_sources']=[src for src in context['execution_sources'] if src not in removed]
                context['sources']=[src['frozen'] for src in context['execution_sources']]
                plan['execution_context_sha256']=p.context_identity(context)
                plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
                (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(context))
                with self.assertRaisesRegex(ValueError,'closure'):p.audit(run)

    def test_progress_worker_is_opt_in_and_cannot_replace_mode_after_freeze(self):
        run=self.root/'progress-tamper';run.mkdir()
        original=self.root/'campaign/smoke/natural'
        plan=p.read(original/'plan.json');context=p.read(original/'context.json')
        context['worker_variant']='semantic_progress_worker_v1.py'
        plan['execution_context_sha256']=p.context_identity(context)
        plan['plan_sha256']=p.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
        (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(context))
        with self.assertRaisesRegex(ValueError,'observer identity'):p.audit(run)


class IndependentMonitorTests(unittest.TestCase):
    def test_manual_and_background_sample_query_midpoints_append_monotonically(self):
        entered=threading.Event()
        class FakeNVML:
            def nvmlDeviceGetPowerUsage(self,*args):
                if threading.current_thread().name=='slow-query':
                    entered.set();time.sleep(.08)
                return 0
            def nvmlDeviceGetComputeRunningProcesses_v2(self,*args):return 0
        def initialize(m,runtime,interval):
            m.lib=FakeNVML();m.runtime=runtime;m.interval=interval;m.handle=C.c_void_p()
            m.samples=[];m.errors=[]
        ownership=dict(owned_used_mib=0,owned_processes_mib={},owned_aliases=[],foreign_pids=[],
                       unresolved_nvml_pids=[],ownership_evidence=[])
        with patch.object(owned.BoardMonitor,'__init__',initialize),patch.object(owned,'memory_ownership',return_value=ownership):
            monitor=owned.StartupBoardMonitor(None)
            slow=threading.Thread(target=monitor.sample,name='slow-query');slow.start()
            self.assertTrue(entered.wait(1));time.sleep(.06)
            # Without whole-query serialization this later quick query appends
            # before the first query, whose midpoint lies earlier than this one.
            monitor.sample();slow.join(1)
            self.assertFalse(slow.is_alive())
            self.assertEqual(len(monitor.samples),2)
            stamps=[row['time_ns'] for row in monitor.samples]
            self.assertEqual(stamps,sorted(stamps))
            self.assertEqual(monitor.errors,[])


class IndependentBudgetTests(unittest.TestCase):
    def test_total_storage_baseline_covers_launch_to_first_watchdog_gap(self):
        class Finished:
            pid=999999998
            def __init__(self, waits=0):self.waits=waits
            def poll(self):
                if self.waits:self.waits-=1;return None
                return 0
            def wait(self,timeout=None):return 0
        class Disk:
            def __init__(self,free):self.free=free
        # Both cases crossed6GiB before the monitor's first ordinary sample.
        # The second crosses during transition from launch accounting to monitor.
        for values in ([20,20,13.5,13.5,13.5],[20,20,14.1,13.9,13.9]):
            with self.subTest(values=values),tempfile.TemporaryDirectory() as directory:
                root=Path(directory);measurements=iter(int(v*2**30) for v in values)
                def disk(_):return Disk(next(measurements,int(values[-1]*2**30)))
                with patch.object(s.shutil,'disk_usage',side_effect=disk),patch.object(s.subprocess,'Popen',return_value=Finished(1 if values[2]>14 else 0)):
                    result=s.supervise(['CPU_FIXTURE'],root/'run',environment={},lock_path=root/'lock')
                self.assertEqual(result['status'],'FAILED_NO_RETRY')
                self.assertEqual(result['watchdog']['status'],'STORAGE_SCENARIO_EXCEEDED')

    def test_preexisting_unrelated_tree_blocks_launch_and_remains_alive(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(10)'],start_new_session=True)
            try:
                result=s.supervise([sys.executable,'-c','pass'],root/'run',environment={},lock_path=root/'lock',
                    preflight=0,budget=dict(s.BUDGET,total_timeout_seconds=1,cleanup_grace_seconds=.01))
                self.assertEqual(result['status'],'FAILED_NO_RETRY')
                self.assertFalse(result['collection_subprocess_launched'])
                self.assertIsNone(unrelated.poll())
            finally:unrelated.kill();unrelated.wait()


if __name__=='__main__':unittest.main()
