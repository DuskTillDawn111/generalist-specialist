import copy
import gc
import json
import os
from pathlib import Path
import tempfile
import sys
import threading
import time
import unittest
import weakref
from unittest.mock import patch

from gsdeploy.runtime.batching import BatchConfiguration
from tests.test_shared_text_batch_v2 import backend,sample,MODEL
from tools.predictor_validation_20261027 import batch_profile_collect_v3 as c


class BackgroundRuntime:
    def __init__(self,*,fail_measurement=False):
        self.lock=threading.RLock();self.ready={};self.errors=[];self.records=[]
        self.measurement_received=threading.Event();self.fail=fail_measurement
    def start(self):self.ready={owner:dict(pid=100+i,CPU_TEST_ONLY=True) for i,owner in enumerate(c.BACKGROUND)}
    def enqueue(self,row,sample):
        with self.lock:
            self.records.append(row)
            row.update(actual_submit_time_ns=time.perf_counter_ns(),status='submitted')
            if self.fail and row['phase']=='measurement':
                row.update(status='error',error='CPU background inference failure');return
            end=time.perf_counter_ns()
            row.update(completion_time_ns=end,status='completed',worker_response=dict(success=True,
                task_type=row['task'],sample_id=sample['id'],end_perf_ns=end),
                previous_handler_next_request_ready=dict(perf_ns=end+1))
            if row['phase']=='measurement':self.measurement_received.set()
    def drain(self,records,timeout):pass
    def close(self):return {owner:dict(exit_code=0,forced=False,bye_received=True) for owner in c.BACKGROUND}


class BatchProfileCollectV3Tests(unittest.TestCase):
    def fixture(self,root):
        source=root/'source/gsdeploy/backends';source.mkdir(parents=True)
        (source/'compact_native.py').write_text('CPU fixture only\n')
        protocol=root/'protocol.json';protocol.write_text('{}')
        classes=root/'classmap.json';classes.write_text('{}')
        spec=root/'spec.json';spec.write_text(json.dumps(dict(candidate=dict(slot='text',model=MODEL),
            execution=dict(family='chatqa'),resources=dict(protocol=str(protocol),classmap=str(classes)))))
        samples={};arrivals={};priming={}
        for split,prefix,tokens in [('training','t',2),('validation','v',3)]:
            image=root/(split+'.image');image.write_bytes(split.encode())
            rows=[sample(prefix+'a',[1,tokens]),sample(prefix+'b',[1,40,tokens]),
                sample(prefix+'g',[1,41,tokens]),dict(id=prefix+'o',task='ocr',image_path=str(image))]
            samples[split]=rows
            arrivals[split]=[dict(request_id=prefix+'1',sample_id=prefix+'a',alias='text#0',arrival_ns=0),
                dict(request_id=prefix+'2',sample_id=prefix+'b',alias='text#1',arrival_ns=0),
                dict(request_id=prefix+'3',sample_id=prefix+'g',alias='generalist_2#0',arrival_ns=2_000_000),
                dict(request_id=prefix+'4',sample_id=prefix+'o',alias='ocr#0',arrival_ns=4_000_000)]
            primes=[dict(r,request_id='prime-'+r['request_id'],arrival_ns=0) for r in arrivals[split] if r['alias']!='text#1']
            priming[split]=dict(samples=copy.deepcopy(rows),arrivals=primes)
        paths=[]
        for name,value in [('samples',samples),('arrivals',arrivals),('priming',priming)]:
            p=root/(name+'.json');p.write_text(json.dumps(value));paths.append(p)
        return (spec,source.parents[1],*paths)

    def prepared(self,root):
        args=self.fixture(root)
        plan=c.prepare(root/'plan',*args,BatchConfiguration(2,100,1_000_000),
            measurement_ns=20_000_000,drain_timeout_seconds=.3,supervisor_timeout_seconds=5.)
        return root/'plan/plan.json',plan,args

    def test_prepare_has_active_background_and_fresh_three_owner_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path,plan,args=self.prepared(Path(directory));self.assertEqual(c.audit(path),plan)
            deployment=c.read(c.checked(plan['deployment']))
            self.assertEqual(deployment['physical_owners'],list(c.PHYSICAL))
            self.assertEqual(deployment['logical_text_aliases'],list(c.TEXT))
            self.assertEqual(plan['partitions']['training']['requested'],4)
            self.assertEqual(plan['partitions']['training']['offered_by_physical_owner'],
                {'shared_text':2,'generalist_2#0':1,'ocr#0':1})
            for field in ('baseline_four_owner_profiles_reused','baseline_capacity_g_inherited',
                          'startup_identity_inherited','memory_identity_inherited','energy_identity_inherited'):
                self.assertFalse(deployment[field])
            self.assertIn('MISSING',deployment['prediction_profiles'])
            self.assertFalse((path.parent/'training/collection').exists())

    def test_background_arrivals_run_while_text_execute_is_blocked_and_all_offered_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path,plan,args=self.prepared(Path(directory));runtime=BackgroundRuntime();calls=[];timing={}
            def factory():
                calls.append('load');owner=backend();original=owner.execute_prepared
                def execute(rows,**kwargs):
                    calls.append('generate')
                    if len(calls)==3:
                        timing['text_start']=time.perf_counter_ns()
                        self.assertTrue(runtime.measurement_received.wait(.2),'Independent background dispatcher did not run')
                        timing['text_unblocked']=time.perf_counter_ns()
                    return original(rows,**kwargs)
                owner.execute_prepared=execute;return owner
            report=c.collect(path,'training',backend_factory=factory,background_runtime_factory=lambda:runtime)
            self.assertEqual(report['status'],'CPU_TEST_ONLY_COMPLETE')
            self.assertEqual(calls.count('load'),1)
            self.assertTrue(report['all_offered_parent_receipts']);self.assertEqual(report['offered'],4)
            self.assertEqual({r['request_id'] for r in report['results']},{'t1','t2','t3','t4'})
            measured=[r for r in runtime.records if r['phase']=='measurement']
            self.assertTrue(all(r['scheduled_time_ns']==report['epoch_ns']+r['scheduled_offset_ns'] for r in measured))
            self.assertLess(measured[0]['actual_submit_time_ns'],timing['text_unblocked'])
            self.assertTrue(all(r['completion_time_ns']<report['epoch_ns'] for r in runtime.records if r['phase']=='priming'))
            self.assertTrue(all(r['parent_receipt_ns']>=r['scheduled_arrival_ns'] for r in report['results']))
            self.assertEqual({r['physical_owner'] for r in report['results']},set(c.PHYSICAL))
            self.assertIsNone(report['resources']['measurement_energy_joules'])
            self.assertFalse(report['resources']['old_four_owner_profiles_reused'])
            self.assertFalse(report['fit_eligible']);self.assertFalse(report['GPU_started'])
            index=c.read(c.checked(report['profile_index']))
            self.assertEqual(set(index['requests']),{'t1','t2','t3','t4'})
            self.assertEqual(len(index['batches']),1)
            with self.assertRaises(FileExistsError):
                c.collect(path,'training',backend_factory=factory,background_runtime_factory=lambda:runtime)

    def test_background_failure_keeps_entire_offered_cohort_and_pending_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path,plan,args=self.prepared(Path(directory));runtime=BackgroundRuntime(fail_measurement=True)
            report=c.collect(path,'training',backend_factory=backend,background_runtime_factory=lambda:runtime)
            self.assertEqual(report['offered'],4)
            self.assertEqual(len(report['offered_cohort']),4)
            self.assertFalse(report['all_offered_parent_receipts'])
            self.assertEqual(report['status'],'INCOMPLETE_OR_FAILED_NOT_QUALIFIED')
            self.assertEqual(set(report['unresolved_request_ids']),{'t3','t4'})
            self.assertFalse(report['retry_permitted']);self.assertFalse(report['final_gain_eligible'])
            index=c.read(c.checked(report['profile_index']))
            self.assertEqual(len(index['requests']),4)
            self.assertTrue(c.read(c.checked(index['requests']['t3']))['unresolved'])

    def test_text_weight_owner_remains_resident_after_text_finishes_until_background_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            path,plan,args=self.prepared(Path(directory));runtime=BackgroundRuntime();owners=[];checks=[]
            original=runtime.drain
            def drain(records,timeout):
                if records and records[0]['phase']=='measurement':
                    gc.collect();checks.append(True)
                    self.assertIsNotNone(owners[0](),'Shared text weights released before background drain')
                return original(records,timeout)
            runtime.drain=drain
            def factory():
                owner=backend();owners.append(weakref.ref(owner));return owner
            result=c.collect(path,'training',backend_factory=factory,background_runtime_factory=lambda:runtime)
            self.assertTrue(checks);self.assertTrue(result['text_weights_retained_until_after_resource_monitor_close'])
            self.assertEqual(set(result['physical_owner_ready']),set(c.PHYSICAL))

    def test_priming_must_cover_every_background_task_and_not_reuse_request_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root);priming=c.read(args[4])
            priming['training']['arrivals'][0]['request_id']='t1';args[4].write_text(json.dumps(priming))
            with self.assertRaisesRegex(ValueError,'Priming identities'):
                c.prepare(root/'plan',*args,BatchConfiguration(2,100,0),measurement_ns=20_000_000,drain_timeout_seconds=.1,supervisor_timeout_seconds=5.)
            self.assertFalse((root/'plan').exists())

    def test_split_overlap_checks_background_media_bytes_not_only_ids_or_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root)
            (root/'validation.image').write_bytes((root/'training.image').read_bytes())
            with self.assertRaisesRegex(ValueError,'overlap'):
                c.prepare(root/'plan',*args,BatchConfiguration(2,100,0),measurement_ns=20_000_000,drain_timeout_seconds=.1,supervisor_timeout_seconds=5.)
            self.assertFalse((root/'plan').exists())

    def test_media_or_source_tamper_stops_before_factory_and_no_implicit_execute(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);path,plan,args=self.prepared(root);calls=[]
            (root/'training.image').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'Frozen source changed'):
                c.collect(path,'training',backend_factory=lambda:calls.append(1),background_runtime_factory=BackgroundRuntime)
            self.assertFalse(calls);self.assertFalse((path.parent/'training/collection').exists())
            with self.assertRaisesRegex(RuntimeError,'--execute'):c.collect('/missing','training')

    def test_measurement_energy_excludes_drain_and_memory_bills_whole_new_deployment(self):
        samples=[dict(time_ns=0,watts=10.,owned_used_mib=3),dict(time_ns=10,watts=10.,owned_used_mib=8),
            dict(time_ns=20,watts=10.,owned_used_mib=9),dict(time_ns=30,watts=10.,owned_used_mib=12)]
        result=c.resource_summary(samples,startup_ns=0,epoch_ns=10,measurement_end_ns=20,drain_end_ns=30,physical_identity='new-three-owner')
        self.assertAlmostEqual(result['startup_energy_joules'],1e-7)
        self.assertAlmostEqual(result['measurement_energy_joules'],1e-7)
        self.assertEqual(result['owned_sampled_peak_startup_through_drain_mib'],12)
        self.assertFalse(result['memory_safety_factor_applied'])
        self.assertEqual(result['physical_deployment_sha256'],'new-three-owner')

    def test_idle_background_and_arrivals_outside_window_rejected(self):
        samples=[sample('a',[1,2]),dict(id='o',task='ocr')]
        arrivals=[dict(request_id='a',sample_id='a',alias='text#0',arrival_ns=0),
            dict(request_id='o',sample_id='o',alias='ocr#0',arrival_ns=0)]
        with self.assertRaisesRegex(ValueError,'Both G2 and OCR'):c._partition(samples,arrivals)
        arrivals[0]['arrival_ns']=10
        with self.assertRaisesRegex(ValueError,'arrival'):c._partition(samples,arrivals,measurement_ns=10)

    def test_unsuccessful_text_priming_blocks_epoch_and_preserves_all_planned_requests(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);args=self.fixture(root)
            samples=c.read(args[2]);priming=c.read(args[4])
            # Actual fake backend greedily exhausts budget for last token 4.
            for row in samples['training']:
                if row['id']=='ta':row['question']='1 4'
            for row in priming['training']['samples']:
                if row['id']=='ta':row['question']='1 4'
            args[2].write_text(json.dumps(samples));args[4].write_text(json.dumps(priming))
            c.prepare(root/'plan',*args,BatchConfiguration(2,100,0),measurement_ns=20_000_000,
                drain_timeout_seconds=.3,supervisor_timeout_seconds=5.)
            runtime=BackgroundRuntime()
            report=c.collect(root/'plan/plan.json','training',backend_factory=backend,background_runtime_factory=lambda:runtime)
            self.assertIsNone(report['epoch_ns']);self.assertEqual(report['offered'],4)
            self.assertEqual(set(report['unresolved_request_ids']),{'t1','t2','t3','t4'})
            self.assertFalse(any(r['phase']=='measurement' for r in runtime.records))
            self.assertIn('Unsuccessful text priming',report['failure'])
            checkpoint=c.read(c.checked(report['checkpoint']))
            self.assertEqual(len(checkpoint['offered_cohort']),4)
            self.assertEqual(len(checkpoint['priming_cohort']),3)

    def test_external_watchdog_kills_blocked_child_and_grandchild_and_preserves_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);script=root/'hung.py'
            script.write_text('import json,os,subprocess,sys,time,signal\n'
                'from pathlib import Path\n'
                'signal.signal(signal.SIGTERM,signal.SIG_IGN)\n'
                'p=subprocess.Popen([sys.executable,"-c","import time,signal;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)"])\n'
                'Path(sys.argv[1]).write_text(json.dumps({"child":os.getpid(),"grandchild":p.pid,"offered":["a","b"],"receipt_prefix":["a"]}))\n'
                'time.sleep(20)\n')
            checkpoint=root/'checkpoint.json'
            report=c.supervise_process([sys.executable,str(script),str(checkpoint)],root,.2,grace_seconds=.05)
            self.assertTrue(report['timed_out']);self.assertTrue(report['all_process_group_members_exited'])
            self.assertIn('SIGKILL',report['signals'])
            self.assertEqual(c.read(checkpoint)['offered'],['a','b'])
            self.assertEqual(c.read(checkpoint)['receipt_prefix'],['a'])
            self.assertEqual(c._live_group_pids(report['process_group']),[])
            self.assertLess(report['elapsed_seconds'],1.)

    def test_ordinary_child_exit_does_not_leave_running_descendant(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            command=[sys.executable,'-c','import subprocess,sys;subprocess.Popen([sys.executable,"-c","import time;time.sleep(20)"])']
            report=c.supervise_process(command,root,.5,grace_seconds=.05)
            self.assertFalse(report['timed_out'])
            self.assertTrue(report['all_process_group_members_exited'])
            self.assertTrue(report['signals'])

    def test_exit_wait_keeps_ownership_boundary_until_live_group_disappears(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            with patch.object(c,'_live_group_pids',side_effect=[[123],[]]), \
                    patch.object(c.os,'killpg') as kill, \
                    patch.object(c.time,'sleep',side_effect=KeyboardInterrupt):
                c._await_group_exit(123,root)
            kill.assert_called_once_with(123,c.signal.SIGKILL)
            saved=c.read(root/'supervisor-exit-await.json')
            self.assertEqual(saved['status'],'GPU_LOCK_HELD_AWAITING_PROCESS_GROUP_EXIT')
            self.assertFalse(saved['full_cleanup_claimed'])

    def test_interrupt_during_stop_group_still_waits_for_all_child_processes(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);groups=[]
            with patch.object(c,'_stop_group',side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    c.supervise_process([sys.executable,'-c','import time;time.sleep(20)'],root,
                        .03,on_start=groups.append)
            self.assertEqual(len(groups),1)
            self.assertEqual(c._live_group_pids(groups[0]),[])
            self.assertTrue(c.read(root/'supervisor-interrupted.json')['interrupted'])

    def test_interrupt_during_exit_checkpoint_cannot_skip_exit_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);original=c._atomic_checkpoint;calls=[]
            def interrupted_once(path,value):
                calls.append(True)
                if len(calls)==1:raise KeyboardInterrupt
                return original(path,value)
            with patch.object(c,'_live_group_pids',side_effect=[[123],[123],[]]), \
                    patch.object(c,'_atomic_checkpoint',side_effect=interrupted_once), \
                    patch.object(c.os,'killpg') as kill,patch.object(c.time,'sleep'):
                c._await_group_exit(123,root)
            self.assertEqual(len(calls),2)
            kill.assert_called_once_with(123,c.signal.SIGKILL)


if __name__=='__main__':unittest.main()
