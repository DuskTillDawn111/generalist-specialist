"""CPU-only supervision, original priming and genuine multi-owner SQLite checks."""
import copy
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import unified_corun_collect_v8 as collect
from tools.predictor_validation_20261027 import unified_corun_review_v8 as review
from tools.predictor_validation_20261027.trace_export import parse_sqlite
from tools.predictor_validation_20261027.smoke_verifier_v3 import validate_parsed


def inputs():
    return [dict(task=t,pool_id=t+str(i),input_size=i+1) for t in ('text','image','asr','ocr') for i in range(2)]


def jobs():
    return [dict(id='ssg',context='SSG',inputs={a:'text0' for a in ('text#0','text#1','generalist_2#0')},
        tasks={a:'text' for a in ('text#0','text#1','generalist_2#0')},start_offsets_ms={a:0 for a in ('text#0','text#1','generalist_2#0')},split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION'),
        dict(id='ocr',context='OCR',inputs={'ocr#0':'ocr0'},tasks={'ocr#0':'ocr'},start_offsets_ms={'ocr#0':0},split='training',validation_view='TRAINING_ONLY_NOT_VALIDATION')]


def context():
    return dict(worker_bindings={f:dict(command=[sys.executable,'original.py','--spec','same-spec'],env={}) for f in set(collect.RESIDENTS.values())},
        threads={'OMP_NUM_THREADS':'1'},execution_nonce='n',controller_python=sys.executable,nsys='/usr/bin/nsys',
        collection_budget=dict(free_disk_reserve_bytes=12*2**30,disk_abort_margin_bytes=2**30,incremental_storage_scenario_bytes=6*2**30,
            total_timeout_seconds=1,watchdog_interval_seconds=.02,cleanup_grace_seconds=.05))


class PlanAndDomainTests(unittest.TestCase):
    def test_audit_rejects_domain_and_parameter_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            run=Path(tmp);original=run/'original.json';parameters=run/'parameters.json'
            original.write_text(json.dumps({'inputs':inputs()}));parameters.write_text('{}')
            ctx=context();ctx.update(measurement_domain=collect.LIGHT,collection_schema=collect.RAW_SCHEMA,
                worker_variant='semantic_worker.py',child_profilers=False,capture_mode='whole_lifetime',
                affine_observation=False,profiler='single-nsys-process-tree-cuda-nvtx-gpuctxsw')
            plan=dict(schema=collect.PLAN_SCHEMA,split='training',training_only=True,domain=collect.LIGHT,
                jobs=jobs(),inputs=inputs(),paired_jobs_sha256=collect.digest(jobs()),
                original_plan=collect.receipt(original),parameters=collect.receipt(parameters))
            def write():
                plan['plan_sha256']=collect.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
                (run/'plan.json').write_text(json.dumps(plan));(run/'context.json').write_text(json.dumps(ctx))
            write()
            with patch.object(collect,'source_audit',return_value={'status':'PASS'}):
                self.assertEqual(collect.audit(run)['priming_requests'],12)
                ctx['profiler']='none-natural-with-semantic-markers';write()
                with self.assertRaisesRegex(ValueError,'profiler identity'):collect.audit(run)
                ctx['profiler']='single-nsys-process-tree-cuda-nvtx-gpuctxsw';write();parameters.write_text('{"changed":true}')
                with self.assertRaisesRegex(ValueError,'parameter receipt'):collect.audit(run)

    def test_original_priming_order_counts_and_no_child_profiler(self):
        plan={'inputs':inputs()};primers=collect.priming_items(plan)
        self.assertEqual(len(primers),12)
        self.assertEqual([a for a,i in primers],['text#0']*2+['text#1']*2+['generalist_2#0']*6+['ocr#0']*2)
        self.assertEqual([i['pool_id'] for a,i in primers if a=='generalist_2#0'],['text1','text0','image1','image0','asr1','asr0'])
        original=context();registry=collect.build_registry(original)
        self.assertEqual(original['worker_bindings']['text']['command'][1],'original.py')
        for entry in registry.values():
            self.assertTrue(entry['command'][1].endswith('/semantic_worker.py'))
            self.assertNotIn('nsys',entry['command'])
            self.assertEqual(entry['command'][2:],['--spec','same-spec'])
            self.assertEqual(entry['env'][collect.NONCE_ENV],'n')
        bad=context();bad['worker_bindings']['text']['command'].insert(0,'nsys')
        with self.assertRaises(ValueError):collect.build_registry(bad)

    def test_one_outer_session_and_no_natural_profiler(self):
        command=collect.outer_command(Path('/tmp/run'),context(),collect.LIGHT)
        for arg in ('--capture-range=none','--gpuctxsw=true','--cuda-trace-scope=process-tree','--trace-fork-before-exec=false','--wait=all'):
            self.assertIn(arg,command)
        self.assertEqual(sum(Path(x).name=='nsys' for x in command),1)
        natural=collect.outer_command(Path('/tmp/run'),context(),collect.NATURAL)
        self.assertNotIn('profile',natural);self.assertEqual(natural[0],sys.executable)

    def test_matrix_rejects_missing_owner_fields_and_nontraining(self):
        collect.validate_jobs(jobs())
        for mutate in (lambda j:j['tasks'].pop('text#0'),lambda j:j.update(split='validation'),lambda j:j['start_offsets_ms'].update({'text#0':float('nan')})):
            altered=copy.deepcopy(jobs());mutate(altered[0])
            with self.assertRaises(ValueError):collect.validate_jobs(altered)
        with self.assertRaises(ValueError):collect.priming_items({'inputs':inputs()[:-1]})

    def test_live_identity_saves_nonce_and_does_not_claim_host_namespace(self):
        identity=collect.process_identity(os.getpid())
        self.assertEqual(identity['pid'],os.getpid())
        self.assertGreater(identity['start_ticks'],0)
        self.assertEqual(identity['process_group'],os.getpgrp())
        # Linux /proc/environ describes the initial exec environment, rather
        # than subsequent in-process os.environ changes.
        code='import json,os;from tools.predictor_validation_20261027.unified_corun_collect_v8 import process_identity;print(json.dumps(process_identity(os.getpid())))'
        child=json.loads(subprocess.check_output([sys.executable,'-c',code],env={**os.environ,collect.NONCE_ENV:'owned-nonce'},text=True))
        self.assertEqual(child['execution_nonce'],'owned-nonce')
        self.assertIn('not assumed',identity['namespace_interpretation'])
        self.assertIn('error',collect.process_identity(999999999))


class WatchdogTests(unittest.TestCase):
    def test_timeout_cleans_real_cpu_child_process_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            pidfile=Path(tmp)/'child.pid'
            code="import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(60)"
            process=subprocess.Popen([sys.executable,'-c',code,str(pidfile)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            budget=dict(free_disk_reserve_bytes=0,disk_abort_margin_bytes=0,incremental_storage_scenario_bytes=2**60,total_timeout_seconds=.3,watchdog_interval_seconds=.02,cleanup_grace_seconds=.1)
            try:
                result=collect.wait_bounded(process,tmp,budget)
                self.assertEqual(result['status'],'TIMEOUT');self.assertEqual(result['cleanup']['live_survivors'],[])
                self.assertTrue(result['cleanup']['events']);self.assertTrue(pidfile.exists())
                child=int(pidfile.read_text());self.assertNotIn(child,collect.group_members(process.pid))
                self.assertIsNotNone(process.poll())
            finally:
                if collect.group_members(process.pid):os.killpg(process.pid,signal.SIGKILL)
                process.wait(timeout=3)

    def test_disk_guard_preempts_even_when_process_still_running(self):
        budget=context()['collection_budget'];process=SimpleNamespace(pid=999,poll=lambda:None)
        with patch.object(collect.shutil,'disk_usage',return_value=SimpleNamespace(free=12*2**30)),patch.object(collect,'cleanup_group',return_value={'live_survivors':[]}) as cleanup:
            result=collect.wait_bounded(process,'.',budget)
        self.assertEqual(result['status'],'DISK_MARGIN');cleanup.assert_called_once()


def make_sqlite(path):
    c=sqlite3.connect(path)
    c.executescript('''
    create table StringIds(id integer,value text);
    create table TARGET_INFO_SESSION_START_TIME(utcEpochNs integer,systemClockNs integer);
    create table DIAGNOSTIC_EVENT(severity integer,text text,globalPid integer);
    create table GPU_CONTEXT_SWITCH_EVENTS(tag integer,contextId integer,timestamp integer,globalPid integer);
    create table NVTX_EVENTS(start integer,end integer,text text,textId integer,globalTid integer);
    create table CUPTI_ACTIVITY_KIND_RUNTIME(start integer,end integer,globalTid integer,correlationId integer);
    create table CUPTI_ACTIVITY_KIND_KERNEL(start integer,end integer,globalPid integer,correlationId integer,contextId integer,
        deviceId integer,streamId integer,registersPerThread integer,staticSharedMemory integer,dynamicSharedMemory integer,
        demangledName integer,blockX integer,blockY integer,blockZ integer,gridX integer,gridY integer,gridZ integer);
    ''')
    c.execute('insert into TARGET_INFO_SESSION_START_TIME values(1000000,1000)')
    c.execute('insert into DIAGNOSTIC_EVENT values(2,?,?)',('No CUDA events collected. Does the process use CUDA?',1<<24))
    c.execute('insert into GPU_CONTEXT_SWITCH_EVENTS values(1,777,35,?)',(999<<24,))
    c.execute('insert into StringIds values(1,"kernel")')
    anchors=[];rows=[]
    for index,pid in enumerate((11,22,33,44)):
        rid='r'+str(pid);start=10+index*5;end=start+100;tid=(pid<<24)+1
        c.execute('insert into NVTX_EVENTS values(?,?,?,?,?)',(start,end,'GS_STAGE:'+rid+':gpu:inference',None,tid))
        # Same correlationId in all four processes must not cross attribute.
        c.execute('insert into CUPTI_ACTIVITY_KIND_RUNTIME values(?,?,?,1)',(start+1,start+2,tid))
        c.execute('insert into CUPTI_ACTIVITY_KIND_KERNEL values(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',(start+3,start+20,pid<<24,1,1,0,7,32,0,0,1,32,1,1,1,1,1))
        anchors.append(dict(pid=pid,request_id=rid,kind='gpu',segment_id='inference',start_ns=start+1001,end_ns=end+999))
        rows.append(dict(request_id=rid,worker_response=dict(start_perf_ns=start+1000,end_perf_ns=end+1000)))
    c.commit();c.close();return anchors,rows


class UnifiedTraceTests(unittest.TestCase):
    def test_four_owners_same_clock_and_scoped_correlations_parsed_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite';anchors,rows=make_sqlite(path)
            parsed=parse_sqlite(path,anchors);metadata=review.trace_metadata(path)
            self.assertEqual(metadata['issues'],[]);self.assertEqual(metadata['switch_pid_values'],[999])
            self.assertEqual(metadata['owned_scheduler_attribution'],'UNKNOWN')
            self.assertEqual(len(metadata['session']),1)
            for anchor,row in zip(anchors,rows):
                issues,summary=validate_parsed(review.owner_view(parsed,anchor['pid']),[anchor],[row],anchor['pid'])
                self.assertEqual(issues,[]);self.assertEqual(summary['kernel_counts'][row['request_id']],1)
                phase=review.owner_view(parsed,anchor['pid'])['phases'][0]
                self.assertEqual(phase['nsys_start_ns'],anchor['start_ns']-1001)
            canonical=review.canonical_measured(parsed,{'path':'raw','sha256':'receipt'})
            self.assertNotIn('nvtx_ranges',canonical);self.assertNotIn('unattributed_kernels',canonical)
            self.assertEqual(len(canonical['phases']),4)
            self.assertIsNone(metadata['calibrated_cross_clock_error_bound_ns'])

    def test_data_loss_and_missing_scheduler_records_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite';make_sqlite(path)
            c=sqlite3.connect(path);c.execute('insert into DIAGNOSTIC_EVENT values(2,"Dropped CUDA events",0)');c.execute('delete from GPU_CONTEXT_SWITCH_EVENTS');c.commit();c.close()
            issues=review.trace_metadata(path)['issues']
            self.assertIn('TRACE_ERROR_OR_DATA_LOSS_DIAGNOSTIC',issues)
            self.assertIn('GPU_CONTEXT_SWITCH_RECORDS_REQUIRED_FOR_UNIFIED_CAPABILITY',issues)

    def test_one_missing_owner_kernel_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'trace.sqlite';anchors,rows=make_sqlite(path)
            c=sqlite3.connect(path);c.execute('delete from CUPTI_ACTIVITY_KIND_KERNEL where globalPid=?',(44<<24,));c.commit();c.close()
            parsed=parse_sqlite(path,anchors)
            issues,_=validate_parsed(review.owner_view(parsed,44),[anchors[-1]],[rows[-1]],44)
            self.assertIn('REAL_KERNEL_REQUEST_COVERAGE_FAILED',issues)


class CohortTests(unittest.TestCase):
    def test_strict_gates_reject_false_domain_and_missing_original_priming(self):
        ctx=context();plan=dict(inputs=inputs(),jobs=jobs(),domain=collect.LIGHT,execution_context_sha256='ctx')
        ready={a:{'pid':100+i} for i,a in enumerate(collect.RESIDENTS)}
        rows=[dict(request_id='r'+str(i),profile_job_id=j['id'],model=a,pool_id=p,task=j['tasks'][a],split='training')
            for i,(j,a,p) in enumerate((j,a,p) for j in plan['jobs'] for a,p in j['inputs'].items())]
        raw=dict(schema=collect.RAW_SCHEMA,status='completed',execution_context_sha256='ctx',measurement_domain=collect.LIGHT,
            child_profilers=False,profile_controls_sent=False,performance_scope='single_session_profiler_perturbed',
            child_commands={a:e['command'] for a,e in collect.build_registry(ctx).items()},ready=ready,
            shutdown={a:dict(exit_code=0,forced=False,bye_received=True) for a in ready},
            live_owned_processes={a:dict(pid=r['pid'],start_ticks=1,nspid=[r['pid']],execution_nonce='n') for a,r in ready.items()},
            live_identity_events=[dict(bridge_status='UNKNOWN')],request_records=rows,
            priming_request_records=[dict(model=a,task=i['task'],pool_id=i['pool_id'],status='completed') for a,i in collect.priming_items(plan)],
            batches=[dict(job_id=j['id'],request_ids=[r['request_id'] for r in rows if r['profile_job_id']==j['id']]) for j in plan['jobs']])
        executed=dict(trace=True,single_session=True,capture_mode='whole_lifetime',child_profilers=False,profile_controls_sent=False,jobs=plan['jobs'])
        execution=dict(schema='gsdeploy.unified_corun_execution.v8',status='COLLECTED',domain=collect.LIGHT,
            watchdog_interval_seconds=.02,free_bytes_at_end=20*2**30,
            watchdog=dict(status='EXITED',return_code=0,cleanup=dict(live_survivors=[]),samples=[dict(free_bytes=20*2**30)]))
        with patch.object(review,'validate_startup_telemetry',return_value=[]):
            self.assertEqual(review.collection_issues(plan,ctx,raw,executed,execution),[])
            raw['performance_scope']='unprofiled_with_phase_markers'
            self.assertIn('PROFILER_DOMAIN_OR_CONTROL_MISMATCH',review.collection_issues(plan,ctx,raw,executed,execution))
            raw['performance_scope']='single_session_profiler_perturbed';raw['priming_request_records'].pop()
            self.assertIn('EXACT_ORIGINAL_TWELVE_PRIMING_REQUESTS_REQUIRED',review.collection_issues(plan,ctx,raw,executed,execution))

    def test_cpu_fake_cohort_preserves_twelve_priming_and_four_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);run=root/'run';run.mkdir();poolpath=root/'artifacts/static-study-v6-20261021/inputs';poolpath.mkdir(parents=True)
            pool=[dict(pool_id=i['pool_id'],task_type=i['task'],sample={}) for i in inputs()]
            (poolpath/'runtime-inputs.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in pool))
            ctx=context();plan=dict(inputs=inputs(),jobs=jobs(),domain=collect.LIGHT,execution_context_sha256='ctx')
            (run/'context.json').write_text(json.dumps(ctx));(run/'plan.json').write_text(json.dumps(plan))
            history=[]
            class Monitor:
                def __init__(self,runtime):self.runtime=runtime;self.samples=[];self.errors=[];self.identity_events=[{'bridge_status':'UNKNOWN'}];self.phase='initial'
                def sample(self):self.samples.append({'phase':self.phase,'time_ns':time.perf_counter_ns()})
                def check(self):pass
                def start(self):history.append('monitor_start')
                def close(self):history.append('monitor_close')
            class Runtime:
                def __init__(self,config,registry):
                    history.append('construct_runtime');self.ready={a:{'pid':i+100} for i,a in enumerate(collect.RESIDENTS)}
                    self.workers={a:SimpleNamespace(process=SimpleNamespace(pid=self.ready[a]['pid'])) for a in collect.RESIDENTS}
                    self.shutdown_report={};self.errors=[]
                    self.registry=registry
                def start(self):history.append('start_runtime')
                def enqueue(self,row,sample):history.append(('enqueue',row['model'],row['task'],row['pool_id']));row['status']='completed'
                def drain(self,*args):pass
                def close(self):history.append('close_runtime')
            def fake_record(kind,index,task,alias,item,target):return dict(request_id=kind+str(index),model=alias,task=task,pool_id=item['pool_id'],status='queued')
            with patch.object(collect,'ROOT',root),patch.object(collect,'audit'),patch.object(collect,'IdentityMonitor',Monitor),patch.object(collect,'DeploymentRuntime',Runtime),patch.object(collect,'descendant_or_self',return_value=True),patch.object(collect,'record',side_effect=fake_record),patch.object(collect,'process_identity',return_value={'nonce':'test'}),patch.object(collect.time,'sleep'),patch.dict(os.environ,{collect.NONCE_ENV:'n'}):
                raw=collect.cohort(run)
            self.assertEqual(raw['status'],'completed');self.assertEqual(len(raw['priming_request_records']),12);self.assertEqual(len(raw['request_records']),4)
            self.assertEqual(raw['performance_scope'],'single_session_profiler_perturbed')
            self.assertFalse(raw['profile_controls_sent']);self.assertFalse(raw['child_profilers'])
            self.assertLess(history.index('monitor_start'),history.index('construct_runtime'))
            self.assertLess(history.index('close_runtime'),history.index('monitor_close'))
            self.assertEqual([e['boundary'] for e in raw['lifecycle_boundaries']],['startup_begin','startup_end','priming_begin','priming_end','active_begin','active_end','shutdown_begin','shutdown_end'])
            executed=json.loads((run/'executed-jobs.json').read_text());self.assertTrue(executed['single_session'])


if __name__=='__main__':unittest.main()
