"""Behavioral tests for finite background coverage, shared G2 tape and failures."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
run=module('background_run',ROOT/'tools/spillover_background_20261022/run.py')
prep=module('background_prepare',ROOT/'tools/spillover_background_20261022/prepare.py')


class Monitor:
    def check(self):pass


class Runtime:
    def __init__(self,delays,fail_role=None):
        self.delays=delays;self.fail_role=fail_role;self.lock=threading.RLock();self.threads=[];self.submitted=[];self.active={};self.peak={}
    def enqueue(self,row,sample):
        alias=row['model'];self.active[alias]=self.active.get(alias,0)+1;self.peak[alias]=max(self.peak.get(alias,0),self.active[alias])
        row.update(status='submitted',actual_submit_time_ns=time.perf_counter_ns());self.submitted.append(row)
        def complete():
            time.sleep(self.delays[alias])
            with self.lock:
                self.active[alias]-=1;row.update(status='error' if row['role']==self.fail_role else 'completed',completion_time_ns=time.perf_counter_ns())
        thread=threading.Thread(target=complete);thread.start();self.threads.append(thread)
    def join(self):
        for thread in self.threads:thread.join()


def small_rosters(g2_count=6,ocr_count=2,cap=200):
    aliases=['text#0','text#1','generalist_2#0','ocr#0'];primary={a:[] for a in aliases};support={}
    for alias,count in [('generalist_2#0',g2_count),('ocr#0',ocr_count)]:
        for i in range(count):
            task=('image' if i%3!=2 else 'asr') if alias=='generalist_2#0' else 'ocr'
            primary[alias].append(dict(request_id=f'measurement:{alias}:{i}',model=alias,task=task,role='primary',pool_id='p',sequence=i,arrival_index=i))
    for alias in aliases[:-1]:
        support[alias]=[dict(request_id=f'support:{alias}:{i}',model=alias,task='text',role='support',pool_id='p',sequence=len(primary[alias])+i,arrival_index=len(primary[alias])+i,support_sequence=i) for i in range(cap)]
    rows=[run.new_record(r,'measurement') for templates in primary.values() for r in templates]
    return primary,support,rows,{'p':{'sample':{}}}


class BackgroundLoopTests(unittest.TestCase):
    def execute(self,delays,g2=6,ocr=2,cap=200):
        p,s,rows,samples=small_rosters(g2,ocr,cap);rt=Runtime(delays);meta={}
        run.background_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=3,request_timeout_seconds=1,support_every=2,metadata=meta)
        rt.join();return p,s,rows,rt,meta

    def test_g2_serial_tape_inserts_support_after_each_primary_block(self):
        p,s,rows,rt,meta=self.execute({'text#0':.003,'text#1':.004,'generalist_2#0':.009,'ocr#0':.002})
        g2=[r for r in rt.submitted if r['model']=='generalist_2#0']
        self.assertEqual([r['role'] for r in g2],['primary','primary','support','primary','primary','support','primary','primary'])
        self.assertEqual(meta['generalist_inserted_support_at_primary_count'],[2,4])
        self.assertEqual(rt.peak,{a:1 for a in p})
        self.assertEqual(meta['primary_completed_by_alias'],{a:len(p[a]) for a in p})
        self.assertTrue(all(r['status']=='completed' for r in rows))
        self.assertEqual(meta['primary_completion_time_ns'],max(r['completion_time_ns'] for r in rows if r['role']=='primary'))
        self.assertTrue(all(r['scheduled_time_ns']<meta['support_admission_stop_time_ns'] for r in rows if r['role']=='support'))
        self.assertGreaterEqual(meta['end_time_ns'],max(r['completion_time_ns'] for r in rows))
        self.assertEqual([r['admission_sequence'] for r in g2],list(range(len(g2))))

    def test_ocr_has_no_support_and_idles_resident_after_primary(self):
        p,s,rows,rt,meta=self.execute({'text#0':.003,'text#1':.004,'generalist_2#0':.009,'ocr#0':.002})
        self.assertEqual(len([r for r in rt.submitted if r['model']=='ocr#0']),2)
        self.assertEqual(meta['support_released_by_alias']['ocr#0'],0)
        self.assertLess(meta['alias_completion_time_ns']['ocr#0'],meta['primary_completion_time_ns'])

    def test_g2_maintains_text_support_if_ocr_finishes_later(self):
        p,s,rows,rt,meta=self.execute({'text#0':.003,'text#1':.004,'generalist_2#0':.002,'ocr#0':.04},g2=2,ocr=2)
        g2=[r for r in rt.submitted if r['model']=='generalist_2#0']
        self.assertEqual([r['role'] for r in g2[:2]],['primary','primary'])
        self.assertTrue(all(r['role']=='support' for r in g2[2:]));self.assertGreater(len(g2),2)

    def test_support_cap_fails_without_extending_or_idle_tail(self):
        p,s,rows,samples=small_rosters(cap=1);rt=Runtime({'text#0':.002,'text#1':.002,'generalist_2#0':.03,'ocr#0':.03});meta={}
        with self.assertRaisesRegex(RuntimeError,'Support cap exhausted'):
            run.background_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=2,request_timeout_seconds=1,metadata=meta)
        rt.join();self.assertIsNone(meta['primary_completion_time_ns'])
        self.assertEqual(meta['support_released_by_alias']['text#0'],1)
        self.assertTrue(any(r['status']=='planned' and r['scheduled_time_ns'] is None for r in rows))

    def test_native_support_failure_preserves_registered_background(self):
        p,s,rows,samples=small_rosters();rt=Runtime({'text#0':.002,'text#1':.003,'generalist_2#0':.02,'ocr#0':.02},fail_role='support')
        with self.assertRaisesRegex(RuntimeError,'request failure: support:'):
            run.background_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=2,request_timeout_seconds=1)
        rt.join();self.assertEqual(sum(r['role']=='primary' for r in rows),8)

    def test_wall_timeout_includes_pending_support_drain(self):
        class SlowSupport(Runtime):
            def enqueue(self,row,sample):
                if row['model'].startswith('text#'):self.delays[row['model']]=.08
                super().enqueue(row,sample)
        p,s,rows,samples=small_rosters(g2_count=1,ocr_count=1)
        rt=SlowSupport({'text#0':.002,'text#1':.002,'generalist_2#0':.005,'ocr#0':.005});meta={}
        with self.assertRaisesRegex(TimeoutError,'including support drain'):
            run.background_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=.04,request_timeout_seconds=1,metadata=meta)
        rt.join();self.assertIsNotNone(meta['primary_completion_time_ns'])

    def test_terminal_request_still_checked_against_request_deadline(self):
        class SlowProgress:
            def __call__(self):time.sleep(.02)
        p,s,rows,samples=small_rosters(g2_count=1,ocr_count=1)
        rt=Runtime({a:.01 for a in p})
        with self.assertRaisesRegex(TimeoutError,'Request timeout'):
            run.background_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=1,request_timeout_seconds=.005,progress=SlowProgress())
        rt.join()

    def test_final_memory_sample_before_close_even_if_check_fails(self):
        class Events:
            def __init__(self):self.events=[]
            def sample(self):self.events.append('sample')
            def check(self):self.events.append('check');raise RuntimeError('sample flagged')
            def close(self):self.events.append('close')
        monitor=Events()
        with self.assertRaises(RuntimeError):run.stop_memory_with_final_sample(monitor)
        self.assertEqual(monitor.events,['sample','check','close'])

    def test_failed_preflight_persists_and_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'attempt1'
            with patch.object(run,'validate_packet',side_effect=ValueError('tamper')):
                raw=run.run_job({'aliases':{},'resident_deployment':{}},{'job_id':'j'},out,'hash')
            self.assertEqual(raw['status'],'failed');self.assertTrue((out/'run.json').exists())
            with self.assertRaises(FileExistsError):run.run_job({'aliases':{},'resident_deployment':{}},{'job_id':'j'},out,'hash')


class RosterTests(unittest.TestCase):
    def setUp(self):self.pools={t:[f'{t}{i}' for i in range(n)] for t,n in [('text',32),('image',64),('asr',32),('ocr',32)]}
    def test_complete_background_pool_repetitions_and_reversed_task_tape(self):
        jobs=prep.build_jobs(self.pools)
        self.assertEqual([(j['block'],j['seed']) for j in jobs],[(1,105),(2,106)])
        self.assertEqual(jobs,prep.build_jobs(self.pools))
        for job in jobs:
            self.assertEqual(run.job_digest(job),job['job_identity_sha256'])
            self.assertEqual(sum(map(len,job['planned_requests'].values())),384)
            self.assertEqual(len(job['planned_requests']['generalist_2#0']),288)
            for alias,rows in job['planned_requests'].items():
                for task in {r['task'] for r in rows}:
                    values=[r for r in rows if r['task']==task];n=len(self.pools[task]);self.assertEqual(len(values),3*n)
                    for i in range(0,len(values),n):self.assertEqual({r['pool_id'] for r in values[i:i+n]},set(self.pools[task]))
        self.assertEqual([r['task'] for r in jobs[0]['planned_requests']['generalist_2#0'][:3]],['image','image','asr'])
        self.assertEqual([r['task'] for r in jobs[1]['planned_requests']['generalist_2#0'][:3]],['asr','image','image'])

    def test_support_and_warmup_cursors_never_consume_primary(self):
        job=prep.build_jobs(self.pools)[0];original=copy.deepcopy(job)
        self.assertEqual(set(job['support_requests']),{'text#0','text#1','generalist_2#0'})
        for alias,rows in job['support_requests'].items():
            self.assertEqual(len(rows),1024)
            for i in range(0,1024,32):self.assertEqual({r['pool_id'] for r in rows[i:i+32]},set(self.pools['text']))
        job['warmup_requests']['ocr#0'][0]['pool_id']='changed'
        self.assertEqual(job['planned_requests'],original['planned_requests'])
        self.assertEqual(job['support_requests'],original['support_requests'])

    def test_cpu_check_builds_valid_native_packet_without_writing_or_GPU(self):
        result=prep.prepare(check_only=True);self.assertEqual(len(result['jobs']),2)
        packet=json.loads((prep.OLD/'experiment-packet.json').read_text());packet.update(protocol=run.PROTOCOL,jobs=result['jobs'],
            support_limit_per_alias=1024,generalist_support_every_primary=8,coverage_policy=dict(minimum_common_handler_fraction=.9,
            minimum_covered_repetitions_per_input=2,cross_block_per_input_median_ratio_min=.5,cross_block_per_input_median_ratio_max=2.))
        source=Path(run.__file__).resolve();packet['bindings'][str(source)]=run.digest(source)
        packet['bindings_sha256']=__import__('hashlib').sha256(run.canonical(packet['bindings'])).hexdigest()
        self.assertEqual(run.validate_packet(packet)['jobs'],2)
        broken=copy.deepcopy(packet);broken['jobs'][0]['support_requests']['text#0'][0]['task']='image'
        broken['jobs'][0]['job_identity_sha256']=run.job_digest(broken['jobs'][0])
        with self.assertRaisesRegex(ValueError,'Invalid support'):run.validate_packet(broken)
        self.assertEqual(result['risk_check']['generalist_inserted_text_support_requests'],35)
        self.assertIsNone(result['risk_check']['formal_rho']);self.assertFalse(result['GPU_started'])


if __name__=='__main__':unittest.main()
