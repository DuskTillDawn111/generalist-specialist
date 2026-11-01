"""Behavioral contracts for symmetric support, global stop and bounded failures."""
import importlib.util
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('sustained_run',ROOT/'tools/spillover_capacity_20261022/run_sustained.py')
run=importlib.util.module_from_spec(spec);spec.loader.exec_module(run)
spec=importlib.util.spec_from_file_location('sustained_prepare',ROOT/'tools/spillover_capacity_20261022/prepare_sustained.py')
prepare=importlib.util.module_from_spec(spec);spec.loader.exec_module(prepare)


class Monitor:
    def check(self): pass


class Runtime:
    def __init__(self,delays,fail_role=None):
        self.delays=delays;self.lock=threading.RLock();self.threads=[];self.submitted=[];self.active={};self.peak={};self.fail_role=fail_role
    def enqueue(self,row,sample):
        alias=row['model'];self.active[alias]=self.active.get(alias,0)+1
        self.peak[alias]=max(self.peak.get(alias,0),self.active[alias])
        row.update(status='submitted',actual_submit_time_ns=time.perf_counter_ns());self.submitted.append(row)
        def complete():
            time.sleep(self.delays[alias])
            with self.lock:
                self.active[alias]-=1
                row.update(status='error' if self.fail_role==row['role'] else 'completed',completion_time_ns=time.perf_counter_ns())
        thread=threading.Thread(target=complete);thread.start();self.threads.append(thread)
    def join(self):
        for thread in self.threads: thread.join()


def rosters(aliases,n=3,cap=40):
    primary={a:[dict(model=a,task='text',pool_id=str(i),sequence=i,arrival_index=i,role='primary',request_id=f'measurement:{a}:{i}') for i in range(n)] for a in aliases}
    support={a:[dict(model=a,task='text',pool_id=str(i%n),sequence=n+i,arrival_index=n+i,support_sequence=i,role='support',request_id=f'support:{a}:{i}') for i in range(cap)] for a in aliases}
    records=[run.new_record(r,'measurement') for rows in primary.values() for r in rows]
    samples={str(i):{'sample':{}} for i in range(n)}
    return primary,support,records,samples


class SustainedLoopTests(unittest.TestCase):
    def exercise(self,fast_alias):
        aliases=['text#0','text#1','generalist_2#0']
        rt=Runtime({a:.004 if a==fast_alias else .016 for a in aliases})
        primary,support,rows,samples=rosters(aliases);meta={}
        run.sustained_loop(rt,primary,support,samples,rows,Monitor(),timeout_seconds=2,request_timeout_seconds=1,metadata=meta)
        rt.join()
        self.assertTrue(all(r['status']=='completed' for r in rows))
        self.assertEqual(meta['primary_completed_by_alias'],{a:3 for a in aliases})
        self.assertGreater(meta['support_released_by_alias'][fast_alias],0)
        self.assertEqual(rt.peak,{a:1 for a in aliases})
        self.assertEqual(meta['primary_completion_time_ns'],max(r['completion_time_ns'] for r in rows if r['role']=='primary'))
        self.assertGreaterEqual(meta['support_admission_stop_time_ns'],meta['primary_completion_time_ns'])
        self.assertGreaterEqual(meta['end_time_ns'],max(r['completion_time_ns'] for r in rows))
        self.assertTrue(all(r['scheduled_time_ns']<meta['support_admission_stop_time_ns'] for r in rows if r['role']=='support'))
        self.assertEqual(len({r['request_id'] for r in rows}),len(rows))
        for alias in aliases:
            self.assertEqual([r['pool_id'] for r in rows if r['role']=='primary' and r['model']==alias],['0','1','2'])
            released=[r for r in rows if r['role']=='support' and r['model']==alias]
            self.assertEqual([r['support_sequence'] for r in released],list(range(len(released))))
        return meta
    def test_specialist_finishes_early_and_supports_until_global_primary_done(self): self.exercise('text#0')
    def test_generalist_support_is_symmetric_when_it_finishes_early(self): self.exercise('generalist_2#0')
    def test_single_alias_never_submits_support(self):
        rt=Runtime({'generalist_2#0':.004});p,s,rows,samples=rosters(['generalist_2#0']);meta={}
        run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=1,request_timeout_seconds=.5,metadata=meta)
        rt.join();self.assertEqual(len(rows),3);self.assertEqual(meta['support_released_by_alias'],{'generalist_2#0':0})
    def test_support_cap_fails_without_idle_tail_or_extending_roster(self):
        rt=Runtime({'text#0':.002,'generalist_2#0':.06});p,s,rows,samples=rosters(list(rt.delays),n=2,cap=1);meta={}
        with self.assertRaisesRegex(RuntimeError,'Support cap exhausted'):
            run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=1,request_timeout_seconds=.5,metadata=meta)
        rt.join();self.assertEqual(meta['support_released_by_alias']['text#0'],1)
        self.assertTrue(any(r['status']=='planned' and r['scheduled_time_ns'] is None for r in rows))
        self.assertIsNone(meta['primary_completion_time_ns'])
    def test_support_failure_stops_and_keeps_all_primary_records(self):
        rt=Runtime({'text#0':.003,'generalist_2#0':.03},fail_role='support');p,s,rows,samples=rosters(list(rt.delays))
        with self.assertRaisesRegex(RuntimeError,'request failure: support:'):
            run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=1,request_timeout_seconds=.5)
        rt.join();self.assertEqual(sum(r['role']=='primary' for r in rows),6)
        self.assertTrue(any(r['status']=='error' and r['role']=='support' for r in rows))
    def test_wall_timeout_includes_pending_support_drain(self):
        class SlowSupport(Runtime):
            def enqueue(self,row,sample):
                if row['role']=='support': self.delays[row['model']]=.08
                super().enqueue(row,sample)
        rt=SlowSupport({'text#0':.003,'generalist_2#0':.02});p,s,rows,samples=rosters(list(rt.delays),n=1);meta={}
        with self.assertRaisesRegex(TimeoutError,'including support drain'):
            run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=.04,request_timeout_seconds=.3,metadata=meta)
        self.assertIsNotNone(meta['primary_completion_time_ns'])
        rt.join()
    def test_failed_preflight_persists_and_does_not_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/'attempt1'
            with patch.object(run,'validate_packet',side_effect=ValueError('tamper')):
                raw=run.run_job({'aliases':{},'resident_deployment':{}},{'job_id':'j'},out,'hash')
            self.assertEqual(raw['status'],'failed');self.assertTrue((out/'run.json').is_file())
            with self.assertRaises(FileExistsError): run.run_job({'aliases':{},'resident_deployment':{}},{'job_id':'j'},out,'hash')
    def test_terminal_response_after_request_deadline_still_fails(self):
        class Late(Runtime):
            def enqueue(self,row,sample):
                row.update(status='completed',actual_submit_time_ns=row['scheduled_time_ns'],completion_time_ns=row['scheduled_time_ns']+2_000_000_000)
        rt=Late({'generalist_2#0':0});p,s,rows,samples=rosters(list(rt.delays),n=1)
        with self.assertRaisesRegex(TimeoutError,'Request timeout'):
            run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=3,request_timeout_seconds=1)
    def test_all_primary_completed_together_does_not_trigger_zero_support_cap(self):
        class Immediate(Runtime):
            def enqueue(self,row,sample):
                row.update(status='completed',actual_submit_time_ns=time.perf_counter_ns(),completion_time_ns=time.perf_counter_ns())
        rt=Immediate({'text#0':0,'generalist_2#0':0});p,s,rows,samples=rosters(list(rt.delays),n=1,cap=0);meta={}
        run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=1,request_timeout_seconds=.5,metadata=meta)
        self.assertEqual(len(rows),2);self.assertEqual(meta['support_released_by_alias'],{'text#0':0,'generalist_2#0':0})
    def test_progress_snapshot_time_counts_toward_measurement_wall(self):
        class Immediate(Runtime):
            def enqueue(self,row,sample): row.update(status='completed',completion_time_ns=time.perf_counter_ns())
        rt=Immediate({'generalist_2#0':0});p,s,rows,samples=rosters(list(rt.delays),n=1)
        def slow_progress(): time.sleep(.02)
        with self.assertRaisesRegex(TimeoutError,'progress snapshots'):
            run.sustained_loop(rt,p,s,samples,rows,Monitor(),timeout_seconds=.01,request_timeout_seconds=.1,progress=slow_progress)
    def test_warmup_terminal_late_response_fails_request_deadline(self):
        class Late(Runtime):
            def enqueue(self,row,sample): row.update(status='completed',completion_time_ns=row['scheduled_time_ns']+2_000_000_000)
        rt=Late({'generalist_2#0':0});p,s,rows,samples=rosters(list(rt.delays),n=1)
        with self.assertRaisesRegex(TimeoutError,'Request timeout'):
            run.closed_loop(rt,p,samples,[],Monitor(),phase='warmup',timeout_seconds=3,
                request_timeout_seconds=1,warmup_seconds=.01)


class RosterTests(unittest.TestCase):
    def test_primary_matches_across_state_and_support_has_separate_balanced_cursor(self):
        ids=[str(i) for i in range(32)];jobs=prepare.build_jobs(ids)
        self.assertEqual([(j['block'],j['state'],j['seed']) for j in jobs],[(1,'B',103),(1,'C',103),(2,'C',104),(2,'B',104)])
        for left,right in [(jobs[0],jobs[1]),(jobs[2],jobs[3])]: self.assertEqual(left['input_pool_ids'],right['input_pool_ids'])
        for job in jobs:
            self.assertEqual(job['job_identity_sha256'],run.job_digest(job))
            for a,rows in job['support_requests'].items():
                self.assertEqual(len(rows),1024)
                self.assertNotEqual([r['pool_id'] for r in rows[:96]],job['input_pool_ids'])
                for i in range(0,1024,32): self.assertEqual({r['pool_id'] for r in rows[i:i+32]},set(ids))
        self.assertEqual(jobs,prepare.build_jobs(ids))


if __name__=='__main__': unittest.main()
