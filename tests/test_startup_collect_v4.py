import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import time
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import startup_collect_v4 as target


class LoadingOwnershipTests(unittest.TestCase):
    def worker(self,pid):return SimpleNamespace(process=SimpleNamespace(pid=pid))

    def test_child_loading_owned_before_ready_foreign_retained(self):
        workers=[('text#0',self.worker(10)),('text#1',self.worker(20))]
        with patch.object(target,'descendant_or_self',side_effect=lambda pid,root:(pid,root) in [(11,10),(21,20)]):
            owned,foreign=target.owned_processes({11:12.,21:15.,99:1.},workers,{})
        self.assertEqual(owned,{11:'text#0',21:'text#1'})
        self.assertEqual(foreign,[99])

    def test_ready_owner_and_absent_not_filled_with_fake_zero(self):
        workers=[('text#0',self.worker(10)),('text#1',self.worker(20))]
        with patch.object(target,'descendant_or_self',return_value=False):
            owned,foreign=target.owned_processes({11:12.},workers,{'text#0':{'pid':11}})
        self.assertEqual(owned,{11:'text#0'})
        self.assertNotIn(20,owned)
        self.assertEqual(foreign,[])

    def test_ambiguous_descendant_fails(self):
        with patch.object(target,'descendant_or_self',return_value=True):
            with self.assertRaisesRegex(ValueError,'Ambiguous'):
                target.owned_processes({11:12.},[('a',self.worker(10)),('b',self.worker(20))],{})


class StartupOrderingTests(unittest.TestCase):
    def run_fake(self,root,fail=False):
        run=root/'run';run.mkdir();(root/'artifacts/static-study-v6-20261021/inputs').mkdir(parents=True)
        pool=root/'artifacts/static-study-v6-20261021/inputs/runtime-inputs.jsonl'
        pool.write_text(json.dumps({'pool_id':'p','task_type':'text','sample':{}})+'\n')
        (run/'context.json').write_text(json.dumps(dict(sources=[],execution_sources=[],worker_bindings={f:dict(command=['python','worker'],env={}) for f in set(target.RESIDENTS.values())},threads={},collection_budget=dict(free_disk_reserve_bytes=12*2**30,total_timeout_seconds=2400))))
        (run/'plan.json').write_text(json.dumps(dict(execution_context_sha256='ctx',inputs=[],jobs=[dict(id='j',context='solo',inputs={'text#0':'p'},tasks={'text#0':'text'},start_offsets_ms={'text#0':0},split='training',validation_view='training') ])))
        history=[];holder={}
        class Monitor:
            def __init__(self,runtime):self.runtime=runtime;self.phase='initial';self.samples=[];self.errors=[];holder['monitor']=self
            def sample(self):self.samples.append({'phase':self.phase,'time_ns':time.perf_counter_ns()});history.append('sample:'+self.phase)
            def check(self):pass
            def start(self):history.append('monitor_start')
            def close(self):history.append('monitor_close')
        class Runtime:
            def __init__(self,*args):
                self.ready={a:{'pid':123} for a in target.RESIDENTS};self.workers={a:SimpleNamespace(process=SimpleNamespace(pid=123)) for a in target.RESIDENTS}
                self.shutdown_report={};self.errors=[];history.append('runtime_construct')
            def start(self):
                history.append('runtime_start')
                if fail:raise RuntimeError('injected startup failure')
            def enqueue(self,row,sample):row['status']='completed'
            def drain(self,*args):pass
            def close(self):history.append('runtime_close')
        with patch.object(target,'ROOT',root),patch.object(target,'StartupBoardMonitor',Monitor),patch.object(target,'DeploymentRuntime',Runtime),patch.object(target,'context_identity',return_value='ctx'),patch.object(target,'descendant_or_self',return_value=True),patch.object(target.time,'sleep'),patch.object(target,'record',side_effect=lambda *args:dict(request_id='r',status='queued')):
            raw=target.collect(run)
        return history,raw,run

    def test_monitor_and_raw_boundaries_cover_startup_and_shutdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            history,raw,run=self.run_fake(Path(tmp))
            self.assertLess(history.index('sample:startup_begin'),history.index('runtime_construct'))
            self.assertLess(history.index('sample:pre_startup'),history.index('sample:startup_begin'))
            self.assertLess(history.index('monitor_start'),history.index('runtime_start'))
            self.assertLess(history.index('runtime_close'),history.index('monitor_close'))
            self.assertEqual([e['boundary'] for e in raw['lifecycle_boundaries']],['startup_begin','startup_end','priming_begin','priming_end','active_begin','active_end','shutdown_begin','shutdown_end'])
            saved=json.loads((run/'raw.json').read_text())
            self.assertEqual(saved['lifecycle_boundaries'],raw['lifecycle_boundaries'])
            self.assertEqual(saved['board_samples'][0]['phase'],'pre_startup')
            boundaries={e['boundary']:e['time_ns'] for e in saved['lifecycle_boundaries']}
            self.assertLessEqual(saved['board_samples'][0]['time_ns'],boundaries['startup_begin'])
            self.assertGreaterEqual(saved['board_samples'][-1]['time_ns'],boundaries['startup_end'])

    def test_failed_startup_preserves_samples_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            history,raw,run=self.run_fake(Path(tmp),fail=True)
            self.assertEqual(raw['status'],'failed')
            self.assertIn('runtime_close',history);self.assertIn('monitor_close',history)
            self.assertTrue((run/'raw.json').is_file())
            self.assertNotIn('startup_end',[e['boundary'] for e in raw['lifecycle_boundaries']])


if __name__=='__main__':unittest.main()
