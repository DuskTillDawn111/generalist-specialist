"""CPU protocol and cache correctness tests; no real GPU calls."""
from collections import Counter
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from joint_deploy.measurement.schema import MeasurementConfig
from joint_deploy.measurement.router import Router
from deployment_core import Workload
from deployment_search.tests.fixtures import model
from deployment_search.runtime_cache import save_json,digest
from .adapter import expand,validate_memory,PlanningMemoryInfeasible,FamilyReplicaRouter
from .runtime import run
from .evaluator import ReplicaRuntimeEvaluator,PilotExecutionBlocked

CAPS={'G':['text','asr'],'S':['text']};BOUNDS={'G':2,'S':1}
MEM={'G':{'0':0.,'1':2.,'2':3.5},'S':{'0':0.,'1':1.}}
REG={m:{'python':sys.executable,'script':str(Path('joint_deploy/measurement/tests/protocol_worker.py').resolve())} for m in CAPS}


def cfg(counts=None):
    return MeasurementConfig(counts or {'G':2,'S':0},{'text':20.},{'text':{'G':20.,'S':0.}},warmup_seconds=.15,
        measurement_seconds=.4,drain_timeout_seconds=2,shutdown_timeout_seconds=1,startup_timeout_seconds=3)


def context(pool):
    value={'models':{m:{'supported_tasks':tasks,'source':{'fixture':'CPU-only'}} for m,tasks in CAPS.items()},
        'registry':REG,'replica_bounds':BOUNDS,'memory_table_mib':MEM,'gpu_capacity_mib':10.,'gpu_uuid':'cpu',
        'input_pool':str(pool),'telemetry_interval_seconds':.01,'source_hashes':{'fixture':'CPU-only'}}
    value['context_sha256']=digest(value);return value


class FakeMonitor:
    def __init__(self,runtime,interval,capacity):
        self.runtime=runtime;self.interval=interval;self.samples=[];self.errors=[];self.phase='load'
        self.foreign_interference=threading.Event()
    def start(self):pass
    def close(self):pass
    def check(self):pass
    def idle(self,phase):
        samples=[{'owned_used_mib':3.5,'used_mib':3.5,'each_owned_pid_has_cuda_context':True,'phase':phase,
            'foreign_pids':[],'processes':[]} for _ in range(3)]
        self.samples+=samples;return samples


class AdapterTests(unittest.TestCase):
    def test_expansion_independent_ids_and_family_conservation(self):
        transport,registry,mapping,memory=expand(cfg(),CAPS,BOUNDS,MEM,10.,REG)
        self.assertEqual(transport.deployment,{'G#0':1,'G#1':1})
        self.assertEqual(transport.routing_rps['text'],{'G#0':10.,'G#1':10.})
        self.assertEqual(memory,3.5);self.assertEqual(mapping,{'G#0':'G','G#1':'G'})
        self.assertEqual(set(registry),set(mapping))

    def test_round_robin_preserves_family_routing_sequence_across_tasks(self):
        routing={'text':{'G':3.,'S':1.},'asr':{'G':1.}}
        expected=Router(routing);actual=FamilyReplicaRouter(routing,{'G':2,'S':1});counts=Counter()
        for task in ['text','asr','text']*50:
            replica,family=actual.choose(task)
            self.assertEqual(family,expected.choose(task));counts[replica]+=1
        self.assertLessEqual(abs(counts['G#0']-counts['G#1']),1)

    def test_evidence_and_memory_gate_before_spawn(self):
        with patch('subprocess.Popen',side_effect=AssertionError('Must not launch')):
            with self.assertRaises(PlanningMemoryInfeasible):expand(cfg(),CAPS,BOUNDS,MEM,3.,REG)
            for bad in ({'G':3},{'S':2},{'G':True},{'G':-1},{'unknown':1}):
                with self.assertRaises(ValueError):validate_memory(bad,CAPS,BOUNDS,MEM,10.)
            with self.assertRaises(ValueError):validate_memory({'G':1},CAPS,BOUNDS,{'G':{'1':float('nan')},'S':{'0':0.}},10.)

    def test_two_real_CPU_processes_metrics_and_shutdown(self):
        with tempfile.TemporaryDirectory() as directory:
            pool=Path(directory)/'pool.jsonl';output=Path(directory)/'result.json'
            pool.write_text(json.dumps({'pool_id':'text:fixture','task_type':'text','sample':{'id':'cpu'}})+'\n')
            with patch('integer_replica_pilot.runtime.snapshot',return_value={'uuid':'cpu','processes':[],'used_mib':0.}),contextlib.redirect_stdout(io.StringIO()):
                result=run(cfg(),context(pool),output,monitor_factory=FakeMonitor,release_fn=lambda *args:{'verified':True,'CPU_fixture':True})
            self.assertTrue(result['valid_execution'],result['errors'])
            self.assertEqual(len({v['pid'] for v in result['worker_identities'].values()}),2)
            self.assertEqual(result['per_replica_request_count'],{'G#0':4,'G#1':4})
            self.assertEqual(result['aggregate_family_metrics']['G']['completed_requests'],8)
            self.assertTrue(all(not v['forced'] and v['bye_received'] and v['exit_code']==0 for v in result['shutdown'].values()))
            self.assertEqual(len(result['priming_request_records']),2)
            self.assertNotEqual(result['per_replica_worker_latency']['G#0']['mean'],None)
            self.assertEqual(result['replica_count_by_model'],{'G':2,'S':0})

    def test_cache_vectors_distinct_and_zeros_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'context.json';save_json(path,context('/unused'))
            c={'G':model('G',2.,text=1.,asr=1.),'S':model('S',1.,text=1.)}
            evaluator=ReplicaRuntimeEvaluator(Path(directory)/'cache',c,path,{},run=lambda *args:None)
            w=Workload({'text':1.});r={('text','G'):1.}
            a=evaluator.identity({'G':1},w,r);b=evaluator.identity({'G':2},w,r)
            self.assertNotEqual(digest(a),digest(b));self.assertEqual(b['deployment'],{'G':2,'S':0})

    def test_resource_failure_preserved_with_no_retry_or_fallback(self):
        calls=[]
        def failed(config,raw,log,config_path,context_path):
            calls.append(config)
            save_json(raw,dict(config,runtime_resource_status='RUNTIME_RESOURCE_INFEASIBLE',execution_status='RUNTIME_RESOURCE_INFEASIBLE',
                errors=['CUDA out of memory'],memory_observations={'peak_owned_mib':11.}))
            return {'returncode':1}
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'context.json';save_json(path,context('/unused'))
            c={'G':model('G',2.,text=1.,asr=1.),'S':model('S',1.,text=1.)}
            evaluator=ReplicaRuntimeEvaluator(Path(directory)/'cache',c,path,{'warmup_seconds':0.,'measurement_seconds':1.},run=failed)
            for _ in range(2):
                with self.assertRaises(PilotExecutionBlocked):evaluator.evaluate({'G':2},Workload({'text':1.}),{('text','G'):1.})
            self.assertEqual(len(calls),1)
            ledger=json.loads((Path(directory)/'cache/runtime_cache_manifest.json').read_text())
            self.assertEqual(len(ledger['run_attempts']),1)
            entry=next(iter(ledger['configurations'].values()))
            self.assertEqual(entry['status'],'RUNTIME_RESOURCE_INFEASIBLE')
            self.assertIsNone(entry['repetitions'][0]['runtime']['stability_state_v2'])

    def test_three_completed_unstable_repetitions_and_exact_cache_hit(self):
        calls=[]
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);pool=root/'pool.jsonl';pool.write_text(json.dumps({'pool_id':'text:fixture','task_type':'text','sample':{'id':'cpu'}})+'\n')
            path=root/'context.json';save_json(path,context(pool))
            c={'G':model('G',2.,text=1.,asr=1.),'S':model('S',1.,text=1.)}
            def cpu_runner(config,raw,log,config_path,context_path):
                calls.append(config)
                with patch('integer_replica_pilot.runtime.snapshot',return_value={'uuid':'cpu','processes':[],'used_mib':0.}),contextlib.redirect_stdout(io.StringIO()):
                    result=run(MeasurementConfig(**config),context(pool),raw,monitor_factory=FakeMonitor,release_fn=lambda *args:{'verified':True,'CPU_fixture':True})
                # A completed timed run with serving errors is valid execution, unstable serving.
                result['status']='failed';result['errors'].append('deliberate completed serving error');save_json(raw,result)
                return {'returncode':1,'timed_out':False}
            evaluator=ReplicaRuntimeEvaluator(root/'cache',c,path,{'warmup_seconds':0.,'measurement_seconds':.2,'shutdown_timeout_seconds':1.},run=cpu_runner)
            w=Workload({'text':20.});routing={('text','G'):20.}
            result=evaluator.evaluate({'G':2},w,routing)
            self.assertEqual(result.valid_repetition_count,3)
            self.assertEqual(result.stability_state_v2,'unstable')
            self.assertEqual(len(result.repetitions),3)
            self.assertEqual(result.per_replica_request_count['G#0']['values'],[2,2,2])
            self.assertEqual(result.aggregate_family_metrics['G']['request_count']['mean'],4.)
            second=evaluator.evaluate({'G':2},w,routing)
            self.assertEqual(len(calls),3);self.assertEqual(result,second)
            self.assertTrue(all(r['run_status']=='completed' for r in result.repetitions))

    def test_cache_budget_counts_reserved_failures_and_caps_six(self):
        from deployment_search.runtime_cache import RuntimeEvaluationCache
        with tempfile.TemporaryDirectory() as directory:
            with RuntimeEvaluationCache(directory,6,3).locked() as cache:
                for i in range(6):cache.reserve({'replica_vector':i})
                with self.assertRaises(RuntimeError):cache.reserve({'replica_vector':6})
                self.assertEqual(len(cache.data['configurations']),6)


if __name__=='__main__':unittest.main()
