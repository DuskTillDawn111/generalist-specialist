"""CPU control-path checks; fake hardware is not native measurement evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import known_input_native_flow_v11 as flow
from tools.predictor_validation_20261027.known_input_observed_v5 import validate_plan as observed_validate_plan


class Runtime:
    def __init__(self):
        self.ready={a:dict(pid=os.getpid()) for a in flow.RESIDENTS}
        self.workers={a:types.SimpleNamespace(process=types.SimpleNamespace(pid=os.getpid())) for a in flow.RESIDENTS}
        self.errors=[];self.shutdown_report={};self.admission=None
        self.measured=[];self.drains=[]
    def start(self):pass
    def close(self):pass
    def enqueue(self,row,sample):
        now=time.perf_counter_ns()
        if row['phase']=='measurement':
            assert now>=row['scheduled_time_ns']
            self.measured.append(copy.deepcopy(row))
        row.update(status='completed',completion_time_ns=now,
            observed_worker_start_time_ns=now,response_reader_event_enqueue_time_ns=now,
            worker_response=dict(task=row['task'],prediction='',success=True,start_perf_ns=now,end_perf_ns=now))
    def drain(self,rows,timeout):
        if rows and rows[0]['phase']=='measurement':self.drains.append((len(rows),len(self.measured)))


class Monitor:
    def __init__(self,runtime):self.samples=[];self.errors=[]
    def sample(self):
        self.samples.append(dict(time_ns=time.perf_counter_ns(),watts=100.,owned_used_mib=1000.,
            owned_aliases=list(flow.RESIDENTS),foreign_pids=[]))
    def start(self):pass
    def check(self):pass
    def close(self):self.sample()


class FreshFlowTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        profile=self.root/'profile';profile.mkdir()
        pool=[json.loads(line) for line in (flow.BASE/'runtime-inputs.jsonl').read_text().splitlines()]
        inputs=[dict(task=task,pool_id=row['pool_id'],input_size=i+1)
                for task in ('text','image','asr','ocr')
                for i,row in enumerate([r for r in pool if r['task_type']==task][:2])]
        source=profile/'snapshot.py';source.write_text('# CPU context fixture\n')
        context=dict(residents=flow.RESIDENTS,sources=[flow.receipt(source)],
            execution_sources=[dict(original_path=str(source),frozen=flow.receipt(source))],
            worker_bindings={f:dict(command=['python','worker.py'],env={}) for f in set(flow.RESIDENTS.values())},
            threads={},worker_variant='semantic_worker.py')
        flow.save(profile/'context.json',context)
        flow.save(profile/'plan.json',dict(execution_context_sha256=flow.context_identity(context),inputs=inputs,
            input_assignments=[dict(task=r['task'],pool_id=r['pool_id'],view='known_input') for r in inputs]))
        self.plan=flow.prepare(profile,self.root/'prepared',duration=.01,text_rate=300.,background_rate=300.)
        for candidate in self.plan['candidates']:
            candidate['routes']={f'fresh-v11:r{i:03d}':candidate['routes'][row['request_id']]
                                  for i,row in enumerate(self.plan['trace'])}
        for i,row in enumerate(self.plan['trace']):row['request_id']=f'fresh-v11:r{i:03d}'
        self.path=self.root/'fresh-plan.json';self.write_plan()

    def write_plan(self):
        self.plan['trace_sha256']=flow.digest(self.plan['trace'])
        self.plan.pop('plan_sha256',None);self.plan['plan_sha256']=flow.digest(self.plan)
        self.path.write_text(json.dumps(self.plan))

    def assert_pre_runtime_rejection(self,exception):
        with patch.object(flow,'DeploymentRuntime') as runtime:
            with self.assertRaises(exception):flow.execute(self.path,'shared_balanced',self.root/'out')
            runtime.assert_not_called()
        self.assertFalse((self.root/'out').exists())

    def test_exact_fresh_ids_routes_full_cohort_and_immutable_native_output(self):
        runtime=Runtime()
        with patch.object(flow,'DeploymentRuntime',return_value=runtime),patch.object(flow,'BoardMonitor',Monitor),patch.object(flow.fcntl,'flock'):
            result=flow.execute(self.path,'shared_balanced',self.root/'out')
        candidate=self.plan['candidates'][0]
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(runtime.measured),len(self.plan['trace']))
        self.assertEqual(runtime.drains,[(len(self.plan['trace']),len(self.plan['trace']))])
        for offered,row in zip(self.plan['trace'],runtime.measured):
            self.assertEqual(row['request_id'],offered['request_id'])
            self.assertEqual(row['model'],candidate['routes'][offered['request_id']])
            self.assertEqual(row['scheduled_time_ns'],result['phase_start_ns']+offered['offset_ns'])
        native=json.loads((self.root/'out/raw.json').read_text())
        self.assertNotIn('quality_evidence',native)
        self.assertEqual(result['native_raw'],flow.receipt(self.root/'out/raw.json'))
        self.assertEqual(native,{k:v for k,v in result.items() if k not in {'quality_evidence','native_raw'}})
        self.assertEqual(set(result['quality_evidence']['totals_by_request']),{r['request_id'] for r in self.plan['trace']})

    def test_v5_observed_schema_and_actual_v11_source_closure(self):
        self.assertEqual(observed_validate_plan(self.plan),self.plan)
        self.assertEqual(self.plan['runner_source'],flow.receipt(flow.__file__))
        refs=self.plan['collector_sources']
        self.assertIn(flow.receipt(flow.__file__),refs)
        self.assertNotIn(flow.receipt(Path(flow.__file__).with_name('known_input_native_flow_v5.py')),refs)
        self.assertEqual(refs,[flow.receipt(p) for p in flow.collection_source_paths()])

    def test_duplicate_id_rejected_before_runtime_or_output(self):
        self.plan['trace'][1]['request_id']=self.plan['trace'][0]['request_id'];self.write_plan()
        self.assert_pre_runtime_rejection(ValueError)

    def test_missing_route_rejected_before_runtime_or_output(self):
        self.plan['candidates'][0]['routes'].pop(self.plan['trace'][0]['request_id']);self.write_plan()
        self.assert_pre_runtime_rejection(ValueError)

    def test_unsupported_route_rejected_before_runtime_or_output(self):
        text=next(r for r in self.plan['trace'] if r['task']=='text')
        self.plan['candidates'][0]['routes'][text['request_id']]='ocr#0';self.write_plan()
        self.assert_pre_runtime_rejection(ValueError)

    def test_existing_output_exclusive_before_runtime(self):
        out=self.root/'out';out.mkdir();sentinel=out/'sentinel';sentinel.write_text('untouched')
        with patch.object(flow,'DeploymentRuntime') as runtime:
            with self.assertRaises(FileExistsError):flow.execute(self.path,'shared_balanced',out)
            runtime.assert_not_called()
        self.assertEqual(list(out.iterdir()),[sentinel]);self.assertEqual(sentinel.read_text(),'untouched')

    def test_stale_context_rejected_before_runtime_or_output(self):
        (self.root/'profile/snapshot.py').write_text('# stale\n')
        self.assert_pre_runtime_rejection(ValueError)


if __name__=='__main__':unittest.main()
