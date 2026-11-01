"""CPU execution-path checks for the flow collector; never GPU evidence."""
import copy
import json
import os
from pathlib import Path
import tempfile
import time
import types
import unittest
from unittest.mock import patch

from tools.predictor_validation_20261027 import native_flow as flow
from gsdeploy.forecasting.prediction_gate import DATA_SCHEMA, freeze_gate_protocol, seal_predictions


class NativeFlowTests(unittest.TestCase):
    def profile(self,root):
        root.mkdir()
        pool=[json.loads(l) for l in (flow.BASE/'runtime-inputs.jsonl').read_text().splitlines()]
        inputs=[]
        for task in ('text','image','asr','ocr'):
            inputs.extend(dict(task=task,pool_id=r['pool_id'],input_size=i+1)
                          for i,r in enumerate([r for r in pool if r['task_type']==task][:4]))
        frozen=root/'snapshot.py';frozen.write_text('# execution fixture\n')
        ctx=dict(residents=flow.RESIDENTS,sources=[flow.receipt(frozen)],
            execution_sources=[dict(original_path=str(frozen),frozen=flow.receipt(frozen))],
            worker_bindings={f:dict(command=['python','worker.py'],env={}) for f in set(flow.RESIDENTS.values())},
            threads={},worker_variant='semantic_worker.py')
        flow.save(root/'context.json',ctx)
        flow.save(root/'plan.json',dict(execution_context_sha256=flow.context_identity(ctx),inputs=inputs,
            input_assignments=[dict(task=r['task'],pool_id=r['pool_id'],view='known_input' if i%4<3 else 'unseen_input')
                               for i,r in enumerate(inputs)]))

    def test_prepare_same_cohort_routes_and_resource_binding(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.profile(root/'profile')
            plan=flow.prepare(root/'profile',root/'pilot',duration=1.,text_rate=20.,background_rate=5.,include_admission=True)
            self.assertEqual(len(plan['candidates']),12)
            ids={r['request_id'] for r in plan['trace']}
            self.assertEqual({r['task'] for r in plan['trace']},{'text','image','asr','ocr'})
            for c in plan['candidates']:
                self.assertEqual(set(c['routes']),ids)
                if not c['allow_spill']:
                    self.assertTrue(all(c['routes'][r['request_id']].startswith('text#') for r in plan['trace'] if r['task']=='text'))
                if c['resource_configuration']['mode']=='request_admission':
                    self.assertTrue(all(v['max_inflight']==1 for v in c['resource_configuration']['allocations'].values()))

    def test_training_supplement_complete_domain_without_holdout_inputs(self):
        from tools.predictor_validation_20261027.domain_collect import prepare
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.profile(root/'profile')
            plan=prepare(root/'profile',root/'supplement')
            self.assertEqual(len(plan['jobs']),138)
            self.assertEqual(len({j['context'] for j in plan['jobs']}),23)
            self.assertTrue(all(j['split']=='training' for j in plan['jobs']))
            known={(a['task'],a['pool_id']) for a in plan['input_assignments']}
            self.assertEqual(len(known),12)
            for job in plan['jobs']:
                self.assertTrue(all((job['tasks'][a],p) in known for a,p in job['inputs'].items()))

    def test_continuous_training_is_predeclared_and_disjoint(self):
        from tools.predictor_validation_20261027.flow_training import prepare_suite
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.profile(root/'profile')
            suite=prepare_suite(root/'profile',root/'training')
            self.assertEqual(len(suite['jobs']),8)
            self.assertEqual(suite['seeds'],[627401,627402])
            for job in suite['jobs']:
                plan=json.loads(Path(job['plan']['path']).read_text())
                self.assertEqual(plan['phase'],'predictor_training')
                self.assertEqual(plan['split'],'training')
                self.assertGreater(plan['created_ns'],0)
                self.assertEqual(job['plan'],flow.receipt(job['plan']['path']))

    def test_dispatches_full_trace_and_scores_after_native_raw_saved(self):
        """Mock hardware only; run real plan/source/dispatch/scoring control path."""
        class Runtime:
            def __init__(self,config,registry):
                self.ready={a:dict(pid=os.getpid()) for a in flow.RESIDENTS}
                self.workers={a:types.SimpleNamespace(process=types.SimpleNamespace(pid=os.getpid())) for a in flow.RESIDENTS}
                self.errors=[];self.shutdown_report={};self.admission=None
                self.measurement_enqueues=0;self.measurement_drains=[]
            def start(self):pass
            def close(self):pass
            def enqueue(self,row,sample):
                now=time.perf_counter_ns()
                if row['phase']=='measurement':
                    self.measurement_enqueues+=1
                    assert now>=row['scheduled_time_ns']
                row.update(status='completed',completion_time_ns=now,
                    observed_worker_start_time_ns=now,
                    worker_response=dict(task=row['task'],prediction='',success=True,start_perf_ns=now,end_perf_ns=now))
            def drain(self,rows,timeout):
                if rows and rows[0]['phase']=='measurement':
                    self.measurement_drains.append((len(rows),self.measurement_enqueues))
        class Monitor:
            def __init__(self,runtime):self.samples=[];self.errors=[]
            def sample(self):
                self.samples.append(dict(time_ns=time.perf_counter_ns(),watts=100.,owned_used_mib=1000.,
                    owned_aliases=list(flow.RESIDENTS),foreign_pids=[]))
            def start(self):pass
            def check(self):pass
            def close(self):self.sample()
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.profile(root/'profile')
            plan=flow.prepare(root/'profile',root/'pilot',duration=.03,text_rate=100.,background_rate=100.)
            runtime=Runtime(None,None)
            with patch.object(flow,'DeploymentRuntime',return_value=runtime),patch.object(flow,'BoardMonitor',Monitor),patch.object(flow.fcntl,'flock'):
                raw=flow.execute(root/'pilot/plan.json','shared_balanced',root/'run')
            self.assertEqual(raw['status'],'completed')
            self.assertEqual(runtime.measurement_drains,[(len(plan['trace']),len(plan['trace']))])
            self.assertEqual(len(raw['quality_evidence']['totals_by_request']),len(plan['trace']))
            native=json.loads((root/'run/raw.json').read_text())
            self.assertNotIn('quality_evidence',native)
            self.assertEqual(raw['native_raw'],flow.receipt(root/'run/raw.json'))
            self.assertLess(raw['startup_begin_ns'],raw['phase_start_ns'])
            self.assertGreaterEqual(raw['observation_end_ns'],raw['deadline_ns'])
            # Exercise the real prospective checker before the same native
            # dispatch path. All hardware remains mocked and is no accuracy evidence.
            plan['queue_depth_boundary']='waiting_until_handler_start'
            plan['plan_sha256']=flow.digest({k:v for k,v in plan.items() if k!='plan_sha256'})
            plan_path=root/'validation-plan.json';flow.save(plan_path,plan)
            model=root/'fixture-model.json';flow.save(model,{'kind':'cpu_fixture'})
            training=root/'training-index.json'
            flow.save(training,dict(schema=DATA_SCHEMA,split='training',records=[
                dict(run_id='train',request_id='t',pool_id='disjoint-training-input',task='text')]))
            cohort=[{k:r[k] for k in ('request_id','pool_id','task')} for r in plan['trace']]
            protocol=freeze_gate_protocol(model_paths={'fixture':model},source_paths=[plan_path,*flow.collection_source_paths()],
                training_data_paths=[training],candidates=[dict(id=c['id'],model_id='fixture',
                    execution_context_sha256=c['execution_context_sha256']) for c in plan['candidates']],
                baseline_candidate_id=plan['baseline_candidate_id'],validation_domain=[dict(run_id='v',requests=cohort)],
                thresholds={},created_unix_ns=100,weights=plan['weights'],queue_sample_offsets_ns=plan['queue_sample_offsets_ns'])
            protocol_path=root/'protocol.json';flow.save(protocol_path,protocol)
            cells=[dict(run_id='v',candidate_id=c['id'],execution_context_sha256=c['execution_context_sha256'],
                requests=[dict(r,service_ms=None,queue_ms=None,e2e_ms=None) for r in cohort],
                terms=dict.fromkeys('QTLPME')) for c in plan['candidates']]
            predictions_path=root/'predictions.json'
            flow.save(predictions_path,seal_predictions(protocol,cells,created_unix_ns=200))
            pre_path=root/'validation.precollection.json'
            class CheckedMonitor(Monitor):
                def __init__(self,runtime):
                    assert pre_path.exists(), 'Collection telemetry started before prospective check'
                    super().__init__(runtime)
            with patch.object(flow,'DeploymentRuntime',return_value=Runtime(None,None)),patch.object(flow,'BoardMonitor',CheckedMonitor),patch.object(flow.fcntl,'flock'):
                gated=flow.execute(plan_path,'shared_balanced',root/'validation',protocol_path=protocol_path,
                    predictions_path=predictions_path,run_id='v')
            pre=json.loads(pre_path.read_text())
            self.assertEqual(gated['status'],'completed')
            self.assertEqual(gated['split'],'validation')
            self.assertGreater(gated['collection_started_unix_ns'],pre['checked_unix_ns'])
            self.assertEqual(gated['precollection'],flow.receipt(pre_path))
            self.assertEqual(json.loads((root/'validation/raw.json').read_text())['precollection'],flow.receipt(pre_path))
            # Changes to a frozen model fail before any runtime construction.
            frozen_model=model.read_bytes()
            model.write_text('{}')
            with patch.object(flow,'DeploymentRuntime') as forbidden:
                with self.assertRaisesRegex(ValueError,'identity'):
                    flow.execute(plan_path,'shared_balanced',root/'tampered',protocol_path=protocol_path,
                        predictions_path=predictions_path,run_id='v')
                forbidden.assert_not_called()
            self.assertFalse((root/'tampered').exists())
            # A source can drift while another experiment owns the GPU lock.
            # The post-lock check must prevent telemetry and worker startup too.
            model.write_bytes(frozen_model)
            def lock_then_drift(handle,mode):
                if mode==flow.fcntl.LOCK_EX:model.write_text('{}')
            runtime=Runtime(None,None)
            with patch.object(flow,'DeploymentRuntime',return_value=runtime),patch.object(flow,'BoardMonitor') as forbidden_monitor,patch.object(flow.fcntl,'flock',side_effect=lock_then_drift),patch.object(runtime,'start') as forbidden_start:
                failed=flow.execute(plan_path,'shared_balanced',root/'changed-during-lock',protocol_path=protocol_path,
                    predictions_path=predictions_path,run_id='v')
                forbidden_monitor.assert_not_called();forbidden_start.assert_not_called()
            self.assertEqual(failed['status'],'failed')
            self.assertNotIn('collection_started_unix_ns',failed)

    def test_partial_validation_options_fail_before_runtime(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);self.profile(root/'profile')
            flow.prepare(root/'profile',root/'pilot',duration=.03,text_rate=100.,background_rate=100.)
            with patch.object(flow,'DeploymentRuntime') as forbidden:
                with self.assertRaisesRegex(ValueError,'together'):
                    flow.execute(root/'pilot/plan.json','shared_balanced',root/'run',run_id='v')
                forbidden.assert_not_called()


if __name__=='__main__':unittest.main()
