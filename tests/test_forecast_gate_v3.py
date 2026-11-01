import copy
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from gsdeploy.forecasting.prediction_gate import freeze_gate_protocol
from tests.test_ordered_kernel_profiles_v2 import document
from tests.test_launch_supply_residual_v2 import residual
from tools.predictor_validation_20261027.composite_execution_v2 import SCHEMA as COMPOSITE_SCHEMA
from tools.predictor_validation_20261027.forecast_gate_v3 import (
    BUNDLE_SCHEMA,execute_bundle,evidence_closure,run_and_seal)
from tools.predictor_validation_20261027.forecast_gate_v2 import INPUT_SCHEMA
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.historical_evidence_v4 import SCHEMA as HISTORY_SCHEMA,resolve_closure


def fixtures(root):
    def save(name,value):
        p=root/name;p.write_text(json.dumps(value));return receipt(p)
    ordered=document();fit=residual(ordered)
    for p in ordered['profiles']:
        p['task']='text'
        for s in p['segments']:s['duration_ms']=dict(profiled=s['duration_ms']['mean'])
    for r in fit['fits']:r['task']='text'
    ordered_ref=save('ordered.json',ordered);residual_ref=save('residual.json',fit)
    lifecycle_ref=save('lifecycle.json',dict(schema='gsdeploy.handler_lifecycle_profiles.v2',execution_context_sha256='natural',
        provenance=dict(kind='cpu_fixture',split='training',sources=['fixture']),profiles=[dict(family='shared-family',task='text',
            pool_id='input',permit_scope='not_applicable',evidence=['fixture'],scenarios=dict(mean=dict(parent_receipt_ms=.5,
                worker_next_ready_ms=.1,permit_release_ms=None)))]))
    target_ref=save('target.json',{})
    composite=dict(schema=COMPOSITE_SCHEMA,components=dict(service=dict(residual=residual_ref,original_context_sha256='context'),
        lifecycle=dict(model=lifecycle_ref,original_context_sha256='natural')),target=dict(context=target_ref,execution_context_sha256='natural'),
        scenario_adapter=dict(profiled='mean'))
    composite['composite_execution_sha256']=identity(composite)
    composite_ref=save('composite.json',composite)
    quality_ref=save('quality.json',{'shared-family':{'text':.1}})
    index=save('training-index.json',dict(schema='gsdeploy.independent_prediction_gate_training_index.v2',split='training',
        records=[dict(run_id='training',request_id='train',pool_id='training-fixture-only',task='text')]))
    defaults=dict(aliases={a:dict(family='shared-family',supported_tasks=['text']) for a in ('a','b')},
        continuation_routing_rps=dict(text=dict(a=1.,b=1.)),resources=None,
        latency_reference_ms=dict(text=1000),memory_budget_mib=49140,energy_reference_watts=450,memory_safe_factor=1.,
        provider_kind='launch_supply_composite_v2',execution_contract='strict_lifecycle_v2')
    bundle=dict(schema=BUNDLE_SCHEMA,kind='cpu_fixture',execution_context_sha256='natural',forecast_defaults=defaults,
        artifacts=dict(ordered=ordered_ref,residual=residual_ref,composite=composite_ref,lifecycle=lifecycle_ref,quality=quality_ref),
        weights={k:1/6 for k in 'QTLPME'},training_indices=[index],startup_amortization_seconds=600.,
        kernel_duration_mode='cpu_fixture',hypothesis_options=dict(allow_independent_host_supply_hypothesis=True,
                                                                  device_progress_hypothesis='equal_context_time_slice'))
    public=dict(phase_start_ns=0,now_ns=0,deadline_ns=1000000000,drain_deadline_ns=2000000000,
        queues={a:[dict(request_id=a,alias=a,task='text',pool_id='input',arrival_ns=0)] for a in ('a','b')},
        future_workload=dict(rates=dict(text=2.),input_pools=dict(text=['input']),process='declared_schedule',declared_arrivals=[]))
    return bundle,public,save,index


class ForecastGateV3Tests(unittest.TestCase):
    def test_real_constructor_simulator_and_scorer_path_fixture(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,_,_=fixtures(Path(temp))
            result=execute_bundle(bundle,public,allow_cpu_fixture=True)
            self.assertEqual(result['forecast']['finishes']['a']['handler_end_ns'],210000000)
            self.assertEqual(result['forecast']['finishes']['a']['finish_ns'],210500000)
            self.assertEqual(result['metrics']['terms']['Q'],.1)
            self.assertIsNone(result['metrics']['terms']['M']);self.assertIsNone(result['metrics']['terms']['E'])
            self.assertIsNone(result['metrics']['full_U'])
            self.assertNotEqual(result['target_execution_context_sha256'],result['model_composite_execution_sha256'])

    def test_fixture_and_prediction_timing_injection_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,_,_=fixtures(Path(temp))
            with self.assertRaises(ValueError):execute_bundle(bundle,public)
            bundle['forecast_defaults']['completed']=[]
            with self.assertRaises(ValueError):execute_bundle(bundle,public,allow_cpu_fixture=True)

    def test_dag_kind_executes_parallel_nodes_and_same_scorer(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,save,_=fixtures(Path(temp))
            bundle['forecast_defaults']['provider_kind']='cuda_dag_v2'
            bundle['dag_hypothesis_options']=dict(allow_context_time_slice_hypothesis=True)
            profiles=[dict(alias=a,task='text',pool_id='input',nodes=[dict(id='x',kind='gpu',duration_ns=10000000),
                dict(id='y',kind='gpu',duration_ns=20000000)],edges=[],unresolved=[],cyclic=False) for a in ('a','b')]
            bundle['artifacts']['dag']=save('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',
                training_only=True,execution_context_sha256='context',profiles=profiles))
            result=execute_bundle(bundle,public,allow_cpu_fixture=True)
            self.assertEqual(result['forecast']['finishes']['a']['handler_end_ns'],40000000)
            self.assertEqual(result['forecast']['max_simultaneous_nodes_per_owner']['a'],2)
            self.assertEqual(result['metrics']['terms']['Q'],.1)

    def test_trained_dag_never_reuses_profiled_release_or_handler_anchor(self):
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,save,_=fixtures(Path(temp))
            bundle.update(kind='trained_hypothesis',kernel_duration_mode='trained_kernel_baseline_v3',baseline_intakes=[])
            bundle['artifacts']['baseline_fit']=save('baseline.json',{})
            bundle['artifacts']['quality_report']=save('quality-report.json',dict(status='FITTED',
                quality_loss=bundle['artifacts']['quality'],execution_context_sha256='natural'))
            bundle['forecast_defaults']['provider_kind']='cuda_dag_v2'
            bundle['dag_hypothesis_options']=dict(allow_context_time_slice_hypothesis=True)
            profiles=[dict(alias=a,task='text',pool_id='input',nodes=[dict(id='x',kind='gpu',duration_ns=999999,
                release_ns=123456789,kernel={})],edges=[],unresolved=[],cyclic=False) for a in ('a','b')]
            bundle['artifacts']['dag']=save('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',
                training_only=True,execution_context_sha256='context',profiles=profiles))
            with patch('tools.predictor_validation_20261027.forecast_gate_v3.verify_target_runtime'), \
                 patch('tools.predictor_validation_20261027.forecast_gate_v3.LifecycleProvider',
                       return_value=SimpleNamespace(execution_context_sha256='natural')), \
                 patch('tools.predictor_validation_20261027.kernel_baseline_query_v3.ExactSourceBaselineResolver',
                       return_value=lambda node:dict(known=True,duration_ns=10)):
                with self.assertRaisesRegex(UnknownEvidence,'TRAINED_DAG_HOST_RELEASE_MODEL_REQUIRED'):
                    execute_bundle(bundle,public,allow_trained_hypothesis=True)

    def test_full_public_cohort_sealed_after_actual_forecast(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle,public,save,index=fixtures(root)
            model=save('bundle.json',bundle);public_ref=save('public.json',dict(schema=INPUT_SCHEMA,forecast_inputs=public))
            source=Path(__file__).resolve()
            closure=evidence_closure([model['path']]);paths=list({r['path'] for r in closure}|{public_ref['path'],str(source)})
            protocol=freeze_gate_protocol(model_paths=dict(model=model['path']),source_paths=paths,training_data_paths=[index['path']],
                candidates=[dict(id=c,model_id='model',execution_context_sha256='natural') for c in ('baseline','candidate')],
                baseline_candidate_id='baseline',validation_domain=[dict(run_id='validation',requests=[
                    dict(request_id=a,pool_id='input',task='text') for a in ('a','b')])],
                thresholds=dict(queue_depth_mae_max=1),created_unix_ns=1,weights=bundle['weights'],queue_sample_offsets_ns=[0,50000000])
            protocol_ref=save('protocol.json',protocol)
            jobs=[dict(run_id='validation',candidate_id=c,public_inputs_path=public_ref['path']) for c in ('baseline','candidate')]
            with patch('tools.predictor_validation_20261027.forecast_gate_v3.prediction_source_paths',return_value=[source]):
                result=run_and_seal(protocol_ref['path'],jobs,root/'output',allow_cpu_fixture=True)
            sealed=json.loads(Path(result['predictions']['path']).read_text())
            self.assertEqual(len(sealed['cells']),2)
            for cell in sealed['cells']:
                self.assertEqual(len(cell['requests']),2);self.assertIsNone(cell['terms']['E'])
                self.assertEqual(cell['terms']['Q'],.1)

    def test_v4_host_supply_executes_trained_shape_and_original_six_term_scorer(self):
        from tests.test_dag_host_release_v4 import document as host_document,profile
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,save,_=fixtures(Path(temp))
            public['queues']={'a':public['queues']['a']};p=profile()
            p.update(task='text',family='shared-family')
            bundle['forecast_defaults']['provider_kind']='cuda_dag_host_supply_v4'
            dag=save('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',training_only=True,
                execution_context_sha256='context',profiles=[p]))
            host=host_document(p);host.update(light_execution_context_sha256='context',dag=dag)
            bundle['artifacts'].update(dag=dag,dag_host_release=save('host.json',host))
            bundle.update(host_supply_scenario='front_loaded_supply',
                allow_host_supply_identification_hypothesis=True,allow_host_supply_pooling_hypothesis=True)
            result=execute_bundle(bundle,public,allow_cpu_fixture=True)
            self.assertEqual(result['forecast']['finishes']['a']['handler_end_ns'],107)
            self.assertEqual(result['metrics']['terms']['Q'],.1)
            self.assertEqual(result['host_supply_scenario'],'front_loaded_supply')
            self.assertFalse(result['measured_target_fallback_used'])
            bundle['allow_host_supply_identification_hypothesis']=False
            with self.assertRaisesRegex(ValueError,'opt-ins'):execute_bundle(bundle,public,allow_cpu_fixture=True)

    def test_optional_resource_provider_reaches_event_axis_and_scorer(self):
        from tests.test_dag_host_release_v4 import document as host_document,profile
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,save,_=fixtures(Path(temp))
            public['queues']={'a':public['queues']['a']};p=profile();p.update(task='text',family='shared-family')
            bundle['forecast_defaults']['provider_kind']='cuda_dag_host_supply_v4'
            dag=save('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',training_only=True,
                execution_context_sha256='context',profiles=[p]))
            host=host_document(p);host.update(light_execution_context_sha256='context',dag=dag)
            bundle['artifacts'].update(dag=dag,dag_host_release=save('host.json',host),natural_resources=save('resources.json',{}))
            bundle.update(host_supply_scenario='front_loaded_supply',allow_host_supply_identification_hypothesis=True,
                allow_host_supply_pooling_hypothesis=True,allow_conditional_resource_hypothesis=True,
                startup_prediction=dict(peak_memory_mib=300,basis='frozen_training_prediction',evidence=['fixture']))
            resource=SimpleNamespace(execution_context_sha256='natural',
                power=lambda *a:dict(known=True,watts=100),memory=lambda *a:dict(known=True,peak_memory_mib=200))
            with patch('tools.predictor_validation_20261027.natural_resources_v5.NaturalResourceProvider',return_value=resource):
                result=execute_bundle(bundle,public,allow_cpu_fixture=True)
            self.assertEqual(result['forecast']['board_energy_joules'],100)
            self.assertEqual(result['forecast']['peak_memory_mib'],200)
            self.assertAlmostEqual(result['metrics']['terms']['M'],300/49140)
            self.assertAlmostEqual(result['metrics']['terms']['E'],100/450)

    def test_explicit_numeric_hypothesis_preserves_strict_rejection_and_no_target_fallback(self):
        from tests.test_dag_host_release_v4 import document as host_document,profile
        with tempfile.TemporaryDirectory() as temp:
            bundle,public,save,_=fixtures(Path(temp));public['queues']={'a':public['queues']['a']}
            p=profile();p.update(task='text',family='shared-family',source_sqlite={})
            p['nodes'].append(dict(id='copy',kind='copy',duration_ns=5,issue_start_ns=104,issue_end_ns=105,issue_thread=1))
            p['edges'].append(dict(before='copy',after='end'))
            for node in p['nodes']:
                if node['kind']=='gpu':node['kernel']={}
            bundle['forecast_defaults']['provider_kind']='cuda_dag_host_supply_v4'
            dag=save('dag.json',dict(schema='gsdeploy.request_cuda_dag_profiles.v2',training_only=True,
                execution_context_sha256='context',profiles=[p]))
            host=host_document(p);host.update(light_execution_context_sha256='context',dag=dag)
            bundle['artifacts'].update(dag=dag,dag_host_release=save('host.json',host),semantic_fit=save('semantic.json',{}),
                copy_memset_fit=save('copy.json',{}))
            bundle.update(host_supply_scenario='front_loaded_supply',allow_host_supply_identification_hypothesis=True,
                allow_host_supply_pooling_hypothesis=True,kernel_duration_mode='numeric_semantic_hypothesis_v5',
                allow_numeric_semantic_hypothesis=True,copy_memset_duration_mode='numeric_copy_memset_hypothesis_v5',
                allow_copy_memset_numeric_hypothesis=True,dag_hypothesis_options=dict(allow_independent_copy_hypothesis=True))
            prediction=dict(known=False,strict_known=False,numeric_hypothesis_available=True,duration_ns=30,
                hypothesis_only=True,deployment_qualified=False,strict_failure_reasons=['original_25_percent_residual'])
            with patch('tools.predictor_validation_20261027.kernel_semantic_query_v5.ExactSourceSemanticHypothesisResolver',
                       return_value=lambda kernel:prediction), \
                 patch('tools.predictor_validation_20261027.copy_memset_baseline_v5.ExactSourceCopyMemsetHypothesisResolver',
                       return_value=lambda node,source:dict(prediction,duration_ns=50)):
                result=execute_bundle(bundle,public,allow_cpu_fixture=True)
                self.assertEqual(result['forecast']['finishes']['a']['handler_end_ns'],137)
                self.assertFalse(prediction['known']);self.assertFalse(prediction['strict_known'])
                self.assertEqual(result['baseline_coverage']['numeric_hypothesis_nodes'],2)
                self.assertEqual(result['baseline_coverage']['strict_qualified_nodes'],0)
                self.assertEqual(result['baseline_coverage']['strict_failure_reason_counts']['original_25_percent_residual'],2)
                self.assertEqual(result['copy_memset_coverage']['numeric_hypothesis_nodes'],1)
                self.assertEqual(result['copy_memset_coverage']['strict_qualified_nodes'],0)
                self.assertFalse(result['physical_accuracy_validated'])
                prediction['numeric_hypothesis_available']=False
                with self.assertRaisesRegex(UnknownEvidence,'HYPOTHESIS_UNAVAILABLE'):
                    execute_bundle(bundle,public,allow_cpu_fixture=True)
            bundle['allow_numeric_semantic_hypothesis']=False
            with self.assertRaisesRegex(ValueError,'evaluation opt-in'):execute_bundle(bundle,public,allow_cpu_fixture=True)

    def test_explicit_snapshot_resolution_preserves_old_bytes_and_current_source_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);bundle,public,save,index=fixtures(root)
            legacy=root/'legacy.py';legacy.write_text('original source\n');old=receipt(legacy)
            snapshot=root/'snapshot.py';snapshot.write_bytes(legacy.read_bytes())
            legacy.write_text('changed current source\n')
            history=save('history.json',dict(schema=HISTORY_SCHEMA,historical_artifacts_rewritten=False,
                mappings=[dict(historical=old,snapshot=receipt(snapshot))]))
            bundle['artifacts']['historical_resolution']=history;bundle['historical_training_source']=old
            model=save('bundle.json',bundle);public_ref=save('public.json',dict(schema=INPUT_SCHEMA,forecast_inputs=public))
            source=Path(__file__).resolve();resolved=resolve_closure([model['path']],history['path'])
            paths=list({r['path'] for r in resolved['materialized_sources']}|{public_ref['path'],str(source)})
            protocol=freeze_gate_protocol(model_paths=dict(model=model['path']),source_paths=paths,
                training_data_paths=[index['path']],candidates=[dict(id=c,model_id='model',execution_context_sha256='natural')
                    for c in ('baseline','candidate')],baseline_candidate_id='baseline',
                validation_domain=[dict(run_id='validation',requests=[dict(request_id=a,pool_id='input',task='text') for a in ('a','b')])],
                thresholds={},created_unix_ns=1,weights=bundle['weights'])
            protocol_ref=save('protocol.json',protocol)
            jobs=[dict(run_id='validation',candidate_id=c,public_inputs_path=public_ref['path']) for c in ('baseline','candidate')]
            with patch('tools.predictor_validation_20261027.forecast_gate_v3.prediction_source_paths',return_value=[source]):
                run_and_seal(protocol_ref['path'],jobs,root/'output',allow_cpu_fixture=True)
            report=json.loads((root/'output/production-receipt.json').read_text())
            for job in report['jobs']:
                resolved=job['historical_evidence_resolution']
                self.assertEqual(resolved['historical_substitutions'],[dict(historical=old,snapshot=receipt(snapshot))])
                self.assertFalse(resolved['historical_direct_path_closure_pass_claimed'])
            self.assertEqual(snapshot.read_text(),'original source\n');self.assertEqual(legacy.read_text(),'changed current source\n')
            # Snapshots cannot substitute for the code actually executing now.
            with patch('tools.predictor_validation_20261027.forecast_gate_v3.prediction_source_paths',return_value=[legacy]):
                with self.assertRaisesRegex(ValueError,'complete v3 package/tools'):
                    run_and_seal(protocol_ref['path'],jobs,root/'other-output',allow_cpu_fixture=True)


if __name__=='__main__':unittest.main()
