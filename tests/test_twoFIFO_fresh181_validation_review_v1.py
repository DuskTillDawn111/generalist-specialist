"""CPU validation role, native quality, unchanged six-term scoring and census."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_review_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_evaluation_v1 as evaluation
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_metrics_v1 as metrics
from tools.predictor_validation_20261027.flow_validation import _score
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode
from tests import test_twoFIFO_native_quality_labels_v1 as quality_fixture
from tests import test_twoFIFO_fresh181_validation_metrics_v1 as metric_fixture


class IndependentValidationTests(unittest.TestCase):
    def test_full_review_wiring_and_outer_report_claim_change_restore_watch(self):
        # Explicit native-shaped CPU interface fixture: semantic provenance
        # parsers are injected, and this grants no native qualification. The
        # outer watch, canonical paths, durable save and orchestration are real.
        for tamper in (None,'report','claim'):
            with self.subTest(tamper=tamper),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);collection=root/'attempt'/'collection';collection.mkdir(parents=True)
                def save(path,value):return m.executor.save(path,value)
                context=save(root/'context.json',dict(worker_bindings={}))
                pool=save(root/'pool.json',dict(CPU_ONLY=True));catalog=save(root/'catalog.json',dict(CPU_ONLY=True))
                original=save(root/'original.json',dict(catalog=catalog));public=save(root/'public.json',dict(inputs=dict(public_plan=original)))
                contract=save(root/'contract.json',dict(queue_sample_offsets_ns=[0],weights={k:1/6 for k in 'QTLPME'}))
                report=save(collection/'expected.json',dict(CPU_ONLY=True));prefix=save(collection/'prefix.json',dict(CPU_ONLY=True))
                gate=save(collection/'gate.json',dict(CPU_ONLY=True));seal=save(collection/'seal.json',dict(CPU_ONLY=True))
                plan=dict(schema=m.PLAN_SCHEMA,split='validation',context=context,prepared_pool=pool,execution_id='e',attempt_id='a',
                    candidate_id='c0',execution_context_sha256='ctx',cases=[dict(case_id='original181')],sources=[],source_freeze=[],
                    public_preparation=public,public_contract=contract,saved_matching_fit=report,priming_semantics={'sha256':'priming'},
                    forecast_barrier=dict(prefix_path=prefix['path'],gate_request_path=gate['path'],expected_report_path=report['path'],seal_path=seal['path']))
                plan_ref=save(root/'attempt'/'plan.json',plan);claim=save(root/'claim.json',dict(CPU_ONLY=True))
                manifest=save(root/'manifest.json',dict(attempts=[dict(candidate_id='c0',plan=plan_ref,collection=str(collection),claim=claim['path'])]))
                execute=save(collection/'execution.json',dict(process_group=123))
                executed=save(collection/'executed.json',dict(attempt_id='a',execution_id='e',cases=plan['cases']))
                save(collection/m.ACTIVITY_FILE,dict(CPU_ONLY=True))
                raw=dict(schema=m.RAW_SCHEMA,collector_schema=m.RAW_SCHEMA,split='validation',backend='native',kind='measured',status='completed',
                    plan=plan_ref,context=context,execution_context_sha256='ctx',prepared_pool=pool,source_bindings=[],source_freeze=[],
                    fresh181_executor_sources=m.implementation_sources(m.executor.__file__),executed_cases=executed,
                    shutdown={o:dict(exit_code=0,forced=False,bye_received=True) for o in m.OWNERS})
                raw_ref=save(collection/'raw.json',raw)
                prediction=dict(report=dict(source_queue_forecast=dict(CPU_ONLY=True),expected_score=dict(expected_U_under_supplied_marginals=-1.)),
                    gate=dict(public_requests=[],forecast_origin_ns=0,fixed_drain_deadline_ns=210))
                def inspected(*args):
                    if tamper:
                        path=Path((report if tamper=='report' else claim)['path']);before=path.read_bytes();path.write_bytes(before+b' ');path.write_bytes(before)
                    return prediction
                with ExitStack() as stack:
                    replacements=[(m.executor,'audit',m.checked(manifest)),(m.executor,'audit_plan',plan),
                        (m,'inspect_cohort',dict(issues=[])),(m,'inspect_histories',{}),(m,'inspect_ledger',{}),
                        (m,'inspect_protocol',dict(issues=[],all_offered_and_priming_prefixes=199,requests=[{}]*199,sources=[])),
                        (m,'inspect_samples',{}),(m.metrics,'observed_labels',{}),(m.solo,'build',dict(sources=[])),
                        (m.solo,'slowdown_labels',{}),(m.metrics,'actual_timeline',[]),(m.metrics,'prediction_errors',{}),
                        (m,'actual_quality',{}),(m,'actual_resources',dict(joint_tuple={})),
                        (m,'actual_score',dict(score=dict(terms={},full_U=-1.))), (m.expected,'evaluate_validation',{})]
                    for obj,name,value in replacements:stack.enter_context(patch.object(obj,name,return_value=value))
                    stack.enter_context(patch.object(m.executor,'completed_execution'))
                    stack.enter_context(patch.object(m,'attempt_events'))
                    stack.enter_context(patch.object(m,'inspect_prediction',side_effect=inspected))
                    stack.enter_context(patch('tools.predictor_validation_20261027.natural_macro_resources_candidate_v1.priming_identity',return_value='priming'))
                    if tamper:
                        with self.assertRaises(ValueError):m.review(raw_ref['path'],root/'out',manifest_path=manifest['path'],execution_path=execute['path'])
                        self.assertFalse((root/'out'/'report.json').exists())
                    else:
                        result=m.checked(m.review(raw_ref['path'],root/'out',manifest_path=manifest['path'],execution_path=execute['path']))
                        self.assertTrue(result['validation_labels_read']);self.assertFalse(result['validation_labels_used_for_fit'])
                        self.assertEqual(result['predefined_forecast_worlds'],12);self.assertFalse(result['final_gain_eligible'])

    def quality_fixture(self,root):
        raw,plan,case,contract,catalog=quality_fixture.NativeQualityTests().fixture(root)
        originals=raw['request_records'];raw['request_records']=[];requests=[]
        for i in range(181):
            row=copy.deepcopy(originals[0 if i==0 else 1]);row['request_id']='r'+str(i);row['worker_response']['request_id']=row['request_id']
            raw['request_records'].append(row);requests.append(dict(request_id=row['request_id'],owner=row['model'],task=row['task'],pool_id=row['pool_id'],arrival_ns=100+i))
            row['scheduled_time_ns']=100+i;row['completion_time_ns']=300+i
        contract_file=root/'contract.json';contract_file.write_text(json.dumps(contract));plan['public_contract']=m.receipt(contract_file)
        return raw,plan,requests,catalog

    def test_actual_quality_uses_exact_native_outputs_summed_units_no_training_fallback(self):
        with tempfile.TemporaryDirectory() as temp:
            raw,plan,requests,catalog=self.quality_fixture(Path(temp))
            quality=m.actual_quality(raw,plan,catalog,requests,1000)
            self.assertEqual(quality['offered_requests'],181);self.assertEqual(quality['measured_Q'],1/(1+180*9))
            self.assertEqual(quality['per_task']['ocr']['scored'],181);self.assertFalse(quality['training_expectation_fallback_used'])
            raw['request_records'][1].update(status='incomplete',worker_response=None,completion_time_ns=None)
            quality=m.actual_quality(raw,plan,catalog,requests,1000)
            self.assertIsNone(quality['measured_Q']);self.assertEqual(quality['per_task']['ocr']['offered'],181)
            with self.assertRaises(ValueError):m.actual_quality(raw,plan,catalog,requests[:-1],1000)

    def test_validation_role_rejects_training_cpu_or_oldschema_before_canonical_access(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for i,change in enumerate([dict(schema='old'),dict(split='training'),dict(backend='CPU_TEST_ONLY'),dict(kind='CPU_TEST_ONLY')]):
                raw=dict(schema=m.RAW_SCHEMA,collector_schema=m.RAW_SCHEMA,split='validation',backend='native',kind='measured');raw.update(change)
                file=root/str(i);file.write_text(json.dumps(raw))
                with self.assertRaisesRegex(ValueError,'before evaluation labels'):
                    m.review(file,root/('out'+str(i)),manifest_path='not-accessed',execution_path='not-accessed')

    def score_fixture(self,root):
        a=1000;b=a+90_000_000_000;d=b+120_000_000_000
        contract=dict(window_seconds=90.,startup_amortization_seconds=3600.,weights={k:1/6 for k in 'QTLPME'},
            aliases={o:dict(supported_tasks=[t],family=o) for o,t in [('text#0','text'),('text#1','image'),('generalist_2#0','asr'),('ocr#0','ocr')]},
            workload_rps=dict(text=1.85,image=.05,asr=.05,ocr=.05),latency_reference_ms=dict(text=100.,image=100.,asr=100.,ocr=100.),
            memory_budget_mib=1000.,energy_reference_watts=100.,memory_safe_factor=1.1)
        path=root/'contract.json';path.write_text(json.dumps(contract));plan=dict(public_contract=m.receipt(path))
        requests=[dict(request_id=str(i),owner=['text#0','text#1','generalist_2#0','ocr#0'][i%4],task=['text','image','asr','ocr'][i%4],pool_id='p',arrival_ns=a+i) for i in range(181)]
        labels=dict(requests=[dict(request_id=r['request_id'],arrival_ns=r['arrival_ns'],reader_E2E_ns=20_000_000,completed_within_fixed_drain=True) for r in requests])
        raw=dict(measurement_windows=[dict(start_ns=a,end_ns=b,fixed_drain_deadline_ns=d)])
        resources=dict(joint_tuple=dict(startup_joules=3600.,premeasurement_owned_sampled_peak_mib=100.,measurement_through_drain_owned_sampled_peak_mib=200.,measurement_board_joules=9000.))
        quality=dict(measured_Q=.3,per_task={t:dict(loss=.3,numerator=3.,denominator=10.,offered=sum(r['task']==t for r in requests),missing=[]) for t in contract['workload_rps']})
        return raw,plan,requests,labels,quality,resources,contract

    def test_original_native_scorer_terms_memory_max_startup_once_and_censoring(self):
        with tempfile.TemporaryDirectory() as temp:
            raw,plan,requests,labels,q,res,contract=self.score_fixture(Path(temp))
            measured=m.actual_score(raw,plan,requests,labels,q,res,[plan['public_contract']])['score']
            cohort={r['request_id']:dict(request_id=r['request_id'],alias=r['owner'],task=r['task'],pool_id=r['pool_id'],arrival_ns=r['arrival_ns']-1000) for r in requests}
            forecast=dict(cohort=cohort,finishes={rid:dict(row,finish_ns=row['arrival_ns']+20_000_000) for rid,row in cohort.items()},peak_memory_mib=200.,board_energy_joules=9000.)
            original=_score(contract,forecast,{o:{contract['aliases'][o]['supported_tasks'][0]:dict(basis='expected_per_request_totals',numerator=3.,denominator=10.,evidence=['CPU comparison'])} for o in contract['aliases']},startup_joules=3600.,historical_peak=100.)
            for term in 'TLPME':self.assertEqual(measured['terms'][term],original['terms'][term])
            self.assertEqual(measured['full_U'],-sum(contract['weights'][k]*v for k,v in measured['terms'].items()))
            self.assertAlmostEqual(measured['terms']['M'],.22);self.assertAlmostEqual(measured['terms']['E'],1.01)
            labels['requests'][0]['completed_within_fixed_drain']=False
            measured=m.actual_score(raw,plan,requests,labels,q,res,[plan['public_contract']])['score']
            self.assertIsNone(measured['terms']['L']);self.assertIsNone(measured['full_U']);self.assertEqual(measured['cohort_size'],181)
            self.assertIsNotNone(measured['terms']['M']);self.assertEqual(measured['terms']['Q'],.3)

    def test_original_queue_grid_threshold_requires_full_world_and_request_coverage(self):
        requests,raw=metric_fixture.population();helper=metric_fixture.ValidationMetricsTests();labels=metrics.observed_labels(raw,requests,210)
        errors=metrics.prediction_errors(helper.report(requests),labels,metrics.actual_timeline(raw,requests,0,210,[]),requests,0,210,sample_times=[0,100,200])
        contract=dict(validation_thresholds=dict(e2e_mare_max=.2,service_mare_max=.2,each_task_p95_absolute_relative_error_max=.2,queue_depth_mean_absolute_error_max_requests=1.))
        result=evaluation.threshold_arithmetic(errors,contract)
        self.assertTrue(result['all_local_pilot_thresholds_pass'])
        errors=decode(errors);errors['point_reader_MARE_request_coverage']=180/181
        self.assertIsNone(evaluation.threshold_arithmetic(errors,contract)['checks']['e2e'])

if __name__=='__main__':unittest.main()
