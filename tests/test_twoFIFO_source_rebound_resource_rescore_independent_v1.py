"""Full public 150-case CPU integration; authentication/provider are mocked.

Actual compact forecast generation, replay, batch scoring, durable persistence,
measured resource labels and score-error propagation execute unchanged.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_source_rebound_phase12_adapter_independent_v1 import fixture
from tests.test_twoFIFO_fold_resource_rescore_v1 import setup_world
from tools.predictor_validation_20261027 import twoFIFO_source_rebound_resource_rescore_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_recovery_generated_v1 as recovery_generated
from tools.predictor_validation_20261027 import twoFIFO_recovery_resource_v1 as recovery_resource
from tools.predictor_validation_20261027 import twoFIFO_recovery_resource_rescore_v1 as recovery_scorer

SECOND = 10**9


def put(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc))
    return m.receipt(path)


def fixture_all(root):
    plans = copy.deepcopy(fixture()[0])
    public_design = json.loads(Path('artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json').read_text())
    score_ref, quality_ref = public_design['scoring_plan'], public_design['quality']
    score, quality = m.checked(score_ref), m.checked(quality_ref)
    plan_refs, reviews, rows = [], [], []
    blocks = sorted({p['capture_id'] for p in plans})
    for index, plan in enumerate(plans):
        plan_ref = put(root/f'plan-{index}.json', plan)
        plan_refs.append(plan_ref)
        windows, native_cases, records = [], [], []
        for number, case in enumerate(plan['cases']):
            epoch = (10 + number*100)*SECOND
            windows.append(dict(case_id=case['case_id'], start_ns=epoch,
                end_ns=epoch+10*SECOND, fixed_drain_deadline_ns=epoch+70*SECOND))
            native_cases.append(dict(case_id=case['case_id'], observation_start_ns=epoch))
            records.extend(dict(request_id=a['request_id'], status='completed',
                completion_time_ns=epoch+a['arrival_ns']+1_000_000) for a in case['arrivals'])
            donors = [b for b in blocks if b != plan['capture_id']]
            forecast = m.source.generated.forecast_case(case, epoch, None, None, score, quality,
                donor_blocks=donors, unqualified_reason='CPU_FIXTURE_COMPLETE_FIT_UNQUALIFIED')
            archive = root/'archived'/str(index)/str(number)
            archive.mkdir(parents=True)
            ref = m.save(archive/'prediction.json',
                m.source.generated.compact_prediction(forecast))
            rows.append(dict(block=plan['capture_id'], attempt_id=plan['attempt_id'],
                case_id=case['case_id'], prediction=ref))
        last = windows[-1]['fixed_drain_deadline_ns']
        raw = dict(schema=m.resource.response.executor.RAW_SCHEMA, backend='native', status='completed',
            split='training', CPU_FIXTURE_NOT_NATIVE=True,
            **{k:plan[k] for k in ('capture_id','attempt_id','execution_id')},
            lifecycle_boundaries=[dict(boundary=k, time_ns=v) for k,v in [
                ('startup_begin',8*SECOND),('priming_end',9*SECOND),
                ('monitoring_end',last+SECOND),('shutdown_begin',last+2*SECOND)]],
            shutdown_memory_sampled=False, cases=native_cases, measurement_windows=windows,
            request_records=records, board_samples=[dict(time_ns=t, watts=20,
                owned_used_mib=3000 if t==79*SECOND else 1000)
                for t in range(8*SECOND,last+SECOND+1,SECOND)])
        raw_ref = put(root/f'raw-{index}.json', raw)
        review_ref = put(root/f'review-{index}.json', {'raw':raw_ref,'CPU_FIXTURE_NOT_NATIVE':True})
        reviews.append(dict(attempt_id=plan['attempt_id'],training_review=review_ref))
    executor_ref = put(root/'executor.json', dict(attempts=[{'plan':r} for r in plan_refs]))
    fit_manifest = dict(executor_manifest=executor_ref)
    fit_manifest_ref = put(root/'fit-manifest.json',fit_manifest)
    final_ref = put(root/'continuation.json',{'CPU_FIXTURE_NOT_NATIVE':True})
    intake = dict(attempts=reviews,CPU_FIXTURE_NOT_NATIVE=True)
    design = dict(fit_manifest=fit_manifest_ref,scoring_plan=score_ref,quality=quality_ref)
    design_ref = put(root/'design.json',design)
    resource_design = dict(fit_manifest=fit_manifest_ref)
    resource_design_ref = put(root/'resource-design.json',resource_design)
    fit = dict(manifest=fit_manifest_ref,continuation=final_ref,intake_evidence=intake)
    fit_ref = put(root/'fit.json',fit)
    model = dict(schema=m.resource.SCHEMA,
        status='COMPLETE_SOURCE_REBOUND_RESOURCE_TRAINING_NOT_QUALIFICATION',resource_fit_run=True,
        validation_labels_read=False,design=resource_design_ref,continuation=final_ref,
        intake_evidence=intake,sources=[])
    model_ref = put(root/'model.json',model)
    report = dict(schema=m.source.SCHEMA,status='COMPLETE_REBOUND_UNQUALIFIED_FIT_ALL_WORLDS_UNKNOWN',
        response_fit_reused=True,validation_labels_read=False,sources=[],design=design_ref,
        response_fit=fit_ref,whole_block_leaveout=[dict(heldout_block_id=b,
            training_block_ids=[x for x in blocks if x!=b],
            donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!=b],
            heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']==b],
            folded_law=None,cases=[{k:v for k,v in r.items() if k!='block'} for r in rows if r['block']==b]) for b in blocks])
    report_ref = put(root/'completed.json',report)
    return dict(plans=plans,report=report,report_ref=report_ref,model=model,model_ref=model_ref,
        fit=fit,design=design,resource_design=resource_design,fit_manifest=fit_manifest)


class IndependentAllcaseResourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.data = fixture_all(cls.root/'data')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def authenticate(self, calls, *, data=None, generated_api=None, resource_api=None):
        d = self.data if data is None else data
        generated_api = m.source if generated_api is None else generated_api
        resource_api = m.resource if resource_api is None else resource_api
        provider_type = resource_api.FoldResourceTrainingProvider
        def provider(model_path,target,context,*,heldout_block_id,scenario,**kwargs):
            calls.append((heldout_block_id,target['attempt_id']))
            obj = object.__new__(provider_type)
            obj.target_plan, obj.target_context = copy.deepcopy(target), copy.deepcopy(context)
            obj.physical_execution_identity = m.resource.resource.physical_execution_identity(target,context)
            obj.fold_binding = dict(full_native_model=d['model_ref'],selected_fold_model_sha256='a'*64,
                heldout_block_id=heldout_block_id,
                training_block_ids=sorted({p['capture_id'] for p in d['plans']} - {heldout_block_id}),
                training_attempt_ids=[p['attempt_id'] for p in d['plans'] if p['capture_id']!=heldout_block_id],
                heldout_attempt_ids=[p['attempt_id'] for p in d['plans'] if p['capture_id']==heldout_block_id],sources=[])
            return obj
        return [patch.object(generated_api,'audit',return_value=d['design']),
            patch.object(generated_api.response,'audit_fit',return_value={'report':d['fit']}),
            patch.object(generated_api.response,'audit',return_value=d['fit_manifest']),
            patch.object(resource_api,'audit',return_value=d['resource_design']),
            patch.object(resource_api,'FoldResourceTrainingProvider',side_effect=provider)]

    def test_actual_full150_gold_after_persistence_and_1200_unknown_mass(self):
        calls, label_calls = [], []
        patches = self.authenticate(calls)
        native = m.gold.score_case
        output = self.root/'complete-output'
        def label(raw,plan,case,epoch,score,quality,*,source_evidence):
            prediction = source_evidence['persisted_resource_prediction']
            self.assertEqual(m.receipt(prediction['path']),prediction)
            doc = m.checked(prediction)
            self.assertEqual(m.checked(doc['target_plan'])['attempt_id'],plan['attempt_id'])
            self.assertEqual(raw['attempt_id'],plan['attempt_id'])
            self.assertEqual(doc['ensemble']['public_requests'],m.batch.canonical_requests(
                m.source.generated.public_case(case,epoch)))
            self.assertEqual(len(list(output.glob('*/*/case-*/prediction.json'))),len(label_calls)+1)
            label_calls.append((plan['capture_id'],plan['attempt_id'],case['case_id']))
            return native(raw,plan,case,epoch,score,quality,source_evidence=source_evidence)
        with patches[0],patches[1],patches[2],patches[3],patches[4], \
             patch.object(m.gold,'score_case',side_effect=label), \
             patch.object(m.source.response,'fit_bundle',side_effect=AssertionError('no solver')):
            result = m.checked(m.run_all(self.data['report_ref']['path'],self.data['model_ref']['path'],output))
        self.assertEqual(len(calls),3)
        self.assertEqual(len({b for b,_ in calls}),3)
        self.assertEqual(len(label_calls),150)
        self.assertEqual(len(set(label_calls)),150)
        self.assertEqual(result['offered_requests'],546)
        worlds = 0
        offered = 0
        score_plan = m.checked(self.data['design']['scoring_plan'])
        for row in result['output_cases']:
            doc = m.checked(row['prediction'])
            original = m.checked(doc['archived_prediction'])
            value = m.batch.single.decode(doc['ensemble'])
            worlds += len(value['draws'])
            offered += value['offered_requests']
            self.assertEqual(value['utility']['unknown_probability_mass'],1)
            self.assertIsNone(value['utility']['expected_value'])
            compact = m.batch.single.encode(m.source.generated.compact_prediction(
                dict(ensemble=doc['ensemble'])))
            self.assertEqual(compact['ensemble']['draws'],original['ensemble']['draws'])
            self.assertEqual(doc['past_memory_sources'],[])
            self.assertFalse(doc['missing_memory_state_assumed_fresh'])
            measured = m.checked(row['measured_label'])
            self.assertEqual(measured['attempt_id'],row['attempt_id'])
            self.assertEqual(measured['case_id'],row['case_id'])
            self.assertTrue(measured['heldout_future_used_for_evaluation_only'])
            self.assertIsNotNone(measured['measured_original_score']['terms']['E'])
            self.assertIsNotNone(measured['measured_original_score']['terms']['M'])
            self.assertEqual(measured['measured_resources']['startup_energy_joules'],40)
            self.assertEqual(measured['measured_resources']['measurement_board_energy_joules'],200)
            self.assertAlmostEqual(measured['measured_original_score']['terms']['M'],
                3000*score_plan['memory_safe_factor']/score_plan['memory_budget_mib'])
            self.assertIsNone(m.batch.single.decode(row['score_errors'])['U_error'])
        self.assertEqual((worlds,offered),(1200,546))
        self.assertFalse(result['native_label_values_used_as_predictions'])
        self.assertFalse(result['validation_labels_read'])

    def test_missing_case_and_donor_overlap_fail_before_provider_or_output(self):
        for kind in ('case','donor'):
            bad = copy.deepcopy(self.data['report'])
            if kind == 'case':
                bad['whole_block_leaveout'][0]['cases'].pop()
            else:
                fold = bad['whole_block_leaveout'][0]
                fold['donor_attempt_ids'][0] = fold['heldout_attempt_ids'][0]
            ref = put(self.root/f'bad-{kind}.json',bad)
            calls=[]
            patches=self.authenticate(calls)
            out = self.root/f'bad-{kind}-output'
            with self.subTest(kind=kind),patches[0],patches[1],patches[2],patches[3],patches[4]:
                with self.assertRaises(ValueError):
                    m.run_all(ref['path'],self.data['model_ref']['path'],out)
            self.assertEqual(calls,[])
            self.assertFalse(out.exists())

    def test_partially_unknown_ensemble_never_uses_conditional_mean_as_complete_error(self):
        ensemble = setup_world(unknown=True)[5]
        result = m.batch.single.decode(m.score_errors(ensemble,
            dict(terms={term:1 for term in 'QTLPME'},full_U=-1,unknown_reasons=[])))
        for term in 'QTLPME':
            row = result['terms'][term]
            self.assertIsNone(row['predicted_unconditional_expected_value'])
            self.assertIsNone(row['error'])
            self.assertGreater(row['original_distribution']['unknown_probability_mass'],0)
        self.assertIsNone(result['predicted_unconditional_expected_U'])
        self.assertIsNone(result['U_error'])
        self.assertFalse(result['conditional_U_used_as_complete_U'])
        self.assertFalse(result['UNKNOWN_mass_dropped'])

    def test_recovery_producers_full150_use_distinct_schemas_without_global_replacement(self):
        d=copy.deepcopy(self.data)
        root=self.root/'recovery-producer'
        d['fit_manifest']['original_executor_manifest']=d['fit_manifest'].pop('executor_manifest')
        fit_manifest_ref=put(root/'fit-manifest.json',d['fit_manifest'])
        d['design']['fit_manifest']=fit_manifest_ref
        design_ref=put(root/'design.json',d['design'])
        d['resource_design']['fit_manifest']=fit_manifest_ref
        resource_design_ref=put(root/'resource-design.json',d['resource_design'])
        d['fit']['manifest']=fit_manifest_ref
        fit_ref=put(root/'fit.json',d['fit'])
        d['model'].update(schema=recovery_resource.SCHEMA,status=recovery_resource.COMPLETE_STATUS,
            design=resource_design_ref)
        d['model_ref']=put(root/'model.json',d['model'])
        d['report'].update(schema=recovery_generated.SCHEMA,design=design_ref,response_fit=fit_ref)
        d['report_ref']=put(root/'completed.json',d['report'])
        calls=[]
        patches=self.authenticate(calls,data=d,generated_api=recovery_generated,resource_api=recovery_resource)
        original_source,original_resource=m.source,m.resource
        with patches[0],patches[1],patches[2],patches[3],patches[4]:
            result=m.checked(recovery_scorer.run_all(d['report_ref']['path'],d['model_ref']['path'],root/'output'))
        self.assertIs(m.source,original_source)
        self.assertIs(m.resource,original_resource)
        self.assertEqual(result['generated_producer_schema'],recovery_generated.SCHEMA)
        self.assertEqual(result['resource_producer_schema'],recovery_resource.SCHEMA)
        self.assertEqual((len(calls),result['cases'],result['offered_requests']),(3,150,546))
        self.assertTrue(result['labels_generated'])
        self.assertTrue(all(r['measured_label'] for r in result['output_cases']))
        for producer in (recovery_generated,recovery_resource):
            self.assertIn(m.receipt(producer.__file__),result['sources'])

    def test_conflicting_original_executor_receipts_reject_before_provider(self):
        d=copy.deepcopy(self.data)
        d['fit_manifest']['original_executor_manifest']=put(self.root/'foreign-executor.json',{'foreign':True})
        calls=[]
        patches=self.authenticate(calls,data=d)
        out=self.root/'conflicting-executor-output'
        with patches[0],patches[1],patches[2],patches[3],patches[4]:
            with self.assertRaisesRegex(ValueError,'unambiguous original executor'):
                m.run_all(d['report_ref']['path'],d['model_ref']['path'],out)
        self.assertEqual(calls,[])
        self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
