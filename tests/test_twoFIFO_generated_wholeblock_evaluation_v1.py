import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tests.test_twoFIFO_settled_history_initial_v1 import snapshot
from tools.predictor_validation_20261027 import twoFIFO_generated_wholeblock_evaluation_v1 as m


def past():
    value=snapshot();rename={'owner':'text#0','idle':'text#1'}
    value['ready']={rename[k]:v for k,v in value['ready'].items()}
    for row in value['request_records']:row['model']=rename[row['model']]
    return value


def public_case():
    return dict(case_id='public-case',arrivals=[dict(request_id='offered-'+str(i),owner='text#0',task='text',pool_id='input',arrival_ns=i*2_000_000) for i in range(2)])


def law():
    *_,vector=source_fixture();vectors=[]
    for block in ('training-block-1','training-block-2'):
        for i in range(2):
            v=copy.deepcopy(vector);v.update(training_block_id=block,vector_id=block+':'+str(i))
            v['transport_constraints'].update(enqueue_after_arrival_ns=0,sender_turn_extra_ns=0,
                submit_to_write_record_ns=1,observed_submit_to_START_ns=1,previous_READY_minus_submit_ns=-10)
            vectors.append(v)
    return dict(schema=m.joint.SCHEMA,blocks=['training-block-1','training-block-2'],heldout_block_id='training-block-3',
        features=list(m.response.FEATURES),coefficients=[0.]*12,vectors=vectors)


def scoring():
    return dict(aliases={owner:dict(family='text',supported_tasks=['text']) for owner in ('text#0','text#1')},
        workload_rps={'text':.2},latency_reference_ms={'text':1.},window_seconds=10.,startup_amortization_seconds=3600,
        memory_budget_mib=1000,memory_safe_factor=1.2,energy_reference_watts=10,weights={k:1/6 for k in 'QTLPME'})
QUALITY={'text':{'text':.25}}


class WholeBlockGeneratedTests(unittest.TestCase):
    def test_real_fold_law_compilation_generates_all_predeclared_worlds_and_public_slots(self):
        case=public_case();value=m.forecast_case(case,1000,past(),law(),scoring(),QUALITY)
        out=m.decode(value['ensemble'])
        self.assertEqual(len(out['draws']),8)
        self.assertEqual(sum(d['weight'] for d in out['draws']),1)
        self.assertEqual(out['world_unknown_probability_mass'],0)
        self.assertEqual(out['offered_requests'],2)
        self.assertEqual(out['utility']['unknown_probability_mass'],1)
        self.assertEqual({d['block_id'] for d in out['draws']},{'training-block-1','training-block-2'})
        self.assertTrue(all(d['forecast']['world']['offered_requests']==2 for d in out['draws']))
        self.assertTrue(value['generation_inputs_exclude_heldout_future_labels'])
        self.assertEqual(value['settled_seed']['producer_next']['text#0:modeled_stdout_lifetime'],3)
        self.assertTrue(all(len(r['selections'])==2 for r in value['selections']))

    def test_missing_public_donor_retains_half_unknown_mass_not_conditional_renormalization(self):
        donor=law();donor['vectors']=[v for v in donor['vectors'] if v['training_block_id']=='training-block-1']
        out=m.decode(m.forecast_case(public_case(),1000,past(),donor,scoring(),QUALITY)['ensemble'])
        self.assertEqual(out['world_unknown_probability_mass'],F(1,2))
        self.assertEqual(len(out['draws']),8)
        self.assertTrue(all(len(d['all_public_request_ids_retained'])==2 for d in out['draws'] if d['status']=='UNKNOWN_WORLD'))
        self.assertIsNone(out['expected_world_complete_cohort_P95_ns']['expected_value'])
        self.assertFalse(out['unknown_worlds_renormalized'])

    def test_missing_or_unsettled_past_keeps_every_world_unknown(self):
        for snapshot_value in (None,dict(past(),queued_runtime_events=[{}])):
            out=m.decode(m.forecast_case(public_case(),1000,snapshot_value,law(),scoring(),QUALITY)['ensemble'])
            self.assertEqual(out['world_unknown_probability_mass'],1)
            self.assertEqual(out['offered_requests'],2);self.assertEqual(len(out['draws']),8)
            self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds']==[0,1] for r in out['fixed_drain_risk']))

    def test_programming_errors_are_not_converted_to_unknown_compilation(self):
        with patch.object(m.joint,'compile_draw_v4',side_effect=ValueError('bad contract')):
            with self.assertRaisesRegex(ValueError,'bad contract'):m.compile_case_draws(m.public_case(public_case(),1000),law())
        bad=law();bad['vectors'][0]['training_block_id']='training-block-3'
        with self.assertRaisesRegex(ValueError,'donor block outside'):m.compile_case_draws(m.public_case(public_case(),1000),bad)

    def test_public_slot_scheme_is_fixed_and_request_ids_do_not_select_donors(self):
        rows=m.public_case(public_case(),1000);one,a=m.compile_case_draws(rows,law())
        renamed=[dict(r,request_id='new-'+r['request_id']) for r in rows];two,b=m.compile_case_draws(renamed,law())
        self.assertEqual(a,b)
        self.assertEqual({m.public_quantile(('text#0','text','input'),2,i) for i in range(4)},{F(1,8),F(3,8),F(5,8),F(7,8)})
        self.assertEqual([r['inputs'] for r in one],[r['inputs'] for r in two])

    def test_heldout_future_macro_ready_query_poison_cannot_change_prediction(self):
        case=public_case();captured_past=past()
        raw=dict(case_histories=[dict(prior_history_snapshot=captured_past)],request_records=[dict(request_id='offered-0',
            completion_time_ns=1200,execution_events=[dict(future='body')],previous_handler_next_request_ready=dict(perf_ns=99999))])
        before=m.forecast_case(case,1000,raw['case_histories'][0]['prior_history_snapshot'],law(),scoring(),QUALITY)
        raw['request_records'][0].update(completion_time_ns=10**18,execution_events=[dict(unbounded_query='poison')],
            previous_handler_next_request_ready=dict(perf_ns=10**18))
        after=m.forecast_case(case,1000,raw['case_histories'][0]['prior_history_snapshot'],law(),scoring(),QUALITY)
        self.assertEqual(before,after)

    def test_compact_errors_keep_world_unknowns_requests_and_late_native_labels(self):
        case=public_case();prediction=m.compact_prediction(m.forecast_case(case,1000,past(),law(),scoring(),QUALITY))
        raw=dict(request_records=[dict(request_id=a['request_id'],status='completed',completion_time_ns=1000+a['arrival_ns']+2000) for a in case['arrivals']])
        labels=m.native_labels(raw,case,1000,scoring(),QUALITY);errors=m.errors(prediction,labels)
        self.assertEqual(errors['offered_requests'],2);self.assertEqual(errors['point_supported_requests'],2)
        self.assertTrue(all(len(r['draws'])==8 for r in errors['requests']))
        self.assertTrue(all(r['actual_reader_E2E_ns']==2000 for r in errors['requests']))
        self.assertFalse(labels['future_macro_READY_query_bodies_read'])
        self.assertTrue(labels['quality_from_frozen_training_expectations_not_native_outcome'])
        self.assertFalse(labels['actual_native_quality_validated'])
        self.assertFalse(labels['full_U_resource_qualification'])
        self.assertTrue(all(d['forecast_sha256'] for d in prediction['ensemble']['draws']))
        self.assertTrue(all('trace' not in d for d in prediction['ensemble']['draws']))

    def test_incomplete_twelve_gate_runs_no_numeric_raw_or_law(self):
        design=dict(campaign={'path':'campaign'},scoring_plan={'path':'score'},quality={'path':'quality'},sources=[])
        bundle=dict(intake=dict(complete=False),documents=None,sources=[])
        def read(ref):return scoring() if ref['path']=='score' else QUALITY
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'audit_design',return_value=({'path':'design','sha256':'x'},design)),\
                patch.object(m,'checked',side_effect=read),patch.object(m.response,'load_complete',return_value=bundle),\
                patch.object(m.response,'fit_new_blocks') as fit,patch.object(m.joint,'build_joint_law_v4') as build,\
                patch.object(m,'strict_references',return_value=[]):
            out=json.loads(Path(m.run('design',[],Path(directory)/'out')['path']).read_text())
        fit.assert_not_called();build.assert_not_called();self.assertFalse(out['numerical_fit_run'])
        self.assertEqual(out['forecasts_generated'],0);self.assertEqual(out['whole_block_leaveout'],[])

    def test_prepare_fixes_catalog_and_quality_receipts_before_fitting(self):
        campaign=dict(schema=m.response.intake.campaign_v2.SCHEMA,attempts=[])
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory)
            paths={}
            for name,value in [('campaign',campaign),('score',scoring()),('quality',QUALITY)]:
                path=directory/(name+'.json');path.write_text(json.dumps(value));paths[name]=path
            with patch.object(m.response.intake.campaign_v2,'audit',return_value=campaign),patch.object(m.response.numeric,'fit') as fit:
                ref=m.prepare(paths['campaign'],paths['score'],paths['quality'],directory/'prepared')
                _,design=m.audit_design(ref['path'])
            fit.assert_not_called();self.assertEqual(design['scheme'],m.SCHEME)
            self.assertEqual(design['quality'],m.receipt(paths['quality']))
            self.assertEqual(design['scoring_plan'],m.receipt(paths['score']))
            value=json.loads(Path(ref['path']).read_text())
            for field,replacement in (('scheme',dict(value['scheme'],slots_per_training_block=8)),
                    ('training_block_ids',['training-block-1']),('heldout_unit','one_raw_SHA')):
                bad=copy.deepcopy(value);bad[field]=replacement;Path(ref['path']).write_text(json.dumps(bad))
                with self.assertRaises(ValueError):m.audit_design(ref['path'])

    def test_prediction_is_durable_before_case_evaluation_label_extraction(self):
        case=public_case();raw=dict(request_records=[dict(request_id=a['request_id'],status='completed',
            completion_time_ns=1000+a['arrival_ns']+2000) for a in case['arrivals']])
        original=m.native_labels
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory)/'case'
            def read_labels(*args):
                self.assertTrue((folder/'prediction.json').exists())
                prediction=json.loads((folder/'prediction.json').read_text())
                self.assertTrue(prediction['generation_inputs_exclude_heldout_future_labels'])
                self.assertEqual(prediction['ensemble']['offered_requests'],2)
                return original(*args)
            with patch.object(m,'native_labels',side_effect=read_labels) as labels:
                report=m.evaluate_case(raw,case,1000,past(),law(),scoring(),QUALITY,folder,
                    donor_blocks=['training-block-1','training-block-2'])
            self.assertEqual(labels.call_count,1);self.assertTrue(report['durable_before_case_evaluation_label_extraction'])
            self.assertTrue((folder/'labels.json').exists())

    def test_unqualified_complete_fit_does_not_compile_law_or_silently_drop_requests(self):
        with patch.object(m,'compile_case_draws') as compile:
            value=m.forecast_case(public_case(),1000,past(),None,scoring(),QUALITY,
                donor_blocks=['training-block-1','training-block-2'],unqualified_reason='UNKNOWN_UNQUALIFIED_ROOT')
        compile.assert_not_called();out=m.decode(value['ensemble'])
        self.assertEqual(out['world_unknown_probability_mass'],1);self.assertEqual(out['offered_requests'],2)
        self.assertTrue(all(r['reason']=='UNKNOWN_UNQUALIFIED_ROOT' for r in out['draws']))

    def test_complete_orchestration_keeps_all_150_cases_546_requests_and_1200_unknown_worlds(self):
        with tempfile.TemporaryDirectory() as directory:
            directory=Path(directory);plans=[];reviews=[];sources=[]
            for block in range(1,4):
                counter=0
                for shard,size in enumerate((13,13,13,11),1):
                    attempt='training-block-'+str(block)+'-shard-'+str(shard);cases=[];observed=[];histories=[];rows=[]
                    for index in range(size):
                        ordinal=counter;counter+=1;epoch=1000+ordinal*70_000_000_000
                        case=public_case();case['case_id']=attempt+':case:'+str(index)
                        case['arrivals']=[dict(case['arrivals'][0],request_id=case['case_id']+':request:'+str(i),arrival_ns=i*2_000_000)
                            for i in range(4 if ordinal<32 else 3)]
                        cases.append(case);observed.append(dict(case_id=case['case_id'],observation_start_ns=epoch))
                        histories.append(dict(case_id=case['case_id'],prior_history_snapshot=past()))
                        rows.extend(dict(request_id=a['request_id'],status='completed',completion_time_ns=epoch+a['arrival_ns']+2000) for a in case['arrivals'])
                    plan=dict(attempt_id=attempt,capture_id='training-block-'+str(block),cases=cases);plans.append(plan)
                    raw=dict(cases=observed,case_histories=histories,request_records=rows)
                    plan_path=directory/(attempt+'-plan.json');plan_path.write_text(json.dumps(plan))
                    raw_path=directory/(attempt+'-raw.json');raw_path.write_text(json.dumps(raw))
                    review=dict(inputs={'plan.json':m.receipt(plan_path)},raw=m.receipt(raw_path))
                    review_path=directory/(attempt+'-review.json');review_path.write_text(json.dumps(review));reviews.append(review_path)
            score_path=directory/'score.json';score_path.write_text(json.dumps(scoring()))
            quality_path=directory/'quality.json';quality_path.write_text(json.dumps(QUALITY))
            design_path=directory/'design.json';design_path.write_text('{}')
            design=dict(campaign={'path':'campaign'},scoring_plan=m.receipt(score_path),quality=m.receipt(quality_path),
                training_block_ids=['training-block-'+str(i) for i in (1,2,3)],sources=[])
            bundle=dict(intake=dict(complete=True),documents=[{}]*12,plans=plans,sources=[],
                denominator=dict(measured_requests=546,priming_requests_excluded_as_labels=216))
            with patch.object(m,'audit_design',return_value=(m.receipt(design_path),design)),\
                    patch.object(m.response,'load_complete',return_value=bundle),\
                    patch.object(m.response,'fit_new_blocks',return_value=dict(response_law_compilation_eligible=False)),\
                    patch.object(m.joint,'build_joint_law_v4') as law_builder:
                ref=m.run(design_path,reviews,directory/'evaluation')
            report=json.loads(Path(ref['path']).read_text());law_builder.assert_not_called()
            self.assertEqual(report['forecasts_generated'],150);self.assertEqual(len(report['whole_block_leaveout']),3)
            self.assertTrue(report['numerical_fit_run']);self.assertIsNone(report['full_law'])
            self.assertEqual(sum(f['summary']['offered_requests'] for f in report['whole_block_leaveout']),546)
            self.assertTrue(all(f['summary']['point_supported_requests']==0 for f in report['whole_block_leaveout']))
            self.assertTrue(all(len(f['cases'])==50 and len(f['donor_attempt_ids'])==8 and len(f['heldout_attempt_ids'])==4
                for f in report['whole_block_leaveout']))
            self.assertIsNone(report['equal_whole_block_mean_absolute_error_ns'])
            self.assertTrue(all(c['durable_before_case_evaluation_label_extraction'] for f in report['whole_block_leaveout'] for c in f['cases']))
            predictions=[json.loads(Path(c['prediction']['path']).read_text()) for f in report['whole_block_leaveout'] for c in f['cases']]
            self.assertEqual(sum(len(p['ensemble']['draws']) for p in predictions),1200)
            self.assertTrue(all(p['ensemble']['world_unknown_probability_mass']=={'numerator':1,'denominator':1} for p in predictions))


if __name__=='__main__':unittest.main()
