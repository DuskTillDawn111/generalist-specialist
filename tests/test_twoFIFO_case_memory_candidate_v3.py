import copy
from fractions import Fraction as F
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_joint_request_law_v2 import source_fixture
from tools.predictor_validation_20261027 import twoFIFO_case_memory_candidate_v3 as m

PHYSICAL='0'*64


def fixture():
    *_,vector=source_fixture()
    for node in vector['nodes']:
        if node['phase']=='prefill':node['forward_shape']=dict(input_sequence_length=8,input_batch_size=1,forward_index=0)
        elif node['phase']=='decode':node['forward_shape']=dict(input_sequence_length=1,input_batch_size=1,forward_index=1)
    rows=[];donors={};owner=vector['public_key'][0];descriptor=m.generated.shapes.CausalState([],[owner]).descriptor()
    context=m.support_context(descriptor,{'text#0':9,'text#1':0,'generalist_2#0':0},[vector['public_key']])
    domain=dict(nominal_poll_cadence_ns=20_000_000,horizon_ns=10_000_000_000)
    for block in (1,2,3):
        for shard in (1,2,3,4):
            attempt='training-block-'+str(block)+'-shard-'+str(shard)
            donors[attempt]=dict(training_block_id='training-block-'+str(block),P0_mib=1000.,shape_descriptor=descriptor)
            for index,x in enumerate(([1.,0.,0.,0.],[1.,10.,0.,0.],[1.,0.,4.,0.],[1.,0.,0.,2.])):
                rows.append(dict(schema=m.ROW_SCHEMA,capture_id='training-block-'+str(block),attempt_id=attempt,case_id='case-'+str(index),
                    physical_execution_identity=PHYSICAL,resident_owners=[owner],features=x,P0_mib=1000.,
                    target_absolute_sampled_peak_mib=1000.+1024*(.1+.2*x[1]+.3*x[2]+.4*x[3]),
                    public_keys=[vector['public_key']],observation_domain=domain,support_context=copy.deepcopy(context),CPU_TEST_ONLY=True))
    return vector,rows,donors,domain


def world_fixture():
    vector,rows,donors,domain=fixture();compiled=m.generated.joint.compile_draw_v4(vector,0)
    key=tuple(vector['public_key']);owner=key[0]
    # Fixed explicit paired constraints keep this independent CPU world simple.
    vector['transport_constraints'].update(enqueue_after_arrival_ns=0,sender_turn_extra_ns=0,
        submit_to_write_record_ns=1,observed_submit_to_START_ns=2,previous_READY_minus_submit_ns=1)
    compiled=m.generated.joint.compile_draw_v4(vector,0)
    requests=[dict(request_id='generated',owner=owner,task=key[1],pool_id=key[2],arrival_ns=100)]
    world=m.generated.engine.simulate(requests=requests,plans={key:compiled['plan']},rates={(owner,n.phase,0,0):F(1) for n in compiled['plan'].macros},
        protocol_slacks={compiled['plan']:compiled['protocol_slacks']},transport_draws={(*key,0):compiled['transport_draw']},
        window_start_ns=100,window_end_ns=101,drain_end_ns=100+domain['horizon_ns'],
        initial_sender_available_ns={owner:100},initial_worker_read_release_ns={owner:102},
        initial_protocol_state=dict(reader_free={},collector_free=100,last_collector_arrival=100,pending_messages=[]))
    sidecar=m.generated.selected_shape_sidecar([dict(vector=vector,public_key_arrival_ordinal=0)],physical_execution_identity=PHYSICAL)
    donors={k:v for k,v in donors.items() if v['training_block_id']!='training-block-3'}
    model=m.fit_rows(rows[:32],startup_donors=donors)
    origin=dict(scope='selected_training_donor_after_original_priming',donor_attempt_id='training-block-1-shard-1')
    return sidecar,world,requests,model,origin,domain


class CaseMemoryCandidateTests(unittest.TestCase):
    def test_fixed_four_column_fit_P0_offset_and_all_wholeblock_folds(self):
        _,rows,donors,_=fixture();model=m.fit_rows(rows,startup_donors=donors)
        self.assertEqual(model['numeric_rank'],4);self.assertEqual(model['columns'],list(m.COLUMNS))
        for actual,expected in zip(model['coefficients'],(.1,.2,.3,.4)):self.assertAlmostEqual(actual,expected)
        for block in ('training-block-1','training-block-2','training-block-3'):
            train=[r for r in rows if r['capture_id']!=block];heldout=[r for r in rows if r['capture_id']==block]
            fold=m.fit_rows(train,startup_donors={k:v for k,v in donors.items() if v['training_block_id']!=block})
            report=m.evaluate_explanatory_rows(heldout,fold)
            self.assertEqual(len(fold['training_attempt_ids']),8);self.assertEqual(report['heldout_cases'],16)
            self.assertTrue(all(abs(r['signed_error_mib'])<1e-8 for r in report['rows']))
            self.assertFalse(report['prospective_queue_memory_prediction_validated'])

    def test_donor_only_generated_features_real_engine_and_queries_do_not_accumulate(self):
        sidecar,world,requests,model,origin,domain=world_fixture()
        features=m.generated_case_features(sidecar,world,requests,model=model,memory_origin=origin,observation_domain=domain)
        self.assertEqual(features['status'],'DONOR_ONLY_GENERATED_CASE_FEATURES')
        self.assertAlmostEqual(features['features'][1],__import__('math').log1p(9))
        before=m.predict_generated_case(model,features,query_times=[200])
        after=m.predict_generated_case(model,features,query_times=[200]*100+[1000])
        self.assertEqual(before['prediction'],after['prediction']);self.assertIsNotNone(after['prediction']['peak_memory_mib'])
        self.assertTrue(all(q['prediction']==after['prediction'] for q in after['queries']))
        self.assertFalse(after['original_safe_factor_applied']);self.assertFalse(after['repeated_queries_accumulate_memory'])

    def test_heldout_memory_poison_cannot_change_fold_model_or_generated_inputs(self):
        _,rows,donors,_=fixture();baseline=m.fit_rows(rows[:32],startup_donors={k:v for k,v in donors.items() if v['training_block_id']!='training-block-3'})
        poisoned=copy.deepcopy(rows)
        for row in poisoned[32:]:row['target_absolute_sampled_peak_mib']=1e15;row['features']=[1.,999.,999.,999.]
        self.assertEqual(baseline,m.fit_rows(poisoned[:32],startup_donors={k:v for k,v in donors.items() if v['training_block_id']!='training-block-3'}))
        with self.assertRaisesRegex(ValueError,'heldout block'):m.evaluate_explanatory_rows(rows[:1],baseline)
        with self.assertRaisesRegex(ValueError,'Startup donor outside'):m.fit_rows(rows[:32],startup_donors=donors)
        sidecar,world,requests,model,origin,domain=world_fixture()
        wrong=copy.deepcopy(sidecar);wrong['slots'][0]['training_block_id']='training-block-3'
        with self.assertRaisesRegex(ValueError,'only from donor blocks'):
            m.generated_case_features(wrong,world,requests,model=model,memory_origin=origin,observation_domain=domain)

    def test_future_memory_labels_are_rejected_and_missing_events_not_repaired(self):
        sidecar,world,requests,model,origin,domain=world_fixture()
        for key in ('board_samples','target_absolute_sampled_peak_mib','request_records'):
            poisoned=copy.deepcopy(world);poisoned[key]=[999999]
            with self.assertRaisesRegex(ValueError,'future execution/memory labels'):
                m.generated_case_features(sidecar,poisoned,requests,model=model,memory_origin=origin,observation_domain=domain)
        missing=m.generated.engine.decode(world)
        index=next(i for i,r in enumerate(missing['trace']) if r['kind']=='processed_notification' and r['action']=='phase')
        missing['trace'].pop(index)
        result=m.generated_case_features(sidecar,missing,requests,model=model,memory_origin=origin,observation_domain=domain)
        self.assertEqual(result['status'],'UNKNOWN');self.assertIn('MISSING_GENERATED_PROCESSED_NOTICE',result['unknown_reasons'])
        predicted=m.predict_generated_case(model,result);self.assertIsNone(predicted['prediction']['peak_memory_mib'])

    def test_unsupported_rows_and_underestimates_remain_and_rank_failure_is_explicit(self):
        _,rows,_,_=fixture();model=m.fit_rows(rows[:32]);heldout=copy.deepcopy(rows[32:])
        heldout[0]['features'][1]=999.;heldout[1]['target_absolute_sampled_peak_mib']+=100.
        report=m.evaluate_explanatory_rows(heldout,model)
        self.assertEqual(len(report['rows']),16);self.assertIsNone(report['rows'][0]['signed_error_mib'])
        self.assertAlmostEqual(report['rows'][1]['signed_error_mib'],-100.)
        self.assertTrue(report['rows'][1]['underestimated'])
        unknown=copy.deepcopy(rows[:32]);unknown[0]['features']=None
        partial=m.fit_rows(unknown);self.assertEqual(partial['unknown_training_rows'],1)
        self.assertEqual(len(partial['training_rows']),32)
        self.assertIn('KNOWN_ROWS_ONLY',partial['status']);self.assertFalse(partial['full_training_denominator_supported'])
        deficient=copy.deepcopy(rows[:32])
        for row in deficient:row['features']=[1.,1.,1.,1.]
        rank=m.fit_rows(deficient);self.assertEqual(rank['numeric_rank'],1);self.assertIsNone(rank['coefficients'])
        evaluated=m.evaluate_explanatory_rows(rows[32:],rank)
        self.assertEqual(evaluated['unknown_predictions'],16)

    def test_support_context_and_design_diagnostics_do_not_silently_extrapolate(self):
        _,rows,_,_=fixture();model=m.fit_rows(rows[:32]);heldout=copy.deepcopy(rows[32:])
        heldout[0]['support_context']['prior_owner_shape_envelope']['text#0']=100
        report=m.evaluate_explanatory_rows(heldout,model)
        self.assertIn('PRIOR_SHAPE_ENVELOPE_OUTSIDE_DONOR_RANGE',report['rows'][0]['prediction']['unknown_reasons'])
        changed=heldout[1]['support_context'];changed['task_state']['owner_prior_envelope_departures']['text#0']=False
        changed['task_state_key']=m.resource.digest(changed['task_state'])
        report=m.evaluate_explanatory_rows(heldout,model)
        self.assertIn('UNKNOWN_EXACT_TASK_STATE_OR_ENVELOPE_DEPARTURE_PATTERN',report['rows'][1]['prediction']['unknown_reasons'])
        diagnostic=report['rows'][2]['prediction']['support_diagnostic']
        self.assertLess(diagnostic['design_span_distance'],1e-8);self.assertGreater(diagnostic['leverage'],0)
        self.assertTrue(diagnostic['leverage_is_diagnostic_only'])
        model['coefficients']=model['coefficients'][:3]
        with self.assertRaisesRegex(ValueError,'Exactly four'):m.evaluate_explanatory_rows(heldout,model)

    def test_nonfinite_memory_arithmetic_and_numerical_fit_failures_remain_UNKNOWN(self):
        _,rows,_,_=fixture();model=m.fit_rows(rows[:32]);row=rows[32]
        for coeff in ([1e308,0.,0.,0.],[-1e308,0.,0.,0.],[1e308,-1e308,0.,0.]):
            with self.subTest(coeff=coeff):
                model['coefficients']=coeff
                values=[1.,10.,0.,0.] if coeff[1] else row['features']
                prediction=m._predict(model,values,row['P0_mib'],row['public_keys'],row['observation_domain'],row['support_context'])
                self.assertEqual(prediction['status'],'UNKNOWN');self.assertIsNone(prediction['peak_memory_mib'])
                self.assertIn('UNKNOWN_NONFINITE_MEMORY_ARITHMETIC',prediction['unknown_reasons'])
        huge=copy.deepcopy(rows[:32]);huge[0]['features'][1]=1e308
        fitted=m.fit_rows(huge)
        self.assertEqual(fitted['status'],'UNKNOWN_MEMORY_FIT_NUMERICAL_FAILURE')
        self.assertEqual(fitted['training_rows'],huge);self.assertEqual(fitted['total_training_rows'],32)
        self.assertIsNone(fitted['coefficients']);self.assertIsNotNone(fitted['fit_numerical_failure'])
        evaluated=m.evaluate_explanatory_rows(rows[32:],fitted)
        self.assertEqual(evaluated['unknown_predictions'],16)
        self.assertIn('MEMORY_FIT_NUMERICAL_FAILURE',evaluated['rows'][0]['prediction']['unknown_reasons'])
        __import__('json').dumps(fitted,allow_nan=False)
        for failure in (FloatingPointError('CPU_TEST_ONLY'),m.np.linalg.LinAlgError('CPU_TEST_ONLY')):
            with self.subTest(failure=type(failure).__name__),patch.object(m.np.linalg,'lstsq',side_effect=failure):
                failed=m.fit_rows(rows[:32])
                self.assertEqual(failed['status'],'UNKNOWN_MEMORY_FIT_NUMERICAL_FAILURE');self.assertEqual(len(failed['training_rows']),32)
        ordinary=m.fit_rows(rows[:32]);extreme=copy.deepcopy(row);extreme['features'][1]=1e308
        predicted=m._predict(ordinary,extreme['features'],extreme['P0_mib'],extreme['public_keys'],extreme['observation_domain'],extreme['support_context'])
        self.assertEqual(predicted['status'],'UNKNOWN');self.assertIn('NONFINITE_DESIGN_GEOMETRY_ARITHMETIC',predicted['unknown_reasons'])
        __import__('json').dumps(predicted,allow_nan=False)

    def test_resident_origin_requires_causal_P0_and_seed_never_implies_memory(self):
        sidecar,world,requests,model,origin,domain=world_fixture()
        descriptor=model['startup_donors']['training-block-1-shard-1']['shape_descriptor']
        resident=dict(scope='resident_observed_before_origin',P0_mib=1000.,shape_descriptor=descriptor,last_observed_poll_end_ns=99)
        features=m.generated_case_features(sidecar,world,requests,model=model,memory_origin=resident,observation_domain=domain)
        self.assertEqual(features['P0_mib'],1000.)
        resident['last_observed_poll_end_ns']=101
        with self.assertRaisesRegex(ValueError,'strictly before origin'):
            m.generated_case_features(sidecar,world,requests,model=model,memory_origin=resident,observation_domain=domain)
        with self.assertRaisesRegex(ValueError,'queue seed is not memory'):
            m.generated_case_features(sidecar,world,requests,model=model,memory_origin={'scope':'settled-seed'},observation_domain=domain)

    def test_actual_target_adapter_prefix_peak_is_causal_and_rows_remain_explanatory(self):
        from tests.test_twoFIFO_owned_training_review_v3 import cohort_fixture
        plan,raw=cohort_fixture()
        # The fixture already carries genuine request/phase ordering; only
        # resource target values are CPU labels, not a native-fit claim.
        raw['lifecycle_boundaries']=[dict(boundary='startup_begin',time_ns=0)]
        raw['measurement_windows']=[dict(case_id=c['case_id'],start_ns=c['epoch_ns'],end_ns=c['measurement_end_ns'],
            fixed_drain_deadline_ns=c['fixed_drain_deadline_ns']) for c in raw['cases']]
        plan['resident_owners']=['text#0','text#1','generalist_2#0','ocr#0']
        start=raw['measurement_windows'][0]['start_ns']
        raw['board_samples']=[dict(time_ns=start-2,query_duration_ns=0,owned_used_mib=100.),
            dict(time_ns=start+1,query_duration_ns=0,owned_used_mib=999999.)]
        native=dict(physical_execution_identity=PHYSICAL,measured_case_resources=dict(cases=[
            dict(case_id=w['case_id'],startup_through_this_drain_owned_sampled_peak_mib=999999.) for w in raw['measurement_windows']]))
        rows=m.actual_rows_from_verified(raw,plan,{},native)
        self.assertEqual(rows[0]['P0_mib'],100.)
        self.assertEqual(rows[0]['target_absolute_sampled_peak_mib'],999999.)
        self.assertTrue(rows[0]['actual_future_trajectory_used']);self.assertFalse(rows[0]['prospective_queue_memory_validation'])

    def test_partial_native_intake_never_extracts_new_targets_or_fits(self):
        campaign=Path('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json');metadata=m.checked(m.receipt(campaign))
        with tempfile.TemporaryDirectory() as directory,patch.object(m.resource.intake.campaign_v2,'audit',return_value=metadata),\
                patch.object(m.resource.macros_v2,'extract') as extract,patch.object(m,'actual_rows_from_verified') as adapter,patch.object(m,'fit_rows') as fit:
            with self.assertRaisesRegex(ValueError,'All twelve'):m.run(campaign,[],Path(directory)/'output')
        extract.assert_not_called();adapter.assert_not_called();fit.assert_not_called()

    def test_complete_native_loader_cannot_bypass_per_attempt_case_order_gate(self):
        tables=[dict(capture_id='training-block-'+str(b),raw='raw',plan='plan',context='context') for b in (1,2,3) for _ in range(4)]
        plan=dict(cases=[dict(case_id='one'),dict(case_id='two')])
        def read(value):return plan if value=='plan' else {}
        with tempfile.TemporaryDirectory() as directory,patch.object(m.resource,'load_complete',return_value=(tables,[])),\
                patch.object(m,'checked',side_effect=read),patch.object(m,'actual_rows_from_verified',return_value=[dict(case_id='two'),dict(case_id='one')]),\
                patch.object(m,'fit_rows') as fit:
            with self.assertRaisesRegex(ValueError,'Every original ordered case'):m.run('unused',[],Path(directory)/'output')
        fit.assert_not_called()


if __name__=='__main__':unittest.main()
