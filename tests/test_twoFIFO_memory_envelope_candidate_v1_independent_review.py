"""Independent causal filtered-memory replay checks, never a future provider."""
import copy
import unittest
from unittest.mock import patch
from tests.test_twoFIFO_memory_envelope_candidate_v1 import training_tables
from tools.predictor_validation_20261027 import twoFIFO_memory_envelope_candidate_v1 as m


def event(at,sequence):
    return dict(request_id='r',event='start',segment_id='prefill_0',semantic_phase='prefill',
        input_batch_size=1,input_sequence_length=sequence,forward_index=0,
        perf_ns=at-3,received_time_ns=at-2,collector_dequeued_time_ns=at-1,collector_processing_time_ns=at)


def causal_fixture():
    raw=dict(priming_request_records=[],request_records=[dict(request_id='r',model='text#0',task='text',
        pool_id='exact',profile_job_id='case',execution_events=[event(139,99999),event(145,888888)])],
        board_samples=[dict(time_ns=50,query_duration_ns=20,owned_used_mib=50),
            dict(time_ns=100,query_duration_ns=20,owned_used_mib=100),
            dict(time_ns=125,query_duration_ns=60,owned_used_mib=999),
            dict(time_ns=150,query_duration_ns=20,owned_used_mib=500)],
        lifecycle_boundaries=[dict(boundary='startup_begin',time_ns=0)],
        measurement_windows=[dict(case_id='case',start_ns=20,end_ns=120,fixed_drain_deadline_ns=200)])
    plan=dict(capture_id='block',attempt_id='block-shard-1',execution_id='CPU',resident_owners=['text#0'])
    return raw,plan


class MemoryCandidateIndependentReview(unittest.TestCase):
    def test_overlapping_prior_poll_is_not_anchor_and_current_label_not_feature(self):
        raw,plan=causal_fixture()
        with patch.object(m.observed,'observed',return_value=dict(cases=[])),patch.object(m.resource,'physical_execution_identity',return_value='CPU'):
            table=m.table_from_verified(raw,plan,{})
            at=table['polls'][3]
            self.assertEqual(at['poll_begin_ns'],140)
            self.assertEqual(at['prior_observed_owned_peak_mib'],100)
            self.assertEqual(at['positive_increment_mib'],400)
            shape=at['state']['current_processed_owner_task_exact_input_phase_shape'][0]['actual_BEGIN_shape']
            self.assertEqual(shape['input_sequence_length'],99999)
            # This BEGIN is visible before poll query start even though its
            # request/future pool fields were known earlier; next BEGIN is not.
            raw['request_records'][0]['execution_events'][1]['input_sequence_length']=123456789
            raw['board_samples'][2]['owned_used_mib']=10**9
            raw['board_samples'][3]['owned_used_mib']=700
            changed=m.table_from_verified(raw,plan,{})['polls'][3]
            self.assertEqual(changed['state'],at['state'])
            self.assertEqual(changed['state_key'],at['state_key'])
            self.assertEqual(changed['prior_observed_owned_peak_mib'],100)
            self.assertEqual(changed['positive_increment_mib'],600)
            self.assertTrue(table['all_original_poll_rows_retained'])

    def test_state_support_in_one_block_stays_unknown_and_all_cases_polls_retained(self):
        tables=training_tables();train=tables[:8]
        for table in train:
            if table['capture_id']=='training-block-2':table['polls'][0]['state_key']='other'
        model=m.fit_tables(train)
        self.assertFalse(model['states']['crossed-state']['known'])
        heldout=copy.deepcopy(tables[-1]);heldout['cases'].append(dict(case_id='empty',startup_through_this_drain_owned_sampled_peak_mib=333))
        out=m.evaluate_table(heldout,model)
        self.assertEqual(out['offered_cases'],2)
        self.assertEqual(len(out['cases']),2)
        self.assertEqual(out['unknown_polls'],1)
        self.assertFalse(out['cases'][0]['all_polls_supported'])
        self.assertIsNone(out['cases'][0]['predicted_owned_sampled_peak_mib'])
        self.assertEqual(out['cases'][1]['polls'],0)
        self.assertIsNone(out['cases'][1]['predicted_owned_sampled_peak_mib'])
        self.assertFalse(out['prospective_generated_resource_prediction'])
        self.assertFalse(out['capacity_bound'])
        self.assertFalse(model['memory_safety_factor_applied'])

    def test_evaluation_rejects_different_physical_identity_and_schema(self):
        tables=training_tables();model=m.fit_tables(tables[:8])
        for mutation in ('physical','table_schema','model_schema'):
            table=copy.deepcopy(tables[-1]);fitted=copy.deepcopy(model)
            if mutation=='physical':table['physical_execution_identity']='different-hardware'
            elif mutation=='table_schema':table['schema']='unreviewed'
            else:fitted['schema']='unreviewed'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):m.evaluate_table(table,fitted)
