import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests import test_twoFIFO_owned_training_review_v3 as fixture
from tests.test_twoFIFO_observed_resources_v1 import fixture as resource_fixture,SECOND
from tools.predictor_validation_20261027 import twoFIFO_memory_envelope_candidate_v1 as m


def training_tables():
    result=[]
    for block in (1,2,3):
        for shard in (1,2,3,4):
            result.append(dict(schema=m.TABLE_SCHEMA,capture_id='training-block-'+str(block),
                attempt_id='training-block-'+str(block)+'-shard-'+str(shard),physical_execution_identity='same-worker',
                cases=[dict(case_id='case',startup_through_this_drain_owned_sampled_peak_mib=100+10*block)],
                polls=[dict(case_id='case',raw_poll_ordinal=0,state_key='crossed-state',state={'CPU_TEST_ONLY':True},
                    prior_observed_owned_peak_mib=100,sampled_owned_mib=100+10*block,positive_increment_mib=10*block)]))
    return result


class MemoryCandidateTests(unittest.TestCase):
    def test_exact_wholeblock_conditional_distribution_and_heldout_underestimate_retained(self):
        tables=training_tables();model=m.fit_tables(tables)
        self.assertEqual(model['states']['crossed-state']['empirical_conditional_max_increment_mib'],30)
        self.assertEqual(model['states']['crossed-state']['equal_block_weights'],{b:1/3 for b in model['training_block_ids']})
        train=m.fit_tables([t for t in tables if t['capture_id']!='training-block-3'])
        report=m.evaluate_table(tables[-1],train)
        self.assertEqual(report['underestimation_count'],1)
        self.assertEqual(report['cases'][0]['signed_error_mib'],-10)
        self.assertEqual(report['empirical_supported_poll_nonunderestimate_fraction'],0)
        self.assertFalse(report['prospective_generated_resource_prediction'])
        self.assertFalse(report['capacity_bound'])

    def test_missing_state_or_missing_wholeblock_never_gets_zero_fallback(self):
        tables=training_tables();model=m.fit_tables(tables[:8]);heldout=copy.deepcopy(tables[-1])
        heldout['polls'][0]['state_key']='unseen'
        report=m.evaluate_table(heldout,model)
        self.assertEqual(report['unknown_polls'],1)
        self.assertIsNone(report['cases'][0]['predicted_owned_sampled_peak_mib'])
        with self.assertRaises(ValueError):m.fit_tables(tables[:-1])
        with self.assertRaises(ValueError):m.evaluate_table(tables[0],model)

    def test_heldout_evaluation_rejects_different_physical_domain_or_schema(self):
        tables=training_tables();model=m.fit_tables(tables[:8]);heldout=copy.deepcopy(tables[-1])
        heldout['physical_execution_identity']='different-priming'
        with self.assertRaisesRegex(ValueError,'physical'):m.evaluate_table(heldout,model)
        heldout=copy.deepcopy(tables[-1]);heldout['schema']='unrelated-table'
        with self.assertRaisesRegex(ValueError,'schemas'):m.evaluate_table(heldout,model)
        wrong_model=copy.deepcopy(model);wrong_model['schema']='unrelated-model'
        with self.assertRaisesRegex(ValueError,'schemas'):m.evaluate_table(tables[-1],wrong_model)

    def test_future_actual_BEGIN_shape_cannot_change_current_candidate(self):
        plan,raw=fixture.cohort_fixture();rows=raw['priming_request_records']+raw['request_records']
        for row in rows:
            for e in row['execution_events']:
                if e.get('event')=='start' and e.get('semantic_phase') in ('prefill','decode'):
                    e.update(input_batch_size=1,input_sequence_length=8 if e['semantic_phase']=='prefill' else 1,forward_index=0 if e['semantic_phase']=='prefill' else 1)
        event=next(e for e in raw['request_records'][0]['execution_events'] if e.get('event')=='start' and e.get('semantic_phase')=='decode')
        cutoff=event['collector_processing_time_ns']-1
        before=m.candidate_at(rows,['text#0','text#1'],cutoff)
        event['input_sequence_length']=99999;event['forward_index']=999
        self.assertEqual(before,m.candidate_at(rows,['text#0','text#1'],cutoff))
        self.assertNotEqual(before,m.candidate_at(rows,['text#0','text#1'],cutoff+1))

    def test_highwater_compresses_order_but_keeps_peak_shape_and_concurrency(self):
        rows=[dict(request_id='a',model='text#0',task='text',pool_id='exact-a'),dict(request_id='b',model='text#1',task='text',pool_id='exact-b')]
        def begin(r,phase,seq,ordinal):return dict(request_id=r,event='start',semantic_phase=phase,segment_id=phase+'_0',input_sequence_length=seq,input_batch_size=1,forward_index=ordinal)
        a=m.CausalState(rows,['text#0','text#1']);b=m.CausalState(rows,['text#0','text#1'])
        events=[begin('a','prefill',16,0),begin('b','prefill',8,0)]
        for e in events:a.process(e)
        for e in reversed(events):b.process(e)
        self.assertEqual(a.descriptor(),b.descriptor())
        self.assertEqual(a.descriptor()['maximum_prior_visible_concurrency'],2)
        a.process(begin('a','decode',1,4))
        high=next(x for x in a.descriptor()['prior_processed_shape_highwater'] if x['owner']=='text#0')
        self.assertEqual(high['max_input_sequence_length'],16);self.assertEqual(high['max_forward_index'],4)

    def test_partial_campaign_never_reads_raw_or_fits(self):
        campaign_path=Path('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json')
        campaign=m.checked(m.receipt(campaign_path))
        with tempfile.TemporaryDirectory() as directory,patch.object(m.intake.campaign_v2,'audit',return_value=campaign),patch.object(m.intake.macros_v2,'extract') as extract,patch.object(m,'fit_tables') as fit:
            with self.assertRaisesRegex(ValueError,'All twelve'):m.run(campaign_path,[],Path(directory)/'out')
        extract.assert_not_called();fit.assert_not_called()

    def test_poll_table_uses_only_prior_completed_peak_and_keeps_all_polls(self):
        campaign=m.checked(m.receipt('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json'))
        plan=m.checked(campaign['attempts'][0]);context=m.checked(plan['context'])
        _,protocol=fixture.cohort_fixture();raw=resource_fixture()
        for poll in raw['board_samples']:poll['query_duration_ns']=0
        raw.update(request_records=protocol['request_records'],priming_request_records=protocol['priming_request_records'])
        before=copy.deepcopy(raw);table=m.table_from_verified(raw,plan,context)
        self.assertEqual(raw,before);self.assertEqual(len(table['polls']),len(raw['board_samples']))
        peak=table['polls'][5]
        self.assertEqual(peak['prior_observed_owned_peak_mib'],5)
        self.assertEqual(peak['positive_increment_mib'],95)
        self.assertEqual(table['polls'][6]['prior_observed_owned_peak_mib'],100)
        raw['board_samples'][9]['owned_used_mib']=99999
        changed=m.table_from_verified(raw,plan,context)
        self.assertEqual(changed['polls'][:9],table['polls'][:9])


if __name__=='__main__':unittest.main()
