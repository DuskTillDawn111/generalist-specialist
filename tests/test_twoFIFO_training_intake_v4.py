import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import twoFIFO_training_intake_v4 as m
from tests.test_twoFIFO_owned_training_review_v3 import cohort_fixture as fixture
from tools.predictor_validation_20261027.twoFIFO_composable_macros_v3 import request_rows

CAMPAIGN=Path('artifacts/predictor-twoFIFO-owned-training-design-v3-20261028-cpu01/campaign.json')


def documents(plans):
    result=[]
    for ordinal,plan in enumerate(plans):
        requests=[]
        for c in plan['cases']:
            for a in c['arrivals']:
                duration=100+ordinal*7+len(requests);left=1000;right=left+duration
                peer_owner='text#1' if a['owner']!='text#1' else 'text#0'
                peers=[] if len(requests)%2==0 else [dict(owner=peer_owner,request_id='actual-peer',phase='prefill',phase_id='processed-BEGIN',available_ns=999)]
                macro=dict(owner=a['owner'],task=a['task'],pool_id=a['pool_id'],request_id=a['request_id'],
                    macro_id='entry',semantic_phase='entry',kind='entry',left_ns=left,right_ns=right,
                    response_wall_ns=duration,response_domain=m.DOMAIN,forward_shape=None,
                    additional_stream_elapsed_or_sync_wait_may_be_added=False,true_SM_service_identified=False,
                    capture_sha256='raw-'+str(ordinal),attempt_id=plan['attempt_id'],training_replication_unit=plan['capture_id'],
                    causal_covariate_path=[dict(left_ns=left,right_ns=right,duration_ns=duration,
                        each_state_uses_only_its_processed_prefix=True,state=dict(cutoff_processing_ns=left,
                        peer_visible_macro_owner_count=len(peers),peer_same_visible_macro_phase_owner_count=0,peer_macros=peers))])
                requests.append(dict(owner=a['owner'],request_id=a['request_id'],capture_id=plan['capture_id'],
                    attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],worker_start_to_ready_ns=duration,macros=[macro]))
        result.append(dict(schema=m.MACRO_SCHEMA,mode='natural',status='COMPLETE_ATTEMPT_DISJOINT_TRAINING_MACROS_NOT_FITTED_PROVIDER',
            validation_labels_read=False,whole_attempt_history_used_for_visible_prefix=True,case_history_reset=False,
            capture_id=plan['capture_id'],attempt_id=plan['attempt_id'],execution_id=plan['execution_id'],requests=requests,
            measured_requests=len(requests),priming_rows_excluded_as_labels=18,macros=len(requests),sources=[]))
    return result


class IntakeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign=m.checked(m.receipt(CAMPAIGN));cls.plans=[m.checked(r) for r in cls.campaign['attempts']]
        cls.docs=documents(cls.plans)

    def make_reviews(self,directory,count=12,failed=None):
        refs=[]
        for i,plan_ref in enumerate(self.campaign['attempts'][:count]):
            review=dict(schema=m.REVIEW_SCHEMA,status=m.COMPLETE,training_feature_extraction_eligible=True,
                issues=[],validation_labels_read=False,inputs={'plan.json':plan_ref})
            if i==failed:review.update(status='NOT_ELIGIBLE',training_feature_extraction_eligible=False,issues=['incomplete'])
            path=Path(directory)/('review-'+str(i)+'.json');path.write_text(json.dumps(review));refs.append(str(path))
        return refs

    def test_partial_and_failed_intake_never_extract_response_documents(self):
        for count,failed in ((0,None),(11,None),(12,4)):
            with self.subTest(count=count,failed=failed),tempfile.TemporaryDirectory() as directory:
                refs=self.make_reviews(directory,count,failed)
                with patch.object(m.campaign_v2,'audit',return_value=self.campaign),patch.object(m.macros_v2,'extract') as extract:
                    report=m.checked(m.run(CAMPAIGN,refs,Path(directory)/'report'))
                extract.assert_not_called();self.assertTrue(report['status'].startswith('WAITING'))
                self.assertEqual(report['intake']['expected_measured_offered'],546)
                self.assertEqual(report['intake']['expected_startups'],12)
                self.assertFalse(report['numerical_fit_run']);self.assertEqual(report['expected_plan_receipts'],self.campaign['attempts'])

    def test_complete_campaign_only_has_three_whole_block_folds(self):
        with tempfile.TemporaryDirectory() as directory:
            refs=self.make_reviews(directory)
            with patch.object(m.campaign_v2,'audit',return_value=self.campaign),patch.object(m.macros_v2,'extract',side_effect=self.docs) as extract:
                result=m.checked(m.run(CAMPAIGN,refs,Path(directory)/'report'))
            self.assertEqual(extract.call_count,12)
        self.assertEqual(result['denominator']['measured_requests'],546)
        self.assertEqual(result['training_replication_units'],3);self.assertEqual(result['planned_startup_attempts'],12)
        self.assertEqual(result['completed_native_startup_attempts'],12)
        self.assertEqual(len(result['designs']),3);self.assertFalse(result['numerical_fit_run'])
        for design in result['designs']:
            self.assertEqual(len(design['whole_block_leaveout']),3)
            for fold in design['whole_block_leaveout']:
                self.assertEqual(len(fold['training_attempt_ids']),8);self.assertEqual(len(fold['heldout_attempt_ids']),4)
                self.assertFalse(any(a.startswith(fold['heldout_block_id']+'-') for a in fold['training_attempt_ids']))
                self.assertFalse(fold['training_diagnostic']['fitted_root_numerical_qualification'])

    def test_duplicate_missing_or_relabelled_attempts_cannot_create_replication(self):
        for mutate in (lambda ds:ds.pop(),lambda ds:ds.__setitem__(0,copy.deepcopy(ds[1])),
            lambda ds:ds[0]['requests'].pop(),lambda ds:ds[0]['requests'][0]['macros'][0].update(training_replication_unit='raw-sha-as-block')):
            ds=copy.deepcopy(self.docs);mutate(ds)
            with self.assertRaises(ValueError):m.complete_documents(self.plans,ds)

    def test_new_schema_adapter_retains_original_entry_forward_and_READY_tail(self):
        plan,raw=fixture();requests,events=request_rows(raw,plan,'raw-provenance')
        doc=dict(schema=m.MACRO_SCHEMA,validation_labels_read=False,capture_id='whole-block',attempt_id='shard',requests=requests)
        rows=m.prepared_rows([doc],m.phase.DESIGNS['global_phase_aware'])
        self.assertEqual(len(rows),12);self.assertEqual({r['capture'] for r in rows},{'whole-block'})
        self.assertEqual({r['raw_sha256'] for r in rows},{'raw-provenance'})
        self.assertEqual({r['phase'] for r in rows},{'entry','prefill','decode','ready_tail'})
        bad=copy.deepcopy(doc);bad['schema']='gsdeploy.composable_response_macro_training.v1'
        with self.assertRaises(ValueError):m.prepared_rows([bad],m.phase.DESIGNS['global_phase_aware'])

    def test_missing_OCR_entry_exposure_stays_unknown_despite_planned_offsets(self):
        support=m.observed_support(self.docs,self.plans)
        self.assertEqual(len(support['OCR_exact_input_by_whole_block']),6)
        ds=copy.deepcopy(self.docs)
        for doc in ds:
            for request in doc['requests']:
                for macro in request['macros']:
                    if macro['owner']=='ocr#0':
                        for piece in macro['causal_covariate_path']:
                            piece['state'].update(peer_visible_macro_owner_count=0,peer_macros=[])
        support=m.observed_support(ds,self.plans)
        self.assertFalse(support['each_target_OCR_input_each_block_n0_and_npositive'])
        self.assertTrue(all(r['status'].startswith('UNKNOWN') for r in support['OCR_exact_input_by_whole_block']))

    def test_rank_failure_and_zero_score_point_do_not_qualify_a_fit(self):
        features=m.phase.DESIGNS['own_family_phase_count_and_same'];rows=m.prepared_rows(self.docs,features)
        result=m.zero_point_diagnostic(rows,features)
        self.assertFalse(result['joint_full_column_rank']);self.assertFalse(result['fitted_root_numerical_qualification'])
        self.assertTrue(result['no_columns_deleted']);self.assertEqual(result['feature_dimension'],22)
        self.assertIn('same@ocr:ocr_inference',result['exact_piecewise_zero_columns'])
        self.assertFalse(result['optimizer_run']);self.assertIn('not_fitted_root',result['evaluation_point'])

    def test_future_BEGIN_or_duplicate_owner_is_not_observed_support(self):
        for which in ('future','duplicate'):
            doc=copy.deepcopy(self.docs[0])
            piece=next(macro['causal_covariate_path'][0] for r in doc['requests'] for macro in r['macros']
                if macro['causal_covariate_path'][0]['state']['peer_macros'])
            state=piece['state']
            if which=='future':state['peer_macros'][0]['available_ns']=piece['left_ns']+1
            else:state['peer_macros'].append(copy.deepcopy(state['peer_macros'][0]));state[m.phase.COUNT]=2
            with self.assertRaises(ValueError):m.prepared_rows([doc],m.phase.DESIGNS['global_phase_aware'])

    def test_full_design_support_cannot_hide_whole_block_leaveout_rank_loss(self):
        rows=[]
        for block in (1,2,3):
            for ordinal in (0,1):
                count=ordinal if block==3 else 0
                rows.append(dict(group=('same-public-nuisance',),capture='training-block-'+str(block),
                    attempt_id='training-block-'+str(block)+'-shard-1',
                    durations=np.asarray([1.+block+ordinal]),matrix=np.asarray([[count,0.]])))
        features=m.phase.DESIGNS['global_phase_aware']
        full=m.zero_point_diagnostic(rows,features)
        heldout=m.zero_point_diagnostic([r for r in rows if r['capture']!='training-block-3'],features)
        self.assertEqual(full['nuisance_adjusted_derivative_rank'],1)
        self.assertEqual(heldout['nuisance_adjusted_derivative_rank'],0)
        self.assertIn(m.phase.COUNT,heldout['exact_piecewise_zero_columns'])
        self.assertFalse(heldout['fitted_root_numerical_qualification'])


if __name__=='__main__':unittest.main()
