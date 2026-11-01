import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from tools.predictor_validation_20261027 import twoFIFO_fresh181_validation_evaluation_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_expected_marginal_score_v1 as marginal
from tools.predictor_validation_20261027 import twoFIFO_forecast_ensemble_v1 as ensemble
from tools.predictor_validation_20261027.stage_response_features_v1 import receipt,checked
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import encode
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode


class IndependentEvaluationTests(unittest.TestCase):
    def fixture(self,root):
        limits=dict(e2e_mare_max=.2,service_mare_max=.2,each_task_p95_absolute_relative_error_max=.2,
            queue_depth_mean_absolute_error_max_requests=1.,candidate_pair_rank_accuracy_min=1.,paired_delta_sign_accuracy_min=1.)
        contract=dict(weights={k:1/6 for k in 'QTLPME'},validation_thresholds=limits)
        contract_ref=m.executor.save(root/'contract.json',contract);attempts=[]
        for i in range(4):
            plan=m.executor.save(root/('plan'+str(i)+'.json'),dict(candidate_id='c'+str(i),public_contract=contract_ref))
            attempts.append(dict(candidate_id='c'+str(i),plan=plan))
        manifest=dict(attempts=attempts);manifest_ref=m.executor.save(root/'manifest.json',manifest)
        return contract,manifest_ref,manifest

    def review(self,root,contract,manifest_ref,manifest,index):
        cid='c'+str(index);value=.5-index*.0001;terms={k:value for k in 'QTLPME'};domain={'candidate':cid}
        draws=[dict(status='GENERATED_WORLD',weight=F(1),forecast=terms)]
        marginals={t:dict(domain=domain,sources=[dict(CPU_TEST_ONLY=True)],distribution=ensemble.distribution(draws,lambda f,t=t:f[t])) for t in 'QTLPME'}
        score=marginal.combine(marginals,candidate_id=cid,candidate_domain=domain,comparison_contract={'same':True},weights=contract['weights'])
        prediction=m.executor.save(root/('pred'+str(index)+'.json'),dict(expected_score=score,evaluation_protocol=m.expected.EVALUATION_PROTOCOL,
            conditional_mean_hypothesis=m.expected.HCPU))
        actual=dict(terms=terms,full_U=-sum(contract['weights'][k]*v for k,v in terms.items()))
        document=dict(schema=m.validation.SCHEMA,manifest=manifest_ref,validation_labels_read=True,validation_labels_used_for_fit=False,
            plan=manifest['attempts'][index]['plan'],status='PASS_CONDITIONAL_NATIVE_VALIDATION_EVIDENCE_AND_DESCRIPTIVE_ERRORS',
            raw=m.executor.save(root/('raw'+str(index)+'.json'),dict(CPU_TEST_ONLY=True)),
            execution=m.executor.save(root/('execution'+str(index)+'.json'),dict(CPU_TEST_ONLY=True)),
            source_authenticated_prediction=dict(report=prediction),actual_six_term_score=dict(score=actual),
            actual_resources=dict(joint_tuple={}),forecast_errors=dict(CPU_TEST_ONLY=True))
        return m.executor.save(root/('review'+str(index)+'.json'),document)

    def test_missing_all_candidates_keeps_four_and_six_unknown_no_zeros(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);contract,manifest_ref,manifest=self.fixture(root)
            with patch.object(m.executor,'audit',return_value=manifest):
                report=m.evaluate(manifest_ref['path'],[],root/'result')
            result=decode(checked(report));self.assertEqual(result['candidate_denominator'],4);self.assertEqual(result['pair_denominator'],6)
            self.assertEqual(result['complete_candidate_coverage'],0);self.assertIsNone(result['all_predeclared_pilot_thresholds_pass'])
            self.assertTrue(all(p['exact_sign_agreement'] is None for p in result['full_six_term_comparison']['pairs']))

    def test_full_review_replay_four_candidates_and_all_nearzero_pair_signs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);contract,manifest_ref,manifest=self.fixture(root)
            refs=[self.review(root,contract,manifest_ref,manifest,i) for i in range(4)];calls=[]
            def recheck(raw_path,out,**kwargs):
                index=int(Path(raw_path).stem[3:]);calls.append(index);out.mkdir(parents=True)
                return m.executor.save(out/'report.json',checked(refs[index]))
            with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.validation,'review',side_effect=recheck),\
                    patch.object(m,'threshold_arithmetic',return_value=dict(all_local_pilot_thresholds_pass=True)):
                result=decode(checked(m.evaluate(manifest_ref['path'],[r['path'] for r in refs],root/'result')))
            self.assertEqual(calls,[0,1,2,3]);self.assertEqual(result['known_pair_coverage'],1)
            self.assertEqual(result['exact_sign_accuracy'],1);self.assertTrue(result['all_predeclared_pilot_thresholds_pass'])
            self.assertTrue(all(p['predicted_engineering_indifference'] and p['predicted_sign']==1 for p in result['full_six_term_comparison']['pairs']))
            self.assertFalse(result['original_stability_evaluated']);self.assertFalse(result['validation_labels_used_for_fit'])

    def test_changed_restored_report_during_recheck_cannot_be_saved_as_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);contract,manifest_ref,manifest=self.fixture(root);ref=self.review(root,contract,manifest_ref,manifest,0)
            def recheck(raw_path,out,**kwargs):
                path=Path(ref['path']);before=path.read_bytes();path.write_bytes(before+b' ');path.write_bytes(before)
                out.mkdir(parents=True);return m.executor.save(out/'report.json',checked(ref))
            with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.validation,'review',side_effect=recheck),\
                    patch.object(m,'threshold_arithmetic',return_value=dict(all_local_pilot_thresholds_pass=True)):
                with self.assertRaises(ValueError):m.evaluate(manifest_ref['path'],[ref['path']],root/'result')
            self.assertFalse((root/'result'/'report.json').exists())

    def test_training_review_role_and_recomputed_outcome_disagreement_rejected(self):
        for mode in ('training','recheck'):
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);contract,manifest_ref,manifest=self.fixture(root);ref=self.review(root,contract,manifest_ref,manifest,0)
                document=checked(ref)
                if mode=='training':document['validation_labels_used_for_fit']=True;Path(ref['path']).write_text(json.dumps(document));ref=receipt(ref['path'])
                def recheck(raw_path,out,**kwargs):
                    out.mkdir(parents=True);return m.executor.save(out/'report.json',dict(document,untrusted_extra=True))
                with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.validation,'review',side_effect=recheck):
                    with self.assertRaises(ValueError):m.evaluate(manifest_ref['path'],[ref['path']],root/'result')

    def test_definite_P95_failure_dominates_missing_service_and_other_task_P95(self):
        with tempfile.TemporaryDirectory() as temp:
            contract,_,_=self.fixture(Path(temp))
        errors=dict(point_queue_grid_errors=dict(arrived_RESULT_unfinished=dict(mean_absolute_error_upper=0.)),
            per_task_P95=dict(text=dict(absolute_relative_error=.3),image=dict(absolute_relative_error=None)),
            point_reader_MARE_request_coverage=1.,point_START_to_READY_MARE_request_coverage=180/181,
            offered_requests=181,point_reader_MARE=0.,point_START_to_READY_MARE=0.)
        result=m.threshold_arithmetic(errors,contract)
        self.assertIsNone(result['checks']['service']);self.assertFalse(result['checks']['each_task_P95'])
        self.assertFalse(result['all_local_pilot_thresholds_pass'])

    def test_known_candidate_or_QME_failure_survives_three_missing_candidates(self):
        for failing in ('candidate','QME'):
            with self.subTest(failing=failing),tempfile.TemporaryDirectory() as temp:
                root=Path(temp);contract,manifest_ref,manifest=self.fixture(root);ref=self.review(root,contract,manifest_ref,manifest,0)
                if failing=='QME':
                    doc=checked(ref);actual=doc['actual_six_term_score']['score'];actual['terms']['Q']=0.
                    actual['full_U']=-sum(contract['weights'][k]*v for k,v in actual['terms'].items())
                    Path(ref['path']).write_text(json.dumps(doc));ref=receipt(ref['path'])
                def recheck(raw_path,out,**kwargs):
                    out.mkdir(parents=True);return m.executor.save(out/'report.json',checked(ref))
                with patch.object(m.executor,'audit',return_value=manifest),patch.object(m.validation,'review',side_effect=recheck),\
                        patch.object(m,'threshold_arithmetic',return_value=dict(all_local_pilot_thresholds_pass=failing!='candidate')):
                    result=decode(checked(m.evaluate(manifest_ref['path'],[ref['path']],root/'result')))
                self.assertEqual(result['candidate_denominator'],4);self.assertEqual(result['pair_denominator'],6)
                self.assertEqual(result['complete_candidate_coverage'],.25);self.assertEqual(result['known_pair_coverage'],0.)
                self.assertFalse(result['all_predeclared_pilot_thresholds_pass'])
                self.assertEqual(sum(r['all_local_pilot_thresholds_pass'] is None for r in result['original_local_thresholds']),3)
                if failing=='QME':self.assertFalse(result['QME_engineering_thresholds_pass'])

    def test_unknown_pair_results_do_not_hide_unrecoverable_sign_accuracy_failure(self):
        self.assertFalse(m.accuracy_threshold(0,1,6,1.))
        self.assertIsNone(m.accuracy_threshold(1,1,6,1.))
        self.assertTrue(m.tri_state_all([True,True]))
        self.assertIsNone(m.tri_state_all([True,None]))
        self.assertFalse(m.tri_state_all([None,False,True]))

if __name__=='__main__':unittest.main()
