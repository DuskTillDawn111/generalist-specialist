"""Independent CPU integration counterexamples; native intake alone is mocked."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_fold_resource_rescore_v1 import setup_world,execute
from tests.test_twoFIFO_resource_fold_provider_v1 import fixture,model_document
from tests import test_twoFIFO_generated_wholeblock_evaluation_v1 as archived
from tools.predictor_validation_20261027 import twoFIFO_fold_resource_rescore_v1 as m


def completed_inputs(folder,mutation=None):
    design_ref,design=m.generated.audit_design(Path(
        'artifacts/predictor-twoFIFO-generated-wholeblock-design-v1-20261028-cpu01/design.json'))
    campaign=m.checked(design['campaign']);plans=[m.checked(ref) for ref in campaign['attempts']]
    plan=next(p for p in plans if p['capture_id']=='training-block-3');case=plan['cases'][0]
    score=m.checked(design['scoring_plan']);quality=m.checked(design['quality'])
    _,_,tables=fixture();model,sources=model_document(tables,folder)
    completed_sources=[design['campaign']]+copy.deepcopy(model['training_reviews'])
    if mutation=='campaign':
        alternate=copy.deepcopy(campaign);alternate['CPU_TEST_ONLY_other_campaign']=True
        alternate_ref=m.save(folder/'other-campaign.json',alternate)
        model['campaign']=alternate_ref;model['sources']=sources=[alternate_ref]
    elif mutation=='reviews':
        replacement=m.save(folder/'other-reviewed-native-run.json',dict(CPU_TEST_ONLY=True,other_run=True))
        model['training_reviews'][0]=replacement
    model_path=folder/'model.json';model_path.write_text(json.dumps(model))
    compact=m.generated.compact_prediction(m.generated.forecast_case(case,1000,None,None,score,quality,
        donor_blocks=['training-block-1','training-block-2'],unqualified_reason='CPU_TEST_ONLY_UNKNOWN'))
    prediction=m.save(folder/'prediction.json',compact)
    fold=dict(heldout_block_id='training-block-3',training_block_ids=['training-block-1','training-block-2'],
        donor_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']!='training-block-3'],
        heldout_attempt_ids=[p['attempt_id'] for p in plans if p['capture_id']=='training-block-3'],folded_law=None,
        cases=[dict(attempt_id=plan['attempt_id'],case_id=case['case_id'],prediction=prediction)])
    report=m.save(folder/'completed.json',dict(schema=m.generated.SCHEMA,numerical_fit_run=True,
        status='COMPLETE_NATIVE_TRAINING_FIT_UNQUALIFIED_ALL_GENERATED_WORLDS_UNKNOWN',validation_labels_read=False,
        sources=completed_sources,design=design_ref,whole_block_leaveout=[fold]))
    return report,model_path,plan,case,tables,sources


class IndependentFoldResourceRescoreTests(unittest.TestCase):
    def test_completed_case_rejects_other_campaign_with_same_ids_and_physical_domain(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report,model,plan,case,tables,sources=completed_inputs(folder,'campaign')
            with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)) as load:
                with self.assertRaisesRegex(ValueError,'campaign'):
                    m.run_completed_case(report['path'],model,heldout_block_id='training-block-3',
                        attempt_id=plan['attempt_id'],case_id=case['case_id'],output=folder/'out')
            self.assertFalse((folder/'out').exists());load.assert_not_called()

    def test_completed_case_rejects_other_review_run_with_same_campaign_and_table_values(self):
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);report,model,plan,case,tables,sources=completed_inputs(folder,'reviews')
            with patch.object(m.folds.resource,'load_complete',return_value=(tables,sources)) as load:
                with self.assertRaisesRegex(ValueError,'review'):
                    m.run_completed_case(report['path'],model,heldout_block_id='training-block-3',
                        attempt_id=plan['attempt_id'],case_id=case['case_id'],output=folder/'out')
            self.assertFalse((folder/'out').exists());load.assert_not_called()

    def test_unknown_draw_cannot_name_heldout_resource_block(self):
        data=setup_world(unknown=True);bad=copy.deepcopy(data[5]);bad['draws'][1]['block_id']='training-block-3'
        with tempfile.TemporaryDirectory() as directory,self.assertRaisesRegex(ValueError,'donor block'):
            execute(directory,data,document=bad)

    def test_explicit_prefix_peak_dominates_once_and_missing_startup_energy_stays_unknown(self):
        data=list(setup_world());data[-1]['prior_owned_sampled_peak_mib']=8000.
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,data)
        scored=out['ensemble']['draws'][0]['forecast']['scoring']
        self.assertEqual(scored['resource_prediction']['premeasurement_owned_sampled_peak_mib'],8000.)
        self.assertEqual(scored['original_score']['terms']['M'],8000*1.2/1000)
        missing=copy.deepcopy(data[2]);missing[0]['startup_joules']=None
        with tempfile.TemporaryDirectory() as directory:out=execute(directory,data,tables=missing)
        score=out['ensemble']['draws'][0]['forecast']['scoring']['original_score']
        self.assertIsNone(score['terms']['E']);self.assertIsNone(score['full_U'])
        self.assertEqual(score['terms']['M'],8000*1.2/1000)

    def test_drain_memory_peak_is_kept_and_drain_exposure_does_not_enter_E(self):
        data=setup_world();board=data[5]['draws'][0]['forecast']['scoring']['resource_timeline']['slices']
        tables=copy.deepcopy(data[2]);changed=False
        for table in tables:
            for row,interval in zip(table['slices'],board):
                if m.clock(interval['measurement_overlap_ns'])==0:
                    row['sampled_owned_peak_mib']=99999.;changed=True
        self.assertTrue(changed)
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            original=execute(a,data);modified=execute(b,data,tables=tables)
        left=original['ensemble']['draws'][0]['forecast']['scoring']['original_score']
        right=modified['ensemble']['draws'][0]['forecast']['scoring']['original_score']
        self.assertEqual(left['terms']['E'],right['terms']['E'])
        predicted=modified['ensemble']['draws'][0]['forecast']['scoring']['resource_prediction']
        self.assertGreater(m.clock(predicted['total_observation_duration_ns']),m.clock(predicted['total_measurement_duration_ns']))
        self.assertEqual(predicted['board_energy_joules'],6e-9)
        self.assertEqual(right['terms']['M'],99999*1.2/1000)

    def test_compact_replay_rejects_summary_and_full_denominator_poison(self):
        law=archived.law();score=archived.scoring();quality=archived.QUALITY
        forecast=m.generated.forecast_case(archived.public_case(),1000,archived.past(),law,score,quality)
        compact=m.encode(m.generated.compact_prediction(forecast))
        for mutation in ('summary','risk','drain'):
            bad=copy.deepcopy(compact)
            if mutation=='summary':bad['ensemble']['draws'][0]['forecast_summary']['original_score']['full_U']=999.
            elif mutation=='risk':bad['ensemble']['fixed_drain_risk'][0]['fixed_drain_incomplete_probability_bounds']=[m.encode(F(1)),m.encode(F(1))]
            else:bad['ensemble']['original_drain_end_ns']=m.encode(m.clock(bad['ensemble']['original_drain_end_ns'])+1)
            with self.subTest(mutation=mutation),self.assertRaisesRegex(ValueError,'archived hash|ensemble contract'):
                m.restore_full_ensemble(bad,law,score,quality,donor_blocks=law['blocks'])

    def test_training_review_byte_drift_rejected_before_native_loader(self):
        data=setup_world()
        with tempfile.TemporaryDirectory() as directory:
            folder=Path(directory);model,sources=model_document(data[2],folder)
            (Path(model['training_reviews'][0]['path'])).write_text('{}')
            path=folder/'model.json';path.write_text(json.dumps(model))
            with patch.object(m.folds.resource,'load_complete') as load,self.assertRaisesRegex(ValueError,'review changed'):
                m.rescore_ensemble(data[5],data[3],path,data[0],data[1],heldout_block_id='training-block-3',
                    scoring_plan=data[4]['scoring_plan'],quality=data[4]['quality'],window_start_ns=data[4]['window_start_ns'],
                    window_end_ns=data[4]['window_end_ns'],drain_end_ns=data[4]['drain_end_ns'])
            load.assert_not_called()


if __name__=='__main__':unittest.main()
