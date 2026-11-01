import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import twoFIFO_phase12_fit_v2 as m
from tests.test_twoFIFO_training_intake_v4 import documents,CAMPAIGN


def synthetic_rows():
    """Crossed independent axes with known log-response slopes and noise."""
    rng=np.random.default_rng(12);beta=np.linspace(-.12,.16,12);rows=[]
    for block in (1,2,3):
        for ordinal in range(120):
            x=rng.integers(0,4,12).astype(float);duration=np.exp(1.3+x@beta+rng.normal(0,.08))
            rows.append(dict(group=('known-public-input',),capture='training-block-'+str(block),
                attempt_id='training-block-'+str(block)+'-shard-'+str(ordinal%4+1),raw_sha256='raw-'+str(block),
                request_id=str(block)+':'+str(ordinal),macro_id='macro',owner='text#0',phase='prefill',kind='language_forward_macro',
                response_ms=duration,durations=np.asarray([duration]),matrix=x.reshape(1,12),
                ordered_durations=np.asarray([duration]),ordered_matrix=x.reshape(1,12)))
    return rows,beta


class PhaseFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.campaign=m.checked(m.receipt(CAMPAIGN));cls.plans=[m.checked(r) for r in cls.campaign['attempts']]
        cls.docs=documents(cls.plans)

    def test_real_numeric_root_recovers_crossed_phase12_slopes(self):
        rows,beta=synthetic_rows();fit,params=m.fitted_rows(rows)
        self.assertTrue(fit['numerical_qualification'])
        self.assertEqual(fit['nuisance_adjusted_derivative_rank'],12)
        self.assertLess(np.max(np.abs(params[0]-beta)),.02)
        self.assertEqual(len(fit['gradient_root_attempts']),3)
        self.assertTrue(all(r['stationary_strict_local_minimum'] for r in fit['gradient_root_attempts']))

    def test_exact_missing_column_stays_unqualified_no_column_deletion(self):
        rows,beta=synthetic_rows()
        for row in rows:row['matrix'][:,-1]=0;row['ordered_matrix'][:,-1]=0
        fit,params=m.fitted_rows(rows)
        self.assertFalse(fit['numerical_qualification']);self.assertEqual(len(fit['features']),12)
        self.assertLess(fit['nuisance_adjusted_derivative_rank'],12)

    def test_partial_campaign_never_runs_numeric_fit(self):
        bundle=dict(intake=dict(complete=False),documents=None,sources=[])
        with tempfile.TemporaryDirectory() as directory,patch.object(m,'load_complete',return_value=bundle),patch.object(m.numeric,'fit') as fit:
            report=m.checked(m.run('campaign',[],Path(directory)/'out'))
        fit.assert_not_called();self.assertFalse(report['numerical_fit_run']);self.assertIsNone(report['fit'])

    def test_v2_contamination_hold_precedes_reviews_raw_extraction_and_fit(self):
        from tools.predictor_validation_20261027 import twoFIFO_joint_request_law_v3 as law
        with patch.object(m.intake.campaign_v2,'audit') as audit,patch.object(m.intake.macros_v2,'extract') as extract,patch.object(m.numeric,'fit') as fit:
            old_campaign=Path('artifacts/predictor-twoFIFO-owned-training-design-v2-20261028-cpu01/campaign.json')
            bundle=m.load_complete(old_campaign,['missing-review-must-not-be-read'])
            self.assertEqual(bundle['status'],m.CONTAMINATION_HOLD)
            with tempfile.TemporaryDirectory() as directory:
                report=m.checked(m.run(old_campaign,['missing-review'],Path(directory)/'out'))
                with self.assertRaisesRegex(ValueError,m.CONTAMINATION_HOLD):
                    law.run(old_campaign,['missing-review'],'missing-fit',Path(directory)/'law')
            self.assertEqual(report['status'],m.CONTAMINATION_HOLD)
            self.assertFalse(report['numerical_fit_run']);self.assertIsNone(report['fit'])
        audit.assert_not_called();extract.assert_not_called();fit.assert_not_called()

    def test_whole_block_fold_scores_equal_weight_and_unknowns_preserved(self):
        rows,beta=synthetic_rows()
        # Test the orchestrator with meaningful real math but replace native
        # macro->row extraction with this crossed CPU-only response design.
        with patch.object(m.intake,'prepared_rows',return_value=rows),patch.object(m.intake,'observed_support',return_value=dict(each_target_OCR_input_each_block_n0_and_npositive=True)):
            result=m.fit_new_blocks(self.docs,self.plans)
        self.assertTrue(result['conditional_response_fit_qualified'])
        self.assertEqual(len(result['whole_block_leaveout']),3)
        self.assertAlmostEqual(result['equal_block_mean_conditional_nll'],sum(x['mean_conditional_nll'] for x in result['block_scores'])/3)
        for fold in result['whole_block_leaveout']:
            self.assertEqual(len(fold['training_attempt_ids']),8);self.assertEqual(len(fold['heldout_attempt_ids']),4)
            self.assertNotIn(fold['heldout_block_id'],fold['training_fit']['whole_blocks'])
            self.assertTrue(all(r['block_id']==fold['heldout_block_id'] and 'raw_sha256' in r for r in fold['evaluation']['records']))
        # A public nuisance group found only in heldout block cannot disappear
        # from its denominator or be repaired by an unknown fallback.
        altered=copy.deepcopy(rows)
        for row in altered:
            if row['capture']=='training-block-3':row['group']=('heldout-only-public-shape',)
        with patch.object(m.intake,'prepared_rows',return_value=altered),patch.object(m.intake,'observed_support',return_value=dict(each_target_OCR_input_each_block_n0_and_npositive=True)):
            result=m.fit_new_blocks(self.docs,self.plans)
        fold=result['whole_block_leaveout'][2]
        self.assertEqual(fold['evaluation']['summary']['unknown_rows'],120)
        self.assertEqual(len(fold['evaluation']['records']),120)
        self.assertFalse(result['response_law_compilation_eligible']);self.assertIsNone(result['equal_block_mean_conditional_nll'])


if __name__=='__main__':unittest.main()
