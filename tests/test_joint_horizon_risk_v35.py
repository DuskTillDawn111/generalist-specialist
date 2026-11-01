import copy
import unittest

from tools.predictor_validation_20261027 import horizon_recovery_risk_v31 as risk
from tools.predictor_validation_20261027 import observed_horizon_risk_sidecar_v32 as sidecar
from tools.predictor_validation_20261027.joint_horizon_risk_v35 import join_predictions,pava_nonincreasing


def fixture(values):
    return [dict(schema=sidecar.SCHEMA,id='held:10',run_id='held',decision_time_ns=10,
        original_execution_context_sha256='original-context',visible_state_sha256='observed-host-state',
        visible_waiting_count=2,risk_domain_sha256='explicit-common-domain',
        horizon_ns=h,model_sha256='frozen-model-'+str(h),risk_score=v,
        status='UNCALIBRATED_TRAINING_DIAGNOSTIC') for h,v in zip(risk.HORIZONS_NS,values)]


class JointHorizonRiskTests(unittest.TestCase):
    def test_extreme_complete_reversal_pools_all_horizons(self):
        self.assertEqual(pava_nonincreasing([0,.5,1]),[.5,.5,.5])
        for value in pava_nonincreasing([0,0,1]):self.assertAlmostEqual(value,1/3)
        row=join_predictions(fixture([0,.5,1]))
        self.assertEqual(row['raw_scores'],[0,.5,1]);self.assertEqual(row['projected_scores'],[.5,.5,.5])
        self.assertEqual(row['raw_monotonicity_violation'],1)
        self.assertEqual(row['projection_l2_squared'],.5)

    def test_adjacent_pooling_and_already_monotone_scores(self):
        self.assertEqual(pava_nonincreasing([0,1,.2]),[.5,.5,.2])
        self.assertEqual(pava_nonincreasing([1,0,1]),[1,.5,.5])
        self.assertEqual(pava_nonincreasing([1,.5,0]),[1,.5,0])
        self.assertEqual(pava_nonincreasing([.7,.7,.7]),[.7,.7,.7])

    def test_UNKNOWN_horizon_is_joint_UNKNOWN_without_zero_imputation(self):
        rows=fixture([.9,.8,.7]);rows[1].update(status='UNKNOWN_OUTSIDE_TRAINING_SUPPORT',risk_score=None)
        row=join_predictions(rows)
        self.assertEqual(row['status'],'JOINT_UNKNOWN');self.assertIsNone(row['projected_scores'])
        self.assertEqual(row['raw_scores'],[.9,None,.7]);self.assertIsNone(row['raw_monotonicity_violation'])

    def test_unknown_or_different_domain_cannot_form_joint_score(self):
        rows=fixture([.9,.8,.7]);rows[1]['risk_domain_sha256']='unbridged-domain'
        self.assertEqual(join_predictions(rows)['status'],'JOINT_UNKNOWN')
        rows=fixture([.9,.8,.7])
        for row in rows:row.update(status='UNKNOWN_EXECUTION_CONTEXT_OUTSIDE_EXPLICIT_RISK_DOMAIN',risk_score=None)
        self.assertEqual(join_predictions(rows)['status'],'JOINT_UNKNOWN')

    def test_different_cutoffs_or_original_contexts_cannot_be_pooled(self):
        for key,value in [('decision_time_ns',11),('original_execution_context_sha256','different-original-context'),('visible_state_sha256','different-state')]:
            rows=fixture([.9,.8,.7]);rows[2][key]=value
            result=join_predictions(rows)
            self.assertEqual(result['status'],'JOINT_UNKNOWN');self.assertIsNone(result['projected_scores'])

    def test_raw_outputs_are_preserved_without_mutation_and_policy_does_not_change_utility(self):
        rows=fixture([.3,.7,.2]);original=copy.deepcopy(rows)
        row=join_predictions(rows)
        self.assertEqual(rows,original);self.assertEqual(row['raw_scores'],[.3,.7,.2])
        self.assertFalse(row['calibrated_probability']);self.assertFalse(row['rho_changed']);self.assertFalse(row['six_term_utility_changed'])

    def test_empty_population_has_no_fabricated_joint_probability(self):
        rows=fixture([0,0,0])
        for row in rows:row.update(visible_waiting_count=0,status='NOT_AT_RISK_CURRENT_BACKLOG_EMPTY',risk_score=None)
        result=join_predictions(rows)
        self.assertEqual(result['status'],'JOINT_NOT_AT_RISK_CURRENT_BACKLOG_EMPTY');self.assertIsNone(result['projected_scores'])

    def test_incomplete_horizon_grid_and_nonfinite_scores_rejected(self):
        with self.assertRaises(ValueError):join_predictions(fixture([1,.5,0])[:2])
        for value in [float('nan'),float('inf'),-.1,1.1,True]:
            with self.assertRaises(ValueError):pava_nonincreasing([1,.5,value])


if __name__=='__main__':unittest.main()
