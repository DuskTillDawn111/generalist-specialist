import math

from tests import test_cumulative_save_hazard_v13 as previous
from tools.predictor_validation_20261027.cumulative_save_hazard_v14 import CumulativeSaveHazardController
from tools.predictor_validation_20261027.native_collect import receipt


class ConditionalEntryHazardTests(previous.CumulativeHazardTests):
    """Run all twelve prior contracts plus explicit left-truncated entry cases."""
    def controller(self,**kwargs):
        return CumulativeSaveHazardController(receipt(self.path),allow_unvalidated_approximation=True,**kwargs)

    def test_age_88ms_conditions_remaining_budget_without_moving_restore(self):
        self.model['coefficients'][self.model['columns'].index('age_bin:15')]=math.log(4.)
        self.rewrite();controller=self.controller()
        observation=88_000_000
        result=controller.start_restore('holder',0,previous.A,'old-restore',observation_ns=observation)
        expected=observation+math.ceil(math.log(2.)/8*1e6)
        self.assertEqual(result['absolute_save_candidate_ns'],expected)
        self.assertGreater(result['absolute_save_candidate_ns'],observation)
        self.assertEqual(result['restore_age_ns'],88_000_000)
        self.assertEqual(result['origin_restore_ns'],0)
        self.assertEqual(result['observation_ns'],observation)
        self.assertEqual(result['consumed_hazard'],0.)
        self.assertEqual(result['state']['preentry_interval_ns'],[0,observation])
        self.assertIsNone(result['state']['preentry_hazard'])
        self.assertFalse(result['preentry_hazard_integrated'])
        young=controller.start_restore('holder',observation,previous.A,'new-restore')
        self.assertNotEqual(young['absolute_save_candidate_ns'],expected)

    def test_age_1ms_uses_attained_bin_and_updates_do_not_recondition(self):
        self.model['coefficients'][self.model['columns'].index('age_bin:10')]=math.log(.25)
        self.rewrite();controller=self.controller()
        observation=1_000_000
        first=controller.start_restore('holder',0,previous.A,'old',observation_ns=observation)
        updated=controller.update(first['state'],observation+10_000,previous.A)
        self.assertAlmostEqual(updated['consumed_hazard'],.005)
        self.assertEqual(updated['hazard_budget'],first['hazard_budget'])
        self.assertEqual(updated['absolute_save_candidate_ns'],first['absolute_save_candidate_ns'])
        self.assertEqual(updated['observation_ns'],observation)
        self.assertEqual(updated['restore_age_ns'],observation+10_000)

    def test_explicit_entry_at_restore_matches_default(self):
        controller=self.controller()
        default=controller.start_restore('holder',100,previous.A,'r')
        explicit=controller.start_restore('holder',100,previous.A,'r',observation_ns=100)
        self.assertEqual(default,explicit)

    def test_conditional_entry_empty_pause_preserves_origin_and_assumption(self):
        controller=self.controller(empty_frontier_policy='NO_QUERY_PAUSE')
        observation=88_000_000
        first=controller.start_restore('holder',0,[], 'r',observation_ns=observation)
        self.assertIsNone(first['absolute_save_candidate_ns'])
        resumed=controller.update(first['state'],observation+100_000,previous.A)
        self.assertEqual(resumed['consumed_hazard'],0.)
        self.assertEqual(resumed['observation_ns'],observation)
        self.assertEqual(resumed['origin_restore_ns'],0)
        self.assertEqual(resumed['restore_age_ns'],88_100_000)
        self.assertIn('NO_QUERY_PAUSE_ASSUMPTION',resumed['uncertainty_reasons'])
        self.assertEqual(resumed['absolute_save_candidate_ns'],88_100_000+math.ceil(math.log(2.)/2*1e6))

    def test_observation_before_restore_and_update_before_observation_rejected(self):
        controller=self.controller()
        for observation in (99,True,100.5):
            with self.assertRaises(ValueError):controller.start_restore('holder',100,previous.A,'r',observation_ns=observation)
        first=controller.start_restore('holder',100,previous.A,'r',observation_ns=1000)
        with self.assertRaises(ValueError):controller.update(first['state'],999,previous.A)
