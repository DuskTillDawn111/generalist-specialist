"""Analytical tests of causal integration, empty risk and single budgets."""
import copy
import math
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import OwnerLifecycle, RESIDENTS


class PiecewiseHazard:
    def query(self,holder,frontier,states,age,horizon_ns=0):
        competitor = states['text#1'].handler_active
        rate = 10. if competitor else 1.
        return dict(hazard_per_ms=rate,OOD_reasons=['EMPTY_RISK_IS_STATISTICAL'] if not frontier else [])


class AgeHazard:
    def query(self,holder,frontier,states,age,horizon_ns=0):
        return dict(hazard_per_ms=0. if age < 100_000 else 2.,OOD_reasons=[])


class ZeroHazard:
    def query(self,*args):
        return dict(hazard_per_ms=0.,OOD_reasons=['EXPLICIT_ZERO_FIXTURE'])


class ControllerTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)/'training.json'
        self.source.write_text('{"test":true}')
        self.low = {owner:OwnerLifecycle(False,False,False) for owner in RESIDENTS}
        self.low['text#0'] = OwnerLifecycle(True,None,False)
        self.high = dict(self.low, **{'text#1':OwnerLifecycle(True,True,True)})
        self.frontier = [['text#0','text','known',['decode']]]

    def controller(self,candidate=None,**kwargs):
        return CumulativeLifecycleHazard(candidate or PiecewiseHazard(),
            age_edges_ns=(0,100_000,1_000_000),sources=[receipt(self.source)],candidate_identity='a'*64,
            allow_unvalidated_approximation=True,**kwargs)

    def start(self,controller,**kwargs):
        return controller.start_restore('text#0',0,self.frontier,'restore-1',self.low,**kwargs)

    def test_lifecycle_update_integrates_old_rate_and_never_redraws(self):
        c = self.controller()
        first = self.start(c)
        updated = c.update(first['state'],100_000,self.frontier,self.high)
        self.assertAlmostEqual(updated['consumed_hazard'],.1)
        self.assertEqual(updated['hazard_budget'],first['hazard_budget'])
        expected = 100_000+math.ceil((math.log(2)-.1)/10*1e6)
        self.assertEqual(updated['absolute_save_candidate_ns'],expected)
        terminal = c.update(updated['state'],expected,self.frontier,self.low)
        self.assertEqual(terminal['status'],'SAVE_BUDGET_REACHED')
        self.assertEqual(terminal['state']['lifecycle'],updated['state']['lifecycle'])

    def test_empty_frontier_consumes_supported_hazard(self):
        c = self.controller()
        first = c.start_restore('text#0',0,[],'empty',self.low)
        changed = c.update(first['state'],200_000,[],self.low)
        self.assertAlmostEqual(changed['consumed_hazard'],.2)
        self.assertEqual(changed['absolute_save_candidate_ns'],math.ceil(math.log(2)*1e6))
        self.assertIn('EMPTY_RISK_IS_STATISTICAL',changed['uncertainty_reasons'])
        self.assertNotIn('empty_intervals',changed['state'])

    def test_conditional_entry_does_not_integrate_preentry_age(self):
        c = self.controller()
        first = self.start(c,observation_ns=400_000)
        self.assertEqual(first['consumed_hazard'],0.)
        self.assertEqual(first['absolute_save_candidate_ns'],400_000+math.ceil(math.log(2)*1e6))
        self.assertFalse(first['preentry_hazard_integrated'])

    def test_age_bin_zero_rate_then_positive_exact_crossing(self):
        c = self.controller(AgeHazard())
        first = self.start(c)
        expected = 100_000+math.ceil(math.log(2)/2*1e6)
        self.assertEqual(first['absolute_save_candidate_ns'],expected)
        self.assertEqual(c.update(first['state'],expected,self.frontier,self.low)['status'],'SAVE_BUDGET_REACHED')

    def test_explicit_zero_fixture_has_no_invented_deadline(self):
        c = self.controller(ZeroHazard())
        first = self.start(c)
        self.assertIsNone(first['absolute_save_candidate_ns'])
        self.assertIsNone(first['next_review_ns'])
        self.assertEqual(c.update(first['state'],10_000_000,[],self.low)['consumed_hazard'],0.)

    def test_refresh_partition_and_unchanged_budget(self):
        c = self.controller()
        first = self.start(c)
        direct = c.update(first['state'],600_000,self.frontier,self.low)
        many = first
        for clock in range(1000,600_001,1000):
            many = c.update(many['state'],clock,self.frontier,self.low)
        self.assertAlmostEqual(many['consumed_hazard'],direct['consumed_hazard'],places=14)
        self.assertEqual(many['absolute_save_candidate_ns'],direct['absolute_save_candidate_ns'])
        self.assertEqual(many['hazard_budget'],first['hazard_budget'])

    def test_seeded_budget_is_repeatable_and_state_mutation_rejected(self):
        c = self.controller(budget_policy='SEEDED_UNIFORM',seed=17)
        first = self.start(c)
        self.assertEqual(first['hazard_budget'],self.start(c)['hazard_budget'])
        bad = copy.deepcopy(first['state']);bad['lifecycle']['text#1']['handler_active'] = True
        with self.assertRaisesRegex(ValueError,'Sealed'):
            c.update(bad,1000,[],self.high)
        self.source.write_text('{"test":fals}')
        with self.assertRaisesRegex(ValueError,'source bytes changed'):
            c.assert_sources_unchanged()


if __name__ == '__main__':
    unittest.main()
