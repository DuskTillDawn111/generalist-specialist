"""Independent CPU-only measured-label scope counterexamples."""
import copy
import unittest

from tests.test_twoFIFO_source_rebound_observed_score_v1 import fixture, execute, SECOND


class IndependentObservedScoreTests(unittest.TestCase):
    def test_future_case_and_after_current_drain_cannot_enter_current_score(self):
        raw, plan, _, _, score, quality = fixture()
        data = (raw, plan, plan['cases'][0], 2*SECOND, score, quality)
        base = execute(data)
        future = copy.deepcopy(data)
        for sample in future[0]['board_samples']:
            if sample['time_ns'] > 72*SECOND:
                sample.update(watts=1e12, owned_used_mib=1e12)
        changed = execute(future)
        self.assertEqual(changed['measured_resources'], base['measured_resources'])
        self.assertEqual(changed['measured_original_score'], base['measured_original_score'])

    def test_drain_boundary_counts_memory_and_never_extends_measurement_energy(self):
        data = fixture()
        base = execute(data)
        data[0]['board_samples'][143]['owned_used_mib'] = 99999
        data[0]['board_samples'][144]['owned_used_mib'] = 1e12
        changed = execute(data)
        self.assertEqual(changed['measured_resources']['phase_start_through_drain_owned_peak_mib'], 99999)
        self.assertEqual(changed['measured_original_score']['terms']['E'], base['measured_original_score']['terms']['E'])
        self.assertEqual(changed['measured_original_score']['terms']['M'],
            99999*data[-2]['memory_safe_factor']/data[-2]['memory_budget_mib'])

    def test_missing_prefix_is_unknown_and_cannot_be_filled_from_future(self):
        raw, plan, _, _, score, quality = fixture()
        # All available polls are after the first case epoch. Even a later
        # large memory sample cannot establish startup-through-start history.
        raw['board_samples'] = [s for s in raw['board_samples'] if s['time_ns'] > 2*SECOND]
        out = execute((raw, plan, plan['cases'][0], 2*SECOND, score, quality))
        self.assertIsNone(out['measured_resources']['startup_through_phase_start_owned_peak_mib'])
        self.assertIsNone(out['measured_original_score']['terms']['M'])
        self.assertIsNone(out['measured_original_score']['terms']['E'])
        self.assertIsNone(out['measured_original_score']['full_U'])
        self.assertEqual(out['measured_original_score']['cohort_size'], 1)

    def test_four_task_production_score_keeps_absent_task_unknown(self):
        out = execute(fixture())
        for term in ('Q', 'L', 'P'):
            self.assertIsNone(out['measured_original_score']['terms'][term])
        self.assertIsNone(out['measured_original_score']['full_U'])
        self.assertIsNotNone(out['measured_original_score']['terms']['M'])
        self.assertIsNotNone(out['measured_original_score']['terms']['E'])
        self.assertFalse(out['native_evidence_authenticated_by_this_pure_helper'])
        self.assertTrue(out['heldout_future_used_for_evaluation_only'])
        self.assertFalse(out['sampled_memory_is_continuous_capacity_bound'])


if __name__ == '__main__':
    unittest.main()
