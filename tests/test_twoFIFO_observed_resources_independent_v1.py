"""Analytic CPU-only boundary counterexamples for measured resources."""
import copy
import unittest

from tests import test_twoFIFO_observed_resources_v1 as fixture_module
from tools.predictor_validation_20261027 import twoFIFO_observed_resources_v1 as m


class ObservedResourcesIndependentTests(unittest.TestCase):
    def test_prestartup_memory_cannot_enter_startup_through_drain_peak(self):
        raw=fixture_module.fixture()
        raw['lifecycle_boundaries'][0]['time_ns']=fixture_module.SECOND
        raw['board_samples'][0]['owned_used_mib']=10000
        result=m.observed(raw)
        self.assertEqual(result['cases'][0]['startup_through_this_drain_owned_sampled_peak_mib'],100)
        self.assertEqual(result['all_startup_through_final_drain_owned_sampled_peak_mib'],100)

    def test_previous_case_drain_peak_retained_but_future_case_peak_not_backfilled(self):
        raw=fixture_module.fixture();baseline=m.observed(raw)
        raw['board_samples'][9]['owned_used_mib']=1000
        changed=m.observed(raw)
        self.assertEqual(changed['cases'][0],baseline['cases'][0])
        self.assertEqual(changed['cases'][1]['startup_through_this_drain_owned_sampled_peak_mib'],1000)
        self.assertEqual(changed['cases'][1]['local_measurement_through_drain_owned_sampled_peak_mib'],1000)

    def test_non_aligned_window_trapezoid_is_exact_and_drain_not_amortized(self):
        raw=fixture_module.fixture();second=fixture_module.SECOND
        raw['measurement_windows'][0].update(start_ns=2*second+second//2,end_ns=3*second+second//2)
        result=m.observed(raw)
        # Integral of 10+2t over [2.5,3.5] is 16J, average16W.
        self.assertEqual(result['cases'][0]['measurement_energy_joules'],16)
        self.assertEqual(result['cases'][0]['measurement_power_watts'],16)
        self.assertFalse(result['cases'][0]['startup_energy_assigned_to_case'])
        self.assertFalse(result['memory_safety_factor_applied'])
        self.assertFalse(result['startup']['amortization_applied'])


if __name__=='__main__':unittest.main()
