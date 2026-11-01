import copy
from fractions import Fraction as F
import unittest

from tools.predictor_validation_20261027.twoFIFO_observed_resources_v1 import observed

SECOND=10**9


def fixture():
    return dict(shutdown_memory_sampled=False,
        lifecycle_boundaries=[dict(boundary=k,time_ns=v*SECOND) for k,v in
            [('startup_begin',0),('priming_end',1),('monitoring_end',11),('shutdown_begin',12)]],
        # Memory during the preceding case's drain must still affect next M.
        board_samples=[dict(time_ns=i*SECOND,watts=10+2*i,owned_used_mib=100 if i==5 else 5)
            for i in range(12)],
        cases=[dict(case_id='a'),dict(case_id='b')],measurement_windows=[
            dict(case_id='a',start_ns=2*SECOND,end_ns=4*SECOND,fixed_drain_deadline_ns=6*SECOND),
            dict(case_id='b',start_ns=7*SECOND,end_ns=8*SECOND,fixed_drain_deadline_ns=10*SECOND)])


class MeasuredResourceTests(unittest.TestCase):
    def test_exact_window_integrals_startup_once_and_cumulative_peak(self):
        raw=fixture();before=copy.deepcopy(raw);out=observed(raw);self.assertEqual(raw,before)
        self.assertEqual(out['startup']['energy_joules'],24)
        self.assertEqual(out['startup']['occurrences'],1)
        self.assertFalse(out['startup']['amortization_applied'])
        # Independent analytic primitive of P(t)=10+2t, not helper replay.
        primitive=lambda x:10*F(x)+F(x)**2
        for row,a,b in zip(out['cases'],(2,7),(4,8)):
            self.assertEqual(row['measurement_energy_joules'],float(primitive(b)-primitive(a)))
        self.assertEqual(out['cases'][1]['startup_through_this_drain_owned_sampled_peak_mib'],100)
        self.assertEqual(out['cases'][1]['local_measurement_through_drain_owned_sampled_peak_mib'],5)

    def test_drain_power_cannot_change_measured_energy(self):
        raw=fixture();original=observed(raw)
        raw['board_samples'][5]['watts']+=1000
        changed=observed(raw)
        self.assertEqual(original['cases'][0]['measurement_energy_joules'],changed['cases'][0]['measurement_energy_joules'])
        self.assertNotEqual(original['cases'][0]['drain_energy_joules_diagnostic_only'],changed['cases'][0]['drain_energy_joules_diagnostic_only'])

    def test_missing_bracket_or_large_gap_preserves_UNKNOWN(self):
        raw=fixture();raw['board_samples']=[s for s in raw['board_samples'] if s['time_ns'] not in (3*SECOND,4*SECOND)]
        out=observed(raw);self.assertIsNone(out['cases'][0]['measurement_energy_joules'])
        self.assertIsNotNone(out['cases'][1]['measurement_energy_joules'])
        raw=fixture();raw['board_samples']=raw['board_samples'][1:]
        self.assertIsNone(observed(raw)['startup']['energy_joules'])

    def test_shutdown_or_overlapping_case_windows_rejected(self):
        raw=fixture();raw['shutdown_memory_sampled']=True
        with self.assertRaises(ValueError):observed(raw)
        raw=fixture();raw['measurement_windows'][1]['start_ns']=5*SECOND
        with self.assertRaises(ValueError):observed(raw)


if __name__=='__main__':unittest.main()
