"""Causal brackets are not simultaneous clock samples or accuracy bounds."""
import unittest

from tools.predictor_validation_20261027.session_clock_audit_v2 import causal_offset_interval


class SessionClockAuditTests(unittest.TestCase):
    def test_nested_boundaries_yield_interval_and_unknown_drift(self):
        phases = [dict(pid=7,start_ns=1110,end_ns=1190,nsys_start_ns=100,nsys_end_ns=200),
                  dict(pid=7,start_ns=1302,end_ns=1395,nsys_start_ns=300,nsys_end_ns=400)]
        result = causal_offset_interval(phases,7)
        self.assertEqual((result['lower_ns'],result['upper_ns'],result['width_ns']),(995,1002,7))
        self.assertIsNone(result['drift_bound_ns'])
        self.assertIsNone(result['device_host_clock_calibration_error_bound_ns'])

    def test_incompatible_brackets_do_not_force_a_point_mapping(self):
        phases = [dict(pid=7,start_ns=1100,end_ns=1200,nsys_start_ns=100,nsys_end_ns=200),
                  dict(pid=7,start_ns=2200,end_ns=2300,nsys_start_ns=300,nsys_end_ns=400)]
        self.assertFalse(causal_offset_interval(phases,7)['known'])

    def test_missing_or_noninteger_boundaries_are_not_filled(self):
        self.assertFalse(causal_offset_interval([],7)['known'])
        with self.assertRaises(ValueError):
            causal_offset_interval([dict(pid=7,start_ns=0.,end_ns=1,nsys_start_ns=0,nsys_end_ns=1)],7)


if __name__ == '__main__':
    unittest.main()
