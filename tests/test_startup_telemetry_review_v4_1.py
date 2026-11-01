import unittest

from tests.test_startup_telemetry_review_v4 import StartupTelemetryTests
from tools.predictor_validation_20261027.startup_telemetry_review_v4_1 import validate_startup_telemetry


class ExplicitObservedZeroTests(unittest.TestCase):
    def test_explicit_shutdown_zero_is_valid_not_an_imputed_missing_pid(self):
        raw=StartupTelemetryTests().fixture()
        end=raw['board_samples'][-1]
        end.update(processes_mib={101:0.},owned_pid_aliases={101:'text#0'},owned_aliases=['text#0'],owned_used_mib=0.)
        self.assertEqual(validate_startup_telemetry(raw),[])

    def test_active_zero_still_fails_original_active_verifier(self):
        raw=StartupTelemetryTests().fixture()
        row=raw['board_samples'][4];row['processes_mib'][101]=0.;row['owned_used_mib']-=10.
        self.assertIn('INVALID_BOARD_OR_OWNED_MEMORY_SAMPLE',validate_startup_telemetry(raw))

    def test_negative_or_nan_reported_usage_always_invalid(self):
        for value in [-1.,float('nan')]:
            raw=StartupTelemetryTests().fixture();row=raw['board_samples'][-1]
            row.update(processes_mib={101:value},owned_pid_aliases={101:'text#0'},owned_aliases=['text#0'],owned_used_mib=value)
            self.assertIn('INVALID_STARTUP_OWNED_BOARD_SAMPLE',validate_startup_telemetry(raw))


if __name__=='__main__':unittest.main()
