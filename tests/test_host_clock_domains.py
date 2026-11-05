"""Separate clock rates cannot justify rejecting an otherwise valid raw bracket."""
import copy
import unittest
from experiments.analyze_activation_host_diagnostic_v2 import validate_host_bracket
from experiments.diagnose_activation_host_brackets_v2 import describe_event


class HostClockDomainTests(unittest.TestCase):
    def event(self):
        return dict(component='model_forward',ordinal=0,thread_id=1,host_start_ns=1000,
            host_end_ns=400_000_000,device_elapsed_ms=399.,
            passive_host_bracket=dict(schema='gsdeploy.passive_host_call_bracket.v1',thread_id=1,native_thread_id=2,
                wall_start_ns=0,wall_end_ns=400_002_000,cpu_start_ns=500_000_000,cpu_end_ns=900_020_000,
                usage_before=dict(user_cpu_seconds=1.,system_cpu_seconds=2.),
                usage_after=dict(user_cpu_seconds=1.1,system_cpu_seconds=2.3)))

    def test_small_cross_clock_inversion_is_retained_not_clipped(self):
        event=self.event();validate_host_bracket(event);value=describe_event(event)
        self.assertAlmostEqual(value['signed_outer_wall_minus_thread_cpu_ms'],-.018)
        self.assertAlmostEqual(value['outer_wall_ms'],400.002)
        self.assertAlmostEqual(value['thread_cpu_ms'],400.02)

    def test_negative_own_clock_or_wrong_identity_still_rejected(self):
        for update in [dict(cpu_end_ns=499_999_999),dict(thread_id=3),dict(wall_end_ns=300_000_000)]:
            event=copy.deepcopy(self.event());event['passive_host_bracket'].update(update)
            with self.subTest(update=update),self.assertRaises(ValueError):validate_host_bracket(event)


if __name__=='__main__':unittest.main()
