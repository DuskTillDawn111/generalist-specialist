"""Keep host, CUDA and outer CPU timing scopes separate in diagnostic reports."""
import unittest
from experiments.diagnose_activation_host_brackets import describe_event


class HostBracketDiagnosisTests(unittest.TestCase):
    def test_cpu_active_and_waiting_are_reported_without_gpu_subtraction(self):
        event=dict(component='prepare_and_media',ordinal=0,thread_id=1,host_start_ns=1_000_000,
            host_end_ns=61_000_000,device_elapsed_ms=5.,
            passive_host_bracket=dict(schema='gsdeploy.passive_host_call_bracket.v1',thread_id=1,native_thread_id=2,
                wall_start_ns=0,wall_end_ns=62_000_000,cpu_start_ns=100_000_000,cpu_end_ns=155_000_000,
                usage_before=dict(user_cpu_seconds=1.,system_cpu_seconds=2.,minor_page_faults=4),
                usage_after=dict(user_cpu_seconds=1.005,system_cpu_seconds=2.05,minor_page_faults=8)))
        value=describe_event(event)
        self.assertEqual(value['semantic_host_ms'],60)
        self.assertEqual(value['outer_wall_ms'],62)
        self.assertEqual(value['thread_cpu_ms'],55)
        self.assertEqual(value['outer_wall_minus_thread_cpu_ms'],7)
        self.assertEqual(value['device_interval_ms'],5)
        self.assertAlmostEqual(value['usage_system_ms'],50)
        self.assertEqual(value['usage_delta']['minor_page_faults'],4)


if __name__=='__main__':unittest.main()
