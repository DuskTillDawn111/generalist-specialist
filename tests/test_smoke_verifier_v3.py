"""CPU regression tests for smoke false-PASS and asynchronous GPU boundaries."""
from copy import deepcopy
import unittest

from tools.predictor_validation_20261027.profiling import RESIDENTS
from tools.predictor_validation_20261027.smoke_verifier_v3 import (
    validate_coverage, validate_parsed, validate_request, validate_telemetry,
)


class SmokeVerifierTests(unittest.TestCase):
    def request(self):
        return dict(request_id='r', model='text#0', task='text', status='completed',
            scheduled_time_ns=5, actual_submit_time_ns=8, completion_time_ns=110,
            response_reader_event_enqueue_time_ns=110, response_collector_dequeued_time_ns=111,
            response_collector_processing_time_ns=112,
            worker_response=dict(request_id='r', task_type='text', success=True,
                start_perf_ns=10, end_perf_ns=100, service_ms=.1, worker_latency_ms=.1,
                semantic_forward_observations=dict(records=[{}], instrumentation_errors=[])),
            execution_events=[
                dict(request_id='r', pid=123, perf_ns=10, received_time_ns=11, collector_dequeued_time_ns=12, collector_processing_time_ns=13),
                dict(request_id='r', kind='gpu', segment_id='prefill', event='start', perf_ns=20, received_time_ns=21, collector_dequeued_time_ns=22, collector_processing_time_ns=23),
                dict(request_id='r', kind='gpu', segment_id='prefill', event='end', perf_ns=30, received_time_ns=31, collector_dequeued_time_ns=32, collector_processing_time_ns=33)],
            previous_handler_next_request_ready=dict(request_id='r', pid=123, perf_ns=105,
                clock='host_perf_counter_ns', boundary='previous_handler_next_request_ready',
                received_time_ns=106, collector_dequeued_time_ns=107, collector_processing_time_ns=108))

    def test_zero_requests_cannot_pass_four_owner_plan(self):
        job = dict(id='job', inputs=dict.fromkeys(RESIDENTS, 'pool'), split='smoke', validation_view='smoke')
        issues = validate_coverage({'jobs': [job]}, {'request_records': [], 'batches': []}, {'jobs': [job], 'trace': True})
        self.assertIn('PLANNED_REQUEST_COVERAGE_MISMATCH', issues)

    def test_actual_start_pid_is_required(self):
        row = self.request()
        self.assertEqual(validate_request(row, 123)[0], [])
        row['execution_events'][0]['pid'] = 999
        self.assertIn('START_PID_IDENTITY_MISMATCH:r', validate_request(row, 123)[0])

    def test_phase_cannot_borrow_another_request_anchor(self):
        row = self.request()
        row['execution_events'][1]['request_id'] = 'another'
        self.assertTrue(validate_request(row, 123)[0])

    def test_same_handler_async_kernel_past_forward_is_valid(self):
        row = self.request()
        _, anchors = validate_request(row, 123)
        phase = {**anchors[0], 'clock_mapping_valid': True,
                 'kernels': [dict(pid=123, start_ns=25, end_ns=90)]}
        parsed = dict(phases=[phase], clock_mappings={'123': dict(valid=True, max_boundary_residual_ns=0)})
        self.assertEqual(validate_parsed(parsed, anchors, [row], 123)[0], [])
        parsed['phases'].append(deepcopy(phase))
        self.assertIn('EXACT_CUPTI_ANCHOR_KEY_COVERAGE_FAILED', validate_parsed(parsed, anchors, [row], 123)[0])

    def test_every_telemetry_sample_must_include_all_owned_pids(self):
        ready = {alias: dict(pid=index+1) for index, alias in enumerate(RESIDENTS)}
        sample = dict(time_ns=1, query_duration_ns=1, watts=50, processes_mib={str(i): 10 for i in range(1, 5)},
                      owned_used_mib=40, owned_aliases=list(RESIDENTS), foreign_pids=[])
        raw = dict(ready=ready, observation_start_ns=2, observation_end_ns=3,
                   board_samples=[sample, {**deepcopy(sample), 'time_ns': 4}])
        self.assertEqual(validate_telemetry(raw), [])
        raw['board_samples'][0]['processes_mib'].pop('4')
        self.assertIn('INVALID_BOARD_OR_OWNED_MEMORY_SAMPLE', validate_telemetry(raw))

    def test_semantic_sink_failure_cannot_pass(self):
        row = self.request()
        row['worker_response']['semantic_forward_observations']['instrumentation_errors'] = [{'where': 'sink'}]
        self.assertIn('SEMANTIC_FORWARD_OBSERVATION_INVALID:r', validate_request(row, 123)[0])


if __name__ == '__main__':
    unittest.main()
