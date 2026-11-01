import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.test_twoFIFO_owned_training_review_v2 import fixture, digest
from tools.predictor_validation_20261027 import twoFIFO_busy_transport_observed_v1 as module
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources


def run(plan, raw):
    return module.summarize(raw, plan, capture_sha256='CPU_FIXTURE_ONLY')


def find(result, rid):
    return next(r for r in result['records'] if r['request_id'] == rid)


class BusyObservedTests(unittest.TestCase):
    def test_complete_denominator_priming_and_unknown_initial_owner(self):
        plan, raw = fixture(); before = copy.deepcopy((plan, raw)); result = run(plan, raw)
        self.assertEqual(before, (plan, raw))
        self.assertEqual(result['counts']['offered'], 3)
        self.assertEqual(result['priming_rows_used_only_for_history'], 18)
        self.assertEqual(find(result, 'first')['previous_request_id'], 'prime17')
        self.assertEqual(find(result, 'first')['public_owner_ordinal_including_priming'], 18)
        peer = find(result, 'peer')
        self.assertIsNone(peer['previous_handler_busy_at_submit'])
        self.assertIsNone(peer['idle_handler_component_bounds'])
        self.assertEqual(result['counts']['predecessor_unknown'], 1)
        self.assertFalse(result['native_evidence_authenticated'])

    def test_busy_sender_pressure_distinct_and_no_idle_bounds_or_demand_fit(self):
        plan, raw = fixture(); first, second = raw['request_records'][:2]
        second['enqueued_time_ns'] = first['enqueued_time_ns'] + 1
        second['actual_submit_time_ns'] = first['submit_write_completed_time_ns'] + 1
        result = run(plan, raw); row = find(result, 'second')
        self.assertTrue(row['previous_handler_busy_at_submit'])
        self.assertTrue(row['sender_predecessor_write_record_pending_at_enqueue'])
        self.assertLess(row['prior_READY_to_submit_signed_ns'], 0)
        self.assertIsNone(row['worker_pre_START_demand_ns'])
        self.assertIsNone(row['stdin_line_available_ns'])
        self.assertIsNone(row['idle_handler_component_bounds'])
        self.assertIsNone(row['idle_handler_joint_endpoint_pairs'])
        self.assertEqual(result['counts']['busy_at_submit'], 1)
        self.assertEqual(result['counts']['sender_predecessor_write_record_pending_at_enqueue'], 1)
        self.assertEqual(row['previous_READY_to_START_ns'], second['observed_worker_start_time_ns'] - first['previous_handler_next_request_ready']['perf_ns'])
        self.assertFalse(result['idle_to_busy_transfer_used'])

    def test_idle_pairs_joint_and_write_record_can_follow_START(self):
        plan, raw = fixture(); result = run(plan, raw); row = find(result, 'second')
        self.assertFalse(row['previous_handler_busy_at_submit'])
        self.assertTrue(row['worker_START_precedes_parent_write_record'])
        self.assertIsNotNone(row['idle_handler_component_bounds'])
        for pair in row['idle_handler_joint_endpoint_pairs']:
            self.assertEqual(pair['line_extra_ns'] + pair['worker_pre_START_ns'], row['observed_submit_to_START_ns'])
        self.assertIsNone(row['worker_pre_START_demand_ns'])
        self.assertFalse(row['pure_stdin_idle_authenticated'])

    def test_late_write_after_RESULT_is_legal_and_not_pipe_line_availability(self):
        plan, raw = fixture(); first, second = raw['request_records'][:2]
        first['submit_write_completed_time_ns'] = second['actual_submit_time_ns'] - 1
        result = run(plan, raw)
        self.assertTrue(find(result, 'first')['write_record_after_RESULT_reader'])
        self.assertEqual(result['counts']['write_record_after_RESULT'], 1)
        self.assertFalse(result['parent_write_record_gates_worker_START'])

    def test_whole_attempt_predecessor_does_not_reset_between_cases(self):
        plan, raw = fixture(); first, second, peer = raw['request_records']; shift = 70_000_000_000
        old = plan['cases'][0]
        first_arrivals = [a for a in old['arrivals'] if a['request_id'] != 'second']
        second_arrivals = [a for a in old['arrivals'] if a['request_id'] == 'second']
        old.update(arrivals=first_arrivals, public_arrivals_sha256=digest(first_arrivals))
        plan['cases'].append(dict(case_id='later', arrivals=second_arrivals, public_arrivals_sha256=digest(second_arrivals)))
        raw['request_records'] = [first, peer, second]; second['profile_job_id'] = 'later'
        for key in ('scheduled_time_ns', 'enqueued_time_ns', 'actual_submit_time_ns', 'submit_write_completed_time_ns',
            'observed_worker_start_time_ns', 'completion_time_ns', 'response_reader_event_enqueue_time_ns',
            'response_collector_dequeued_time_ns', 'response_collector_processing_time_ns'):
            second[key] += shift
        for key in ('start_perf_ns', 'end_perf_ns'): second['worker_response'][key] += shift
        for e in second['execution_events'] + [second['previous_handler_next_request_ready']]:
            for key in ('perf_ns', 'received_time_ns', 'collector_dequeued_time_ns', 'collector_processing_time_ns'): e[key] += shift
        observed = raw['cases'][0]
        observed.update(request_ids=['first', 'peer'], public_arrivals_sha256=digest(first_arrivals))
        later = copy.deepcopy(observed); later.update(case_id='later', request_ids=['second'], public_arrivals_sha256=digest(second_arrivals))
        for key in ('epoch_ns', 'measurement_end_ns', 'fixed_drain_deadline_ns', 'observation_finalized_ns'): later[key] += shift
        raw['cases'].append(later)
        result = run(plan, raw)
        self.assertEqual(find(result, 'second')['previous_request_id'], 'first')
        self.assertEqual(find(result, 'second')['previous_case_id'], 'case')
        self.assertEqual(result['case_histories'][1]['prior_public_request_ids'][-2:], ['first', 'peer'])
        self.assertFalse(result['case_history_reset'])

    def test_incomplete_denominator_old_schema_and_validation_roles_rejected(self):
        for mode in ('incomplete', 'delete', 'oldschema', 'validation'):
            plan, raw = fixture()
            if mode == 'incomplete': raw['request_records'][1]['status'] = 'incomplete'
            elif mode == 'delete': raw['request_records'].pop()
            elif mode == 'oldschema': raw['schema'] = 'old150'
            else: raw['split'] = 'validation'
            with self.subTest(mode=mode), self.assertRaises(ValueError): run(plan, raw)

    def test_cross_request_stdout_causality_not_processing_gate(self):
        plan, raw = fixture(); first = raw['request_records'][0]
        first['previous_handler_next_request_ready']['collector_processing_time_ns'] += 1_000_000
        with self.assertRaisesRegex(ValueError, 'producer READY'): run(plan, raw)

    def test_native_entry_cannot_bypass_exact_review_and_reverse_collector_closure(self):
        with patch.object(module.macros_v2, 'extract', side_effect=ValueError('exact native rederive required')) as call:
            with self.assertRaisesRegex(ValueError, 'exact native rederive'): module.extract('forged')
            call.assert_called_once_with('forged')
        root = Path(__file__).resolve().parents[1]
        names = {Path(r['path']).name for r in implementation_sources(root / 'tools/predictor_validation_20261027/twoFIFO_owned_collect_v2.py')}
        self.assertNotIn('twoFIFO_busy_transport_observed_v1.py', names)


if __name__ == '__main__': unittest.main()
