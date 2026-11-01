"""Independent CPU counterexamples for exact anchors and atomic prefix resumption."""
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.dag_busy_posterior_v8 import key, _emissions
from tools.predictor_validation_20261027.dag_checkpoint_v9 import decode
from tools.predictor_validation_20261027.dag_visible_anchor_v10 import (
    ConditionalVisibleStartSupply, UnsupportedStartObservation, start_lag_radii)
from tests import test_dag_request_start_v9 as start_fixtures
from tests import test_dag_busy_posterior_v8 as busy_fixtures


class IndependentVisiblePrefixTests(unittest.TestCase):
    def test_occurrence_support_is_owner_specific_and_sorted_by_public_arrival(self):
        rows = {rid: dict(request_id=rid, alias=alias, task='text', pool_id='x', arrival_ns=at)
                for rid, alias, at in [('late', 'a', 30), ('other', 'b', 10), ('first', 'a', 10)]}
        snapshot = dict(full_public_cohort=rows, snapshot_sha256='visible', visible_submissions={},
                        observed_host_state=dict(requests={'late': dict(handler_start_ns=35)}))
        scenario = dict(id='A', training_run_ids=['A'], request_start_delay_ns_by_input={
            key('a', 'text', 'x'): [1, 7], key('b', 'text', 'x'): [100]})
        supply = ConditionalVisibleStartSupply(snapshot, scenario, {(key('a', 'text', 'x'), 1): 2})
        self.assertEqual(supply.from_owner_ready(rows['late'], owner_ready_ns=20), 35)
        self.assertEqual(supply.from_owner_ready(rows['first'], owner_ready_ns=10), 11)
        self.assertEqual(supply.from_owner_ready(rows['other'], owner_ready_ns=10), 110)
        with self.assertRaisesRegex(UnsupportedStartObservation, 'SUPPORT_UNAVAILABLE'):
            ConditionalVisibleStartSupply(snapshot, scenario, {(key('b', 'text', 'x'), 1): 1000})

    def test_radius_is_maximum_of_only_same_occurrence_start_lag_and_rejects_invalid_ns(self):
        coordinate = key('a', 'image', 'x')
        rows = [dict(input_key=coordinate, occurrence=0, absolute_channel_errors_ns={
                    'start_lag': e, 'handler_wall_diagnostic': 10**15}) for e in (2, 7, 3)]
        self.assertEqual(start_lag_radii(dict(leave_one_run_out_errors=rows)), {(coordinate, 0): 7})
        for bad in (True, 1.0, -1, None):
            rows[0]['absolute_channel_errors_ns']['start_lag'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                start_lag_radii(dict(leave_one_run_out_errors=rows))

    def test_pause_every_integer_threshold_uses_real_boundary_and_preserves_suffix(self):
        fixture = start_fixtures.RequestStartTests()
        model, rows = fixture.model({'a1': 13, 'a2': 9})
        expected = model.forecast(rows, 0, 60, drain_deadline_ns=200)
        baseline_rows = copy.deepcopy(rows)
        for threshold in range(1, 61):
            model, rows = fixture.model({'a1': 13, 'a2': 9})
            partial = model.forecast(rows, 0, 60, drain_deadline_ns=200,
                                     pause_at_or_after_ns=threshold)
            if 'checkpoint' not in partial:
                self.assertEqual(partial, expected)
                continue
            state = decode(partial['checkpoint']['state'])
            self.assertGreaterEqual(state['clock'], threshold)
            # Every clock returned by a time pause must also occur in an
            # uninterrupted event-count pause; no synthetic cutoff edge.
            reference_model, _ = fixture.model({'a1': 13, 'a2': 9})
            reference = reference_model.forecast(rows, 0, 60, drain_deadline_ns=200,
                                                pause_after_event_count=state['event_count'])
            self.assertEqual(reference['checkpoint'], partial['checkpoint'])
            continued_model, _ = fixture.model({'a1': 13, 'a2': 9})
            self.assertEqual(continued_model.forecast(rows, 0, 60, drain_deadline_ns=200,
                checkpoint=partial['checkpoint']), expected)
            self.assertEqual(rows, baseline_rows)

    def test_resume_consumes_original_absolute_event_budget(self):
        fixture = start_fixtures.RequestStartTests()
        model, rows = fixture.model()
        expected = model.forecast(rows, 0, 60, drain_deadline_ns=200)
        budget = expected['events'] - 1
        model, rows = fixture.model(); model.max_events = budget
        paused = model.forecast(rows, 0, 60, drain_deadline_ns=200, pause_at_or_after_ns=1)
        self.assertIn('checkpoint', paused)
        with self.assertRaisesRegex(UnknownEvidence, 'EVENT_BUDGET_EXCEEDED'):
            model.forecast(rows, 0, 60, drain_deadline_ns=200, checkpoint=paused['checkpoint'])
        changed_model, _ = fixture.model(); changed_model.max_events = budget + 1
        with self.assertRaises(ValueError):
            changed_model.forecast(rows, 0, 60, drain_deadline_ns=200, checkpoint=paused['checkpoint'])

    def test_partial_projection_keeps_future_phase_timestamps_without_clipping(self):
        row = dict(request_id='r', alias='a', task='text', pool_id='x', arrival_ns=100)
        profile = dict(start_reader_delay_ns=0, ready_reader_delay_ns=0, phases=[
            dict(segment_id='p', kind='gpu', event='start', perf_offset_ns=5, reader_delay_ns=0),
            dict(segment_id='p', kind='gpu', event='end', perf_offset_ns=500, reader_delay_ns=0)])
        scenario = dict(emissions={key('a', 'text', 'x'): profile})
        forecast = dict(cohort={'r': row}, starts={'r': 100}, finishes={})
        before = copy.deepcopy((scenario, forecast))
        projected = _emissions(forecast, scenario, partial=True)
        self.assertEqual([e['perf_ns'] for e in projected], [100, 105, 600])
        forecast['finishes']['r'] = dict(handler_end_ns=120, finish_ns=130, worker_next_ready_ns=130)
        with self.assertRaisesRegex(ValueError, 'boundary chronology'):
            _emissions(forecast, scenario)
        self.assertEqual(scenario, before[0])

    def test_chronology_failure_removes_only_bad_scenario(self):
        fixture = busy_fixtures.BusyPosteriorTests(); fixture.setUp(); self.addCleanup(fixture.doCleanups)
        fixture.support['scenarios'][2]['emissions'][key('a', 'text', 'x')]['phases'][1]['perf_offset_ns'] = 1000
        result = fixture.condition(capture_latent_checkpoints=False)
        self.assertEqual([c['id'] for c in result['candidates']], ['A', 'B'])
        self.assertEqual(len(result['rejected']), 1)
        self.assertEqual(result['rejected'][0]['id'], 'C')
        self.assertEqual(result['rejected'][0]['reason'], 'DAG_READER_JOINT_CHRONOLOGY_INCONSISTENT')


if __name__ == '__main__':
    unittest.main()
