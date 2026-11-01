"""Independent CPU counterexamples for boundaries and declared scheduling.

These tests encode required behavior, including regressions reported during
review. They neither collect GPU data nor assert physical prediction accuracy.
"""
import copy
import unittest
from unittest.mock import patch

from scipy.stats import gamma

from gsdeploy.forecasting import queue_diagnostics
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.runtime.batching import BatchConfiguration, select_batch
from tests.test_full_horizon_forecast import aliases, provider, profiles, resources, request, MS, SECOND
from tests.test_queue_prediction_v2 import lifecycle


class IndependentArrivalTests(unittest.TestCase):
    def test_exact_gamma_conditioned_residual_in_integrated_rate_time(self):
        windows = [dict(start_seconds=0., end_seconds=1., rates={'text': 2.}),
                   dict(start_seconds=1., end_seconds=2., rates={'text': 0.}),
                   dict(start_seconds=2., end_seconds=4., rates={'text': 4.})]
        workload = FutureWorkload({'text': 1.}, {'text': ['x']}, process='gamma_renewal',
                                  interarrival_cv={'text': .5}, windows=windows)
        class FixedRandom:
            def __init__(self, *args): pass
            def random(self): return .25
            def choices(self, *args, **kwargs): return [{'pool_id': 'x'}]
            def gammavariate(self, *args): return 100.
        # Last arrival .5s => integrated age at 2s is1; renewal quantile
        # conditioned on this survival must be inverted through rate4 after2s.
        interval = gamma.isf(.75 * gamma.sf(1., a=4., scale=.25), a=4., scale=.25)
        expected = round((2. + (interval - 1.) / 4.) * SECOND)
        with patch('gsdeploy.forecasting.workload.random.Random', FixedRandom):
            rows = workload.suffix(0, 2 * SECOND, 4 * SECOND,
                                   observed_last_arrivals_ns={'text': SECOND // 2})['arrivals']
        self.assertEqual([r['arrival_ns'] for r in rows], [expected])


class IndependentEventTests(unittest.TestCase):
    def test_oldest_waiting_age_does_not_assume_arrival_sorted_fifo(self):
        result = FullHorizonModel(aliases(), provider()).forecast(
            {'s': [request('active', arrival_ns=0), request('recent', arrival_ns=90 * MS),
                   request('old', arrival_ns=10 * MS)]}, 100 * MS, 120 * MS,
            drain_deadline_ns=150 * MS)
        self.assertEqual(result['queue_diagnostics']['max_oldest_waiting_age_ms'], 140.)

    def test_tiny_stalled_cursor_cannot_complete_without_progress(self):
        class Stalled:
            execution_context_sha256 = 'fixture'
            def __init__(self): self.base = provider()
            def plan(self, *args): return self.base.plan(*args)
            def power(self, *args): return self.base.power(*args)
            def memory(self, *args): return self.base.memory(*args)
            def service(self, row, active, *args):
                if row['alias'] == 's' and 'g' in active:
                    return dict(known=True, context_supported=True, stalled=True, service_ms=None)
                return dict(known=True, context_supported=True, service_ms=row['segment_duration_ms'])
        cursor = dict(profile_id='s', segment_index=0, remaining_fraction=1e-12,
                      observed_at_ns=0, evidence='synthetic-tiny-positive-progress')
        result = FullHorizonModel(aliases(), Stalled()).forecast(
            {'s': [request('s', forecast_cursor=cursor)], 'g': [request('g')]},
            0, 10 * MS, drain_deadline_ns=20 * MS)
        self.assertEqual(result['finishes'], {})
        self.assertEqual(result['incomplete_ids'], ['g', 's'])

    def test_diagnostic_work_calls_scale_with_events_not_pending_product(self):
        n = 40
        original = queue_diagnostics.remaining_work
        with patch.object(queue_diagnostics, 'remaining_work', wraps=original) as counted:
            FullHorizonModel(aliases(), provider(s_ms=1)).forecast(
                {'s': [request(str(i)) for i in range(n)]}, 0, SECOND)
        # Generous bound permits per-profile and per-active-event work. It
        # rejects the observed N(N+1) full-pending recalculation regression.
        self.assertLess(counted.call_count, 10 * n)


class IndependentBatchTests(unittest.TestCase):
    def test_known_future_ready_is_next_formation_event_before_timeout(self):
        rows = [dict(request_id=str(i), owner='a', task='text', execution_key='e',
                     input_tokens=4, ready_ns=i * 10) for i in range(2)]
        config = BatchConfiguration(2, 16, 100)
        first = select_batch(rows, 0, config)
        self.assertEqual(first['status'], 'WAIT')
        self.assertEqual(first['next_event_ns'], 10)
        self.assertEqual(select_batch(rows, 10, config)['members'], ['0', '1'])


class IndependentCLITests(unittest.TestCase):
    def request(self):
        return dict(schema='gsdeploy.full_horizon_forecast_request.v1',
                    execution_context_sha256='fixture', profiles=profiles(), aliases=aliases(), resources=resources(),
                    queues={'s': [request('a')]}, now_ns=0, phase_start_ns=0,
                    deadline_ns=SECOND, drain_deadline_ns=2 * SECOND,
                    future_workload=dict(rates={'text': 1.}, input_pools={'text': ['x']},
                                         process='declared_schedule', declared_arrivals=[]),
                    continuation_routing_rps={'text': {'s': 1.}}, latency_reference_ms={'text': 100.},
                    quality_loss={'text': {'text': 0.}}, memory_budget_mib=1000., memory_safe_factor=1.,
                    energy_reference_watts=100., observed_peak_memory_mib=100., observed_board_energy_joules=0.)

    def test_lifecycle_fixture_does_not_inherit_measured_service_physical_evidence(self):
        from gsdeploy.forecasting.__main__ import forecast_request
        value = self.request()
        value['profiles']['provenance']['kind'] = 'measured'
        value['lifecycle_profiles'] = copy.deepcopy(lifecycle(gated=False).document)
        value['execution_contract'] = 'strict_lifecycle_v2'
        with patch('subprocess.Popen', side_effect=AssertionError('CPU review cannot launch processes')):
            result = forecast_request(value, allow_cpu_fixture=True)
        self.assertFalse(result['physical_evidence'])
        self.assertEqual(result['forecast']['finishes']['a']['finish_ns'], 120 * MS)

    def test_strict_lifecycle_cannot_silently_use_instant_handoff(self):
        from gsdeploy.forecasting.__main__ import forecast_request
        value = self.request(); value['execution_contract'] = 'strict_lifecycle_v2'
        with self.assertRaises(ValueError): forecast_request(value, allow_cpu_fixture=True)

    def test_lifecycle_presence_infers_strict_and_rejects_explicit_legacy(self):
        from gsdeploy.forecasting.__main__ import forecast_request
        value = self.request(); value['lifecycle_profiles'] = copy.deepcopy(lifecycle(gated=False).document)
        result = forecast_request(value, allow_cpu_fixture=True)
        self.assertEqual(result['execution_contract'], 'strict_lifecycle_v2')
        self.assertEqual(result['forecast']['finishes']['a']['finish_ns'], 120 * MS)
        value['execution_contract'] = 'legacy_instant_handoff'
        with self.assertRaises(ValueError): forecast_request(value, allow_cpu_fixture=True)


class IndependentScoreTests(unittest.TestCase):
    def forecast(self):
        row = dict(request('a'), alias='s')
        return dict(cohort={'a': row}, finishes={'a': {**row, 'finish_ns': 100 * MS}},
                    board_energy_joules=1., peak_memory_mib=1.)

    def score(self, forecast, *, completed=(), now=0, start=0):
        return score_forecast(forecast, list(completed), aliases(), {'text': 1.}, {'text': 100.},
                              {'text': {'text': 0.}, 'generalist_2': {'text': .1}},
                              phase_start_ns=start, now_ns=now, deadline_ns=SECOND,
                              memory_budget_mib=1000., energy_reference_watts=100.,
                              observed_peak_memory_mib=1., observed_board_energy_joules=0.)

    def test_outside_measurement_cohort_arrival_is_rejected(self):
        for arrival in (-1, SECOND):
            forecast = self.forecast()
            for mapping in ('cohort', 'finishes'):
                forecast[mapping]['a']['arrival_ns'] = arrival
            forecast['finishes']['a']['finish_ns'] = SECOND + MS
            with self.assertRaises(ValueError): self.score(forecast)

    def test_forecast_finish_must_match_cohort_public_identity(self):
        forecast = self.forecast(); forecast['finishes']['a']['arrival_ns'] = -SECOND
        with self.assertRaises(ValueError): self.score(forecast)

    def test_forecast_finish_time_is_integer_and_not_before_visible_now(self):
        for finish in (100.5, 100 * MS):
            forecast = self.forecast(); forecast['finishes']['a']['finish_ns'] = finish
            with self.assertRaises(ValueError): self.score(forecast, now=200 * MS)

    def test_historical_completion_arrivals_belong_to_measurement_phase(self):
        row = dict(request('past', arrival_ns=-SECOND), alias='s', completion_ns=0)
        forecast = self.forecast(); forecast['cohort'] = {}; forecast['finishes'] = {}
        with self.assertRaises(ValueError): self.score(forecast, completed=[row])


if __name__ == '__main__': unittest.main()
