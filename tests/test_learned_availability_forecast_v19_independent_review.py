"""Independent two-task CPU DAG audit and diagnostic-retention prototype.

The retention helpers below are review-only: no runtime/core file is changed.
"""
import copy
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027 import availability_forecast_v11 as original
from tools.predictor_validation_20261027.compiled_static_providers_v16 import CompiledStaticProviders
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import (
    DagPhaseFullHorizonModel, SchedulingDecision)
from tools.predictor_validation_20261027.learned_availability_forecast_v19 import (
    FullPublicDagQueueDiagnostics, LearnedAvailabilityForecast, QueueDiagnosticEngine, V16Forecast)
from tools.predictor_validation_20261027.native_collect import receipt
from tools.predictor_validation_20261027.prospective_flow_contract_v11 import score_complete_forecast


class RetainReturnedForecast:
    """Proposed independent engine proxy: retain only an actual normal return."""
    def __init__(self, engine):
        self.engine = engine
        self.last_returned_forecast = None

    def __getattr__(self, name):
        return getattr(self.engine, name)

    def forecast(self, *args, **kwargs):
        result = self.engine.forecast(*args, **kwargs)
        self.last_returned_forecast = result
        return result


def attach_unscored_diagnostic(result, engine):
    """Proposed outer execute step; preserve original status/reason/metrics."""
    if (result.get('status') == 'UNKNOWN' and 'forecast' not in result
            and engine.last_returned_forecast is not None):
        result['unscored_forecast_diagnostic'] = dict(
            forecast=engine.last_returned_forecast,
            score_rejection_reason=result.get('reason'), diagnostic_only=True,
            complete_prediction_or_gate_evidence=False)
    return result


class FixedScheduler:
    def decide(self, view):
        return SchedulingDecision({n.key: .5 if n.key[0] == 'text' else 1.
                                   for n in view.active_gpu_nodes}, None)

    def provenance(self):
        return {'kind': 'independent-fixed-rates'}


class IndependentV19Tests(unittest.TestCase):
    BASE = 42_000_000

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        source = Path(temporary.name) / 'synthetic-training.json'
        source.write_text('{"synthetic_only":true}')
        self.aliases = {task: dict(family=task, supported_tasks=[task]) for task in ('text', 'ocr')}
        self.plans = {}
        work = {'text': dict(copy=2, left=6, right=4, memset=1),
                'ocr': dict(copy=1, left=8, right=3, memset=2)}
        for task in self.aliases:
            nodes = [dict(id=i, kind=k) for i, k in (
                ('copy', 'copy'), ('record', 'event_record'), ('left', 'gpu'),
                ('right', 'gpu'), ('wait', 'event_wait'), ('memset', 'memset'),
                ('sync', 'host_sync'), ('end', 'handler_end'))]
            self.plans[task, task, 'known'] = dict(nodes=nodes, edges=[
                dict(before=a, after=b) for a, b in (
                    ('copy', 'record'), ('record', 'left'), ('record', 'right'),
                    ('left', 'wait'), ('right', 'wait'), ('wait', 'memset'),
                    ('memset', 'sync'), ('sync', 'end'))],
                default_stream_semantics='UNMEASURED_EXPLICIT_HYPOTHESIS')
        self.compiled = CompiledStaticProviders(self.aliases, self.plans,
            duration=lambda n, r: work[r['task']][n['id']]*1_000_000,
            release=lambda n, r: dict(not_before_offset_ns=0,
                                      after_predecessors_ns=1_000_000 if n['id'] == 'end' else 0),
            phase=lambda n, r: 'synthetic-phase', sources=[receipt(source)])
        self.rows = [dict(request_id=f'independent-{i}', alias=task, task=task,
                          pool_id='known', arrival_ns=self.BASE + int(offset*1e6))
                     for i, (task, offset) in enumerate(
                         [('text', 0), ('ocr', 1), ('text', 5), ('ocr', 70), ('text', 90)])]
        self.plan = dict(aliases=self.aliases, window_seconds=.1, drain_seconds=.2,
            workload_rps={'text': 30., 'ocr': 20.}, latency_reference_ms={'text': 100., 'ocr': 200.},
            memory_budget_mib=10000., memory_safe_factor=1.2,
            energy_reference_watts=450., weights={k: 1/6 for k in 'QTLPME'})
        self.startup = dict(energy_joules=7200., peak_memory_mib=3500.,
            energy_includes_priming=True, safety_factor_applied=False,
            startup_amortization_seconds=3600., warmup_seconds=0.)
        self.quality = {'text': {'text': .2}, 'ocr': {'ocr': dict(
            basis='expected_per_request_totals', numerator=1., denominator=10.,
            evidence='synthetic training expectation')}}

    def engine(self, observed=False):
        engine = DagPhaseFullHorizonModel(self.compiled.graph.aliases, self.compiled.graph.plans,
            duration_provider=self.compiled.duration, release_provider=self.compiled.release,
            phase_provider=self.compiled.phase, scheduler=FixedScheduler(),
            lifecycle_provider=SimpleNamespace(delays=lambda *a: dict(
                known=True, parent_receipt_ms=2., worker_next_ready_ms=3.)),
            power_provider=SimpleNamespace(power=lambda handlers, *a:
                                           dict(known=True, watts=100+10*len(handlers))),
            memory_provider=SimpleNamespace(memory=lambda handlers, *a:
                                            dict(known=True, peak_memory_mib=1000+1000*len(handlers))),
            require_energy=True, require_memory=True, max_events=10000,
            allow_independent_copy_hypothesis=True, allow_no_implicit_default_stream_hypothesis=True)
        return QueueDiagnosticEngine(engine, self.compiled, list(self.compiled.sources)) if observed else engine

    def forecast(self, engine, drain_ms=300, rows=None):
        return engine.forecast(self.rows if rows is None else rows, self.BASE,
            self.BASE+100_000_000, drain_deadline_ns=self.BASE+int(drain_ms*1e6))

    def test_all_original_fields_and_hand_calculated_two_task_six_terms(self):
        for drain in (.5, 5, 20, 130, 300):
            with self.subTest(drain_ms=drain):
                plain = self.forecast(self.engine(), drain)
                wrapped = self.engine(True)
                observed = self.forecast(wrapped, drain)
                original_fields = dict(observed)
                original_fields['queue_diagnostics'] = plain['queue_diagnostics']
                self.assertEqual(original_fields, plain)
                self.assertEqual(observed['cohort'], {r['request_id']: r for r in self.rows})
                if drain == 300:
                    self.assertEqual(len(observed['finishes']), 5)
                    e2e = [(observed['finishes'][r['request_id']]['finish_ns']-r['arrival_ns'])/1e6
                           for r in self.rows]
                    self.assertEqual(e2e, [18., 14., 32., 14., 18.])
                    scores = [score_complete_forecast(f, self.rows, self.plan, self.quality,
                        self.startup, phase_start_ns=self.BASE) for f in (plain, observed)]
                    self.assertEqual(scores[0], scores[1])
                    expected = dict(Q=.16, T=.2, L=.164, P=.2116, M=.42, E=10.86/45)
                    for key, value in expected.items():
                        self.assertAlmostEqual(scores[1]['terms'][key], value, msg=key)
                    self.assertAlmostEqual(scores[1]['full_U'], -sum(expected.values())/6)
                    self.assertEqual(observed['max_simultaneous_nodes_per_owner']['text'], 2)

    def test_future_costs_do_not_become_live_backlog_and_zero_delta_does_not_consume(self):
        diagnostic = FullPublicDagQueueDiagnostics(self.compiled.graph.aliases,
            self.compiled.graph.plans, self.compiled.duration, cohort=self.rows, now_ns=self.BASE)
        initial = diagnostic.result({r['request_id']: r for r in self.rows})
        self.assertEqual(initial['registered_request_count'], 0)
        self.assertEqual(initial['remaining_activity_work_ms'], 0.)
        self.assertEqual(initial['offered_work_envelopes'][0]['max_offered_solo_work_ms'], 67.)
        wrapped = self.engine(True)
        forecast = self.forecast(wrapped, .5)
        q = forecast['queue_diagnostics']
        self.assertEqual(q['registered_request_count'], 1)
        self.assertEqual(q['max_waiting_requests'], 0)
        self.assertEqual(len(q['not_yet_registered_request_ids']), 4)
        self.assertEqual(q['remaining_activity_work_ms'], 12.5)
        self.assertEqual(q['zero_duration_state_snapshots_without_rate_imputation'], 1)
        self.assertTrue(all(value is None for value in q['max_normalized_resource_work_ms'].values()))
        self.assertEqual(q['offered_work_envelopes'][0]['max_offered_solo_work_ms'], 67.)

    def test_unknown_future_profile_is_not_zero_and_original_engine_reason_survives(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['pool_id'] = 'unsupported'
        wrapped = self.engine(True)
        forecast = self.forecast(wrapped, .5, rows)
        self.assertIsNone(forecast['queue_diagnostics']['offered_work_envelopes'][0]['max_offered_solo_work_ms'])
        self.assertEqual(forecast['queue_diagnostics']['offered_work_envelopes'][0]['max_arrival_count'], 5)
        self.assertEqual(forecast['queue_diagnostics']['remaining_activity_work_ms'], 12.5)
        reasons = []
        for engine in (self.engine(), self.engine(True)):
            with self.assertRaises(UnknownEvidence) as caught:
                self.forecast(engine, 300, rows)
            reasons.append(str(caught.exception))
        self.assertEqual(reasons[0], reasons[1])
        self.assertIn('UNKNOWN_DAG_INPUT', reasons[0])
        self.assertIsNone(engine.sidecar['diagnostics']['remaining_activity_work_ms'])

    def original_execute(self, engine, drain_ms):
        planref = {'path': '/synthetic/plan.json', 'sha256': 'synthetic'}
        contextref = {'path': '/synthetic/context.json', 'sha256': 'synthetic'}
        startupref = {'path': '/synthetic/startup.json', 'sha256': 'synthetic'}
        plan = dict(self.plan, context=contextref, execution_context_sha256='context')
        instance = SimpleNamespace(aliases=self.aliases, defaults={},
            configuration={k: startupref for k in ('parameters', 'availability_model', 'component_bundle')},
            bundle={'weights': plan['weights']}, source_native={'projection_sha256': 'projection'},
            source_target_context_ref=contextref, generation=SimpleNamespace(check=lambda rows: {}),
            lifecycle=SimpleNamespace(quality=self.quality), engine=lambda **kwargs: engine)
        public = dict(now_ns=self.BASE, phase_start_ns=self.BASE,
            deadline_ns=self.BASE+100_000_000, drain_deadline_ns=self.BASE+int(drain_ms*1e6))
        with mock.patch.object(original, 'checked', side_effect=lambda ref: plan if ref == planref else {}), \
             mock.patch.object(original, 'original_flow_defaults', return_value={}), \
             mock.patch.object(original, 'bind_public', return_value=self.rows), \
             mock.patch.object(original, 'native_execution_projection', return_value=dict(
                 original_full_context_sha256='context', projection_sha256='projection')), \
             mock.patch.object(original, 'startup_prediction', return_value=self.startup):
            return original.AvailabilityForecast.execute(instance, planref, 'synthetic', public, {}, startupref)

    def test_engine_resource_unknown_is_not_converted_to_zero_or_masked(self):
        for kind, reason in (('power', 'UNKNOWN_DAG_WHOLE_BOARD_POWER_CONTEXT'),
                             ('memory', 'UNKNOWN_DAG_WHOLE_BOARD_PEAK_MEMORY_CONTEXT')):
            reasons = []
            wrapped = None
            for observed in (False, True):
                engine = self.engine(observed)
                target = engine._engine if observed else engine
                setattr(target, kind+'_provider', SimpleNamespace(**{
                    kind: lambda *a: dict(known=False)}))
                with self.assertRaises(UnknownEvidence) as caught:
                    self.forecast(engine)
                reasons.append(str(caught.exception))
                if observed:
                    wrapped = engine
            self.assertEqual(reasons, [reason, reason])
            self.assertEqual(wrapped.sidecar['original_forecast_unknown_reason'], reason)
            self.assertTrue(all(value is None for value in
                wrapped.sidecar['diagnostics']['max_normalized_resource_work_ms'].values()))

    def test_actual_v11_score_rejection_loses_forecast_and_minimal_proxy_preserves_only_diagnostic(self):
        captured = RetainReturnedForecast(self.engine(True))
        rejected = self.original_execute(captured, .5)
        self.assertEqual(rejected['status'], 'UNKNOWN')
        self.assertNotIn('forecast', rejected)
        self.assertNotIn('metrics', rejected)
        before = copy.deepcopy(rejected)
        attach_unscored_diagnostic(rejected, captured)
        self.assertEqual({k: v for k, v in rejected.items() if k != 'unscored_forecast_diagnostic'}, before)
        diagnostic = rejected['unscored_forecast_diagnostic']
        self.assertIs(diagnostic['forecast'], captured.last_returned_forecast)
        self.assertEqual(len(diagnostic['forecast']['cohort']), 5)
        self.assertEqual(len(diagnostic['forecast']['finishes']), 0)
        self.assertFalse(diagnostic['complete_prediction_or_gate_evidence'])
        complete = RetainReturnedForecast(self.engine(True))
        result = self.original_execute(complete, 300)
        attach_unscored_diagnostic(result, complete)
        self.assertEqual(result['status'], 'COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST')
        self.assertNotIn('unscored_forecast_diagnostic', result)
        self.assertIs(result['forecast'], complete.last_returned_forecast)
        engine_unknown = RetainReturnedForecast(self.engine(True))
        unsupported = [dict(self.rows[0], pool_id='unsupported')]
        with self.assertRaises(UnknownEvidence) as caught:
            self.forecast(engine_unknown, 300, unsupported)
        self.assertIsNone(engine_unknown.last_returned_forecast)
        failed = dict(status='UNKNOWN', reason=str(caught.exception))
        attach_unscored_diagnostic(failed, engine_unknown)
        self.assertNotIn('unscored_forecast_diagnostic', failed)

    def test_actual_v19_entry_injects_at_forecast_without_model_constructors(self):
        instance = LearnedAvailabilityForecast.__new__(LearnedAvailabilityForecast)
        instance.compiled = self.compiled

        def parent_execute(model, *args, **kwargs):
            return dict(status='UNCHANGED', forecast=self.forecast(model.engine(max_events=10000)))

        with mock.patch.object(V16Forecast, 'engine', return_value=self.engine()), \
             mock.patch.object(V16Forecast, 'execute', new=parent_execute):
            result = instance.execute('same-plan', 'same-candidate')
        self.assertEqual(result['status'], 'UNCHANGED')
        self.assertEqual(result['queue_diagnostic_sidecar']['full_public_cohort'], self.rows)
        self.assertEqual(result['queue_diagnostic_sidecar']['now_ns'], self.BASE)
        self.assertEqual(result['queue_diagnostic_instrumentation_version'], 19)


if __name__ == '__main__':
    unittest.main()
