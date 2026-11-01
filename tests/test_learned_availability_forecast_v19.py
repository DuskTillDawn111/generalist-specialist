"""Full trajectories and unsupported/stalled paths with/without sidecar."""
import copy
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_compiled_static_providers_v16 as compilation_tests
from tools.predictor_validation_20261027.learned_availability_forecast_v19 import (
    FullPublicDagQueueDiagnostics, LearnedAvailabilityForecast, QueueDiagnosticEngine,
    V16Forecast, implementation_sources)


class StallScheduler:
    def decide(self, view):
        from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import SchedulingDecision
        return SchedulingDecision({node.key: 0. for node in view.active_gpu_nodes}, None)

    def provenance(self):
        return dict(kind='explicit_test_stall_no_fallback')


class V19QueueTests(unittest.TestCase):
    provider = compilation_tests.CompilationTests.provider
    compiled = compilation_tests.CompilationTests.compiled
    engine = compilation_tests.CompilationTests.engine

    def setUp(self):
        compilation_tests.CompilationTests.setUp(self)
        for alias in self.aliases.values():
            alias['family'] = 'text'
        for key, profile in self.plans.items():
            profile['nodes'].append(dict(id='parallel',kind='gpu',retained_source_metadata=['parallel']))
            profile['edges'].extend([dict(before='e',after='parallel'),dict(before='parallel',after='w')])
            self.work[(*key,'parallel')] = 13
            self.releases[(*key,'parallel')] = dict(not_before_offset_ns=2,after_predecessors_ns=0)
            self.phases[(*key,'parallel')] = 'decode'

    def rows(self, offsets=(0,4,9,17), now=1000):
        return [dict(alias=a,task='text',pool_id='short',arrival_ns=now+t,request_id=f'{a}-{t}')
                for t in offsets for a in self.aliases]

    def wrapped(self, compiled, *, stall=False):
        engine = self.engine(compiled)
        if stall:
            engine.scheduler = StallScheduler()
        return QueueDiagnosticEngine(engine, compiled, list(compiled.sources))

    def equivalent(self, old, new):
        new = copy.deepcopy(new)
        new['queue_diagnostics'] = old['queue_diagnostics']
        self.assertEqual(old,new)

    def test_all_original_outputs_identical_complete_and_drain_limited_parallel_late(self):
        compiled = self.compiled()
        rows = self.rows()
        for drain in (1040,2000):
            with self.subTest(drain=drain):
                plain = self.engine(compiled).forecast(rows,1000,1030,drain_deadline_ns=drain)
                wrapper = self.wrapped(compiled)
                observed = wrapper.forecast(rows,1000,1030,drain_deadline_ns=drain)
                self.equivalent(plain,observed)
                self.assertEqual(observed['cohort'],{r['request_id']:r for r in rows})
                self.assertGreaterEqual(observed['max_simultaneous_nodes_per_owner']['a'],2)
                self.assertTrue(all(v is None for v in observed['queue_diagnostics']['max_normalized_resource_work_ms'].values()))
                self.assertIs(wrapper._engine.duration_provider,compiled.duration)
                self.assertIs(wrapper._engine.plans,compiled.graph.plans)
                self.assertEqual(wrapper.sidecar['now_ns'],1000)
        q = observed['queue_diagnostics']
        self.assertAlmostEqual(sum(q['owner_fifo_request_wait_ms'].values()),sum(observed['queue_wait_ns'].values())/1e6)
        self.assertAlmostEqual(q['remaining_activity_work_ms'],0.)
        self.assertTrue(q['queue_recovery_episodes'])

    def test_future_supported_work_envelope_known_but_not_live_backlog_at_early_cutoff(self):
        compiled = self.compiled()
        rows = [self.rows((0,))[0],self.rows((17,))[0],self.rows((20,))[1]]
        plain = self.engine(compiled).forecast(rows,1000,1030,drain_deadline_ns=1002)
        wrapper = self.wrapped(compiled)
        observed = wrapper.forecast(rows,1000,1030,drain_deadline_ns=1002)
        self.equivalent(plain,observed)
        q = observed['queue_diagnostics']
        self.assertEqual(q['registered_request_count'],1)
        self.assertEqual(q['max_waiting_requests'],0)
        self.assertEqual(len(q['not_yet_registered_request_ids']),2)
        per_a = 11+11+11+13
        per_b = 5+5+5+13
        self.assertAlmostEqual(q['remaining_activity_work_ms'],per_a/1e6)
        envelope = q['offered_work_envelopes'][0]
        self.assertEqual(envelope['max_arrival_count'],3)
        self.assertAlmostEqual(envelope['max_offered_solo_work_ms'],(2*per_a+per_b)/1e6)
        self.assertEqual(q['zero_duration_state_snapshots_without_rate_imputation'],1)

    def test_original_six_terms_U_and_partial_score_rejection_are_identical(self):
        from tools.predictor_validation_20261027.prospective_flow_contract_v11 import score_complete_forecast
        compiled = self.compiled()
        rows = self.rows()
        plan = dict(aliases=self.aliases,window_seconds=30e-9,drain_seconds=1970e-9,
            workload_rps={'text':len(rows)/30e-9},latency_reference_ms={'text':1.},
            memory_budget_mib=100.,memory_safe_factor=1.2,energy_reference_watts=50.,
            weights={k:1/6 for k in 'QTLPME'})
        startup = dict(energy_joules=360.,peak_memory_mib=70.,energy_includes_priming=True,
            safety_factor_applied=False,startup_amortization_seconds=3600.,warmup_seconds=0.)
        for drain in (1040,2000):
            values = []
            for engine in (self.engine(compiled),self.wrapped(compiled)):
                forecast = engine.forecast(rows,1000,1030,drain_deadline_ns=drain)
                try:
                    values.append(score_complete_forecast(forecast,rows,plan,
                        {'text':{'text':.2}},startup,phase_start_ns=1000))
                except UnknownEvidence as exc:
                    values.append(str(exc))
            self.assertEqual(values[0],values[1])
            if drain == 2000:
                self.assertEqual(set(values[0]['terms']),set('QTLPME'))
                self.assertIsNotNone(values[0]['full_U'])
            else:
                self.assertEqual(values[0],'COMPLETE_DECLARED_FLOW_DAG_AND_PARENT_RECEIPTS_REQUIRED')

    def test_unknown_future_template_preserves_count_and_unknown_work(self):
        compiled = self.compiled()
        rows = self.rows((0,17))[:3]
        rows[2]['pool_id'] = 'unsupported'
        plain = self.engine(compiled).forecast(rows,1000,1030,drain_deadline_ns=1002)
        wrapper = self.wrapped(compiled)
        observed = wrapper.forecast(rows,1000,1030,drain_deadline_ns=1002)
        self.equivalent(plain,observed)
        q = observed['queue_diagnostics']
        self.assertEqual(q['full_public_request_count'],3)
        self.assertEqual(q['offered_work_envelopes'][0]['max_arrival_count'],3)
        self.assertIsNone(q['offered_work_envelopes'][0]['max_offered_solo_work_ms'])
        self.assertIn(rows[2]['request_id'],q['unknown_public_profile_by_request'])
        self.assertEqual(q['registered_request_count'],2)
        self.assertTrue(any('PUBLIC_OFFERED_PROFILE_UNKNOWN:' in s for s in q['unknown_reasons']))

    def test_arrived_unsupported_template_keeps_original_engine_unknown(self):
        compiled = self.compiled()
        rows = [dict(self.rows((0,))[0],pool_id='unsupported')]
        reasons = []
        wrapper = self.wrapped(compiled)
        for engine in (self.engine(compiled),wrapper):
            with self.assertRaises(UnknownEvidence) as caught:
                engine.forecast(rows,1000,1030,drain_deadline_ns=2000)
            reasons.append(str(caught.exception))
        self.assertEqual(reasons[0],reasons[1])
        self.assertIn('UNKNOWN_DAG_INPUT',reasons[0])
        q = wrapper.sidecar['diagnostics']
        self.assertEqual(q['registered_request_count'],1)
        self.assertIsNone(q['remaining_activity_work_ms'])
        self.assertEqual(wrapper.sidecar['original_forecast_unknown_reason'],reasons[0])

    def test_stall_retains_work_and_matches_plain_original_unknown(self):
        compiled = self.compiled()
        rows = self.rows((0,4))
        plain = self.engine(compiled)
        plain.scheduler = StallScheduler()
        wrapper = self.wrapped(compiled,stall=True)
        reasons = []
        for engine in (plain,wrapper):
            with self.assertRaises(UnknownEvidence) as caught:
                engine.forecast(rows,1000,1030)
            reasons.append(str(caught.exception))
        self.assertEqual(reasons[0],reasons[1])
        self.assertEqual(reasons[0],'PHASE_SCHEDULER_ZERO_PROGRESS_WITHOUT_TRANSITION')
        q = wrapper.sidecar['diagnostics']
        # Initial copies complete independently; every GPU/memset dependency
        # after them remains. No stalled GPU work was charged as consumption.
        self.assertGreater(q['remaining_activity_work_ms'],0)
        self.assertGreater(q['max_oldest_waiting_age_ms'],0)
        self.assertIsNotNone(q['unrecovered_backlog_since_ns'])
        self.assertEqual(q['registered_request_count'],4)

    def test_v19_engine_and_execute_entry_inject_without_real_initialization_or_rescore(self):
        compiled = self.compiled()
        instance = LearnedAvailabilityForecast.__new__(LearnedAvailabilityForecast)
        instance.compiled = compiled
        rows = self.rows((0,4))
        metrics = dict(U=.25,rho=.5,Q=.7,T=.6,L=.5,P=.4,M=.3,E=.2)

        def fake_parent_execute(self,*args,**kwargs):
            forecast = self.engine(max_events=1000).forecast(rows,1000,1030,drain_deadline_ns=2000)
            return dict(schema='unchanged_original_schema',metrics=copy.deepcopy(metrics),forecast=forecast)

        with patch.object(V16Forecast,'engine',return_value=self.engine(compiled)), \
             patch.object(V16Forecast,'execute',new=fake_parent_execute):
            result = instance.execute('unchanged','arguments')
        self.assertEqual(result['metrics'],metrics)
        self.assertEqual(result['schema'],'unchanged_original_schema')
        self.assertEqual(result['queue_diagnostic_sidecar']['full_public_cohort'],rows)
        self.assertEqual(result['queue_diagnostic_instrumentation_version'],19)
        refs = result['queue_diagnostic_sidecar']['implementation_sources']
        names = {r['path'].rsplit('/',1)[-1] for r in refs}
        self.assertTrue({'learned_availability_forecast_v19.py','dag_queue_diagnostics_v6.py',
                         'queue_diagnostics.py','dag_phase_scheduler_v8.py',
                         'compiled_static_providers_v16.py'} <= names)

    def test_public_prepass_does_not_register_or_charge_future_work(self):
        compiled = self.compiled()
        rows = self.rows((0,17))
        diagnostic = FullPublicDagQueueDiagnostics(compiled.graph.aliases,compiled.graph.plans,
            compiled.duration,cohort=rows,now_ns=1000)
        result = diagnostic.result({r['request_id']:r for r in rows})
        self.assertEqual(result['registered_request_count'],0)
        self.assertEqual(result['remaining_activity_work_ms'],0)
        self.assertEqual(result['max_waiting_requests'],0)
        self.assertGreater(result['offered_work_envelopes'][0]['max_offered_solo_work_ms'],0)
        self.assertTrue(all(v is None for v in result['max_normalized_resource_work_ms'].values()))


if __name__ == '__main__': unittest.main()
