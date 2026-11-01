"""Independent CPU checks for finite hypothetical busy-world continuation."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_twoFIFO_busy_candidate_continuation_v1 import fixture
from tests.test_twoFIFO_native_busy_state_independent_v1 import Poison
from tools.predictor_validation_20261027 import twoFIFO_busy_candidate_continuation_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_native_busy_state_v1 as native
from tools.predictor_validation_20261027.actual_worker_twoFIFO_reference_v5_2 import decode
from tools.predictor_validation_20261027.actual_worker_world_forecast_v4 import forecast


class IndependentBusyContinuationTests(unittest.TestCase):
    def test_future_result_end_and_reader_queued_bodies_stay_opaque(self):
        original, draws, args, snap = fixture()
        row = snap['request_records'][0]
        row['response_collector_processing_time_ns'] = 21
        row['worker_response'] = Poison(end_perf_ns=1)
        row['previous_handler_next_request_ready'] = Poison(collector_processing_time_ns=22)
        snap['queued_runtime_events'] = [Poison(end_perf_ns=0)]
        state = native.from_snapshot(snap, public_cohort=original['binding']['public_cohort'], cutoff_ns=20,
            owner_producers={'owner': 'owner'}, execution_context_sha256='b' * 64)
        self.assertEqual(state, original)
        self.assertEqual(m.evaluate(state, draws, **args), m.evaluate(original, draws, **args))

    def test_each_resume_receives_its_own_exact_checkpoint_and_original_calendar(self):
        state, draws, args, _ = fixture()
        calls = []
        original = m.ensemble.evaluate

        def observe(requests, catalog, **kwargs):
            if 'continuation_state' in kwargs:
                calls.append((copy.deepcopy(requests), copy.deepcopy(catalog), copy.deepcopy(kwargs)))
            return original(requests, catalog, **kwargs)

        with patch.object(m.ensemble, 'evaluate', side_effect=observe):
            result = decode(m.evaluate(state, draws, **args))
        self.assertEqual(len(calls), 2)
        for call, row, draw in zip(calls, result['draws'], draws):
            requests, singleton, keywords = call
            self.assertEqual(requests, state['binding']['public_cohort'])
            self.assertEqual(len(requests), 2)
            self.assertGreater(requests[1]['arrival_ns'], state['cutoff_ns'])
            self.assertEqual(singleton[0]['draw_id'], draw['draw_id'])
            self.assertEqual(singleton[0]['weight'], 1)
            for key in ('window_start_ns', 'window_end_ns', 'drain_end_ns'):
                self.assertEqual(keywords[key], args[key])
            self.assertEqual(decode(keywords['continuation_state']), row['candidate_checkpoint_at_cut'])
        self.assertNotEqual(calls[0][2]['continuation_state'], calls[1][2]['continuation_state'])
        self.assertEqual(result['draws'][1]['candidate_checkpoint_at_cut']['payload']['active']['owner']['age'], F(19, 2))

    def test_no_start_notice_does_not_collapse_started_candidates_to_waiting(self):
        _, draws, args, snap = fixture()
        snap['request_records'][0]['execution_events'] = []
        public = [dict(request_id='r0', owner='owner', task='text', pool_id='input', arrival_ns=0),
                  dict(request_id='r1', owner='owner', task='text', pool_id='input', arrival_ns=30)]
        state = native.from_snapshot(snap, public_cohort=public, cutoff_ns=20,
            owner_producers={'owner': 'owner'}, execution_context_sha256='b' * 64)
        out = decode(m.evaluate(state, draws, **args))
        self.assertEqual(out['world_unknown_probability_mass'], 0)
        self.assertEqual(out['draws'][0]['candidate_checkpoint_at_cut']['payload']['starts']['r0'], 1)
        self.assertTrue(out['draws'][1]['candidate_checkpoint_at_cut']['payload']['active'])
        self.assertFalse(out['start_notice_transport_modeled'])
        self.assertFalse(out['native_full_prefix_likelihood_identified'])

    def test_fixed_short_drain_keeps_real_simulated_miss_plus_unknown_mass(self):
        state, draws, args, _ = fixture()
        args['drain_end_ns'] = 40
        draws[0]['weight'], draws[1]['weight'] = F(1, 7), F(2, 7)
        draws.append(dict(draw_id='unmodeled', block_id='missing-block', weight=F(4, 7),
            inputs=m.ensemble.UnknownForecastInputs('no supported joint work draw')))
        result = decode(m.evaluate(state, draws, **args))
        self.assertEqual(result['world_unknown_probability_mass'], F(4, 7))
        risk = {r['request_id']: r['fixed_drain_incomplete_probability_bounds'] for r in result['fixed_drain_risk']}
        self.assertEqual(risk['r0'], [0, F(4, 7)])
        self.assertEqual(risk['r1'], [F(3, 7), 1])
        self.assertEqual([r['weight'] for r in result['draws']], [F(1, 7), F(2, 7), F(4, 7)])
        self.assertIsNone(result['expected_world_complete_cohort_P95_ns']['expected_value'])
        self.assertEqual(result['expected_world_complete_cohort_P95_ns']['unknown_probability_mass'], 1)
        for row in result['draws'][:2]:
            self.assertEqual(row['forecast']['scoring']['original_score']['cohort_size'], 2)
            self.assertEqual(row['forecast']['world']['unfinished_request_ids'], ['r1'])

    def test_all_unknown_catalog_still_checks_score_and_has_no_checkpoint_work(self):
        state, draws, args, _ = fixture()
        for draw in draws:
            draw['inputs'] = m.ensemble.UnknownForecastInputs('missing training support')
        with patch.object(m.engine, 'simulate', side_effect=AssertionError('must not simulate unknown')):
            out = decode(m.evaluate(state, draws, **args))
            self.assertEqual(out['world_unknown_probability_mass'], 1)
            self.assertTrue(all(r['forecast'] is None for r in out['draws']))
            self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds'] == [0, 1]
                                for r in out['fixed_drain_risk']))
            args['scoring_plan']['weights']['P'] = 2
            with self.assertRaises(ValueError):
                m.evaluate(state, draws, **args)

    def test_engine_missing_support_keeps_slot_and_implementation_failure_propagates(self):
        state, draws, args, _ = fixture()
        draws[0]['inputs']['plans'] = {}
        out = decode(m.evaluate(state, draws, **args))
        self.assertEqual(out['world_unknown_probability_mass'], F(1, 2))
        self.assertIn('UNKNOWN_PUBLIC_INPUT_RESPONSE_TEMPLATE', out['draws'][0]['reason'])
        self.assertEqual(out['draws'][1]['status'], 'GENERATED_WORLD')
        state, draws, args, _ = fixture()
        with patch.object(m.engine, 'simulate', side_effect=RuntimeError('engine bug')):
            with self.assertRaisesRegex(RuntimeError, 'engine bug'):
                m.evaluate(state, draws, **args)

    def test_prior_native_history_and_query_remain_unsupported_full_mass(self):
        state, draws, args, snap = fixture()
        prior = copy.deepcopy(snap['request_records'][0])
        prior['request_id'] = 'not-in-public-cohort'
        prior['execution_events'][0]['request_id'] = prior['request_id']
        snap['request_records'].append(prior)
        snap['pending_request_ids'].append(prior['request_id'])
        s = native.from_snapshot(snap, public_cohort=state['binding']['public_cohort'], cutoff_ns=20,
            owner_producers={'owner': 'owner'}, execution_context_sha256='b' * 64)
        out = decode(m.evaluate(s, draws, **args))
        self.assertEqual(out['world_unknown_probability_mass'], 1)
        self.assertTrue(all(r['reason'] == 'UNMODELED_NATIVE_PRIOR_HISTORY' for r in out['draws']))
        _, _, _, snap = fixture()
        snap['request_records'][0]['execution_events'].append(dict(
            request_id='r0', kind='device_event_query', event='observation', schema=native.QUERY_SCHEMA,
            perf_ns=2, received_time_ns=3, collector_dequeued_time_ns=4, collector_processing_time_ns=5))
        s = native.from_snapshot(snap, public_cohort=state['binding']['public_cohort'], cutoff_ns=20,
            owner_producers={'owner': 'owner'}, execution_context_sha256='b' * 64)
        out = decode(m.evaluate(s, draws, **args))
        self.assertEqual(out['world_unknown_probability_mass'], 1)
        self.assertTrue(all(r['reason'] == 'UNMAPPED_NATIVE_PHASE_OR_QUERY' for r in out['draws']))

    def test_generated_checkpoints_match_uninterrupted_world_when_cut_is_at_drain(self):
        _, draws, args, snap = fixture()
        # With only START observed, postpone result/ready processing beyond drain.
        # The engine can complete physical work while the collector prefix stays START-only.
        for draw in draws:
            for mapping in draw['inputs']['protocol_slacks'].values():
                for slack in mapping.values():
                    slack['collector_processing_slack_ns'] = 1000
        args['drain_end_ns'] = 35
        public = [dict(request_id='r0', owner='owner', task='text', pool_id='input', arrival_ns=0),
                  dict(request_id='r1', owner='owner', task='text', pool_id='input', arrival_ns=30)]
        state = native.from_snapshot(snap, public_cohort=public, cutoff_ns=35,
            owner_producers={'owner': 'owner'}, execution_context_sha256='b' * 64)
        result = decode(m.evaluate(state, draws, **args))
        for candidate, row in zip(draws, result['draws']):
            self.assertEqual(row['status'], 'GENERATED_WORLD')
            inputs = candidate['inputs']
            direct = decode(forecast(public, **{k: inputs[k] for k in m.ensemble.INPUT_FIELDS},
                                     **inputs['initial_state'], **args))
            self.assertEqual(row['forecast'], direct)


if __name__ == '__main__':
    unittest.main()
