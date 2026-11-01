import copy
import json
from fractions import Fraction
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_actual_worker_twoFIFO_reference_v4 as v4_fixture
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5 as module
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as frozen
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Mark, Macro, RequestPlan
from tools.predictor_validation_20261027.worker_stdin_fifo_contract_v1 import TransportDraw
from tools.predictor_validation_20261027.implementation_source_closure_v27 import implementation_sources

decode=v4_fixture.decode
slacks=v4_fixture.slacks


def inputs(**kwargs):
    value = v4_fixture.ActualWorkerTwoFIFOTests().inputs(**kwargs)
    value['rates'] = {('owner', *key): rate for key, rate in value['rates'].items()}
    return value


def coupled_inputs():
    requests = []; plans = {}; timing = {}; transport = {}
    for owner, speed in [('text#0', Fraction(1, 2)), ('text#1', Fraction(2)), ('generalist_2#0', Fraction(1))]:
        key = owner, 'text', 'public-input'; versions = []
        for ordinal in range(2):
            producer = owner + ':stdout'
            marks = (Mark('begin', 0, 0, 'phase', producer, phase='decode'),
                Mark('query', 3, 0, 'query', producer, event_id='event'+str(ordinal), stream_id='stream', query_complete=True),
                Mark('RESULT', 7, 0, 'result', producer),
                Mark('clear', 9, 0, 'clear', producer, clear_scope='request'))
            plan = RequestPlan((Macro('decode_0000', 'decode', 10, marks),), 0, producer)
            versions.append(plan)
            timing[plan] = {m.identity: slacks(1, 3) for m in marks}
            timing[plan]['worker_READY'] = slacks(1, 3)
            requests.append(dict(request_id=owner+':r'+str(ordinal), owner=owner, task='text', pool_id='public-input', arrival_ns=ordinal))
            transport[(*key, ordinal)] = TransportDraw(0, 0, 5 if ordinal == 0 else 1, 0, 2)
        plans[key] = tuple(versions)
    rates = {(owner, 'decode', n, same): speed/Fraction(1+same)
        for owner, speed in [('text#0', Fraction(1, 2)), ('text#1', Fraction(2)), ('generalist_2#0', Fraction(1))]
        for n in range(3) for same in range(n+1)}
    owners = {r['owner'] for r in requests}
    return dict(requests=requests, plans=plans, rates=rates, window_start_ns=0, window_end_ns=2,
        drain_end_ns=200, protocol_slacks=timing, transport_draws=transport,
        initial_sender_available_ns={o: 0 for o in owners}, initial_worker_read_release_ns={o: 0 for o in owners},
        initial_protocol_state=dict(reader_free={}, collector_free=0, last_collector_arrival=0, pending_messages=()))


class ReferenceV5Tests(unittest.TestCase):
    def assert_split_equal(self, kwargs, cut):
        full = module.simulate(**kwargs)
        first = module.simulate(**dict(kwargs, cutoff_ns=cut))
        saved = json.loads(json.dumps(first['continuation_state']))
        before = copy.deepcopy(saved)
        resumed = module.simulate(**dict(kwargs, continuation_state=saved))
        self.assertEqual(saved, before)
        self.assertEqual(resumed, full)
        return decode(first)

    def test_actual_FIFO_release_and_parent_notice_non_gates_match_frozen_v4(self):
        kwargs = inputs(); result = decode(module.simulate(**kwargs))
        self.assertEqual(result['starts'], {'r0': 1, 'r1': 12})
        self.assertEqual(result['READY_occurrences'], {'r0': 11, 'r1': 22})
        self.assertEqual(result['RESULT_receipts'], {'r0': 19, 'r1': 30})
        self.assertEqual(result['RESULT_processed'], {'r0': 119, 'r1': 119})
        old = decode(frozen.simulate(**v4_fixture.ActualWorkerTwoFIFOTests().inputs()))
        for field in ('trace', 'starts', 'READY_occurrences', 'RESULT_receipts', 'per_owner_FIFO', 'transport_schedule'):
            self.assertEqual(result[field], old[field])
        self.assertFalse(result['READY_processing_gates_worker_START'])

    def test_split_exact_at_active_pending_sender_START_reader_processing_and_READY_boundaries(self):
        for write in (1, 20):
            for cut in (0, Fraction(1, 2), 1, 5, 9, 10, 11, 12, 19, 20, 22, 30, 118, 119, 150):
                with self.subTest(write=write, cut=cut): self.assert_split_equal(inputs(write=write), cut)
        first = self.assert_split_equal(inputs(write=20, worker=7), 5)
        payload = first['continuation_state']['payload']
        self.assertTrue(payload['worker_starts'])
        self.assertTrue(payload['sender_events'])

    def test_resume_JSON_with_initial_fields_omitted_and_multiple_pauses(self):
        kwargs = inputs(); full = module.simulate(**kwargs); state = None
        for cutoff in (3, 10, 20, 21, 45, 119, 150):
            supplied = dict(kwargs, cutoff_ns=cutoff)
            if state is not None:
                for key in ('initial_sender_available_ns', 'initial_worker_read_release_ns', 'initial_protocol_state'): supplied.pop(key)
                supplied['continuation_state'] = json.loads(json.dumps(state))
            state = module.simulate(**supplied)['continuation_state']
        self.assertEqual(state, full['continuation_state'])

    def test_owner_four_key_rates_preserve_same_semantic_phase_and_coupled_resume(self):
        kwargs = coupled_inputs(); full = decode(module.simulate(**kwargs))
        positive = [r for r in full['response_exposure'] if r['evidence_key'][3] > 0]
        self.assertTrue(positive)
        self.assertTrue(all(r['actual_phase'] == 'decode' and r['evidence_key'][1] == 'decode' for r in positive))
        self.assertTrue(any(r['evidence_key'][3] == 2 for r in positive))
        self.assertNotEqual(full['READY_occurrences']['text#0:r0'], full['READY_occurrences']['text#1:r0'])
        for cut in (0, 1, 2, 4, 8, 12, Fraction(35, 2), 35, 80, 199, 200):
            with self.subTest(cut=cut): self.assert_split_equal(kwargs, cut)

    def test_complete_nonempty_state_preserves_producer_lifetime_and_virtual_age(self):
        first = decode(module.simulate(**dict(coupled_inputs(), cutoff_ns=12)))
        payload = first['continuation_state']['payload']
        self.assertTrue(payload['active']); self.assertTrue(payload['visible'])
        self.assertTrue(payload['producer_next']); self.assertTrue(payload['owner_issued'])
        self.assertTrue(payload['processed_issuance_watermarks']); self.assertTrue(payload['phase_watermarks'])
        self.assertTrue(payload['notifications'] or payload['protocol_readers'])
        self.assertEqual(set(payload), module.STATE_FIELDS)
        self.assertTrue(any(r['age'] > 0 for r in payload['active'].values()))

    def test_missing_unknown_transport_and_partial_or_future_initial_never_zero_filled(self):
        kwargs = inputs()
        for field in ('transport_draws', 'initial_sender_available_ns', 'initial_worker_read_release_ns', 'initial_protocol_state'):
            with self.subTest(field=field), self.assertRaises(UnknownEvidence): module.simulate(**dict(kwargs, **{field: None}))
        first = module.simulate(**dict(kwargs, cutoff_ns=5)); checkpoint = first['continuation_state']
        for field in module.STATE_FIELDS:
            state = copy.deepcopy(checkpoint); del state['payload'][field]
            with self.subTest(field=field), self.assertRaises(UnknownEvidence): module.simulate(**dict(kwargs, continuation_state=state))
        future = module.simulate(**kwargs)['continuation_state']
        with self.assertRaisesRegex(UnknownEvidence, 'FUTURE_OR_OUTSIDE'): module.simulate(**dict(kwargs, cutoff_ns=5, continuation_state=future))
        state = copy.deepcopy(checkpoint); state['payload']['ready_occurrences']['r0'] = {'numerator': 11, 'denominator': 1}
        with self.assertRaisesRegex(UnknownEvidence, 'FUTURE_COMPLETED'): module.simulate(**dict(kwargs, continuation_state=state))

    def test_changed_world_or_initial_station_binding_and_three_key_rate_rejected(self):
        kwargs = inputs(); state = module.simulate(**dict(kwargs, cutoff_ns=5))['continuation_state']
        changes = [dict(rates={('owner', 'work', 0, 0): 2}), dict(drain_end_ns=151),
                   dict(initial_worker_read_release_ns={'owner': 1}),
                   dict(transport_draws={('owner', 'text', 'input', i): TransportDraw(0, 0, 1, 1, 1) for i in range(2)})]
        for change in changes:
            with self.subTest(change=change), self.assertRaises(UnknownEvidence): module.simulate(**dict(kwargs, continuation_state=state, **change))
        with self.assertRaises(ValueError): module.simulate(**dict(kwargs, rates={('work', 0, 0): 1}))

    def test_unissued_producer_state_future_trace_and_invalid_active_cursor_rejected(self):
        kwargs = coupled_inputs(); state = module.simulate(**dict(kwargs, cutoff_ns=12))['continuation_state']
        for mode in ('producer', 'issuance', 'trace', 'cursor', 'duplicate'):
            changed = copy.deepcopy(state); payload = changed['payload']
            if mode == 'producer': payload['producer_next'].clear()
            elif mode == 'issuance': payload['owner_issued'].clear()
            elif mode == 'trace': payload['trace'].append(dict(kind='request_start', at_ns=13))
            elif mode == 'cursor': next(iter(payload['active'].values()))['next_mark'] = 100
            else:
                owner, rid = next(iter(payload['occupied'].items()))
                payload['pending'].setdefault(owner, []).append(rid)
            with self.subTest(mode=mode), self.assertRaises(UnknownEvidence): module.simulate(**dict(kwargs, continuation_state=changed))

    def test_multimacro_boundaries_query_and_tail_restore_without_future_mark_replay(self):
        kwargs = inputs(); key = ('owner', 'text', 'input'); versions = []; timing = {}
        for ordinal in range(2):
            producer = 'owner'
            plan = RequestPlan((
                Macro('entry', 'entry', 2, (Mark('entry-begin', 0, 0, 'phase', producer, phase='entry'),)),
                Macro('decode', 'decode', 5, (
                    Mark('decode-begin', 0, 0, 'phase', producer, phase='decode'),
                    Mark('query', 2, 0, 'query', producer, event_id='q'+str(ordinal), stream_id='stream', query_complete=False))),
                Macro('tail', 'ready_tail', 3, (
                    Mark('clear', 0, 0, 'clear', producer, clear_scope='request'),
                    Mark('result', 1, 0, 'result', producer)))), 0, producer)
            versions.append(plan); timing[plan] = {m.identity: slacks(1, 2) for macro in plan.macros for m in macro.marks}
            timing[plan]['worker_READY'] = slacks(1, 2)
        kwargs.update(plans={key: tuple(versions)}, protocol_slacks=timing,
            rates={('owner', phase, 0, 0): rate for phase, rate in [('entry', Fraction(2)), ('decode', Fraction(1, 2)), ('ready_tail', Fraction(3, 2))]})
        for cut in (1, 2, 3, 6, 8, 12, 13, 14, 17, 25):
            with self.subTest(cut=cut): self.assert_split_equal(kwargs, cut)
        missing = dict(kwargs); missing['rates'] = {k: v for k, v in kwargs['rates'].items() if k[1] != 'decode'}
        with self.assertRaisesRegex(UnknownEvidence, 'UNKNOWN_VISIBLE_MACRO_RESPONSE_RATE'): module.simulate(**missing)

    def test_immutable_inputs_v5_resource_adapter_and_collector_reverse_closure(self):
        kwargs = inputs(); before = copy.deepcopy(kwargs); result = module.simulate(**kwargs)
        self.assertEqual(kwargs, before)
        view = module.resource_timeline_view(result)
        self.assertEqual(view['origin_schema'], module.SCHEMA)
        self.assertFalse(view['schema_adapter_confers_native_qualification'])
        for actual, adapted in zip(result['response_exposure'], view['response_exposure']):
            self.assertEqual(adapted['owner_phase_rate_key'], actual['evidence_key'])
            self.assertEqual(adapted['evidence_key'], actual['evidence_key'][1:])
        from tools.predictor_validation_20261027.macro_resource_timeline_v1 import timeline
        resource=decode(timeline(view,kwargs['requests'],resident_owners=('owner',)))
        self.assertEqual(resource['total_wall_ns'],150)
        with self.assertRaises(ValueError): frozen.resource_timeline_view(result)
        root = Path(__file__).resolve().parents[1]
        names = {Path(r['path']).name for r in implementation_sources(root/'tools/predictor_validation_20261027/twoFIFO_owned_collect_v2.py')}
        self.assertNotIn('actual_worker_twoFIFO_reference_v5.py', names)


if __name__ == '__main__': unittest.main()
