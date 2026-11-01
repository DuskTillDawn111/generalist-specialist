import unittest
from dataclasses import replace
from fractions import Fraction as F

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import (
    Macro, Mark, RequestPlan, simulate,
)


def ns(value):
    return F(value['numerator'], value['denominator'])


def request(identity='a', owner='s0', arrival=0):
    return dict(request_id=identity, owner=owner, task='text', pool_id='known', arrival_ns=arrival)


def plan(work=10, *, producer='worker', result_delay=0, ready_delay=0):
    return RequestPlan((Macro('final', 'prefill', F(work), (
        Mark('begin', F(0), F(0), 'phase', producer, phase='prefill'),
        Mark('native_end', F(work), F(0), 'clear', producer, clear_scope='request'),
        Mark('result', F(work), F(result_delay), 'result', producer),
    )),), F(ready_delay), 'ready-transport')


def run(rows, values, *, cutoff=None, rates=None, fifo=(), drain=100):
    plans = {(r['owner'], r['task'], r['pool_id']): values[r['owner']] for r in rows}
    return simulate(rows, plans, rates or {('prefill', n, n): F(1,n+1) for n in range(5)},
        window_start_ns=0, window_end_ns=80, drain_end_ns=drain,
        cutoff_ns=cutoff, producer_fifo=fifo)


class MacroQueueReferenceTests(unittest.TestCase):
    def test_all_future_arrivals_change_response_only_after_arrival(self):
        rows = [request(), request('b', 's1', 5)]
        result = run(rows, {'s0':plan(producer='s0'), 's1':plan(producer='s1')})
        self.assertEqual(ns(result['RESULT_receipts']['a']), 15)
        self.assertEqual(ns(result['RESULT_receipts']['b']), 20)
        prefix = run(rows, {'s0':plan(producer='s0'), 's1':plan(producer='s1')}, cutoff=4)
        self.assertEqual(ns(prefix['active_response_states']['s0']['virtual_age_ns']), 4)
        self.assertEqual(prefix['future_public_arrivals'][0]['request_id'], 'b')

    def test_four_owners_run_without_global_cap_and_full_latencies_retained(self):
        rows = [request(str(i), 's'+str(i)) for i in range(4)]
        result = run(rows, {r['owner']:plan(producer=r['owner']) for r in rows})
        self.assertEqual(result['maximum_parallel_handlers'], 4)
        self.assertIsNone(result['global_admission_cap'])
        self.assertEqual([ns(x) for x in result['task_E2E_latency_arrays_ns']['text']], [40]*4)
        self.assertEqual(result['offered_requests'], 4)

    def test_delayed_BEGIN_changes_peer_evidence_at_processing_not_occurrence(self):
        slow_notice = plan(producer='s1')
        macro = slow_notice.macros[0]
        slow_notice = replace(slow_notice, macros=(replace(macro,
            marks=(replace(macro.marks[0], processing_delay_ns=F(8)), *macro.marks[1:])),))
        rows = [request(), request('b', 's1')]
        result = run(rows, {'s0':plan(producer='s0'), 's1':slow_notice}, cutoff=9)
        self.assertEqual(ns(result['active_response_states']['s0']['virtual_age_ns']), F(17,2))
        self.assertEqual(ns(result['active_response_states']['s1']['virtual_age_ns']), F(9,2))
        self.assertEqual(result['visible_macro_evidence']['s1']['phase'], 'prefill')

    def test_query_success_does_not_end_macro_and_pending_notice_crosses_boundary(self):
        p = RequestPlan((
            Macro('prefill','prefill',F(10),(
                Mark('begin',F(0),F(0),'phase','events',phase='prefill'),
                Mark('q',F(8),F(5),'query','query',event_id='prefill-end',
                    stream_id='stream0',query_complete=True),)),
            Macro('last','decode',F(20),(
                Mark('decode',F(0),F(0),'phase','events',phase='decode'),
                Mark('clear',F(20),F(0),'clear','events',clear_scope='request'),
                Mark('result',F(20),F(0),'result','events'),)),
        ),F(0),'events')
        rates = {('prefill',0,0):F(1),('decode',0,0):F(1)}
        a = run([request()], {'s0':p}, cutoff=11, rates=rates)
        self.assertEqual(a['pending_notifications'][0]['body']['identity'], 'q')
        b = run([request()], {'s0':p}, cutoff=14, rates=rates)
        self.assertEqual(b['active_response_states']['s0']['macro_id'], 'last')
        self.assertEqual(ns(b['active_response_states']['s0']['virtual_age_ns']),4)
        self.assertEqual(b['visible_macro_evidence']['s0']['phase'],'decode')
        self.assertNotIn('a',b['RESULT_receipts'])

    def test_READY_can_release_next_handler_with_old_RESULT_still_pending(self):
        # This fixture deliberately declares separate transports. Same-producer
        # RESULT-before-READY FIFO would prohibit this particular joint world.
        p = plan(result_delay=20)
        result = run([request(),request('b',arrival=1)], {'s0':p}, cutoff=11)
        self.assertEqual(ns(result['starts']['b']),10)
        self.assertEqual(result['active_response_states']['s0']['request_id'],'b')
        self.assertEqual(result['pending_notifications'][0]['body']['request_id'],'a')
        self.assertFalse(result['main_queue_provider_eligible'])

    def test_ready_delay_preserves_owner_FIFO_without_global_admission(self):
        result = run([request(),request('b',arrival=1)], {'s0':plan(ready_delay=5)})
        self.assertEqual(ns(result['starts']['b']),15)
        self.assertEqual(ns(result['RESULT_receipts']['b']),25)
        self.assertEqual([ns(x) for x in result['task_E2E_latency_arrays_ns']['text']], [10,24])

    def test_explicit_FIFO_joint_contradiction_is_rejected_not_clamped(self):
        p = replace(plan(result_delay=20), ready_producer_id='worker')
        with self.assertRaisesRegex(UnknownEvidence,'PRODUCER_FIFO'):
            run([request()], {'s0':p}, fifo=('worker',))
        # Without a certified FIFO contract the world remains an explicit
        # alternative hypothesis; the reference does not invent one.
        result = run([request()], {'s0':p}, cutoff=11)
        self.assertIn('a',result['READY_occurrences'])

    def test_drain_does_not_drop_unfinished_or_extend_original_horizon(self):
        result = run([request()], {'s0':plan(200)}, drain=90)
        self.assertEqual(result['unfinished_request_ids'],['a'])
        self.assertEqual(ns(result['cutoff_ns']),90)
        self.assertEqual(ns(result['active_response_states']['s0']['virtual_age_ns']),90)
        self.assertEqual(result['six_item_resource_predictions'],'UNKNOWN_NO_RESOURCE_PROVIDER')

    def test_virtual_age_conserved_across_notifications_no_resampling(self):
        result = run([request(),request('b','s1',5)],
            {'s0':plan(producer='s0'), 's1':plan(producer='s1')})
        for identity in ('a','b'):
            pieces = [p for p in result['response_exposure'] if p['request_id']==identity]
            self.assertEqual(sum((ns(p['right_ns'])-ns(p['left_ns']))*ns(p['rate']) for p in pieces),10)
            self.assertTrue(all(ns(a['new_virtual_age_ns'])==ns(b['previous_virtual_age_ns'])
                for a,b in zip(pieces,pieces[1:])))

    def test_old_macro_clear_does_not_clear_next_decode_macro(self):
        p = RequestPlan((
            Macro('pre','prefill',F(10),(
                Mark('pre_begin',F(0),F(0),'phase','pre',phase='prefill'),
                Mark('pre_end',F(10),F(5),'clear','pre',clear_scope='macro'),)),
            Macro('dec','decode',F(20),(
                Mark('dec_begin',F(0),F(0),'phase','dec',phase='decode'),
                Mark('result',F(20),F(0),'result','dec'),)),
        ),F(0),'ready')
        result = run([request()], {'s0':p}, cutoff=16,
            rates={('prefill',0,0):F(1),('decode',0,0):F(1)})
        self.assertEqual(result['visible_macro_evidence']['s0']['macro_id'],'dec')

    def test_true_query_occurring_in_ready_tail_retains_source_event_identity(self):
        p = RequestPlan((
            Macro('final','prefill',F(10),()),
            Macro('tail','ready_tail',F(5),(
                Mark('late_query',F(2),F(1),'query','query',event_id='final:stream-end',
                    stream_id='s0',query_complete=True),
                Mark('result',F(5),F(0),'result','result'),)),
        ),F(0),'ready')
        result = run([request()], {'s0':p}, cutoff=14,
            rates={('prefill',0,0):F(1),('ready_tail',0,0):F(1)})
        query = next(x for x in result['trace'] if x['kind']=='processed_notification')
        self.assertEqual(query['event_id'],'final:stream-end')
        self.assertEqual(query['macro_id'],'tail')
        self.assertEqual(result['active_response_states']['s0']['macro_id'],'tail')

    def test_old_request_BEGIN_cannot_silently_overwrite_new_request_phase(self):
        p = plan(); m=p.macros[0]
        p=replace(p,macros=(replace(m,marks=(replace(m.marks[0],processing_delay_ns=F(15)),
            *m.marks[1:])),))
        # At 15 the first delayed BEGIN would overwrite the current second
        # handler's later-issued visible phase in an unqualified multi-producer
        # world; give second handler a different public template.
        rows=[request(),dict(request('b',arrival=1),pool_id='second')]
        plans={('s0','text','known'):p,('s0','text','second'):plan(producer='second')}
        with self.assertRaisesRegex(UnknownEvidence,'STALE_PHASE'):
            simulate(rows,plans,{('prefill',0,0):F(1)},
                window_start_ns=0,window_end_ns=80,drain_end_ns=100)

    def test_unknown_input_rate_and_invalid_mark_support_rejected(self):
        with self.assertRaisesRegex(UnknownEvidence,'PUBLIC_INPUT'):
            simulate([request()],{}, {},window_start_ns=0,window_end_ns=80,drain_end_ns=100)
        with self.assertRaisesRegex(UnknownEvidence,'RESPONSE_RATE'):
            run([request()], {'s0':plan()}, rates={('prefill',1,1):F(1)})
        p=plan();m=p.macros[0]
        invalid=replace(p,macros=(replace(m,marks=(replace(m.marks[0],work_ns=F(11)),*m.marks[1:])),))
        with self.assertRaises(ValueError):run([request()],{'s0':invalid})

    def test_distinct_prechosen_draws_consumed_by_public_ordinal_not_request_identity(self):
        rows=[request(),request('b',arrival=1)]
        values={('s0','text','known'):(plan(10),plan(20))}
        kwargs=dict(window_start_ns=0,window_end_ns=80,drain_end_ns=100)
        rates={('prefill',0,0):F(1)}
        result=simulate(rows,values,rates,**kwargs)
        self.assertEqual(ns(result['RESULT_receipts']['b']),30)
        renamed=[dict(row,request_id='new'+str(i)) for i,row in enumerate(rows)]
        changed=simulate(renamed,values,rates,**kwargs)
        self.assertEqual(ns(changed['RESULT_receipts']['new1']),30)
        self.assertEqual([r['public_key_dispatch_ordinal'] for r in
            result['consumed_public_key_joint_draws']],[0,1])
        with self.assertRaisesRegex(UnknownEvidence,'ONE_JOINT_DRAW'):
            simulate(rows,{('s0','text','known'):(plan(),)},rates,**kwargs)

    def test_E2E_RESULT_reader_endpoint_precedes_processing(self):
        p=plan(result_delay=8,ready_delay=10);m=p.macros[0]
        p=replace(p,macros=(replace(m,marks=(*m.marks[:2],
            replace(m.marks[2],reader_delay_ns=F(2)))),))
        result=run([request()],{'s0':p},cutoff=15)
        self.assertEqual(ns(result['RESULT_receipts']['a']),12)
        self.assertNotIn('a',result['RESULT_processed'])
        self.assertEqual([ns(x) for x in result['task_E2E_latency_arrays_ns']['text']],[12])
        self.assertEqual(result['unfinished_request_ids'],[])
        self.assertTrue(result['pending_notifications'])

    def test_dispatch_to_worker_START_delay_keeps_pending_start_and_owner_FIFO(self):
        p=replace(plan(),dispatch_to_worker_start_ns=F(3))
        rows=[request(),request('b',arrival=1)]
        prefix=run(rows,{'s0':p},cutoff=2)
        self.assertEqual(ns(prefix['parent_dispatches']['a']),0)
        self.assertNotIn('a',prefix['starts'])
        self.assertFalse(prefix['active_response_states'])
        self.assertEqual(ns(prefix['pending_worker_starts'][0]['at_ns']),3)
        full=run(rows,{'s0':p})
        self.assertEqual(ns(full['starts']['a']),3)
        self.assertEqual(ns(full['parent_dispatches']['b']),13)
        self.assertEqual(ns(full['starts']['b']),16)
        self.assertEqual(ns(full['RESULT_receipts']['b']),26)
        self.assertFalse(full['dispatch_contract_matches_original_sender'])

    def test_joint_reader_FIFO_and_clock_inversion_rejected(self):
        p=plan(result_delay=20);m=p.macros[0]
        marks=(replace(m.marks[0],processing_delay_ns=F(20),reader_delay_ns=F(15)),
            replace(m.marks[2],reader_delay_ns=F(0)))
        p=replace(p,macros=(replace(m,marks=marks),))
        with self.assertRaisesRegex(UnknownEvidence,'READER_FIFO'):
            run([request()],{'s0':p},fifo=('worker',))
        bad=replace(p,ready_reader_delay_ns=F(2))
        with self.assertRaises(ValueError):run([request()],{'s0':bad})


if __name__ == '__main__':
    unittest.main()
