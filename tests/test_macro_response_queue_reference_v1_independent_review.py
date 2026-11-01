"""Independent exact-world counterexamples and cutoff/FIFO conservation checks."""
import unittest
import copy
from dataclasses import replace
from fractions import Fraction as F

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Macro,Mark,RequestPlan,simulate


def val(x):return F(x['numerator'],x['denominator'])


def request(identity='a',owner='x',arrival=0):
    return dict(request_id=identity,owner=owner,task='text',pool_id='p',arrival_ns=arrival)


def run(rows,templates,cutoff=100,rates=None,fifo=()):
    plans={(r['owner'],'text','p'):templates[r['owner']] for r in rows}
    return simulate(rows,plans,rates or {('prefill',0,0):F(1),('decode',0,0):F(1)},
        window_start_ns=0,window_end_ns=80,drain_end_ns=100,cutoff_ns=cutoff,producer_fifo=fifo)


def simple(work=10):
    return RequestPlan((Macro('m','prefill',F(work),(
        Mark('begin',F(0),F(0),'phase','phase',phase='prefill'),
        Mark('result',F(work),F(0),'result','result'),)),),F(0),'ready')


class MacroQueueIndependentReviewTests(unittest.TestCase):
    def test_reader_completion_at_cutoff_survives_unprocessed_result_and_ready(self):
        p=simple();m=p.macros[0]
        p=replace(p,macros=(replace(m,marks=(m.marks[0],replace(m.marks[1],
            reader_delay_ns=F(2),processing_delay_ns=F(20)))),),
            ready_reader_delay_ns=F(3),ready_processing_delay_ns=F(30))
        result=run([request(),request('queued',arrival=1)],{'x':p},cutoff=12)
        self.assertEqual(val(result['RESULT_receipts']['a']),12)
        self.assertEqual(result['unfinished_request_ids'],['queued'])
        self.assertFalse(result['RESULT_processed'])
        self.assertEqual(result['occupied_owners'],{'x':'a'})
        self.assertEqual(result['per_owner_FIFO']['x'],['queued'])
        self.assertEqual([val(v) for v in result['task_E2E_latency_arrays_ns']['text']],[12])

    def test_phase_and_query_reader_receipt_does_not_change_processing_visibility(self):
        p=RequestPlan((Macro('m','prefill',F(40),(
            Mark('begin',F(0),F(10),'phase','x',phase='prefill',reader_delay_ns=F(1)),
            Mark('query',F(2),F(10),'query','x',event_id='end',stream_id='stream',
                query_complete=True,reader_delay_ns=F(1)),
            Mark('result',F(40),F(0),'result','r'))),),F(0),'ready')
        prefix=run([request()],{'x':p},cutoff=3)
        self.assertFalse(prefix['visible_macro_evidence'])
        self.assertFalse([r for r in prefix['trace'] if r['kind']=='processed_notification'])
        after=run([request()],{'x':p},cutoff=13)
        self.assertEqual(after['visible_macro_evidence']['x']['phase'],'prefill')
        self.assertEqual(after['active_response_states']['x']['macro_id'],'m')
        self.assertFalse(after['RESULT_receipts'])

    def test_READY_reader_does_not_release_owner_before_processing(self):
        p=replace(simple(),ready_reader_delay_ns=F(1),ready_processing_delay_ns=F(10))
        rows=[request(),request('b',arrival=1)]
        prefix=run(rows,{'x':p},cutoff=15)
        self.assertEqual(prefix['occupied_owners'],{'x':'a'})
        self.assertNotIn('b',prefix['parent_dispatches'])
        ready=next(n['body'] for n in prefix['pending_notifications'] if n['body']['action']=='ready')
        self.assertEqual(val(ready['reader_receipt_ns']),11)
        full=run(rows,{'x':p})
        self.assertEqual(val(full['parent_dispatches']['b']),20)

    def test_handoff_cutoff_retains_draw_and_excludes_unstarted_macro_exposure(self):
        p=replace(simple(),dispatch_to_worker_start_ns=F(7,2))
        rows=[request('a',arrival=2),request('queued',arrival=3)]
        prefix=run(rows,{'x':p},cutoff=5)
        self.assertFalse(prefix['starts']);self.assertFalse(prefix['response_exposure'])
        self.assertEqual(val(prefix['pending_worker_starts'][0]['at_ns']),F(11,2))
        self.assertEqual(len(prefix['consumed_public_key_joint_draws']),1)
        full=run(rows,{'x':p})
        self.assertEqual(val(full['RESULT_receipts']['a']),F(31,2))
        self.assertEqual(val(full['task_E2E_latency_arrays_ns']['text'][0]),F(27,2))
        self.assertFalse(full['dispatch_contract_matches_original_sender'])

    def test_ready_reader_FIFO_is_checked_separately_from_processing_FIFO(self):
        p=RequestPlan((Macro('m','prefill',F(10),(
            Mark('result',F(10),F(20),'result','worker',reader_delay_ns=F(5)),)),),
            F(30),'worker',ready_reader_delay_ns=F(1))
        with self.assertRaisesRegex(UnknownEvidence,'READER_FIFO'):
            run([request()],{'x':p},fifo=('worker',))

    def test_reader_before_occurrence_and_reader_after_processing_are_rejected(self):
        p=simple();m=p.macros[0]
        for reader in (F(-1),F(2)):
            with self.subTest(reader=reader),self.assertRaises(ValueError):
                bad=replace(p,macros=(replace(m,marks=(m.marks[0],replace(m.marks[1],
                    processing_delay_ns=F(1),reader_delay_ns=reader))),))
                run([request()],{'x':bad})

    def test_public_draw_ordinal_follows_arrival_fifo_not_request_input_order(self):
        rows=[request('later',arrival=5),request('earlier',arrival=0)]
        values={('x','text','p'):(simple(10),simple(20))}
        result=simulate(rows,values,{('prefill',0,0):F(1)},window_start_ns=0,window_end_ns=80,drain_end_ns=100)
        self.assertEqual([(r['request_id'],r['public_key_dispatch_ordinal']) for r in result['consumed_public_key_joint_draws']],
            [('earlier',0),('later',1)])
        self.assertEqual(val(result['RESULT_receipts']['earlier']),10)
        self.assertEqual(val(result['RESULT_receipts']['later']),30)

    def test_public_draw_cursors_are_separate_for_interleaved_pool_keys(self):
        rows=[request('p0'),dict(request('q0',arrival=1),pool_id='q'),request('p1',arrival=2)]
        values={('x','text','p'):(simple(3),simple(5)),('x','text','q'):(simple(7),)}
        result=simulate(rows,values,{('prefill',0,0):F(1)},window_start_ns=0,window_end_ns=80,drain_end_ns=100)
        self.assertEqual([(r['pool_id'],r['public_key_dispatch_ordinal']) for r in result['consumed_public_key_joint_draws']],
            [('p',0),('q',0),('p',1)])
        self.assertEqual({k:val(v) for k,v in result['RESULT_receipts'].items()},{'p0':3,'q0':10,'p1':15})

    def test_cutoff_does_not_consume_queued_or_future_joint_draws_or_mutate_inputs(self):
        rows=[request('a'),request('queued',arrival=1),request('future',arrival=60)]
        values={('x','text','p'):(simple(10),simple(20),simple(30))}
        rates={('prefill',0,0):F(1)}
        originals=copy.deepcopy((rows,values,rates))
        prefix=simulate(rows,values,rates,window_start_ns=0,window_end_ns=80,drain_end_ns=100,cutoff_ns=5)
        self.assertEqual(len(prefix['consumed_public_key_joint_draws']),1)
        self.assertEqual(prefix['per_owner_FIFO']['x'],['queued'])
        self.assertEqual(prefix['future_public_arrivals'][0]['request_id'],'future')
        self.assertEqual((rows,values,rates),originals)
        again=simulate(rows,values,rates,window_start_ns=0,window_end_ns=80,drain_end_ns=100,cutoff_ns=5)
        self.assertEqual(prefix,again)
        full=simulate(rows,values,rates,window_start_ns=0,window_end_ns=80,drain_end_ns=100)
        self.assertEqual([r['public_key_dispatch_ordinal'] for r in full['consumed_public_key_joint_draws']],[0,1,2])
        self.assertEqual((rows,values,rates),originals)

    def test_joint_draw_count_checked_against_full_offered_cohort_at_prefix(self):
        rows=[request('a'),request('future',arrival=60)]
        for draws in ((),(simple(),),(simple(),simple(),simple())):
            with self.subTest(draws=len(draws)),self.assertRaises(UnknownEvidence):
                simulate(rows,{('x','text','p'):draws},{('prefill',0,0):F(1)},
                    window_start_ns=0,window_end_ns=80,drain_end_ns=100,cutoff_ns=5)

    def test_explicit_macro_clear_cannot_remove_visible_later_decode(self):
        p=RequestPlan((
            Macro('pre','prefill',F(5),(
                Mark('pre_begin',F(0),F(0),'phase','pre',phase='prefill'),
                Mark('pre_end',F(5),F(10),'clear','pre',clear_scope='macro'))),
            Macro('dec','decode',F(30),(
                Mark('dec_begin',F(0),F(0),'phase','dec',phase='decode'),
                Mark('result',F(30),F(0),'result','result')))),F(0),'ready')
        result=run([request()],{'x':p},cutoff=16)
        self.assertEqual(result['visible_macro_evidence']['x']['macro_id'],'dec')

    def test_processed_terminal_clear_leaves_watermark_for_older_begin(self):
        p=RequestPlan((Macro('m','prefill',F(30),(
            Mark('begin',F(0),F(15),'phase','begin',phase='prefill'),
            Mark('native_end',F(10),F(0),'clear','end',clear_scope='request'),
            Mark('result',F(30),F(0),'result','result'))),),F(0),'ready')
        with self.assertRaisesRegex(UnknownEvidence,'STALE|ORDER|REORDER|CAUSAL'):
            run([request()],{'x':p},cutoff=16)

    def test_equal_occurrence_late_old_phase_cannot_resurrect_after_clear(self):
        p=RequestPlan((Macro('m','prefill',F(30),(
            Mark('begin',F(0),F(15),'phase','begin',phase='prefill'),
            Mark('native_end',F(0),F(0),'clear','end',clear_scope='request'),
            Mark('result',F(30),F(0),'result','result'))),),F(0),'ready')
        with self.assertRaisesRegex(UnknownEvidence,'STALE|ORDER|REORDER|CAUSAL'):
            run([request()],{'x':p},cutoff=16)

    def test_old_macro_delayed_phase_cannot_overwrite_new_macro(self):
        plan=RequestPlan((
            Macro('first','prefill',F(10),(
                Mark('old_begin',F(0),F(15),'phase','oldphase',phase='prefill'),)),
            Macro('second','decode',F(30),(
                Mark('new_begin',F(0),F(0),'phase','newphase',phase='decode'),
                Mark('result',F(30),F(0),'result','result'),)),),F(0),'ready')
        with self.assertRaisesRegex(UnknownEvidence,'STALE|ORDER|REORDER|CAUSAL'):
            run([request()],{'x':plan},cutoff=16)

    def test_old_request_phase_after_ready_cannot_replace_new_request(self):
        p=replace(simple(),macros=(replace(simple().macros[0],marks=(
            Mark('begin',F(0),F(15),'phase','phase',phase='prefill'),
            Mark('result',F(10),F(0),'result','result'))),))
        with self.assertRaisesRegex(UnknownEvidence,'STALE|ORDER|REORDER|CAUSAL'):
            run([request(),request('b',arrival=1)],{'x':p},cutoff=16)

    def test_actual_macro_completion_not_delayed_until_begin_processing(self):
        p=simple();p=replace(p,macros=(replace(p.macros[0],marks=(
            replace(p.macros[0].marks[0],processing_delay_ns=F(30)),p.macros[0].marks[1])),))
        result=run([request()],{'x':p},cutoff=11)
        self.assertEqual(val(result['RESULT_receipts']['a']),10)
        self.assertEqual(val(result['READY_occurrences']['a']),10)
        self.assertFalse(result['active_response_states'])
        self.assertEqual(result['pending_notifications'][0]['body']['identity'],'begin')

    def test_same_transport_fifo_rejects_inverted_phase_availability(self):
        p=RequestPlan((Macro('m','prefill',F(10),(
            Mark('begin',F(0),F(20),'phase','worker',phase='prefill'),
            Mark('result',F(10),F(0),'result','worker'))),),F(0),'worker')
        with self.assertRaisesRegex(UnknownEvidence,'PRODUCER_FIFO'):
            run([request()],{'x':p},fifo=('worker',))

    def test_future_offered_request_retained_at_cutoff_and_prefix_invariant(self):
        rows=[request(),request('b','y',60)]
        one=run(rows,{'x':simple(50),'y':simple()},cutoff=5)
        two=run([request()],{'x':simple(50)},cutoff=5)
        self.assertEqual(one['starts'],two['starts'])
        self.assertEqual(one['response_exposure'],two['response_exposure'])
        self.assertEqual(one['offered_requests'],2)
        self.assertEqual(one['future_public_arrivals'][0]['request_id'],'b')
        self.assertEqual(set(one['unfinished_request_ids']),{'a','b'})

    def test_fractional_response_clock_conservation_across_macro_transition(self):
        p=RequestPlan((
            Macro('a','prefill',F(2),()),
            Macro('b','decode',F(3),(Mark('r',F(3),F(0),'result','r'),))),F(0),'ready')
        result=run([request()],{'x':p},rates={('prefill',0,0):F(3,2),('decode',0,0):F(2,3)})
        self.assertEqual(val(result['RESULT_receipts']['a']),F(35,6))
        per={}
        for row in result['response_exposure']:
            per[row['macro_id']]=per.get(row['macro_id'],F(0))+(val(row['right_ns'])-val(row['left_ns']))*val(row['rate'])
        self.assertEqual(per,{'a':F(2),'b':F(3)})

    def test_ready_processing_at_cutoff_preserves_new_handler_with_zero_exposure(self):
        result=run([request(),request('b',arrival=1)],{'x':simple()},cutoff=10)
        self.assertEqual(val(result['starts']['b']),10)
        self.assertEqual(val(result['active_response_states']['x']['virtual_age_ns']),0)
        self.assertEqual(result['unfinished_request_ids'],['b'])

    def test_late_result_pending_is_retained_after_actual_handler_drained(self):
        p=simple();p=replace(p,macros=(replace(p.macros[0],marks=(
            p.macros[0].marks[0],replace(p.macros[0].marks[1],processing_delay_ns=F(150)))),))
        result=run([request()],{'x':p})
        self.assertEqual(result['unfinished_request_ids'],['a'])
        self.assertEqual(len(result['pending_notifications']),1)
        self.assertEqual(val(result['pending_notifications'][0]['available_ns']),160)
        self.assertFalse(result['active_response_states']);self.assertFalse(result['occupied_owners'])


if __name__=='__main__':unittest.main()
