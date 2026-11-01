"""Distinct residual hypotheses, full-calendar resume, explicit UNKNOWN mass."""
import copy
from fractions import Fraction as F
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_actual_worker_twoFIFO_reference_v5 import inputs, decode
from tests.test_actual_worker_world_forecast_v4 import inputs as forecast_inputs
from tests.test_twoFIFO_native_busy_state_independent_v1 import Poison
from tools.predictor_validation_20261027 import twoFIFO_busy_candidate_continuation_v1 as m
from tools.predictor_validation_20261027 import twoFIFO_native_busy_state_v1 as native
from tools.predictor_validation_20261027.actual_worker_world_forecast_v4 import forecast


def fixture():
    args = inputs(); args.pop('cutoff_ns')
    args['requests'][1]['arrival_ns'] = 30
    args['window_end_ns'] = 31
    args = forecast_inputs(args)
    public = args.pop('requests')
    initial = {k: args.pop(k) for k in m.INITIAL}
    joint = {k: args.pop(k) for k in m.ensemble.INPUT_FIELDS}
    draws = [dict(draw_id=name, block_id='CPU-fixture-'+name, weight=F(1,2),
                  inputs=dict(copy.deepcopy(joint), initial_state=copy.deepcopy(initial)))
             for name in ('ended', 'active')]
    draws[1]['inputs']['rates'] = {k:F(1,2) for k in joint['rates']}
    row = dict(request_id='r0', model='owner', scheduled_time_ns=0, enqueued_time_ns=0,
        actual_submit_time_ns=0, submit_write_completed_time_ns=1,
        execution_events=[dict(request_id='r0',pid=41,perf_ns=1,received_time_ns=2,
            collector_dequeued_time_ns=3,collector_processing_time_ns=4)])
    snap = dict(snapshot_time_ns=150, request_records=[row], pending_request_ids=['r0'],
        ready={'owner':dict(pid=41,spec_sha256='a'*64)}, queued_runtime_events=[Poison()],
        runtime_errors=[],empty_initial_state_assumed=False)
    state = native.from_snapshot(snap, public_cohort=public, cutoff_ns=20,
        owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
    return state, draws, args, snap


class BusyCandidateTests(unittest.TestCase):
    def test_two_residual_worlds_resume_full_future_calendar_exactly(self):
        state, draws, args, _ = fixture(); before=copy.deepcopy((state,draws,args))
        out=decode(m.evaluate(state,draws,**args))
        self.assertEqual((state,draws,args),before)
        self.assertEqual(out['world_unknown_probability_mass'],0)
        ended, active = out['draws']
        self.assertEqual(ended['candidate_checkpoint_at_cut']['payload']['active'],{})
        self.assertTrue(active['candidate_checkpoint_at_cut']['payload']['active'])
        # Reader RESULT already arrived in first world, but collector has not processed it.
        self.assertEqual(ended['candidate_checkpoint_at_cut']['payload']['receipts']['r0'],19)
        self.assertEqual(ended['candidate_checkpoint_at_cut']['payload']['ready_occurrences']['r0'],11)
        for draw, row in zip(draws,out['draws']):
            value=draw['inputs']; joint={k:value[k] for k in m.ensemble.INPUT_FIELDS}
            direct=decode(forecast(state['binding']['public_cohort'],**joint,**value['initial_state'],**args))
            self.assertEqual(row['forecast'],direct)
            self.assertEqual(row['offered_requests'],2)
            self.assertEqual(row['forecast']['scoring']['original_score']['cohort_size'],2)
            self.assertIn('r1',row['forecast']['world']['starts'])
        self.assertFalse(out['native_full_prefix_likelihood_identified'])
        self.assertFalse(out['native_busy_forecast_eligible'])
        self.assertFalse(out['calibrated_posterior'])

    def test_missing_and_incompatible_slots_keep_original_weight(self):
        state, draws, args, _=fixture()
        draws[0]['weight']=F(1,3);draws[1]['weight']=F(1,3)
        draws[1]['inputs']['initial_state']['initial_worker_read_release_ns']['owner']=2
        draws.append(dict(draw_id='missing',block_id='unmodeled',weight=F(1,3),
            inputs=m.ensemble.UnknownForecastInputs('no joint donor')))
        out=decode(m.evaluate(state,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],F(2,3))
        self.assertEqual(out['six_terms']['P']['known_probability_mass'],F(1,3))
        self.assertIsNone(out['six_terms']['P']['expected_value'])
        self.assertTrue(all(r['fixed_drain_incomplete_probability_bounds']==[0,F(2,3)] for r in out['fixed_drain_risk']))
        self.assertEqual(out['draws'][1]['all_public_request_ids_retained'],['r0','r1'])
        self.assertFalse(out['unknown_worlds_renormalized'])

    def test_unmapped_phase_keeps_all_mass_unknown(self):
        state, draws, args, snap=fixture()
        snap['request_records'][0]['execution_events'].append(dict(request_id='r0',kind='gpu',
            event='start',segment_id='p',perf_ns=2,received_time_ns=3,
            collector_dequeued_time_ns=4,collector_processing_time_ns=5))
        state=native.from_snapshot(snap,public_cohort=state['binding']['public_cohort'],cutoff_ns=20,
            owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
        out=decode(m.evaluate(state,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],1)
        self.assertTrue(all('UNMAPPED_NATIVE_PHASE_OR_QUERY' in r['reason'] for r in out['draws']))

    def test_absent_processed_result_rejects_incompatible_world(self):
        state, draws, args, _=fixture();draws=draws[:1];draws[0]['weight']=F(1)
        for slacks in draws[0]['inputs']['protocol_slacks'].values():
            for value in slacks.values():value['collector_processing_slack_ns']=0
        out=decode(m.evaluate(state,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],1)
        self.assertIn('PROCESSED_BOUNDARY_PATTERN',out['draws'][0]['reason'])

    def test_observed_result_and_ready_require_all_four_modeled_clocks(self):
        state,draws,args,snap=fixture();draws=draws[:1];draws[0]['weight']=F(1)
        for slacks in draws[0]['inputs']['protocol_slacks'].values():
            for value in slacks.values():value['collector_processing_slack_ns']=0
        row=snap['request_records'][0]
        row.update(worker_response=dict(end_perf_ns=9,success=True),
            response_reader_event_enqueue_time_ns=19,response_collector_dequeued_time_ns=19,
            response_collector_processing_time_ns=19,
            previous_handler_next_request_ready=dict(request_id='r0',pid=41,perf_ns=11,
                received_time_ns=21,collector_dequeued_time_ns=21,collector_processing_time_ns=21))
        for perturb in (False,True):
            if perturb:row['response_reader_event_enqueue_time_ns']=18
            s=native.from_snapshot(snap,public_cohort=state['binding']['public_cohort'],cutoff_ns=22,
                owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
            out=decode(m.evaluate(s,draws,**args))
            self.assertEqual(out['world_unknown_probability_mass'],int(perturb))

    def test_wrong_native_producer_binding_is_not_a_compatible_candidate(self):
        state,draws,args,_=fixture()
        state['binding']['owner_producers']['owner']='other-stdout'
        state['state_sha256']=m.digest({k:v for k,v in state.items() if k!='state_sha256'})
        out=decode(m.evaluate(state,draws,**args))
        self.assertEqual(out['world_unknown_probability_mass'],1)
        self.assertTrue(all('PRODUCER_BINDING_MISMATCH' in r['reason'] for r in out['draws']))

    def test_future_native_body_stays_opaque(self):
        state, draws, args, snap=fixture()
        snap['request_records'][0]['execution_events'].append(Poison(collector_processing_time_ns=21))
        later=native.from_snapshot(snap,public_cohort=state['binding']['public_cohort'],cutoff_ns=20,
            owner_producers={'owner':'owner'},execution_context_sha256='b'*64)
        self.assertEqual(later,state)
        self.assertEqual(m.evaluate(later,draws,**args),m.evaluate(state,draws,**args))

    def test_invalid_contracts_raise_and_native_guard_stays_closed(self):
        state, draws, args, _=fixture()
        for kind in ('hash','mass','initial','score','override'):
            s=copy.deepcopy(state);d=copy.deepcopy(draws);a=copy.deepcopy(args)
            if kind=='hash':s['cutoff_ns']=21
            elif kind=='mass':d.pop()
            elif kind=='initial':d[0]['inputs']['initial_state']={'continuation_state':{}}
            elif kind=='score':a['scoring_plan']['weights']['P']=2
            else:a['continuation_state']={}
            with self.subTest(kind=kind),self.assertRaises(ValueError):m.evaluate(s,d,**a)
        with patch.object(m.engine,'simulate',side_effect=ValueError('implementation failure')):
            with self.assertRaisesRegex(ValueError,'implementation failure'):m.evaluate(state,draws,**args)
        with self.assertRaisesRegex(UnknownEvidence,'busy_state_support_not_identified'):
            native.forecast(state,requests=state['binding']['public_cohort'],window_start_ns=20)


if __name__=='__main__':unittest.main()
