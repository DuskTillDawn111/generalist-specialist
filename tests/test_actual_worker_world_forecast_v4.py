import copy
from dataclasses import replace
from fractions import Fraction as F
import json
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_actual_worker_world_forecast_v2 as fixtures
from tests import test_actual_worker_twoFIFO_reference_v4 as clock_fixtures
from tools.predictor_validation_20261027 import actual_worker_world_forecast_v4 as module
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as reference
from tools.predictor_validation_20261027.actual_worker_twoFIFO_query_adapter_v5_2 import query_bound_view


def inputs(args=None):
    args=fixtures.forecast_inputs(args)
    owners={row['owner'] for row in args['requests']}
    if all(len(key)==3 for key in args['rates']):
        args['rates']={(owner,*key):value for owner in owners for key,value in args['rates'].items()}
    return args


class NewWorldIntegrationTests(unittest.TestCase):
    def test_complete_future_arrivals_and_original_six_score_use_same_reader_world(self):
        args=inputs();before=copy.deepcopy(args);out=module.forecast(**args)
        self.assertEqual(args,before)
        q=clock_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(q['event_slices'][0]['state']['complete_cohort_remaining_response_reference_work_ns'],40)
        self.assertEqual(q['full_public_arrival_envelopes'][-1]['max_offered_response_reference_work_ns'],40)
        self.assertEqual(q['reader_E2E_P95_ns'],F(327,10))
        self.assertEqual(q['FIFO_convoy_pending_request_ns'],47)
        self.assertEqual(out['query_fingerprint_sha256'],q['query_fingerprint_sha256'])
        self.assertAlmostEqual(out['scoring']['original_score']['terms']['P'],32.7)
        self.assertEqual(out['scoring']['original_score']['cohort_size'],4)
        self.assertIsNone(out['scoring']['original_score']['full_U'])
        self.assertTrue(out['settled_history_seed_implemented']);self.assertFalse(out['active_native_history_assimilation_implemented'])
        self.assertFalse(out['final_gain_eligible'])

    def test_owner_rates_preserve_phase_semantics_on_board_timeline(self):
        args=inputs();base=args['plans'][('owner','text','input')][0]
        second=replace(base,ready_producer_id='second',macros=(replace(base.macros[0],
            marks=tuple(replace(mark,producer_id='second') for mark in base.macros[0].marks)),))
        args['requests']=[dict(request_id='first',owner='owner',task='text',pool_id='input',arrival_ns=0),
                          dict(request_id='second',owner='second',task='text',pool_id='input',arrival_ns=0)]
        args['plans']={('owner','text','input'):base,('second','text','input'):second}
        args['protocol_slacks']={base:args['protocol_slacks'][base],second:args['protocol_slacks'][base]}
        draw=next(iter(args['transport_draws'].values()))
        args['transport_draws']={(owner,'text','input',0):draw for owner in ('owner','second')}
        args['rates']={(owner,'work',n,same):(1 if owner=='owner' else 2)
                       for owner in ('owner','second') for n,same in ((0,0),(1,0),(1,1))}
        args['initial_sender_available_ns']={'owner':0,'second':0}
        args['initial_worker_read_release_ns']={'owner':0,'second':0}
        args['scoring_plan']['aliases']['second']=dict(family='text',supported_tasks=['text'])
        out=module.forecast(**args)
        world=clock_fixtures.decode(out['world'])
        self.assertEqual(world['READY_occurrences'],{'second':5,'first':10})
        phases={r['response_phase'] for s in out['scoring']['resource_timeline']['slices']
                for r in s['actual_START_to_READY_handlers']}
        self.assertEqual(phases,{'work'})
        self.assertTrue(all(len(row['evidence_key'])==4 for row in world['response_exposure']))

    def test_resume_with_active_and_unprocessed_notices_preserves_score_and_queue(self):
        args=inputs();engine={k:v for k,v in args.items() if k not in ('scoring_plan','quality','scales_ns')}
        partial=reference.simulate(**engine,cutoff_ns=13)
        checkpoint=json.loads(json.dumps(partial['continuation_state']))
        self.assertTrue(partial['active_response_states'])
        self.assertTrue(partial['pending_notifications'])
        continued=module.forecast(**dict(args,continuation_state=checkpoint))
        continuous=module.forecast(**args)
        self.assertEqual(continued,continuous)

    def test_fixed_short_drain_retains_all_four_requests_and_incomplete_P95(self):
        args=inputs();args['drain_end_ns']=11
        out=module.forecast(**args);q=clock_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(out['scoring']['original_score']['cohort_size'],4)
        self.assertEqual(len(out['scoring']['original_score']['incomplete_ids']),4)
        self.assertIsNone(out['scoring']['original_score']['terms']['P'])
        self.assertIsNone(q['reader_E2E_P95_ns'])
        self.assertEqual(q['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'],29)
        self.assertEqual(q['full_public_arrival_envelopes'][-1]['max_arrival_count'],4)

    def test_public_boundary_route_and_missing_transport_cannot_be_silently_changed(self):
        args=inputs();out=module.forecast(**args)
        boundaries=dict(window_start_ns=0,window_end_ns=11,drain_end_ns=70)
        with self.assertRaises(ValueError):query_bound_view(out['world'],args['requests'][:-1],**boundaries)
        with self.assertRaises(ValueError):query_bound_view(out['world'],args['requests'],**dict(boundaries,drain_end_ns=71))
        with self.assertRaises(UnknownEvidence):module.forecast(**dict(args,transport_draws={}))
        with self.assertRaises(TypeError):module.forecast(**dict(args,global_admission_cap=1))

    def test_paired_constraints_do_not_fabricate_line_clock_in_queue_diagnostics(self):
        from tests.test_actual_worker_twoFIFO_reference_v5_2 import paired_inputs
        args=inputs(paired_inputs(write=5));args.pop('cutoff_ns',None)
        out=module.forecast(**args)
        q=clock_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(out['world']['offered_requests'],2)
        self.assertEqual(q['reader_E2E_offered_requests'],2)
        self.assertEqual(q['reader_E2E_P95_ns'],F(57,2))
        self.assertEqual(q['unknown_work_public_keys'],[])
        self.assertTrue(all(send['stdin_line_available_ns'] is None
                            for send in out['world']['transport_schedule'].values()))
        self.assertFalse(out['busy_transport_transfer_qualified'])

    def test_settled_native_prefix_remains_outside_current_cohort_and_keeps_offsets(self):
        from tests.test_twoFIFO_settled_history_initial_v1 import seeded_inputs
        engine=seeded_inputs();args=inputs(engine)
        for key in ('initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state'):
            args.pop(key,None)
        out=module.forecast(**args);world=clock_fixtures.decode(out['world'])
        self.assertEqual(out['scoring']['original_score']['cohort_size'],2)
        self.assertEqual(set(r['request_id'] for r in world['public_offered_requests']),{'r0','r1'})
        first=next(row for row in world['trace'] if row['kind']=='latent_occurrence')
        self.assertEqual(first['producer_sequence'],3)
        self.assertEqual(first['owner_issuance_ordinal'],3)
        self.assertEqual(out['query_fingerprint_sha256'],out['queue_diagnostics']['query_fingerprint_sha256'])


if __name__=='__main__':unittest.main()
