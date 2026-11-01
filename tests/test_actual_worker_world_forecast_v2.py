import copy
from dataclasses import replace
from fractions import Fraction as F
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tests import test_actual_twoFIFO_queue_diagnostics_v1 as queue_fixtures
from tests import test_actual_worker_twoFIFO_reference_v4 as twofifo_fixtures
from tests import test_actual_worker_world_score_v1 as score_fixtures
from tools.predictor_validation_20261027 import actual_worker_world_forecast_v2 as module
from tools.predictor_validation_20261027.macro_response_queue_reference_v1 import Macro, Mark


def forecast_inputs(args=None):
    args=queue_fixtures.burst_inputs() if args is None else args
    args=dict(args);args.pop('cutoff_ns',None)
    n=len(args['requests']);duration=args['window_end_ns']-args['window_start_ns']
    args.update(scoring_plan=dict(aliases={'owner':dict(family='text',supported_tasks=['text'])},
        workload_rps={'text':n*1e9/duration},latency_reference_ms={'text':1e-6},
        window_seconds=duration/1e9,startup_amortization_seconds=3600,memory_budget_mib=1000,
        memory_safe_factor=1.2,energy_reference_watts=10,weights={k:1/6 for k in 'QTLPME'}),
        quality={'text':{'text':.25}},scales_ns=(1,2,10,11))
    return args


def with_resources(args,provider):
    return dict(args,resource_provider=provider,target_context=provider.context,
        resource_source_evidence=['Synthetic CPU-only declared binding, not physical GPU evidence'])


class ActualWorkerWorldForecastTests(unittest.TestCase):
    def test_complete_trace_future_work_score_and_queue_share_one_public_identity(self):
        args=forecast_inputs();before=copy.deepcopy(args);out=module.forecast(**args)
        self.assertEqual(args,before)
        self.assertEqual(out['public_offered_requests'],4)
        self.assertEqual(out['scoring']['original_score']['cohort_size'],4)
        self.assertEqual(out['query_fingerprint_sha256'],out['queue_diagnostics']['query_fingerprint_sha256'])
        queue=twofifo_fixtures.decode(out['queue_diagnostics'])
        first=queue['event_slices'][0]['state']
        self.assertEqual(first['lifecycle_count']['scheduled_not_arrived'],3)
        self.assertEqual(first['complete_cohort_remaining_response_reference_work_ns'],40)
        self.assertEqual(queue['full_public_arrival_envelopes'][-1]['max_offered_response_reference_work_ns'],40)
        self.assertEqual(queue['queue_recovery_episodes'][0]['clear_ns'],30)
        self.assertEqual(queue['reader_E2E_P95_ns'],F(327,10))
        self.assertAlmostEqual(out['scoring']['original_score']['terms']['P'],32.7)
        self.assertIsNone(out['scoring']['original_score']['terms']['M'])
        self.assertIsNone(out['scoring']['original_score']['terms']['E'])
        self.assertIsNone(out['scoring']['original_score']['full_U'])
        self.assertEqual(out['queue_diagnostics']['unknown_work_public_keys'],[])
        self.assertIsNone(out['native_current_visible_risk_features'])
        self.assertFalse(out['main_queue_provider_eligible']);self.assertFalse(out['stability_qualified'])

    def test_heterogeneous_multimacro_fractional_work_is_public_arrival_ordinal_not_input_order(self):
        args=forecast_inputs();key=('owner','text','input');old=args['plans'][key]
        new=tuple(replace(plan,macros=(Macro('entry'+str(i),'entry',3+F(i,2),()),
            Macro('work'+str(i),'work',7,(Mark('result'+str(i),4,0,'result','owner'),))))
            for i,plan in enumerate(old))
        args['plans']={key:new};args['rates'][('entry',0,0)]=2
        args['protocol_slacks']={n:args['protocol_slacks'][o] for o,n in zip(old,new)}
        args['requests']=[args['requests'][3],args['requests'][0],args['requests'][2],args['requests'][1]]
        costs,details=module.public_plan_work(args['requests'],args['plans'])
        self.assertEqual(list(costs.values()),[10,F(21,2),11,F(23,2)])
        out=module.forecast(**args);work=twofifo_fixtures.decode(out['supplied_plan_reference_work'])
        self.assertEqual([r['response_reference_work_ns'] for r in work],[10,F(21,2),11,F(23,2)])
        self.assertTrue(all(len(r['macros'])==2 for r in work))
        queue=twofifo_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(queue['full_public_arrival_envelopes'][-1]['max_offered_response_reference_work_ns'],43)
        self.assertEqual(out['world']['public_key_arrival_ordinals'],{'b0':0,'b1':1,'b2':2,'b3':3})

    def test_enqueue_reorder_does_not_reassign_public_plan_costs(self):
        args=forecast_inputs(twofifo_fixtures.ActualWorkerTwoFIFOTests().inputs(enqueue=[10,0],worker=0))
        key=('owner','text','input');p0,p1=args['plans'][key]
        second=replace(p1,macros=(replace(p1.macros[0],reference_work_ns=20),))
        args['plans']={key:(p0,second)};args['protocol_slacks'][second]=args['protocol_slacks'].pop(p1)
        out=module.forecast(**args)
        self.assertEqual(list(out['world']['starts']),['r1','r0'])
        self.assertEqual([r['response_reference_work_ns']['numerator'] for r in out['supplied_plan_reference_work']],[10,20])
        queue=twofifo_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(queue['full_public_arrival_envelopes'][-1]['max_offered_response_reference_work_ns'],30)

    def test_reference_work_is_not_slowed_wall_service(self):
        slow=forecast_inputs();fast=copy.deepcopy(slow);fast['rates']={('work',0,0):2}
        a=module.forecast(**slow);b=module.forecast(**fast)
        self.assertEqual(a['supplied_plan_reference_work'],b['supplied_plan_reference_work'])
        self.assertEqual(a['queue_diagnostics']['full_public_arrival_envelopes'],b['queue_diagnostics']['full_public_arrival_envelopes'])
        self.assertGreater(a['scoring']['original_score']['terms']['L'],b['scoring']['original_score']['terms']['L'])
        self.assertNotEqual(a['world']['response_exposure'],b['world']['response_exposure'])

    def test_optional_resources_and_UNKNOWN_cannot_promote(self):
        args=forecast_inputs(twofifo_fixtures.ActualWorkerTwoFIFOTests().inputs())
        known=module.forecast(**with_resources(args,score_fixtures.ExplicitCPUResources()))
        self.assertAlmostEqual(known['scoring']['original_score']['terms']['M'],.72)
        self.assertAlmostEqual(known['scoring']['original_score']['terms']['E'],10.1)
        missing=module.forecast(**with_resources(args,score_fixtures.ExplicitCPUResources(startup=None)))
        self.assertIsNone(missing['scoring']['original_score']['terms']['E'])
        self.assertIsNone(missing['scoring']['original_score']['full_U'])
        for output in (known,missing):
            self.assertFalse(output['resource_provider_qualified']);self.assertFalse(output['final_gain_eligible'])
        with self.assertRaises(ValueError):module.forecast(**dict(args,resource_provider=score_fixtures.ExplicitCPUResources()))

    def test_incomplete_at_original_drain_keeps_future_cohort_costs_and_unknown_score(self):
        args=forecast_inputs(queue_fixtures.burst_inputs(drain=11))
        out=module.forecast(**args);score=out['scoring']['original_score'];queue=twofifo_fixtures.decode(out['queue_diagnostics'])
        self.assertEqual(score['cohort_size'],4);self.assertEqual(len(score['incomplete_ids']),4)
        self.assertIsNone(score['terms']['L']);self.assertIsNone(score['terms']['P'])
        self.assertIsNone(queue['reader_E2E_P95_ns'])
        self.assertEqual(queue['cutoff_state']['complete_cohort_remaining_response_reference_work_ns'],29)
        self.assertEqual(len(queue['original_drain']['RESULT_unfinished_request_ids']),4)
        self.assertEqual(queue['full_public_arrival_envelopes'][-1]['max_arrival_count'],4)
        self.assertTrue(out['full_original_drain_and_public_cohort_preserved'])

    def test_missing_explicit_plan_transport_initial_state_and_overrides_rejected(self):
        for field in ('protocol_slacks','transport_draws','initial_sender_available_ns',
                      'initial_worker_read_release_ns','initial_protocol_state'):
            args=forecast_inputs();args.pop(field)
            with self.assertRaises(TypeError):module.forecast(**args)
        args=forecast_inputs();args['plans']={}
        with self.assertRaises(UnknownEvidence):module.forecast(**args)
        args=forecast_inputs();args['plans'][('owner','text','input')]=args['plans'][('owner','text','input')][:-1]
        with self.assertRaises(UnknownEvidence):module.forecast(**args)
        for extra in ({'cutoff_ns':1},{'work_by_public_key_ordinal':{}},{'global_admission_cap':1}):
            with self.assertRaises(TypeError):module.forecast(**dict(forecast_inputs(),**extra))

    def test_shared_plan_and_public_identifier_rename_preserve_exact_costs(self):
        args=forecast_inputs();key=('owner','text','input');plan=args['plans'][key][0]
        args['plans']={key:plan};args['protocol_slacks']={plan:args['protocol_slacks'][plan]}
        out=module.forecast(**args)
        for row in args['requests']:row['request_id']='public-renamed-'+row['request_id']
        renamed=module.forecast(**args)
        self.assertEqual(out['supplied_plan_reference_work'],renamed['supplied_plan_reference_work'])
        self.assertEqual(out['scoring']['original_score']['terms'],renamed['scoring']['original_score']['terms'])
        self.assertEqual(len(renamed['supplied_plan_reference_work']),4)


if __name__=='__main__':unittest.main()
