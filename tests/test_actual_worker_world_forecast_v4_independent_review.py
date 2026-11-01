"""Independent public/source-clock/resource boundary counterexamples."""
import copy
from fractions import Fraction
import unittest

from tests import test_actual_worker_world_forecast_v4 as fixtures
from tests import test_actual_worker_world_forecast_v2 as oldfixtures
from tests import test_actual_worker_twoFIFO_reference_v4 as clockfixtures
from tools.predictor_validation_20261027 import actual_worker_world_forecast_v4 as forecast
from tools.predictor_validation_20261027 import actual_worker_world_score_v3 as scoring
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v5_2 as engine
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_query_adapter_v5_2 as query
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_query_adapter_v4_1 as oldquery
from tools.predictor_validation_20261027 import twoFIFO_resource_provider_v3 as resources


def resource_fixture(startup=3600., memory_unknown=False, power_unknown=False):
    campaign=resources.checked(resources.receipt(resources.intake.campaign_v2.ROOT /
        'artifacts/predictor-twoFIFO-owned-training-design-v2-20261028-cpu01/campaign.json'))
    plan=resources.checked(campaign['attempts'][0]);context=resources.checked(plan['context'])
    class CPUProvider:
        target_plan=plan
        physical_execution_identity=resources.physical_execution_identity(plan,context)
        document={'priming_content_identity':'CPU_ONLY_EXPLICIT_RESOURCE_SCENARIO'}
        def power(self,active,*,resident_owners):
            return dict(known=not power_unknown,watts=None if power_unknown else (300. if active else 100.))
        def memory(self,active,*,resident_owners,ordered_state_history_identity):
            known=not(memory_unknown and not active)
            return dict(known=known,peak_memory_mib=(900. if active else 400.) if known else None)
        def startup(self):return dict(joules=startup,premeasurement_owned_sampled_peak_mib=600.)
    return CPUProvider(),context


def with_resources(args,**kwargs):
    provider,context=resource_fixture(**kwargs)
    return dict(args,resource_provider=provider,target_context=context,
        resource_memory_initial_state=dict(basis='fresh_after_original_priming'),
        resource_source_evidence=['Explicit CPU scenario only; no native resource accuracy claim'])


def small_args():
    return fixtures.inputs(clockfixtures.ActualWorkerTwoFIFOTests().inputs())


class IndependentWorldBridgeTests(unittest.TestCase):
    def test_real_resource_predictor_API_hash_energy_measurement_and_peak_once(self):
        args=small_args();out=forecast.forecast(**with_resources(args))
        score=out['scoring']['original_score'];pred=out['scoring']['resource_prediction']
        self.assertAlmostEqual(score['terms']['E'],20.1)
        self.assertAlmostEqual(score['terms']['M'],1.08)
        self.assertAlmostEqual(pred['board_energy_joules'],4e-7)
        self.assertEqual(pred['resource_timeline_sha256'],resources.digest(out['scoring']['resource_timeline']))
        self.assertFalse(pred['caller_execution_binding_independently_authenticated'])
        self.assertFalse(out['resource_provider_qualified'])

    def test_extreme_parent_processing_delay_does_not_replace_RESULT_reader(self):
        args=small_args();baseline=forecast.forecast(**with_resources(args))
        for marks in args['protocol_slacks'].values():
            for slack in marks.values():slack['collector_processing_slack_ns']=10**9
        delayed=forecast.forecast(**with_resources(args))
        self.assertEqual(delayed['world']['RESULT_processed'],{})
        self.assertEqual(delayed['world']['RESULT_receipts'],baseline['world']['RESULT_receipts'])
        self.assertEqual(delayed['scoring']['original_score']['terms'],baseline['scoring']['original_score']['terms'])
        self.assertEqual(delayed['scoring']['original_score']['terms']['L'],24)
        self.assertAlmostEqual(delayed['scoring']['original_score']['terms']['P'],28.5)

    def test_unknown_startup_power_or_any_drain_memory_never_fills_full_U(self):
        args=small_args()
        for options,term in (({'startup':None},'E'),({'power_unknown':True},'E'),({'memory_unknown':True},'M')):
            out=forecast.forecast(**with_resources(args,**options))
            self.assertIsNone(out['scoring']['original_score']['terms'][term])
            self.assertIsNone(out['scoring']['original_score']['full_U'])
        missing=forecast.forecast(**with_resources(args,startup=None))
        self.assertIsNotNone(missing['scoring']['resource_prediction']['board_energy_joules'])
        self.assertIsNone(missing['scoring']['resource_prediction']['startup_energy_joules'])

    def test_memory_requires_explicit_history_and_prior_peak_is_preserved_once(self):
        args=small_args();missing=with_resources(args);missing.pop('resource_memory_initial_state')
        unknown=forecast.forecast(**missing)
        self.assertIsNone(unknown['scoring']['original_score']['terms']['M'])
        self.assertFalse(unknown['scoring']['resource_prediction']['memory_initial_history_supplied'])
        prior=with_resources(args);prior['resource_memory_initial_state']=dict(
            basis='explicit_prior_attempt_history',ordered_state_history_identity='a'*64,
            last_state_key='b'*64,prior_owned_sampled_peak_mib=1500.)
        result=forecast.forecast(**prior)
        self.assertAlmostEqual(result['scoring']['original_score']['terms']['M'],1.8)
        self.assertEqual(result['scoring']['resource_prediction']['premeasurement_owned_sampled_peak_mib'],1500.)
        with self.assertRaises(ValueError):forecast.forecast(**dict(args,resource_memory_initial_state=dict(basis='fresh_after_original_priming')))

    def test_query_namespace_binding_full_cohort_route_and_future_arrival(self):
        args=fixtures.inputs();world=forecast.forecast(**args)['world']
        bounds={k:args[k]for k in ('window_start_ns','window_end_ns','drain_end_ns')}
        for mutate in ('schema','route','arrival','order','input','delete'):
            altered=copy.deepcopy(world);public=copy.deepcopy(args['requests'])
            if mutate=='schema':altered['schema']='gsdeploy.actual_worker_twoFIFO_cpu_reference.v4'
            elif mutate=='route':public[-1]['owner']='foreign'
            elif mutate=='arrival':public[-1]['arrival_ns']+=1
            elif mutate=='order':public.reverse()
            elif mutate=='input':public[-1]['pool_id']='foreign'
            else:public.pop()
            with self.subTest(mutate=mutate),self.assertRaises(ValueError):query.query_bound_view(altered,public,**bounds)
        with self.assertRaises(ValueError):oldquery.query_bound_view(world,args['requests'],**bounds)
        self.assertEqual(world['schema'],engine.SCHEMA)
        partial=engine.simulate(**{k:v for k,v in args.items()if k not in ('scoring_plan','quality','scales_ns')},cutoff_ns=5)
        with self.assertRaisesRegex(ValueError,'complete drain'):scoring.score_world(partial,args['requests'],args['scoring_plan'],args['quality'],**bounds)

    def test_resource_resume_cumulative_past_not_dropped_and_forecast_wrapper_cannot_override_cutoff(self):
        args=small_args();engine_args={k:v for k,v in args.items()if k not in ('scoring_plan','quality','scales_ns')}
        partial=engine.simulate(**engine_args,cutoff_ns=5)
        continuous=forecast.forecast(**with_resources(args))
        resumed=forecast.forecast(**dict(with_resources(args),continuation_state=partial['continuation_state']))
        self.assertEqual(resumed,continuous)
        for extra in ({'cutoff_ns':5},{'work_by_public_key_ordinal':{}},{'global_admission_cap':1}):
            with self.assertRaises(TypeError):forecast.forecast(**dict(args,**extra))


if __name__=='__main__':unittest.main()


class PairedSeededBridgeIndependentReview(unittest.TestCase):
    def test_paired_seed_uses_explicit_resource_history_and_current_reader_cohort(self):
        from tests.test_actual_worker_twoFIFO_reference_v5_2 import paired_inputs
        args=fixtures.inputs(paired_inputs(seeded=True,write=5));args.pop('cutoff_ns',None)
        for k in ('initial_sender_available_ns','initial_worker_read_release_ns','initial_protocol_state'):args.pop(k,None)
        missing=with_resources(args);missing.pop('resource_memory_initial_state')
        unknown=forecast.forecast(**missing)
        self.assertEqual(unknown['scoring']['original_score']['cohort_size'],2)
        self.assertIsNone(unknown['scoring']['original_score']['terms']['M'])
        prior=with_resources(args);prior['resource_memory_initial_state']=dict(
            basis='explicit_prior_attempt_history',ordered_state_history_identity='a'*64,
            last_state_key='b'*64,prior_owned_sampled_peak_mib=1500.)
        out=forecast.forecast(**prior)
        self.assertAlmostEqual(out['scoring']['original_score']['terms']['M'],1.8)
        self.assertAlmostEqual(out['scoring']['original_score']['terms']['P'],28.5)
        self.assertEqual(clockfixtures.decode(out['world']['RESULT_receipts']),{'r0':119,'r1':130})
        self.assertTrue(all(send['stdin_line_available_ns'] is None for send in out['world']['transport_schedule'].values()))
        self.assertEqual(out['scoring']['resource_prediction']['premeasurement_owned_sampled_peak_mib'],1500.)
        # The seed provides protocol history, never a silent allocator-history reset.
        self.assertEqual(out['queue_diagnostics']['reader_E2E_offered_requests'],2)

    def test_nonsingleton_paired_transport_never_reaches_score_or_queue(self):
        from unittest.mock import patch
        from tests.test_actual_worker_twoFIFO_reference_v5_2 import paired_inputs
        from tools.predictor_validation_20261027.twoFIFO_paired_transport_v3 import PairedTransportConstraints,UnknownTransportInterval
        args=fixtures.inputs(paired_inputs());args.pop('cutoff_ns',None)
        args['transport_draws'][('owner','text','input',1)]=PairedTransportConstraints(0,0,10,12,-3)
        with patch.object(forecast,'score_world',side_effect=AssertionError('must not score incomplete transport world')):
            with self.assertRaises(UnknownTransportInterval) as caught:forecast.forecast(**args)
        self.assertEqual(caught.exception.evidence['offered_requests'],2)
        self.assertEqual(caught.exception.evidence['public_key_arrival_ordinal'],1)
        self.assertEqual(caught.exception.evidence['START_ns_bounds'],[13,23])
