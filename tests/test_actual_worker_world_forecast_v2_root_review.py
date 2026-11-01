"""Independent whole-entry concurrency and original integer-score checks."""
import copy
from dataclasses import replace
from fractions import Fraction as F
import unittest

from tests import test_actual_worker_world_forecast_v2 as fixtures
from tests.test_actual_worker_twoFIFO_reference_v4 import decode
from tools.predictor_validation_20261027.actual_worker_world_forecast_v2 import forecast


class WholeForecastIndependentReview(unittest.TestCase):
    def test_four_lanes_are_concurrent_with_all_offered_and_work_preserved(self):
        args=fixtures.forecast_inputs();source=copy.deepcopy(args)
        for owner in ('second','third','fourth'):
            args['requests'].extend(dict(r,owner=owner,request_id=owner+':'+r['request_id']) for r in source['requests'])
            old=source['plans'][('owner','text','input')]
            new=tuple(replace(p,ready_producer_id=owner,macros=tuple(replace(m,marks=tuple(
                replace(mark,producer_id=owner) for mark in m.marks)) for m in p.macros)) for p in old)
            args['plans'][owner,'text','input']=new
            args['protocol_slacks'].update({p:source['protocol_slacks'][q] for p,q in zip(new,old)})
            args['transport_draws'].update({(owner,*k[1:]):v for k,v in source['transport_draws'].items()})
            for field in ('initial_sender_available_ns','initial_worker_read_release_ns'):args[field][owner]=0
            args['scoring_plan']['aliases'][owner]=dict(family='text',supported_tasks=['text'])
        args['scoring_plan']['workload_rps']['text']*=4
        out=decode(forecast(**args));world=out['world'];queue=out['queue_diagnostics']
        self.assertEqual(world['maximum_parallel_handlers'],4)
        self.assertEqual(world['global_admission_cap'],None)
        self.assertEqual(out['scoring']['original_score']['cohort_size'],16)
        self.assertEqual(queue['max_arrived_waiting_START'],8)
        self.assertEqual(queue['full_public_arrival_envelopes'][-1]['max_offered_response_reference_work_ns'],160)
        self.assertAlmostEqual(out['scoring']['original_score']['terms']['P'],33)
        self.assertFalse(out['main_queue_provider_eligible'])

    def test_score_rounding_and_exact_diagnostic_latency_have_explicit_sub_ns_difference(self):
        args=fixtures.forecast_inputs();args['rates']={('work',0,0):F(7,11)}
        out=decode(forecast(**args));world=out['world'];score=out['scoring']['original_score']
        arrivals={r['request_id']:r['arrival_ns'] for r in args['requests']}
        latencies=[world['RESULT_receipts'][r]-a for r,a in arrivals.items()]
        exact_mean=sum(latencies)/len(latencies)
        exact_p95=out['queue_diagnostics']['reader_E2E_P95_ns']
        self.assertGreaterEqual(score['terms']['L'],float(exact_mean))
        self.assertLess(score['terms']['L']-float(exact_mean),1)
        self.assertGreaterEqual(score['terms']['P'],float(exact_p95))
        self.assertLess(score['terms']['P']-float(exact_p95),1)
        bridge=out['scoring']['original_score_bridge']
        self.assertGreater(bridge['maximum_timestamp_roundup_ns'],0)
        self.assertLess(bridge['maximum_timestamp_roundup_ns'],1)


if __name__=='__main__':unittest.main()
