import copy
from fractions import Fraction as F
import unittest

from tests import test_actual_worker_twoFIFO_reference_v4 as fixtures
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as reference
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_query_adapter_v4_1 as adapter
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_training_diagnostics_v4 as diagnostic


class ActualTwoFIFODiagnosticsIndependentReview(unittest.TestCase):
    def test_query_binding_preserves_all_routes_arrivals_and_native_ineligibility(self):
        args=fixtures.ActualWorkerTwoFIFOTests().inputs();world=reference.simulate(**args);before=copy.deepcopy(world)
        public=copy.deepcopy(args['requests'])
        for row in public:row['arrival_ns']=dict(numerator=row['arrival_ns'],denominator=1)
        view=adapter.query_bound_resource_view(world,public,window_start_ns=F(0),window_end_ns=2,drain_end_ns=150)
        self.assertEqual(world,before)
        self.assertFalse(view['main_queue_provider_eligible']);self.assertFalse(view['query_identity_confers_native_qualification'])
        self.assertTrue(view['source_FIFO_order_match_only'])
        public[0]['request_id']='foreign'
        with self.assertRaises(ValueError):adapter.query_bound_view(world,public,window_start_ns=0,window_end_ns=2,drain_end_ns=150)

    def test_incomplete_four_draw_ensemble_keeps_unknown_denominator(self):
        rows=[dict(heldout_training_capture_sha256='heldout',evaluation_only_heldout_request_id='a',
            status='SIMULATED',predicted_latency_ns=10,actual_latency_ns=10) for _ in range(4)]
        rows[-1].update(status='UNKNOWN',predicted_latency_ns=None)
        result=diagnostic.ensemble(rows)
        self.assertEqual(result['offered_world_requests'],4)
        self.assertEqual(result['offered_unique_requests'],1)
        self.assertEqual(result['complete_four_world_ensembles'],0)
        self.assertEqual(result['incomplete_or_unknown_ensembles'],1)
        self.assertNotIn('four_world_mean_MAE_ms',result)

    def test_random_draw_MAE_differs_from_ensemble_point_error(self):
        rows=[dict(heldout_training_capture_sha256='heldout',evaluation_only_heldout_request_id='a',
            status='SIMULATED',predicted_latency_ns=x,actual_latency_ns=10) for x in (5,5,15,15)]
        result=diagnostic.ensemble(rows)
        self.assertEqual(result['mean_random_world_MAE_ms'],5e-6)
        self.assertEqual(result['four_world_mean_MAE_ms'],0)


if __name__=='__main__':unittest.main()
