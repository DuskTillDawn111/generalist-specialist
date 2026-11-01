from copy import deepcopy
import unittest

from tests import test_actual_worker_twoFIFO_reference_v4 as fixtures
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_reference_v4 as reference
from tools.predictor_validation_20261027 import actual_worker_twoFIFO_query_adapter_v4_1 as adapter


class QueryAndInitialStateTests(unittest.TestCase):
    def world(self):
        kwargs=fixtures.ActualWorkerTwoFIFOTests().inputs()
        return kwargs,reference.simulate(**kwargs)

    def view(self,world,public):
        return adapter.query_bound_view(world,public,window_start_ns=0,window_end_ns=2,drain_end_ns=150)

    def test_full_public_identity_and_boundaries_are_bound_without_mutation(self):
        kwargs,world=self.world();prior=deepcopy(world);view=self.view(world,kwargs['requests'])
        self.assertEqual(world,prior)
        self.assertEqual(len(view['query_fingerprint_sha256']),64)
        self.assertTrue(view['public_query_identity_verified'])
        self.assertFalse(view['query_identity_confers_native_qualification'])
        for field in ['owner','task','pool_id','arrival_ns']:
            altered=deepcopy(kwargs['requests']);altered[0][field]=1 if field=='arrival_ns' else 'other'
            with self.assertRaises(ValueError):self.view(world,altered)
        with self.assertRaises(ValueError):self.view(world,list(reversed(kwargs['requests'])))
        with self.assertRaises(ValueError):self.view(world,kwargs['requests'][:1])
        with self.assertRaises(ValueError):adapter.query_bound_view(world,kwargs['requests'],window_start_ns=0,window_end_ns=3,drain_end_ns=150)

    def test_past_initial_FIFO_boundary_cannot_generate_past_submit_or_START(self):
        kwargs=fixtures.ActualWorkerTwoFIFOTests().inputs()
        for row in kwargs['requests']:row['arrival_ns']+=100
        kwargs.update(window_start_ns=100,window_end_ns=102,drain_end_ns=250,cutoff_ns=250)
        kwargs['initial_protocol_state']=dict(reader_free={},collector_free=100,last_collector_arrival=100,pending_messages=())
        world=fixtures.decode(reference.simulate(**kwargs))
        self.assertEqual(world['parent_dispatches'],{'r0':100,'r1':101})
        self.assertEqual(world['starts'],{'r0':101,'r1':112})
        self.assertTrue(all(at>=100 for at in world['starts'].values()))


if __name__=='__main__':unittest.main()
