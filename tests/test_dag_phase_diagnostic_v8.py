import unittest
from tools.predictor_validation_20261027.dag_phase_diagnostic_v8 import peer_exposure,union_ns


class PhaseExposureTests(unittest.TestCase):
    def test_joint_peer_exposure_partitions_overlapping_owners_once(self):
        peers=[(0,7,'a','prefill'),(3,10,'b','decode')]
        self.assertEqual(peer_exposure(0,10,peers),{'a:prefill':3,'a:prefill|b:decode':4,'b:decode':3})

    def test_outside_and_partial_ranges_keep_the_full_own_phase(self):
        self.assertEqual(peer_exposure(5,15,[(0,8,'a','prefill'),(12,20,'a','decode')]),
            {'a:prefill':3,'NO_PEER_PYTHON_PHASE':4,'a:decode':3})

    def test_api_wall_union_never_sums_nested_thread_intervals(self):
        self.assertEqual(union_ns([(1,10),(3,5),(7,15),(20,22)]),16)
        self.assertEqual(union_ns([]),0)

    def test_reverse_observations_are_rejected(self):
        with self.assertRaises(ValueError):union_ns([(2,1)])
        with self.assertRaises(ValueError):peer_exposure(2,1,[])


if __name__=='__main__':unittest.main()
