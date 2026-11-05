"""Profiler intervals must retain long/nested APIs without double counting."""
import unittest
from experiments.analyze_activation_cupti_host_probe import IntervalIndex,union_ns,detail_key


class CuptiHostProbeTests(unittest.TestCase):
    def test_nested_api_union_and_long_prior_interval(self):
        rows=[dict(start_ns=0,end_ns=100,api='runtime'),dict(start_ns=10,end_ns=15,api='driver'),
              dict(start_ns=20,end_ns=25,api='driver'),dict(start_ns=60,end_ns=80,api='driver')]
        index=IntervalIndex(rows);selected=index.overlaps(50,70)
        self.assertEqual([r['api'] for r in selected],['runtime','driver'])
        self.assertEqual(union_ns(selected,50,70),20)
        self.assertEqual(index.preceding(50)['end_ns'],25)
        self.assertEqual(index.preceding(101)['api'],'runtime')
        self.assertEqual(index.overlaps(100,110),[])

    def test_marker_preserves_request_colons_and_ordinal(self):
        self.assertEqual(detail_key('GS_DETAIL:training:157:prepare_and_media:0'),('training:157','prepare_and_media',0))
        self.assertIsNone(detail_key('GS_STAGE:training:157:gpu:native_inference'))
        with self.assertRaises(ValueError):detail_key('GS_DETAIL:training:157:prepare_and_media:invalid')


if __name__=='__main__':unittest.main()
