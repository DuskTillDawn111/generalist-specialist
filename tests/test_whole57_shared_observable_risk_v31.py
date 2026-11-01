import unittest

from tools.predictor_validation_20261027 import whole57_shared_observable_risk_v31 as v
from tools.predictor_validation_20261027.owner_lifecycle_mark_v21 import IntervalIndex


class SharedProjectionTests(unittest.TestCase):
    def test_future_phase_end_does_not_change_current_projection(self):
        a={o:IntervalIndex([]) for o in v.OWNERS};b=dict(a)
        a['ocr#0']=IntervalIndex([(10,30)]);b['ocr#0']=IntervalIndex([(10,300000)])
        self.assertEqual(v.handler_projection(a,20),v.handler_projection(b,20))
        self.assertTrue(v.handler_projection(a,20)['ocr#0'])
        self.assertIsNone(v.handler_projection(a,30)['ocr#0'])

    def test_absence_of_NVTX_is_unknown_including_engine_idle(self):
        source=v.handler_projection({o:IntervalIndex([]) for o in v.OWNERS},50)
        self.assertTrue(all(x is None for x in source.values()))
        self.assertEqual(v.engine_projection({o:False for o in v.OWNERS}),source)
        self.assertEqual(v.shared_features('ocr#0',None,123,source)['context_age_ns'],123)
        bad=dict(source);bad['ocr#0']=False
        with self.assertRaises(ValueError):v.shared_features('ocr#0',None,123,bad)

    def test_only_complete_cutoff_fields_accepted(self):
        with self.assertRaises(ValueError):v.engine_projection({'ocr#0':True})
        with self.assertRaises(ValueError):v.engine_projection({o:None for o in v.OWNERS})
        states={o:None for o in v.OWNERS}
        with self.assertRaises(ValueError):v.shared_features('ocr#0',None,-1,states)
        a=v.shared_features('ocr#0','text#0',77768,states)
        self.assertEqual(set(a),{'held_owner','previous_held_owner','context_age_ns','handler_active'})


if __name__=='__main__':unittest.main()
