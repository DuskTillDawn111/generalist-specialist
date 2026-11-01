"""Projected worker handlers preserve causal cutoff and physical accounting."""
import copy
import unittest

from tools.predictor_validation_20261027 import worker_handler_projection_adapter_v41 as v


def fixture(end=150):
    indexes={o:v.IntervalIndex([]) for o in v.OWNERS};indexes['text#0']=v.IntervalIndex([(50,end)])
    cuts=sorted(t for index in indexes.values() for span in index.spans for t in span)
    residuals={o:10 for o in v.OWNERS}
    row=dict(capture='a',held_owner='ocr#0',driver_context=[1,0,10,100],restore_start_ns=0,
        start_ns=0,end_ns=200,age_entry_ns=0,age_exit_ns=200,physical_SAVE_END_observed=True,
        physical_SAVE_seqNo=129,exposure_qualified=False,
        shared_left_features=dict(held_owner='ocr#0',previous_held_owner='text#1',context_age_ns=0,
            handler_active={o:True for o in v.OWNERS}))
    return row,indexes,cuts,residuals


class WorkerHandlerProjectionTests(unittest.TestCase):
    def test_worker_half_open_mask_does_not_OR_native_stage_evidence(self):
        row,indexes,cuts,residuals=fixture()
        rows=list(v.split_risk(row,indexes,cuts,residuals))
        self.assertEqual([(x['start_ns'],x['end_ns']) for x in rows],[(0,50),(50,150),(150,200)])
        self.assertEqual([x['shared_left_features']['handler_active']['text#0'] for x in rows],[False,True,False])
        self.assertTrue(all(x['shared_left_features']['handler_active']['ocr#0'] is False for x in rows))

    def test_split_retains_unresolved_exposure_and_one_exact_terminal_SAVE(self):
        row,indexes,cuts,residuals=fixture();rows=list(v.split_risk(row,indexes,cuts,residuals))
        self.assertEqual(sum(x['end_ns']-x['start_ns'] for x in rows),200)
        self.assertEqual(sum(x['physical_SAVE_END_observed'] for x in rows),1)
        self.assertEqual([x['physical_SAVE_seqNo'] for x in rows],[None,None,129])
        self.assertTrue(all(x['exposure_qualified'] is False for x in rows))
        self.assertTrue(all(x['age_entry_ns']==x['start_ns'] for x in rows))

    def test_future_END_mutation_and_residual_do_not_change_cutoff_features(self):
        a,ia,ca,ra=fixture(150);b,ib,cb,rb=fixture(190)
        # Query at 100, before both observed projected END events.
        a.update(start_ns=100,age_entry_ns=100);b.update(start_ns=100,age_entry_ns=100)
        rb={o:1000 for o in v.OWNERS}
        xa=next(v.split_risk(a,ia,ca,ra));xb=next(v.split_risk(b,ib,cb,rb))
        self.assertEqual(xa['shared_left_features'],xb['shared_left_features'])
        self.assertNotEqual(xa['handler_boundary_proximity_diagnostic'],xb['handler_boundary_proximity_diagnostic'])
        self.assertEqual(xb['end_ns'],190)  # No END-minus-residual split is added.

    def test_exact_boolean_mask_and_attained_age_required(self):
        mask={o:False for o in v.OWNERS}
        self.assertEqual(v.features('ocr#0',None,0,mask)['handler_active'],mask)
        for bad in [dict(mask,**{'ocr#0':None}),dict(mask,**{'ocr#0':1})]:
            with self.assertRaises(ValueError):v.features('ocr#0',None,0,bad)
        with self.assertRaises(ValueError):v.features('ocr#0',None,-1,mask)


if __name__=='__main__':unittest.main()
