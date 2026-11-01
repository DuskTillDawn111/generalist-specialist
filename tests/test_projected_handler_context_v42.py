import copy
import unittest

from tools.predictor_validation_20261027 import projected_handler_context_v42 as v


def state(held='text#0',age=0,active=('text#0',)):
    return dict(held_owner=held,previous_held_owner=None,context_age_ns=age,
        handler_active={o:o in active for o in v.OWNERS})


def row(capture,begin,end,event,active):
    return dict(capture=capture,shared_left_features=state(age=begin,active=active),
        age_entry_ns=begin,age_exit_ns=end,start_ns=begin,end_ns=end,restore_start_ns=0,
        physical_SAVE_END_observed=event,exposure_qualified=True,
        held_owner='text#0',driver_context=[1,0,1,99])


class ProjectedHandlerTests(unittest.TestCase):
    def test_handler_projection_is_used_without_phase_or_unknown_coercion(self):
        alone=state();co=state(active=('text#0','text#1'))
        self.assertNotEqual(v.mark_vector(alone),v.mark_vector(co))
        self.assertNotEqual(v.hazard_tokens(alone),v.hazard_tokens(co))
        invalid=copy.deepcopy(alone);invalid['handler_active']['ocr#0']=None
        with self.assertRaises(ValueError):v.mark_vector(invalid)
        invalid=dict(alone,future_phase_END=100)
        with self.assertRaises(ValueError):v.hazard_tokens(invalid)

    def test_single_handler_risk_is_not_deleted_or_assigned_zero_hazard(self):
        risks=[row('a',0,1000000,False,('text#0',)),
               row('a',1000000,1001000,True,('text#0','text#1')),
               row('b',0,1000000,False,('text#0',)),
               row('b',1000000,1001000,True,('text#0','text#1'))]
        marks=[dict(capture=c,features=state(age=1001000,active=('text#0','text#1')),
                    next_owner='text#1',same_owner=False) for c in ('a','b')]
        fit=v.fit_models(marks,risks)
        self.assertEqual(fit['risk_accounting']['counts']['exposure_ns'],2002000)
        only=[r for r in fit['risk_accounting']['rows'] if 'held_handler:text#0:0010' in r['features']]
        self.assertEqual(sum(r['exposure_ns'] for r in only),2000000)
        self.assertEqual(sum(r['events'] for r in only),0)
        # Finite regularized hazard retains uncertainty; no hard zero rule.
        candidate=object.__new__(v.ObservableContextCandidate)
        candidate.model={'fitted':fit};h=fit['hazard'];candidate.coefficients=dict(zip(h['columns'],h['coefficients']))
        query=candidate.save_hazard(state(age=100),100)
        self.assertGreater(query['hazard_per_ms'],0.)
        self.assertIn('UNCALIBRATED_FIXED_MEDIAN_WORKER_HANDLER_PROJECTION',query['OOD_reasons'])
        self.assertIsNone(query['conditional_CDF'])

    def test_inactive_mask_does_not_fabricate_previous_or_reset_attained_age(self):
        q=state(age=23000,active=())
        self.assertEqual(v.signature(q),'text#0:0000')
        self.assertEqual(v.validate_features(q)['context_age_ns'],23000)
        self.assertIn('previous:None',v.mark_names())


if __name__=='__main__':unittest.main()
