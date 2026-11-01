import copy
import math
import unittest

import numpy as np

from tools.predictor_validation_20261027 import whole57_observable_context_v34 as v


def state(owner='text#0', age=2000, previous=None, active=('text#0',)):
    return v.shared_features(owner, previous, age,
        {o:True if o in active else None for o in v.OWNERS})


def risk(capture, start, stop, event, *, qualified=True, owner='text#0'):
    return dict(capture=capture, shared_left_features=state(owner, start),
        age_entry_ns=start, age_exit_ns=stop, start_ns=100000+start,
        end_ns=100000+stop, physical_SAVE_END_observed=event,
        exposure_qualified=qualified, held_owner=owner,
        driver_context=[0,0,17,99], restore_start_ns=100000)


class ObservableTrainingTests(unittest.TestCase):
    def test_native_sequence_number_is_not_global_event_identity(self):
        a=v.event_identity(100,[1,0,5,99],7)
        b=v.event_identity(101,[1,0,5,99],7)
        c=v.event_identity(100,[1,0,6,99],7)
        self.assertEqual(len({a,b,c}),3)
        self.assertEqual(a,v.event_identity(100,[1,0,5,99],7))
        with self.assertRaises(ValueError):v.event_identity(-1,[1,0,5,99],7)

    def test_only_cutoff_fields_and_real_handler_states(self):
        value=state()
        changed=copy.deepcopy(value);changed['future_label']='ocr#0'
        with self.assertRaises(ValueError):v.mark_vector(changed)
        changed=copy.deepcopy(value);changed['handler_active']['ocr#0']=False
        with self.assertRaises(ValueError):v.mark_vector(changed)
        self.assertEqual(len(v.mark_vector(value)),len(v.mark_names()))

    def test_split_preserves_exact_exposure_events_and_unresolved(self):
        rows=[risk('a',500,10000,True),risk('a',10000,12000,False),
              risk('a',12000,20000,True,qualified=False)]
        data=v.aggregate(rows)
        self.assertEqual(data['counts']['all_exposure_ns'],19500)
        self.assertEqual(data['counts']['exposure_ns'],11500)
        self.assertEqual(data['counts']['events'],1)
        self.assertEqual(data['counts']['unresolved_paired_SAVE'],1)
        self.assertEqual(sum(r['exposure_ns'] for r in data['rows']),11500)
        self.assertEqual(sum(r['events'] for r in data['rows']),1)
        last_event=next(r for r in data['rows'] if r['events'])
        self.assertEqual(last_event['age_bin'],v.EDGES_NS.index(10000)-1)

    def test_unresolved_bad_exposure_and_holder_identity_rejected_before_accounting(self):
        good=risk('a',0,1000,True)
        for bad in [risk('a',2000,1000,True,qualified=False),
                    risk('a',1000,1000,False,qualified=False),
                    dict(good,physical_SAVE_END_observed=1),
                    dict(good,held_owner='ocr#0'),
                    dict(good,restore_start_ns=99999),
                    dict(good,exposure_qualified=None)]:
            with self.subTest(bad=bad),self.assertRaises(ValueError):v.aggregate([good,bad])

    def test_observable_changes_do_not_reset_context_age_or_add_future_features(self):
        a=state(age=77768);b=state(age=77768,active=('text#0','ocr#0'))
        self.assertEqual(v.mark_vector(a)[1:3],v.mark_vector(b)[1:3])
        self.assertEqual(v.mark_vector(a),v.mark_vector(b))
        self.assertEqual(v.hazard_tokens(a),v.hazard_tokens(b))
        self.assertEqual(v.hazard_tokens(a),v.hazard_tokens(dict(a,previous_held_owner='ocr#0')))

    def test_training_and_heldout_rows_have_independent_capture_weights(self):
        marks=[];risks=[]
        for capture in ('a','b','heldout'):
            for i,owner in enumerate(v.OWNERS):
                # All physical classes retained, including same held owner.
                marks.append(dict(capture=capture,features=state(age=1000*(i+1)),
                    next_owner=owner,same_owner=owner=='text#0'))
                risks.append(risk(capture,0,1000*(i+1),True))
        fitted=v.fit_models([r for r in marks if r['capture']!='heldout'],
            [r for r in risks if r['capture']!='heldout'])
        self.assertEqual(set(fitted['hazard']['capture_envelopes']),{'a','b'})
        self.assertTrue(fitted['mark']['optimizer']['converged'])
        self.assertEqual(set(fitted['mark']['label_counts']),set(v.OWNERS))
        evaluation=v.evaluate_models(fitted,[r for r in marks if r['capture']=='heldout'],
            [r for r in risks if r['capture']=='heldout'])
        self.assertEqual(evaluation['physical_mark']['all']['n'],4)
        self.assertEqual(evaluation['same_owner_observed_count'],1)
        self.assertEqual(evaluation['physical_SAVE'][0]['observed_SAVE'],4)
        self.assertFalse(evaluation['independent_validation'])

    def test_piecewise_hazard_integral_and_state_specific_support(self):
        value=state(age=1000);tokens=v.hazard_tokens(value)
        candidate=object.__new__(v.ObservableContextCandidate)
        coefficients={'intercept':math.log(2.),'age_bin:1':math.log(3.),'age_bin:2':math.log(5.)}
        candidate.coefficients=coefficients
        candidate.model={'fitted':{'hazard':dict(training_vocabulary=list(tokens),
            state_age_support_ns={v.signature(value):[[0,5000]]})}}
        query=candidate.save_hazard(value,3000)
        expected=2*3*1000/1e6+2*5*2000/1e6
        self.assertAlmostEqual(query['hazard_per_ms'],6.)
        self.assertAlmostEqual(query['conditional_CDF'],-math.expm1(-expected))
        query=candidate.save_hazard(value,10000)
        self.assertIsNone(query['conditional_CDF'])
        self.assertEqual(query['restricted_horizon_ns'],4000)
        unsupported=state(owner='ocr#0')
        self.assertIn('AGE_OUTSIDE_OBSERVED_STATE_RISK',candidate.save_hazard(unsupported)['OOD_reasons'])


if __name__=='__main__':unittest.main()
