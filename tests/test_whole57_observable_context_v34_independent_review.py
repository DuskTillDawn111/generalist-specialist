"""Independent input-integrity and probability-accounting fixtures."""
import copy
import math
import unittest
from unittest.mock import patch

import numpy as np

from tools.predictor_validation_20261027 import whole57_observable_context_v34 as v


def features(owner='ocr#0',age=0,active=()):
    return dict(held_owner=owner,previous_held_owner=None,context_age_ns=age,
        handler_active={o:True if o in active else None for o in v.OWNERS})


def risk(start,stop,event=False,qualified=True):
    return dict(capture='a',held_owner='ocr#0',driver_context=[1,0,10,100],restore_start_ns=10000,
        start_ns=10000+start,end_ns=10000+stop,age_entry_ns=start,age_exit_ns=stop,
        physical_SAVE_END_observed=event,exposure_qualified=qualified,shared_left_features=features(age=start))


def accounting_fixture():
    """A qualified mark, an unmarked paired SAVE, and an unpaired SAVE."""
    ctx=[1,0,10,100]
    rows=[dict(risk(0,4,True),physical_SAVE_seqNo=1),
        dict(risk(5,8,True),physical_SAVE_seqNo=2)]
    # The second envelope starts at a new RESTORE, not five units of attained age.
    rows[1].update(restore_start_ns=10005,age_entry_ns=0,age_exit_ns=3,
        shared_left_features=features(age=0))
    native_mark=dict(held_owner='ocr#0',held_driver_context=ctx,current_restore_start_ns=10000,
        save_end_ns=10004,next_owner='text#0',next_driver_context=[1,0,11,101],next_restore_start_ns=10005)
    mark=dict(capture='a',features=features(age=4),held_driver_context=ctx,SAVE_END_ns=10004,
        next_owner='text#0',next_driver_context=[1,0,11,101],next_restore_start_ns=10005,
        same_owner=False,same_driver_context=False)
    events=[dict(capture='a',seqNo=seq,physical_SAVE_END_ns=end,owner='ocr#0',driver_context=ctx,
        restore_start_ns=start,next_owner='text#0' if seq==1 else None)
        for seq,start,end in [(1,10000,10004),(2,10005,10008),(3,None,10009)]]
    docs={'saves':{'events':events},'gaps':{'native_window_ns':[10000,10010],
        'gaps':[[10004,10005],[10008,10010]]},'native':{'qualified':[native_mark]}}
    report=dict(capture='a',native_window_ns=[10000,10010],physical_SAVE_event_ledger='saves',
        native_context_gaps='gaps',physical_SAVE_event_count=3,SAVE_without_next_mark_count=2,
        gap_duration_ns=3,mark_target_counts={'text#0':1},qualified_same_owner_count=0)
    native=dict(capture='a',native_window_ns=[10000,10010],native_transition_ledger='native')
    return report,native,[mark],rows,docs


class ObservableContextIndependentTests(unittest.TestCase):
    def test_candidate_rejects_resigned_malformed_numeric_model_arrays(self):
        marks=[dict(capture='a',features=features(age=10+i),next_owner=o,same_owner=o=='ocr#0')
            for i,o in enumerate(v.OWNERS)]
        fitted=v.fit_models(marks,[risk(0,10000,True)])
        model=dict(schema=v.SCHEMA,parameters=v.PARAMETERS,training_only=True,
            fitted=fitted,sources=[])
        model['model_sha256']=v.digest(model)
        valid=v.ObservableContextCandidate(model)
        self.assertAlmostEqual(sum(valid.next_owner(features(age=10))['physical_owner_probabilities'].values()),1.)
        mutations=(
            lambda m:m['fitted']['mark']['scale'].__setitem__(0,0.),
            lambda m:m['fitted']['mark']['coefficients'][0].__setitem__(0,float('nan')),
            lambda m:m['fitted']['mark']['feature_names'].reverse(),
            lambda m:m['fitted']['hazard']['coefficients'].pop(),
            lambda m:m['fitted']['hazard']['columns'].append('intercept'),
            lambda m:m['fitted']['hazard']['fixed_age_edges_ns'].append(123),
        )
        for mutate in mutations:
            bad=copy.deepcopy(model);mutate(bad)
            with self.subTest(mutation=mutate),self.assertRaises(ValueError):
                bad['model_sha256']=v.digest({k:value for k,value in bad.items() if k!='model_sha256'})
                v.ObservableContextCandidate(bad)

    def test_capture_event_normalization_rejects_capture_without_events(self):
        marks=[dict(capture='a',features=features(age=10+i),next_owner=o,same_owner=o=='ocr#0')
            for i,o in enumerate(v.OWNERS)]
        with self.assertRaisesRegex(ValueError,'every training capture'):
            v.fit_models(marks,[risk(0,10000,True),dict(risk(0,10000,False),capture='b')])

    def test_exact_native_accounting_preserves_unmarked_and_unpaired_SAVE(self):
        report,native,marks,risks,docs=accounting_fixture()
        with patch.object(v,'checked',side_effect=docs.__getitem__):
            v.verify_native_accounting(report,native,marks,risks)
        self.assertEqual(sum(r['physical_SAVE_END_observed'] for r in risks),2)
        self.assertEqual(report['physical_SAVE_event_count'],3)

    def test_native_SAVE_seqNo_can_repeat_without_merging_distinct_physical_events(self):
        report,native,marks,risks,docs=accounting_fixture()
        for event in docs['saves']['events']:event['seqNo']=129
        for row in risks:row['physical_SAVE_seqNo']=129
        with patch.object(v,'checked',side_effect=docs.__getitem__):
            v.verify_native_accounting(report,native,marks,risks)
        # Equal seqNo at distinct physical timestamps cannot hide a missing risk event.
        risks[1]['physical_SAVE_END_observed']=False
        with patch.object(v,'checked',side_effect=docs.__getitem__),self.assertRaises(ValueError):
            v.verify_native_accounting(report,native,marks,risks)

    def test_native_accounting_rejects_missing_duplicate_SAVE_gap_and_mark_mutations(self):
        mutations=(
            lambda r,n,m,rs,d:rs[1].update(physical_SAVE_END_observed=False),
            lambda r,n,m,rs,d:rs[1].update(physical_SAVE_seqNo=1),
            lambda r,n,m,rs,d:d['gaps']['gaps'][0].__setitem__(1,10006),
            lambda r,n,m,rs,d:m[0].update(next_owner='text#1'),
            lambda r,n,m,rs,d:m.append(copy.deepcopy(m[0])),
        )
        for mutate in mutations:
            fixture=accounting_fixture();mutate(*fixture)
            report,native,marks,risks,docs=fixture
            with self.subTest(mutation=mutate),patch.object(v,'checked',side_effect=docs.__getitem__):
                with self.assertRaises(ValueError):v.verify_native_accounting(report,native,marks,risks)

    def test_all_unresolved_rows_require_positive_original_time_and_boolean_flags(self):
        good=risk(0,1000,True)
        for bad in (risk(2000,1000,True,False),risk(1000,1000,False,False),
                    dict(risk(0,1000,False,False),physical_SAVE_END_observed=1)):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):v.aggregate([good,bad])

    def test_feature_holder_restore_age_cannot_disagree_with_native_identity(self):
        for bad in (dict(risk(0,1000,True),held_owner='text#0'),
                    dict(risk(0,1000,True),restore_start_ns=9999)):
            with self.assertRaises(ValueError):v.aggregate([bad])

    def test_handler_diagnostics_and_future_labels_do_not_change_model_features(self):
        a=features(age=77768);b=features(age=77768,active=tuple(v.OWNERS))
        self.assertEqual(v.mark_vector(a),v.mark_vector(b))
        self.assertEqual(v.hazard_tokens(a),v.hazard_tokens(b))
        changed=dict(a,previous_held_owner='text#0')
        self.assertNotEqual(v.mark_vector(a),v.mark_vector(changed))
        self.assertEqual(v.hazard_tokens(a),v.hazard_tokens(changed))
        with self.assertRaises(ValueError):v.mark_vector(dict(a,next_owner='text#1'))

    def test_real_Poisson_fit_keeps_zero_event_risk_and_is_split_invariant(self):
        original=[risk(0,10000,True),risk(20000,30000,False)]
        split=[risk(0,4000,False),risk(4000,10000,True),risk(20000,30000,False)]
        data=v.aggregate(original);split_data=v.aggregate(split)
        fit=v.normalized_fit(data);other=v.normalized_fit(split_data)
        self.assertTrue(fit['optimizer']['success']);self.assertTrue(other['optimizer']['success'])
        self.assertEqual(fit['columns'],other['columns'])
        self.assertTrue(np.allclose(fit['coefficients'],other['coefficients'],atol=1e-8))
        candidate=object.__new__(v.ObservableContextCandidate);candidate.model={'fitted':{'hazard':dict(fit,state_age_support_ns=data['state_age_support_ns'])}}
        candidate.coefficients=dict(zip(fit['columns'],fit['coefficients']))
        self.assertGreater(candidate.save_hazard(features(age=25000))['hazard_per_ms'],0.)
        self.assertIn('AGE_OUTSIDE_OBSERVED_STATE_RISK',candidate.save_hazard(features(age=15000))['OOD_reasons'])


if __name__=='__main__':unittest.main()
