"""Independent statistical counterexamples; synthetic CPU intervals only."""
import copy
from fractions import Fraction
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.context_survival_v12 import ContextEnvelopeSurvival,SCHEMA
from tools.predictor_validation_20261027.profiling import digest

F=[['pending','text','exact',['phase']]]


def interval(c,e,a,b,event):
    return dict(capture=c,envelope_id=e,held_owner='held',frontier=copy.deepcopy(F),
                age_entry_ns=a,age_exit_ns=b,save_observed=event)


def provider(rows,**kwargs):
    model=dict(schema=SCHEMA,training_only=True,episodes=rows);model['model_sha256']=digest(model)
    return ContextEnvelopeSurvival(model,allow_unvalidated_survival_hypothesis=True,allow_cpu_fixture=True,**kwargs)


class ContextSurvivalIndependentTests(unittest.TestCase):
    def test_delayed_entry_risk_denominator_conditional_age_and_restricted_integral(self):
        rows=[interval('a','one',0,10,True),interval('a','two',5,20,True),interval('a','tail',0,30,False),
              interval('b','one',0,10,True),interval('b','two',0,20,True),interval('b','tail',15,30,False)]
        fit=provider(rows);r=fit.query('held',F,0)
        # t10: 2 events / 5 at risk; t20: 2 / 4 remaining.
        first=Fraction(2,5);second=(1-first)*Fraction(2,4);tail=1-first-second
        self.assertEqual([m['residual_ns'] for m in r['residual_event_mass']],[10,20])
        self.assertAlmostEqual(r['residual_event_mass'][0]['probability'],float(first))
        self.assertAlmostEqual(r['residual_event_mass'][1]['probability'],float(second))
        self.assertAlmostEqual(r['unresolved_tail_probability'],float(tail))
        self.assertAlmostEqual(r['restricted_mean_residual_ns'],float(10+(1-first)*10+tail*10))
        self.assertEqual(r['median_residual_ns'],20)
        aged=fit.query('held',F,12)
        self.assertEqual(aged['residual_event_mass'],[dict(residual_ns=8,probability=.5)])
        self.assertEqual(aged['restricted_mean_residual_ns'],13.)
        self.assertEqual(aged['median_residual_ns'],8)
        self.assertEqual(aged['unresolved_tail_probability'],.5)

    def test_adjacent_same_envelope_censor_and_reentry_do_not_double_risk(self):
        fit=provider([interval('a','same',0,5,False),interval('a','same',5,10,True),interval('b','one',0,10,True)])
        r=fit.query('held',F,0)
        self.assertEqual(r['residual_event_mass'],[dict(residual_ns=10,probability=1.)])
        self.assertEqual(r['unresolved_tail_probability'],0.)
        self.assertEqual(r['restricted_mean_residual_ns'],10.)
        self.assertEqual(fit.query('held',F,5)['median_residual_ns'],5)

    def test_gap_in_any_capture_stops_forecast_and_exact_age_exit_is_outside_support(self):
        fit=provider([interval('a','one',0,10,True),interval('a','late',12,20,True),interval('b','one',0,30,False)])
        r=fit.query('held',F,0)
        self.assertEqual(r['restricted_residual_horizon_ns'],10)
        # Capture a has two envelope weights (.25 each), b one (.5):
        # only a's first and b's one are at risk at t=10.
        self.assertAlmostEqual(r['residual_event_mass'][0]['probability'],1/3)
        self.assertEqual(r['residual_event_mass'][0]['residual_ns'],10)
        self.assertAlmostEqual(r['unresolved_tail_probability'],2/3)
        for age in (10,11,20,30):
            with self.subTest(age=age),self.assertRaises(UnknownEvidence):fit.query('held',F,age)
        self.assertEqual(fit.query('held',F,12)['restricted_residual_horizon_ns'],8)

    def test_all_censored_tail_stays_one_and_returns_no_median_or_next_owner(self):
        fit=provider([interval('a','one',2,20,False),interval('b','one',0,30,False)])
        r=fit.query('held',F,5)
        self.assertEqual(r['residual_event_mass'],[]);self.assertEqual(r['unresolved_tail_probability'],1.)
        self.assertEqual(r['restricted_mean_residual_ns'],15.)
        self.assertIsNone(r['median_residual_ns']);self.assertFalse(r['complete_finite_residual_distribution'])
        self.assertFalse(r['next_owner_policy_identified']);self.assertFalse(r['pending_work_and_waiting_age_conditioned'])
        with self.assertRaises(UnknownEvidence):fit.query('pending',F,5)
        with self.assertRaises(UnknownEvidence):fit.query('held',[['pending','text','unseen',['phase']]],5)


if __name__=='__main__':unittest.main()
