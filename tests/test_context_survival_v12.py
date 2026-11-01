"""Age, censoring, capture balance and unsupported empirical tail checks."""
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.context_survival_v12 import ContextEnvelopeSurvival,SCHEMA
from tools.predictor_validation_20261027.profiling import digest


FRONTIER = [['ready','text','input',['prefill']]]


def episode(capture,eid,end,*,begin=0,event=True):
    return dict(capture=capture,envelope_id=eid,held_owner='held',frontier=FRONTIER,
                age_entry_ns=begin,age_exit_ns=end,save_observed=event)


def model(rows):
    result = dict(schema=SCHEMA,training_only=True,episodes=rows)
    result['model_sha256'] = digest(result)
    return result


def provider(rows,**kwargs):
    return ContextEnvelopeSurvival(model(rows),allow_unvalidated_survival_hypothesis=True,
                                   allow_cpu_fixture=True,**kwargs)


class ContextSurvivalTests(unittest.TestCase):
    def test_residual_uses_real_age_and_preserves_nonpending_holder(self):
        rows = [episode(c,str(t),t) for c in ('a','b') for t in (10,20)]
        fit = provider(rows)
        initial = fit.query('held',FRONTIER,0)
        self.assertEqual(initial['median_residual_ns'],10)
        aged = fit.query('held',FRONTIER,15)
        self.assertEqual(aged['median_residual_ns'],5)
        self.assertEqual(aged['unresolved_tail_probability'],0.)
        self.assertFalse(aged['context_age_reset'])
        with self.assertRaises(UnknownEvidence):
            fit.query('ready',FRONTIER,0)

    def test_right_censored_tail_never_becomes_zero_or_event(self):
        rows = [episode(c,'ten',10) for c in ('a','b')]
        rows += [episode(c,'tail',20,event=False) for c in ('a','b')]
        result = provider(rows).query('held',FRONTIER,0)
        self.assertEqual(result['residual_event_mass'],[dict(residual_ns=10,probability=.5)])
        self.assertEqual(result['unresolved_tail_probability'],.5)
        self.assertFalse(result['complete_finite_residual_distribution'])
        self.assertFalse(result['tail_filled_with_zero'])
        self.assertEqual(result['restricted_mean_residual_ns'],15.)

    def test_delayed_entry_cannot_fill_unobserved_age_gap(self):
        rows = [episode(c,'early',10,event=False) for c in ('a','b')]
        rows += [episode(c,'late',20,begin=12) for c in ('a','b')]
        fit = provider(rows)
        result = fit.query('held',FRONTIER,0)
        self.assertEqual(result['restricted_residual_horizon_ns'],10)
        self.assertEqual(result['unresolved_tail_probability'],1.)
        self.assertIsNone(result['median_residual_ns'])
        with self.assertRaises(UnknownEvidence):
            fit.query('held',FRONTIER,11)
        self.assertEqual(fit.query('held',FRONTIER,15)['median_residual_ns'],5)

    def test_capture_balance_is_not_kernel_count_weighting(self):
        rows = [episode('a',str(i),10) for i in range(10)]+[episode('b','one',20)]
        result = provider(rows).query('held',FRONTIER,0)
        self.assertAlmostEqual(result['residual_event_mass'][0]['probability'],.5)
        # The second capture has no age>10 support in the first capture;
        # stopping at the common endpoint retains the other half as unknown.
        self.assertAlmostEqual(result['unresolved_tail_probability'],.5)
        self.assertEqual(result['restricted_residual_horizon_ns'],10)

    def test_each_independent_capture_must_support_query_age(self):
        fit = provider([episode('a','one',10),episode('b','two',20)])
        with self.assertRaises(UnknownEvidence):
            fit.query('held',FRONTIER,15)
        with self.assertRaises(UnknownEvidence):
            provider([episode('a','one',10)]).query('held',FRONTIER,0)
        result = provider([episode('a','one',10)],allow_single_capture_training_diagnostic=True).query('held',FRONTIER,0)
        self.assertEqual(result['capture_count'],1)
        self.assertFalse(result['independent_prediction_accuracy_validated'])

    def test_invalid_episodes_and_overlapping_same_envelope_rejected(self):
        for rows in ([episode('a','one',0)],
                     [episode('a','one',10),episode('a','one',20,begin=5)]):
            with self.assertRaises(ValueError):
                provider(rows)


if __name__ == '__main__':
    unittest.main()
