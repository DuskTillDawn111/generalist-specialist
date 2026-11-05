import copy
import math
import unittest

from gsdeploy.utility_tradeoffs import (TERMS, WEIGHT_PROFILES, compare_panel,
                                       episode_metrics, score_terms)


class UtilityTradeoffTests(unittest.TestCase):
    def evaluate(self, errors=(0, 0, 1, 1), latency=(1, 1, 3, 3), task='text', units=None, **changes):
        rows = [dict(request_id=str(i), task=task, success=True, latency_ms=l,
                     errors=e, units=1 if units is None else units[i])
                for i, (e, l) in enumerate(zip(errors, latency))]
        kwargs = dict(expected_tasks_by_id={r['request_id']: task for r in rows},
            task_references={task: dict(best_loss=.1, worst_loss=.5, deadline_ms=2, latency_scale_ms=2)},
            task_weights={task: 1}, episode_joules=20., energy_scale_joules=100.,
            owned_peak_mib=10., memory_budget_mib=100.)
        kwargs.update(changes)
        return episode_metrics(rows, **kwargs)

    def test_identical_marginals_hide_different_joint_delivery(self):
        a = self.evaluate()
        b = self.evaluate(errors=(1, 1, 0, 0))
        self.assertEqual(a['terms'], b['terms'])
        self.assertEqual(a['per_task']['text']['correct_and_timely_fraction'], .5)
        self.assertEqual(b['per_task']['text']['correct_and_timely_fraction'], 0)
        # Multiplying .5 accuracy by .5 timely would incorrectly give .25 for both.

    def test_edit_errors_use_total_units_and_can_exceed_one(self):
        result = self.evaluate(task='asr', errors=(8, 0), units=(2, 4), latency=(1, 3))
        self.assertAlmostEqual(result['per_task']['asr']['native_loss'], 8 / 6)
        self.assertIsNone(result['per_task']['asr']['correct_and_timely_fraction'])
        self.assertNotEqual(result['per_task']['asr']['native_loss'], (8 / 2 + 0 / 4) / 2)

    def test_deadline_changes_do_not_change_latency_or_quality_scales(self):
        a = self.evaluate()
        b = self.evaluate(task_references={'text': dict(best_loss=.1, worst_loss=.5, deadline_ms=3, latency_scale_ms=2)})
        self.assertEqual(a['terms']['T'], .5)
        self.assertEqual(b['terms']['T'], 0)  # Equality with deadline counts as timely.
        for t in ('Q', 'L', 'P', 'M', 'E'):
            self.assertEqual(a['terms'][t], b['terms'][t])

    def test_missing_duplicate_failed_or_task_mismatched_cohort_rejected(self):
        row = dict(request_id='a', task='text', success=True, latency_ms=1, errors=0, units=1)
        kwargs = dict(expected_tasks_by_id={'a': 'text'},
            task_references={'text': dict(best_loss=0, worst_loss=1, deadline_ms=2, latency_scale_ms=2)},
            task_weights={'text': 1}, episode_joules=1, energy_scale_joules=1, owned_peak_mib=1, memory_budget_mib=1)
        for rows in ([], [row, row], [dict(row, success=False)], [dict(row, task='image')],
                     [dict(row, latency_ms=float('nan'))], [dict(row, units=0)],
                     [dict(row, errors=.5)], [dict(row, errors=float('inf'))]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                episode_metrics(rows, **kwargs)
        with self.assertRaises(ValueError):
            episode_metrics([row], **dict(kwargs, expected_tasks_by_id={'a': 'text', 'b': 'text'}))
        with self.assertRaises(ValueError):
            episode_metrics([row], **dict(kwargs, task_references={**kwargs['task_references'], 'ocr': kwargs['task_references']['text']},
                                        task_weights={'text': .5, 'ocr': .5}))

    def test_invalid_scales_and_reference_spans_rejected(self):
        for changes in ({'energy_scale_joules': 0}, {'memory_budget_mib': -1}, {'episode_joules': None},
                        {'owned_peak_mib': float('nan')}, {'task_weights': {'text': .5}},
                        {'task_references': {'text': dict(best_loss=.5, worst_loss=.5, deadline_ms=2, latency_scale_ms=2)}}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.evaluate(**changes)

    def test_missing_terms_nan_or_invalid_weights_never_produce_a_ranking(self):
        base = dict.fromkeys(TERMS, 1.)
        for bad in ({k: v for k, v in base.items() if k != 'E'}, dict(base, Q=None), dict(base, L=math.nan)):
            with self.assertRaises(ValueError):
                score_terms(bad, WEIGHT_PROFILES['four_groups_v3'])
        for bad in (dict.fromkeys(TERMS, .1), dict(Q=1), dict(WEIGHT_PROFILES['four_groups_v3'], Q=-1)):
            with self.assertRaises(ValueError):
                score_terms(base, bad)

    def test_grouped_weights_and_monotonicity(self):
        base = dict(Q=-.5, T=.5, L=1., P=3., M=.8, E=.4)
        for weights in WEIGHT_PROFILES.values():
            result = score_terms(base, weights)
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertGreater(result['contributions']['Q'], 0)  # Better than best anchor, no clamping.
            for term in TERMS:
                worse = dict(base)
                worse[term] += .1
                self.assertLess(score_terms(worse, weights)['U'], result['U'])
        self.assertAlmostEqual(sum(WEIGHT_PROFILES['four_groups_v3'][t] for t in ('T', 'L', 'P')), .25)

    def test_scale_sensitivity_changes_only_named_terms(self):
        terms = dict(Q=.3, T=.2, L=.6, P=.8, M=.7, E=.4)
        result = score_terms(terms, WEIGHT_PROFILES['four_groups_v3'], energy_scale_factor=2, quality_span_factor=.5)
        self.assertEqual(result['terms'], dict(terms, Q=.6, E=.2))
        self.assertEqual(terms['Q'], .3)

    def panel(self):
        return [dict(candidate_id=c, round=i, observed_feasible=True, predicted_feasible=True,
            observed=score_terms(dict.fromkeys(TERMS, cost), WEIGHT_PROFILES['four_groups_v3']),
            predicted=score_terms(dict.fromkeys(TERMS, cost), WEIGHT_PROFILES['four_groups_v3']))
            for c, cost in [('a', .1), ('b', .2)] for i in (1, 2)]

    def test_predicted_infeasible_candidate_cannot_be_selected(self):
        rows = self.panel()
        rows[0]['predicted_feasible'] = False
        result = compare_panel(rows)
        self.assertEqual(result['predicted_selected'], 'b')
        self.assertAlmostEqual(result['mean_regret'], .1)

    def test_observed_infeasible_selection_has_unknown_regret(self):
        rows = self.panel()
        rows[0]['observed_feasible'] = False
        result = compare_panel(rows)
        self.assertEqual(result['predicted_selected'], 'a')
        self.assertIsNone(result['mean_regret'])
        self.assertIsNone(result['per_round_regret'][0])
        self.assertEqual(result['observed_best'], ['b'])

    def test_no_feasible_predictions_and_duplicate_panel(self):
        rows = self.panel()
        for row in rows:
            row['predicted_feasible'] = False
        self.assertIsNone(compare_panel(rows)['predicted_selected'])
        for bad in (rows[:-1], [rows[0], rows[0], *rows[2:]]):
            with self.assertRaises(ValueError):
                compare_panel(bad)

    def test_bad_round_not_hidden_by_mean_and_ties_are_explicit(self):
        rows = self.panel()
        rows[0]['observed'] = score_terms(dict.fromkeys(TERMS, .25), WEIGHT_PROFILES['four_groups_v3'])
        result = compare_panel(rows)
        self.assertEqual(result['mean_regret'], 0)
        self.assertAlmostEqual(result['per_round_regret'][0], .05)
        rows[2]['predicted'] = copy.deepcopy(rows[0]['predicted'])
        rows[3]['predicted'] = copy.deepcopy(rows[1]['predicted'])
        self.assertEqual(compare_panel(rows)['predicted_tied_winners'], ['a', 'b'])


if __name__ == '__main__':
    unittest.main()
