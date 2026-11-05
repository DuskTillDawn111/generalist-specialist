import copy
import unittest

from experiments.accept_bilevel_algorithm import exhaustive, initial_signature, verify_inner
from gsdeploy.continuous.contracts import load_experiment
from tests import test_continuous_routing as fixture


class FinalAcceptanceTests(unittest.TestCase):
    def test_initialization_identity_includes_scores_and_charged_work(self):
        raw = dict(events=[dict(source='initialization', routing_shares={'text': {'g': 1}}, lb=-1,
                                ub=-1, certified=True)],
            convergence=[dict(source='initialization', full_prediction_calls=2,
                outer_evaluations=1, milp_solve_calls=2, best_feasible_utility=-1)])
        for changed in ('lower', 'calls', 'route'):
            other = copy.deepcopy(raw)
            if changed == 'lower': other['events'][0]['lb'] = -2
            if changed == 'calls': other['convergence'][0]['full_prediction_calls'] = 1
            if changed == 'route': other['events'][0]['routing_shares']['text']['g'] = .5
            self.assertNotEqual(initial_signature(raw), initial_signature(other))

    def test_empty_initialization_is_not_matched_evidence(self):
        with self.assertRaises(ValueError):
            initial_signature(dict(events=[], convergence=[]))

    def test_partial_bounds_and_false_certificates(self):
        truth = dict(U=-1)
        row = dict(best_feasible_utility=-2, valid_upper_bound=0, certified=False)
        verify_inner(row, truth)
        for bad in (dict(row, certified=True), dict(row, valid_upper_bound=-1.5),
                    dict(row, best_feasible_utility=-.5)):
            with self.assertRaises(ValueError): verify_inner(bad, truth)
        verify_inner(dict(best_feasible_utility=-1, valid_utility_upper_bound=-1, certified=True), truth)

    def test_infeasible_domain_cannot_get_a_feasible_score(self):
        with self.assertRaises(ValueError):
            verify_inner(dict(best_feasible_utility=-1, valid_upper_bound=0, certified=False), dict(U=None))

    def test_independent_reference_exercises_zero_and_two_replica_states(self):
        _, p = load_experiment(fixture.CONFIG)
        def utility(x, routing, trace):
            return -1. if x == {'g': 2} else -3.
        ref = exhaustive(p, fixture.TableOracle(utility), {t: {'g': 1.} for t in p.config.workload})
        self.assertEqual(ref['domain_size'], 18)
        self.assertEqual(ref['deployment'], {'g': 2})
        self.assertEqual(ref['U'], -1.)
        self.assertTrue(any(r['deployment'].get('s_text') == 2 for r in ref['records']))


if __name__ == '__main__':
    unittest.main()
