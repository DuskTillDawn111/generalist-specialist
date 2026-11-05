"""Independent domain counts and instrumentation equivalence for Experiment A."""

from fractions import Fraction
import itertools
import unittest

from gsdeploy.route_domain import fraction_rows
from gsdeploy.search_session import SearchSession
from tests.test_calibrated_search_bounds_v6 import search_fixture
from experiments.joint_scalability import ROOT, domain_audit, probes


class JointScalabilityTests(unittest.TestCase):
    def test_grid_is_nested_rational_union_and_counts_are_exact(self):
        rows=list(fraction_rows(('a','b'),4))
        self.assertEqual(len(rows),7)
        self.assertIn((('a',Fraction(1,3)),('b',Fraction(2,3))),rows)
        self.assertEqual(len(list(fraction_rows(('a','b'),8))),23)
        self.assertTrue(set(rows)<=set(fraction_rows(('a','b'),8)))

    def test_raw_domain_audit_matches_independent_cartesian_counts(self):
        c=search_fixture()
        states={m:[1,2] for m in c.catalog['models']}
        audit=domain_audit(c,states)
        self.assertEqual(audit['raw_deployment_combinations'],9)
        # K=1; two pure models with two counts each for each of two tasks.
        self.assertEqual(audit['raw_task_count_columns'],8)
        self.assertEqual(audit['memory_bound_survivors'],9)
        self.assertEqual(audit['memory_and_task_cover_survivors'],8)

    def test_probes_keep_complete_prediction_and_certificate_and_restore(self):
        c=search_fixture()
        with SearchSession(ROOT,c.as_dict()) as session:
            module=session.module('milp')
            before=session.optimizer('milp').solve(K=2)
            original=module.solve_master
            optimizer=session.optimizer('milp')
            with probes(module,optimizer) as timing:
                after=optimizer.solve(K=2)
            self.assertIs(module.solve_master,original)
            self.assertEqual(before['winner'],after['winner'])
            self.assertEqual(before['optimality_certified'],after['optimality_certified'])
            self.assertEqual(before['metrics']['full_prediction_calls'],timing['prediction_calls'])
            self.assertEqual(timing['columns_returned'],after['metrics']['task_columns'])
            self.assertGreater(timing['highs_seconds'],0)
            self.assertGreaterEqual(timing['master_seconds'],timing['highs_seconds'])
            self.assertTrue(all(r['variables']>0 and r['constraints']>0 for r in timing['rounds']))
            with self.assertRaises(RuntimeError):
                with probes(module):
                    raise RuntimeError('restore on exception')
            self.assertIs(module.solve_master,original)


if __name__=='__main__':
    unittest.main()
