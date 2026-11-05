"""Diagnostic probes preserve solver decisions and close their time ledger."""

import unittest

from gsdeploy.continuous import Budget, ContinuousRoutingEvaluator
from tests.test_continuous_routing import TableOracle, tiny
from experiments.continuous_diagnostic_timing import PhaseClock, diagnostics


class DiagnosticTests(unittest.TestCase):
    def test_fixed_solver_probe_preserves_result_and_restores_methods(self):
        import gsdeploy.continuous.fixed_routing_milp as module
        original=module.FixedRoutingMilpOptimizer.solve
        route={'text':{'g':1},'image':{'g':1}}
        expected=ContinuousRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1 if 's_shared' in x else -.8),'milp').evaluate(route)
        clock=PhaseClock()
        with diagnostics(clock):
            e=ContinuousRoutingEvaluator(tiny(),TableOracle(lambda x,r,t:-.1 if 's_shared' in x else -.8),'milp',Budget(20))
            actual=e.evaluate(route)
            report=clock.report()
        self.assertIs(module.FixedRoutingMilpOptimizer.solve,original)
        for key in ('deployment','best_feasible_utility','certified','full_prediction_calls','milp_solve_calls'):
            self.assertEqual(expected[key],actual[key])
        self.assertEqual(len([r for r in report['rounds'] if 'solver_status' in r]),actual['milp_solve_calls'])
        self.assertLess(abs(report['accounting_error_seconds']),1e-7)
        self.assertIn('highs_solver',report['seconds'])

    def test_restores_on_exception(self):
        from gsdeploy.milp import JointMilpOptimizer
        original=JointMilpOptimizer.solve
        with self.assertRaisesRegex(RuntimeError,'probe'):
            with diagnostics(PhaseClock()):
                raise RuntimeError('probe')
        self.assertIs(JointMilpOptimizer.solve,original)

    def test_joint_solver_probe_preserves_complete_candidate_scoring(self):
        from dataclasses import replace
        from gsdeploy import DeploymentSystem
        from tests.test_continuous_routing import fixture
        _,problem=fixture()
        c=replace(problem.config,weights=dict(Q=0.,T=.2,L=.4,P=.2,M=.2),
                  utility_schema='legacy_four_term',milp_capacity_witness=False)
        options={m:[1] for m in c.catalog['models']}
        expected=DeploymentSystem(c).plan(replica_options=options,route_grid=2)
        clock=PhaseClock()
        with diagnostics(clock):
            actual=DeploymentSystem(c).plan(replica_options=options,route_grid=2)
            report=clock.report()
        self.assertEqual(expected['winner'],actual['winner'])
        self.assertEqual(expected['optimality_certified'],actual['optimality_certified'])
        self.assertEqual(sum(r['prediction_bundles'] for r in report['rounds']),
                         actual['metrics']['full_prediction_calls'])
        self.assertLess(abs(report['accounting_error_seconds']),1e-7)


if __name__=='__main__':
    unittest.main()
