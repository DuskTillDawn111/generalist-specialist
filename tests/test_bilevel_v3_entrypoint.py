import copy
from dataclasses import replace
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.solve_bilevel_v3 import solve, GroupedReferenceOracle, OBJECTIVE
from gsdeploy.config import SystemConfig
from gsdeploy.continuous import ImmutableContinuousProblem, SessionPredictorOracle, Budget
from gsdeploy.continuous.contracts import digest
from gsdeploy.utility_tradeoffs import SCHEMA

ROOT = Path(__file__).resolve().parents[1]


class FinalV3EntrypointTests(unittest.TestCase):
    def config(self):
        return SystemConfig.load(ROOT/'configs/paper_fifo_reference_v2_cpu_fixture.json')

    def test_full_controller_runs_declared_multi_deployment_v3(self):
        c = self.config()
        before = copy.deepcopy(c.as_dict())
        result = solve(c, seconds=10, predictions=12, routes=6)
        self.assertEqual(c.as_dict(), before)
        self.assertIsNotNone(result['winner'])
        self.assertEqual(result['declared_replica_states'], {'g': [0, 1], 's': [0, 1]})
        self.assertEqual(result['objective_contract_sha256'], digest(OBJECTIVE))
        prediction = result['winner']['prediction']['traces'][0]['prediction']
        self.assertEqual(prediction['utility_schema'], SCHEMA)
        self.assertAlmostEqual(prediction['Q_contribution'], -.25*prediction['Q_norm'])
        self.assertFalse(result['physical_qualification_granted'])
        self.assertLessEqual(result['budget']['full_prediction_calls'], 12)

    def test_oracle_keeps_unknown_unknown_and_identity_distinct(self):
        p = ImmutableContinuousProblem(replace(self.config(), weights=OBJECTIVE['weights']))
        with SessionPredictorOracle(p) as base:
            oracle = GroupedReferenceOracle(base, p)
            self.assertNotEqual(oracle.version, base.version)
            with patch.object(base, 'predict', return_value=dict(U=None, performance_unknown=True)):
                out = oracle.predict(p, {'g': 1}, {'text': {'g': 2}}, p.train_traces[0], Budget())
            self.assertIsNone(out['U'])
            self.assertTrue(out['performance_unknown'])
            with self.assertRaises(TypeError): p.objective_id = 'changed'
        with self.assertRaises(ValueError): oracle.check()

    def test_reject_mixed_weights_and_unsupported_policy(self):
        p = ImmutableContinuousProblem(self.config())
        with SessionPredictorOracle(p) as base, self.assertRaises(ValueError):
            GroupedReferenceOracle(base, p)
        with self.assertRaises(ValueError): solve(replace(self.config(), search_policy='robust'))


if __name__ == '__main__':
    unittest.main()
