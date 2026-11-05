import unittest

from experiments.solve_bilevel_v3 import ROOT, solve
from gsdeploy.config import SystemConfig


class SchedulerEntrypointTests(unittest.TestCase):
    def config(self):
        return SystemConfig.load(ROOT/'configs/paper_fifo_reference_v2_cpu_fixture.json')

    def test_generic_v3_defaults_to_balanced_with_bounded_work(self):
        result = solve(self.config(), seconds=10, predictions=16, routes=8, milp=32)
        self.assertEqual(result['scheduler_mode'], 'balanced')
        self.assertEqual(result['scheduler']['version'], 'bounded_interleaving_v1')
        self.assertLessEqual(result['budget']['outer_evaluations'],8)
        self.assertLessEqual(result['budget']['full_prediction_calls'],16)
        self.assertFalse(result['physical_qualification_granted'])
        self.assertIsNotNone(result['winner'])

    def test_legacy_option_remains_explicit_and_invalid_option_rejected(self):
        result = solve(self.config(),scheduler='legacy',seconds=10,predictions=16,routes=8,milp=32)
        self.assertEqual(result['scheduler_mode'],'legacy')
        self.assertNotIn('scheduler',result)
        with self.assertRaisesRegex(ValueError,'Unknown bilevel scheduler'):
            solve(self.config(),scheduler='other')


if __name__=='__main__':
    unittest.main()
