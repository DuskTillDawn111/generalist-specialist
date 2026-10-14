"""Regression for numerical residual flow on an absent model in the probe."""

import unittest
from dataclasses import replace
from unittest.mock import patch
from scipy.optimize import milp as real_milp
from gsdeploy import DeploymentSystem
from tests.test_search_contract import config


class CapacityWitnessTests(unittest.TestCase):
    def test_tiny_absent_flow_does_not_discard_the_capacity_witness(self):
        c = replace(
            config({"text": 1.0}, {"g": {"text": 2.0}, "s": {"text": 4.0}}),
            solver="milp",
            milp_capacity_witness=True,
        )
        calls = 0

        def residual(*args, **kwargs):
            nonlocal calls
            fit = real_milp(*args, **kwargs)
            if calls == 0:
                # Probe states are g0,g1,s0,s1; flow paths are text/g,text/s.
                self.assertGreater(fit.x[1], 0.5)
                self.assertLess(fit.x[3], 0.5)
                fit.x[4] -= 1e-14
                fit.x[5] = 1e-14
            calls += 1
            return fit

        with patch("gsdeploy.milp.milp", residual):
            r = DeploymentSystem(c).plan(route_grid=2)
        witness = r["domain"]["capacity_witness"]
        self.assertIsNotNone(witness)
        self.assertEqual(witness["deployment"], {"g": 1})
        self.assertEqual(witness["routing"], {"text": {"g": 1.0}})
        self.assertTrue(r["constraints"]["constraints_valid"])


if __name__ == "__main__":
    unittest.main()
