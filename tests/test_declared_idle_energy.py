"""Long idle must remain causal, fully accounted, and empirically bounded."""
import copy
import unittest

import numpy as np

from experiments.fit_activation_declared_idle_energy import fit_declared_idle_age_energy
from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_energy import (
    IDLE_EDGES_SECONDS, NativeIdleAgeEnergy, SCHEMA, exposure, fit_idle_age_energy,
)


class DeclaredIdleEnergyTests(unittest.TestCase):
    def test_long_idle_carries_between_jobs_and_retains_measured_bound(self):
        edges = [0., 2., 10., 20., 121.]
        parts = [(i*30_000_000_000, (i+1)*30_000_000_000, [0.]) for i in range(3)]
        with self.assertRaisesRegex(UnknownEvidence, 'DESCRIPTOR_DOMAIN'):
            exposure(parts, [[]], 0)
        full, state, age = exposure(parts, [[]], 0, idle_edges=edges)
        np.testing.assert_allclose(full, [0., 2., 8., 10., 70.])
        self.assertEqual((state, age), (0, 90.))
        accumulated = np.zeros(5)
        carried = 0
        for part in parts:
            row, carried, _ = exposure([part], [[]], carried, idle_edges=edges)
            accumulated += row
        np.testing.assert_allclose(accumulated, full)
        law = NativeIdleAgeEnergy(dict(schema=SCHEMA, status='FITTED_UNVALIDATED',
            admission_enabled=False, idle_edges_seconds=edges,
            independent_run_ids=['a', 'b'], active_terms=[[]],
            coefficients=[100., 90., 60., 40., 30.],
            maximum_observed_idle_age_seconds=90., support_vertices=[[0.], [1.]]))
        joules, carried = law.integrate(parts, 0)
        self.assertEqual(joules, 3160.)
        self.assertEqual(carried, 0)
        with self.assertRaisesRegex(UnknownEvidence, 'MEASURED_SUPPORT'):
            law.integrate([(90_000_000_000, 91_000_000_000, [0.])], 0)
        with self.assertRaisesRegex(UnknownEvidence, 'DESCRIPTOR_DOMAIN'):
            law.integrate([(0, 122_000_000_000, [0.])], 0)

    def test_default_descriptor_reproduces_original_fit_and_does_not_mutate(self):
        profiles = {'p': dict(family='image', task='image', segments=[dict(id='native', kind='gpu')])}
        runs = [dict(run_id=name, split='training', telemetry_errors=[],
            window_ns=[0, 30_000_000_000], initial_last_native_end_ns=0,
            spans=[dict(alias='image#0', profile_id='p', segment_id='native', kind='gpu', start_ns=0,
                        end_ns=5_000_000_000)],
            board_samples=[dict(time_ns=i*100_000_000, watts=50., foreign_pids=[])
                           for i in range(301)]) for name in ['a', 'b']]
        before = copy.deepcopy(runs)
        expected = fit_idle_age_energy(profiles, runs)
        actual = fit_declared_idle_age_energy(profiles, runs, idle_edges=IDLE_EDGES_SECONDS)
        self.assertEqual(actual, expected)
        self.assertEqual(runs, before)
        self.assertEqual(IDLE_EDGES_SECONDS, [0., 2., 10., 20., 31.])
        for edges in ([1., 121.], [0., 10., 2.], [0., float('inf')]):
            with self.assertRaises(ValueError):
                fit_declared_idle_age_energy(profiles, runs, idle_edges=edges)


if __name__ == '__main__':
    unittest.main()
