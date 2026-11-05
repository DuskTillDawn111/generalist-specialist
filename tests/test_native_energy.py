"""Causal board-energy accounting; CPU fixtures do not qualify measurements."""
import numpy as np
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.native_energy import exposure, NativeIdleAgeEnergy, SCHEMA, IDLE_EDGES_SECONDS
from gsdeploy.forecasting.native_context import NativeContextProvider, fit_native_progress
from gsdeploy.forecasting.simulator import FullHorizonModel
from tests.test_native_activity_context import fixture


class NativeEnergyTests(unittest.TestCase):
    def test_fifo_uses_carried_idle_age_and_unknown_prefix_is_not_cold(self):
        template, observations = fixture()
        template['native_activity_law'] = fit_native_progress(template, observations)
        template['native_idle_age_energy'] = dict(
            schema=SCHEMA, status='FITTED_UNVALIDATED', admission_enabled=False,
            execution_context_sha256='fixture', categories=[['a','text'],['b','image']],
            idle_edges_seconds=IDLE_EDGES_SECONDS, independent_run_ids=['one','two'],
            active_terms=[[]], coefficients=[150., 90., 90., 90., 30.],
            maximum_observed_idle_age_seconds=30., support_vertices=[[0.,0.],[1.,0.],[0.,1.]])
        provider = NativeContextProvider(template,'fixture',allow_cpu_fixture=True)
        model = FullHorizonModel({'a#0':dict(family='a',supported_tasks=['text'])}, provider)
        empty = model.forecast({},20_000_000_000,25_000_000_000,
                               energy_state=0,require_episode_energy=True)
        self.assertEqual(empty['through_drain_board_energy_joules'],150.)
        self.assertEqual(empty['final_energy_state'],0)
        unknown = model.forecast({},20_000_000_000,25_000_000_000,
                                 require_episode_energy=True,require_energy=False)
        self.assertFalse(unknown['energy_known'])
        with self.assertRaisesRegex(UnknownEvidence,'PREFIX'):
            model.forecast({},20_000_000_000,25_000_000_000,require_episode_energy=True)
        queues = {'a#0':[dict(request_id='a',alias='a#0',task='text',pool_id='a1',arrival_ns=20_000_000_000)]}
        busy = model.forecast(queues,20_000_000_000,20_025_000_000,
                              energy_state=0,require_episode_energy=True)
        self.assertAlmostEqual(busy['through_drain_board_energy_joules'],2.85,places=7)
        self.assertEqual(busy['final_energy_state'],20_010_000_000)

    def test_busy_to_idle_energy_carries_across_calls_and_partial_blocks(self):
        terms = [[], [0]]
        windows = [(0, 1_000_000_000, [1]), (1_000_000_000, 16_000_000_000, [0])]
        row, state, age = exposure(windows, terms, 0)
        a, carried, _ = exposure(windows[:1], terms, 0)
        b, carried, _ = exposure([(1_000_000_000, 6_000_000_000, [0])], terms, carried)
        c, carried, _ = exposure([(6_000_000_000, 16_000_000_000, [0])], terms, carried)
        np.testing.assert_allclose(row, a+b+c)
        assert state == carried == 1_000_000_000
        assert age == 15
        # The active intercept is absent from idle exposure.
        assert row[0] == row[1] == 1
        assert sum(row[2:]) == 15


    def test_idle_prefix_must_be_known_and_causal(self):
        with self.assertRaisesRegex(UnknownEvidence, 'PREFIX'):
            exposure([(0, 1, [0])], [[]], None)
        with self.assertRaisesRegex(ValueError, 'future'):
            exposure([(0, 1, [0])], [[]], 1)
        with self.assertRaisesRegex(UnknownEvidence, 'DOMAIN'):
            exposure([(0, 32_000_000_000, [0])], [[]], 0)


    def test_idle_energy_changes_with_known_prefix_and_cannot_extrapolate(self):
        spec = dict(schema=SCHEMA, status='FITTED_UNVALIDATED', admission_enabled=False,
                    idle_edges_seconds=IDLE_EDGES_SECONDS, independent_run_ids=['a', 'b'],
                active_terms=[[]], coefficients=[150., 90., 90., 90., 30.],
                    maximum_observed_idle_age_seconds=30., support_vertices=[[0.], [1.]])
        law = NativeIdleAgeEnergy(spec)
        hot, _ = law.integrate([(20_000_000_000, 25_000_000_000, [0.])], 20_000_000_000)
        cold, _ = law.integrate([(20_000_000_000, 25_000_000_000, [0.])], 0)
        assert hot == 450. and cold == 150.
        with self.assertRaisesRegex(UnknownEvidence, 'MEASURED_SUPPORT'):
            law.integrate([(30_000_000_000, 30_500_000_000, [0.])], 0)
