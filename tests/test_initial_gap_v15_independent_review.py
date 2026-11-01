"""Actual frozen-prior composition checks; no source event extraction or GPU."""
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from tests.test_learned_context_oracle_v14 import Controller, Mark, Handoff
from tools.predictor_validation_20261027.context_initial_prior_v14 import InitialContextPrior
from tools.predictor_validation_20261027.context_initial_prior_v15 import (
    JointInitialContextPrior, realize_joint_initial_scenario,
)
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode, SchedulingView
from tools.predictor_validation_20261027.learned_context_oracle_v15 import LearnedContextOracle, LearnedContextScheduler

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT/'artifacts/predictor-context-initial-prior-v15-20261028-training01/model.json'


def view(clock, frontier):
    return SchedulingView(clock, tuple(ActiveGpuNode((owner, 'independent-node'), 100., phases[0],
        'independent:'+owner, task, pool) for owner, task, pool, phases in frontier))


class ActualInitialGapReview(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = json.loads(MODEL_PATH.read_text())
        cls.prior = JointInitialContextPrior(cls.model)
        old = json.loads(Path(cls.model['original_prior']['path']).read_text())
        cls.old_prior = InitialContextPrior(old)
        cls.old = old
        cls.gap = next(r for r in cls.model['scenarios'] if r['state_kind'] == 'IN_TRANSITION')

    def oracle(self, rule='lower_bound_hypothesis'):
        row = self.gap
        prediction = self.prior.predict(first_pending_frontier=row['first_pending_frontier'],
            public_execution_setup=row['public_setup'])
        preceding = 0.; quantile = None
        for scenario in prediction['scenarios']:
            if scenario['scenario_id'] == row['scenario_id']:
                quantile = preceding + scenario['probability']/2; break
            preceding += scenario['probability']
        self.assertIsNotNone(quantile)
        return LearnedContextOracle(initial_prior=self.prior, save_controller=Controller(),
            next_owner=Mark(), handoff=Handoff(), public_setup=copy.deepcopy(row['public_setup']),
            public_workload=[], source_identity='a'*64, initial_quantile=quantile,
            initial_readiness_rule=rule, handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
            allow_initial_pooling=True, allow_observer_transfer=True, allow_flagged_statistical_extrapolation=True)

    def test_all_original_rows_and_each_actual_query_probability_mass_are_preserved(self):
        self.assertEqual(len(self.model['scenarios']), 28)
        self.assertEqual(self.model['state_counts'], {'HELD_CONTEXT':27, 'IN_TRANSITION':1})
        self.assertEqual([r['scenario_id'] for r in self.model['scenarios']],
                         [r['scenario_id'] for r in self.old['scenarios']])
        for row in self.old['scenarios']:
            args = dict(first_pending_frontier=row['first_pending_frontier'], public_execution_setup=row['public_setup'])
            before = self.old_prior.predict(**args); after = self.prior.predict(**args)
            self.assertEqual([(r['scenario_id'], r['probability']) for r in before['scenarios']],
                             [(r['scenario_id'], r['probability']) for r in after['scenarios']])
            self.assertAlmostEqual(before['unknown_holder_mass'], after['in_transition_mass']+after['unknown_state_mass'])
            self.assertEqual(after['unknown_state_mass'], 0.)

    def test_actual_gap_clocks_history_and_empty_frontier_restore_without_future_reads(self):
        model = self.oracle(); scheduler = LearnedContextScheduler(self.gap['public_setup']['residents'], oracle=model)
        now = 1_000_000_000; restore = now+self.gap['residual_restore_gap_ns']; frontier = self.gap['first_pending_frontier']
        with patch('sqlite3.connect', side_effect=AssertionError('prospective source event lookup')), \
                patch('pathlib.Path.read_bytes', side_effect=AssertionError('prospective source file read')):
            first = scheduler.decide(view(now, frontier))
            self.assertEqual(first.next_transition_ns, restore)
            self.assertTrue(all(v == 0. for v in first.gpu_rates.values()))
            self.assertIsNone(scheduler.held_owner); self.assertIsNone(scheduler.restore_start_ns)
            self.assertEqual(scheduler.transitions, []); self.assertEqual(model.controller.starts, [])
            middle = scheduler.decide(view(restore-1, []))
            self.assertEqual(middle.next_transition_ns, restore)
            self.assertIsNone(scheduler.snapshot(view(restore-1, [])).context_age_ns)
            scheduler.decide(view(restore, []))
            self.assertEqual(scheduler.held_owner, self.gap['next_owner'])
            self.assertEqual(scheduler.restore_start_ns, restore)
            self.assertTrue(scheduler.service_available)
            self.assertEqual(model.previous_held_owner, self.gap['previous_held_owner'])
            self.assertEqual(model.controller.starts, [(self.gap['next_owner'], restore, restore)])
            self.assertEqual(scheduler.snapshot(view(restore+1, frontier)).context_age_ns, 1)
        self.assertIn('INITIAL_SERVICE_READINESS_LOWER_BOUND_HYPOTHESIS', model.uncertainty)
        self.assertFalse(scheduler.provenance()['initial_gap_is_physical_no_execution'])
        self.assertEqual([e['kind'] for e in scheduler.transitions], ['RESTORE_START', 'SERVICE_READY'])

    def test_actual_unwitnessed_gap_cannot_supply_upper_or_exact_point(self):
        self.assertIsNone(self.gap['service_ready_delay_after_restore_bounds_ns']['upper_ns'])
        for rule in ['bounds_only', 'upper_bound_hypothesis', 'witnessed_point_only']:
            with self.subTest(rule=rule):
                model = self.oracle(rule); scheduler = LearnedContextScheduler(self.gap['public_setup']['residents'], oracle=model)
                with self.assertRaises(UnknownEvidence):
                    scheduler.decide(view(1_000_000_000, self.gap['first_pending_frontier']))
                self.assertIsNone(scheduler.held_owner)
                self.assertEqual(scheduler.transitions, [])

    def test_actual_label_relative_and_absolute_gap_arithmetic(self):
        row = self.gap; evidence = row['state_evidence']; now = 2_000_000_000
        self.assertEqual(row['elapsed_gap_ns']+row['residual_restore_gap_ns'], row['complete_restore_gap_ns'])
        self.assertEqual(evidence['query_clock_ns']-evidence['previous_SAVE_ns'], row['elapsed_gap_ns'])
        self.assertEqual(evidence['next_RESTORE_ns']-evidence['query_clock_ns'], row['residual_restore_gap_ns'])
        realized = realize_joint_initial_scenario(row, query_clock_ns=now, readiness_rule='lower_bound_hypothesis')
        self.assertEqual(realized['previous_SAVE_ns'], now-row['elapsed_gap_ns'])
        self.assertEqual(realized['next_RESTORE_ns'], now+row['residual_restore_gap_ns'])
        self.assertEqual(realized['service_ready_ns'], realized['next_RESTORE_ns'])
        self.assertTrue(realized['numerical_readiness_is_hypothesis'])
        self.assertIsNone(realized['held_owner']); self.assertIsNone(realized['context_age_ns'])


if __name__ == '__main__':
    unittest.main()
