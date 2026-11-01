"""Read-only reproduction from preflight01 clocks and actual frozen hazard model.

The rejected query's precise frontier was not serialized. Reconstruction uses
another nonempty frontier actually logged by the same candidate, and states this
limitation instead of claiming an exact replay of its unrecorded contents.
"""
import copy
import json
from pathlib import Path
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from tools.predictor_validation_20261027.cumulative_save_hazard_v14 import CumulativeSaveHazardController
from tools.predictor_validation_20261027.dag_phase_scheduler_v8 import ActiveGpuNode, SchedulingView
from tools.predictor_validation_20261027.held_context_scheduler_v12_1 import ContextAvailabilityDecision
from tools.predictor_validation_20261027.learned_context_oracle_v14 import LearnedContextOracle, LearnedContextScheduler

ROOT = Path(__file__).resolve().parents[1]
PREFLIGHT = ROOT/'artifacts/predictor-learned-availability-v14-20261028-preflight01'


def view(clock, frontier):
    return SchedulingView(clock, tuple(ActiveGpuNode((owner, 'review-node'), 1000000., phases[0],
        'review:'+owner, task, pool) for owner, task, pool, phases in frontier))


class Mark:
    def predict_given_save_now(self, **q):
        return dict(most_likely_next_owner='text#1' if q['held_owner']=='text#0' else 'text#0', ood_flags=[])


class Handoff:
    def query(self, **q):
        return dict(save_to_restore_gap_ns=2, service_ready_delay_after_restore_ns=0, ood_flags=[])


class Preflight01HoldOracle(LearnedContextOracle):
    """Isolate the original preflight01 hold branch after the parent fixes it."""
    def decide(self, query):
        self.observe_query(query)
        if self.hazard is None:
            raise UnknownEvidence('CONDITIONAL_SAVE_MODEL_NOT_INITIALIZED')
        if not self.save_due(query):
            review = self.hazard['next_review_ns']
            if review is None or review <= query.clock_ns:
                raise UnknownEvidence('FUTURE_CUMULATIVE_SAVE_REVIEW_UNAVAILABLE')
            return ContextAvailabilityDecision('hold', review)
        return super().decide(query)


def reconstructed(candidate, clipped):
    log = json.loads((PREFLIGHT/('forecast-'+candidate+'.json')).read_text())
    diag = log['scheduler_diagnostic']; last = diag['queries'][-1]
    old_frontier = last['pending_frontier']
    frontier = next(q['pending_frontier'] for q in reversed(diag['queries'])
                    if q['pending_frontier'] and q['pending_frontier'] != old_frontier)
    cfg = log['configuration']
    controller = CumulativeSaveHazardController(cfg['save_model'], allow_unvalidated_approximation=True,
        budget_policy=cfg['options']['save_budget_policy'], empty_frontier_policy=cfg['options']['empty_frontier_policy'])
    cls = LearnedContextOracle if clipped else Preflight01HoldOracle
    oracle = cls(initial_prior=None, save_controller=controller, next_owner=Mark(), handoff=Handoff(),
        public_setup={}, public_workload=log['full_public_cohort'], source_identity='a'*64,
        initial_quantile=.5, initial_readiness_rule='lower_bound_hypothesis',
        handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
        allow_initial_pooling=True, allow_observer_transfer=True, allow_flagged_statistical_extrapolation=True)
    scheduler = LearnedContextScheduler(('text#0', 'text#1', 'generalist_2#0', 'ocr#0'), oracle=oracle)
    scheduler._event(last['clock_ns']-last['context_age_ns'], 'RESTORE_START', last['held_owner'], 'RECONSTRUCTED_LOG_STATE')
    scheduler._event(last['clock_ns'], 'SERVICE_READY', last['held_owner'], 'RECONSTRUCTED_LOG_STATE')
    query = scheduler.snapshot(view(last['clock_ns'], old_frontier))
    scheduler._accept(ContextAvailabilityDecision(**last['decision']), query)
    scheduler.last_frontier = query.pending_frontier
    oracle.last_restored_owner = last['held_owner']
    return scheduler, oracle, last['decision']['service_ready_ns'], frontier, diag


class IndependentDeadlineReview(unittest.TestCase):
    def test_all_four_actual_logs_pin_failure_to_new_service_clock(self):
        for candidate in ['shared_balanced', 'shared_skewed', 'shared_spill_low', 'shared_spill_high']:
            with self.subTest(candidate=candidate):
                scheduler, oracle, service, frontier, diag = reconstructed(candidate, False)
                self.assertEqual(diag['review_deadline_ns'], service+1)
                self.assertIsNone(diag['pending_switch'])
                self.assertEqual(diag['transitions'][-1]['kind'], 'SERVICE_READY')
                self.assertEqual(diag['transitions'][-1]['clock_ns'], service)
                with self.assertRaisesRegex(UnknownEvidence, 'CONTEXT_DEADLINE_EXTENSION_WITHOUT_RESTORE'):
                    scheduler.decide(view(service, frontier))
                self.assertGreater(oracle.hazard['next_review_ns'], service+1)
                self.assertEqual(scheduler.review_deadline_ns, service+1)
                repaired, actual, service, frontier, _ = reconstructed(candidate, True)
                self.assertEqual(repaired.decide(view(service, frontier)).next_transition_ns, service+1)
                budget = copy.deepcopy(actual.hazard)
                self.assertEqual(repaired.decide(view(service+1, frontier)).next_transition_ns,
                                 budget['next_review_ns'])
                self.assertEqual(actual.hazard, budget)

    def test_clipped_review_expires_without_redrawing_or_reconditioning_hazard(self):
        scheduler, oracle, service, frontier, diag = reconstructed('shared_balanced', True)
        result = scheduler.decide(view(service, frontier))
        first = copy.deepcopy(oracle.hazard)
        self.assertEqual(result.next_transition_ns, service+1)
        self.assertGreater(first['next_review_ns'], service+1)
        result = scheduler.decide(view(service+1, frontier))
        self.assertEqual(result.next_transition_ns, first['next_review_ns'])
        self.assertEqual(oracle.hazard, first)
        self.assertEqual(oracle.controller._check(oracle.hazard['state'])['observation_ns'], service)
        crossing = first['absolute_save_candidate_ns']
        scheduler.decide(view(crossing, frontier))
        self.assertEqual(scheduler.queries[-1]['decision']['kind'], 'switch')
        self.assertEqual(scheduler.queries[-1]['decision']['save_end_ns'], crossing)
        self.assertEqual(oracle.hazard['hazard_budget'], first['hazard_budget'])
        self.assertAlmostEqual(oracle.hazard['consumed_hazard'], first['hazard_budget'])

    def test_later_frontier_update_keeps_earliest_future_review(self):
        scheduler, oracle, service, frontier, diag = reconstructed('shared_balanced', True)
        scheduler.decide(view(service, frontier))
        other = diag['queries'][-1]['pending_frontier']
        scheduler.decide(view(service, other))
        self.assertEqual(scheduler.review_deadline_ns, service+1)
        self.assertEqual(oracle.hazard['observation_ns'], service)
        self.assertEqual(oracle.hazard['consumed_hazard'], 0.)


if __name__ == '__main__':
    unittest.main()
