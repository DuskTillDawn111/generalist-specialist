"""Independent hand-calculated initial post-handler state transitions."""

import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import LifecycleProvider, LIFECYCLE_SCHEMA
from gsdeploy.forecasting.profiles import ProfileProvider
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.forecasting.observed_state import (new_observed_state, advance_observed_state,
    feedback_event, forecast_snapshot_from_observed)
from gsdeploy.forecasting.__main__ import forecast_request
from gsdeploy.resources import identity
from tools.resource_control_20261024.forecast_policy import FullHorizonPolicy
from tests.test_full_horizon_forecast import aliases, profiles, resources, request, MS


def life(gated=True, *, ready=(2., 2.)):
    doc = {"schema": LIFECYCLE_SCHEMA, "execution_context_sha256": "fixture",
        "provenance": {"kind": "cpu_fixture", "split": "training", "sources": ["joint-handoff-fixture"]},
        "profiles": []}
    for family in ("text", "generalist_2"):
        samples = [{"id": family + str(i), "delays": {"parent_receipt_ms": 4.,
            "permit_release_ms": release if gated else None, "worker_next_ready_ms": ready[i]}}
            for i, release in enumerate((10., 12.))]
        doc["profiles"].append({"family": family, "task": "text", "pool_id": "x",
            "permit_scope": "request_admission" if gated else "not_applicable", "evidence": [r["id"] for r in samples],
            "scenarios": {"mean": {"parent_receipt_ms": 4., "permit_release_ms": 11. if gated else None,
                                     "worker_next_ready_ms": sum(ready) / 2}}, "joint_observations": samples})
    return LifecycleProvider(doc, "fixture", allow_cpu_fixture=True)


def model(gated=True, **kwargs):
    provider = ProfileProvider(profiles(s_ms=10., g_ms=10., speed=lambda *args: 1.),
                               "fixture", allow_cpu_fixture=True)
    return FullHorizonModel(aliases(), provider, resources=resources(gated), lifecycle_provider=life(gated), **kwargs)


def handoff(now=105, *, received=True, ready=True, gated=True):
    return {"request_id": "old", "alias": "s", "task": "text", "pool_id": "x", "arrival_ns": 0,
        "handler_end_ns": 100 * MS, "parent_receipt_ns": 104 * MS if received else None,
        "permit_release_ns": None, "worker_next_ready_ns": 102 * MS if ready else None,
        "observed_at_ns": now * MS, "evidence": "coherent-cpu-handoff", "start_ns": 90 * MS,
        "boundary_observation_status": {"parent_receipt_ns": "observed" if received else "confirmed_pending",
            "permit_release_ns": "confirmed_pending" if gated else "not_applicable",
            "worker_next_ready_ns": "observed" if ready else "confirmed_pending"}}


def score(result, *, completed=(), now=105, deadline=150):
    return score_forecast(result, completed, aliases(), {"text": 1.}, {"text": 100.},
        {"text": {"text": .1}, "generalist_2": {"text": .2}}, phase_start_ns=0,
        now_ns=now * MS, deadline_ns=deadline * MS, memory_budget_mib=1000., energy_reference_watts=100.,
        observed_peak_memory_mib=100., observed_board_energy_joules=0.)


def visible(now=105, *, gated=True):
    state = new_observed_state(aliases(), "fixture", configuration_sha256=identity(resources(gated)))
    arrival = {"sequence": 1, "request_id": "old", "alias": "s", "task": "text", "pool_id": "x",
               "arrival_ns": 0, "observed_time_ns": 0}
    feedback = [feedback_event(1, "START", {"request_id": "old", "alias": "s", "perf_ns": 90 * MS}, 90 * MS, "fixture"),
        feedback_event(2, "HANDLER_END", {"request_id": "old", "alias": "s", "perf_ns": 100 * MS}, 104 * MS, "fixture"),
        feedback_event(3, "WORKER_READY", {"request_id": "old", "alias": "s", "perf_ns": 102 * MS,
            "clock": "host_perf_counter_ns", "boundary": "previous_handler_next_request_ready",
            "scope": "after_response_flush_before_next_read"}, 104 * MS + 1, "fixture")]
    snapshot = None
    if gated:
        snapshot = {"event_contract": "gsdeploy.admission_events.v2", "clock": "host_perf_counter_ns",
            "configuration_sha256": identity(resources(True)), "observed_time_ns": now * MS,
            "events_after_sequence": 0, "last_sequence": 2, "active": {"old": "s"},
            "waiting": [], "closed": False, "peak_active": 1, "events": [
                {"sequence": 1, "kind": "enqueue", "time_ns": 10 * MS, "alias": "s", "request_id": "old",
                 "reason": None, "active_count_after": 0, "waiting_count_after": 1},
                {"sequence": 2, "kind": "grant", "time_ns": 20 * MS, "alias": "s", "request_id": "old",
                 "reason": None, "active_count_after": 1, "waiting_count_after": 0}]}
    return advance_observed_state(state, now * MS, admission_snapshot=snapshot, arrivals=[arrival], feedback=feedback)


def cli_request():
    return {"schema": "gsdeploy.full_horizon_forecast_request.v1", "execution_context_sha256": "fixture",
        "profiles": profiles(s_ms=10., g_ms=10., speed=lambda *args: 1.), "aliases": aliases(), "resources": resources(True),
        "lifecycle_profiles": life().document, "queues": {"g": [request("next", arrival_ns=105 * MS)]},
        "now_ns": 105 * MS, "phase_start_ns": 0, "deadline_ns": 150 * MS, "drain_deadline_ns": 200 * MS,
        "future_workload": {"rates": {"text": 1}, "input_pools": {"text": ["x"]},
                            "process": "declared_schedule", "declared_arrivals": []},
        "continuation_routing_rps": {"text": {"s": 1}}, "latency_reference_ms": {"text": 100},
        "quality_loss": {"text": {"text": .1}, "generalist_2": {"text": .2}}, "memory_budget_mib": 1000,
        "memory_safe_factor": 1., "energy_reference_watts": 100, "observed_peak_memory_mib": 100.,
        "observed_board_energy_joules": 0., "handoff_state": [handoff()],
        "completed": [{"request_id": "old", "task": "text", "pool_id": "x", "alias": "s",
                       "arrival_ns": 0, "completion_ns": 104 * MS}]}


class InitialHandoffTests(unittest.TestCase):
    def test_pending_handoff_members_count_toward_request_budget(self):
        first = handoff(103, received=False, ready=True, gated=False)
        second = {**copy.deepcopy(first), "request_id": "other", "alias": "g"}
        with self.assertRaisesRegex(UnknownEvidence, "BUDGET_EXCEEDED_NO_TRUNCATION"):
            model(False, max_requests=1).forecast({}, 103 * MS, 150 * MS, handoff_state=[first, second])

    def test_visible_snapshot_helper_and_cli_consume_matching_observed_state(self):
        state = visible()
        projected = forecast_snapshot_from_observed(state, aliases(), "fixture", 105 * MS, ["text"])
        self.assertEqual(projected["observed_last_arrivals_ns"], {"text": 0})
        self.assertEqual(projected["admission"]["active"], {"old": "s"})
        value = cli_request()
        value["visible_host_state"] = state
        result = forecast_request(value, allow_cpu_fixture=True)
        self.assertEqual(result["visible_state_sha256"], state["state_sha256"])
        self.assertEqual(result["metrics"]["cohort_size"], 2)
        self.assertEqual(result["forecast"]["finishes"]["next"]["start_ns"], 111 * MS)
        with self.assertRaises(ValueError):
            forecast_snapshot_from_observed(state, aliases(), "fixture", 106 * MS, ["text"])

    def test_advancing_observer_clock_does_not_make_old_gate_snapshot_current(self):
        state = advance_observed_state(visible(), 106 * MS)
        with self.assertRaises((ValueError, UnknownEvidence)):
            forecast_snapshot_from_observed(state, aliases(), "fixture", 106 * MS, ["text"])

    def test_visible_observed_ready_conflicts_with_stale_pending_handoff(self):
        value = cli_request()
        value["visible_host_state"] = visible()
        value["handoff_state"][0]["worker_next_ready_ns"] = None
        value["handoff_state"][0]["boundary_observation_status"]["worker_next_ready_ns"] = "confirmed_pending"
        value["lifecycle_profiles"] = life(ready=(8., 10.)).document
        with self.assertRaises(ValueError):
            forecast_request(value, allow_cpu_fixture=True)

    def test_policy_terms_include_pending_receipt_handoff_cohort(self):
        source = model().service_provider
        policy = FullHorizonPolicy(aliases(), source,
            {"text": {"text": .1}, "generalist_2": {"text": .2}}, {k: 1 / 6 for k in "QTLPME"},
            {"text": 100.}, {"text": 1.}, scenarios=("mean",),
            future_workload=FutureWorkload({"text": 1.}, {"text": ["x"]}, process="declared_schedule", declared_arrivals=[]),
            memory_budget_mib=1000., resources=resources(True), arrival_scenarios=1, lifecycle_provider=life())
        policy.start_phase(0, 150 * MS)
        queues = policy._queues({"s": {"outstanding": []}, "g": {"outstanding": [
            {"request_id": "next", "task": "text", "pool_id": "x", "arrival_ns": 101 * MS, "status": "queued"}]},
            "_forecast_state": {"admission": {"active": {"old": "s"}, "waiting": [("g", "next")]},
                "handoff_state": [handoff(101, received=False, ready=False)],
                "observed_peak_memory_mib": 100., "observed_board_energy_joules": 0.}}, 101 * MS)
        forecast = policy._forecast(queues, 101 * MS, 150 * MS, "mean@arrival-0")
        metric = policy._terms(forecast, queues)
        self.assertEqual(metric["cohort_ids"], ["next", "old"])

    def test_received_old_request_holds_permit_until_conditional_release(self):
        result = model().forecast({"g": [request("next", arrival_ns=105 * MS)]}, 105 * MS, 150 * MS,
            admission_state={"active": {"old": "s"}, "waiting": [("g", "next")]}, handoff_state=[handoff()])
        self.assertEqual(result["finishes"]["next"]["start_ns"], 111 * MS)
        self.assertEqual(result["finishes"]["next"]["finish_ns"], 125 * MS)
        self.assertNotIn("old", result["cohort"])
        self.assertEqual(result["initial_residual_evidence"][0]["evidence"]["permit_release_remaining_ms"], 6.)
        self.assertIn({"kind": "permit_release", "request_id": "old", "alias": "s", "time_ns": 111 * MS},
                      result["lifecycle_events"])

    def test_pending_receipt_remains_in_cohort_and_finishes_from_residual(self):
        result = model().forecast({}, 101 * MS, 150 * MS,
            admission_state={"active": {"old": "s"}, "waiting": []},
            handoff_state=[handoff(101, received=False, ready=False)])
        self.assertEqual(set(result["cohort"]), {"old"})
        self.assertEqual(result["finishes"]["old"]["finish_ns"], 104 * MS)
        self.assertEqual(result["finishes"]["old"]["predicted_remaining_service_ms"], 0.)
        metric = score(result, now=101)
        self.assertEqual(metric["cohort_size"], 1)
        self.assertAlmostEqual(metric["terms"]["L"], 1.04)

    def test_shared_ready_allows_next_handler_before_old_receipt_without_permit(self):
        result = model(False).forecast({"s": [request("next", arrival_ns=101 * MS)]}, 101 * MS, 150 * MS,
            handoff_state=[handoff(101, received=False, ready=False, gated=False)])
        self.assertEqual(result["finishes"]["next"]["start_ns"], 102 * MS)
        self.assertEqual(result["finishes"]["old"]["finish_ns"], 104 * MS)
        self.assertEqual(result["finishes"]["next"]["finish_ns"], 116 * MS)
        self.assertFalse(any(e["kind"] == "permit_release" for e in result["lifecycle_events"]))

    def test_unknown_boundary_state_does_not_become_default_zero(self):
        state = handoff()
        del state["boundary_observation_status"]
        with self.assertRaisesRegex(UnknownEvidence, "CONFIRMED_HANDOFF_BOUNDARY_STATE_REQUIRED"):
            model().forecast({}, 105 * MS, 150 * MS,
                admission_state={"active": {"old": "s"}, "waiting": []}, handoff_state=[state])

    def test_completed_history_scores_old_once_without_future_cohort_overlap(self):
        result = model().forecast({"g": [request("next", arrival_ns=105 * MS)]}, 105 * MS, 150 * MS,
            admission_state={"active": {"old": "s"}, "waiting": [("g", "next")]}, handoff_state=[handoff()])
        completed = [{"request_id": "old", "task": "text", "pool_id": "x", "alias": "s",
                      "arrival_ns": 0, "completion_ns": 104 * MS}]
        metric = score(result, completed=completed)
        self.assertEqual(metric["cohort_ids"], ["next", "old"])
        self.assertEqual(metric["cohort_size"], 2)
        self.assertAlmostEqual(metric["terms"]["L"], .62)

    def test_receipt_exactly_at_drain_boundary_is_counted_but_not_earlier(self):
        kwargs = {"admission_state": {"active": {"old": "s"}, "waiting": []},
                  "handoff_state": [handoff(101, received=False, ready=False)]}
        censored = model().forecast({}, 101 * MS, 103 * MS, drain_deadline_ns=104 * MS - 1, **kwargs)
        self.assertEqual(censored["incomplete_ids"], ["old"])
        at_boundary = model().forecast({}, 101 * MS, 103 * MS, drain_deadline_ns=104 * MS, **kwargs)
        self.assertEqual(at_boundary["incomplete_ids"], [])
        self.assertEqual(at_boundary["finishes"]["old"]["finish_ns"], 104 * MS)
        self.assertEqual(score(at_boundary, now=101, deadline=103)["terms"]["T"], 1.)

    def test_handoff_gate_held_state_must_agree(self):
        with self.assertRaisesRegex(ValueError, "release and observed gate state disagree"):
            model().forecast({}, 105 * MS, 150 * MS,
                admission_state={"active": {}, "waiting": []}, handoff_state=[handoff()])


if __name__ == "__main__":
    unittest.main()
