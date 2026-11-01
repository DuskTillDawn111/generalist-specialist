"""Post-handler boundaries remain separate, evidence bound, and training only."""

import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.lifecycle import (
    LifecycleProvider, export_training_observations, fit_lifecycle,
)
from gsdeploy.runtime.admission import RequestAdmission
from tests.test_resource_control import resources


CONTEXT = "lifecycle-test-context"
ALIASES = {"s#0": {"family": "specialist"}, "g#0": {"family": "generalist"}}
CLOCK = {"handler": "host_perf_counter_ns", "runtime": "host_perf_counter_ns",
         "worker_next_ready": "host_perf_counter_ns", "same_host_verified": True,
         "source": "cpu-fixture-common-clock"}


def raw_fixture(rid="one", admission=True):
    gate = RequestAdmission(resources("request_admission"))
    gate.acquire("s#0", rid)
    gate.release(rid, "worker_result")
    snapshot = gate.snapshot()
    # Keep a real gate stream and move all timestamps to deterministic units.
    for index, event in enumerate(snapshot["events"]):
        event["time_ns"] = (index + 1) * 10_000_000
    snapshot["observed_time_ns"] = 40_000_000
    row = {"request_id": rid, "split": "training", "status": "completed",
           "model": "s#0", "task": "text", "pool_id": "input-a",
           "worker_response": {"end_perf_ns": 21_000_000},
           "response_reader_event_enqueue_time_ns": 25_000_000,
           "completion_time_ns": 25_000_000}
    if admission:
        row["admission_released_time_ns"] = 30_000_000
    return {"split": "training", "kind": "cpu_fixture", "execution_context_sha256": CONTEXT,
            "admission_enabled": admission, "request_records": [row],
            "admission_snapshot_after": snapshot if admission else None}


def ready_fixture(rid="one", timestamp=22_000_000):
    return {"request_id": rid, "alias": "s#0", "time_ns": timestamp,
            "clock": "host_perf_counter_ns", "boundary": "previous_handler_next_request_ready",
            "source": "cpu-ready-event"}


def export(raw, ready=None, source="training-a", clock=None):
    return export_training_observations(raw, ALIASES, execution_context_sha256=CONTEXT,
                                        clock_contract=CLOCK if clock is None else clock,
                                        source=source, worker_ready_events=ready)


def document(admission=True, ready=True):
    observations = []
    for rid in ("one", "two"):
        observations += export(raw_fixture(rid, admission), [ready_fixture(rid)] if ready else None, source=rid)
    return fit_lifecycle(observations, CONTEXT)


class LifecycleTests(unittest.TestCase):
    def test_joint_residual_conditions_on_event_pattern_not_unconditional_mean_minus_age(self):
        doc = document()
        profile = doc["profiles"][0]
        vectors = [(4., 10., 2.), (4., 12., 3.), (8., 20., 9.)]
        profile["evidence"] = ["joint-" + str(i) for i in range(3)]
        profile["joint_observations"] = [{"id": oid, "delays": dict(zip(
            ("parent_receipt_ms", "permit_release_ms", "worker_next_ready_ms"), values))}
            for oid, values in zip(profile["evidence"], vectors)]
        provider = LifecycleProvider(doc, CONTEXT, allow_cpu_fixture=True)
        observed = {"handler_end_ns": 100 * 1_000_000, "parent_receipt_ns": 104 * 1_000_000,
                    "permit_release_ns": None, "worker_next_ready_ns": 102 * 1_000_000}
        statuses = {"parent_receipt_ns": "observed", "permit_release_ns": "confirmed_pending", "worker_next_ready_ns": "observed"}
        result = provider.residual_handoff({"alias": "s#0", "task": "text", "pool_id": "input-a"},
            ALIASES, "mean", 105 * 1_000_000, observed, boundary_observation_status=statuses)
        self.assertEqual(result["parent_receipt_remaining_ms"], 0.)
        self.assertEqual(result["worker_next_ready_remaining_ms"], 0.)
        self.assertEqual(result["permit_release_remaining_ms"], 6.)
        self.assertEqual(result["condition_evidence"]["joint_support_count"], 2)
        self.assertEqual(result["evidence"], ["joint-0", "joint-1"])
        self.assertNotAlmostEqual(result["permit_release_remaining_ms"], (10. + 12. + 20.) / 3 - 5.)

    def test_handoff_requires_confirmed_status_joint_samples_and_enough_support(self):
        doc = document()
        provider = LifecycleProvider(doc, CONTEXT, allow_cpu_fixture=True)
        row = {"alias": "s#0", "task": "text", "pool_id": "input-a"}
        observed = {"handler_end_ns": 100 * 1_000_000, "parent_receipt_ns": None,
                    "permit_release_ns": None, "worker_next_ready_ns": None}
        statuses = {key: "confirmed_pending" for key in observed if key != "handler_end_ns"}
        with self.assertRaisesRegex(UnknownEvidence, "CONFIRMED_HANDOFF_BOUNDARY_STATE_REQUIRED"):
            provider.residual_handoff(row, ALIASES, "mean", 100 * 1_000_000, observed)
        unknown_ready = {**statuses, "worker_next_ready_ns": "unknown"}
        with self.assertRaisesRegex(UnknownEvidence, "UNKNOWN_HANDOFF_BOUNDARY_STATE"):
            provider.residual_handoff(row, ALIASES, "mean", 100 * 1_000_000, observed,
                                      boundary_observation_status=unknown_ready)
        # Same two empirical vectors support a pending event at age zero.
        result = provider.residual_handoff(row, ALIASES, "mean", 100 * 1_000_000, observed,
                                          boundary_observation_status=statuses)
        self.assertEqual(result["parent_receipt_remaining_ms"], 4.)
        # Strict > age excludes samples exactly at the purported pending boundary.
        with self.assertRaisesRegex(UnknownEvidence, "INSUFFICIENT_CONDITIONAL_JOINT_HANDOFF_SUPPORT"):
            provider.residual_handoff(row, ALIASES, "mean", 101 * 1_000_000, observed,
                                      boundary_observation_status=statuses)
        del doc["profiles"][0]["joint_observations"]
        legacy = LifecycleProvider(doc, CONTEXT, allow_cpu_fixture=True)
        with self.assertRaisesRegex(UnknownEvidence, "JOINT_HANDOFF_TRAINING_OBSERVATIONS_REQUIRED"):
            legacy.residual_handoff(row, ALIASES, "mean", 100 * 1_000_000, observed,
                                    boundary_observation_status=statuses)
        self.assertTrue(legacy.delays(row, ALIASES, "mean")["known"])

    def test_handoff_rejects_future_or_inconsistent_observed_times_and_nonmean_scenario(self):
        provider = LifecycleProvider(document(), CONTEXT, allow_cpu_fixture=True)
        row = {"alias": "s#0", "task": "text", "pool_id": "input-a"}
        observed = {"handler_end_ns": 100, "parent_receipt_ns": 104,
                    "permit_release_ns": 109, "worker_next_ready_ns": 101}
        statuses = {key: "observed" for key in observed if key != "handler_end_ns"}
        for mutate in (lambda value: value.update(handler_end_ns=111),
                       lambda value: value.update(parent_receipt_ns=111),
                       lambda value: value.update(permit_release_ns=103),
                       lambda value: value.update(worker_next_ready_ns=99)):
            broken = dict(observed)
            mutate(broken)
            with self.assertRaises(ValueError):
                provider.residual_handoff(row, ALIASES, "mean", 110, broken,
                                          boundary_observation_status=statuses)
        with self.assertRaises(UnknownEvidence):
            provider.residual_handoff(row, ALIASES, "p95", 110, observed,
                                      boundary_observation_status=statuses)

    def test_shared_joint_residual_has_no_permit_value(self):
        provider = LifecycleProvider(document(admission=False), CONTEXT, allow_cpu_fixture=True)
        observed = {"handler_end_ns": 0, "parent_receipt_ns": None,
                    "permit_release_ns": None, "worker_next_ready_ns": None}
        statuses = {"parent_receipt_ns": "confirmed_pending", "permit_release_ns": "not_applicable",
                    "worker_next_ready_ns": "confirmed_pending"}
        result = provider.residual_handoff({"alias": "s#0", "task": "text", "pool_id": "input-a"},
            ALIASES, "mean", 0, observed, boundary_observation_status=statuses)
        self.assertIsNone(result["permit_release_remaining_ms"])
        self.assertEqual(result["worker_next_ready_remaining_ms"], 1.)

    def test_three_boundaries_are_distinct_and_worker_ready_can_precede_receipt(self):
        observations = export(raw_fixture(), [ready_fixture()])
        self.assertEqual(observations[0]["delays"], {
            "parent_receipt_ms": 4., "permit_release_ms": 9., "worker_next_ready_ms": 1.})
        provider = LifecycleProvider(document(), CONTEXT, allow_cpu_fixture=True)
        result = provider.delays({"alias": "s#0", "task": "text", "pool_id": "input-a"}, ALIASES, "mean")
        self.assertTrue(result["known"])
        self.assertEqual(result["parent_receipt_ms"], 4.)
        self.assertEqual(result["permit_release_ms"], 9.)
        self.assertEqual(result["worker_next_ready_ms"], 1.)
        self.assertFalse(provider.physical_evidence)

    def test_missing_ready_never_becomes_receipt_or_zero(self):
        result = LifecycleProvider(document(ready=False), CONTEXT, allow_cpu_fixture=True).delays(
            {"alias": "s#0", "task": "text", "pool_id": "input-a"}, ALIASES, "mean")
        self.assertFalse(result["known"])
        self.assertIsNone(result["worker_next_ready_ms"])
        self.assertEqual(result["permit_release_ms"], 9.)

    def test_recorded_readiness_is_exported_and_conflicts_are_rejected(self):
        raw = raw_fixture()
        raw["request_records"][0]["previous_handler_next_request_ready"] = {
            "request_id": "one", "pid": 123, "perf_ns": 22_000_000,
            "received_time_ns": 26_000_000, "clock": "host_perf_counter_ns",
            "boundary": "previous_handler_next_request_ready",
            "scope": "after_response_flush_before_next_read"}
        self.assertEqual(export(raw)[0]["delays"]["worker_next_ready_ms"], 1.)
        with self.assertRaises(ValueError):
            export(raw, [ready_fixture(timestamp=23_000_000)])

    def test_shared_has_no_invented_permit(self):
        result = LifecycleProvider(document(admission=False), CONTEXT, allow_cpu_fixture=True).delays(
            {"alias": "s#0", "task": "text", "pool_id": "input-a"}, ALIASES, "mean")
        self.assertTrue(result["known"])
        self.assertEqual(result["permit_scope"], "not_applicable")
        self.assertIsNone(result["permit_release_ms"])

    def test_missing_gate_evidence_remains_unknown_and_old_snapshot_is_rejected(self):
        raw = raw_fixture()
        raw["admission_snapshot_after"] = None
        del raw["request_records"][0]["admission_released_time_ns"]
        observation = export(raw, [ready_fixture()])[0]
        self.assertIsNone(observation["delays"]["permit_release_ms"])
        raw["admission_snapshot_after"] = {"active": {}, "waiting": []}
        with self.assertRaises(ValueError):
            export(raw)

    def test_release_cannot_be_fabricated_from_record_or_earlier_than_receipt(self):
        for mutate in (
            lambda r: r.update(admission_snapshot_after=None),
            lambda r: r["admission_snapshot_after"]["events"][-1].update(time_ns=24_000_000),
            lambda r: r["request_records"][0].update(admission_released_time_ns=31_000_000),
            lambda r: r["admission_snapshot_after"]["events"][-1].update(alias="g#0"),
        ):
            raw = raw_fixture()
            mutate(raw)
            with self.assertRaises(ValueError):
                export(raw)

    def test_training_split_context_and_cpu_evidence_are_enforced(self):
        for mutate in (
            lambda r: r.update(split="validation"),
            lambda r: r.update(not_for_training=True),
            lambda r: r.update(execution_context_sha256="other"),
            lambda r: r["request_records"][0].update(split="validation"),
        ):
            raw = raw_fixture()
            mutate(raw)
            with self.assertRaises(ValueError):
                export(raw)
        with self.assertRaises(ValueError):
            LifecycleProvider(document(), CONTEXT)
        with self.assertRaises(ValueError):
            LifecycleProvider(document(), "other", allow_cpu_fixture=True)

    def test_mixed_unverified_clocks_and_invented_readiness_boundaries_rejected(self):
        for mutate in (
            lambda c: c.update(handler="unix_ns"),
            lambda c: c.update(same_host_verified=False),
        ):
            clock = copy.deepcopy(CLOCK)
            mutate(clock)
            with self.assertRaises(ValueError):
                export(raw_fixture(), clock=clock)
        for mutate in (
            lambda e: e.update(boundary="next_handler_start"),
            lambda e: e.update(time_ns=20_000_000),
            lambda e: e.update(clock="device_ns"),
            lambda e: e.update(alias="g#0"),
            lambda e: e.update(request_id="foreign"),
        ):
            event = ready_fixture()
            mutate(event)
            with self.assertRaises(ValueError):
                export(raw_fixture(), [event])

    def test_unknown_inputs_and_scenarios_are_explicit(self):
        provider = LifecycleProvider(document(), CONTEXT, allow_cpu_fixture=True)
        with self.assertRaises(UnknownEvidence):
            provider.delays({"alias": "s#0", "task": "text", "pool_id": "unseen"}, ALIASES, "mean")
        with self.assertRaises(UnknownEvidence):
            provider.delays({"alias": "s#0", "task": "text", "pool_id": "input-a"}, ALIASES, "p95")

    def test_fitter_rejects_duplicates_validation_and_insufficient_data(self):
        observations = export(raw_fixture(), [ready_fixture()])
        with self.assertRaises(UnknownEvidence):
            fit_lifecycle(observations, CONTEXT)
        with self.assertRaises(ValueError):
            fit_lifecycle(observations * 2, CONTEXT)
        observations[0]["split"] = "validation"
        with self.assertRaises(ValueError):
            fit_lifecycle(observations, CONTEXT)

    def test_gate_replay_and_observed_delay_consistency_are_required(self):
        raw = raw_fixture()
        raw["admission_snapshot_after"]["events"].pop(1)
        with self.assertRaises(ValueError):
            export(raw)
        observations = export(raw_fixture("one"), [ready_fixture("one")], source="one")
        observations += export(raw_fixture("two"), [ready_fixture("two")], source="two")
        observations[0]["delays"]["worker_next_ready_ms"] = 0.
        with self.assertRaises(ValueError):
            fit_lifecycle(observations, CONTEXT)

    def test_fit_preserves_unknown_boundary_even_if_other_rows_observe_it(self):
        observations = export(raw_fixture("one"), [ready_fixture("one")], source="one")
        observations += export(raw_fixture("two"), source="two")
        fitted = fit_lifecycle(observations, CONTEXT)
        self.assertIsNone(fitted["profiles"][0]["scenarios"]["mean"]["worker_next_ready_ms"])


if __name__ == "__main__":
    unittest.main()
