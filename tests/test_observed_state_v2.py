"""Visible observation replay never extrapolates kernel progress or future data."""

import copy
import unittest

from gsdeploy.forecasting.observed_state import (
    new_observed_state, advance_observed_state, feedback_event, feedback_from_runtime_records,
)
from tests.test_forecast_lifecycle_v2 import raw_fixture


CONTEXT = "visible-context"
ALIASES = {"s#0": {}, "g#0": {}}


def initial(mode="online", cutoff=None):
    snapshot = raw_fixture()["admission_snapshot_after"]
    return new_observed_state(ALIASES, CONTEXT, mode=mode,
        configuration_sha256=snapshot["configuration_sha256"], frozen_cutoff_ns=cutoff)


def event(sequence, event_kind, timestamp, received, **payload):
    return feedback_event(sequence, event_kind,
        {"alias": "s#0", "request_id": "one", "perf_ns": timestamp, **payload}, received, CONTEXT)


def arrival(sequence=1, rid="one", timestamp=1_000_000):
    return {"sequence": sequence, "request_id": rid, "alias": "s#0", "task": "text",
            "pool_id": "input-a", "arrival_ns": timestamp, "observed_time_ns": timestamp + 1}


class ObservedStateTests(unittest.TestCase):
    def test_runtime_record_adapter_filters_by_reader_visibility_and_is_incremental(self):
        records = [{"request_id": "one", "model": "s#0", "execution_events": [
            {"request_id": "one", "perf_ns": 1, "received_time_ns": 2},
            {"request_id": "one", "perf_ns": 3, "received_time_ns": 4,
             "event": "start", "kind": "gpu", "segment_id": "g"},
            {"request_id": "one", "perf_ns": 5, "received_time_ns": 6,
             "event": "end", "kind": "gpu", "segment_id": "g"}],
            "worker_response": {"end_perf_ns": 7, "prediction": "not-a-state-feature"},
            "response_reader_event_enqueue_time_ns": 10,
            "previous_handler_next_request_ready": {"request_id": "one", "perf_ns": 8,
                "received_time_ns": 11, "clock": "host_perf_counter_ns",
                "boundary": "previous_handler_next_request_ready", "scope": "after_response_flush_before_next_read"}}]
        first = feedback_from_runtime_records(records, CONTEXT, 6)
        state = advance_observed_state(initial(), 6, feedback=first)
        self.assertIsNone(state["requests"]["one"]["handler_end_ns"])
        suffix = feedback_from_runtime_records(records, CONTEXT, 11, since_sequence=state["last_feedback_sequence"])
        self.assertEqual([e["kind"] for e in suffix], ["HANDLER_END", "WORKER_READY"])
        later = advance_observed_state(state, 11, feedback=suffix)
        self.assertEqual(later["requests"]["one"]["worker_next_ready_ns"], 8)
        self.assertNotIn("prediction", str(later))

    def test_gate_incremental_replay_matches_full_history_and_preserves_previous(self):
        snapshot = raw_fixture()["admission_snapshot_after"]
        prefix = copy.deepcopy(snapshot)
        prefix.update(events=prefix["events"][:2], active={"one": "s#0"}, last_sequence=2,
                      observed_time_ns=22_000_000)
        original = initial()
        middle = advance_observed_state(original, 22_000_000, admission_snapshot=prefix, arrivals=[arrival()])
        self.assertIsNone(original["admission_snapshot"])
        self.assertEqual(middle["admission_snapshot"]["active"], {"one": "s#0"})
        suffix = copy.deepcopy(snapshot)
        suffix.update(events=suffix["events"][2:], events_after_sequence=2)
        finished = advance_observed_state(middle, 40_000_000, admission_snapshot=suffix)
        full = advance_observed_state(original, 40_000_000, admission_snapshot=snapshot, arrivals=[arrival()])
        self.assertEqual(finished, full)
        self.assertEqual(middle["admission_snapshot"]["active"], {"one": "s#0"})
        self.assertEqual(finished["admission_snapshot"]["active"], {})
        self.assertEqual(finished["last_arrivals"]["text"]["request_id"], "one")

    def test_phase_feedback_preserves_unknown_kernel_cursor_despite_elapsed_time(self):
        state = advance_observed_state(initial(), 15_000_000, arrivals=[arrival()], feedback=[
            event(1, "START", 10_000_000, 11_000_000),
            event(2, "PHASE", 12_000_000, 13_000_000, event="start", kind="gpu", segment_id="gemm")])
        later = advance_observed_state(state, 500_000_000)
        row = later["requests"]["one"]
        self.assertIsNone(row["kernel_cursor"])
        self.assertEqual(row["kernel_cursor_status"], "UNKNOWN")
        self.assertIsNone(row["phases"]["gemm"]["end_ns"])
        self.assertNotIn("forecast_cursor", row)
        finished = advance_observed_state(later, 600_000_000, feedback=[
            event(3, "PHASE", 550_000_000, 551_000_000, event="end", kind="gpu", segment_id="gemm"),
            event(4, "HANDLER_END", 560_000_000, 570_000_000),
            event(5, "WORKER_READY", 561_000_000, 571_000_000, clock="host_perf_counter_ns",
                  boundary="previous_handler_next_request_ready", scope="after_response_flush_before_next_read")])
        self.assertEqual(finished["requests"]["one"]["worker_next_ready_ns"], 561_000_000)
        self.assertEqual(finished["requests"]["one"]["parent_receipt_ns"], 570_000_000)
        self.assertIn("INITIAL_HANDOFF_RESIDUAL_UNMODELLED", finished["unknowns"])

    def test_future_visibility_is_rejected_even_if_worker_timestamp_is_past(self):
        with self.assertRaises(ValueError):
            advance_observed_state(initial(), 10, feedback=[event(1, "START", 1, 11)])
        with self.assertRaises(ValueError):
            advance_observed_state(initial(), 10, arrivals=[arrival(timestamp=10)])
        with self.assertRaises(ValueError):
            advance_observed_state(initial(), 10, admission_snapshot=raw_fixture()["admission_snapshot_after"])

    def test_feedback_gaps_duplicates_context_and_clocks_are_rejected(self):
        started = advance_observed_state(initial(), 10, feedback=[event(1, "START", 1, 2)])
        for bad in (
            event(1, "START", 3, 4), event(3, "HANDLER_END", 3, 4),
            {**event(2, "HANDLER_END", 3, 4), "clock": "unix_ns"},
            {**event(2, "HANDLER_END", 3, 4), "execution_context_sha256": "other"},
            event(2, "PHASE", 0, 4, event="start", kind="gpu", segment_id="x"),
        ):
            with self.assertRaises(ValueError):
                advance_observed_state(started, 10, feedback=[bad])

    def test_gate_snapshot_mutation_cursor_and_state_mismatch_are_rejected(self):
        snapshot = raw_fixture()["admission_snapshot_after"]
        for mutate in (
            lambda s: s.update(events_after_sequence=1),
            lambda s: s.update(configuration_sha256="other"),
            lambda s: s.update(active={"ghost": "s#0"}),
            lambda s: s.update(closed=True),
            lambda s: s["events"].pop(1),
            lambda s: s["events"][-1].update(active_count_after=1),
            lambda s: s["events"][-1].update(alias="g#0"),
        ):
            broken = copy.deepcopy(snapshot)
            mutate(broken)
            with self.assertRaises(ValueError):
                advance_observed_state(initial(), 40_000_000, admission_snapshot=broken)
        state = advance_observed_state(initial(), 40_000_000, admission_snapshot=snapshot)
        with self.assertRaises(ValueError):
            advance_observed_state(state, 40_000_000, admission_snapshot=snapshot)

    def test_gate_closure_cancels_waiter_without_releasing_active(self):
        snapshot = raw_fixture()["admission_snapshot_after"]
        snapshot["events"] = snapshot["events"][:2]
        snapshot["events"] += [
            {"sequence": 3, "kind": "enqueue", "time_ns": 21_000_000,
             "alias": "g#0", "request_id": "two", "reason": None,
             "active_count_after": 1, "waiting_count_after": 1},
            {"sequence": 4, "kind": "close", "time_ns": 22_000_000,
             "alias": None, "request_id": None, "reason": "runtime_shutdown",
             "active_count_after": 1, "waiting_count_after": 1},
            {"sequence": 5, "kind": "cancel", "time_ns": 23_000_000,
             "alias": "g#0", "request_id": "two", "reason": "gate_closed",
             "active_count_after": 1, "waiting_count_after": 0}]
        snapshot.update(active={"one": "s#0"}, waiting=[], closed=True, last_sequence=5)
        state = advance_observed_state(initial(), 40_000_000, admission_snapshot=snapshot)
        self.assertTrue(state["admission_snapshot"]["closed"])
        self.assertEqual(state["admission_snapshot"]["active"], {"one": "s#0"})
        self.assertNotIn("admission_release_ns", state["requests"]["one"])
        self.assertEqual(state["requests"]["two"]["admission_cancel_ns"], 23_000_000)

    def test_frozen_observer_cannot_consume_validation_future_or_switch_mode(self):
        state = advance_observed_state(initial("frozen", 10), 10, feedback=[event(1, "START", 1, 2)])
        with self.assertRaises(ValueError):
            advance_observed_state(state, 11)
        with self.assertRaises(ValueError):
            advance_observed_state(state, 10, feedback=[event(2, "HANDLER_END", 9, 11)])
        with self.assertRaises(ValueError):
            advance_observed_state(state, 10, mode="online")
        online = advance_observed_state(initial(), 11, feedback=[event(1, "START", 1, 11)])
        self.assertEqual(online["observed_at_ns"], 11)

    def test_mutated_checkpoint_and_future_labels_in_arrivals_are_rejected(self):
        state = initial()
        state["requests"]["fake"] = {"kernel_cursor": .5}
        with self.assertRaises(ValueError):
            advance_observed_state(state, 10)
        with self.assertRaises(ValueError):
            advance_observed_state(initial(), 10_000_000, arrivals=[{**arrival(), "label": "hidden"}])
        with self.assertRaises(ValueError):
            advance_observed_state(initial(), 10_000_000, arrivals=[arrival(sequence=2)])

    def test_phase_and_ready_boundaries_must_be_observed_in_order(self):
        for bad in (
            event(1, "PHASE", 1, 2, event="end", kind="gpu", segment_id="x"),
            event(1, "HANDLER_END", 1, 2),
            event(1, "WORKER_READY", 1, 2, clock="host_perf_counter_ns",
                  boundary="next_handler_start", scope="after_response_flush_before_next_read"),
        ):
            with self.assertRaises(ValueError):
                advance_observed_state(initial(), 10, feedback=[bad])


if __name__ == "__main__":
    unittest.main()
