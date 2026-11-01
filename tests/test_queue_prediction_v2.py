"""CPU analytic/statistical checks of queue adaptations; no GPU measurements."""
import math
import statistics
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.forecasting.queue_diagnostics import remaining_work, workload_envelopes
from gsdeploy.forecasting.lifecycle import LifecycleProvider
from tests.test_full_horizon_forecast import aliases, provider, profiles, request, resources, MS, SECOND
from gsdeploy.forecasting.simulator import FullHorizonModel


def lifecycle(receipt=20., release=30., ready=10., gated=True):
    document = {"schema": "gsdeploy.handler_lifecycle_profiles.v2", "execution_context_sha256": "fixture",
                "provenance": {"kind": "cpu_fixture", "split": "training", "sources": ["analytic"]},
                "profiles": []}
    for family in ("text", "generalist_2"):
        document["profiles"].append({"family": family, "task": "text", "pool_id": "x",
                                    "permit_scope": "request_admission" if gated else "not_applicable",
                                    "scenarios": {"mean": {"parent_receipt_ms": receipt,
                                        "permit_release_ms": release if gated else None, "worker_next_ready_ms": ready}},
                                    "evidence": ["analytic"]})
    return LifecycleProvider(document, "fixture", allow_cpu_fixture=True)


class ArrivalCVTests(unittest.TestCase):
    def test_gamma_rate_and_cv_are_separate_distribution_controls(self):
        for cv in (.5, 2.):
            w = FutureWorkload({"text": 1000}, {"text": ["x"]}, process="gamma_renewal",
                               interarrival_cv={"text": cv}, max_arrivals=20000)
            rows = w.suffix(0, 0, 10 * SECOND)["arrivals"]
            intervals = [b["arrival_ns"] - a["arrival_ns"] for a, b in zip(rows, rows[1:])]
            mean = statistics.mean(intervals)
            self.assertAlmostEqual(mean / MS, 1., delta=.08)
            self.assertAlmostEqual(statistics.stdev(intervals) / mean, cv, delta=.12)

    def test_gamma_midrun_requires_visible_age_and_rejects_future(self):
        w = FutureWorkload({"text": 1}, {"text": ["x"]}, process="gamma_renewal", interarrival_cv={"text": .3})
        with self.assertRaisesRegex(UnknownEvidence, "VISIBLE_LAST"):
            w.suffix(0, SECOND, 10 * SECOND)
        with self.assertRaisesRegex(ValueError, "no future"):
            w.suffix(0, SECOND, 10 * SECOND, observed_last_arrivals_ns={"text": SECOND + 1})
        args = dict(observed_last_arrivals_ns={"text": 0})
        a = w.suffix(0, SECOND, 10 * SECOND, **args)
        self.assertEqual(a, w.suffix(0, SECOND, 10 * SECOND, **args))
        self.assertFalse(a["actual_future_trace_used"])

    def test_conditional_residual_does_not_restart_renewal(self):
        w = FutureWorkload({"text": 1}, {"text": ["x"]}, process="gamma_renewal", interarrival_cv={"text": .2})
        fresh, aged = [], []
        for scenario in range(100):
            for last, target in ((900 * MS, fresh), (0, aged)):
                rows = w.suffix(0, 900 * MS, 5 * SECOND, scenario_index=scenario,
                                observed_last_arrivals_ns={"text": last})["arrivals"]
                target.append((rows[0]["arrival_ns"] - 900 * MS) / SECOND)
        self.assertGreater(statistics.mean(fresh), .85)
        self.assertLess(statistics.mean(aged), .3)

    def test_gamma_integrates_whole_rate_schedule_including_idle_gap(self):
        w = FutureWorkload({"text": 10}, {"text": ["x"]}, process="gamma_renewal", interarrival_cv={"text": 1.5},
                          windows=[{"start_seconds": 0., "end_seconds": 1., "rates": {"text": 50}},
                                   {"start_seconds": 1., "end_seconds": 2., "rates": {"text": 0}},
                                   {"start_seconds": 2., "end_seconds": 3., "rates": {"text": 100}}])
        rows = w.suffix(0, 0, 3 * SECOND)["arrivals"]
        self.assertTrue(any(r["arrival_ns"] < SECOND for r in rows))
        self.assertTrue(any(r["arrival_ns"] > 2 * SECOND for r in rows))
        self.assertFalse(any(SECOND <= r["arrival_ns"] < 2 * SECOND for r in rows))


class QueueWorkTests(unittest.TestCase):
    def test_remaining_resource_work_uses_cursor_not_queue_length(self):
        p = provider().plan({"alias": "s", **request("a")}, aliases(), "mean")
        work = remaining_work(p, {"segment_index": 0, "remaining_fraction": .25})
        self.assertEqual(work["solo_work_ms"], 25.)
        self.assertEqual(work["normalized_resource_work_ms"]["sm"], 20.)
        p["segments"][0]["demand"]["dram"] = None
        self.assertIsNone(remaining_work(p)["normalized_resource_work_ms"]["dram"])

    def test_multiscale_work_envelope_distinguishes_equal_count(self):
        rows = [request("a", arrival_ns=0), request("b", arrival_ns=10 * MS), request("c", arrival_ns=100 * MS)]
        e = workload_envelopes(rows, {"a": 10., "b": 100., "c": 1.}, scales_ns=(20 * MS, 100 * MS))
        self.assertEqual(e[0]["max_arrival_count"], 2)
        self.assertEqual(e[0]["max_offered_solo_work_ms"], 110.)
        self.assertEqual(e[1]["max_arrival_count"], 2)  # left endpoint is excluded
        self.assertIsNone(workload_envelopes(rows, {"a": 10.})[0]["max_offered_solo_work_ms"])

    def test_real_fifo_hol_and_recovery_are_reported_without_bypass(self):
        m = FullHorizonModel(aliases(), provider(), resources=resources(True, max_active=2))
        result = m.forecast({"s": [request("s0"), request("s1")]}, 0, SECOND,
                            future_arrivals=[request("g0", arrival_ns=25 * MS)], route_future=lambda *args: "g")
        self.assertEqual(result["queue_diagnostics"]["strict_fifo_hol_blocked_ms"], 75.)
        self.assertEqual(result["finishes"]["g0"]["start_ns"], 100 * MS)
        self.assertTrue(result["queue_diagnostics"]["queue_recovery_episodes"])
        self.assertFalse(result["queue_diagnostics"]["stability_rule_changed"])


class LifecycleEventTests(unittest.TestCase):
    def test_receipt_is_score_endpoint_but_release_controls_next_admission(self):
        m = FullHorizonModel(aliases(), provider(), resources=resources(True), lifecycle_provider=lifecycle())
        r = m.forecast({"s": [request("a"), request("b")]}, 0, SECOND)
        self.assertEqual(r["finishes"]["a"]["handler_end_ns"], 100 * MS)
        self.assertEqual(r["finishes"]["a"]["finish_ns"], 120 * MS)
        self.assertEqual(r["finishes"]["b"]["start_ns"], 130 * MS)
        events = {e["kind"]: e["time_ns"] for e in r["lifecycle_events"] if e.get("request_id") == "a"}
        self.assertEqual(events["permit_release"], 130 * MS)

    def test_worker_can_start_next_before_prior_parent_receipt_without_gate(self):
        m = FullHorizonModel(aliases(), provider(), lifecycle_provider=lifecycle(gated=False))
        r = m.forecast({"s": [request("a"), request("b")]}, 0, SECOND)
        self.assertEqual(r["finishes"]["a"]["finish_ns"], 120 * MS)
        self.assertEqual(r["finishes"]["b"]["start_ns"], 110 * MS)

    def test_receipt_after_deadline_and_bounded_drain_is_incomplete(self):
        m = FullHorizonModel(aliases(), provider(), lifecycle_provider=lifecycle(gated=False))
        r = m.forecast({"s": [request("a")]}, 0, 105 * MS, drain_deadline_ns=115 * MS)
        self.assertEqual(r["incomplete_ids"], ["a"])
        self.assertEqual(r["finishes"], {})

    def test_missing_readiness_never_defaults_to_zero(self):
        m = FullHorizonModel(aliases(), provider(), lifecycle_provider=lifecycle(ready=None))
        with self.assertRaisesRegex(UnknownEvidence, "worker_next_ready_ms"):
            m.forecast({"s": [request("a")]}, 0, SECOND)

    def test_explicit_stall_preserves_work_until_competitor_completion(self):
        class StallProvider:
            execution_context_sha256 = "fixture"
            def __init__(self):
                self.base = provider()
            def plan(self, *args): return self.base.plan(*args)
            def memory(self, *args): return self.base.memory(*args)
            def power(self, *args): return self.base.power(*args)
            def service(self, row, active, *args):
                if row["alias"] == "s" and "g" in active:
                    return dict(known=True, context_supported=True, stalled=True, service_ms=None)
                return dict(known=True, context_supported=True, service_ms=row["segment_duration_ms"])
        r = FullHorizonModel(aliases(), StallProvider()).forecast({"s": [request("s0")], "g": [request("g0")]}, 0, SECOND)
        self.assertEqual(r["finishes"]["s0"]["handler_end_ns"], 150 * MS)
        self.assertEqual(r["transitions"][0]["stalled_aliases"], ["s"])
        import json
        json.dumps(r, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
