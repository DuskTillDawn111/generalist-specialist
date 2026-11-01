"""Analytical CPU scenarios; values are fixtures, never GPU measurements."""

import copy
import contextlib
import io
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting import (FutureWorkload, FullHorizonModel, ProfileProvider,
                                  context_key, fit_progress, score_forecast)
from gsdeploy.forecasting.metrics import assess_prediction_pairs
from gsdeploy.forecasting.__main__ import forecast_request, main
from gsdeploy.runtime.admission import RequestAdmission
from tools.resource_control_20261024.forecast_policy import FullHorizonPolicy
from tools.resource_control_20261024.policy import ResourceBoundProvider


MS = 1_000_000
SECOND = 1_000_000_000


def aliases():
    return {"s": {"family": "text", "supported_tasks": ["text"]},
            "g": {"family": "generalist_2", "supported_tasks": ["text"]}}


def resources(gated=False, *, max_active=1):
    value = {"schema": "gsdeploy.resource_configuration.v1", "mode": "shared", "device": "0",
             "allocations": {a: {} for a in aliases()}}
    if gated:
        value.update(mode="request_admission", allocations={a: {"max_inflight": 1} for a in aliases()},
                     groups={"all": {"aliases": list(aliases()), "max_active": max_active}})
    return value


def request(rid, *, task="text", pool_id="x", arrival_ns=0, **kwargs):
    return {"request_id": rid, "task": task, "pool_id": pool_id, "arrival_ns": arrival_ns, **kwargs}


def profiles(*, ordered=False, s_ms=100., g_ms=50., speed=None, memory=None):
    """Enumerate tiny fixture contexts, not a physical interpolation model."""
    document = {"schema": "gsdeploy.segment_resource_profiles.v1", "execution_context_sha256": "fixture",
                "provenance": {"kind": "cpu_fixture", "sources": ["analytical-test"], "split": "training"},
                "profiles": [], "contexts": {}}
    for alias, family, ms in (("s", "text", s_ms), ("g", "generalist_2", g_ms)):
        segments = ([{"id": "host", "kind": "host", "duration_ms": {"mean": 10.},
                      "demand": {"sm": 0., "dram": 0., "l2": 0.}}] if ordered else [])
        segments.append({"id": "gpu", "kind": "gpu", "duration_ms": {"mean": ms},
                         "demand": {"sm": .8 if alias == "s" else .1,
                                    "dram": .1 if alias == "s" else .8, "l2": .2}})
        document["profiles"].append({"id": alias, "family": family, "task": "text", "pool_id": "x",
                                     "execution": {"batch_size": 1, "backend": "cpu_fixture", "weight_reuse": False},
                                     "segments": segments})
    options = [[None] + [{"profile_id": p["id"], "segment_id": s["id"]} for s in p["segments"]]
               for p in document["profiles"]]
    for pair in itertools.product(*options):
        members = {a: member for a, member in zip(("s", "g"), pair) if member is not None}
        speeds = {a: speed(a, members) if speed else 1 / max(1, len(members)) for a in members}
        document["contexts"][context_key(members)] = {"members": members, "evidence": "analytical-fixture-only",
            "scenarios": {"mean": {"progress": speeds, "board_watts": 50. + 100. * len(members),
                                    "peak_memory_mib": memory(members) if memory else 100. + 10. * len(members)}}}
    return document


def provider(**kwargs):
    return ProfileProvider(profiles(**kwargs), "fixture", allow_cpu_fixture=True)


class FutureWorkloadTests(unittest.TestCase):
    def test_same_suffix_independent_of_route_and_does_not_mutate(self):
        pools = {"text": ["a", "b"]}
        workload = FutureWorkload({"text": 50}, pools, namespace="forecast-only")
        first = workload.suffix(0, SECOND, 2 * SECOND, decision_id="current")
        self.assertEqual(first, workload.suffix(0, SECOND, 2 * SECOND, decision_id="current"))
        self.assertNotEqual(first["cohort_sha256"], workload.suffix(0, SECOND, 2 * SECOND, scenario_index=1, decision_id="current")["cohort_sha256"])
        self.assertTrue(all(SECOND < r["arrival_ns"] < 2 * SECOND for r in first["arrivals"]))
        self.assertFalse(first["actual_future_trace_used"])
        self.assertEqual(pools, {"text": ["a", "b"]})

    def test_poisson_preserves_bursts_not_regular_spacing(self):
        rows = FutureWorkload({"text": 100}, {"text": ["x"]}).suffix(0, 0, SECOND)["arrivals"]
        intervals = [b["arrival_ns"] - a["arrival_ns"] for a, b in zip(rows, rows[1:])]
        self.assertGreater(len(rows), 50)
        self.assertLess(min(intervals), 2 * MS)
        self.assertGreater(max(intervals), 20 * MS)

    def test_piecewise_all_future_rates_used(self):
        workload = FutureWorkload({"text": 50}, {"text": ["x"]}, windows=[
            {"start_seconds": 0., "end_seconds": 1., "rates": {"text": 0}},
            {"start_seconds": 1., "end_seconds": 2., "rates": {"text": 100}}])
        rows = workload.suffix(0, 0, 2 * SECOND)["arrivals"]
        self.assertGreater(len(rows), 50)
        self.assertTrue(all(r["arrival_ns"] > SECOND for r in rows))
        with self.assertRaisesRegex(ValueError, "full measurement"):
            workload.suffix(0, 0, 3 * SECOND)

    def test_periodic_suffix_retains_phase(self):
        workload = FutureWorkload({"text": 10}, {"text": ["x"]}, process="deterministic_periodic")
        full = workload.suffix(0, 0, SECOND)["arrivals"]
        suffix = workload.suffix(0, 450 * MS, SECOND)["arrivals"]
        self.assertEqual([r["arrival_ns"] for r in full if r["arrival_ns"] > 450 * MS],
                         [r["arrival_ns"] for r in suffix])

    def test_declared_planning_and_no_answer_or_actual_trace_fields(self):
        workload = FutureWorkload({"text": 1}, {"text": ["x"]}, process="declared_schedule",
            declared_arrivals=[{"request_id": "known", "offset_ns": 10 * MS, "task": "text", "pool_id": "x"}])
        suffix = workload.suffix(0, 0, SECOND)
        self.assertEqual(suffix["arrivals"][0]["request_id"], "known")
        self.assertTrue(suffix["declared_future_plan_used"])
        with self.assertRaisesRegex(ValueError, "labels"):
            FutureWorkload({"text": 1}, {"text": [{"pool_id": "x", "weight": 1, "answer": "secret"}]})
        with self.assertRaisesRegex(ValueError, "Actual future"):
            FutureWorkload({"text": 1}, {"text": ["x"]}, declared_arrivals=[])

    def test_budget_is_unknown_never_truncated(self):
        workload = FutureWorkload({"text": 100}, {"text": ["x"]}, max_arrivals=1)
        with self.assertRaisesRegex(UnknownEvidence, "NO_TRUNCATION"):
            workload.suffix(0, 0, SECOND)


class EventTests(unittest.TestCase):
    def model(self, source=None, **kwargs):
        return FullHorizonModel(aliases(), source or provider(), **kwargs)

    def test_future_arrival_reprices_running_request(self):
        result = self.model().forecast({"s": [request("s0")]}, 0, SECOND,
            future_arrivals=[request("g0", arrival_ns=25 * MS)], route_future=lambda *args: "g")
        self.assertEqual(result["finishes"]["g0"]["finish_ns"], 125 * MS)
        self.assertEqual(result["finishes"]["s0"]["finish_ns"], 150 * MS)
        self.assertEqual(result["future_arrival_count"], 1)
        # 25ms*150 + 100ms*250 + 25ms*150 + 850ms*50.
        self.assertAlmostEqual(result["board_energy_joules"], 75.)
        self.assertEqual(result["transitions"][1]["repriced_aliases"], ["g", "s"])

    def test_future_same_alias_waits_in_fifo(self):
        result = self.model().forecast({"s": [request("a")]}, 0, SECOND,
            future_arrivals=[request("b", arrival_ns=25 * MS)], route_future=lambda *args: "s")
        self.assertEqual(result["finishes"]["b"]["finish_ns"], 200 * MS)
        self.assertEqual(result["finishes"]["b"]["start_ns"], 100 * MS)

    def test_ordered_segments_reprice_at_internal_boundary(self):
        def speed(alias, active):
            return .5 if len(active) == 2 and all(r["segment_id"] == "gpu" for r in active.values()) else 1.
        result = self.model(provider(ordered=True, speed=speed)).forecast(
            {"s": [request("a")], "g": [request("b")]}, 0, SECOND)
        self.assertEqual(result["finishes"]["b"]["finish_ns"], 110 * MS)
        self.assertEqual(result["finishes"]["a"]["finish_ns"], 160 * MS)
        self.assertEqual(result["finishes"]["a"]["remaining_service_by_kind_ms"], {"host": 10., "gpu": 150.})
        self.assertEqual(result["transitions"][0]["segments"], {"s": "host", "g": "host"})
        self.assertEqual(result["transitions"][1]["segments"], {"s": "gpu", "g": "gpu"})

    def test_phase_alignment_changes_prediction_at_same_model_counts(self):
        def speed(alias, active):
            return .25 if len(active) == 2 and all(r["segment_id"] == "gpu" for r in active.values()) else 1.
        model = self.model(provider(ordered=True, speed=speed))
        same = model.forecast({"s": [request("a")], "g": [request("b")]}, 0, SECOND)
        shifted = model.forecast({"s": [request("a")]}, 0, SECOND,
            future_arrivals=[request("b", arrival_ns=100 * MS)], route_future=lambda *args: "g")
        self.assertLess(shifted["finishes"]["a"]["finish_ns"], same["finishes"]["a"]["finish_ns"])

    def test_current_progress_required_not_inferred_from_elapsed_age(self):
        with self.assertRaisesRegex(UnknownEvidence, "SEGMENT_PROGRESS"):
            self.model().forecast({"s": [request("a", head_age_ms=50.)]}, 100 * MS, SECOND)
        cursor = {"profile_id": "s", "segment_index": 0, "remaining_fraction": .25,
                  "observed_at_ns": 100 * MS, "evidence": "cpu-observed-progress"}
        result = self.model().forecast({"s": [request("a", forecast_cursor=cursor)]}, 100 * MS, SECOND)
        self.assertEqual(result["finishes"]["a"]["finish_ns"], 125 * MS)
        cursor["observed_at_ns"] = 99 * MS
        with self.assertRaisesRegex(ValueError, "current conditional"):
            self.model().forecast({"s": [request("a", forecast_cursor=cursor)]}, 100 * MS, SECOND)

    def test_nonpreemptive_admission_serializes_and_counts_waiting(self):
        result = self.model(resources=resources(True)).forecast(
            {"s": [request("s0")], "g": [request("g0")]}, 0, SECOND)
        # Predicted simultaneous sender offers are ordered by alias: g then s.
        self.assertEqual(result["finishes"]["g0"]["finish_ns"], 50 * MS)
        self.assertEqual(result["finishes"]["s0"]["start_ns"], 50 * MS)
        self.assertEqual(result["finishes"]["s0"]["finish_ns"], 150 * MS)
        self.assertEqual(result["finishes"]["s0"]["arrival_ns"], 0)
        gate = RequestAdmission(resources(True))
        self.assertTrue(gate.acquire("g", "g0"))
        self.assertFalse(gate._available("s"))
        gate.release("g0")
        self.assertTrue(gate._available("s"))

    def test_strict_fifo_sender_head_of_line_blocking(self):
        result = self.model(resources=resources(True, max_active=2)).forecast(
            {"s": [request("s0"), request("s1")], "g": []}, 0, SECOND,
            future_arrivals=[request("g0", arrival_ns=10 * MS)], route_future=lambda *args: "g")
        # s1 is already blocked behind active s0. g0 cannot bypass that sender
        # ticket even though the global second permit is initially free.
        self.assertEqual(result["finishes"]["g0"]["start_ns"], 100 * MS)
        self.assertEqual(result["finishes"]["g0"]["finish_ns"], 200 * MS)

    def test_observed_admission_snapshot_required_and_validated(self):
        rows = {"s": [request("s0", observed_pending=True)], "g": [request("g0", observed_pending=True)]}
        with self.assertRaisesRegex(UnknownEvidence, "ADMISSION_STATE"):
            self.model(resources=resources(True)).forecast(rows, 0, SECOND)
        state = {"active": {"s0": "s"}, "waiting": [("g", "g0")]}
        result = self.model(resources=resources(True)).forecast(rows, 0, SECOND, admission_state=state)
        self.assertEqual(result["finishes"]["g0"]["start_ns"], 100 * MS)
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.model(resources=resources(True)).forecast(rows, 0, SECOND,
                admission_state={"active": {"ghost": "s"}, "waiting": []})

    def test_drain_censoring_and_energy_window(self):
        result = self.model().forecast({"s": [request("a")]}, 0, 20 * MS, drain_deadline_ns=40 * MS)
        self.assertEqual(result["incomplete_ids"], ["a"])
        self.assertEqual(result["finishes"], {})
        self.assertAlmostEqual(result["board_energy_joules"], 3.)

    def test_idle_future_gap_and_peak_residency(self):
        result = self.model().forecast({}, 0, SECOND,
            future_arrivals=[request("b", arrival_ns=500 * MS)], route_future=lambda *args: "g")
        self.assertAlmostEqual(result["board_energy_joules"], 55.)
        self.assertEqual(result["peak_memory_mib"], 110.)

    def test_unknown_context_not_approximated_from_pairwise_or_counts(self):
        document = profiles()
        members = {a: {"profile_id": a, "segment_id": "gpu"} for a in aliases()}
        document["contexts"].pop(context_key(members))
        p = ProfileProvider(document, "fixture", allow_cpu_fixture=True)
        with self.assertRaisesRegex(UnknownEvidence, "ALIGNED_SEGMENT_CONTEXT"):
            self.model(p).forecast({"s": [request("a")], "g": [request("b")]}, 0, SECOND)

    def test_unimplemented_weight_merger_and_joint_batch_cannot_claim_gains(self):
        for field, value in (("weight_reuse", True), ("batch_size", 2)):
            document = profiles()
            document["profiles"][0]["execution"][field] = value
            p = ProfileProvider(document, "fixture", allow_cpu_fixture=True)
            with self.assertRaisesRegex(UnknownEvidence, "EXECUTOR_REQUIRED"):
                self.model(p).forecast({"s": [request("a")]}, 0, SECOND)

    def test_missing_memory_remains_unknown_and_event_budget_does_not_truncate(self):
        result = self.model(provider(memory=lambda active: None)).forecast({"s": [request("a")]}, 0, SECOND)
        self.assertIsNone(result["peak_memory_mib"])
        with self.assertRaisesRegex(UnknownEvidence, "PEAK_MEMORY"):
            self.model(provider(memory=lambda active: None)).forecast({"s": [request("a")]}, 0, SECOND, require_memory=True)
        with self.assertRaisesRegex(UnknownEvidence, "BUDGET"):
            self.model(max_events=1).forecast({"s": [request("a")]}, 0, SECOND)
        with self.assertRaisesRegex(UnknownEvidence, "COHORT_BUDGET"):
            self.model(max_requests=1).forecast({"s": [request("a"), request("b")]}, 0, SECOND)

    def test_alias_and_input_identity_and_mutation_checks(self):
        rows = {"s": [request("a")]}
        original = copy.deepcopy(rows)
        self.model().forecast(rows, 0, SECOND)
        self.assertEqual(rows, original)
        with self.assertRaisesRegex(UnknownEvidence, "INPUT_EXECUTION"):
            self.model().forecast({"s": [request("a", pool_id="unseen")]}, 0, SECOND)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.model().forecast({"s": [request("a"), request("a")]}, 0, SECOND)
        with self.assertRaisesRegex(ValueError, "no labels"):
            self.model().forecast({"s": [request("a", answer="unavailable")]}, 0, SECOND)


class ProfileTests(unittest.TestCase):
    def test_evidence_and_execution_configuration_gates(self):
        with self.assertRaisesRegex(ValueError, "CPU fixture"):
            ProfileProvider(profiles(), "fixture")
        with self.assertRaisesRegex(ValueError, "context mismatch"):
            ProfileProvider(profiles(), "other", allow_cpu_fixture=True)
        document = profiles()
        document["provenance"]["split"] = "validation"
        with self.assertRaisesRegex(ValueError, "training evidence"):
            ProfileProvider(document, "fixture", allow_cpu_fixture=True)

    def test_resource_classification_is_not_just_resident_memory(self):
        p = provider()
        self.assertEqual(p.demand_summary("s")["classification"], "compute")
        self.assertEqual(p.demand_summary("g")["classification"], "memory_bandwidth")

    def fit_input(self):
        document = profiles()
        solo = {"s": {"profile_id": "s", "segment_id": "gpu"}}
        shared = {**solo, "g": {"profile_id": "g", "segment_id": "gpu"}}
        def observation(rid, intervals):
            return {"id": rid, "split": "training", "execution_context_sha256": "fixture",
                    "target_alias": "s", "profile_id": "s", "segment_id": "gpu", "intervals": intervals}
        # Work equation: 100*solo_speed=100, then
        # 50*solo_speed + 100*shared_speed=100 => shared_speed=.5.
        rows = [observation("solo", [{"duration_ms": 100., "members": solo}]),
                observation("mixed", [{"duration_ms": 50., "members": solo}, {"duration_ms": 100., "members": shared}])]
        return document, rows, solo, shared

    def test_fit_uses_changing_phase_overlap_intervals(self):
        document, rows, solo, shared = self.fit_input()
        # Add corresponding complete observations for g so shared context is
        # fully supported, rather than assuming symmetric slowdown.
        gsolo = {"g": {"profile_id": "g", "segment_id": "gpu"}}
        for rid, member, duration in (("gsolo", gsolo, 50.), ("gshared", shared, 100.)):
            rows.append({"id": rid, "split": "training", "execution_context_sha256": "fixture",
                         "target_alias": "g", "profile_id": "g", "segment_id": "gpu",
                         "intervals": [{"duration_ms": duration, "members": member}]})
        fitted = fit_progress(document, rows)
        self.assertAlmostEqual(fitted["contexts"][context_key(shared)]["scenarios"]["mean"]["progress"]["s"], .5)
        self.assertAlmostEqual(fitted["contexts"][context_key(shared)]["scenarios"]["mean"]["progress"]["g"], .5)
        self.assertTrue(all(row["accepted"] for row in fitted["fit_diagnostics"]["groups"]))
        self.assertFalse("p90" in fitted["contexts"][context_key(shared)]["scenarios"])

    def test_rank_deficient_contexts_stay_unknown(self):
        document, rows, solo, shared = self.fit_input()
        rows = [copy.deepcopy(rows[1]), copy.deepcopy(rows[1])]
        rows[1]["id"] = "mixed2"
        fitted = fit_progress(document, rows)
        self.assertFalse(fitted["fit_diagnostics"]["groups"][0]["accepted"])
        self.assertNotIn(context_key(shared), fitted["contexts"])

    def test_training_rejects_heldout_and_wrong_execution_evidence(self):
        for key, value in (("split", "validation"), ("execution_context_sha256", "old")):
            document, rows, _, _ = self.fit_input()
            rows[0][key] = value
            with self.assertRaises(ValueError):
                fit_progress(document, rows)

    def test_poor_work_reconstruction_rejects_fit_instead_of_certifying(self):
        document, rows, solo, shared = self.fit_input()
        inconsistent = copy.deepcopy(rows[0])
        inconsistent["id"] = "bad-solo"
        inconsistent["intervals"][0]["duration_ms"] = 1000.
        rows.append(inconsistent)
        fitted = fit_progress(document, rows)
        self.assertFalse(fitted["fit_diagnostics"]["groups"][0]["accepted"])


class MetricTests(unittest.TestCase):
    def score(self, forecast, **kwargs):
        args = dict(phase_start_ns=0, now_ns=0, deadline_ns=SECOND, memory_budget_mib=1000.,
                    energy_reference_watts=100., observed_peak_memory_mib=0., observed_board_energy_joules=0.)
        args.update(kwargs)
        return score_forecast(forecast, [], aliases(), {"text": 1}, {"text": 100.},
                              {"text": {"text": 0.}, "generalist_2": {"text": .1}}, **args)

    def forecast(self, latencies):
        rows = {str(i): {**request(str(i)), "alias": "s", "finish_ns": round(ms * MS)}
                for i, ms in enumerate(latencies)}
        return {"cohort": copy.deepcopy(rows), "finishes": rows, "board_energy_joules": 50., "peak_memory_mib": 100.}

    def test_p95_recomputed_for_complete_cohort(self):
        even = self.score(self.forecast([100, 100, 100]))
        tail = self.score(self.forecast([0, 0, 300]))
        self.assertEqual(even["terms"]["L"], tail["terms"]["L"])
        self.assertAlmostEqual(tail["terms"]["P"], 2.7)
        self.assertGreater(tail["cost"], even["cost"])
        self.assertAlmostEqual(tail["full_U"], tail["metric_components"]["U"])

    def test_historical_peak_cannot_cancel_as_common_additive_constant(self):
        forecast = self.forecast([100])
        self.assertEqual(self.score(forecast, observed_peak_memory_mib=700.)["terms"]["M"], .7)
        metric = self.score(forecast, observed_peak_memory_mib=None)
        self.assertIsNone(metric["cost"])
        self.assertIn("UNKNOWN_WHOLE_PHASE_PEAK_MEMORY", metric["unknown_reasons"])

    def test_memory_safety_factor_matches_native_runtime_peak_accounting(self):
        metric = self.score(self.forecast([100]), observed_peak_memory_mib=700., memory_safe_factor=1.1)
        self.assertAlmostEqual(metric["terms"]["M"], .77)
        self.assertAlmostEqual(metric["safe_memory_mib"], 770.)

    def test_missing_past_energy_allows_comparison_but_no_absolute_U(self):
        metric = self.score(self.forecast([100]), observed_board_energy_joules=None)
        self.assertIsNotNone(metric["cost"])
        self.assertIsNone(metric["full_U"])
        self.assertTrue(metric["six_term_comparison_complete"])

    def test_drain_censored_latency_is_unknown_not_truncated_or_completed(self):
        forecast = self.forecast([100, 200])
        forecast["finishes"].pop("1")
        metric = self.score(forecast)
        self.assertIsNone(metric["cost"])
        self.assertIsNone(metric["terms"]["P"])
        self.assertEqual(metric["terms"]["T"], .5)

    def test_missing_quality_is_not_zero(self):
        forecast = self.forecast([100])
        forecast["cohort"]["0"]["alias"] = "g"
        forecast["finishes"]["0"]["alias"] = "g"
        metric = score_forecast(forecast, [], aliases(), {"text": 1}, {"text": 100}, {},
            phase_start_ns=0, now_ns=0, deadline_ns=SECOND, memory_budget_mib=1000,
            energy_reference_watts=100, observed_peak_memory_mib=0)
        self.assertIsNone(metric["terms"]["Q"])

    def test_ocr_edit_error_loss_above_one_keeps_existing_quality_semantics(self):
        row = {**request("ocr", task="ocr"), "alias": "ocr", "finish_ns": 100 * MS}
        forecast = {"cohort": {"ocr": row}, "finishes": {"ocr": row},
                    "board_energy_joules": 50., "peak_memory_mib": 100.}
        metric = score_forecast(forecast, [], {"ocr": {"family": "ocr", "supported_tasks": ["ocr"]}},
            {"ocr": 1}, {"ocr": 100}, {"ocr": {"ocr": {
                "basis": "expected_per_request_totals", "numerator": 15., "denominator": 10.,
                "evidence": "cpu-training"}}}, phase_start_ns=0, now_ns=0,
            deadline_ns=SECOND, memory_budget_mib=1000, energy_reference_watts=100,
            observed_peak_memory_mib=100)
        self.assertEqual(metric["terms"]["Q"], 1.5)
        self.assertTrue(metric["six_term_comparison_complete"])

    def test_heldout_delta_accuracy_not_training_fit_or_confidence_bound(self):
        rows = [{"id": "a", "split": "validation", "predicted_delta_U": .1, "observed_delta_U": -.1},
                {"id": "b", "split": "validation", "predicted_delta_U": .1, "observed_delta_U": .1}]
        metric = assess_prediction_pairs(rows)
        self.assertEqual(metric["positive_gain_classification_accuracy"], .5)
        self.assertAlmostEqual(metric["delta_mean_absolute_error"], .1)
        self.assertFalse(metric["confidence_bound_calibrated"])


class PolicyTests(unittest.TestCase):
    def policy(self, *, scheduled=None, source=None, **kwargs):
        workload = FutureWorkload({"text": 1}, {"text": ["x"]}, process="declared_schedule", declared_arrivals=scheduled or [])
        policy = FullHorizonPolicy(aliases(), source or provider(),
            {"text": {"text": 0.}, "generalist_2": {"text": 0.}}, {k: 1/6 for k in "QTLPME"},
            {"text": 100.}, {"text": 1.}, scenarios=("mean",), future_workload=workload,
            memory_budget_mib=1000., resources=resources(), arrival_scenarios=2, **kwargs)
        policy.start_phase(0, SECOND)
        return policy

    def snapshot(self):
        return {**{a: {"outstanding": []} for a in aliases()},
                "_forecast_state": {"observed_peak_memory_mib": 100., "observed_board_energy_joules": 0.}}

    def test_policy_really_compares_same_full_future_cohorts(self):
        p = self.policy(scheduled=[{"request_id": "future", "offset_ns": 10 * MS, "task": "text", "pool_id": "x"}])
        p.set_arrival("current", 0)
        result = p.choose("text", "x", 0, self.snapshot())
        scores = [m for a in result["audit"] for m in a["scenarios"].values()]
        self.assertEqual(len(scores), 4)
        self.assertTrue(all(m["cohort_ids"] == ["current", "future"] for m in scores))
        self.assertEqual(len({m["future_cohort_sha256"] for m in scores}), 1)
        self.assertTrue(result["future_arrival_scenarios_used"])
        self.assertFalse(result["physical_six_term_improvement_certified"])

    def test_full_future_externality_can_reverse_spill_decision(self):
        # g is individually faster, but a subsequent request goes to s.
        # Cross-model interference can outweigh current-request waiting gains.
        source = provider(s_ms=100., g_ms=90., speed=lambda alias, active: .05 if len(active) == 2 else 1.)
        p = self.policy(source=source)
        p.set_arrival("current", 0)
        solo = p.choose("text", "x", 0, self.snapshot())
        self.assertTrue(solo["actual_spill_selected"])
        # A same-alias future request has different queueing externalities.
        scheduled = [{"request_id": "future", "offset_ns": 10 * MS, "task": "text", "pool_id": "x"}]
        p = self.policy(source=source, scheduled=scheduled, continuation_factory=lambda: lambda *args: "s")
        p.set_arrival("current", 0)
        future = p.choose("text", "x", 0, self.snapshot())
        self.assertFalse(future["actual_spill_selected"])

    def test_unknown_memory_cannot_pass_as_zero_delta(self):
        p = self.policy()
        snapshot = self.snapshot()
        snapshot["_forecast_state"].pop("observed_peak_memory_mib")
        p.set_arrival("current", 0)
        result = p.choose("text", "x", 0, snapshot)
        self.assertFalse(result["actual_spill_selected"])
        self.assertTrue(any("PEAK_MEMORY" in reason for row in result["audit"] for reason in row["unknown_reasons"]))

    def test_no_spill_control_and_unknown_tail_scenario(self):
        p = self.policy(allow_spill=False)
        p.set_arrival("current", 0)
        self.assertFalse(p.choose("text", "x", 0, self.snapshot())["actual_spill_selected"])

    def test_context_bound_segment_provider_and_all_scenarios_are_exposed(self):
        bound = ResourceBoundProvider(provider(), "fixture")
        p = self.policy(source=bound)
        p.set_arrival("current", 0)
        result = p.choose("text", "x", 0, self.snapshot())
        self.assertTrue(result["six_term_comparison_complete"])
        self.assertEqual(set(result["predicted_U_by_scenario"]), {"mean@arrival-0", "mean@arrival-1"})
        self.assertIsNone(result["full_U"])

    def test_outside_window_falls_back_and_does_not_sample_invalid_future(self):
        p = self.policy()
        p.set_arrival("late", 2 * SECOND)
        result = p.choose("text", "x", 2 * SECOND, self.snapshot())
        self.assertFalse(result["actual_spill_selected"])
        self.assertIn("OUTSIDE_MEASUREMENT_ARRIVAL_WINDOW", result["gates"])


class CLITests(unittest.TestCase):
    def test_forecast_cpu_only_and_non_overwrite(self):
        value = {"schema": "gsdeploy.full_horizon_forecast_request.v1", "execution_context_sha256": "fixture",
                 "profiles": profiles(), "aliases": aliases(), "resources": resources(),
                 "queues": {"s": [request("a")]}, "now_ns": 0, "phase_start_ns": 0,
                 "deadline_ns": SECOND, "drain_deadline_ns": 2 * SECOND,
                 "future_workload": {"rates": {"text": 1}, "input_pools": {"text": ["x"]},
                                     "process": "declared_schedule", "declared_arrivals": []},
                 "continuation_routing_rps": {"text": {"s": 1}}, "latency_reference_ms": {"text": 100},
                 "quality_loss": {"text": {"text": 0.}}, "memory_budget_mib": 1000,
                 "memory_safe_factor": 1.,
                 "energy_reference_watts": 100, "observed_peak_memory_mib": 100., "observed_board_energy_joules": 0.}
        with patch("subprocess.Popen", side_effect=AssertionError("No processes/GPU probes")):
            result = forecast_request(value, allow_cpu_fixture=True)
        self.assertFalse(result["GPU_started"])
        self.assertFalse(result["physical_evidence"])
        self.assertIsNotNone(result["metrics"]["full_U"])
        with tempfile.TemporaryDirectory() as folder:
            source, output = Path(folder) / "input.json", Path(folder) / "output.json"
            source.write_text(json.dumps(value))
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(main(["forecast", "--input", str(source), "--output", str(output), "--allow-cpu-fixture"]), 0)
                with self.assertRaises(SystemExit):
                    main(["forecast", "--input", str(source), "--output", str(output), "--allow-cpu-fixture"])


if __name__ == "__main__":
    unittest.main()
