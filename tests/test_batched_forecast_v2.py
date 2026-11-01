"""Independent hand-calculated CPU batch formation/service/cohort fixtures."""

import itertools
import copy
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.batch_profiles import BatchedProfileProvider, batch_input_key
from gsdeploy.forecasting.lifecycle import LifecycleProvider, LIFECYCLE_SCHEMA
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.profiles import ProfileProvider, PROFILE_SCHEMA, context_key
from gsdeploy.forecasting.simulator import FullHorizonModel
from gsdeploy.forecasting.__main__ import forecast_request
from gsdeploy.runtime.batching import BatchConfiguration, select_batch


MS = 1_000_000
CONTEXT = "independent-cpu-batch-fixture"
METADATA = {"text": {
    pool: {"input_tokens": tokens, "execution_key": "weights-v1", "input_ready_delay_ns": 0,
           "evidence": "public-cpu-input-fixture"} for pool, tokens in (("a", 2), ("b", 5), ("c", 9))}}


def row(rid, pool="a", arrival=0):
    value = {"request_id": rid, "task": "text", "pool_id": pool, "arrival_ns": arrival}
    if arrival == 0:
        value.update(batch_ready_ns=0, batch_ready_evidence="cpu-observed-prepared")
    return value


def make_provider(recipes, metadata=None):
    """Recipes maps owner to [(pool tuple, exact execution milliseconds)]."""
    metadata = METADATA if metadata is None else metadata
    aliases = {alias: {"family": "family-" + alias, "supported_tasks": ["text"]} for alias in recipes}
    doc = {"schema": PROFILE_SCHEMA, "execution_context_sha256": CONTEXT,
           "provenance": {"kind": "cpu_fixture", "split": "training", "sources": ["hand-calculated-batch-fixture"]},
           "profiles": [], "contexts": {}}
    keys = {}
    for alias, configurations in recipes.items():
        for index, (pools, duration) in enumerate(configurations):
            public = [{**row(str(i), pool), **metadata["text"][pool], "owner": alias, "alias": alias, "ready_ns": 0}
                      for i, pool in enumerate(pools)]
            selection = select_batch(public, 0, BatchConfiguration(len(pools), 1000, 0))
            pool_key = batch_input_key(public, selection)
            keys[alias, tuple(pools)] = pool_key
            doc["profiles"].append({"id": alias + "-batch-" + str(index), "family": aliases[alias]["family"],
                "task": "text", "pool_id": pool_key,
                "execution": {"batch_size": len(pools), "backend": "exact-cpu-batch-fixture", "weight_reuse": False},
                "segments": [{"id": "gpu", "kind": "gpu", "duration_ms": {"mean": duration},
                              "demand": {"sm": .2, "dram": .3, "l2": .1}}]})
        # Solo profiles are distinct cost evidence used only by diagnostics.
        for pool in metadata["text"]:
            doc["profiles"].append({"id": alias + "-solo-" + pool, "family": aliases[alias]["family"],
                "task": "text", "pool_id": pool,
                "execution": {"batch_size": 1, "backend": "exact-cpu-solo-fixture", "weight_reuse": False},
                "segments": [{"id": "gpu", "kind": "gpu", "duration_ms": {"mean": 20.},
                              "demand": {"sm": .2, "dram": .3, "l2": .1}}]})
    choices = [[None] + [{"profile_id": p["id"], "segment_id": "gpu"} for p in doc["profiles"]
                        if p["family"] == aliases[alias]["family"]] for alias in aliases]
    for selection in itertools.product(*choices):
        members = {alias: value for alias, value in zip(aliases, selection) if value is not None}
        doc["contexts"][context_key(members)] = {"members": members, "evidence": "exact-cpu-context",
            "scenarios": {"mean": {"progress": {alias: 1. for alias in members},
                "board_watts": 10. + 10. * len(members), "peak_memory_mib": 100. + 5. * len(members)}}}
    provider = BatchedProfileProvider(ProfileProvider(doc, CONTEXT, allow_cpu_fixture=True), metadata,
                                      execution_context_sha256=CONTEXT)
    return aliases, provider, keys


def lifecycle(aliases, keys, *, receipt_ms, ready_ms):
    doc = {"schema": LIFECYCLE_SCHEMA, "execution_context_sha256": CONTEXT,
           "provenance": {"kind": "cpu_fixture", "split": "training", "sources": ["cpu-handoff"]}, "profiles": []}
    for (alias, _), key in keys.items():
        doc["profiles"].append({"family": aliases[alias]["family"], "task": "text", "pool_id": key,
            "permit_scope": "not_applicable", "evidence": ["cpu-joint-handoff-vector"],
            "scenarios": {"mean": {"parent_receipt_ms": receipt_ms, "permit_release_ms": None,
                                    "worker_next_ready_ms": ready_ms}}})
    return LifecycleProvider(doc, CONTEXT, allow_cpu_fixture=True)


class BatchedForecastTests(unittest.TestCase):
    def test_cli_contract_connects_exact_batch_profiles_lifecycle_and_full_score(self):
        aliases, provider, keys = make_provider({"owner": [(("a", "b"), 12.)]})
        handoff = lifecycle(aliases, keys, receipt_ms=3., ready_ms=1.)
        request = {"schema": "gsdeploy.full_horizon_forecast_request.v1", "aliases": aliases,
            "execution_context_sha256": CONTEXT, "profiles": provider.provider.document,
            "lifecycle_profiles": handoff.document, "execution_contract": "strict_lifecycle_v2",
            "batch_configurations": {"owner": {"max_batch_size": 2, "max_padded_tokens": 30,
                                               "max_wait_ns": 20 * MS, "length_buckets": []}},
            "batch_input_metadata": METADATA,
            "future_workload": {"rates": {"text": 1.}, "input_pools": {"text": ["a", "b"]},
                "process": "declared_schedule", "declared_arrivals": [
                    {"request_id": "second", "task": "text", "pool_id": "b", "offset_ns": 5 * MS}]},
            "now_ns": 0, "phase_start_ns": 0, "deadline_ns": 40 * MS, "drain_deadline_ns": 50 * MS,
            "queues": {"owner": [row("first")]}, "continuation_routing_rps": {"text": {"owner": 1.}},
            "resources": {"schema": "gsdeploy.resource_configuration.v1", "mode": "shared", "device": "0",
                          "allocations": {"owner": {}}},
            "latency_reference_ms": {"text": 100.}, "quality_loss": {"family-owner": {"text": .2}},
            "memory_budget_mib": 1000., "energy_reference_watts": 100., "memory_safe_factor": 1.,
            "observed_peak_memory_mib": 100., "observed_board_energy_joules": 0.}
        result = forecast_request(request, allow_cpu_fixture=True)
        self.assertFalse(result["GPU_started"])
        self.assertFalse(result["physical_evidence"])
        self.assertEqual(result["execution_contract"], "strict_lifecycle_v2")
        self.assertEqual(result["metrics"]["cohort_size"], 2)
        self.assertEqual({r["pool_id"] for r in result["forecast"]["finishes"].values()}, {"a", "b"})
        self.assertTrue(all(r["finish_ns"] == 20 * MS for r in result["forecast"]["finishes"].values()))
        self.assertAlmostEqual(result["metrics"]["terms"]["L"], .175)
        self.assertAlmostEqual(result["metrics"]["terms"]["P"], .1975)

    def test_serial_preparation_accepts_different_input_costs_without_reordering_fifo(self):
        metadata = copy.deepcopy(METADATA)
        metadata["text"]["a"]["input_ready_delay_ns"] = 10 * MS
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]}, metadata)
        pending = [row("first"), row("second", "b")]
        for member in pending:
            member.pop("batch_ready_ns")
            member.pop("batch_ready_evidence")
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)}).forecast(
                {"owner": pending}, 0, 40 * MS, require_energy=False)
        self.assertEqual(result["batch_execution"][0]["members"], ["first", "second"])
        self.assertEqual(result["batch_execution"][0]["dispatch_ns"], 10 * MS)
        self.assertEqual(result["finishes"]["first"]["finish_ns"], 22 * MS)
        self.assertEqual(result["finishes"]["second"]["finish_ns"], 22 * MS)

    def test_actual_serial_preparation_finishes_both_inputs_before_full_batch(self):
        metadata = copy.deepcopy(METADATA)
        metadata["text"]["a"]["input_ready_delay_ns"] = 4 * MS
        metadata["text"]["b"]["input_ready_delay_ns"] = 6 * MS
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]}, metadata)
        pending = [row("first"), row("second", "b")]
        for member in pending:
            member.pop("batch_ready_ns")
            member.pop("batch_ready_evidence")
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)}).forecast(
                {"owner": pending}, 0, 40 * MS, require_energy=False)
        self.assertEqual(result["batch_execution"][0]["dispatch_ns"], 10 * MS)
        self.assertEqual(result["batch_execution"][0]["formation_wait_ns"], {"first": 6 * MS, "second": 0})
        self.assertTrue(all(member["finish_ns"] == 22 * MS for member in result["finishes"].values()))

    def test_maxwait_zero_dispatches_ready_singleton_and_preparation_waits_for_generation(self):
        metadata = copy.deepcopy(METADATA)
        metadata["text"]["a"]["input_ready_delay_ns"] = 4 * MS
        metadata["text"]["b"]["input_ready_delay_ns"] = 6 * MS
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.), (("b",), 5.)]}, metadata)
        pending = [row("first"), row("second", "b")]
        for member in pending:
            member.pop("batch_ready_ns")
            member.pop("batch_ready_evidence")
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 0)}).forecast(
                {"owner": pending}, 0, 40 * MS, require_energy=False)
        self.assertEqual([batch["members"] for batch in result["batch_execution"]], [["first"], ["second"]])
        self.assertEqual([batch["dispatch_ns"] for batch in result["batch_execution"]], [4 * MS, 13 * MS])
        self.assertEqual(result["finishes"]["first"]["finish_ns"], 7 * MS)
        self.assertEqual(result["finishes"]["second"]["finish_ns"], 18 * MS)

    def test_future_arrival_fills_batch_before_timeout_with_exact_padding_and_cost(self):
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]})
        model = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)})
        result = model.forecast({"owner": [row("first")]}, 0, 40 * MS,
            future_arrivals=[row("second", "b", 5 * MS)], route_future=lambda *args: "owner")
        batch = result["batch_execution"][0]
        self.assertEqual(batch["dispatch_ns"], 5 * MS)
        self.assertEqual(batch["dispatch_reason"], "full")
        self.assertEqual(batch["formation_wait_ns"], {"first": 5 * MS, "second": 0})
        self.assertEqual((batch["useful_input_tokens"], batch["padded_input_tokens"], batch["padding_tokens"]), (7, 10, 3))
        self.assertAlmostEqual(batch["padding_fraction"], .3)
        self.assertEqual(set(result["cohort"]), {"first", "second"})
        self.assertEqual(set(result["finishes"]), {"first", "second"})
        self.assertTrue(all(r["finish_ns"] == 17 * MS for r in result["finishes"].values()))
        self.assertTrue(all(r["predicted_remaining_service_ms"] == 12. for r in result["finishes"].values()))
        # Board power is charged once: idle 5ms + active 12ms + idle 23ms.
        self.assertAlmostEqual(result["board_energy_joules"], .52)
        self.assertEqual(result["peak_memory_mib"], 105.)
        score = score_forecast(result, [], aliases, {"text": 1.}, {"text": 100.},
            {"family-owner": {"text": .2}}, phase_start_ns=0, now_ns=0, deadline_ns=40 * MS,
            memory_budget_mib=1000., energy_reference_watts=100.,
            observed_peak_memory_mib=100., observed_board_energy_joules=0.)
        self.assertEqual(score["cohort_size"], 2)
        self.assertAlmostEqual(score["terms"]["L"], .145)
        self.assertAlmostEqual(score["terms"]["P"], .1675)
        self.assertEqual(score["terms"]["T"], 0.)
        self.assertAlmostEqual(score["terms"]["M"], .105)
        self.assertAlmostEqual(score["terms"]["E"], .13)

    def test_timeout_dispatches_singleton_and_does_not_assume_batch_speedup(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 23.)]})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(3, 30, 7 * MS)}).forecast(
                {"owner": [row("first")]}, 0, 40 * MS)
        self.assertEqual(result["batch_execution"][0]["dispatch_reason"], "timeout")
        self.assertEqual(result["batch_execution"][0]["dispatch_ns"], 7 * MS)
        self.assertEqual(result["finishes"]["first"]["finish_ns"], 30 * MS)

    def test_padding_budget_barrier_dispatches_fifo_prefix(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.), (("b",), 5.)]})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 8, 20 * MS)}).forecast(
                {"owner": [row("a"), row("b", "b")]}, 0, 40 * MS)
        self.assertEqual([b["members"] for b in result["batch_execution"]], [["a"], ["b"]])
        self.assertEqual(result["batch_execution"][0]["dispatch_reason"], "fifo_barrier")
        self.assertEqual(result["finishes"]["a"]["finish_ns"], 3 * MS)
        self.assertEqual(result["finishes"]["b"]["finish_ns"], 25 * MS)

    def test_bounded_drain_keeps_every_member_incomplete_and_energy_window_only(self):
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)}).forecast(
                {"owner": [row("first")]}, 0, 10 * MS, drain_deadline_ns=15 * MS,
                future_arrivals=[row("second", "b", 5 * MS)], route_future=lambda *args: "owner")
        self.assertEqual(result["incomplete_ids"], ["first", "second"])
        self.assertEqual(result["finishes"], {})
        self.assertEqual(result["drain_finish_ns"], 15 * MS)
        self.assertAlmostEqual(result["board_energy_joules"], .15)

    def test_drain_completion_does_not_erase_deadline_timeout_for_batch_members(self):
        aliases, provider, _ = make_provider({"owner": [(("a", "b"), 12.)]})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 20 * MS)}).forecast(
                {"owner": [row("first")]}, 0, 10 * MS, drain_deadline_ns=30 * MS,
                future_arrivals=[row("second", "b", 5 * MS)], route_future=lambda *args: "owner")
        score = score_forecast(result, [], aliases, {"text": 1.}, {"text": 100.},
            {"family-owner": {"text": .2}}, phase_start_ns=0, now_ns=0, deadline_ns=10 * MS,
            memory_budget_mib=1000., energy_reference_watts=100.,
            observed_peak_memory_mib=100., observed_board_energy_joules=0.)
        self.assertEqual(result["incomplete_ids"], [])
        self.assertEqual(score["cohort_size"], 2)
        self.assertEqual(score["terms"]["T"], 1.)
        self.assertAlmostEqual(score["terms"]["L"], .145)
        self.assertAlmostEqual(score["terms"]["P"], .1675)
        self.assertAlmostEqual(result["board_energy_joules"], .15)

    def test_arrival_age_changes_timeout_without_restarting_wait_at_forecast_now(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.)]})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={"owner": BatchConfiguration(2, 30, 10 * MS)}).forecast(
                {"owner": [row("first", arrival=0)]}, 7 * MS, 30 * MS)
        self.assertEqual(result["batch_execution"][0]["dispatch_ns"], 10 * MS)
        self.assertEqual(result["batch_execution"][0]["formation_wait_ns"], {"first": 10 * MS})
        self.assertEqual(result["finishes"]["first"]["finish_ns"], 13 * MS)

    def test_request_gate_and_inflight_batch_require_real_matching_runtime(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 3.)]})
        resources = {"schema": "gsdeploy.resource_configuration.v1", "mode": "request_admission", "device": "0",
            "allocations": {"owner": {"max_inflight": 8}}, "groups": {"all": {"aliases": ["owner"], "max_active": 8}}}
        with self.assertRaisesRegex(ValueError, "matching runtime gate"):
            FullHorizonModel(aliases, provider, resources=resources,
                batch_configurations={"owner": BatchConfiguration(2, 30, 0)})
        with self.assertRaisesRegex(UnknownEvidence, "OBSERVED_INFLIGHT_BATCH_STATE_REQUIRED"):
            FullHorizonModel(aliases, provider,
                batch_configurations={"owner": BatchConfiguration(2, 30, 0)}).forecast(
                    {"owner": [{**row("first"), "head_age_ms": 1.}]}, 0, 10 * MS)

    def test_three_owners_start_together_without_hidden_cap_two(self):
        aliases, provider, _ = make_provider({alias: [(("a", "a"), 10.)] for alias in ("one", "two", "three")})
        result = FullHorizonModel(aliases, provider,
            batch_configurations={alias: BatchConfiguration(2, 30, 0) for alias in aliases}).forecast(
                {alias: [row(alias + "a"), row(alias + "b")] for alias in aliases}, 0, 20 * MS)
        self.assertEqual(len(result["transitions"][0]["active_requests"]), 3)
        self.assertEqual(len(result["finishes"]), 6)
        self.assertTrue(all(r["finish_ns"] == 10 * MS for r in result["finishes"].values()))
        self.assertAlmostEqual(result["board_energy_joules"], .5)

    def test_batch_handoff_delivers_members_after_receipt_but_starts_next_on_ready(self):
        aliases, provider, keys = make_provider({"owner": [(("a", "a"), 10.)]})
        model = FullHorizonModel(aliases, provider, lifecycle_provider=lifecycle(aliases, keys, receipt_ms=5., ready_ms=2.),
            batch_configurations={"owner": BatchConfiguration(2, 30, 0)})
        result = model.forecast({"owner": [row(str(i)) for i in range(4)]}, 0, 40 * MS)
        self.assertEqual([b["dispatch_ns"] for b in result["batch_execution"]], [0, 12 * MS])
        self.assertEqual([result["finishes"][str(i)]["finish_ns"] for i in range(4)],
                         [15 * MS, 15 * MS, 27 * MS, 27 * MS])
        self.assertEqual([result["finishes"][str(i)]["start_ns"] for i in range(4)],
                         [0, 0, 12 * MS, 12 * MS])
        self.assertIn("explicit", result["lifecycle_scope"])

    def test_missing_exact_batch_profile_is_unknown_without_solo_substitution(self):
        aliases, provider, _ = make_provider({"owner": [(("a",), 10.)]})
        with self.assertRaisesRegex(UnknownEvidence, "UNKNOWN_INPUT_EXECUTION_PROFILE"):
            FullHorizonModel(aliases, provider,
                batch_configurations={"owner": BatchConfiguration(2, 30, 0)}).forecast(
                    {"owner": [row("a"), row("b", "b")]}, 0, 20 * MS)

    def test_resource_stall_never_becomes_instant_completion(self):
        aliases, provider, _ = make_provider({"owner": [(("a", "a"), 10.)]})

        class StalledProvider(BatchedProfileProvider):
            def service(self, *args):
                return {"known": True, "context_supported": True, "stalled": True, "service_ms": None}

        stalled = StalledProvider(provider.provider, METADATA, execution_context_sha256=CONTEXT)
        result = FullHorizonModel(aliases, stalled,
            batch_configurations={"owner": BatchConfiguration(2, 30, 0)}).forecast(
                {"owner": [row("a"), row("b")]}, 0, 10 * MS, drain_deadline_ns=20 * MS)
        self.assertEqual(result["finishes"], {})
        self.assertEqual(result["incomplete_ids"], ["a", "b"])
        self.assertEqual(result["transitions"][0]["stalled_aliases"], ["owner"])


if __name__ == "__main__":
    unittest.main()
