"""Preparation never upgrades field presence or synthetic claims to evidence."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.c1_qualification import build_packet, inventory_run, paired_route_evidence
from gsdeploy.config import SystemConfig


ROOT = Path(__file__).resolve().parents[1]


class QualificationTests(unittest.TestCase):
    def test_legacy_field_inventory_cannot_invent_stages_or_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory)/"run.json"
            p.write_text(json.dumps(dict(request_records=[dict(worker_response=dict(start_ns=1, end_ns=2))])))
            result = inventory_run(p)
            self.assertFalse(result["matching_training_qualified"])
            self.assertEqual(result["ordered_stage_fields"], 0)
            self.assertEqual(result["explicit_ready_fields"], 0)
            self.assertEqual(len(result["rejection_reasons"]), 2)

    def pairs(self):
        return [dict(kind="measured", split="confirmation", used_for_selection=False,
            execution_sha256="execution", normalization_sha256="reference", comparison_sha256="fixed-routes",
            trace_sha256=f"trace-{i}", seed=i, candidate_U=-.5, baseline_U=-.6)
            for i in (1, 2, 3)]

    def test_paired_interval_does_not_grant_physical_qualification(self):
        row = paired_route_evidence(self.pairs(), minimum_gain=.01)
        self.assertTrue(row["stable_gain_supported"])
        self.assertFalse(row["physical_qualification_granted"])

    def test_duplicate_selected_synthetic_or_changed_comparison_rejected(self):
        for field, value in (("seed", 1), ("used_for_selection", True), ("kind", "cpu_fixture"),
                             ("comparison_sha256", "other"), ("normalization_sha256", "other")):
            rows = self.pairs(); rows[1][field] = value
            with self.assertRaises(ValueError):
                paired_route_evidence(rows)

    def test_uncertain_small_difference_is_not_stable_gain(self):
        rows = self.pairs()
        for row, value in zip(rows, (-.3, -.9, -.5)):
            row["candidate_U"] = value
        self.assertFalse(paired_route_evidence(rows)["stable_gain_supported"])

    def test_current_packet_is_explicitly_blocked_without_fabricated_load(self):
        path = ROOT/"configs/system_reference_v2_20261103.json"
        if not (ROOT/"artifacts/utility-normalization-audit-20261101/catalog-calibrated-replay-v2.json").exists():
            self.skipTest("Optional local real catalog is absent")
        c = SystemConfig.load(path)
        packet = build_packet(c, hardware=dict(usable=False))
        self.assertIn("NORMALIZATION_REFERENCES_PENDING", packet["blockers"])
        self.assertFalse(packet["collector_implemented_by_this_packet"])
        self.assertFalse(packet["GPU_started"])
        self.assertEqual(packet["qualification_decision"]["status"], "NOT_RUN")
        hot = next(r for r in packet["scenarios"] if r["load_class"] != "original")
        self.assertIsNone(hot["workload_rps"])
        for row in packet["scenarios"]:
            for shares in row["routing_shares"].values():
                self.assertAlmostEqual(sum(shares.values()), 1.)


if __name__ == "__main__":
    unittest.main()
