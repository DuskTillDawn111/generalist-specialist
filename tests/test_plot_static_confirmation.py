"""CPU fixtures only: no native import, measured observations, or GPU launch."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from gsdeploy.evidence import digest
from gsdeploy.quality import quality_digest
from gsdeploy.study_reporting import measured_summary
from scripts.plot_static_confirmation import plot, verify_report
from tests.test_confirmation import native_fixture, prepared


def fixture(root, *, absent_role=False, missing_candidate=False):
    packet, _ = prepared(root)
    if absent_role:
        role = packet["candidate_roles"][-1]
        key = role["plan_key"]
        role.update({"plan_key": None, "status": "INFEASIBLE", "optimality_certified": True})
        packet["candidates"] = [c for c in packet["candidates"] if c["plan_key"] != key]
        packet["jobs"] = [j for j in packet["jobs"] if j["candidate_key"] != key]
        packet["confirmation_sha256"] = quality_digest({k: v for k, v in packet.items() if k != "confirmation_sha256"})
    packet_path = root / "packet.json"
    packet_path.write_text(json.dumps(packet))
    rows = []
    for index, job in enumerate(packet["jobs"]):
        if missing_candidate and job["candidate_key"] == packet["candidates"][-1]["plan_key"]:
            continue
        raw = native_fixture(packet, job)
        path = root / (job["id"] + ".json")
        path.write_text(json.dumps(raw))
        utility = {"eligible": True, "energy_verified": True}
        utility.update({term + "_norm": .05 + index / 100 for term in ("Q", "T", "L", "P", "M", "E")})
        utility["U"] = -sum(packet["config"]["weights"][t] * utility[t + "_norm"] for t in packet["config"]["weights"])
        rows.append({"job_id": job["id"], "plan_key": job["candidate_key"], "seed": job["seed"],
                     "labels": job["labels"], "path": str(path), "sha256": digest(path), "utility": utility})
    seen = {r["job_id"] for r in rows}
    missing = sorted(j["id"] for j in packet["jobs"] if j["id"] not in seen)
    report = {"schema": "gsdeploy.independent_confirmation_report.v1",
        "evidence_kind": "CPU_FIXTURE_NOT_MEASURED", "packet": {"path": str(packet_path), "sha256": digest(packet_path)},
        "status": "INCOMPLETE_CONFIRMATION" if missing else "ALL_FROZEN_JOBS_VERIFIED",
        "completed_jobs": len(rows), "planned_jobs": len(packet["jobs"]), "missing_job_ids": missing,
        "runs": measured_summary(rows, packet["config"]["weights"]), "candidate_roles": packet["candidate_roles"],
        "physical_global_optimality_certified": False}
    path = root / "report.json"
    path.write_text(json.dumps(report))
    return path, report, packet


class ConfirmationPlotTests(unittest.TestCase):
    def test_exports_all_formats_and_keeps_absent_role(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, packet = fixture(root, absent_role=True)
            output = root / "figures"
            with self.assertRaisesRegex(ValueError, "require --cpu-fixture"):
                plot(path, root / "unmarked")
            self.assertFalse((root / "unmarked").exists())
            result = plot(path, output, cpu_fixture=True)
            self.assertEqual(result["evidence_kind"], "CPU_FIXTURE_NOT_MEASURED")
            self.assertFalse(result["physical_global_optimality_certified"])
            self.assertFalse(result["gpu_started"])
            self.assertEqual(len(list(output.glob("*.png"))), 3)
            self.assertEqual(len(list(output.glob("*.svg"))), 3)
            self.assertEqual(len(list(output.glob("*.pdf"))), 3)
            self.assertIn("CPU FIXTURE", (output / "confirmation-utility.svg").read_text())
            roster = json.loads((output / "candidate-roster.json").read_text())
            self.assertEqual(roster["roles"], packet["candidate_roles"])
            self.assertTrue(any(c["plan_key"] is None and c["role_status"] == "INFEASIBLE" for c in roster["candidates"]))
            for name, sha in result["files"].items():
                self.assertEqual(digest(output / name), sha)

    def test_partial_export_requires_flag_and_preserves_all_missing_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, _, packet = fixture(root, missing_candidate=True)
            with self.assertRaisesRegex(ValueError, "Incomplete report"):
                plot(path, root / "refused", cpu_fixture=True)
            self.assertFalse((root / "refused").exists())
            result = plot(path, root / "partial", allow_incomplete=True, cpu_fixture=True)
            self.assertEqual(result["status"], "INCOMPLETE_CONFIRMATION")
            roster = json.loads((root / "partial" / "candidate-roster.json").read_text())
            self.assertEqual(len(roster["candidates"]), len(packet["candidates"]))
            self.assertEqual(roster["candidates"][-1]["missing_seeds"], [43, 44, 45])
            self.assertIn("NO VALID RUNS", (root / "partial" / "confirmation-costs.svg").read_text())

    def test_packet_and_raw_run_sha_changes_rejected(self):
        for target in ("packet", "run"):
            with self.subTest(target=target), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                path, report, _ = fixture(root)
                receipt = report["packet"] if target == "packet" else report["runs"][0]
                with Path(receipt["path"]).open("a") as stream:
                    stream.write(" ")
                with self.assertRaisesRegex(ValueError, "Missing or changed receipt"):
                    verify_report(path, allow_incomplete=True)

    def test_duplicate_unknown_wrong_seed_and_utility_changes_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path, original, _ = fixture(Path(temporary))
            changes = ("duplicate", "unknown", "seed", "cost", "utility", "count", "global", "nan")
            for change in changes:
                with self.subTest(change=change):
                    report = copy.deepcopy(original)
                    if change == "duplicate": report["runs"].append(copy.deepcopy(report["runs"][0]))
                    if change == "unknown": report["runs"][0]["job_id"] = "unknown"
                    if change == "seed": report["runs"][0]["seed"] = 46
                    if change == "cost": report["runs"][0]["weighted_costs"]["Q"] += .1
                    if change == "utility": report["runs"][0]["utility"]["U"] += .1
                    if change == "count": report["completed_jobs"] -= 1
                    if change == "global": report["physical_global_optimality_certified"] = True
                    if change == "nan": report["runs"][0]["utility"]["Q_norm"] = float("nan")
                    path.write_text(json.dumps(report))
                    with self.assertRaises(ValueError):
                        verify_report(path, allow_incomplete=True)

    def test_rehashed_run_with_wrong_phase_or_identity_rejected(self):
        for change in ("phase", "identity"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                path, report, _ = fixture(Path(temporary))
                receipt = report["runs"][0]
                raw_path = Path(receipt["path"])
                raw = json.loads(raw_path.read_text())
                if change == "phase": raw["experiment_phase"] = "calibration"
                else:
                    raw["executed_measurement_identity"]["random_seed"] = 42
                    raw["executed_measurement_id"] = quality_digest(raw["executed_measurement_identity"])
                raw_path.write_text(json.dumps(raw))
                receipt["sha256"] = digest(raw_path)
                path.write_text(json.dumps(report))
                with self.assertRaises(ValueError):
                    verify_report(path, allow_incomplete=True)


if __name__ == "__main__":
    unittest.main()
