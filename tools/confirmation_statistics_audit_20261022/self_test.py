"""Temporary CPU fixtures and rejection tests; never produce measured evidence."""

import copy
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from audit_statistics import audit, holm, recompute, t_df2_quantile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts/static-study-gpu-20261022"
VERIFIER = BASE / "confirmation-plot-source-v1/scripts/plot_static_confirmation.py"


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, allow_nan=False))


def fixture(folder):
    # Use production reporting only as the independent test oracle, never in
    # the auditor. These are declared synthetic utilities, not native outputs.
    spec = importlib.util.spec_from_file_location("upstream_fixture_oracle", BASE / "execution-source-v5/gsdeploy/study_reporting.py")
    oracle = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(oracle)
    labels = ["free_search_winner", "strongest_alternative_deployment", "strongest_alternative_route",
              "best_generalist_only", "best_specialist_only"]
    candidates = []
    for index, label in enumerate(labels):
        plan = dict(deployment={f"model_{index}": 1}, routing={"text": {f"model_{index}": 1.}})
        candidates.append(dict(plan_key=canonical(plan), plan=plan,
            labels=[label, "best_no_split"] if index == 0 else [label]))
    roles = [{"label": label, "plan_key": candidates[index]["plan_key"], "status": "OPTIMAL_FIXTURE",
              "optimality_certified": True} for index, label in enumerate(labels)]
    roles.insert(3, dict(label="best_no_split", plan_key=candidates[0]["plan_key"],
                         status="OPTIMAL_FIXTURE", optimality_certified=True))
    jobs = []
    for seed in (43, 44, 45):
        for c in candidates:
            identity = dict(seed=seed, deployment=c["plan"]["deployment"], routing=c["plan"]["routing"], workers={})
            jobs.append(dict(id=canonical(identity), identity=identity, candidate_key=c["plan_key"],
                             seed=seed, phase="confirmation", purpose="independent_confirmation_not_training",
                             plan=c["plan"], labels=c["labels"]))
    packet = dict(schema="gsdeploy.independent_confirmation.v1", candidates=candidates,
        candidate_roles=roles, jobs=jobs, config={"weights": {t: 1 / 6 for t in ("Q", "T", "L", "P", "M", "E")}},
        confirmation={"seeds": [43, 44, 45], "training_forbidden": True}, winner_label="free_search_winner")
    packet["confirmation_sha256"] = canonical(packet)
    packet_path = folder / "packet.json"
    save(packet_path, packet)
    rows = []
    # Cover positive, mixed-sign, zero-variance and all-zero differences.
    shifts = [(0., 0., 0.), (.1, .2, .3), (.1, -.2, .3), (.2, .2, .2), (0., 0., 0.)]
    for job in jobs:
        index = next(i for i, c in enumerate(candidates) if c["plan_key"] == job["candidate_key"])
        slot = job["seed"] - 43
        loss = .5 + slot / 10 + shifts[index][slot]
        utility = dict(eligible=True, energy_verified=True, **{t + "_norm": loss for t in packet["config"]["weights"]})
        utility["U"] = -math.fsum(packet["config"]["weights"][t] * loss for t in packet["config"]["weights"])
        raw = dict(experiment_job_id=job["id"], status="completed", backend="native", experiment_phase="confirmation",
            confirmation_packet_sha256=packet["confirmation_sha256"], plan=job["plan"],
            executed_measurement_id=canonical(job["identity"]), executed_measurement_identity=job["identity"])
        path = folder / (job["id"] + ".json")
        save(path, raw)
        rows.append(dict(job_id=job["id"], plan_key=job["candidate_key"], seed=job["seed"], labels=job["labels"],
            path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(), utility=utility))
    report = dict(schema="gsdeploy.independent_confirmation_report.v1", evidence_kind="CPU_FIXTURE_NOT_MEASURED",
        packet={"path": str(packet_path), "sha256": hashlib.sha256(packet_path.read_bytes()).hexdigest()},
        status="ALL_FROZEN_JOBS_VERIFIED", completed_jobs=15, planned_jobs=15, missing_job_ids=[],
        candidate_roles=roles, runs=oracle.measured_summary(rows, packet["config"]["weights"]),
        paired=oracle.paired_comparisons(rows, candidates[0]["plan_key"], [43, 44, 45],
                                       planned_plan_keys=[c["plan_key"] for c in candidates]),
        physical_global_optimality_certified=False)
    path = folder / "report.json"
    save(path, report)
    return path, report, packet


class StatisticsAuditTests(unittest.TestCase):
    def test_analytical_t_and_independent_holm_known_values(self):
        self.assertAlmostEqual(t_df2_quantile(.975), 4.302652729749464, places=12)
        self.assertAlmostEqual(t_df2_quantile(.025), -t_df2_quantile(.975), places=12)
        self.assertEqual(holm([.01, .04, .03, .2]), [.04, .09, .09, .2])
        self.assertEqual(holm([.25, .5, 1., .75]), [1., 1., 1., 1.])

    def test_temporary_complete_fixture_matches_upstream_all_statistics(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report, packet = fixture(Path(tmp))
            result = audit(path, VERIFIER, cpu_fixture=True)
            self.assertEqual(result["status"], "CPU_FIXTURE_STATISTICS_VERIFIED")
            self.assertFalse(result["full_statistical_audit_complete"])
            self.assertEqual(len(result["independent_comparisons"]), 4)
            p = [c["exact_two_sided_sign_flip_p"] for c in result["independent_comparisons"]]
            self.assertIn(.25, p)
            self.assertIn(1., p)
            with self.assertRaisesRegex(ValueError, "CPU fixtures require"):
                audit(path, VERIFIER)

    def test_all_comparison_fields_are_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, report, _ = fixture(Path(tmp))
            for field, value in report["paired"]["comparisons"][0].items():
                with self.subTest(field=field):
                    modified = copy.deepcopy(report)
                    if isinstance(value, bool): replacement = not value
                    elif isinstance(value, str): replacement = "altered"
                    elif isinstance(value, list): replacement = [] if value else [99]
                    elif isinstance(value, dict): replacement = {}
                    else: replacement = value + .1
                    modified["paired"]["comparisons"][0][field] = replacement
                    save(path, modified)
                    with self.assertRaisesRegex(ValueError, "differs"):
                        audit(path, VERIFIER, cpu_fixture=True)

    def test_duplicate_and_missing_blocks_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, report, packet = fixture(Path(tmp))
            rows = report["runs"]
            keys = [c["plan_key"] for c in packet["candidates"]]
            winner = packet["candidates"][0]["plan_key"]
            with self.assertRaisesRegex(ValueError, "all 15"):
                recompute(rows[:-1], winner, keys, [43, 44, 45])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                recompute(rows + [rows[0]], winner, keys, [43, 44, 45])

    def test_existing_real_four_group_partial_refused(self):
        with self.assertRaisesRegex(ValueError, "FINAL_AUDIT_REFUSED_INCOMPLETE"):
            audit(BASE / "confirmation-audit-seed43-four/report.json", VERIFIER)


if __name__ == "__main__":
    unittest.main(verbosity=2)
