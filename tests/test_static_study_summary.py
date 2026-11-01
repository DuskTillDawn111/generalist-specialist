"""Actual JSON contracts with explicitly synthetic CPU-only study evidence."""
import copy
import json
from pathlib import Path
import shutil
import tempfile
import unittest

from gsdeploy.study_reporting import measured_summary, paired_comparisons
from scripts.report_static_study import (ALGORITHMS, ROLES, TERMS, assemble, canonical,
    confirmation_summary, efficiency_summary, load_search, receipt, sha)


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def cpu_fixture(root, *, certified=False, with_confirmation=False):
    failed = {"schema": "gsdeploy.calibrated_surrogate.v1", "status": "VALIDATION_FAILED",
        "validation": {"conditions": [{"errors": {"utility_absolute_error": .28}, "passed": False}],
                       "ranking": [{"ordering_consistent": False}]}}
    validated = {"schema": "gsdeploy.calibrated_replay.v2", "status": "VALIDATED_ON_HELDOUT_CONDITIONS",
        "version_sha256": "CPU_FIXTURE_VERSION", "validation": {"conditions": [{"passed": True}]}}
    models = {m: {"role": "generalist" if m.startswith("g") else "specialist", "name": "CPU_FIXTURE_" + m,
                  "supported_tasks": ["text"]} for m in ("g1", "g2", "g3", "text", "image", "ocr", "asr")}
    config = {"catalog": {"models": models, "capacity_factors": {m: {"0": 0, "1": 1, "2": 2} for m in models},
        "memory_table_mib": {m: {"1": 1, "2": 2} for m in models}, "performance_mode": "calibrated_replay_v2",
        "calibrated_replay_v2": {"version_sha256": validated["version_sha256"], "status": validated["status"]}},
        "weights": {t: 1 / 6 for t in TERMS}, "workload": {"text": 1.}, "route_grid": 4,
        "horizon_seconds": 1800, "random_seed": 42, "arrival_process": "poisson", "gpu_memory_mib": 49140,
        "search_policy": "nominal", "milp_time_limit_seconds": 120., "milp_absolute_gap": 1e-6}
    def utility(value):
        return {"eligible": True, "U": -value, **{t + "_norm": value for t in TERMS}}
    def result(deployment, routing, value=.2, problem=None):
        return {"status": "OPTIMAL" if certified else "MILP_BUDGET_LIMIT", "search_completed": certified,
            "optimality_certified": certified, "utility_upper_bound": -value if certified else -.1,
            "optimality_gap": 0 if certified else value - .1, "unknown_domain_lower_cost": None,
            "certificate_scope": "CPU fixture finite predictor domain", "unresolved_candidates": [],
            "metrics": {"full_prediction_calls": 10, "unknown_candidates": 0},
            "domain": {"K": 4}, "search_restrictions": {"replica_options": None},
            "external_wall_seconds": 120., "winner": {"deployment": deployment, "routing": routing,
                "prediction": utility(value), "score": -value, "objective": "nominal"},
            "problem": copy.deepcopy(problem or config)}
    plans = [({"g2": 1, "text": 1}, {"text": {"g2": .25, "text": .75}}),
        ({"g2": 1}, {"text": {"g2": 1.}}),
        ({"g2": 1, "text": 1}, {"text": {"g2": .5, "text": .5}}),
        ({"g2": 1}, {"text": {"g2": 1.}}),
        ({"g2": 1}, {"text": {"g2": 1.}}),
        ({"text": 1}, {"text": {"text": 1.}})]
    search_dir = root / "search"
    search_dir.mkdir()
    candidates = []
    for index, (label, (deployment, routing)) in enumerate(zip(ROLES, plans)):
        problem = copy.deepcopy(config)
        if label in {"best_generalist_only", "best_specialist_only"}:
            role = "generalist" if label == "best_generalist_only" else "specialist"
            selected = {m for m, spec in models.items() if spec["role"] == role}
            for field in ("models", "capacity_factors", "memory_table_mib"):
                problem["catalog"][field] = {m: v for m, v in problem["catalog"][field].items() if m in selected}
        value = .2 + index / 100
        row = result(deployment, routing, value, problem)
        candidates.append({"label": label, "search": row})
        write(search_dir / (label + ".json"), row)
    write(search_dir / "system-config.json", config)
    bundle = {"schema": "gsdeploy.static_search_roles.v1", "config_sha256": canonical(config),
        "candidates": candidates, "physical_global_optimality_certified": False, "gpu_started": False}
    write(search_dir / "search-results.json", bundle)
    write(search_dir / "manifest.json", {"files": {p.name: sha(p) for p in search_dir.iterdir()}})
    benchmark = root / "efficiency"
    benchmark.mkdir()
    preflight = {"predictor_version": validated["version_sha256"], "effective_problem_sha256": "fixture-problem",
        "trace_sha256": "fixture-trace", "fairness": {"absolute_gap": 1e-6}, "source_root": "CPU_FIXTURE_SOURCE",
        "domain": {"grid": 4, "all_route_deployment_identities": 4097}, "replica_options": {"g2": [1], "text": [1]}}
    records, summaries = [], []
    for index, algorithm in enumerate(ALGORITHMS):
        timing = float(10 * (index + 1))
        row = {"repeat": 0, "algorithm": algorithm, "search_wall_seconds": timing,
            "parent_process_wall_seconds": timing + 1, "peak_rss_kib": 1024, **preflight,
            "result": result({"g2": 1}, {"text": {"g2": 1.}})}
        records.append(row)
        write(benchmark / f"run-0-{algorithm}.json", row)
        summaries.append({"algorithm": algorithm, "completed_processes": 1, "certificates": int(certified),
            "median_search_wall_seconds": timing, "median_parent_wall_seconds": timing + 1,
            "median_full_predictions": 10, "max_peak_rss_kib": 1024})
    efficiency = {"preflight": preflight, "records": records, "summaries": summaries,
        "per_repeat": [{"repeat": 0, "objective_agrees": True, "all_certified": certified,
            "certified_objective_agrees": certified, "plan_identity_agrees": True}],
        "source_before": {"fixture.py": "fixture"}, "source_after": {"fixture.py": "fixture"},
        "source_unchanged": True, "benchmark_source_before": "fixture", "benchmark_source_after": "fixture",
        "config_unchanged": True, "same_contract": True, "benchmark_complete": certified,
        "timing_scope": "CPU_FIXTURE_NOT_MEASURED", "scope": "CPU fixture comparison only", "gpu_started": False}
    write(benchmark / "preflight.json", preflight)
    write(benchmark / "report.json", efficiency)
    write(root / "failed.json", failed)
    write(root / "validated.json", validated)
    inputs = {"calibration_report": root / "failed.json", "validated_report": root / "validated.json",
              "search_dir": search_dir, "efficiency_report": benchmark / "report.json"}
    if with_confirmation:
        unique, roles = {}, []
        for candidate in candidates:
            row = candidate["search"]
            plan = {k: row["winner"][k] for k in ("deployment", "routing")}
            key = canonical(plan)
            roles.append({"label": candidate["label"], "plan_key": key, "status": row["status"],
                "optimality_certified": row["optimality_certified"], "search": row})
            unique.setdefault(key, {"plan_key": key, "plan": plan, "labels": []})["labels"].append(candidate["label"])
        jobs = []
        for seed in (43, 44, 45):
            for key, candidate in unique.items():
                identity = {"seed": seed, "deployment": candidate["plan"]["deployment"], "routing": candidate["plan"]["routing"], "workers": {}}
                jobs.append({"id": canonical(identity), "identity": identity, "candidate_key": key,
                    "seed": seed, "labels": candidate["labels"], "plan": candidate["plan"], "phase": "confirmation",
                    "purpose": "independent_confirmation_not_training"})
        packet = {"config": config, "candidate_roles": roles, "candidates": list(unique.values()), "jobs": jobs,
            "frozen_search": receipt(search_dir / "search-results.json"), "confirmation": {"seeds": [43, 44, 45]}}
        packet["confirmation_sha256"] = canonical(packet)
        write(root / "packet.json", packet)
        rows = []
        for index, job in enumerate(jobs):
            raw = {"experiment_job_id": job["id"], "experiment_phase": "confirmation", "backend": "native", "status": "completed",
                "confirmation_packet_sha256": packet["confirmation_sha256"], "plan": job["plan"],
                "executed_measurement_identity": job["identity"], "executed_measurement_id": canonical(job["identity"])}
            raw_path = root / (job["id"] + ".json")
            write(raw_path, raw)
            rows.append({"job_id": job["id"], "plan_key": job["candidate_key"], "seed": job["seed"], "labels": job["labels"],
                **receipt(raw_path), "utility": utility(.3 + index / 100)})
        rows = measured_summary(rows, config["weights"])
        confirmation = {"schema": "gsdeploy.independent_confirmation_report.v1", "packet": receipt(root / "packet.json"),
            "candidate_roles": roles, "status": "ALL_FROZEN_JOBS_VERIFIED", "physical_global_optimality_certified": False,
            "runs": rows, "completed_jobs": len(jobs), "planned_jobs": len(jobs), "missing_job_ids": [],
            "paired": paired_comparisons(rows, roles[0]["plan_key"], [43, 44, 45], planned_plan_keys=list(unique))}
        write(root / "confirmation.json", confirmation)
        inputs["confirmation_report"] = root / "confirmation.json"
    return inputs


class StaticStudySummaryTests(unittest.TestCase):
    def test_incomplete_has_no_solve_speed_claim_and_missing_confirmation_is_null(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = cpu_fixture(root)
            report = assemble(**inputs, output=root / "summary", cpu_fixture=True)
            self.assertEqual(report["v1_failure"]["status"], "VALIDATION_FAILED")
            self.assertEqual(report["confirmation"]["status"], "NOT_DONE")
            self.assertIsNone(report["confirmation"]["completed_jobs"])
            self.assertIsNone(report["confirmation"]["paired"])
            self.assertEqual(len(report["static_search"]["models"]), 7)
            self.assertEqual(len(report["static_search"]["roles"]), 6)
            self.assertEqual(report["static_search"]["roles"][0]["winner"]["routing_share"]["text"], {"g2": .25, "text": .75})
            self.assertEqual(report["efficiency"]["observed_wall_ratio_rival_over_milp"], {"dfs": 2., "plain_dfs": 3.})
            self.assertEqual(report["efficiency"]["certified_solve_speedup_rival_over_milp"], {"dfs": None, "plain_dfs": None})
            self.assertFalse(report["physical_global_optimality_certified"])
            self.assertFalse(report["gpu_started"])
            text = (root / "summary/report.zh.md").read_text()
            self.assertIn("CPU_FIXTURE_NOT_MEASURED", text)
            self.assertIn("不能用 0 或预测值代替", text)
            self.assertIn("纯通预算候选", text)
            self.assertNotIn("### 最优纯通", text)
            self.assertNotIn("| 最优纯通 |", text)
            for item in report["receipts"]["search_files"] + report["receipts"]["efficiency_files"]:
                self.assertEqual(sha(item["path"]), item["sha256"])

    def test_certified_speed_and_formal_pairs_preserve_the_actual_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = assemble(**cpu_fixture(root, certified=True, with_confirmation=True), output=root / "summary", cpu_fixture=True)
            self.assertTrue(report["efficiency"]["milp_faster_than_both_at_certified_same_objective"])
            self.assertFalse(report["efficiency"]["full_seven_model_domain_proved_by_this_comparison"])
            self.assertEqual(report["confirmation"]["status"], "ALL_FROZEN_JOBS_VERIFIED")
            self.assertEqual(len(report["confirmation"]["paired"]["comparisons"]), 3)
            self.assertIn("Student", (root / "summary/report.zh.md").read_text())
            self.assertIn("### 最优纯通", (root / "summary/report.zh.md").read_text())

    def test_search_manifest_and_bundle_tamper_rejected(self):
        for mutation in ("sha", "bundle", "weight", "pool"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = cpu_fixture(root)
                search_dir = inputs["search_dir"]
                if mutation == "sha":
                    with (search_dir / "free_search_winner.json").open("a") as stream: stream.write(" ")
                else:
                    bundle = json.loads((search_dir / "search-results.json").read_text())
                    if mutation == "bundle": bundle["candidates"] = bundle["candidates"][:-1]
                    if mutation == "weight": bundle["candidates"][0]["search"]["winner"]["prediction"]["U"] = 0
                    if mutation == "pool": bundle["candidates"][0]["search"]["problem"]["catalog"]["models"].pop("g3")
                    write(search_dir / "search-results.json", bundle)
                    manifest = json.loads((search_dir / "manifest.json").read_text())
                    manifest["files"]["search-results.json"] = sha(search_dir / "search-results.json")
                    write(search_dir / "manifest.json", manifest)
                with self.assertRaises(ValueError):
                    assemble(**inputs, output=root / "refused", cpu_fixture=True)
                self.assertFalse((root / "refused").exists())

    def test_resealed_wrong_utility_rejected_and_no_plan_retained(self):
        for mutation in ("utility", "no_plan"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = cpu_fixture(root)
                directory = inputs["search_dir"]
                bundle = json.loads((directory / "search-results.json").read_text())
                candidate = bundle["candidates"][0 if mutation == "utility" else -1]
                if mutation == "utility":
                    candidate["search"]["winner"]["prediction"]["U"] = 0
                else:
                    candidate["search"].update({"winner": None, "status": "UNKNOWN_EVIDENCE", "optimality_gap": None})
                    candidate["search"]["metrics"]["unknown_candidates"] = 4
                write(directory / (candidate["label"] + ".json"), candidate["search"])
                write(directory / "search-results.json", bundle)
                write(directory / "manifest.json", {"files": {p.name: sha(p) for p in directory.iterdir() if p.name != "manifest.json"}})
                if mutation == "utility":
                    with self.assertRaisesRegex(ValueError, "U does not match"):
                        assemble(**inputs, output=root / "refused", cpu_fixture=True)
                else:
                    report = assemble(**inputs, output=root / "summary", cpu_fixture=True)
                    role = report["static_search"]["roles"][-1]
                    self.assertIsNone(role["winner"])
                    self.assertIsNone(role["optimality_gap"])
                    self.assertEqual(role["unknown_candidates"], 4)

    def test_efficiency_false_certificate_speed_or_missing_algorithm_rejected(self):
        for mutation in ("certificate", "median", "algorithm", "fairness", "file"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = cpu_fixture(root)
                path = inputs["efficiency_report"]
                report = json.loads(path.read_text())
                if mutation == "certificate": report["benchmark_complete"] = True
                if mutation == "median": report["summaries"][0]["median_search_wall_seconds"] = .1
                if mutation == "algorithm": report["records"] = report["records"][:-1]
                if mutation == "fairness": report["same_contract"] = False
                if mutation == "file": report["records"][0]["result"]["winner"]["score"] = 100
                write(path, report)
                with self.assertRaises(ValueError):
                    efficiency_summary(path, json.loads(inputs["validated_report"].read_text()))

    def test_confirmation_sha_seed_pair_and_winner_changes_rejected(self):
        for mutation in ("sha", "seed", "pair", "winner", "p"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = cpu_fixture(root, with_confirmation=True)
                path = inputs["confirmation_report"]
                report = json.loads(path.read_text())
                if mutation == "sha": report["runs"][0]["sha256"] = "wrong"
                if mutation == "seed": report["runs"][0]["seed"] = 42
                if mutation == "pair": report["paired"]["comparisons"][0]["mean_U_difference"] = 100
                if mutation == "winner": report["paired"]["winner_key"] = "changed"
                if mutation == "p": report["paired"]["comparisons"][0]["exact_two_sided_sign_flip_p"] = .001
                write(path, report)
                with self.assertRaises(ValueError):
                    assemble(**inputs, output=root / "refused", cpu_fixture=True)

    def test_incomplete_confirmation_keeps_missing_blocks_and_no_inference(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = cpu_fixture(root, with_confirmation=True)
            path = inputs["confirmation_report"]
            report = json.loads(path.read_text())
            removed = report["runs"].pop()
            report.update({"status": "INCOMPLETE_CONFIRMATION", "completed_jobs": len(report["runs"]), "missing_job_ids": [removed["job_id"]]})
            packet = json.loads((root / "packet.json").read_text())
            report["paired"] = paired_comparisons(report["runs"], report["paired"]["winner_key"], [43, 44, 45],
                planned_plan_keys=[c["plan_key"] for c in packet["candidates"]])
            write(path, report)
            result = assemble(**inputs, output=root / "partial", cpu_fixture=True)
            self.assertEqual(result["confirmation"]["status"], "INCOMPLETE_CONFIRMATION")
            self.assertEqual(result["confirmation"]["missing_job_ids"], [removed["job_id"]])
            partial = [p for p in result["confirmation"]["paired"]["comparisons"] if p["missing_seeds"]]
            self.assertTrue(partial)
            self.assertNotIn("exact_two_sided_sign_flip_p", partial[0])

    def test_nested_evidence_keeps_stronger_pure_candidate_and_frozen_formal_packet(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = cpu_fixture(root, with_confirmation=True)
            # The untouched packet is retained as an input-mutation sentinel;
            # it is not passed as confirmation after constructing another search.
            inputs.pop("confirmation_report")
            packet_before = (root / "packet.json").read_bytes()
            additional = root / "nested-search"
            shutil.copytree(inputs["search_dir"], additional)
            for directory, K, value, upper in ((inputs["search_dir"], 8, 5.676, -.555), (additional, 4, 3.367, -.635)):
                config = json.loads((directory / "system-config.json").read_text())
                config["route_grid"] = K
                write(directory / "system-config.json", config)
                bundle = json.loads((directory / "search-results.json").read_text())
                bundle["config_sha256"] = canonical(config)
                for candidate in bundle["candidates"]:
                    result = candidate["search"]
                    result["problem"]["route_grid"] = K
                    result["domain"]["K"] = K
                    if candidate["label"] == "best_generalist_only":
                        result["winner"]["prediction"].update({"U": -value, **{t + "_norm": value for t in TERMS}})
                        result["winner"]["score"] = -value
                        result["utility_upper_bound"] = upper
                        result["optimality_gap"] = upper + value
                    write(directory / (candidate["label"] + ".json"), result)
                write(directory / "search-results.json", bundle)
                write(directory / "manifest.json", {"files": {p.name: sha(p) for p in directory.iterdir() if p.name != "manifest.json"}})
            frozen_main = {p.name: sha(p) for p in inputs["search_dir"].iterdir()}
            report = assemble(**inputs, output=root / "summary", additional_search_evidence_dirs=[additional], cpu_fixture=True)
            pure = report["additional_search_evidence"]["pure_candidate_assessments"][0]
            self.assertAlmostEqual(pure["frozen_main_candidate"]["U"], -5.676)
            self.assertAlmostEqual(pure["best_reported_predictor_candidate"]["winner"]["U"], -3.367)
            self.assertEqual(pure["best_reported_predictor_candidate"]["source_K"], 4)
            self.assertTrue(pure["additional_candidate_stronger_than_frozen_main"])
            self.assertTrue(pure["excluded_from_full_pool_optimum_by_main_upper_bound"])
            self.assertAlmostEqual(pure["main_pure_domain_upper_bound"], -.555)
            self.assertFalse(pure["main_pure_domain_optimality_certified"])
            self.assertFalse(pure["pure_subdomain_proven_by_this_additional_evidence"])
            self.assertTrue(pure["formal_confirmation_roster_unchanged"])
            self.assertEqual((root / "packet.json").read_bytes(), packet_before)
            self.assertEqual({p.name: sha(p) for p in inputs["search_dir"].iterdir()}, frozen_main)
            text = (root / "summary/report.zh.md").read_text()
            self.assertIn("U=-3.367", text)
            self.assertIn("纯通预算候选", text)

    def test_additional_evidence_cannot_change_mathematical_contract_or_expand_main_grid(self):
        for mutation in ("larger_K", "workload", "receipt"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = cpu_fixture(root)
                additional = root / "additional"
                shutil.copytree(inputs["search_dir"], additional)
                if mutation == "receipt":
                    with (additional / "system-config.json").open("a") as stream: stream.write(" ")
                else:
                    config = json.loads((additional / "system-config.json").read_text())
                    if mutation == "larger_K": config["route_grid"] = 8
                    else: config["workload"]["text"] = 2.
                    write(additional / "system-config.json", config)
                    bundle = json.loads((additional / "search-results.json").read_text())
                    bundle["config_sha256"] = canonical(config)
                    for candidate in bundle["candidates"]:
                        candidate["search"]["problem"]["route_grid"] = config["route_grid"]
                        if mutation == "workload":
                            candidate["search"]["problem"]["workload"]["text"] = 2.
                            for model in candidate["search"]["winner"]["routing"]["text"]:
                                candidate["search"]["winner"]["routing"]["text"][model] *= 2
                        write(additional / (candidate["label"] + ".json"), candidate["search"])
                    write(additional / "search-results.json", bundle)
                    write(additional / "manifest.json", {"files": {p.name: sha(p) for p in additional.iterdir() if p.name != "manifest.json"}})
                with self.assertRaises(ValueError):
                    assemble(**inputs, output=root / "refused", additional_search_evidence_dirs=[additional], cpu_fixture=True)


if __name__ == "__main__":
    unittest.main()
