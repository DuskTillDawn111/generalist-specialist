"""CPU-only behavioral tests; no model imports or GPU execution."""

import json
from pathlib import Path
import unittest

from placement.framework import (
    Constraints, Deployment, Hardware, MissingPerformanceError, Model,
    PerformanceModel, PerformanceResult, TablePerformanceModel, Workload,
    generate_candidates, optimize,
)
from placement.framework.__main__ import load_catalog

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "placement/examples"


class FrameworkTests(unittest.TestCase):
    def setUp(self):
        self.catalog = load_catalog(ROOT / "benchmark/profiles/model_service_profile_6task.json")
        self.hardware = Hardware(49140)
        self.backend = TablePerformanceModel.from_json(EXAMPLES / "performance_table.synthetic.json")
        self.workload = Workload(json.loads((EXAMPLES / "workload_mixed.json").read_text()))

    def test_vqa_requires_qwen_and_enumerates_all_subsets(self):
        candidates = list(generate_candidates(self.catalog, self.workload, self.hardware))
        self.assertEqual(len(candidates), 16)
        self.assertTrue(all("qwen" in d.model_counts for d in candidates))
        self.assertEqual(len({d.key for d in candidates}), 16)

    def test_specialists_cover_five_tasks_without_vqa(self):
        workload = Workload({task: 1 for task in ("text", "image", "ocr", "asr", "lid")})
        candidates = list(generate_candidates(self.catalog, workload, self.hardware))
        self.assertIn(Deployment({m: 1 for m in self.catalog if m != "qwen"}), candidates)
        lid = list(generate_candidates(self.catalog, Workload({"lid": 1}), self.hardware))
        self.assertIn(Deployment({"whisper": 1}), lid)

    def test_memory_boundary_and_reserve(self):
        memory = self.catalog["qwen"].memory_mib
        self.assertEqual(list(generate_candidates(self.catalog, self.workload, Hardware(memory))),
                         [Deployment({"qwen": 1})])
        self.assertEqual(list(generate_candidates(self.catalog, self.workload, Hardware(memory, 1))), [])

    def test_workload_changes_selected_deployment(self):
        for fixture, expected in (("mixed", "whisper"), ("text_heavy", "distilbert")):
            workload = Workload(json.loads((EXAMPLES / f"workload_{fixture}.json").read_text()))
            result = optimize(self.catalog, workload, self.hardware, self.backend)
            self.assertEqual(result.status, "ok")
            self.assertEqual(result.best.deployment, Deployment({"qwen": 1, expected: 1}))
            self.assertEqual(result.best.performance.mean_latency_ms, 120)
            self.assertEqual(len(result.evaluations), 16)

    def test_table_key_normalizes_zero_tasks_and_model_order(self):
        backend = TablePerformanceModel(self.backend.context, [{
            "model_counts": {"whisper": 1, "qwen": 1},
            "arrival_rates": {"text": 1},
            "performance": {"mean_latency_ms": 2, "p95_latency_ms": 3, "throughput_rps": 1},
        }])
        self.assertEqual(backend.predict(Deployment({"qwen": 1, "whisper": 1}),
                                        Workload({"text": 1, "asr": 0})).mean_latency_ms, 2)
        with self.assertRaises(MissingPerformanceError):
            backend.predict(Deployment({"qwen": 1, "whisper": 1}), Workload({"text": 2}))

    def test_unseen_load_is_not_interpolated(self):
        result = optimize(self.catalog, Workload({"text": 100, "visual_qa": 1}),
                          self.hardware, self.backend)
        self.assertEqual(result.status, "incomplete_table")
        self.assertIsNone(result.best)
        self.assertEqual(len(result.missing), 16)

    def test_partial_table_requires_explicit_opt_in(self):
        data = json.loads((EXAMPLES / "performance_table.synthetic.json").read_text())
        backend = TablePerformanceModel(data["context"], [data["rows"][4]])
        strict = optimize(self.catalog, self.workload, self.hardware, backend)
        partial = optimize(self.catalog, self.workload, self.hardware, backend, allow_partial=True)
        self.assertEqual(strict.status, "incomplete_table")
        self.assertIsNone(strict.best)
        self.assertEqual(partial.status, "partial")
        self.assertIsNotNone(partial.best)

    def test_constraints_can_make_all_candidates_infeasible(self):
        for constraints in (Constraints(30), Constraints(0, 100)):
            result = optimize(self.catalog, self.workload, self.hardware, self.backend,
                              constraints=constraints)
            self.assertEqual(result.status, "infeasible")
            self.assertIsNone(result.best)

    def test_optimizer_accepts_independent_backend_and_maximizes_throughput(self):
        class CustomBackend(PerformanceModel):
            def predict(self, deployment, workload):
                fast = "distilbert" in deployment.model_counts
                return PerformanceResult(20 if fast else 5, 30, 10 if fast else 3)

        result = optimize(self.catalog, Workload({"text": 1}), self.hardware, CustomBackend(),
                          objective="throughput")
        self.assertEqual(result.best.deployment, Deployment({"distilbert": 1}))
        self.assertEqual(result.best.performance.throughput_rps, 10)

    def test_idle_workload_requires_no_backend(self):
        result = optimize(self.catalog, Workload({}), Hardware(0), self.backend)
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.best.deployment, Deployment({}))
        self.assertEqual(result.best.performance.completed_requests, 0)

    def test_invalid_input_is_rejected(self):
        for rates in ({"vqa": 1}, {"text": -1}, {"text": float("nan")},
                      {"text": float("inf")}, {"text": True}, {"text": "1"}):
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                Workload(rates)
        for count in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                Deployment({"qwen": count})
        with self.assertRaises(ValueError):
            Hardware(1, 2)
        with self.assertRaises(ValueError):
            PerformanceResult(1, 2, 3, slo_violation_rate=2)
        with self.assertRaises(ValueError):
            PerformanceResult(1, 2, 3, completed_requests=0.5)

    def test_duplicate_measurements_need_explicit_aggregation(self):
        data = json.loads((EXAMPLES / "performance_table.synthetic.json").read_text())
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            TablePerformanceModel(data["context"], [data["rows"][0]] * 2)

    def test_p95_need_not_exceed_mean(self):
        # Rare extreme outliers can put the mean above P95.
        self.assertEqual(PerformanceResult(100, 10, 1).p95_latency_ms, 10)


if __name__ == "__main__":
    unittest.main()
