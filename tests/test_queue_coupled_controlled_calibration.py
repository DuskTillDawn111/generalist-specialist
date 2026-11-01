"""Controlled calibration identities, intervals and CPU-only worker lifecycle."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.queue_coupled_utility_20261023 import controlled_calibration as cc


class ControlledCalibration(unittest.TestCase):
    def test_schedule_covers_each_input_context_once_per_seed(self):
        pools = [f"t{i:02}" for i in range(32)]
        rows = cc.schedule(pools)
        self.assertEqual(len(rows), 320)
        self.assertEqual(sum(len(row["aliases"]) for row in rows), 576)
        self.assertEqual(len({row["id"] for row in rows}), 320)
        self.assertEqual(rows, cc.schedule(list(reversed(pools))))
        for seed in cc.SEEDS:
            selected = [row for row in rows if row["seed"] == seed]
            self.assertEqual({(row["context"], row["pool_id"]) for row in selected},
                             {(context, pool) for context in cc.CONTEXTS for pool in pools})
            self.assertTrue(all(set(row["aliases"]) == set(cc.CONTEXTS[row["context"]]) for row in selected))
        self.assertNotEqual([(r["context"], r["pool_id"]) for r in rows[:160]],
                            [(r["context"], r["pool_id"]) for r in rows[160:]])

    def test_incomplete_input_catalog_rejected(self):
        for pools in (["x"] * 32, [str(i) for i in range(31)]):
            with self.assertRaisesRegex(ValueError, "32 distinct"):
                cc.schedule(pools)

    def test_actual_interval_uses_worker_end_and_service_and_host_clock_anchor(self):
        row = {"completion_time_ns": 500_000_000, "completion_unix_ns": 2_000_000_000,
               "worker_response": {"end_ns": 1_990_000_000, "start_ns": 1_900_000_000,
                                   "service_ms": 50.}}
        value = cc.interval(row)
        self.assertEqual(value["start_time_ns"], 440_000_000)
        self.assertEqual(value["end_time_ns"], 490_000_000)
        self.assertEqual(value["start_unix_ns"], 1_940_000_000)
        self.assertIn("not observed GPU kernel", value["scope"])
        for bad in (None, -1., float("nan"), True):
            changed = copy.deepcopy(row); changed["worker_response"]["service_ms"] = bad
            self.assertIsNone(cc.interval(changed))

    def test_execute_requires_explicit_flag_before_any_worker_or_output(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(cc, "validate_packet", side_effect=AssertionError("must not validate/launch")):
            output = Path(temp) / "output"
            with self.assertRaisesRegex(ValueError, "Explicit"):
                cc.execute({}, output)
            self.assertFalse(output.exists())

    def _fixture(self, path, fail=False):
        asset = path / "asset"; asset.write_text("cpu")
        samples = path / "pool.jsonl"
        rows = [{"pool_id": f"t{i:02}", "task_type": "text", "sample": {"context": "c", "question": str(i), "label": "ok", "mock_error": fail}}
                for i in range(32)]
        rows += [{"pool_id": t, "task_type": t, "sample": {"input": str(asset), "label": "ok"}}
                 for t in ("image", "asr", "ocr")]
        samples.write_text("".join(json.dumps(row)+"\n" for row in rows))
        models = {"text": {"supported_tasks": ["text"], "service_time_ms": 1.},
                  "generalist_2": {"supported_tasks": ["text", "image", "asr", "ocr"], "service_time_ms": 1.},
                  "ocr": {"supported_tasks": ["ocr"], "service_time_ms": 1.}}
        system = SimpleNamespace(workload={"text": 1.85, "image": .01, "asr": .01, "ocr": .01},
            catalog={"models": models, "safe_factor": 1.1}, gpu_memory_mib=1000.)
        aliases = {f"{family}#{i}": {"family": family, "supported_tasks": models[family]["supported_tasks"]}
                   for family, count in cc.PRIMARY.items() for i in range(count)}
        contract = {"strategy": "queue_quality", "text_rate": 1.85, "aliases": aliases,
                    "deployment": cc.PRIMARY,
                    "background_routing": {"image": {"generalist_2": .01}, "asr": {"generalist_2": .01}, "ocr": {"ocr": .01}}}
        return system, samples, contract

    def test_real_cpu_workers_batch_isolation_and_no_fabricated_resources(self):
        with tempfile.TemporaryDirectory() as temp:
            system, samples, contract = self._fixture(Path(temp))
            batches = cc.schedule([f"t{i:02}" for i in range(32)])[:12]
            events = []
            with patch("subprocess.run", side_effect=AssertionError("No GPU/system query in mock")):
                raw = cc.collect(system, samples, contract, batches, backend="mock", idle_seconds=0,
                                 startup_timeout=10., drain_timeout=10., progress=events.append)
            self.assertEqual(raw["status"], "completed", raw["errors"])
            self.assertEqual(len(raw["batches"]), len(batches))
            self.assertEqual(len(raw["request_records"]), sum(len(b["aliases"]) for b in batches))
            self.assertEqual(events[-1]["completed_batches"], len(batches))
            self.assertEqual(events[-1]["completed_requests"], len(raw["request_records"]))
            self.assertGreaterEqual(events[-1]["elapsed_seconds"], 0)
            self.assertTrue(all(b["all_completed"] for b in raw["batches"]))
            by_id = {r["request_id"]: r for r in raw["request_records"]}
            for previous, current in zip(raw["batches"], raw["batches"][1:]):
                self.assertGreaterEqual(current["start_time_ns"], previous["end_time_ns"])
                self.assertLessEqual(max(by_id[rid]["completion_time_ns"] for rid in previous["request_ids"]), current["start_time_ns"])
            self.assertTrue(all(not v["forced"] and v["exit_code"] == 0 and v["bye_received"] for v in raw["shutdown"].values()))
            self.assertFalse(raw["calibration_usable"])
            self.assertFalse(raw["energy_evidence"]["coverage_valid"])
            self.assertIsNone(raw["energy_evidence"]["run_joules"])
            self.assertIsNone(raw["memory_evidence"]["safe_memory_mib"])
            self.assertIsNone(raw["U"])
            self.assertFalse(raw["original_stability_evaluated"])
            for worker in raw["worker_registry"].values():
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    self.assertEqual(worker["env"][key], "1")

    def test_priming_failure_keeps_raw_and_shutdown_without_batches(self):
        with tempfile.TemporaryDirectory() as temp:
            system, samples, contract = self._fixture(Path(temp), fail=True)
            with patch("subprocess.run", side_effect=AssertionError("No GPU query")):
                raw = cc.collect(system, samples, contract, [], backend="mock", idle_seconds=0,
                                 startup_timeout=10., drain_timeout=10.)
            self.assertEqual(raw["status"], "failed")
            self.assertIn("Controlled priming failed", " ".join(raw["errors"]))
            self.assertTrue(raw["priming_request_records"])
            self.assertFalse(raw["batches"])
            self.assertTrue(raw["shutdown"])
            self.assertFalse(raw["calibration_usable"])


if __name__ == "__main__":
    unittest.main()
