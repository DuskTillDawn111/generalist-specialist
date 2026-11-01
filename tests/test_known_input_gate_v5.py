"""Prospectively fixed CPU gate fixtures; never claims GPU validation PASS."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from tools.predictor_validation_20261027.known_input_gate_v5 import (freeze_gate_protocol, seal_predictions,
    evaluate_prediction_gate, DATA_SCHEMA, OBSERVED_SCHEMA, main)


class KnownInputGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.model = self.put("model.json", {"frozen": "model-input-only"})
        self.source = self.put("source.json", {"code": "cpu-fixture"})
        self.training = self.put("training.json", {"schema": DATA_SCHEMA, "split": "training", "records": [
            {"run_id": "train-run", "request_id": "train-id", "pool_id": "heldout-one", "task": "text"},
            {"run_id": "train-run", "request_id": "train-id2", "pool_id": "heldout-two", "task": "text"}]})
        self.catalog=self.base/'public.jsonl'
        self.catalog.write_text('\n'.join(json.dumps(dict(pool_id=pool,task_type='text',
            sample=dict(context=pool,question='public'))) for pool in ('heldout-one','heldout-two')))
        self.domain = [{"run_id": "validation-run", "requests": [
            {"request_id": "one", "pool_id": "heldout-one", "task": "text"},
            {"request_id": "two", "pool_id": "heldout-two", "task": "text"}]}]
        self.candidates = [{"id": cid, "model_id": "frozen", "execution_context_sha256": "execution-" + cid}
                           for cid in ("base", "low", "high")]
        self.thresholds = {"service_mare_max": .2, "queue_mae_ms_max": 1., "e2e_mare_max": .2,
                           "p95_mare_max": .2, "delta_sign_accuracy_min": 1., "pair_rank_accuracy_min": 1.}

    def put(self, name, value):
        path = self.base / name
        path.write_text(json.dumps(value))
        return path

    def protocol(self, **changes):
        values = dict(model_paths={"frozen": self.model}, source_paths=[self.source,self.catalog],public_input_catalog_path=self.catalog, training_data_paths=[self.training],
            candidates=self.candidates, baseline_candidate_id="base", validation_domain=self.domain,
            thresholds=self.thresholds, created_unix_ns=100, weights={key: 1 / 6 for key in "QTLPME"})
        values.update(changes)
        return freeze_gate_protocol(**values)

    def cells(self):
        cells = []
        for cid, cost in (("base", .6), ("low", .4), ("high", .2)):
            cells.append({"run_id": "validation-run", "candidate_id": cid,
                "execution_context_sha256": "execution-" + cid,
                "requests": [{"request_id": "one", "pool_id": "heldout-one", "task": "text",
                              "service_ms": 10., "queue_ms": 2., "e2e_ms": 15.},
                             {"request_id": "two", "pool_id": "heldout-two", "task": "text",
                              "service_ms": 20., "queue_ms": 4., "e2e_ms": 25.}],
                "terms": {key: cost for key in "QTLPME"}})
        return cells

    def observed(self, cells=None, **changes):
        document = {"schema": OBSERVED_SCHEMA, "split": "validation", "kind": "cpu_fixture",
            "run_id": "validation-run", "collection_started_unix_ns": 300, "cells": self.cells() if cells is None else cells}
        document.update(changes)
        return self.put("observed.json", document)

    def test_complete_gate_measures_service_queue_e2e_p95_delta_and_all_pair_ranks(self):
        protocol = self.protocol()
        predictions = seal_predictions(protocol, self.cells(), created_unix_ns=200)
        report = evaluate_prediction_gate(protocol, predictions, [self.observed()])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["evidence_kind"], "cpu_fixture")
        self.assertTrue(report["coverage_complete"])
        self.assertEqual(report["delta_sign"], {"expected": 2, "known": 2, "accuracy": 1.})
        self.assertEqual(report["pair_ranking"], {"expected": 3, "known": 3, "accuracy": 1.})
        for candidate in report["diagnostics"]:
            self.assertEqual(set(candidate["errors"]), {"service", "queue", "e2e", "p95"})
            self.assertTrue(all(metric["mae"] == 0. for metric in candidate["errors"].values()))
        self.assertFalse(report["physical_gain_certified"])
        self.assertFalse(report["final_four_groups_enabled"])
        self.assertFalse(report["legacy_v4_artifacts_modified"])

    def test_command_evaluates_sealed_artifacts_without_overwriting_report(self):
        protocol = self.protocol()
        protocol_path = self.put("protocol.json", protocol)
        prediction_path = self.put("predictions.json", seal_predictions(protocol, self.cells(), created_unix_ns=200))
        actual = self.observed()
        output = self.base / "report.json"
        arguments = ["--protocol", str(protocol_path), "--predictions", str(prediction_path),
                     "--observed", str(actual), "--output", str(output)]
        self.assertEqual(main(arguments), 0)
        report = json.loads(output.read_text())
        self.assertFalse(report["GPU_started"])
        self.assertEqual(report["status"], "PASS")
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            main(arguments)

    def test_bad_service_or_queue_fails_prelocked_threshold_without_retuning(self):
        protocol = self.protocol()
        cells = self.cells()
        cells[0]["requests"][0]["service_ms"] = 20.
        cells[0]["requests"][0]["queue_ms"] = 7.
        sealed = seal_predictions(protocol, cells, created_unix_ns=200)
        report = evaluate_prediction_gate(protocol, sealed, [self.observed()])
        self.assertEqual(report["status"], "FAIL")
        base = report["diagnostics"][0]["errors"]
        self.assertEqual(base["service"]["mare"], .5)
        self.assertEqual(base["queue"]["mae"], 2.5)
        self.assertFalse(report["thresholds_changed"])
        self.assertEqual(protocol["thresholds"], self.thresholds)

    def test_missing_request_or_candidate_is_unknown_not_removed_from_denominator(self):
        protocol = self.protocol()
        cells = self.cells()[:2]
        cells[0]["requests"].pop()
        with self.assertRaisesRegex(ValueError,'Cartesian'):
            seal_predictions(protocol,cells,created_unix_ns=200)
        cells.append(self.cells()[2])
        report = evaluate_prediction_gate(protocol, seal_predictions(protocol, cells, created_unix_ns=200), [self.observed()])
        self.assertEqual(report["status"], "UNKNOWN")
        self.assertFalse(report["coverage_complete"])
        base = report["diagnostics"][0]["errors"]["service"]
        self.assertEqual((base["expected"], base["known"]), (2, 1))
        self.assertEqual(report["pair_ranking"]["expected"], 3)
        self.assertEqual(report["pair_ranking"]["known"], 3)

    def test_input_run_leakage_foreign_timing_context_and_preseal_observation_rejected(self):
        for change in ({"run_id": "train-run"}, {"requests": [
            {"request_id": "train-id", "pool_id": "heldout-one", "task": "text"}]}):
            domain = copy.deepcopy(self.domain)
            domain[0].update(change)
            with self.assertRaises(ValueError):
                self.protocol(validation_domain=domain)
        protocol = self.protocol()
        cells = self.cells()
        cells[0]["execution_context_sha256"] = "other"
        with self.assertRaises(ValueError):
            seal_predictions(protocol, cells, created_unix_ns=200)
        seal = seal_predictions(protocol, self.cells(), created_unix_ns=200)
        with self.assertRaises(ValueError):
            evaluate_prediction_gate(protocol, seal, [self.observed(collection_started_unix_ns=200)])

    def test_frozen_source_model_data_and_threshold_identities_are_checked(self):
        for path in (self.model, self.source, self.training):
            original = path.read_text()
            protocol = self.protocol()
            seal = seal_predictions(protocol, self.cells(), created_unix_ns=200)
            path.write_text(original + " ")
            with self.assertRaises(ValueError):
                evaluate_prediction_gate(protocol, seal, [self.observed()])
            path.write_text(original)
        protocol = self.protocol()
        seal = seal_predictions(protocol, self.cells(), created_unix_ns=200)
        protocol["thresholds"]["queue_mae_ms_max"] = 100.
        with self.assertRaises(ValueError):
            evaluate_prediction_gate(protocol, seal, [self.observed()])

    def test_p95_uses_full_task_cohort_and_ranking_detects_reversed_candidate(self):
        protocol = self.protocol()
        cells = self.cells()
        cells[0]["requests"][1]["e2e_ms"] = 35.
        cells[2]["terms"] = {key: .8 for key in "QTLPME"}
        report = evaluate_prediction_gate(protocol, seal_predictions(protocol, cells, created_unix_ns=200), [self.observed()])
        self.assertEqual(report["status"], "FAIL")
        # True task P95=24.5; predicted=34, so error=9.5 ms.
        self.assertAlmostEqual(report["diagnostics"][0]["errors"]["p95"]["mae"], 9.5)
        self.assertEqual(report["delta_sign"]["accuracy"], .5)
        self.assertAlmostEqual(report["pair_ranking"]["accuracy"], 1 / 3)

    def test_no_thresholds_only_assess_and_zero_denominators_remain_explicit(self):
        protocol = self.protocol(thresholds={})
        report = evaluate_prediction_gate(protocol, seal_predictions(protocol, self.cells(), created_unix_ns=200), [self.observed()])
        self.assertEqual(report["status"], "ASSESSED_NOT_GATED")
        protocol = self.protocol()
        actual, predicted = self.cells(), self.cells()
        actual[0]["requests"][0]["service_ms"] = 0.
        report = evaluate_prediction_gate(protocol, seal_predictions(protocol, predicted, created_unix_ns=200), [self.observed(actual)])
        self.assertEqual(report["status"], "UNKNOWN")
        error = report["diagnostics"][0]["errors"]["service"]
        self.assertIsNone(error["mare"])
        self.assertEqual(error["undefined_relative_denominators"], 1)

    def test_queue_depth_grid_is_separate_from_queue_delay_and_missing_samples_are_unknown(self):
        protocol = self.protocol(queue_sample_offsets_ns=(0, 10),
            thresholds={"queue_depth_mae_max": 1.})
        actual, predicted = self.cells(), self.cells()
        for cell in actual:
            cell["queue_depth_samples"] = [{"offset_ns": 0, "depth": 1}, {"offset_ns": 10, "depth": 3}]
        for cell in predicted:
            cell["queue_depth_samples"] = [{"offset_ns": 0, "depth": 2}, {"offset_ns": 10, "depth": 4}]
        report = evaluate_prediction_gate(protocol, seal_predictions(protocol, predicted, created_unix_ns=200), [self.observed(actual)])
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["diagnostics"][0]["errors"]["queue_depth"]["mae"], 1.)
        self.assertEqual(report["diagnostics"][0]["errors"]["queue"]["mae"], 0.)
        predicted[0]["queue_depth_samples"].pop()
        incomplete = evaluate_prediction_gate(protocol, seal_predictions(protocol, predicted, created_unix_ns=200), [self.observed(actual)])
        self.assertEqual(incomplete["status"], "UNKNOWN")
        self.assertEqual(incomplete["diagnostics"][0]["errors"]["queue_depth"]["known"], 1)


if __name__ == "__main__":
    unittest.main()
