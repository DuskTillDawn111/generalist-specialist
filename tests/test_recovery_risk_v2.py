"""Synthetic run/group holdout validates contracts, not real risk accuracy."""

import copy
import json
import unittest

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.recovery_risk import (build_features, fit_recovery_risk,
    RecoveryRiskProvider, evaluate_holdout)
from tests.test_batched_forecast_v2 import MS
from tests.test_initial_handoff_forecast_v2 import model, handoff, score


CONTEXT = "recovery-cpu-fixture"
HORIZON = 100 * MS


def features(work, now=10 * MS):
    return build_features(now_ns=now, execution_context_sha256=CONTEXT, queue_work_ms=work,
        oldest_waiting_ms=work / 2, hol_blocked_ms=work / 10, waiting_requests=2,
        visible_state_time_ns=now, declared_at_ns=0, future_source="sampled_declared_workload_scenario",
        future_work_envelopes=[{"window_ns": 10 * MS, "max_offered_solo_work_ms": 5.},
                               {"window_ns": 100 * MS, "max_offered_solo_work_ms": 20.}])


def observation(index, work, split="training"):
    now = 10 * MS
    return {"id": split + str(index), "run_id": split + "-run-" + str(index),
        "group_id": split + "-group-" + str(index % 2), "split": split,
        "kind": "cpu_fixture", "execution_context_sha256": CONTEXT, "decision_time_ns": now,
        "features": features(work, now), "outcome": {"window_start_ns": now, "window_end_ns": now + HORIZON,
            "observed_until_ns": now + HORIZON, "violation": work > 5., "recovery_ms": 60. if work > 5 else 10.}}


def training():
    return [observation(index, work) for index, work in enumerate((1., 2., 3., 4., 6., 7., 8., 9.))]


def provider():
    document = fit_recovery_risk(training(), CONTEXT, horizon_ns=HORIZON, recovery_threshold_ms=50.)
    return RecoveryRiskProvider(document, CONTEXT, allow_cpu_fixture=True)


class RecoveryRiskTests(unittest.TestCase):
    def test_frozen_joint_contract_and_run_group_holdout(self):
        risk = provider()
        before = json.dumps(risk.model, sort_keys=True)
        report = evaluate_holdout(risk, [observation(20, 2.5, "validation"), observation(21, 7.5, "validation")])
        self.assertEqual(report["count"], 2)
        self.assertEqual(report["rank_auc"], 1.)  # Analytic separable fixture only.
        self.assertFalse(report["physical_risk_accuracy_established"])
        self.assertFalse(report["training_refit"])
        self.assertFalse(report["calibrated_probability"])
        self.assertIsNone(report["confidence_interval"])
        self.assertEqual(json.dumps(risk.model, sort_keys=True), before)
        self.assertLess(risk.predict(features(2.5), 10 * MS)["risk_score"],
                        risk.predict(features(7.5), 10 * MS)["risk_score"])
        loaded = RecoveryRiskProvider(json.loads(before), CONTEXT, allow_cpu_fixture=True)
        self.assertEqual(loaded.predict(features(2.5), 10 * MS), risk.predict(features(2.5), 10 * MS))

    def test_short_violation_and_long_recovery_both_form_offline_target(self):
        rows = training()
        for row in rows:
            row["outcome"]["violation"] = False
        fitted = fit_recovery_risk(rows, CONTEXT, horizon_ns=HORIZON, recovery_threshold_ms=50.)
        self.assertEqual(fitted["training_class_counts"], {"0": 4, "1": 4})

    def test_validation_cannot_enter_fit_and_both_label_classes_need_support(self):
        with self.assertRaises(ValueError):
            fit_recovery_risk([*training(), observation(30, 5., "validation")], CONTEXT,
                              horizon_ns=HORIZON, recovery_threshold_ms=50.)
        rows = training()
        for row in rows:
            row["outcome"].update(violation=False, recovery_ms=10.)
        with self.assertRaisesRegex(UnknownEvidence, "BOTH_RISK_LABEL_CLASSES"):
            fit_recovery_risk(rows, CONTEXT, horizon_ns=HORIZON, recovery_threshold_ms=50.)

    def test_actual_future_or_label_fields_never_enter_online_predict(self):
        risk = provider()
        for mutate in (
            lambda f: f.update(outcome={"violation": True}),
            lambda f: f.update(future_source="actual_validation_arrivals"),
            lambda f: f.update(visible_state_time_ns=11 * MS),
            lambda f: f.update(declared_at_ns=11 * MS),
            lambda f: f.update(observed_at_ns=11 * MS),
            lambda f: f.update(clock="unix_ns"),
            lambda f: f.update(execution_context_sha256="other"),
        ):
            value = features(2.)
            mutate(value)
            with self.assertRaises(ValueError):
                risk.predict(value, 10 * MS)

    def test_missing_feature_new_scale_and_outside_training_support_are_unknown(self):
        risk = provider()
        for mutate in (lambda f: f.update(queue_work_ms=None),
                       lambda f: f.update(queue_work_ms=100.),
                       lambda f: f["future_work_envelopes"][0].update(window_ns=5 * MS),
                       lambda f: f["future_work_envelopes"][0].update(max_offered_solo_work_ms=None)):
            value = features(2.)
            mutate(value)
            with self.assertRaises(UnknownEvidence):
                risk.predict(value, 10 * MS)

    def test_holdout_requires_disjoint_runs_groups_and_ids(self):
        risk = provider()
        for key, source in (("run_id", "run_ids"), ("group_id", "group_ids"), ("id", "ids")):
            rows = [observation(20, 2., "validation"), observation(21, 7., "validation")]
            rows[0][key] = risk.model["provenance"][source][0]
            with self.assertRaises(ValueError):
                evaluate_holdout(risk, rows)
        with self.assertRaisesRegex(UnknownEvidence, "BOTH_HOLDOUT_CLASSES"):
            evaluate_holdout(risk, [observation(20, 2., "validation")])

    def test_censored_future_labels_and_single_run_training_are_rejected(self):
        rows = training()
        rows[0]["outcome"]["observed_until_ns"] = 11 * MS
        with self.assertRaises(ValueError):
            fit_recovery_risk(rows, CONTEXT, horizon_ns=HORIZON, recovery_threshold_ms=50.)
        rows = training()
        for row in rows:
            row["run_id"] = "same-run"
        with self.assertRaisesRegex(UnknownEvidence, "INDEPENDENT_TRAINING_RUNS_GROUPS_REQUIRED"):
            fit_recovery_risk(rows, CONTEXT, horizon_ns=HORIZON, recovery_threshold_ms=50.)

    def test_cpu_evidence_context_and_frozen_model_mutation_are_rejected(self):
        document = provider().model
        with self.assertRaises(ValueError):
            RecoveryRiskProvider(document, CONTEXT)
        with self.assertRaises(ValueError):
            RecoveryRiskProvider(document, "other", allow_cpu_fixture=True)
        risk = provider()
        risk.model["base_score"] += 1.
        with self.assertRaises(ValueError):
            risk.predict(features(2.), 10 * MS)

    def test_risk_prediction_does_not_change_six_term_score_or_original_rho(self):
        forecast = model().forecast({}, 101 * MS, 150 * MS,
            admission_state={"active": {"old": "s"}, "waiting": []},
            handoff_state=[handoff(101, received=False, ready=False)])
        before, forecast_before = score(forecast, now=101), copy.deepcopy(forecast)
        result = provider().predict(features(2.), 10 * MS)
        self.assertFalse(result["stability_rule_changed"])
        self.assertFalse(result["six_term_utility_changed"])
        self.assertFalse(result["reject_requests"])
        self.assertEqual(score(forecast, now=101), before)
        self.assertEqual(forecast, forecast_before)


if __name__ == "__main__":
    unittest.main()
