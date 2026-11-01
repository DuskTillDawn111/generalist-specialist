"""Optional Sinan-inspired delayed-backlog risk, separate from queue execution.

Small logistic gradient-boosted decision stumps use numpy only. The output is
an uncalibrated model score, never a probability guarantee or admission rule.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from ..forecast_errors import UnknownEvidence
from ..resources import identity
from .workload import number

FEATURE_SCHEMA = "gsdeploy.visible_recovery_risk_features.v2"
MODEL_SCHEMA = "gsdeploy.frozen_recovery_risk.v2"
PUBLIC_SOURCES = {"declared_schedule", "sampled_declared_workload_scenario"}


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit " + name + " required")
    return value


def _ns(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer " + name + " required")
    return value


def build_features(*, now_ns, execution_context_sha256, queue_work_ms, oldest_waiting_ms,
                   hol_blocked_ms, waiting_requests, visible_state_time_ns, declared_at_ns,
                   future_source, future_work_envelopes):
    """Accept current visible diagnostics and declared future work only.

    Do not pass the full-horizon simulator's future maxima as current queue
    features. Future envelopes describe the public planning scenario, not a
    replay of actual held-out arrivals or outcomes.
    """
    features = {"schema": FEATURE_SCHEMA, "execution_context_sha256": execution_context_sha256,
        "clock": "host_perf_counter_ns", "observed_at_ns": now_ns,
        "visible_state_time_ns": visible_state_time_ns, "declared_at_ns": declared_at_ns,
        "future_source": future_source, "queue_work_ms": queue_work_ms,
        "oldest_waiting_ms": oldest_waiting_ms, "hol_blocked_ms": hol_blocked_ms,
        "waiting_requests": waiting_requests, "future_work_envelopes": copy.deepcopy(future_work_envelopes)}
    _vector(features, now_ns, execution_context_sha256)
    return features


def _vector(features, now_ns, context):
    fields = {"schema", "execution_context_sha256", "clock", "observed_at_ns", "visible_state_time_ns",
              "declared_at_ns", "future_source", "queue_work_ms", "oldest_waiting_ms", "hol_blocked_ms",
              "waiting_requests", "future_work_envelopes"}
    _ns(now_ns, "risk decision time")
    _text(context, "execution identity")
    if (not isinstance(features, dict) or set(features) != fields or features["schema"] != FEATURE_SCHEMA
            or features["execution_context_sha256"] != context or features["clock"] != "host_perf_counter_ns"):
        raise ValueError("Strict public risk feature/clock/execution contract required; labels are forbidden")
    observed = _ns(features["observed_at_ns"], "feature observation time")
    if (observed != now_ns or _ns(features["visible_state_time_ns"], "visible state time") > observed
            or _ns(features["declared_at_ns"], "workload declaration time") > observed):
        raise ValueError("Future or stale risk features are not current visible state")
    if features["future_source"] not in PUBLIC_SOURCES:
        raise ValueError("Actual future workload cannot be a risk feature")
    names = ["queue_work_ms", "oldest_waiting_ms", "hol_blocked_ms", "waiting_requests"]
    if type(features["waiting_requests"]) is not int or features["waiting_requests"] < 0:
        raise ValueError("Nonnegative integer visible waiting count required")
    values = []
    for name in names:
        if features[name] is None:
            raise UnknownEvidence("UNKNOWN_VISIBLE_RECOVERY_RISK_FEATURE:" + name)
        values.append(number(features[name], name))
    envelopes = features["future_work_envelopes"]
    if not isinstance(envelopes, list) or not envelopes:
        raise UnknownEvidence("DECLARED_MULTISCALE_WORK_ENVELOPES_REQUIRED")
    previous = 0
    for envelope in envelopes:
        if set(envelope) != {"window_ns", "max_offered_solo_work_ms"}:
            raise ValueError("Only declared scale/work envelope fields accepted")
        scale = _ns(envelope["window_ns"], "workload envelope window")
        if scale <= previous:
            raise ValueError("Distinct increasing workload-envelope scales required")
        work = envelope["max_offered_solo_work_ms"]
        if work is None:
            raise UnknownEvidence("UNKNOWN_DECLARED_WORK_ENVELOPE")
        names.append("declared_work_ms@" + str(scale))
        values.append(number(work, "declared offered solo work"))
        previous = scale
    return names, np.asarray(values, dtype=float)


def _label(row, context, split, horizon_ns, recovery_threshold_ms):
    required = {"id", "run_id", "group_id", "split", "kind", "execution_context_sha256",
                "decision_time_ns", "features", "outcome"}
    if (set(row) != required or row["split"] != split or row["kind"] not in {"measured", "cpu_fixture"}
            or row["execution_context_sha256"] != context):
        raise ValueError("Explicit split/context-bound offline risk row required")
    for name in ("id", "run_id", "group_id"):
        _text(row[name], name)
    now = _ns(row["decision_time_ns"], "training decision time")
    names, vector = _vector(row["features"], now, context)
    outcome = row["outcome"]
    if (set(outcome) != {"window_start_ns", "window_end_ns", "observed_until_ns", "violation", "recovery_ms"}
            or _ns(outcome["window_start_ns"], "offline label window start") != now
            or _ns(outcome["window_end_ns"], "offline label window end") != now + horizon_ns
            or _ns(outcome["observed_until_ns"], "offline label observation end") < now + horizon_ns
            or type(outcome["violation"]) is not bool):
        raise ValueError("Complete prospectively defined future-window outcomes required offline")
    recovery = number(outcome["recovery_ms"], "observed recovery duration")
    if now + round(recovery * 1e6) > outcome["observed_until_ns"]:
        raise ValueError("Recovery outcome is censored at observation end")
    return names, vector, int(outcome["violation"] or recovery > recovery_threshold_ms)


def _sigmoid(value):
    return 1. / (1. + np.exp(-np.clip(value, -50., 50.)))


def fit_recovery_risk(training_rows, execution_context_sha256, *, horizon_ns,
                      recovery_threshold_ms, n_estimators=40, learning_rate=.2, minimum_leaf=2):
    """Freeze a JSON model from training runs/groups; no validation row allowed."""
    _text(execution_context_sha256, "execution context")
    if _ns(horizon_ns, "risk horizon") == 0:
        raise ValueError("Positive risk horizon required")
    number(recovery_threshold_ms, "recovery threshold", positive=True)
    if (type(n_estimators) is not int or not 1 <= n_estimators <= 200
            or type(minimum_leaf) is not int or minimum_leaf < 2
            or not 0 < number(learning_rate, "boosting rate", positive=True) <= 1):
        raise ValueError("Bounded boosting configuration and at least two samples per leaf required")
    rows = list(training_rows)
    parsed = [_label(row, execution_context_sha256, "training", horizon_ns, recovery_threshold_ms) for row in rows]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Nonempty unique training risk observations required")
    names = parsed[0][0]
    if any(p[0] != names for p in parsed):
        raise ValueError("Training feature scales differ")
    kinds = {row["kind"] for row in rows}
    runs, groups = {row["run_id"] for row in rows}, {row["group_id"] for row in rows}
    if len(kinds) != 1 or len(runs) < 3 or len(groups) < 2:
        raise UnknownEvidence("INDEPENDENT_TRAINING_RUNS_GROUPS_REQUIRED")
    X, y = np.stack([p[1] for p in parsed]), np.asarray([p[2] for p in parsed], dtype=float)
    if min(np.sum(y == 0), np.sum(y == 1)) < 2:
        raise UnknownEvidence("BOTH_RISK_LABEL_CLASSES_REQUIRE_SUPPORT")
    base = math.log(float(np.mean(y)) / (1. - float(np.mean(y))))
    scores, trees = np.full(len(y), base), []
    for _ in range(n_estimators):
        residual, best = y - _sigmoid(scores), None
        for feature in range(X.shape[1]):
            unique = np.unique(X[:, feature])
            if len(unique) > 33:
                unique = unique[np.unique(np.linspace(0, len(unique) - 1, 33).astype(int))]
            for left, right in zip(unique, unique[1:]):
                threshold = float(left + (right - left) / 2.)
                mask = X[:, feature] <= threshold
                if min(np.sum(mask), np.sum(~mask)) < minimum_leaf:
                    continue
                low, high = float(np.mean(residual[mask])), float(np.mean(residual[~mask]))
                prediction = np.where(mask, low, high)
                loss = float(np.sum((residual - prediction) ** 2))
                if best is None or loss < best[0]:
                    best = (loss, feature, threshold, low, high, prediction)
        if best is None:
            raise UnknownEvidence("NO_SUPPORTED_RECOVERY_RISK_TREE_SPLIT")
        _, feature, threshold, low, high, prediction = best
        trees.append({"feature": feature, "threshold": threshold,
                      "left": learning_rate * low, "right": learning_rate * high})
        scores += learning_rate * prediction
    model = {"schema": MODEL_SCHEMA, "execution_context_sha256": execution_context_sha256,
        "provenance": {"kind": next(iter(kinds)), "split": "training", "ids": sorted(row["id"] for row in rows),
                       "run_ids": sorted(runs), "group_ids": sorted(groups)},
        "target": {"horizon_ns": horizon_ns, "recovery_threshold_ms": recovery_threshold_ms},
        "feature_names": names, "support_min": np.min(X, axis=0).tolist(), "support_max": np.max(X, axis=0).tolist(),
        "base_score": base, "trees": trees, "algorithm": "logistic_gradient_boosted_stumps_numpy_v1",
        "training_class_counts": {"0": int(np.sum(y == 0)), "1": int(np.sum(y == 1))}}
    model["model_sha256"] = identity(model)
    return model


class RecoveryRiskProvider:
    def __init__(self, model, execution_context_sha256, *, allow_cpu_fixture=False):
        if (model.get("schema") != MODEL_SCHEMA or model.get("execution_context_sha256") != execution_context_sha256
                or model.get("model_sha256") != identity({k: v for k, v in model.items() if k != "model_sha256"})
                or model.get("provenance", {}).get("split") != "training"):
            raise ValueError("Frozen risk model identity/context/training provenance mismatch")
        if model["provenance"]["kind"] not in {"measured", "cpu_fixture"}:
            raise ValueError("Unknown risk evidence kind")
        if model["provenance"]["kind"] == "cpu_fixture" and not allow_cpu_fixture:
            raise ValueError("CPU fixture is not physical recovery-risk evidence")
        if set(model["target"]) != {"horizon_ns", "recovery_threshold_ms"} or not _ns(model["target"]["horizon_ns"], "model risk horizon"):
            raise ValueError("Explicit positive risk target horizon required")
        number(model["target"]["recovery_threshold_ms"], "model recovery threshold", positive=True)
        dimension = len(model["feature_names"])
        if (not dimension or len(set(model["feature_names"])) != dimension
                or len(model["support_min"]) != dimension or len(model["support_max"]) != dimension
                or not model["trees"] or model["algorithm"] != "logistic_gradient_boosted_stumps_numpy_v1"
                or any(not math.isfinite(float(v)) for v in [model["base_score"], *model["support_min"], *model["support_max"]])
                or any(lo > hi for lo, hi in zip(model["support_min"], model["support_max"]))):
            raise ValueError("Malformed frozen risk ensemble/support")
        for tree in model["trees"]:
            if (set(tree) != {"feature", "threshold", "left", "right"} or type(tree["feature"]) is not int
                    or not 0 <= tree["feature"] < dimension
                    or any(type(tree[k]) not in (int, float) or not math.isfinite(tree[k]) for k in ("threshold", "left", "right"))):
                raise ValueError("Malformed frozen risk tree")
        self.model = copy.deepcopy(model)
        self.execution_context_sha256 = execution_context_sha256
        self.physical_evidence = model["provenance"]["kind"] == "measured"

    def predict(self, features, now_ns):
        names, vector = _vector(features, now_ns, self.execution_context_sha256)
        model = self.model
        if model["model_sha256"] != identity({k: v for k, v in model.items() if k != "model_sha256"}):
            raise ValueError("Frozen risk model was mutated after construction")
        if names != model["feature_names"]:
            raise UnknownEvidence("RECOVERY_RISK_FEATURE_SCALES_UNSUPPORTED")
        if np.any(vector < model["support_min"]) or np.any(vector > model["support_max"]):
            raise UnknownEvidence("RECOVERY_RISK_OUTSIDE_TRAINING_FEATURE_SUPPORT")
        score = model["base_score"]
        for tree in model["trees"]:
            score += tree["left"] if vector[tree["feature"]] <= tree["threshold"] else tree["right"]
        return {"known": True, "risk_score": float(_sigmoid(score)), "decision_time_ns": now_ns,
            "target": copy.deepcopy(model["target"]), "model_sha256": model["model_sha256"],
            "calibrated_probability": False, "confidence_interval": None,
            "scope": "optional delayed violation or slow recovery model score",
            "stability_rule_changed": False, "six_term_utility_changed": False, "reject_requests": False}


def evaluate_holdout(provider, rows):
    """Run/group-disjoint evaluation only; this function cannot refit a model."""
    rows = list(rows)
    model, predictions, labels, ids = provider.model, [], [], set()
    provenance, target = model["provenance"], model["target"]
    if not rows:
        raise ValueError("Nonempty held-out risk rows required")
    for row in rows:
        _, _, label = _label(row, provider.execution_context_sha256, "validation", **target)
        if (row["id"] in ids or row["id"] in provenance["ids"] or row["run_id"] in provenance["run_ids"]
                or row["group_id"] in provenance["group_ids"]):
            raise ValueError("Holdout observations/runs/groups overlap training")
        if row["kind"] != provenance["kind"]:
            raise ValueError("Held-out evidence kind differs from training")
        ids.add(row["id"])
        labels.append(label)
        predictions.append(provider.predict(row["features"], row["decision_time_ns"])["risk_score"])
    if set(labels) != {0, 1}:
        raise UnknownEvidence("BOTH_HOLDOUT_CLASSES_REQUIRED")
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    binary = predictions >= .5
    positives, negatives = predictions[labels == 1], predictions[labels == 0]
    auc = float(np.mean((positives[:, None] > negatives[None, :]) + .5 * (positives[:, None] == negatives[None, :])))
    return {"split": "validation", "count": len(rows), "run_ids": sorted({r["run_id"] for r in rows}),
        "group_ids": sorted({r["group_id"] for r in rows}), "model_sha256": model["model_sha256"],
        "accuracy_at_score_half": float(np.mean(binary == labels)),
        "balanced_accuracy_at_score_half": float(np.mean([np.mean(binary[labels == c] == c) for c in (0, 1)])),
        "rank_auc": auc, "calibrated_probability": False, "confidence_interval": None,
        "training_refit": False, "physical_risk_accuracy_established": False}
