"""Versioned quality loss for offline routing replay and actual output scoring.

Quality is never an absolute accuracy gate. Missing records remain unknown.
Per-sample prediction reuses observed outputs, and is explicitly not a guarantee
that concurrent or stochastic inference will reproduce them. Aggregated evidence
can only be used when its mode is explicitly ``aggregate_proxy``.
"""

import hashlib
import json
import math
import random

from .routing import Router
from .normalization import quality_loss
from .trace import expand_routes

SCHEMA = "six_term_quality_v1"
METRICS = {"text": "em", "image": "accuracy", "ocr": "cer", "asr": "wer"}


def quality_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _number(value, label, upper=None):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < 0 or (upper is not None and value > upper)):
        raise ValueError(f"Invalid quality {label}")
    return float(value)


def _document(data):
    quality = data.get("quality", {})
    if quality.get("schema_version") != 1:
        raise ValueError("MISSING_QUALITY_SCHEMA")
    if quality.get("mode") not in {"per_sample_replay", "aggregate_proxy"}:
        raise ValueError("MISSING_QUALITY_MODE")
    identity = quality.get("identity", {})
    if not all(identity.get(k) for k in
               ("input_pool_sha256", "reference_sha256", "protocol_sha256", "source")):
        raise ValueError("MISSING_QUALITY_IDENTITY")
    if "references" in quality and quality_digest(quality["references"]) != identity["reference_sha256"]:
        raise ValueError("QUALITY_REFERENCE_FINGERPRINT_MISMATCH")
    return quality


def _path(quality, model, task):
    record = quality["paths"][model][task]
    if (record.get("metric", "").lower() != METRICS[task]
            or record.get("unit") != "fraction" or not record.get("source")):
        raise ValueError(f"MISSING_QUALITY_PATH_IDENTITY:{model}/{task}")
    identity = record.get("checkpoint_identity", {})
    if not record.get("checkpoint_sha256") and not (
            identity.get("kind") and identity.get("sha256") and identity.get("source")):
        raise ValueError(f"MISSING_CHECKPOINT_IDENTITY:{model}/{task}")
    return record


def sample_totals(task, score):
    """Return numerator/denominator; edit errors can exceed reference length."""
    if task in {"text", "image"}:
        correct = score.get("correct", score.get("EM") if task == "text" else None)
        if type(correct) is bool:
            correct = int(correct)
        return 1 - _number(correct, "correct", 1), 1.0
    errors = _number(score.get("errors"), "errors")
    units = _number(score.get("reference_units"), "reference_units")
    if units == 0:
        raise ValueError("EMPTY_QUALITY_REFERENCE")
    return errors, units


def _unknown(exc, **context):
    return None, {"status": "missing", "kind": "unknown", "reason": str(exc), **context}


def _native_task_quality(data, task, x, row, trace, seed=42):
    """Return native task loss and provenance for one frozen routing column.

    ``input_order`` is the source JSONL order, shuffled exactly as SamplePool.
    The task-specific router is unaffected by other task columns. This makes
    the resulting value an exact column cost for the frozen quality predictor.
    """
    try:
        quality = _document(data)
        active = {m: q for m, q in row.items() if q > 0}
        if not active or not all(m in x and x[m] > 0 for m in active):
            raise ValueError("INVALID_QUALITY_ROUTE")
        paths = {m: _path(quality, m, task) for m in active}
        if quality["mode"] == "aggregate_proxy":
            total = math.fsum(active.values())
            losses = {}
            for model, record in paths.items():
                value = _number(record.get("value"), "aggregate value",
                                1 if task in {"text", "image"} else None)
                losses[model] = 1 - value if task in {"text", "image"} else value
            return math.fsum(active[m] / total * losses[m] for m in active), {
                "status": "available", "kind": "aggregate_proxy",
                "scope": "same-distribution route-share prediction; not per-input measurement",
                "identity": quality["identity"], "per_model_loss": losses,
            }
        order = list(quality["input_order"][task])
        if not order or len(set(order)) != len(order):
            raise ValueError("INVALID_QUALITY_INPUT_ORDER")
        random.Random(f"{seed}:samples:{task}").shuffle(order)
        aliases, routes = expand_routes(x, {task: row})
        router = Router(routes)
        numerator = denominator = 0.0
        count = 0
        variability = False
        for _, offered_task in trace:
            if offered_task != task:
                continue
            model = aliases[router.choose(task)]
            sample_id = order[count % len(order)]
            score = paths[model]["samples"][sample_id]
            a, b = sample_totals(task, score)
            numerator += a
            denominator += b
            count += 1
            variation = score.get("variability", {})
            variability |= bool(score.get("variable_output") or paths[model].get("variable_output")
                                or variation.get("deterministic") is False
                                or variation.get("cross_replica_change_observed")
                                or variation.get("n1_output_variants", 1) > 1)
        if not count:
            raise ValueError("MISSING_QUALITY_TASK_OBSERVATIONS")
        return numerator / denominator, {
            "status": "available", "kind": "offline_per_sample_replay",
            "scope": "frozen observed outputs replayed through runtime routing and input order",
            "identity": quality["identity"], "requests": count,
            "unique_input_count": min(count, len(order)),
            "error_numerator": numerator, "reference_denominator": denominator,
            "variable_output_observed": variability,
            "runtime_output_reproduction_verified": False,
        }
    except (KeyError, TypeError, ValueError) as exc:
        return _unknown(exc, task=task)


def task_quality(data, task, x, row, trace, seed=42):
    native, evidence = _native_task_quality(data, task, x, row, trace, seed)
    value = quality_loss(data, task, native)
    return value, {**evidence, "native_loss": native,
                   "normalized_loss": value}


def predict_quality(data, x, routing, rates, trace, seed=42):
    total = math.fsum(rates.values())
    per_task = {}
    for task in sorted(rates):
        loss, evidence = task_quality(data, task, x, routing.get(task, {}), trace, seed)
        per_task[task] = {"loss": loss, **evidence}
    complete = all(v["loss"] is not None for v in per_task.values())
    return (math.fsum(rates[t] / total * p["loss"] for t, p in per_task.items())
            if complete else None), {
                "status": "available" if complete else "missing",
                "kind": data.get("quality", {}).get("mode", "unknown"),
                "per_task": per_task,
                "task_weights": {t: q / total for t, q in rates.items()},
                "scope": "offline prediction; actual output scoring required for measured utility",
            }


def measure_quality(data, raw, rates):
    """Score actual worker predictions against frozen references, with fixed weights.

    Never substitute an aggregate proxy or an offline prediction for observed Q.
    Reference payloads use the existing tasks.score_result protocol; references
    stay entirely in the evaluator and are not added to worker requests here.
    """
    try:
        quality = _document(data)
        if raw.get("preflight", {}).get("sample_pool_sha256") != quality["identity"]["input_pool_sha256"]:
            raise ValueError("QUALITY_INPUT_POOL_MISMATCH")
        references = quality["references"]
        from .tasks import score_result, page_ocr_text
        totals = {task: [0.0, 0.0, 0] for task in rates}
        orders = {task: list(quality["input_order"][task]) for task in rates}
        seed = raw["transport_config"]["random_seed"]
        for task, order in orders.items():
            if not order or len(set(order)) != len(order):
                raise ValueError("INVALID_QUALITY_INPUT_ORDER")
            random.Random(f"{seed}:samples:{task}").shuffle(order)
        for record in raw["request_records"]:
            task, alias = record["task"], record["model"]
            model = raw["replica_to_family"][alias]
            _path(quality, model, task)
            if record.get("status") != "completed":
                raise ValueError("INCOMPLETE_QUALITY_COHORT")
            if record["pool_id"] != orders[task][totals[task][2] % len(orders[task])]:
                raise ValueError("QUALITY_INPUT_ORDER_MISMATCH")
            sample = references[task][record["pool_id"]]
            response = record["worker_response"]
            result = dict(response)
            result["task"] = response.get("task", response.get("task_type"))
            # Success must be reported by the worker; timestamp completion alone
            # cannot turn an invalid model output into a quality observation.
            if result.get("success") is not True:
                raise ValueError("UNVERIFIED_QUALITY_OUTPUT")
            if task == "ocr":
                result["prediction"] = page_ocr_text(result["prediction"])
            a, b = sample_totals(task, score_result(sample, result))
            totals[task][0] += a
            totals[task][1] += b
            totals[task][2] += 1
        if any(v[1] == 0 for v in totals.values()):
            raise ValueError("MISSING_QUALITY_TASK_OBSERVATIONS")
        total = math.fsum(rates.values())
        per_task = {t: {"loss": quality_loss(data, t, a / b), "native_loss": a / b, "error_numerator": a,
                        "reference_denominator": b, "requests": n}
                    for t, (a, b, n) in totals.items()}
        if any(p["loss"] is None for p in per_task.values()):
            raise ValueError("MISSING_QUALITY_NORMALIZATION_REFERENCES")
        return math.fsum(rates[t] / total * p["loss"] for t, p in per_task.items()), {
            "status": "available", "kind": "measured_outputs",
            "identity": quality["identity"], "per_task": per_task,
            "task_weights": {t: q / total for t, q in rates.items()},
        }
    except (KeyError, TypeError, ValueError) as exc:
        return _unknown(exc)
