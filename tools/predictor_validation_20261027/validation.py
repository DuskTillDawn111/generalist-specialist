"""Assess frozen forecasts against independent physical observations.

The manifest references byte-hashed model, protocol, training sources,
predictions and observations. Predictions must be sealed before observations
become available. This tool never fits a model or invokes a GPU. CPU fixtures
exercise report arithmetic but cannot pass physical validation. Hash checks
establish artifact identity, not the authenticity of hardware measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


SCHEMA = "gsdeploy.independent_predictor_validation.v1"
PREDICTION_SCHEMA = "gsdeploy.sealed_predictor_prediction.v1"
OBSERVATION_SCHEMA = "gsdeploy.predictor_validation_observation.v1"
TERMS = set("QTLPME")


def _finite(value, name, *, nonnegative=True):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError("Finite " + ("nonnegative " if nonnegative else "") + name + " required")
    return value


def _stamp(value, name):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer " + name + " required")
    return value


def _identifier(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit " + name + " required")
    return value


def _artifact(ref, root):
    if not isinstance(ref, dict) or set(ref) != {"path", "sha256"}:
        raise ValueError("Artifact path and SHA256 required")
    path = Path(ref["path"])
    path = path if path.is_absolute() else root / path
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != ref["sha256"]:
        raise ValueError("Artifact SHA256 mismatch: " + str(path))
    return json.loads(raw), actual


def percentile(values, q=.95):
    values = sorted(values)
    if not values:
        return None
    pos = (len(values) - 1) * q
    lo, hi = math.floor(pos), math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def errors(pairs):
    """Pairs are (predicted, observed); zero denominators stay undefined."""
    if not pairs:
        return {"count": 0, "mae": None, "bias_predicted_minus_observed": None,
                "absolute_error_p95": None, "relative_error_count": 0, "mean_absolute_relative_error": None}
    residuals = [p - o for p, o in pairs]
    relative = [abs(p - o) / abs(o) for p, o in pairs if o != 0]
    return {"count": len(pairs), "mae": statistics.mean(map(abs, residuals)),
            "bias_predicted_minus_observed": statistics.mean(residuals),
            "absolute_error_p95": percentile(list(map(abs, residuals))),
            "relative_error_count": len(relative),
            "mean_absolute_relative_error": statistics.mean(relative) if relative else None}


def _indexed(rows, key, name):
    if not isinstance(rows, list):
        raise ValueError(name + " must be an explicit list")
    result = {}
    for row in rows:
        identity = row.get(key)
        if type(identity) is not int:
            _identifier(identity, name + " identity")
        if identity in result:
            raise ValueError("Duplicate " + name + " identity")
        result[identity] = row
    return result


def _utility(row, weights):
    terms = row.get("terms")
    if terms is None:
        return None
    if not isinstance(terms, dict) or set(terms) != TERMS:
        raise ValueError("All six utility terms required; missing terms cannot become zero")
    if any(v is None for v in terms.values()):
        return None
    return -sum(weights[key] * _finite(terms[key], "utility term") for key in TERMS)


def assess(manifest, *, artifact_root="."):
    """Return diagnostics and preregistered gates, with UNKNOWN coverage.

    No validation observation, including future arrivals, is supplied to the
    predictor here. Offline trace replay may diagnose conditional service but
    is explicitly excluded from online queue/utility validation.
    """
    if manifest.get("schema") != SCHEMA:
        raise ValueError("Explicit independent validation schema required")
    root = Path(artifact_root)
    model = manifest["model"]
    model_document, model_sha = _artifact(model["artifact"], root)
    model_frozen = _stamp(model["frozen_ns"], "model freeze timestamp")
    protocol_ref = manifest["protocol"]
    protocol, protocol_sha = _artifact(protocol_ref["artifact"], root)
    protocol_frozen = _stamp(protocol_ref["frozen_ns"], "protocol freeze timestamp")
    weights = protocol["weights"]
    if set(weights) != TERMS or abs(sum(_finite(v, "weight") for v in weights.values()) - 1) > 1e-12:
        raise ValueError("Six nonnegative weights summing to one required")
    margin = _finite(protocol.get("utility_tie_tolerance", 0.), "utility tie tolerance")
    required = protocol.get("required_categories", ["service", "slowdown", "queue", "p95", "delta_U", "ranking"])
    if not required or set(required) - {"service", "slowdown", "queue", "p95", "delta_U", "ranking"}:
        raise ValueError("Explicit recognized validation categories required")
    plan = _indexed(protocol["planned_cases"], "id", "planned validation case")
    if not plan:
        raise ValueError("Freeze a nonempty validation case plan before observation")
    for planned in plan.values():
        categories = planned["categories"]
        if not categories or set(categories) - set(required) or len(set(categories)) != len(categories):
            raise ValueError("Planned case categories must be unique protocol categories")
        tasks = planned["expected_tasks"]
        if not tasks or len(set(tasks)) != len(tasks) or any(not isinstance(t, str) or not t for t in tasks):
            raise ValueError("Freeze explicit unique task coverage for every planned case")
        if {"ranking", "delta_U"} & set(categories):
            candidates = planned["expected_candidate_ids"]
            if len(candidates) < 2 or len(set(candidates)) != len(candidates):
                raise ValueError("Freeze at least two unique candidate IDs for decision validation")
    training_runs, training_sha, training_paths, training_source_sha = set(), {model_sha}, set(), set()
    for source in model["training_sources"]:
        run = _identifier(source["run_id"], "training run")
        _, digest = _artifact(source["artifact"], root)
        training_runs.add(run)
        training_sha.add(digest)
        training_source_sha.add(digest)
        training_paths.add(source["artifact"]["path"])
    if not training_runs:
        raise ValueError("Nonempty, explicitly identified training sources required")
    # Native profile documents already identify their training sources. Check
    # that the validation manifest does not silently omit a fitting source.
    if model_document.get("schema") == "gsdeploy.segment_resource_profiles.v1":
        declared_sources = model_document.get("provenance", {}).get("sources", [])
        if not declared_sources or set(declared_sources) not in (training_paths, training_source_sha):
            raise ValueError("Model training provenance differs from manifest training sources")
    observed_pairs = {name: [] for name in ("service", "wait", "e2e", "slowdown", "queue", "p95", "delta_U")}
    eligible_pairs = {name: [] for name in observed_pairs}
    case_reports, seen_ids, seen_runs, seen_prediction_sha, seen_observation_sha = [], set(), set(), set(), set()
    sign_pairs, rank_correct, rank_total, regrets = [], 0, 0, []
    requested, known, eligible = {k: 0 for k in required}, {k: 0 for k in required}, {k: 0 for k in required}
    for planned in plan.values():
        for category in planned["categories"]:
            requested[category] += 1
    points = {k: {"requested": 0, "known": 0, "physical_online": 0} for k in observed_pairs}
    case_list = manifest["cases"]
    if not isinstance(case_list, list):
        raise ValueError("Explicit validation cases list required")
    for case in case_list:
        cid = _identifier(case["id"], "validation case")
        if cid not in plan:
            raise ValueError("Validation case was not in frozen protocol plan")
        run = _identifier(case["run_id"], "validation run")
        if cid in seen_ids or run in seen_runs or run in training_runs:
            raise ValueError("Validation case/run duplicated or overlaps training")
        seen_ids.add(cid)
        seen_runs.add(run)
        if case.get("split") != "validation":
            raise ValueError("Only independent validation split is accepted")
        prediction, psha = _artifact(case["prediction"], root)
        observation, osha = _artifact(case["observation"], root)
        if psha in seen_prediction_sha or osha in seen_observation_sha or osha in training_sha or psha in training_sha:
            raise ValueError("Prediction/observation reused or overlaps training evidence")
        seen_prediction_sha.add(psha)
        seen_observation_sha.add(osha)
        if prediction.get("schema") != PREDICTION_SCHEMA or observation.get("schema") != OBSERVATION_SCHEMA:
            raise ValueError("Explicit sealed prediction and observation schemas required")
        if observation.get("run_id") != run:
            raise ValueError("Observation run identity mismatch")
        if prediction.get("model_sha256") != model_sha or prediction.get("protocol_sha256") != protocol_sha:
            raise ValueError("Prediction does not bind frozen model and validation protocol")
        execution = _identifier(prediction["execution_context_sha256"], "execution context")
        if execution != observation.get("execution_context_sha256"):
            raise ValueError("Prediction/observation execution contexts differ")
        cohort = _identifier(prediction["offered_cohort_sha256"], "offered cohort")
        # For stochastic online forecasting, the predicted random cohort is
        # different. This identity binds the actual experiment's declared
        # workload/input distribution and paired realized cohorts, not a claim
        # that sampled future request identities equal actual future requests.
        if cohort != observation.get("offered_cohort_sha256"):
            raise ValueError("Prediction and observation comparison cohorts differ")
        created = _stamp(prediction["created_ns"], "prediction seal timestamp")
        origin = _stamp(prediction["origin_ns"], "forecast origin")
        cutoff = _stamp(prediction["information_cutoff_ns"], "information cutoff")
        available = _stamp(observation["available_ns"], "observation availability")
        if not max(model_frozen, protocol_frozen) <= created < available or cutoff > origin or origin > created:
            raise ValueError("Model/protocol not frozen, prediction unsealed, or future information cutoff")
        mode = prediction["mode"]
        if mode not in {"online_distribution", "declared_plan", "trace_replay"}:
            raise ValueError("Explicit prediction mode required")
        used_future = prediction.get("actual_future_trace_used")
        if type(used_future) is not bool or (used_future and mode != "trace_replay"):
            raise ValueError("Actual future leakage in online prediction")
        if mode == "declared_plan" and not prediction.get("schedule_declared_before_origin", False):
            raise ValueError("Known planning schedule must be declared before forecast origin")
        kind = observation["evidence_kind"]
        if kind not in {"measured", "cpu_fixture"}:
            raise ValueError("Explicit physical/CPU observation provenance required")
        physical = kind == "measured"
        online_eligible = physical and mode != "trace_replay"
        diagnostic = {"id": cid, "run_id": run, "evidence_kind": kind, "prediction_mode": mode,
                      "physical_online_eligible": online_eligible, "unknown_requests": [], "per_task": {}}
        categories = case.get("categories", plan[cid]["categories"])
        if set(categories) != set(plan[cid]["categories"]):
            raise ValueError("Case categories differ from frozen case plan")
        present = set()

        def point_enabled(category):
            return ("service" if category in {"wait", "e2e"} else category) in categories

        def add_pair(category, pair):
            observed_pairs[category].append(pair)
            if point_enabled(category):
                points[category]["known"] += 1
            if online_eligible:
                eligible_pairs[category].append(pair)
                if point_enabled(category):
                    points[category]["physical_online"] += 1

        predicted = _indexed(prediction.get("requests", []), "request_id", "predicted request")
        actual = _indexed(observation.get("requests", []), "request_id", "observed request")
        # Request matching is meaningful only for fixed-input service probes or
        # a declared schedule. Stochastic future aggregates are assessed via
        # p95_by_task and queue_samples, never by fabricated identity matches.
        if set(predicted) - set(actual):
            raise ValueError("Foreign predicted request identities")
        actual_latencies = {}
        for rid, row in actual.items():
            task = _identifier(row["task"], "task")
            _identifier(row["pool_id"], "public input")
            service = _finite(row["service_ms"], "observed service")
            wait = _finite(row["wait_ms"], "observed wait")
            latency = _finite(row["e2e_ms"], "observed e2e")
            if abs(service + wait - latency) > max(1e-6, latency * 1e-6):
                raise ValueError("Observed wait + whole-handler service must equal E2E")
            actual_latencies.setdefault(task, []).append(latency)
            for name in ("service", "wait", "e2e"):
                if point_enabled(name):
                    points[name]["requested"] += 1
            if row.get("solo_ms") is not None and point_enabled("slowdown"):
                points["slowdown"]["requested"] += 1
            pred = predicted.get(rid)
            if pred is None:
                diagnostic["unknown_requests"].append(rid)
                continue
            if (pred.get("task"), pred.get("pool_id")) != (task, row["pool_id"]):
                raise ValueError("Request input/task mismatch")
            for name, observed in (("service", service), ("wait", wait), ("e2e", latency)):
                value = pred.get(name + "_ms")
                if value is not None:
                    pair = (_finite(value, "predicted " + name), observed)
                    add_pair(name, pair)
                    present.add("service" if name == "service" else name)
            if row.get("solo_ms") is not None and pred.get("solo_ms") is not None and pred.get("service_ms") is not None:
                solo = _finite(row["solo_ms"], "observed solo")
                predicted_solo = _finite(pred["solo_ms"], "predicted solo")
                if solo == 0 or predicted_solo == 0:
                    raise ValueError("Positive solo references required for slowdown")
                pair = (pred["service_ms"] / predicted_solo, service / solo)
                add_pair("slowdown", pair)
                present.add("slowdown")

        p95_predictions = prediction.get("p95_by_task", {})
        if set(p95_predictions) - set(actual_latencies):
            raise ValueError("Predicted P95 for absent validation task")
        expected_tasks = set(plan[cid]["expected_tasks"])
        if set(actual_latencies) - expected_tasks:
            raise ValueError("Observed tasks outside frozen case task coverage")
        diagnostic["missing_observed_tasks"] = sorted(expected_tasks - set(actual_latencies))
        if point_enabled("p95"):
            points["p95"]["requested"] += len(expected_tasks - set(actual_latencies))
        for task, latencies in sorted(actual_latencies.items()):
            if point_enabled("p95"):
                points["p95"]["requested"] += 1
            observed_p95 = percentile(latencies)
            predicted_p95 = p95_predictions.get(task)
            diagnostic["per_task"][task] = {"request_count": len(latencies), "observed_p95_ms": observed_p95,
                                              "predicted_p95_ms": predicted_p95}
            if predicted_p95 is not None:
                pair = (_finite(predicted_p95, "predicted task P95"), observed_p95)
                add_pair("p95", pair)
                present.add("p95")
        pqueue = _indexed(prediction.get("queue_samples", []), "offset_ns", "predicted queue sample")
        oqueue = _indexed(observation.get("queue_samples", []), "offset_ns", "observed queue sample")
        if set(pqueue) - set(oqueue):
            raise ValueError("Queue samples must use the same declared observation grid")
        for offset, sample in oqueue.items():
            if point_enabled("queue"):
                points["queue"]["requested"] += 1
            _stamp(offset, "queue sample offset")
            actual_depth = _finite(sample["depth"], "observed queue depth")
            if offset in pqueue and pqueue[offset].get("depth") is not None:
                pair = (_finite(pqueue[offset]["depth"], "predicted queue depth"), actual_depth)
                add_pair("queue", pair)
                present.add("queue")
        pc = _indexed(prediction.get("candidates", []), "candidate_id", "predicted candidate")
        oc = _indexed(observation.get("candidates", []), "candidate_id", "observed candidate")
        expected_candidates = set(plan[cid].get("expected_candidate_ids", []))
        if expected_candidates and (set(pc) | set(oc)) - expected_candidates:
            raise ValueError("Candidate outside frozen validation domain")
        diagnostic["missing_observed_candidates"] = sorted(expected_candidates - set(oc))
        if set(pc) - set(oc):
            raise ValueError("Foreign predicted candidates")
        candidate_values = {}
        for name, observed in oc.items():
            if observed.get("offered_cohort_sha256") != cohort:
                raise ValueError("Candidate observations are not paired by arrivals and inputs")
            if name not in pc:
                continue
            if pc[name].get("execution_context_sha256", execution) != observed.get("execution_context_sha256", execution):
                raise ValueError("Candidate execution context mismatch")
            pu, ou = _utility(pc[name], weights), _utility(observed, weights)
            if pu is not None and ou is not None:
                candidate_values[name] = (pu, ou)
        baseline = prediction.get("baseline_candidate_id")
        if oc and baseline not in oc:
            raise ValueError("Observed candidate set must include the declared baseline")
        if point_enabled("delta_U"):
            points["delta_U"]["requested"] += max(0, len(expected_candidates or oc) - 1)
        delta_rows = []
        if baseline in candidate_values:
            pb, ob = candidate_values[baseline]
            for name, (pu, ou) in candidate_values.items():
                if name == baseline:
                    continue
                pair = (pu - pb, ou - ob)
                add_pair("delta_U", pair)
                if online_eligible:
                    sign_pairs.append(pair)
                delta_rows.append({"candidate_id": name, "predicted_delta_U": pair[0], "observed_delta_U": pair[1]})
                present.add("delta_U")
        diagnostic["paired_deltas"] = delta_rows
        if len(candidate_values) >= 2:
            present.add("ranking")
            # Ties count as a distinct outcome; a predicted tie does not count
            # as a correct strict ranking, nor does zero count as negative.
            if online_eligible:
                for a, b in itertools.combinations(sorted(candidate_values), 2):
                    pa, oa = candidate_values[a]
                    pb, ob = candidate_values[b]
                    sign = lambda v: 0 if abs(v) <= margin else (1 if v > 0 else -1)
                    rank_correct += sign(pa - pb) == sign(oa - ob)
                    rank_total += 1
                winner = max(sorted(candidate_values), key=lambda k: candidate_values[k][0])
                regret = max(v[1] for v in candidate_values.values()) - candidate_values[winner][1]
                regrets.append(regret)
                diagnostic.update(predicted_winner=winner, observed_selection_regret=regret,
                                  ranking_candidate_count=len(candidate_values),
                                  requested_candidate_count=len(oc))
        diagnostic["known_candidate_count"] = len(candidate_values)
        diagnostic["requested_candidate_count"] = len(oc)
        for category in categories:
            if category in present:
                known[category] += 1
                eligible[category] += online_eligible
        case_reports.append(diagnostic)

    sign = lambda v: 0 if abs(v) <= margin else (1 if v > 0 else -1)
    positives = [pair for pair in sign_pairs if sign(pair[0]) == 1]
    sign_metrics = {"count": len(sign_pairs),
                    "three_way_sign_accuracy": statistics.mean(sign(p) == sign(o) for p, o in sign_pairs) if sign_pairs else None,
                    "predicted_positive_count": len(positives),
                    "positive_gain_precision": statistics.mean(sign(o) == 1 for _, o in positives) if positives else None,
                    "false_positive_gain_count": sum(sign(o) <= 0 for _, o in positives),
                    "pairwise_rank_accuracy": rank_correct / rank_total if rank_total else None,
                    "rank_pair_count": rank_total,
                    "mean_selection_regret": statistics.mean(regrets) if regrets else None,
                    "max_selection_regret": max(regrets) if regrets else None}
    coverage = {key: {"requested_cases": requested[key], "known_cases": known[key],
                      "physical_online_cases": eligible[key],
                      "physical_online_fraction": eligible[key] / requested[key] if requested[key] else 0.}
                for key in required}
    metrics = {name: errors(pairs) for name, pairs in eligible_pairs.items()}
    criteria = protocol.get("acceptance", {})
    available_metrics = {"service_mare": metrics["service"]["mean_absolute_relative_error"],
                         "slowdown_mare": metrics["slowdown"]["mean_absolute_relative_error"],
                         "queue_mae": metrics["queue"]["mae"],
                         "p95_mare": metrics["p95"]["mean_absolute_relative_error"],
                         "delta_U_mae": metrics["delta_U"]["mae"], **sign_metrics}
    gates = []
    for name, rule in criteria.items():
        if name not in available_metrics or not isinstance(rule, dict) or set(rule) != {"operator", "value"}:
            raise ValueError("Recognized metric and explicit acceptance rule required")
        target = _finite(rule["value"], "acceptance target")
        if rule["operator"] not in {">=", "<="}:
            raise ValueError("Acceptance operator must be >= or <=")
        value = available_metrics[name]
        passed = None if value is None else (value >= target if rule["operator"] == ">=" else value <= target)
        gates.append({"metric": name, "value": value, "rule": rule, "passed": passed})
    missing_cases = sorted(set(plan) - seen_ids)
    complete = (bool(case_list) and not missing_cases
                and not any(c["missing_observed_tasks"] or c["missing_observed_candidates"] for c in case_reports)
                and all(requested[k] > 0 and eligible[k] == requested[k] for k in required)
                and all(points[k]["requested"] > 0 and points[k]["physical_online"] == points[k]["requested"]
                        for k in required if k in points)
                and ("ranking" not in required or all(c["known_candidate_count"] == c["requested_candidate_count"]
                                                     for c in case_reports if c["requested_candidate_count"])))
    status = ("NOT_RUN" if not case_list else "INCOMPLETE" if not complete else
              "UNASSESSED" if not gates else "PASS" if all(g["passed"] is True for g in gates) else "FAIL")
    return {"schema": "gsdeploy.independent_predictor_validation_report.v1", "status": status,
            "model_sha256": model_sha, "protocol_sha256": protocol_sha,
            "training_run_count": len(training_runs), "validation_case_count": len(case_list),
            "planned_case_count": len(plan), "missing_case_ids": missing_cases,
            "coverage": coverage, "point_coverage": points, "physical_online_metrics": metrics,
            "all_evidence_diagnostics": {name: errors(pairs) for name, pairs in observed_pairs.items()},
            "physical_online_decisions": sign_metrics, "acceptance_gates": gates, "cases": case_reports,
            "GPU_started": False, "confidence_bounds_calibrated": False,
            "physical_improvement_certified": False,
            "limitations": ["SHA256 verifies artifact identity, not measurement authenticity",
                            "No accuracy claim for unseen inputs or execution contexts",
                            "UNKNOWN cases remain in coverage denominator",
                            "Trace replay and CPU fixtures excluded from physical online acceptance",
                            "No confidence interval or physical optimality certificate"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing validation report")
    manifest = json.loads(args.input.read_text())
    report = assess(manifest, artifact_root=args.input.parent)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "status": report["status"], "GPU_started": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
