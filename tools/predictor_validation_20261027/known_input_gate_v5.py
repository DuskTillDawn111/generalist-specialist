"""Known-input conditional prediction gate with fresh heldout co-run requests.

Evaluation never fits models or changes thresholds. Old V4 protocols/results
are neither imported nor modified. All error/ranking coverage is explicit.
"""
from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics

from gsdeploy.resources import identity
from gsdeploy.forecasting.metrics import percentile
from gsdeploy.forecasting.workload import number

PROTOCOL_SCHEMA = "gsdeploy.known_input_prediction_gate_protocol.v5"
PREDICTION_SCHEMA = "gsdeploy.known_input_prediction_gate_seal.v5"
OBSERVED_SCHEMA = "gsdeploy.known_input_prediction_gate_observed_run.v5"
DATA_SCHEMA = "gsdeploy.independent_prediction_gate_training_index.v2"
THRESHOLDS = {"service_mare_max", "queue_mae_ms_max", "e2e_mare_max", "p95_mare_max",
              "queue_depth_mae_max", "delta_sign_accuracy_min", "pair_rank_accuracy_min"}


def _text(value, name):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit " + name + " required")
    return value


def _ns(value):
    if type(value) is not int or value < 0:
        raise ValueError("Nonnegative integer Unix metadata time required")
    return value


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _verify(reference):
    if set(reference) != {"path", "sha256"} or receipt(reference["path"]) != reference:
        raise ValueError("Frozen evidence identity changed")


def _read(path):
    return json.loads(Path(path).read_text())


def _requests(rows):
    result = {}
    for row in rows:
        if set(row) != {"request_id", "pool_id", "task"}:
            raise ValueError("Explicit public request/input/task index required")
        for key in row:
            _text(row[key], key)
        if row["request_id"] in result:
            raise ValueError("Duplicate indexed request")
        result[row["request_id"]] = row
    if not result:
        raise ValueError("Nonempty request domain required")
    return result


def freeze_gate_protocol(*, model_paths, source_paths, training_data_paths, candidates,
                         baseline_candidate_id, validation_domain, thresholds, created_unix_ns,
                         weights, utility_tie_tolerance=0., queue_sample_offsets_ns=(),
                         public_input_catalog_path=None):
    """Bind a prospective gate; thresholds are supplied, with no built-in retune.

    Training indices must list all training run/input identities. Their source
    raw data/model fitting lineage remains a separate, auditable prerequisite.
    This index cannot magically prove an externally false split declaration.
    """
    _ns(created_unix_ns)
    if set(weights) != set("QTLPME") or abs(sum(weights.values()) - 1.) > 1e-12:
        raise ValueError("Six original utility weights summing to one required")
    for weight in weights.values():
        number(weight, "utility weight")
    if not isinstance(thresholds, dict) or set(thresholds) - THRESHOLDS:
        raise ValueError("Only prospectively declared v2 thresholds accepted")
    for threshold in thresholds.values():
        number(threshold, "frozen gate threshold")
    number(utility_tie_tolerance, "utility tie tolerance")
    offsets = list(queue_sample_offsets_ns)
    if offsets != sorted(set(offsets)) or any(type(v) is not int or v < 0 for v in offsets):
        raise ValueError("Distinct increasing nonnegative queue sampling offsets required")
    if "queue_depth_mae_max" in thresholds and not offsets:
        raise ValueError("Queue-depth threshold requires predeclared sampling offsets")
    if not model_paths or not source_paths or not training_data_paths:
        raise ValueError("Frozen model/source/training-data identities required")
    models = {key: receipt(path) for key, path in model_paths.items()}
    for key in models:
        _text(key, "model identity")
    candidate_map = {}
    for candidate in candidates:
        if set(candidate) != {"id", "model_id", "execution_context_sha256"}:
            raise ValueError("Candidate/model/execution binding required")
        for value in candidate.values():
            _text(value, "candidate identity")
        if candidate["id"] in candidate_map or candidate["model_id"] not in models:
            raise ValueError("Duplicate candidate or missing bound model")
        candidate_map[candidate["id"]] = candidate
    if len(candidate_map) < 2 or baseline_candidate_id not in candidate_map:
        raise ValueError("At least two candidates and an explicit baseline required")
    training_refs, train_runs, train_inputs, train_ids = [], set(), set(), set()
    for path in training_data_paths:
        data = _read(path)
        if set(data) != {"schema", "split", "records"} or data["schema"] != DATA_SCHEMA or data["split"] != "training":
            raise ValueError("Explicit training-only gate index required")
        for row in data["records"]:
            if set(row) != {"run_id", "request_id", "pool_id", "task"}:
                raise ValueError("Complete training run/input index required")
            for key in row:
                _text(row[key], key)
            if (row["run_id"], row["request_id"]) in train_ids:
                raise ValueError("Duplicate training-index request")
            train_ids.add((row["run_id"], row["request_id"]))
            train_runs.add(row["run_id"])
            train_inputs.add(row["pool_id"])
        training_refs.append(receipt(path))
    if not train_ids:
        raise ValueError("Nonempty training identity index required")
    if public_input_catalog_path is None:
        raise ValueError('Known-input scope requires immutable public input catalog')
    from .generation_work_v4 import input_content_identity
    catalog_ref=receipt(public_input_catalog_path)
    if catalog_ref not in [receipt(p) for p in source_paths]:
        raise ValueError('Freeze public known-input content catalog before prediction')
    catalog={row['pool_id']:row for row in map(json.loads,Path(public_input_catalog_path).read_text().splitlines())}
    content_index={pool:dict(task=row['task_type'],input_content_sha256=input_content_identity(row)) for pool,row in catalog.items()}
    domains = {}
    for run in validation_domain:
        if set(run) != {"run_id", "requests"} or run["run_id"] in domains or run["run_id"] in train_runs:
            raise ValueError("Unique training-disjoint validation runs required")
        _text(run["run_id"], "validation run")
        index = _requests(run["requests"])
        for row in index.values():
            if (row['pool_id'] not in train_inputs or row['pool_id'] not in content_index
                    or content_index[row['pool_id']]['task']!=row['task']):
                raise ValueError('Every known-input validation content must be explicitly trained and public')
            if row['request_id'] in {request for run,request in train_ids}:
                raise ValueError('Known-input validation requires fresh heldout request identities')
        domains[run["run_id"]] = list(index.values())
    if not domains:
        raise ValueError("Prospective validation input/run domain required")
    protocol = {"schema": PROTOCOL_SCHEMA, "created_unix_ns": created_unix_ns,
        "evaluator_source": receipt(__file__),
        "models": models, "sources": [receipt(path) for path in source_paths], "training_data": training_refs,
        "training_run_ids": sorted(train_runs), "training_input_ids": sorted(train_inputs),
        "candidates": list(candidate_map.values()), "baseline_candidate_id": baseline_candidate_id,
        "validation_domain": domains, "thresholds": dict(thresholds), "weights": dict(weights),
        "utility_tie_tolerance": utility_tie_tolerance, "queue_sample_offsets_ns": offsets}
    protocol.update(validation_scope='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS',
        unseen_input_generalization_claimed=False,training_input_overlap_explicit=True,
        public_input_catalog=catalog_ref,
        known_public_contents={pool:content_index[pool] for pool in sorted({r['pool_id'] for domain in domains.values() for r in domain})})
    protocol["protocol_sha256"] = identity(protocol)
    return protocol


def _verify_protocol(protocol):
    if (protocol.get("schema") != PROTOCOL_SCHEMA
            or protocol.get("protocol_sha256") != identity({k: v for k, v in protocol.items() if k != "protocol_sha256"})):
        raise ValueError("Frozen v2 gate protocol changed")
    if (protocol.get('validation_scope')!='FRESH_HELDOUT_CO_RUN_REQUESTS_OF_EXACT_TRAINED_PUBLIC_INPUTS'
            or protocol.get('unseen_input_generalization_claimed') is not False):
        raise ValueError('Known-input conditional scope must remain explicit')
    _verify(protocol['public_input_catalog'])
    for reference in [protocol["evaluator_source"], *protocol["models"].values(), *protocol["sources"], *protocol["training_data"]]:
        _verify(reference)


def _cells(cells, protocol):
    candidates = {candidate["id"]: candidate for candidate in protocol["candidates"]}
    result = {}
    for cell in cells:
        required = {"run_id", "candidate_id", "execution_context_sha256", "requests", "terms"}
        if required - set(cell) or set(cell) - required - {"queue_depth_samples"}:
            raise ValueError("Explicit candidate request timings and original six terms required")
        key = cell["run_id"], cell["candidate_id"]
        candidate = candidates.get(cell["candidate_id"])
        if (key in result or cell["run_id"] not in protocol["validation_domain"] or candidate is None
                or cell["execution_context_sha256"] != candidate["execution_context_sha256"]):
            raise ValueError("Duplicate or foreign validation cell/execution context")
        expected = _requests(protocol["validation_domain"][cell["run_id"]])
        seen = set()
        for row in cell["requests"]:
            if set(row) != {"request_id", "pool_id", "task", "service_ms", "queue_ms", "e2e_ms"}:
                raise ValueError("Explicit service/queue/end-to-end timing rows required")
            rid = row["request_id"]
            if rid not in expected or rid in seen or any(row[k] != expected[rid][k] for k in ("pool_id", "task")):
                raise ValueError("Timing identity/input/task differs from frozen validation domain")
            seen.add(rid)
            for name in ("service_ms", "queue_ms", "e2e_ms"):
                if row[name] is not None:
                    number(row[name], name)
        if cell["terms"] is not None:
            if set(cell["terms"]) != set("QTLPME"):
                raise ValueError("Six-term utility evidence required")
            for term in cell["terms"].values():
                if term is not None:
                    number(term, "utility term")
        sampled = set()
        for sample in cell.get("queue_depth_samples", []):
            if (set(sample) != {"offset_ns", "depth"} or type(sample["offset_ns"]) is not int
                    or sample["offset_ns"] not in protocol["queue_sample_offsets_ns"] or sample["offset_ns"] in sampled):
                raise ValueError("Queue sample differs from predeclared observation grid")
            sampled.add(sample["offset_ns"])
            if sample["depth"] is not None:
                number(sample["depth"], "queue depth")
        result[key] = cell
    return result


def seal_predictions(protocol, cells, *, created_unix_ns):
    _verify_protocol(protocol)
    if _ns(created_unix_ns) < protocol["created_unix_ns"]:
        raise ValueError("Prediction seal precedes frozen protocol")
    mapped=_cells(cells, protocol)
    expected={(run,c['id']) for run in protocol['validation_domain'] for c in protocol['candidates']}
    if set(mapped)!=expected:raise ValueError('Complete known-input candidate/run prediction Cartesian product required')
    sealed = {"schema": PREDICTION_SCHEMA, "protocol_sha256": protocol["protocol_sha256"],
              "created_unix_ns": created_unix_ns, "cells": copy.deepcopy(cells)}
    sealed["prediction_sha256"] = identity(sealed)
    return sealed


def _errors(pairs, expected):
    known = [(p, o) for p, o in pairs if p is not None and o is not None]
    absolute = [abs(p - o) for p, o in known]
    relative = [abs(p - o) / o if o else (0. if p == 0 else None) for p, o in known]
    return {"expected": expected, "known": len(known), "complete": len(known) == expected,
        "mae": statistics.mean(absolute) if absolute else None,
        "mare": statistics.mean(relative) if relative and all(v is not None for v in relative) else None,
        "undefined_relative_denominators": sum(v is None for v in relative),
        "max_absolute_error": max(absolute) if absolute else None}


def _utility(cell, weights):
    terms = cell.get("terms") if cell else None
    return -sum(weights[key] * terms[key] for key in weights) if terms and all(v is not None for v in terms.values()) else None


def evaluate_prediction_gate(protocol, predictions, observed_paths):
    """Evaluate a presealed prediction against independent validation receipts."""
    _verify_protocol(protocol)
    if (predictions.get("schema") != PREDICTION_SCHEMA or predictions.get("protocol_sha256") != protocol["protocol_sha256"]
            or predictions.get("prediction_sha256") != identity({k: v for k, v in predictions.items() if k != "prediction_sha256"})
            or predictions["created_unix_ns"] < protocol["created_unix_ns"]):
        raise ValueError("Frozen prediction identity/protocol mismatch")
    predicted = _cells(predictions["cells"], protocol)
    observed, refs, kinds, run_ids = {}, [], set(), set()
    for path in observed_paths:
        run = _read(path)
        if (set(run) != {"schema", "split", "kind", "run_id", "collection_started_unix_ns", "cells"}
                or run["schema"] != OBSERVED_SCHEMA or run["split"] != "validation"
                or run["kind"] not in {"measured", "cpu_fixture"}
                or _ns(run["collection_started_unix_ns"]) <= predictions["created_unix_ns"]
                or run["run_id"] not in protocol["validation_domain"]
                or any(cell["run_id"] != run["run_id"] for cell in run["cells"])):
            raise ValueError("Independent post-seal validation run required")
        mapped = _cells(run["cells"], protocol)
        if set(observed) & set(mapped) or run["run_id"] in run_ids:
            raise ValueError("Duplicate observed candidate/run cells")
        run_ids.add(run["run_id"])
        kinds.add(run["kind"])
        observed.update(mapped)
        refs.append(receipt(path))
    if len(kinds) > 1:
        raise ValueError("Physical and CPU-fixture observations cannot be mixed")
    diagnostics, utilities, checks = [], {}, []
    for run_id, domain in protocol["validation_domain"].items():
        for candidate in protocol["candidates"]:
            key = run_id, candidate["id"]
            p, o = predicted.get(key), observed.get(key)
            pp = {r["request_id"]: r for r in p["requests"]} if p else {}
            oo = {r["request_id"]: r for r in o["requests"]} if o else {}
            errors = {name: _errors([(pp.get(r["request_id"], {}).get(name + "_ms"),
                                     oo.get(r["request_id"], {}).get(name + "_ms")) for r in domain], len(domain))
                      for name in ("service", "queue", "e2e")}
            p95_pairs = []
            for task in sorted({r["task"] for r in domain}):
                values = []
                for indexed in (pp, oo):
                    latencies = [indexed.get(r["request_id"], {}).get("e2e_ms") for r in domain if r["task"] == task]
                    values.append(percentile(latencies) if all(v is not None for v in latencies) else None)
                p95_pairs.append(tuple(values))
            errors["p95"] = _errors(p95_pairs, len(p95_pairs))
            per_task_p95={task:_errors([pair],1) for task,pair in zip(sorted({r['task'] for r in domain}),p95_pairs)}
            if 'p95_mare_max' in protocol['thresholds']:
                for task,metric in per_task_p95.items():
                    value=metric['mare'] if metric['complete'] else None
                    checks.append(dict(run_id=run_id,candidate_id=candidate['id'],task=task,
                        threshold='p95_mare_max',value=value,limit=protocol['thresholds']['p95_mare_max'],
                        passed=value<=protocol['thresholds']['p95_mare_max'] if value is not None else None))
            if protocol["queue_sample_offsets_ns"]:
                ps = {s["offset_ns"]: s["depth"] for s in p.get("queue_depth_samples", [])} if p else {}
                os = {s["offset_ns"]: s["depth"] for s in o.get("queue_depth_samples", [])} if o else {}
                errors["queue_depth"] = _errors([(ps.get(offset), os.get(offset))
                    for offset in protocol["queue_sample_offsets_ns"]], len(protocol["queue_sample_offsets_ns"]))
            utilities[key] = (_utility(p, protocol["weights"]), _utility(o, protocol["weights"]))
            diagnostics.append({"run_id": run_id, "candidate_id": candidate["id"], "errors": errors,
                                "per_task_p95": per_task_p95,
                                "predicted_U": utilities[key][0], "observed_U": utilities[key][1]})
            for threshold, metric, field in (("service_mare_max", "service", "mare"),
                ("queue_mae_ms_max", "queue", "mae"), ("e2e_mare_max", "e2e", "mare"), ("p95_mare_max", "p95", "mare")):
                if threshold in protocol["thresholds"]:
                    value = errors[metric][field] if errors[metric]["complete"] else None
                    checks.append({"run_id": run_id, "candidate_id": candidate["id"], "threshold": threshold,
                        "value": value, "limit": protocol["thresholds"][threshold],
                        "passed": value <= protocol["thresholds"][threshold] if value is not None else None})
            if "queue_depth_mae_max" in protocol["thresholds"]:
                metric = errors["queue_depth"]
                value = metric["mae"] if metric["complete"] else None
                checks.append({"run_id": run_id, "candidate_id": candidate["id"], "threshold": "queue_depth_mae_max",
                    "value": value, "limit": protocol["thresholds"]["queue_depth_mae_max"],
                    "passed": value <= protocol["thresholds"]["queue_depth_mae_max"] if value is not None else None})
    tolerance = protocol["utility_tie_tolerance"]
    sign = lambda value: 0 if abs(value) <= tolerance else (1 if value > 0 else -1)
    paired, ranks = [], []
    for run_id in protocol["validation_domain"]:
        base = utilities[run_id, protocol["baseline_candidate_id"]]
        for candidate in protocol["candidates"]:
            if candidate["id"] == protocol["baseline_candidate_id"]:
                continue
            value = utilities[run_id, candidate["id"]]
            known = all(v is not None for v in (*base, *value))
            predicted_delta, observed_delta = (value[0] - base[0], value[1] - base[1]) if known else (None, None)
            paired.append({"run_id": run_id, "candidate_id": candidate["id"], "predicted_delta_U": predicted_delta,
                "observed_delta_U": observed_delta, "correct": sign(predicted_delta) == sign(observed_delta) if known else None})
        for a, b in itertools.combinations(sorted(c["id"] for c in protocol["candidates"]), 2):
            aa, bb = utilities[run_id, a], utilities[run_id, b]
            known = all(v is not None for v in (*aa, *bb))
            ranks.append({"run_id": run_id, "a": a, "b": b,
                "correct": sign(aa[0] - bb[0]) == sign(aa[1] - bb[1]) if known else None})
    def accuracy(rows):
        values = [r["correct"] for r in rows if r["correct"] is not None]
        return {"expected": len(rows), "known": len(values), "accuracy": sum(values) / len(values) if values else None}
    delta, rank = accuracy(paired), accuracy(ranks)
    for threshold, value in (("delta_sign_accuracy_min", delta), ("pair_rank_accuracy_min", rank)):
        if threshold in protocol["thresholds"]:
            score = value["accuracy"] if value["known"] == value["expected"] else None
            checks.append({"threshold": threshold, "value": score, "limit": protocol["thresholds"][threshold],
                           "passed": score >= protocol["thresholds"][threshold] if score is not None else None})
    complete = (all(e["complete"] for d in diagnostics for e in d["errors"].values())
                and all(d["predicted_U"] is not None and d["observed_U"] is not None for d in diagnostics))
    status = ("FAIL" if any(c["passed"] is False for c in checks) else "UNKNOWN" if not complete
              or any(c["passed"] is None for c in checks) else "PASS" if checks else "ASSESSED_NOT_GATED")
    return {"schema": "gsdeploy.known_input_prediction_gate_report.v5", "status": status,
        "protocol_sha256": protocol["protocol_sha256"], "prediction_sha256": predictions["prediction_sha256"],
        "observed_sources": refs, "diagnostics": diagnostics, "paired_deltas": paired, "delta_sign": delta,
        "candidate_pair_ranks": ranks, "pair_ranking": rank, "checks": checks, "coverage_complete": complete,
        "delta_U_mae": statistics.mean(abs(r["predicted_delta_U"] - r["observed_delta_U"])
            for r in paired if r["correct"] is not None) if delta["known"] else None,
        "validation_scope": protocol['validation_scope'], "unseen_input_generalization_claimed": False,
        "thresholds_changed": False, "validation_refit": False, "legacy_v4_artifacts_modified": False,
        "evidence_kind": next(iter(kinds)) if kinds else None,
        "uncertainty_calibrated": False, "physical_gain_certified": False, "final_four_groups_enabled": False}


def main(argv=None):
    """Evaluate already sealed artifacts; never collect profiles or launch GPU."""
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--observed", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Refusing to overwrite an existing gate report")
    report = evaluate_prediction_gate(_read(args.protocol), _read(args.predictions), args.observed)
    report["input_artifacts"] = {"protocol": receipt(args.protocol), "predictions": receipt(args.predictions)}
    report["GPU_started"] = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
