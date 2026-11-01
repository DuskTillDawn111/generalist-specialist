#!/usr/bin/env python3
"""Read frozen study plans and produce CPU diagnostics; never launch workers."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.static_study import _candidate_calibrations, validate_packet
from tests.test_milp import add_energy, grid_plans
from tests.test_search_contract import config as synthetic_config


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize(prediction):
    keys = ["U", "diagnostic_partial_U", "utility_complete", "utility_schema", "reason",
            "constraints_eligible", "prediction_scope", "max_rho", "safe_memory_mib",
            "offered_requests", "within_window_requests", "end_backlog_requests",
            "incomplete_after_drain", "trace_sha256", "per_task", "quality_evidence",
            "performance_evidence", "energy_evidence", "energy_profile_key",
            "scenario_U_values", "scenario_U_low", "stress_constraints_valid"]
    return {**{k: prediction.get(k) for k in keys},
            "normalized": {k: prediction.get(k + "_norm") for k in "QTLPME"},
            "contributions": {k: prediction.get(k + "_contribution") for k in "QTLPME"}}


def actual_diagnostics(inputs, packet_path):
    c = SystemConfig.load(inputs / "system-config.json")
    packet = json.loads(packet_path.read_text())
    validation = validate_packet(packet)
    if not validation["valid"]:
        raise ValueError("Frozen packet has changed files: " + repr(validation["changed"]))
    calibrations = []
    for job in packet["jobs"]:
        if job["phase"] != "calibration":
            continue
        conditions = job["identity"]
        matched = replace(c, horizon_seconds=conditions["horizon_seconds"],
                          random_seed=job["seed"])
        p = DeploymentSystem(matched).predict(**{
            "deployment": job["plan"]["deployment"], "routing": job["plan"]["routing"]})
        calibrations.append(dict(
            job_id=job["id"], labels=job["labels"], plan=job["plan"],
            horizon_seconds=matched.horizon_seconds, seed=matched.random_seed,
            prediction=summarize(p),
        ))
    proposals = {label: dict(deployment=x, routing=r)
                 for label, x, r in _candidate_calibrations(c)}
    controls = []
    for pool, label in [
        ("full_retained", "mixed_generalist_2_text_specialist_share_0.75"),
        ("generalist_only", "shared_generalist_2"),
        ("specialist_only", "specialists_shared_board"),
    ]:
        plan = proposals[label]
        controls.append(dict(
            pool=pool, label=label, plan=plan, selection="fixed representative, not searched or selected by utility",
            horizon_seconds=c.horizon_seconds, seed=c.random_seed,
            prediction=summarize(DeploymentSystem(c).predict(**plan)),
        ))
    return dict(
        evidence_kind="retained model pool: recorded offline quality plus closed-loop replica-mean FIFO transfer proxy",
        GPU_started=False, winner=None, optimality_claim=False,
        limitation="No measured mixed interference, tails or whole-board energy; partial diagnostics cannot select a six-term winner.",
        packet_validation=validation, workload=c.workload,
        formal_horizon_seconds=c.horizon_seconds, weights=c.weights,
        calibration_plans=calibrations, fixed_structural_representatives=controls,
    )


def synthetic_diagnostics():
    # Same independently published CPU fixture used by the splitting acceptance
    # test. These numbers and energy records are deliberately invented.
    c = synthetic_config({"text": 4.8, "image": 3.0},
                         {"g": {"text": 200.0, "image": 50.0}, "s": {"text": 160.0}},
                         horizon=120)
    c.catalog["capability_contract"] = "verified_executable_v1"
    for spec in c.catalog["models"].values():
        spec["executable_tasks"] = spec["supported_tasks"][:]
    paths = {"g": {"text": ("em", 0.8), "image": ("accuracy", 0.9)},
             "s": {"text": ("em", 0.95)}}
    c.catalog["quality"] = dict(
        schema_version=1, mode="aggregate_proxy",
        identity={k: "synthetic CPU fixture" for k in (
            "input_pool_sha256", "reference_sha256", "protocol_sha256", "source")},
        paths={m: {t: dict(metric=metric, value=value, unit="fraction",
                           checkpoint_sha256="synthetic checkpoint", source="CPU fixture")
                   for t, (metric, value) in tasks.items()}
               for m, tasks in paths.items()},
    )
    c = replace(c, weights={k: 1 / 6 for k in "QTLPME"}, solver="milp",
                utility_schema="six_term_quality_v1", arrival_process="poisson",
                latency_reference_ms={"text": 1000.0, "image": 250.0},
                milp_time_limit_seconds=15)
    add_energy(c, list(grid_plans(c, 4)), lambda *_: 12000.0)
    sensitivity = []
    for q in [0.001, 0.1, 1 / 6, 0.3, 0.6, 0.9]:
        matched = replace(c, weights={k: q if k == "Q" else (1 - q) / 5 for k in "QTLPME"})
        result = DeploymentSystem(matched).plan(route_grid=4)
        best = result.get("winner")
        sensitivity.append(dict(
            quality_weight=q, weights=matched.weights, status=result["status"],
            optimality_certified=result["optimality_certified"], optimality_gap=result["optimality_gap"],
            plan={"deployment": best["deployment"], "routing": best["routing"]} if best else None,
            prediction=summarize(best["prediction"]) if best else None,
        ))
    system = DeploymentSystem(c)
    best = system.plan(route_grid=4)
    plan = best["winner"]
    comparisons = []
    for label, result in [
        ("free_joint_optimum", best),
        ("best_no_split_global", system.plan(route_grid=4, no_split=True)),
        ("best_no_split_same_deployment", system.plan(
            deployment=plan["deployment"], route_grid=4, no_split=True)),
        ("strongest_alternative_deployment", system.plan(
            route_grid=4, exclude_deployments=[plan["deployment"]])),
        ("strongest_alternative_same_deployment_route", system.plan(
            deployment=plan["deployment"], route_grid=4, exclude_plans=[plan])),
    ]:
        winner = result.get("winner")
        comparisons.append(dict(
            label=label, status=result["status"], optimality_certified=result["optimality_certified"],
            optimality_gap=result["optimality_gap"],
            plan={"deployment": winner["deployment"], "routing": winner["routing"]} if winner else None,
            prediction=summarize(winner["prediction"]) if winner else None,
        ))
    return dict(
        evidence_kind="synthetic_cpu_only", claim_scope="correctness of frozen finite-domain scoring/search, not real model/GPU performance",
        workload=c.workload, input_model_accuracy={"g_text": 0.8, "s_text": 0.95, "g_image": 0.9},
        quality_weight_sensitivity=sensitivity, strong_challengers=comparisons,
    )


def number(value):
    return "unknown" if value is None else f"{value:.6f}"


def render(data):
    lines = ["# CPU diagnostics: frozen pool and synthetic routing controls", "",
             "No GPU jobs were launched. The frozen-pool plans have no complete utility or winner: board energy and mixed-run validation are missing.",
             "Recorded offline quality is replayed; latency and throughput use a closed-loop replica-mean FIFO transfer proxy, not measured mixed GPU performance.", "",
             "## Frozen-pool calibration plans", "",
             "Each calibration plan uses its packet horizon and seed. Normalized values below are predictions; E and U remain unknown.", "",
             "| Plan | H/s | Q | T | L | P | M | E | U |",
             "|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for item in data["actual_frozen_pool"]["calibration_plans"]:
        p = item["prediction"]
        lines.append("| " + " / ".join(item["labels"]) + f" | {item['horizon_seconds']} | "
                     + " | ".join(number(p["normalized"][k]) for k in "QTLPME")
                     + " | " + number(p["U"]) + " |")
    lines += ["", "## Fixed structural representatives", "",
              "These are three fixed feasible representatives at the formal horizon, not pool optima. Their partial scores must not rank the pools.", "",
              "| Pool | Fixed representative | H/s | Q contribution | T contribution | L contribution | P contribution | M contribution | E | U |",
              "|---|---|---:|---:|---:|---:|---:|---:|---|---|"]
    for item in data["actual_frozen_pool"]["fixed_structural_representatives"]:
        p = item["prediction"]
        lines.append(f"| {item['pool']} | {item['label']} | {item['horizon_seconds']} | "
                     + " | ".join(number(p["contributions"][k]) for k in "QTLPM")
                     + " | unknown | " + number(p["U"]) + " |")
    lines += ["", "## Synthetic quality-weight sensitivity", "",
              "All values in this section, including energy, are invented CPU fixtures. Specialist text accuracy is 0.95, generalist text accuracy 0.80.",
              "All other five weights share (1-Q weight)/5. Routing is reoptimized over the same finite domain.", "",
              "| Q weight | Generalist text share | U | Q loss | Text P95/ms | Certified |",
              "|---|---:|---:|---:|---:|---|"]
    for item in data["synthetic_controls"]["quality_weight_sensitivity"]:
        p, plan = item["prediction"], item["plan"]
        share = plan["routing"]["text"].get("g", 0) / data["synthetic_controls"]["workload"]["text"]
        lines.append(f"| {item['quality_weight']:.6f} | {share:.2f} | {number(p['U'])} | "
                     f"{number(p['normalized']['Q'])} | {p['per_task']['text']['p95_ms']:.3f} | {item['optimality_certified']} |")
    lines += ["", "## Synthetic strong challengers at equal six-term weights", "",
              "Each challenger is reoptimized with its stated restriction. This includes alternative deployments and routes, beyond pure generalist/specialist comparisons.", "",
              "| Candidate | Generalist text share | U | Q loss | Text P95/ms | Certified |",
              "|---|---:|---:|---:|---:|---|"]
    for item in data["synthetic_controls"]["strong_challengers"]:
        p, plan = item["prediction"], item["plan"]
        if p is None:
            lines.append(f"| {item['label']} | — | — | — | — | {item['status']} |")
            continue
        share = plan["routing"]["text"].get("g", 0) / data["synthetic_controls"]["workload"]["text"]
        lines.append(f"| {item['label']} | {share:.2f} | {number(p['U'])} | {number(p['normalized']['Q'])} | "
                     f"{p['per_task']['text']['p95_ms']:.3f} | {item['optimality_certified']} |")
    lines += ["", "diagnostics.json preserves all Q/T/L/P/M/E normalized values and contributions, per-task counts/latencies/quality provenance, trace hashes, input hashes, constraints and statuses.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=Path("artifacts/static-study-v6-20261021/inputs"))
    parser.add_argument("--packet", type=Path, default=Path("artifacts/static-study-v6-20261021/experiments-final/experiment-packet.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/static_joint_cpu_v6_20261021"))
    args = parser.parse_args()
    args.inputs, args.packet, args.output = args.inputs.resolve(), args.packet.resolve(), args.output.resolve()
    source_files = [args.inputs / name for name in ["system-config.json", "catalog.json", "quality-index.json", "pool-manifest.json"]]
    source_files += [args.packet, Path(__file__).resolve()]
    source_files += [Path(__file__).resolve().parents[1] / "tests" / name for name in (
        "test_search_contract.py", "test_milp.py", "test_static_milp_v6.py")]
    data = dict(schema_version=1, GPU_started=False,
                sources={str(p): sha(p) for p in source_files},
                actual_frozen_pool=actual_diagnostics(args.inputs, args.packet),
                synthetic_controls=synthetic_diagnostics())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "diagnostics.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    (args.output / "diagnostics.md").write_text(render(data))
    print(json.dumps(dict(actual_calibrations=len(data["actual_frozen_pool"]["calibration_plans"]),
                          fixed_structural_representatives=len(data["actual_frozen_pool"]["fixed_structural_representatives"]),
                          GPU_started=False, output=str(args.output))))


if __name__ == "__main__":
    main()
