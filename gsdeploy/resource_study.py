"""Prepare and independently reconstruct CPU-only resource-study packets.

There is deliberately no GPU execution command. Neither old service profiles
nor old search certificates are transferred to newly controlled resources.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

from .config import SystemConfig
from .io import dumps, save
from .resources import identity, validate_resources, _keys
from .runtime.runner import registry_for
from .runtime.schema import MeasurementConfig, number
from .runtime.workload import SamplePool
from .trace import expand_routes, make_trace, trace_digest


DESIGN_SCHEMA = "gsdeploy.resource_study_design.v1"
PACKET_SCHEMA = "gsdeploy.resource_study_packet.v1"
ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["gsdeploy/resource_study.py", "gsdeploy/resources.py", "gsdeploy/config.py",
           "gsdeploy/runtime/runner.py", "gsdeploy/runtime/transport.py",
           "gsdeploy/runtime/deployment.py", "gsdeploy/runtime/schema.py",
           "gsdeploy/runtime/admission.py", "gsdeploy/cli.py",
           "gsdeploy/runtime/workload.py", "gsdeploy/trace.py", "gsdeploy/routing.py",
           "gsdeploy/energy.py", "gsdeploy/evaluation.py", "gsdeploy/prediction.py",
           "gsdeploy/stability.py", "gsdeploy/quality.py",
           "gsdeploy/metrics.py", "gsdeploy/normalization.py", "gsdeploy/runtime/protocol.py",
           "tools/queue_coupled_utility_20261023/policy.py",
           "tools/queue_coupled_utility_20261023/model.py",
           "tools/resource_control_20261024/policy.py",
           "tools/resource_control_20261024/forecast_policy.py",
           "gsdeploy/forecast_errors.py", "gsdeploy/forecasting/__init__.py",
           "gsdeploy/forecasting/workload.py", "gsdeploy/forecasting/profiles.py",
           "gsdeploy/forecasting/simulator.py", "gsdeploy/forecasting/metrics.py",
           "gsdeploy/forecasting/__main__.py"]


def read(path):
    return json.loads(Path(path).read_text())


def receipt(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def objective_contract(document):
    expected = {k: 1 / 6 for k in "QTLPME"}
    if ((document.get("schema"), document.get("utility_schema")) not in {
                ("gsdeploy.objective_contract.v1", "six_term_quality_v1"),
                ("gsdeploy.objective_contract.v2", "six_term_reference_v2")}
            or document.get("weights") != expected
            or set(document.get("active_utility_terms", [])) != set(expected)
            or document.get("P_evaluation_role") != "objective"
            or document.get("absolute_quality_gate") is not False
            or document.get("stability_policy") != "existing_thresholds_unchanged"):
        raise ValueError("Resource study requires six equal terms including P95 and unchanged stability")
    return copy.deepcopy(document)


def _path(base, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Explicit input path required")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _measurement(system, deployment, routing, design, seed):
    aliases, routes = expand_routes(deployment, routing)
    measurement = MeasurementConfig(
        deployment={a: 1 for a in aliases}, workload_rps=system.workload,
        routing_rps=routes, warmup_seconds=design["warmup_seconds"],
        measurement_seconds=system.horizon_seconds, random_seed=seed,
        arrival_process=system.arrival_process, drain_timeout_seconds=system.drain_seconds)
    measurement.validate({a: system.catalog["models"][m]["supported_tasks"] for a, m in aliases.items()})
    return aliases, measurement


def build_packet(design_path):
    """Read and hash inputs, expand per-process controls, freeze paired cohorts.

    All results are prospective. Source identity is frozen, GPU capacities,
    runtime compatibility and seed reservation remain explicit launch gates.
    """
    design_path = Path(design_path).resolve()
    design = read(design_path)
    _keys(design, {"schema", "system_config", "workers", "samples", "objective",
                   "deployment", "workload_rps", "latency_reference_ms", "duration_seconds",
                   "warmup_seconds", "drain_seconds", "arrival_process", "seeds",
                   "target_task", "specialist_family", "generalist_family", "background_routing",
                   "resource_variants", "policy_parameters"}, {"notes", "unresolved_modes"})
    if design["schema"] != DESIGN_SCHEMA:
        raise ValueError("Unknown resource study design schema")
    paths = {k: _path(design_path.parent, design[k])
             for k in ("system_config", "workers", "samples", "objective")}
    objective = objective_contract(read(paths["objective"]))
    system = replace(SystemConfig.load(paths["system_config"]),
                     workload=design["workload_rps"], latency_reference_ms=design["latency_reference_ms"],
                     weights=objective["weights"], utility_schema=objective["utility_schema"],
                     horizon_seconds=design["duration_seconds"], drain_seconds=design["drain_seconds"],
                     arrival_process=design["arrival_process"])
    if objective['utility_schema'] == 'six_term_reference_v2':
        expected = read(_path(paths['objective'].parent, objective['normalization_reference']))
        if expected != system.catalog.get('utility_normalization'):
            raise ValueError('Objective reference version differs from the effective system')
    number(design["warmup_seconds"], "warmup_seconds")
    if system.drain_seconds <= 0:
        raise ValueError("A positive bounded drain is required for this study")
    deployment = design["deployment"]
    if not isinstance(deployment, dict) or not deployment:
        raise ValueError("Nonempty deployment required")
    for family, count in deployment.items():
        if family not in system.catalog["models"] or type(count) is not int or count <= 0:
            raise ValueError("Invalid deployed family/count")
        if str(count) not in system.catalog["capacity_factors"][family]:
            raise ValueError("Unprofiled physical replica count")
    task, specialist, generalist = (design[k] for k in
                                    ("target_task", "specialist_family", "generalist_family"))
    if (task not in system.workload or specialist == generalist
            or specialist not in deployment or generalist not in deployment):
        raise ValueError("Target and two distinct deployed route families required")
    for family, role in ((specialist, "specialist"), (generalist, "generalist")):
        model = system.catalog["models"][family]
        if model.get("role") != role or task not in model["supported_tasks"]:
            raise ValueError("Target family role/capabilities mismatch")
    if (task, specialist, generalist) != ("text", "text", "generalist_2"):
        raise ValueError("The current resource-policy adapter supports text/text/generalist_2 only")
    params = design["policy_parameters"]
    _keys(params, {"scenarios", "utility_margin", "exposure_rate_rps", "exposure_burst",
                   "cooldown_ms", "drift_ratio_limit", "memory_delta_penalty_upper_norm",
                   "maximum_pending_for_forecast"})
    if params["scenarios"] != ["mean", "p90"]:
        raise ValueError("Explicit mean/p90 development scenarios required")
    for field in ("utility_margin", "cooldown_ms"):
        number(params[field], field)
    for field in ("exposure_rate_rps", "exposure_burst", "drift_ratio_limit"):
        number(params[field], field, positive=True)
    if params["exposure_burst"] < 1 or params["drift_ratio_limit"] <= 1:
        raise ValueError("Invalid exposure burst or drift ratio")
    if params["memory_delta_penalty_upper_norm"] is not None:
        number(params["memory_delta_penalty_upper_norm"], "memory_delta_penalty_upper_norm")
    if type(params["maximum_pending_for_forecast"]) is not int or params["maximum_pending_for_forecast"] < 1:
        raise ValueError("Positive integer forecast budget required")
    background = design["background_routing"]
    if not isinstance(background, dict) or set(background) != set(system.workload) - {task}:
        raise ValueError("Every background task must retain its fixed routing")
    for route_task, row in background.items():
        if not isinstance(row, dict):
            raise ValueError("Background route must be a family/rate mapping")
        for family, rate in row.items():
            number(rate, "background routing rate")
            if family not in deployment or route_task not in system.catalog["models"][family]["supported_tasks"]:
                raise ValueError("Unsupported background route")
    nominal = copy.deepcopy(background)
    nominal[task] = {specialist: system.workload[task]}
    seeds = design["seeds"]
    if (not isinstance(seeds, list) or len(seeds) < 2
            or any(type(s) is not int for s in seeds) or len(set(seeds)) != len(seeds)):
        raise ValueError("At least two distinct explicit development seeds required")
    aliases, _ = _measurement(system, deployment, nominal, design, seeds[0])
    workers = read(paths["workers"])
    registry = registry_for(system, aliases, "native", workers)
    variants = design["resource_variants"]
    if not isinstance(variants, dict) or not variants:
        raise ValueError("Nonempty named resource variants required")
    resource_ids, variant_rows = set(), {}
    for name, resources in variants.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Resource variant names must be nonempty strings")
        resources = validate_resources(resources, aliases)
        rid = identity(resources)
        if rid in resource_ids:
            raise ValueError("Duplicate physical resource variants")
        resource_ids.add(rid)
        bound = registry_for(system, aliases, "native", workers, resources)
        variant_rows[name] = {"resource_configuration": resources,
                              "resource_sha256": rid, "prepared_worker_registry": bound,
                              "launch_blockers": sorted({b for e in bound.values()
                                                         for b in e["resource_binding"]["launch_blockers"]})}
    if not any(v["resource_configuration"]["mode"] == "shared" for v in variant_rows.values()):
        raise ValueError("An explicit shared-resource baseline is required")
    inputs = [receipt(design_path), *(receipt(p) for p in paths.values())]
    # SystemConfig.load resolves the external catalog; bind its bytes as well.
    config_source = read(paths["system_config"])
    if isinstance(config_source.get("catalog"), str):
        inputs.append(receipt(_path(paths["system_config"].parent, config_source["catalog"])))
    for entry in registry.values():
        spec = entry.get("spec")
        if isinstance(spec, str):
            inputs.append(receipt(_path(Path(entry["cwd"]), spec)))
    sources = [receipt(ROOT / p) for p in SOURCES]
    system_dict = system.as_dict()
    scope = {"system": system_dict, "aliases": aliases, "objective": objective,
             "inputs": inputs, "sources": sources, "warmup_seconds": design["warmup_seconds"],
             "target_task": task, "background_routing": background, "policy_parameters": params}
    scope_id = identity(scope)
    jobs, cohorts = [], {}
    pairs = [(name, allow) for name in variants for allow in (False, True)]
    for index, seed in enumerate(seeds):
        trace = make_trace(system.workload, system.horizon_seconds, seed, system.arrival_process)
        pool = SamplePool(paths["samples"], list(system.workload), seed)
        input_rows = []
        counts = dict.fromkeys(system.workload, 0)
        for offset, request_task in trace:
            item = pool.next(request_task)
            input_rows.append([offset, request_task, item["pool_id"]])
            counts[request_task] += 1
        if any(n == 0 for n in counts.values()):
            raise ValueError("Each task needs observations in every paired cohort")
        cohorts[str(seed)] = {"trace_sha256": trace_digest(trace),
                              "input_cohort_sha256": identity(input_rows),
                              "task_counts": counts, "offered_requests": len(trace)}
        for variant, allow_spill in (pairs if index % 2 == 0 else list(reversed(pairs))):
            resources = variant_rows[variant]
            candidate = {"deployment": deployment, "resource_configuration": resources["resource_configuration"],
                         "routing_policy": {"allow_spill": allow_spill, "target_task": task,
                                            "specialist_family": specialist, "generalist_family": generalist,
                                            "background_routing": background,
                                            "parameters": params},
                         "backend": "independent_native_workers", "admission": "existing_full_generate"}
            execution_context = {k: candidate[k] for k in
                                 ("deployment", "resource_configuration", "backend", "admission")}
            execution_context.update(workload_rps=system.workload, scope_sha256=scope_id)
            job_id = identity({"scope_sha256": scope_id, "candidate": candidate, "seed": seed})
            jobs.append({"id": job_id, "candidate_sha256": identity(candidate), "seed": seed,
                         "execution_context_sha256": identity(execution_context),
                         "phase": "development", "variant": variant, "candidate": candidate,
                         "pair_id": identity({"scope_sha256": scope_id, "variant": variant, "seed": seed}),
                         "cohort": cohorts[str(seed)], "weights": objective["weights"],
                         "nominal_no_spill_plan": {"deployment": deployment, "routing": nominal},
                         "execution_status": "NOT_RUN", "U": None,
                         "resource_prediction": {"service": None, "memory": None, "energy": None,
                                                 "reason": "RESOURCE_SPECIFIC_EVIDENCE_REQUIRED"}})
    packet = {"schema": PACKET_SCHEMA, "phase": "CPU_PREPARATION_ONLY", "GPU_started": False,
              "native_execution_enabled": False, "design": receipt(design_path),
              "inputs": inputs, "sources": sources, "scope_sha256": scope_id,
              "effective_system": system_dict, "objective": objective, "aliases": aliases,
              "resource_variants": variant_rows, "cohorts": cohorts, "jobs": jobs,
              "launch_blockers": ["GPU_EXECUTION_NOT_AUTHORIZED_IN_THIS_CPU_PACKET",
                                  "PLATFORM_AND_WORKER_COMPATIBILITY_NOT_VERIFIED",
                                  "RESOURCE_SPECIFIC_PREDICTOR_NOT_VALIDATED",
                                  "DEVELOPMENT_SEED_RESERVATION_NOT_VERIFIED"],
              "comparison_contract": {
                  "matched_no_spill_per_resource_variant": True,
                  "optimize_resources_on_both_sides": True,
                  "no_spill_retains_background_tasks": True,
                  "selected_optimality_certified": False,
                  "physical_optimality_certified": False,
                  "policy_score_scope": "observed_prefix_proxy_not_full_future_six_term_guarantee",
                  "new_resource_domain_requires_new_bounds_and_DFS": True},
              "execution_contract": {"fresh_workers_per_job": True, "no_retries": True,
                                     "priming_policy": "largest_three_per_task",
                                     "resource_effect_must_be_observed_not_assumed": True,
                                     "host_admission_forecast_validated": False},
              "unresolved_modes": design.get("unresolved_modes", [])}
    packet["packet_sha256"] = identity(packet)
    return packet


def validate_packet(packet):
    """Rebuild the entire packet, not merely trust its self-reported hash."""
    issues = []
    try:
        if packet.get("schema") != PACKET_SCHEMA:
            raise ValueError("Unknown packet schema")
        for ref in [packet["design"], *packet["inputs"], *packet["sources"]]:
            if receipt(ref["path"]) != ref:
                raise ValueError("Input/source SHA256 mismatch: " + ref["path"])
        expected = build_packet(packet["design"]["path"])
        if packet != expected:
            raise ValueError("Packet does not reconstruct from frozen design/inputs/sources")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        issues.append(str(exc))
    return {"valid": not issues, "issues": issues, "GPU_started": False,
            "native_execution_enabled": False,
            "job_count": len(packet.get("jobs", [])),
            "scope": "CPU structural/provenance validation only; no hardware or performance claim"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--design", required=True)
    prepare.add_argument("--output", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("--packet", required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            if Path(args.output).exists():
                raise FileExistsError("Output already exists: " + args.output)
            packet = build_packet(args.design)
            check = validate_packet(packet)
            if not check["valid"]:
                raise ValueError(check["issues"])
            save(args.output, packet)
            print(dumps({**check, "packet": str(Path(args.output).resolve()),
                         "packet_sha256": packet["packet_sha256"]}), end="")
            return 0
        result = validate_packet(read(args.packet))
        print(dumps(result), end="")
        return 0 if result["valid"] else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"resource-study: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
