"""Legacy exhaustive baseline CLI; use deployment_core.seed for initialization."""

import argparse
import json
from pathlib import Path

from .candidate_generator import screen_deployments
from .models import DEFAULT_SOURCES, Hardware, load_hardware, load_model_catalog
from .routing_solver import LP_TOLERANCE, ROUTING_REL_TOLERANCE, RoutingSolverError
from .workload import TASKS, parse_workload


def report_as_dict(report, catalog, workload, hardware, rho_max):
    return {
        "schema_version": 1,
        "screening": "coverage + memory + standalone capacity LP",
        "arrival_rate_rps": dict(workload.arrival_rate),
        "hardware": {"name": hardware.name, "gpu_memory_mib": hardware.gpu_memory_mib,
                     "source": hardware.source},
        "rho_max": rho_max,
        "numerical_tolerances": {"lp_feasibility": LP_TOLERANCE,
                                 "routing_conservation_relative": ROUTING_REL_TOLERANCE},
        "model_catalog": {
            name: {"supported_tasks": [task for task in TASKS if task in model.supported_tasks],
                   "memory_mib": model.memory_mib, "service_time_ms": dict(model.service_time_ms),
                   "sources": model.sources}
            for name, model in sorted(catalog.items())
        },
        "counts": report.counts,
        "feasible_deployments": [
            {"models": list(row.deployment.models), "memory_mib": row.memory_mib,
             "routing_rps": [{"task": task, "model": model, "rate_rps": rate}
                             for (task, model), rate in row.routing.routing_rps.items()],
             "utilization": row.routing.utilization}
            for row in report.feasible_deployments
        ],
        "rejected_deployments": [
            {"models": list(row.deployment.models), "stage": row.stage, "reason": row.reason}
            for row in report.rejected_deployments
        ],
        "scope": "Fast screening using standalone measurements; not a guarantee of colocated capacity. "
                 "Routing is a feasibility witness. Omitted routing pairs have zero rate.",
    }


def print_report(report, workload, hardware, rho_max):
    print("===== WORKLOAD =====")
    for task in TASKS:
        print(f"{task:12s} {workload.arrival_rate[task]:.9g} req/s")
    print(f"\nStandalone capacity screening; rho_max={rho_max:g}.")
    print("===== FILTERING =====")
    for key, value in report.counts.items():
        print(f"{key.replace('_', ' '):20s}: {value}")
    print("\n===== FEASIBLE DEPLOYMENTS =====")
    print("Order: model count, then names; each routing is one feasible witness.")
    if not report.feasible_deployments:
        print("No feasible deployments under the supplied constraints.")
    for i, row in enumerate(report.feasible_deployments, 1):
        print(f"\n[{i}] {' + '.join(row.deployment.models)}")
        print(f"Memory: {row.memory_mib:g} MiB / {hardware.gpu_memory_mib:g} MiB")
        print("Feasible routing (nonzero rates; omitted pairs are zero):")
        for (task, model), rate in row.routing.routing_rps.items():
            if rate > 0:
                print(f"  {task:12s} -> {model:12s} {rate:.9g} req/s")
        if not workload.active_tasks:
            print("  No active tasks; all routing rates are zero.")
        print("Utilization:")
        for model, value in row.routing.utilization.items():
            print(f"  {model:12s} {value:.9g}")
    print("\n===== REJECTED DEPLOYMENTS =====")
    for row in report.rejected_deployments:
        print(f"{' + '.join(row.deployment.models)}: {row.reason}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", required=True, help="Comma-separated task=req/s; omitted tasks are zero")
    parser.add_argument("--rho-max", type=float, default=0.8)
    parser.add_argument("--gpu-memory-mib", type=float, help="Override measured single-GPU memory budget")
    parser.add_argument("--catalog-sources", type=Path, default=DEFAULT_SOURCES,
                        help="Source manifest; relative paths resolve against the manifest's directory")
    parser.add_argument("--output", type=Path, help="Save all feasible routings, rejections and profile provenance as JSON")
    args = parser.parse_args(argv)
    try:
        workload = parse_workload(args.workload)
        catalog = load_model_catalog(args.catalog_sources)
        hardware = load_hardware(args.catalog_sources)
        if args.gpu_memory_mib is not None:
            hardware = Hardware(args.gpu_memory_mib, hardware.name,
                                {"override": "--gpu-memory-mib", "measured_snapshot": hardware.source})
        report = screen_deployments(catalog, workload, hardware, args.rho_max)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            data = report_as_dict(report, catalog, workload, hardware, args.rho_max)
            args.output.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                   encoding="utf-8")
    except (ValueError, KeyError, TypeError, OSError, RoutingSolverError) as exc:
        parser.error(str(exc))
    print_report(report, workload, hardware, args.rho_max)
    if args.output:
        print(f"\nJSON saved to {args.output.resolve()}")
    # An empty feasible set is a valid result, not a solver failure.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
