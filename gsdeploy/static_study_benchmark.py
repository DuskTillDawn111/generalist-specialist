"""Cold-process CPU correctness/efficiency study; synthetic evidence only.

Run ``python -m gsdeploy.static_study_benchmark --output artifacts/static_cpu_benchmark``.
Five repeats are the default. Each solver invocation gets an isolated process,
the same frozen six-term problem and its own state/column/plan caches. Fixture
generation and library import are excluded; all search/model construction is
included. Peak RSS includes the interpreter, libraries and frozen evidence.
"""

import argparse
from dataclasses import replace
import hashlib
import itertools
import json
import os
from pathlib import Path
import resource
import statistics
import subprocess
import sys
import time

from .config import SystemConfig
from .energy import energy_key
from .route_domain import fraction_rows


CASES = (
    dict(name="tiny", models=2, tasks=1, replicas=1, K=2),
    dict(name="small", models=3, tasks=2, replicas=2, K=3),
    dict(name="medium", models=4, tasks=2, replicas=2, K=4),
)


def grid_plans(config):
    """Cartesian plans using the same route domain, without solver columns/bounds."""
    models = sorted(config.catalog["models"])
    for counts in itertools.product(*(
        sorted(map(int, config.catalog["capacity_factors"][m])) for m in models
    )):
        deployment = {m: n for m, n in zip(models, counts) if n}
        if not deployment:
            continue
        if sum(config.catalog["memory_table_mib"][m][str(n)]
               for m, n in deployment.items()) > config.gpu_memory_mib:
            continue
        options = []
        for task, rate in sorted(config.workload.items()):
            supported = [m for m in deployment
                         if task in config.catalog["models"][m]["supported_tasks"]]
            options.append([
                {m: rate * f.numerator / f.denominator for m, f in row}
                for row in fraction_rows(supported, config.route_grid,
                    config.route_pair_denominators_by_task.get(task, ()))
            ])
        for rows in itertools.product(*options):
            yield deployment, dict(zip(sorted(config.workload), rows))


def make_problem(case, time_limit=10.0):
    tasks = ["text", "image"][:case["tasks"]]
    # Fast but less accurate generalists and slower, more accurate specialists.
    specs = [
        ("g0", "generalist", tasks, 55.0, 0.72),
        ("s0", "specialist", [tasks[0]], 72.0, 0.97),
        ("s1", "specialist", [tasks[-1]], 65.0, 0.96),
        ("g1", "generalist", tasks, 43.0, 0.79),
    ][:case["models"]]
    catalog = dict(
        safe_factor=1.0, capability_contract="verified_executable_v1",
        performance_mode="independent", models={}, capacity_factors={},
        memory_table_mib={}, quality=dict(
            schema_version=1, mode="aggregate_proxy", paths={},
            identity={k: "synthetic CPU benchmark fixture" for k in (
                "input_pool_sha256", "reference_sha256", "protocol_sha256", "source")},
        ),
    )
    for index, (model, role, supported, service, accuracy) in enumerate(specs):
        catalog["models"][model] = dict(
            role=role, supported_tasks=supported, executable_tasks=supported,
            service_time_ms={t: service * (1.0 + 0.1 * tasks.index(t)) for t in supported},
        )
        catalog["capacity_factors"][model] = {
            str(n): (0.0 if n == 0 else 1.0 if n == 1 else 1.65)
            for n in range(case["replicas"] + 1)
        }
        catalog["memory_table_mib"][model] = {
            str(n): n * (4.0 + index) for n in range(case["replicas"] + 1)
        }
        catalog["quality"]["paths"][model] = {
            t: dict(metric="em" if t == "text" else "accuracy", unit="fraction",
                    value=accuracy, checkpoint_sha256="synthetic checkpoint",
                    source="invented CPU benchmark values, no GPU evidence")
            for t in supported
        }
    config = SystemConfig(
        catalog=catalog, workload={t: 5.0 for t in tasks}, gpu_memory_mib=80.0,
        latency_reference_ms={t: 200.0 for t in tasks},
        weights={k: 1 / 6 for k in "QTLPME"}, utility_schema="six_term_quality_v1",
        horizon_seconds=4.0, arrival_process="poisson", random_seed=42,
        route_grid=case["K"], milp_capacity_witness=False,
        route_pair_denominators_by_task=case.get("route_pair_denominators_by_task", {}),
        milp_time_limit_seconds=time_limit, milp_max_iterations=10000,
        milp_max_columns=100000, search_max_states=10_000_000,
    )
    profiles, plan_count = {}, 0
    for deployment, routing in grid_plans(config):
        plan_count += 1
        # This is a complete whole-board fixture. The explicit interaction term
        # forbids treating the energy as a sum of independent model energies.
        active = sum(n for n in deployment.values())
        split_tasks = sum(len(row) > 1 for row in routing.values())
        watts = 35.0 + 12.0 * active + 3.0 * active * active + 8.0 * split_tasks
        key = energy_key(deployment, routing, config.workload,
                         config.horizon_seconds, config.random_seed,
                         config.arrival_process, catalog=config.catalog)
        profiles[key] = dict(
            kind="synthetic_fixture", scope="whole_gpu_board", coverage_valid=True,
            run_joules=watts * config.horizon_seconds, startup_joules=2.0 * active,
            provenance="invented whole-board CPU benchmark fixture; not GPU evidence",
        )
    catalog["energy_profiles"] = profiles
    return config, plan_count


def worker(problem, solver):
    # Import both backends before timing. The problem and solver caches remain
    # cold; repeated timing does not pay asymmetric numerical-library imports.
    from .milp import JointMilpOptimizer  # noqa: F401
    from .optimizer import UtilityOptimizer  # noqa: F401
    from .system import DeploymentSystem

    config = replace(SystemConfig.load(problem), solver=solver)
    baseline_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    start = time.perf_counter()
    result = DeploymentSystem(config).plan()
    elapsed = time.perf_counter() - start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    winner = result.get("winner")
    return dict(
        solver=solver, status=result["status"],
        search_completed=result["search_completed"],
        optimality_certified=result["optimality_certified"],
        U=winner["score"] if winner else None,
        utility_upper_bound=result["utility_upper_bound"],
        optimality_gap=result["optimality_gap"],
        wall_seconds_including_model_build=elapsed,
        solver_elapsed_seconds=result["elapsed_seconds"],
        peak_rss_mib=peak_rss, baseline_peak_rss_mib=baseline_rss,
        peak_rss_growth_mib=max(0.0, peak_rss - baseline_rss),
        metrics=result["metrics"],
        domain=result["domain"],
        deployment=winner["deployment"] if winner else None,
        routing=winner["routing"] if winner else None,
    )


def run_benchmark(output, repeats=5, time_limit=10.0, cases=CASES):
    output = Path(output).resolve()
    fixtures = output / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    report = dict(
        schema_version=1, evidence_kind="synthetic_cpu_only", repeats=repeats,
        per_solver_time_limit_seconds=time_limit, cases=[],
        protocol=dict(
            cache="fresh child process and fresh optimizer per invocation",
            timing="includes trace, model construction and all scoring; excludes fixture generation and library import",
            memory="Linux process peak RSS includes interpreter, libraries and frozen evidence; growth is relative to pre-search high-water mark",
            energy="complete nominal whole-board synthetic evidence over declared domain",
            correctness="same finite replica and rational route grids; compare objective, allow tied plans",
            comparison="no prior assertion of MILP speed advantage",
        ),
        code_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                     for name in ["optimizer.py", "milp.py", "prediction.py", "quality.py",
                                  "performance.py", "system.py", "static_study_benchmark.py"]},
    )
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for case in cases:
        config, plan_count = make_problem(case, time_limit)
        problem = fixtures / f"{case['name']}.json"
        problem.write_text(json.dumps(config.as_dict(), sort_keys=True, indent=2) + "\n")
        entry = dict(**case, domain_plans_before_capacity_pruning=plan_count,
                     problem_sha256=hashlib.sha256(problem.read_bytes()).hexdigest(), runs=[])
        for repeat in range(repeats):
            for solver in (["milp", "dfs"] if repeat % 2 == 0 else ["dfs", "milp"]):
                process_start = time.perf_counter()
                completed = subprocess.run(
                    [sys.executable, "-m", "gsdeploy.static_study_benchmark",
                     "--worker", str(problem), "--solver", solver],
                    check=True, text=True, capture_output=True, env=env,
                    timeout=time_limit + 60,
                )
                result = json.loads(completed.stdout)
                result["subprocess_wall_seconds_including_startup_and_imports"] = time.perf_counter() - process_start
                entry["runs"].append(dict(repeat=repeat + 1, **result))
                print(f"{case['name']} repeat={repeat + 1} {solver}: "
                      f"{result['status']} U={result['U']} "
                      f"wall={result['wall_seconds_including_model_build']:.4f}s", flush=True)
        entry["summary"] = {}
        for solver in ["milp", "dfs"]:
            runs = [r for r in entry["runs"] if r["solver"] == solver]
            entry["summary"][solver] = dict(
                all_certified=all(r["optimality_certified"] for r in runs),
                median_wall_seconds=statistics.median(r["wall_seconds_including_model_build"] for r in runs),
                median_full_prediction_calls=statistics.median(r["metrics"]["full_prediction_calls"] for r in runs),
                median_peak_rss_mib=statistics.median(r["peak_rss_mib"] for r in runs),
            )
        values = [r["U"] for r in entry["runs"]]
        entry["all_observed_objectives_agree"] = (
            all(v is not None for v in values) and max(values) - min(values) <= 1e-6
        )
        entry["all_runs_optimal_objectives_agree"] = (
            all(r["optimality_certified"] for r in entry["runs"])
            and entry["all_observed_objectives_agree"]
        )
        entry["dfs_over_milp_median_time"] = (
            entry["summary"]["dfs"]["median_wall_seconds"]
            / entry["summary"]["milp"]["median_wall_seconds"]
        )
        report["cases"].append(entry)
        (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report["code_sha256_after"] = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in report["code_sha256"]
    }
    report["code_unchanged_during_benchmark"] = report["code_sha256"] == report["code_sha256_after"]
    (output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    lines = ["# Synthetic CPU finite-domain benchmark", "",
             "All service, quality and energy values are synthetic; these are algorithm experiments, not hardware conclusions.",
             "Each invocation starts in a new process. Search wall time includes model construction. Library imports and fixture generation are excluded.", "",
             "| Case | Plans | Solver | All certified | Median seconds | Median full evaluations | Median peak RSS MiB |",
             "|---|---:|---|---|---:|---:|---:|"]
    for case in report["cases"]:
        for solver, summary in case["summary"].items():
            lines.append(f"| {case['name']} | {case['domain_plans_before_capacity_pruning']} | {solver} | "
                         f"{summary['all_certified']} | {summary['median_wall_seconds']:.6f} | "
                         f"{summary['median_full_prediction_calls']} | {summary['median_peak_rss_mib']:.2f} |")
    lines += ["", "See report.json for every repeat, statuses, scores, upper bounds, gaps, and code/problem hashes.", ""]
    (output / "report.md").write_text("\n".join(lines))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/static_cpu_benchmark")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--worker")
    parser.add_argument("--solver", choices=["milp", "dfs"])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(worker(args.worker, args.solver)))
    else:
        if args.repeats < 1 or args.time_limit <= 0:
            parser.error("repeats and time-limit must be positive")
        run_benchmark(args.output, args.repeats, args.time_limit)


if __name__ == "__main__":
    main()
