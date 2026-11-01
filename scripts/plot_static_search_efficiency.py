"""Strictly audit an existing CPU solver comparison and export efficiency figures."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
import types

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ALGORITHMS = ("milp", "dfs", "plain_dfs")
LABELS = ("MILP", "Bounded DFS", "Plain DFS")
COLORS = ("#2563a6", "#319881", "#d07b32")
SCOPE = "Five-model n0/n1, K4 finite calibrated predictor domain; CPU search, no physical GPU optimality claim."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(report_path, config_path):
    """Recompute the upstream contract and summaries; never run a solver or GPU."""
    report_path, config_path = Path(report_path).resolve(), Path(config_path).resolve()
    report = read(report_path)
    preflight = report["preflight"]
    root = Path(preflight["source_root"])
    receipts = [{"path": str(report_path), "sha256": sha(report_path)},
                {"path": str(config_path), "sha256": sha(config_path)}]
    require(read(report_path.parent / "preflight.json") == preflight, "Preflight file differs from report")
    receipts.append({"path": str(report_path.parent / "preflight.json"),
                     "sha256": sha(report_path.parent / "preflight.json")})
    current = {str(p.relative_to(root)): sha(p) for p in sorted((root / "gsdeploy").rglob("*.py"))}
    require(current == report["source_before"] == report["source_after"], "Frozen mathematical source changed")
    benchmark = root / "scripts/compare_calibrated_pool_solvers.py"
    require(sha(benchmark) == report["benchmark_source_before"] == report["benchmark_source_after"],
            "Frozen benchmark entry changed")
    receipts.append({"path": str(benchmark), "sha256": sha(benchmark)})
    wrapper = root / "gsdeploy/search_session.py"
    require(report["session_enabled"] and sha(wrapper) == report["session_wrapper_sha256"],
            "Session wrapper changed or session disabled")
    spec = importlib.util.spec_from_file_location("frozen_efficiency_contract", benchmark)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    args = types.SimpleNamespace(source_root=root, config=config_path,
        models=list(preflight["replica_options"]), route_grid=preflight["domain"]["grid"],
        time_limit=preflight["fairness"]["time_limit_seconds"])
    config, original_hash, bounds = module.prepare(args)
    require(Path(sys.modules["gsdeploy"].__file__).resolve().parent == root / "gsdeploy",
            "Existing Python module cache does not belong to the frozen source; use a fresh process")
    # prepare imports the explicitly frozen package before any mathematical helper.
    from gsdeploy.trace import make_trace, trace_digest
    require(original_hash == preflight["original_problem_sha256"], "Original config digest changed")
    require(canonical(config.as_dict()) == preflight["effective_problem_sha256"], "Effective config digest changed")
    require(module.audit_domain(config, args.models, args.route_grid, bounds) == preflight["domain"],
            "Recomputed finite domain differs")
    trace = make_trace(config.workload, config.horizon_seconds, config.random_seed, config.arrival_process)
    require(trace_digest(trace) == preflight["trace_sha256"] and len(trace) == preflight["offered_requests"],
            "Recomputed offered trace differs")
    require(bounds.spec["version_sha256"] == preflight["predictor_version"], "Predictor version differs")
    require(args.route_grid == 4 and set(args.models) == {"generalist_2", "text", "image", "ocr", "asr"}
            and all(v == [1] for v in preflight["replica_options"].values()), "Unexpected figure domain")
    fairness = dict(time_limit_seconds=args.time_limit, search_max_states=config.search_max_states,
        milp_max_iterations=config.milp_max_iterations, milp_max_columns=config.milp_max_columns,
        absolute_gap=config.milp_absolute_gap, policy=config.search_policy, capacity_witness=False,
        task_replica_profiles_retained=True, baseline_count_rows_retained=True)
    require(fairness == preflight["fairness"], "Recomputed fairness differs")
    records, identities, scores = report["records"], set(), set()
    repeats = sorted({row["repeat"] for row in records})
    require(repeats == list(range(5)) and len(records) == 15, "Expected five complete three-algorithm rounds")
    expected_order = [(r, a) for r in repeats for a in (ALGORITHMS if r % 2 == 0 else ALGORITHMS[::-1])]
    require([(r["repeat"], r["algorithm"]) for r in records] == expected_order, "Interleaved roster differs")
    common = ("original_problem_sha256", "effective_problem_sha256", "predictor_schema", "predictor_version",
              "source_root", "domain", "replica_options", "trace_sha256", "offered_requests", "fairness")
    recomputed_summaries = []
    for row in records:
        path = report_path.parent / f"run-{row['repeat']}-{row['algorithm']}.json"
        require(read(path) == row, "Raw run file differs from embedded record: " + path.name)
        receipts.append({"path": str(path), "sha256": sha(path)})
        require(all(row[k] == preflight[k] for k in common), "Algorithm input/fairness differs")
        require(row["wrapper_sha256"] == sha(wrapper) and Path(row["wrapper_path"]) == wrapper,
                "Algorithm session wrapper differs")
        require(row["session_cache"]["digest_hits"] > 0, "Session memo not observed")
        result, winner = row["result"], row["result"]["winner"]
        require(result["optimality_certified"] and result["search_completed"] and winner,
                "Incomplete finite-domain certificate")
        require(result["optimality_gap"] == 0 and not result["unresolved_candidates"]
                and result["metrics"]["unknown_candidates"] == 0, "Unresolved or nonzero-gap result")
        require(result["trace_sha256"] == preflight["trace_sha256"]
                and result["domain"]["K"] == args.route_grid
                and result["domain"]["continuous_optimality"] is False
                and result["domain"]["surrogate_bounds_version"] == preflight["predictor_version"],
                "Certificate domain differs")
        require(winner["score"] == result["utility_upper_bound"] == winner["prediction"]["U"],
                "Winner, prediction and proven bound differ")
        if row["algorithm"] == "plain_dfs":
            require(result["domain"]["utility_bound_pruning"] is False
                    and result["metrics"]["utility_bound_evaluations"] == 0
                    and result["metrics"]["bound_pruned"] == 0, "Plain DFS used a utility bound")
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in
                    ("search_wall_seconds", "parent_process_wall_seconds", "peak_rss_kib")), "Invalid timing/RSS")
        require(row["parent_process_wall_seconds"] >= row["search_wall_seconds"], "Process wall shorter than search")
        require(row["gpu_started"] is False and row["cpu_affinity"] == records[0]["cpu_affinity"],
                "GPU activity flag or CPU affinity differs")
        identities.add(canonical({k: winner[k] for k in ("deployment", "routing")}))
        scores.add(winner["score"].hex())
    require(len(identities) == len(scores) == 1, "Algorithms/repeats chose different optima")
    for algorithm in ALGORITHMS:
        rows = [row for row in records if row["algorithm"] == algorithm]
        require(len({row["result"]["metrics"]["full_prediction_calls"] for row in rows}) == 1,
                "Prediction counts vary across repeats")
        recomputed_summaries.append(dict(algorithm=algorithm, completed_processes=len(rows),
            certificates=sum(row["result"]["optimality_certified"] for row in rows),
            median_search_wall_seconds=statistics.median(row["search_wall_seconds"] for row in rows),
            median_parent_wall_seconds=statistics.median(row["parent_process_wall_seconds"] for row in rows),
            median_full_predictions=statistics.median(row["result"]["metrics"]["full_prediction_calls"] for row in rows),
            max_peak_rss_kib=max(row["peak_rss_kib"] for row in rows)))
    require(recomputed_summaries == report["summaries"], "Upstream summary arithmetic differs")
    expected_repeat = [dict(repeat=r, objective_agrees=True, all_certified=True,
                            certified_objective_agrees=True, plan_identity_agrees=True) for r in repeats]
    require(report["per_repeat"] == expected_repeat, "Upstream agreement arithmetic differs")
    require(all(report[k] is True for k in ("source_unchanged", "session_wrapper_unchanged", "config_unchanged",
        "same_contract", "benchmark_complete")) and report["gpu_started"] is False, "Upstream checks not complete")
    ratios = []
    for baseline in ("dfs", "plain_dfs"):
        for field in ("search_wall_seconds", "parent_process_wall_seconds"):
            paired = [next(row[field] for row in records if row["repeat"] == r and row["algorithm"] == baseline)
                      / next(row[field] for row in records if row["repeat"] == r and row["algorithm"] == "milp")
                      for r in repeats]
            groups = {a: [row[field] for row in records if row["algorithm"] == a] for a in (baseline, "milp")}
            ratios.append(dict(baseline=baseline, faster_algorithm="milp", timing_field=field,
                ratio_of_medians=statistics.median(groups[baseline]) / statistics.median(groups["milp"]),
                paired_ratios=paired, median_paired_ratio=statistics.median(paired),
                min_paired_ratio=min(paired), max_paired_ratio=max(paired)))
    require(all(sha(row["path"]) == row["sha256"] for row in receipts), "Input changed during audit")
    counts = {a: {key: [row["result"]["metrics"].get(key) for row in records if row["algorithm"] == a]
                  for key in ("full_prediction_calls", "search_states", "utility_bound_evaluations", "bound_pruned", "milp_calls")}
              for a in ALGORITHMS}
    return report, dict(schema="gsdeploy.cpu_efficiency_independent_audit.v1", status="PASS", source_root=str(root),
        input_receipts=receipts, mathematical_source_receipts=current, recomputed_summaries=recomputed_summaries,
        speed_ratios=ratios, evaluation_counts=counts,
        prediction_call_reduction_vs_milp={a: counts[a]["full_prediction_calls"][0] / counts["milp"]["full_prediction_calls"][0]
                                          for a in ("dfs", "plain_dfs")},
        identical_optimum=True, plan_sha256=next(iter(identities)),
        utility=float.fromhex(next(iter(scores))), certificates=15, repeats=5, gpu_started=False,
        scope=SCOPE, limitations="Single host; five repeated serial processes; elapsed time is observational; no confidence-interval or full seven-model/replica-domain claim.")


def plot(report_path, config_path, output):
    report, checked = audit(report_path, config_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
    banner = f"Same certified optimum in all 15 runs: U = {checked['utility']:.9f} | 5 repeats"
    files = []

    def save(fig, name):
        fig.suptitle(banner, fontsize=11)
        fig.text(.5, .015, SCOPE, ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .06, 1, .93))
        for ext in ("png", "svg", "pdf"):
            path = output / f"{name}.{ext}"
            fig.savefig(path, dpi=220, metadata={"Creator": "Receipt-verified CPU benchmark exporter"})
            files.append(path)
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, field, title in zip(axes, ("search_wall_seconds", "parent_process_wall_seconds"),
                                ("Session construction + solve", "Fresh process wall time")):
        for i, algorithm in enumerate(ALGORITHMS):
            vals = [row[field] for row in report["records"] if row["algorithm"] == algorithm]
            med = statistics.median(vals)
            ax.bar(i, med, width=.55, color=COLORS[i], alpha=.7)
            ax.scatter(i + np.linspace(-.12, .12, len(vals)), vals, s=22, color=COLORS[i], edgecolor="white", linewidth=.4)
            ax.text(i, max(vals) * 1.15, f"{med:.3f} s", ha="center", fontsize=9)
        ax.set(yscale="log", ylabel="Seconds (log scale)", title=title, xticks=range(3), xticklabels=LABELS)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.5)
        ax.grid(axis="y", alpha=.2)
    save(fig, "cpu-search-wall-time")

    fig, ax = plt.subplots(figsize=(7.7, 4.3))
    counts = [s["median_full_predictions"] for s in checked["recomputed_summaries"]]
    ax.bar(range(3), counts, width=.55, color=COLORS)
    for i, count in enumerate(counts):
        ax.text(i, count * 1.18, f"{count:g}", ha="center")
    ax.set(yscale="log", ylabel="Complete three-scenario prediction calls (log scale)",
           xticks=range(3), xticklabels=LABELS, title="Median complete evaluations; identical counts across all five repeats")
    ax.set_ylim(.7, max(counts) * 3)
    ax.grid(axis="y", alpha=.2)
    save(fig, "cpu-full-prediction-calls")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    for ax, field, title in zip(axes, ("search_wall_seconds", "parent_process_wall_seconds"),
                                ("Session + solve speed ratio", "Fresh process speed ratio")):
        for i, baseline in enumerate(("dfs", "plain_dfs")):
            row = next(r for r in checked["speed_ratios"] if r["baseline"] == baseline and r["timing_field"] == field)
            vals = row["paired_ratios"]
            ax.scatter(i + np.linspace(-.10, .10, len(vals)), vals, s=28, color=COLORS[i + 1])
            ax.hlines(row["ratio_of_medians"], i - .2, i + .2, color="black", linewidth=2)
            ax.text(i, max(vals) * 1.15, f"{row['ratio_of_medians']:.2f}×", ha="center")
        ax.axhline(1, color="gray", linestyle="--", linewidth=1)
        ax.set(yscale="log", ylabel="Baseline / MILP time (log scale)", title=title,
               xticks=(0, 1), xticklabels=("Bounded DFS / MILP", "Plain DFS / MILP"))
        ax.set_ylim(.7, ax.get_ylim()[1] * 1.6)
        ax.grid(axis="y", alpha=.2)
    fig.text(.5, .065, "Points: paired repeats. Black line and label: ratio of median times.", ha="center", fontsize=8)
    save(fig, "cpu-milp-speed-ratio")
    (output / "audit-report.json").write_text(json.dumps(checked, indent=2, allow_nan=False) + "\n")
    files.append(output / "audit-report.json")
    manifest = dict(schema="gsdeploy.cpu_efficiency_figures.v1", input_receipts=checked["input_receipts"],
        identical_optimum=True, utility=checked["utility"], exporter={"path": str(Path(__file__).resolve()),
        "sha256": sha(__file__)}, outputs={p.name: sha(p) for p in files}, scope=SCOPE)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return checked


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = plot(args.report, args.config, args.output)
    print(json.dumps({key: result[key] for key in ("status", "identical_optimum", "utility", "certificates", "speed_ratios")}))
