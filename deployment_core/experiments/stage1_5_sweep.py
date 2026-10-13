"""Stage 1.5: workload/rho sweeps and independent feasibility-witness validation."""

import argparse
import itertools
import json
import math
from pathlib import Path

import scipy

from deployment_core import Workload, screen_deployments
from deployment_core.models import load_hardware, load_model_catalog
from deployment_core.routing_solver import LP_TOLERANCE, ROUTING_REL_TOLERANCE
from deployment_core.solve import report_as_dict
from deployment_core.workload import TASKS

BASE = dict(zip(TASKS, (10, 5, 3, 2, 1, 1)))
SCALES = (0.25, 0.50, 0.75, 1.00, 1.25, 1.50)
RHOS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)
COMPOSITIONS = {
    "balanced": (5, 4, 3, 2, 2, 4),
    "text-heavy": (14, 2, 1, 1, 1, 1),
    "image-heavy": (2, 13, 1, 1, 1, 2),
    "ocr-heavy": (2, 1, 13, 1, 1, 2),
    "asr-heavy": (2, 1, 1, 8, 1, 1),
    "vqa-heavy": (2, 2, 1, 1, 8, 1),
    "lid-heavy": (2, 1, 1, 1, 1, 14),
}
VALIDATIONS = ("routing_conservation", "capability", "capacity", "non_negativity")
DEFAULT_OUTPUT = Path(__file__).resolve().parents[2] / "results/stage1_5"


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def validate_routing(row, catalog, workload, rho_max, scenario):
    """Audit the actual returned witness; do not solve or repair it."""
    failures = []

    def fail(check, **details):
        failures.append({"check": check, "scenario": scenario,
                         "deployment": list(row.deployment.models), **details})

    deployed = set(row.deployment.models)
    routing = row.routing.routing_rps
    for (task, model), rate in routing.items():
        if not finite(rate) or rate < 0:
            fail("non_negativity", task=task, model=model, rate=rate)
        if (task not in workload.arrival_rate or model not in catalog
                or (rate != 0 and (model not in deployed or task not in catalog[model].supported_tasks))):
            fail("capability", task=task, model=model, rate=rate)
    for task, offered in workload.arrival_rate.items():
        values = [rate for (k, _), rate in routing.items() if k == task]
        total = math.fsum(values) if all(finite(v) for v in values) else float("nan")
        # Same test as Stage 1: relative tolerance, no absolute allowance for zero load.
        if not math.isclose(total, offered, rel_tol=ROUTING_REL_TOLERANCE, abs_tol=0.0):
            fail("routing_conservation", task=task, offered_rps=offered, routed_rps=total,
                 absolute_error=abs(total - offered))
    if set(row.routing.utilization) != deployed:
        fail("capacity", reason="utilization model keys differ from deployed models")
    for model in sorted(deployed):
        if model not in catalog:
            fail("capability", model=model, reason="unknown deployed model")
            continue
        terms = []
        invalid = False
        for (task, m), rate in routing.items():
            if m != model:
                continue
            if not finite(rate) or task not in catalog[model].service_time_ms:
                if rate != 0:
                    invalid = True
                continue
            terms.append(rate * catalog[model].service_time_ms[task] / 1000.0)
        recomputed = math.fsum(terms) if not invalid else float("nan")
        stored = row.routing.utilization.get(model)
        if (not finite(recomputed) or recomputed < 0 or recomputed > rho_max + LP_TOLERANCE
                or not finite(stored) or stored < 0 or stored > rho_max + LP_TOLERANCE
                or not math.isclose(stored, recomputed, rel_tol=0.0, abs_tol=LP_TOLERANCE)):
            fail("capacity", model=model, recomputed_utilization=recomputed,
                 stored_utilization=stored, rho_max=rho_max)
    return failures


def deployment_set(scenario):
    return {tuple(sorted(names)) for names in scenario["feasible_set"]}


def validate_monotonicity(scenarios, parameter, expanding):
    """Check inclusion for ALL parameter pairs, including equal-count set swaps."""
    failures = []
    complete = sorted((s for s in scenarios if s["completed"]), key=lambda s: s[parameter])
    for lower, higher in itertools.combinations(complete, 2):
        low, high = deployment_set(lower), deployment_set(higher)
        violations = low - high if expanding else high - low
        for deployment in sorted(violations):
            failures.append({
                "check": f"{parameter}_monotonicity", "message": "MONOTONICITY VIOLATION",
                "deployment": list(deployment),
                f"lower_{parameter}": lower[parameter], f"higher_{parameter}": higher[parameter],
                "lower_scenario": lower["name"], "higher_scenario": higher["name"],
            })
    return failures


def run_scenario(name, rates, rho_max, catalog, hardware, **parameters):
    workload = Workload(rates)
    row = {"name": name, "arrival_rate_rps": dict(workload.arrival_rate),
           "total_rps": math.fsum(workload.arrival_rate.values()), "rho_max": rho_max,
           **parameters, "completed": False, "counts": None, "feasible_set": [],
           "feasible_deployments": [], "rejected_deployments": [], "failures": []}
    try:
        # This is the same public Stage 1 engine used by generate_feasible_deployments;
        # its report also supplies stage counts without running every LP a second time.
        report = screen_deployments(catalog, workload, hardware, rho_max)
        serialized = report_as_dict(report, catalog, workload, hardware, rho_max)
        row.update(completed=True, counts=report.counts,
                   feasible_set=[list(d.deployment.models) for d in report.feasible_deployments],
                   feasible_deployments=serialized["feasible_deployments"],
                   rejected_deployments=serialized["rejected_deployments"])
        for deployment in report.feasible_deployments:
            row["failures"].extend(validate_routing(deployment, catalog, workload, rho_max, name))
            if workload.arrival_rate["visual_qa"] > 0 and "qwen" not in deployment.deployment.models:
                row["failures"].append({"check": "vqa_coverage", "scenario": name,
                                        "deployment": list(deployment.deployment.models)})
    except Exception as exc:
        # Preserve the failed scenario and continue collecting evidence; never call it infeasible.
        row["completed"] = False
        row["failures"].append({"check": "execution", "scenario": name,
                                "error_type": type(exc).__name__, "error": str(exc)})
    return row


def composition_observations(scenarios):
    observations = {"distinct_feasible_sets": len({frozenset(deployment_set(s)) for s in scenarios
                                                   if s["completed"]}), "specialists": {}}
    for name, specialist in (("asr-heavy", "whisper"), ("text-heavy", "distilbert"),
                             ("image-heavy", "resnet50"), ("ocr-heavy", "paddleocr")):
        row = next(s for s in scenarios if s["name"] == name)
        feasible = deployment_set(row)
        capacity_rejected = {tuple(r["models"]) for r in row["rejected_deployments"]
                             if r["stage"] == "capacity"}
        rescued = []
        for models in sorted(feasible):
            without = tuple(m for m in models if m != specialist)
            if specialist in models and without in capacity_rejected:
                rescued.append({"without_specialist": list(without), "with_specialist": list(models)})
        observations["specialists"][name] = {
            "specialist": specialist, "completed": row["completed"],
            "feasible_with": sum(specialist in d for d in feasible),
            "feasible_without": sum(specialist not in d for d in feasible),
            "capacity_rejected_without_to_feasible_with": rescued,
            "interpretation": ("No feasible deployment at this workload; the sweep cannot demonstrate "
                               "a feasibility gain from this specialist here." if not feasible else
                               "Matched pairs describe feasibility changes when adding the specialist. "
                               "They do not imply an optimal routing or a minimum Qwen utilization."),
        }
    return observations


def safe_json(value):
    """Keep invalid numerical evidence readable in a standards-compliant failure report."""
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: safe_json(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [safe_json(item) for item in value]
    return value


def render_markdown(groups, summary):
    def names(row):
        return "; ".join(" + ".join(d) for d in row["feasible_set"]) or ("∅" if row["completed"] else "ERROR")

    def count(row, key):
        return row["counts"][key] if row["counts"] is not None else "ERROR"

    lines = ["# Stage 1.5 Feasibility Sanity Sweep", "",
             "仅验证 Stage 1 可行集合及 routing；所有组合按模型数量和名称展示。",
             "A/B 使用 rho_max=0.8；C 使用固定基准 workload。显存预算来自现有 snapshot。",
             "完整 arrival rates、routing、utilization、淘汰原因保存在对应 JSON 中。", "",
             "## Table 1：Load Intensity", "",
             "Scale | Total RPS | Coverage | Memory | Capacity | Feasible Set",
             "--- | ---: | ---: | ---: | ---: | ---"]
    for row in groups["load"]:
        lines.append(f"{row['scale']:.2f} | {row['total_rps']:g} | {count(row, 'coverage_feasible')} | "
                     f"{count(row, 'memory_feasible')} | {count(row, 'capacity_feasible')} | {names(row)}")
    lines += ["", "## Table 2：Composition", "",
              "Workload | Total RPS | Capacity Feasible | Feasible Deployments", "--- | ---: | ---: | ---"]
    for row in groups["composition"]:
        lines.append(f"{row['name']} | {row['total_rps']:g} | {count(row, 'capacity_feasible')} | {names(row)}")
    lines += ["", "## Table 3：rho_max", "", "rho_max | Capacity Feasible | Feasible Deployments",
              "--- | ---: | ---"]
    for row in groups["rho"]:
        lines.append(f"{row['rho_max']:.2f} | {count(row, 'capacity_feasible')} | {names(row)}")
    lines += ["", "## Composition observations", "",
              f"本次 sweep 得到 {summary['composition_observations']['distinct_feasible_sets']} 种不同 feasible set。",
              f"VQA 正负载时包含 Qwen 的检查：{'PASS' if summary['vqa_coverage_passed'] else 'FAIL'}。",
              "空集合中的该性质仅为真空成立，不表示存在可服务组合。", ""]
    for name, obs in summary["composition_observations"]["specialists"].items():
        lines.append(f"- {name} / {obs['specialist']}：含该模型的可行组合 {obs['feasible_with']} 个，"
                     f"不含的 {obs['feasible_without']} 个；仅添加该模型就从 Capacity 淘汰变为可行的配对 "
                     f"{len(obs['capacity_rejected_without_to_feasible_with'])} 个。")
        if not obs["feasible_with"] and not obs["feasible_without"]:
            lines.append("  当前负载无可行组合，不能由此证明该 specialist 在本场景恢复可行性。")
    lines += ["", "配对结果检验 specialist 对共享 Qwen capacity 约束的作用；零目标 LP 的 witness "
              "不保证 Qwen 利用率最小，因此不从单个 witness 推断最优卸载量。",
              "这些结论仅适用于 standalone service-demand screening，不代表真实共置性能。", "",
              "## Validation", ""]
    for label, key in (("Load monotonicity", "load_monotonicity_passed"),
                       ("rho monotonicity", "rho_monotonicity_passed"),
                       ("Routing conservation", "routing_conservation_passed"),
                       ("Capability validation", "capability_passed"),
                       ("Capacity validation", "capacity_passed"),
                       ("Non-negativity", "non_negativity_passed")):
        lines.append(f"{label}: {'PASS' if summary[key] else 'FAIL'}  ")
    lines += ["", f"Scenarios: {summary['num_scenarios']}; routing witnesses: {summary['num_routing_witnesses']}; "
              f"failures: {summary['num_failures']}.", "",
              "集合单调性检查覆盖 A/C 各自所有 15 对参数；不是仅比较可行组合数量。",
              "routing 守恒复用相对容差 1e-7（零到达率不加绝对余量）；capacity 容差 1e-9。"]
    if summary["failures"]:
        lines += ["", "## Violations", "", "```json", json.dumps(safe_json(summary["failures"]),
                   ensure_ascii=False, indent=2), "```"]
    lines += ["", "STAGE 1.5 VALIDATION: " + ("PASS" if summary["passed"] else "FAIL"), ""]
    return "\n".join(lines)


def run_sweep(output_dir=DEFAULT_OUTPUT):
    output_dir = Path(output_dir)
    catalog, hardware = load_model_catalog(), load_hardware()
    groups = {
        "load": [run_scenario(f"load_{scale:.2f}", {t: r * scale for t, r in BASE.items()},
                              0.8, catalog, hardware, scale=scale) for scale in SCALES],
        "composition": [run_scenario(name, dict(zip(TASKS, rates)), 0.8, catalog, hardware)
                        for name, rates in COMPOSITIONS.items()],
        "rho": [run_scenario(f"rho_{rho:.2f}", BASE, rho, catalog, hardware) for rho in RHOS],
    }
    scenarios = [row for group in groups.values() for row in group]
    failures = [failure for row in scenarios for failure in row["failures"]]
    load_failures = validate_monotonicity(groups["load"], "scale", expanding=False)
    rho_failures = validate_monotonicity(groups["rho"], "rho_max", expanding=True)
    failures += load_failures + rho_failures
    complete = all(row["completed"] for row in scenarios)
    summary = {
        "schema_version": 1, "scipy_version": scipy.__version__,
        "load_monotonicity_passed": all(r["completed"] for r in groups["load"]) and not load_failures,
        "rho_monotonicity_passed": all(r["completed"] for r in groups["rho"]) and not rho_failures,
        **{f"{key}_passed": complete and not any(f["check"] == key for f in failures) for key in VALIDATIONS},
        "vqa_coverage_passed": complete and not any(f["check"] == "vqa_coverage" for f in failures),
        "num_scenarios": len(scenarios), "num_completed_scenarios": sum(r["completed"] for r in scenarios),
        "num_routing_witnesses": sum(len(row["feasible_deployments"]) for row in scenarios),
        "num_failures": len(failures), "failures": failures,
        "composition_observations": composition_observations(groups["composition"]),
    }
    summary["routing_validation_passed"] = all(summary[f"{key}_passed"] for key in VALIDATIONS)
    summary["passed"] = complete and not failures
    # Share the existing serializer for profile provenance; no feasibility logic is duplicated.
    from deployment_core.candidate_generator import ScreeningReport
    provenance = report_as_dict(ScreeningReport(), catalog, Workload(BASE), hardware, 0.8)
    summary["profile_context"] = {key: provenance[key] for key in
                                  ("model_catalog", "hardware", "numerical_tolerances")}
    output_dir.mkdir(parents=True, exist_ok=True)
    for group, filename in (("load", "load_intensity_sweep.json"),
                            ("composition", "composition_sweep.json"), ("rho", "rho_sweep.json")):
        data = {"schema_version": 1, "scenarios": groups[group]}
        (output_dir / filename).write_text(json.dumps(safe_json(data), ensure_ascii=False,
                                                     indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "stage1_5_summary.json").write_text(json.dumps(safe_json(summary), ensure_ascii=False,
                                                               indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output_dir / "summary.md").write_text(render_markdown(groups, summary), encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        summary = run_sweep(args.output_dir)
    except Exception as exc:
        print(f"STAGE 1.5 VALIDATION: FAIL\n{type(exc).__name__}: {exc}")
        return 1
    for failure in summary["failures"]:
        print(json.dumps(safe_json(failure), ensure_ascii=False, allow_nan=False))
    print(f"Scenarios: {summary['num_scenarios']}; routing witnesses: {summary['num_routing_witnesses']}; "
          f"failures: {summary['num_failures']}")
    print(f"Results: {args.output_dir.resolve()}")
    print("STAGE 1.5 VALIDATION: " + ("PASS" if summary["passed"] else "FAIL"))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
