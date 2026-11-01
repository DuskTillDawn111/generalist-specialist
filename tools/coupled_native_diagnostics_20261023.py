"""Read-only endpoint decomposition for the frozen coupled native study."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


TERMS = "QTLPME"


def receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def quantile(values, q):
    if not values:
        return None
    values = sorted(values)
    position = (len(values) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def summarize(raw, path):
    rows = raw.get("request_records", [])
    utility = raw.get("evaluation", {}).get("utility", {})
    counts, gates, reasons, exposure = Counter(), Counter(), Counter(), []
    compute_ms, pending = [], []
    for row in rows:
        routing = row.get("routing_decision", {})
        decision = routing.get("decision", {})
        family = raw.get("replica_to_family", {}).get(row.get("model"))
        counts[f"{row['task']}->{family}"] += 1
        gates.update(decision.get("gates", []))
        if routing.get("decision_compute_ns") is not None:
            compute_ms.append(routing["decision_compute_ns"] / 1e6)
        budget = decision.get("forecast_computation_budget", {})
        if budget.get("pending_count") is not None:
            pending.append(budget["pending_count"])
        for entry in decision.get("audit", []):
            reasons.update(entry.get("unknown_reasons", []))
        if row["task"] == "text" and family == "generalist_2":
            entry = next((e for e in decision.get("audit", []) if e.get("alias") == row["model"]), {})
            exposure.append(dict(request_id=row["request_id"], pool_id=row.get("pool_id"),
                                 scheduled_offset_seconds=row.get("scheduled_offset_seconds"),
                                 predicted_delta_cost=entry.get("scenario_delta_cost_vs_selected_specialist"),
                                 trigger=decision.get("trigger")))
    cohort = [(r.get("arrival_index"), r["task"], r.get("pool_id"), r.get("scheduled_offset_seconds")) for r in rows]
    return dict(raw=receipt(path), status=raw.get("status"), errors=raw.get("errors", []),
                utility=utility, stability=raw.get("evaluation", {}).get("stability", {}),
                causal_verified=raw.get("evaluation", {}).get("dynamic_audit", {}).get("verified"),
                routing_counts=dict(counts), gates=dict(gates), forecast_unknown_reasons=dict(reasons),
                text_generalist_exposure=exposure,
                offered_cohort_sha256=hashlib.sha256(json.dumps(cohort, separators=(",", ":")).encode()).hexdigest(),
                per_task_metrics=raw.get("per_task_metrics"), system_metrics=raw.get("system_metrics"),
                cohort_energy_diagnostic=raw.get("cohort_energy_diagnostic"),
                energy={k: raw.get("energy_evidence", {}).get(k) for k in ("run_joules", "startup_joules", "coverage_valid")},
                safe_memory_mib=raw.get("memory_evidence", {}).get("safe_memory_mib"),
                router_compute_ms=dict(count=len(compute_ms), mean=statistics.fmean(compute_ms) if compute_ms else None,
                                       p95=quantile(compute_ms, .95), maximum=max(compute_ms, default=None)),
                maximum_observed_pending=max(pending, default=None))


def build(packet_path, execution, output):
    packet_path, execution, output = map(Path, (packet_path, execution, output))
    packet = json.loads(packet_path.read_text())
    if not (execution / "terminal.json").exists():
        raise ValueError("Wait for the frozen execution terminal before endpoint analysis")
    terminal = json.loads((execution / "terminal.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    runs = {}
    for job in packet["jobs"]:
        path = execution / job["label"] / "run.json"
        if path.exists():
            raw = json.loads(path.read_text())
            runs[job["label"]] = summarize(raw, path)
            runs[job["label"]].update(role=job["role"], seed=job["seed"], deployment=job["nominal_plan"]["deployment"])
    weights = packet["jobs"][0]["policy_contract"]["weights"]
    for label, run in runs.items():
        u = run["utility"]
        if u.get("eligible") is True:
            if not all(isinstance(u.get(t + "_norm"), (int, float))
                       and math.isfinite(u[t + "_norm"]) for t in TERMS):
                raise ValueError(f"Eligible run missing a finite utility term: {label}")
            expected = -math.fsum(weights[t] * u[t + "_norm"] for t in TERMS)
            if not math.isclose(expected, u["U"], rel_tol=0, abs_tol=1e-10):
                raise ValueError(f"Six-term measured utility arithmetic failed: {label}")
    cohorts = {}
    for seed in packet["seeds"]:
        selected = {label: run["offered_cohort_sha256"] for label, run in runs.items()
                    if run["seed"] == seed}
        cohorts[str(seed)] = dict(by_run=selected, same_offered_cohort=len(set(selected.values())) == 1,
                                  measured_roles=len(selected))
        if len(set(selected.values())) > 1:
            raise ValueError(f"Offered cohort differs among candidates for seed {seed}")
    ranking = []
    for role in sorted({j["role"] for j in packet["jobs"]}):
        selected = [r for r in runs.values() if r["role"] == role]
        complete = (len(selected) == len(packet["seeds"])
                    and all(r["utility"].get("eligible") is True
                            and r["utility"].get("U") is not None for r in selected))
        ranking.append(dict(role=role, complete_eligible=complete,
                            mean_U=statistics.fmean(r["utility"]["U"] for r in selected) if complete else None,
                            mean_normalized_cost={t: statistics.fmean(r["utility"][t+"_norm"] for r in selected)
                                                  for t in TERMS} if complete else None))
    ranking.sort(key=lambda r: (r["mean_U"] is None, -(r["mean_U"] or 0), r["role"]))
    paired = []
    for seed in packet["seeds"]:
        spill = runs.get(f"coupled_spill-seed{seed}")
        control = runs.get(f"matched_no_spill-seed{seed}")
        complete = bool(spill and control and all(r["utility"].get("eligible") is True
                        and r["utility"].get("U") is not None for r in (spill, control)))
        row = dict(seed=seed, complete_eligible=complete, delta_U=None, gain_by_term=None,
                   same_offered_cohort=(spill["offered_cohort_sha256"] == control["offered_cohort_sha256"])
                   if spill and control else None)
        if complete:
            gain = {t: -weights[t] * (spill["utility"][t + "_norm"] - control["utility"][t + "_norm"]) for t in TERMS}
            delta = spill["utility"]["U"] - control["utility"]["U"]
            if not math.isclose(math.fsum(gain.values()), delta, rel_tol=0, abs_tol=1e-10):
                raise ValueError("Six-term decomposition does not reproduce measured delta_U")
            row.update(delta_U=delta, gain_by_term=gain,
                       strictly_positive=delta > 0,
                       pareto_gain_impossible_with_nonnegative_weights=all(v <= 0 for v in gain.values()) and any(v < 0 for v in gain.values()))
        paired.append(row)
    comparisons = []
    for seed in packet["seeds"]:
        control = runs.get(f"matched_no_spill-seed{seed}")
        for role in ("coupled_spill", "legacy_latency", "other_mixed"):
            candidate = runs.get(f"{role}-seed{seed}")
            if not (candidate and control and candidate["utility"].get("eligible") is True
                    and control["utility"].get("eligible") is True):
                continue
            gain = {t: -weights[t] * (candidate["utility"][t+"_norm"]
                                     - control["utility"][t+"_norm"]) for t in TERMS}
            delta = candidate["utility"]["U"] - control["utility"]["U"]
            if not math.isclose(math.fsum(gain.values()), delta, rel_tol=0, abs_tol=1e-10):
                raise ValueError("Candidate-to-control utility decomposition failed")
            comparisons.append(dict(seed=seed, role=role, delta_U=delta, gain_by_term=gain,
                                    same_residency=candidate["deployment"] == control["deployment"],
                                    exact_spill_ablation=role == "coupled_spill",
                                    scope="Matched spill ablation" if role == "coupled_spill"
                                    else "Whole deployment/policy candidate comparison; not an isolated spill effect"))
    result = dict(schema="gsdeploy.coupled_native_endpoint_diagnostics.v1", packet=receipt(packet_path),
                  terminal=receipt(execution / "terminal.json"), source=receipt(__file__),
                  execution_status=terminal["status"], expected_runs=len(packet["jobs"]),
                  measured_runs=len(runs), weights=weights, runs=runs, paired=paired,
                  offered_cohort_checks=cohorts, ranking=ranking,
                  candidate_comparisons_against_control=comparisons,
                  sign_convention="Positive gain_by_term improves U; sum reproduces spill U minus matched control U.",
                  interpretation="Descriptive decomposition only. No new weight choice, policy selection or confirmation gate.",
                  physical_full_pool_optimality=False)
    (output / "diagnostics.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    fields = ["label", "seed", "role", "deployment", "U", "eligible", "stability", "text_requests", "text_to_s0",
              "text_to_s1", "text_to_g2", "text_accuracy", "text_mean_ms", "text_p95_ms", "text_window_throughput_rps",
              "run_joules", "cohort_joules", "safe_memory_mib", *list(TERMS)]
    with (output / "deployment-routing-metrics.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label, run in runs.items():
            u = run["utility"]
            task = (run.get("per_task_metrics") or {}).get("text", {})
            routed = task.get("actual_routed_request_count", {})
            q = u.get("quality_evidence", {}).get("per_task", {}).get("text", {}).get("loss")
            writer.writerow(dict(label=label, seed=run["seed"], role=run["role"], deployment=json.dumps(run["deployment"], sort_keys=True),
                                 U=u.get("U"), eligible=u.get("eligible"), stability=run["stability"].get("stability_state_v2"),
                                 text_requests=task.get("offered_requests"), text_to_s0=routed.get("text#0", 0),
                                 text_to_s1=routed.get("text#1", 0), text_to_g2=routed.get("generalist_2#0", 0),
                                 text_accuracy=1-q if q is not None else None, text_mean_ms=task.get("mean_e2e_latency_ms"),
                                 text_p95_ms=task.get("p95_e2e_latency_ms"),
                                 text_window_throughput_rps=task.get("within_window_completion_throughput_rps"),
                                 run_joules=run["energy"].get("run_joules"), cohort_joules=(run.get("cohort_energy_diagnostic") or {}).get("joules"),
                                 safe_memory_mib=run["safe_memory_mib"], **{t:u.get(t+"_norm") for t in TERMS}))
    return result


def plot(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output)
    roles = ["coupled_spill", "matched_no_spill", "legacy_latency", "other_mixed"]
    labels = ["Coupled spill", "Matched no spill", "Legacy latency", "Other mixed"]
    seeds = [p["seed"] for p in result["paired"]]
    colors = ["#1768AC", "#D97706", "#2F855A", "#805AD5"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    width = .8 / max(len(seeds), 1)
    for index, seed in enumerate(seeds):
        positions = [i - .4 + width * (index + .5) for i in range(len(roles))]
        values = []
        for role in roles:
            run = result["runs"].get(f"{role}-seed{seed}", {})
            value = run.get("utility", {}).get("U")
            values.append(value if value is not None else float("nan"))
        bars = axes[0].bar(positions, values, width=width * .92, label=f"seed {seed}", color=colors[index % len(colors)])
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        for position, value in zip(positions, values):
            if math.isnan(value):
                axes[0].text(position, 0, "UNKNOWN", rotation=90, ha="center", va="bottom", fontsize=7)
    axes[0].set_xticks(range(len(roles)), labels, rotation=18, ha="right")
    axes[0].set_ylabel("Original measured six-term U (higher is better)")
    axes[0].set_title("Four declared development candidates")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=.2)
    for index, pair in enumerate(result["paired"]):
        positions = [i - .4 + width * (index + .5) for i in range(len(TERMS))]
        values = [(pair.get("gain_by_term") or {}).get(t, float("nan")) for t in TERMS]
        axes[1].bar(positions, values, width=width * .92, label=f"seed {pair['seed']}", color=colors[index % len(colors)])
    axes[1].set_xticks(range(len(TERMS)), list(TERMS))
    axes[1].axhline(0, color="#444444", linewidth=.8)
    axes[1].set_ylabel("Contribution to U(spill) - U(matched control)")
    axes[1].set_title("Paired gain decomposition; positive favors spill")
    axes[1].grid(axis="y", alpha=.2)
    fig.suptitle("Prospective development results; descriptive, not independent confirmation", fontsize=12)
    fig.savefig(output / "measured-results.png", dpi=180)
    fig.savefig(output / "measured-results.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--execution", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    result = build(args.packet, args.execution, args.output)
    if args.plot:
        plot(result, args.output)
    print(json.dumps({"runs": len(result["runs"]), "paired": result["paired"]}, ensure_ascii=False))
