"""Plot receipt-verified independent confirmation summaries, without GPU use.

The upstream report performs native import and measured re-evaluation. This
exporter verifies its frozen packet/run receipts and scoring arithmetic; it does
not re-run the importer or require the current checkout to be the frozen runtime.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


TERMS = ("Q", "T", "L", "P", "M", "E")
TERM_LABELS = ("Quality loss", "Throughput loss", "Mean latency", "P95 latency", "Memory", "Board energy")
ROLE_NAMES = {
    "free_search_winner": "Selected by free search",
    "strongest_alternative_deployment": "Alternative deployment",
    "strongest_alternative_route": "Alternative route",
    "best_no_split": "Best without splitting",
    "best_generalist_only": "Best generalist only",
    "best_specialist_only": "Best specialist only",
}
SCOPE = "Only frozen measured candidates/seeds; no physical global optimality claim."


def _read(path):
    return json.loads(Path(path).read_text())


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def _receipt(receipt):
    path = Path(receipt["path"])
    if not path.is_absolute() or not path.is_file() or _sha(path) != receipt["sha256"]:
        raise ValueError("Missing or changed receipt: " + str(path))
    return path


def _finite(value):
    return type(value) in (float, int) and math.isfinite(value)


def verify_report(report_path, *, allow_incomplete=False):
    """Fail closed on receipt, roster or scoring mismatch, even in partial mode."""
    report_path = Path(report_path).resolve()
    report = _read(report_path)
    if report.get("schema") != "gsdeploy.independent_confirmation_report.v1":
        raise ValueError("Expected a strict independent confirmation report")
    if report.get("status") not in {"ALL_FROZEN_JOBS_VERIFIED", "INCOMPLETE_CONFIRMATION"}:
        raise ValueError("Unknown or invalid confirmation status")
    if report["status"] != "ALL_FROZEN_JOBS_VERIFIED" and not allow_incomplete:
        raise ValueError("Incomplete report; explicitly pass --allow-incomplete for marked descriptive plots")
    if report.get("physical_global_optimality_certified") is not False:
        raise ValueError("Unsupported physical global optimality claim")
    packet_path = _receipt(report["packet"])
    packet = _read(packet_path)
    seal = _canonical({k: v for k, v in packet.items() if k != "confirmation_sha256"})
    if packet.get("schema") != "gsdeploy.independent_confirmation.v1" or packet.get("confirmation_sha256") != seal:
        raise ValueError("Changed or invalid frozen confirmation packet")
    roles = packet["candidate_roles"]
    if (report.get("candidate_roles") != roles or len(roles) != len(ROLE_NAMES)
            or {r["label"] for r in roles} != set(ROLE_NAMES)):
        raise ValueError("Changed or incomplete candidate role roster")
    candidates = {c["plan_key"]: c for c in packet["candidates"]}
    if not candidates or len(candidates) != len(packet["candidates"]):
        raise ValueError("Missing or duplicate frozen candidates")
    for key, candidate in candidates.items():
        labels = [r["label"] for r in roles if r["plan_key"] == key]
        if key != _canonical(candidate["plan"]) or not labels or candidate["labels"] != labels:
            raise ValueError("Changed candidate plan or role binding")
    if any(r["plan_key"] is not None and r["plan_key"] not in candidates for r in roles):
        raise ValueError("Role refers to an unknown candidate")
    seeds = packet["confirmation"]["seeds"]
    if seeds != [43, 44, 45] or packet["confirmation"].get("training_forbidden") is not True:
        raise ValueError("Changed independent seed contract")
    jobs = {j["id"]: j for j in packet["jobs"]}
    expected_pairs = {(key, seed) for key in candidates for seed in seeds}
    if (len(jobs) != len(packet["jobs"]) or len(jobs) != len(expected_pairs)
            or {(j["candidate_key"], j["seed"]) for j in jobs.values()} != expected_pairs):
        raise ValueError("Incomplete or duplicate frozen job roster")
    for job in jobs.values():
        candidate = candidates[job["candidate_key"]]
        if (job["phase"] != "confirmation" or job.get("purpose") != "independent_confirmation_not_training"
                or job["id"] != _canonical(job["identity"]) or job["plan"] != candidate["plan"]
                or job["labels"] != candidate["labels"] or job["identity"].get("seed") != job["seed"]
                or job["identity"].get("deployment") != job["plan"]["deployment"]
                or job["identity"].get("routing") != job["plan"]["routing"]):
            raise ValueError("Changed confirmation job identity")
    weights = packet["config"]["weights"]
    if set(weights) != set(TERMS) or not all(_finite(v) and v >= 0 for v in weights.values()):
        raise ValueError("Invalid six-term scoring weights")
    seen = set()
    for row in report["runs"]:
        job_id = row["job_id"]
        if job_id not in jobs or job_id in seen:
            raise ValueError("Unknown or duplicate report job")
        seen.add(job_id)
        job = jobs[job_id]
        if (row["plan_key"] != job["candidate_key"] or row["seed"] != job["seed"]
                or row["labels"] != job["labels"]):
            raise ValueError("Run summary differs from frozen candidate/seed")
        raw = _read(_receipt(row))
        actual = raw.get("executed_measurement_identity", {})
        if (raw.get("experiment_job_id") != job_id or raw.get("status") != "completed"
                or raw.get("backend") != "native" or raw.get("experiment_phase") != "confirmation"
                or raw.get("confirmation_packet_sha256") != seal or raw.get("plan") != job["plan"]
                or raw.get("executed_measurement_id") != _canonical(actual)):
            raise ValueError("Raw run is not the declared completed native confirmation")
        # The importer permits only one derived worker environment field.
        actual = json.loads(json.dumps(actual))
        for worker in actual.get("workers", {}).values():
            worker.get("env", {}).pop("GS_CHECKPOINT_FILES_JSON", None)
        if actual != job["identity"]:
            raise ValueError("Raw executed identity differs from the frozen job")
        utility = row["utility"]
        if utility.get("eligible") is not True or not _finite(utility.get("U")):
            raise ValueError("Ineligible or invalid measured utility")
        costs = {}
        for term in TERMS:
            value = utility.get(term + "_norm")
            if not _finite(value):
                raise ValueError("Missing or nonfinite utility term")
            costs[term] = weights[term] * value
            given = row.get("weighted_costs", {}).get(term)
            if not _finite(given) or not math.isclose(given, costs[term], rel_tol=1e-8, abs_tol=1e-8):
                raise ValueError("Weighted cost differs from frozen scoring weights")
        if not math.isclose(utility["U"], -math.fsum(costs.values()), rel_tol=1e-8, abs_tol=1e-8):
            raise ValueError("Total utility differs from its six weighted costs")
    missing = sorted(set(jobs) - seen)
    expected_status = "INCOMPLETE_CONFIRMATION" if missing else "ALL_FROZEN_JOBS_VERIFIED"
    if (report["status"] != expected_status or report["missing_job_ids"] != missing
            or report["completed_jobs"] != len(seen) or report["planned_jobs"] != len(jobs)):
        raise ValueError("Confirmation completeness claim differs from the frozen jobs")
    return report, packet


def plot(report_path, output, *, allow_incomplete=False, cpu_fixture=False):
    report_path, output = Path(report_path).resolve(), Path(output).resolve()
    report, packet = verify_report(report_path, allow_incomplete=allow_incomplete)
    if report.get("evidence_kind") == "CPU_FIXTURE_NOT_MEASURED" and not cpu_fixture:
        raise ValueError("CPU fixture reports require --cpu-fixture labels")
    output.mkdir(parents=True, exist_ok=False)
    candidates = packet["candidates"]
    ids = {c["plan_key"]: "C" + str(i + 1) for i, c in enumerate(candidates)}
    entries = [{"id": ids[c["plan_key"]], **c} for c in candidates]
    entries += [{"id": "NO PLAN: " + ROLE_NAMES[r["label"]], "plan_key": None,
                 "labels": [r["label"]], "role_status": r["status"]}
                for r in packet["candidate_roles"] if r["plan_key"] is None]
    seeds = packet["confirmation"]["seeds"]
    rows = {(r["plan_key"], r["seed"]): r for r in report["runs"]}
    groups = [[rows[e["plan_key"], s] for s in seeds if (e["plan_key"], s) in rows] for e in entries]
    incomplete = report["status"] != "ALL_FROZEN_JOBS_VERIFIED"
    marker = "CPU FIXTURE - NOT MEASURED" if cpu_fixture else "Independent confirmation"
    if incomplete:
        marker += " | INCOMPLETE - DESCRIPTIVE ONLY"
    x = np.arange(len(entries))
    ticks = [e["id"] + "\n" + (str(len(g)) + "/" + str(len(seeds)) + " seeds" if e["plan_key"] else str(e["role_status"])) for e, g in zip(entries, groups)]

    def finish(fig, name):
        fig.text(.01, .012, marker + " | " + SCOPE, fontsize=8, color="#9c2020" if cpu_fixture or incomplete else "#444444")
        if cpu_fixture or incomplete:
            fig.text(.5, .53, marker, ha="center", rotation=20, alpha=.12, fontsize=23)
        fig.tight_layout(rect=(0, .065, 1, .96))
        for ext in ("png", "svg", "pdf"):
            fig.savefig(output / (name + "." + ext), dpi=180, bbox_inches="tight")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(9, len(entries) * 1.5), 5))
    offsets = np.linspace(-.18, .18, len(seeds))
    for seed, offset, color in zip(seeds, offsets, plt.cm.tab10.colors):
        values = [rows[e["plan_key"], seed]["utility"]["U"] if (e["plan_key"], seed) in rows else np.nan for e in entries]
        ax.scatter(x + offset, values, label=f"Seed {seed}", color=color, s=36)
    for index, group in enumerate(groups):
        if group:
            mean = statistics.fmean(r["utility"]["U"] for r in group)
            ax.scatter([index], [mean], marker="D", s=60, color="black", label="Arithmetic mean" if index == next(i for i, g in enumerate(groups) if g) else None)
            ax.annotate(f"{mean:.4f}", (index, mean), xytext=(0, 10), textcoords="offset points", ha="center", fontsize=9)
        missing = [str(s) for s in seeds if entries[index]["plan_key"] is not None and (entries[index]["plan_key"], s) not in rows]
        if missing:
            ax.text(index, .03, "MISSING: " + ",".join(missing), transform=ax.get_xaxis_transform(), ha="center", fontsize=7, color="#aa2222", rotation=18)
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set_ylabel("Measured U (larger is better)")
    ax.set_title("Each frozen candidate: seed-level utility and observed mean")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, ncol=4, fontsize=8)
    finish(fig, "confirmation-utility")

    fig, ax = plt.subplots(figsize=(max(9, len(entries) * 1.5), 5))
    bottom = np.zeros(len(entries))
    for term, label, color in zip(TERMS, TERM_LABELS, plt.cm.tab10.colors):
        values = np.array([statistics.fmean(r["weighted_costs"][term] for r in group) if group else 0 for group in groups])
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += values
    for index, group in enumerate(groups):
        ax.annotate(f"-U={bottom[index]:.4f}" if group else ("NO PLAN" if entries[index]["plan_key"] is None else "NO VALID RUNS"), (index, bottom[index]),
                    xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, ticks, fontsize=8)
    ax.set_ylabel("Mean weighted cost = -mean(U), smaller is better")
    ax.set_title("Six weighted utility costs (observed seeds only)")
    ax.legend(frameon=False, ncol=3, fontsize=8)
    ax.margins(y=.2)
    finish(fig, "confirmation-costs")

    selected_role = next(r for r in packet["candidate_roles"] if r["label"] == packet["winner_label"])
    selected = next((c for c in candidates if c["plan_key"] == selected_role["plan_key"]), None)
    fig, axes = plt.subplots(1, 2, figsize=(13, 8))
    if selected:
        plan = selected["plan"]
        models = list(plan["deployment"])
        axes[0].barh(models, [plan["deployment"][m] for m in models], color="#4477aa")
        axes[0].set_xlabel("Resident replica count")
        axes[0].set_title("Frozen selected deployment: " + ids[selected["plan_key"]])
        tasks = list(plan["routing"])
        shares = np.array([[plan["routing"][t].get(m, 0) / sum(plan["routing"][t].values()) for m in models] for t in tasks])
        axes[1].imshow(shares, cmap="Blues", vmin=0, vmax=1, aspect="auto")
        axes[1].set_xticks(range(len(models)), models, rotation=25, ha="right", fontsize=8)
        axes[1].set_yticks(range(len(tasks)), tasks)
        for i in range(len(tasks)):
            for j in range(len(models)):
                axes[1].text(j, i, f"{shares[i,j]:.0%}", ha="center", va="center", color="white" if shares[i,j] > .55 else "black")
        axes[1].set_title("Frozen per-task routing shares")
    else:
        for ax in axes:
            ax.axis("off")
            ax.text(.5, .5, "NO SELECTED PLAN: " + str(selected_role["status"]), ha="center", transform=ax.transAxes)
    # Keep all six roles, including infeasible/absent searches and deduplicated roles.
    role_rows = [[ROLE_NAMES[r["label"]], ids.get(r["plan_key"], "NO PLAN"), str(r["status"]), str(r["optimality_certified"])] for r in packet["candidate_roles"]]
    table_ax = fig.add_axes([.05, .075, .9, .24])
    table_ax.axis("off")
    table = table_ax.table(cellText=role_rows, colLabels=["Frozen role", "Candidate", "Search status", "Proxy optimum certified"], loc="center", cellLoc="left", colWidths=[.35, .12, .26, .27])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    # Manual layout reserves space for the complete role table.
    for ax in axes:
        box = ax.get_position()
        ax.set_position([box.x0, .47, box.width, .4])
    fig.text(.01, .012, marker + " | " + SCOPE, fontsize=8)
    if cpu_fixture or incomplete:
        fig.text(.5, .45, marker, ha="center", alpha=.45, color="#9c2020")
    for ext in ("png", "svg", "pdf"):
        fig.savefig(output / ("confirmation-selected-plan." + ext), dpi=180, bbox_inches="tight")
    plt.close(fig)
    roster = {"roles": packet["candidate_roles"], "candidates": [{"plot_id": e["id"], **e,
        "observed_seeds": [s for s in seeds if (e["plan_key"], s) in rows],
        "missing_seeds": [s for s in seeds if e["plan_key"] is not None and (e["plan_key"], s) not in rows]} for e in entries]}
    (output / "candidate-roster.json").write_text(json.dumps(roster, indent=2, ensure_ascii=False) + "\n")
    manifest = {"schema": "gsdeploy.confirmation_figures.v1", "report": {"path": str(report_path), "sha256": _sha(report_path)},
        "packet": report["packet"], "runs": [{k: r[k] for k in ("job_id", "plan_key", "seed", "path", "sha256")} for r in report["runs"]],
        "script": {"path": str(Path(__file__).resolve()), "sha256": _sha(__file__)},
        "status": report["status"], "evidence_kind": "CPU_FIXTURE_NOT_MEASURED" if cpu_fixture else "STRICT_UPSTREAM_CONFIRMATION_REPORT",
        "receipt_verification": "packet seal, job/candidate roster, raw-run SHA and identities, six-term arithmetic; upstream native importer is not rerun",
        "environment": {"python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__},
        "missing_job_ids": report["missing_job_ids"], "scope": SCOPE,
        "physical_global_optimality_certified": False, "gpu_started": False,
        "files": {p.name: _sha(p) for p in sorted(output.iterdir()) if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true", help="Allow clearly marked partial reports; never bypass receipt/utility validation")
    parser.add_argument("--cpu-fixture", action="store_true", help="Mark every output CPU FIXTURE - NOT MEASURED (test data only)")
    args = parser.parse_args()
    result = plot(args.report, args.output, allow_incomplete=args.allow_incomplete, cpu_fixture=args.cpu_fixture)
    print(json.dumps({k: result[k] for k in ("status", "evidence_kind", "gpu_started")}))
