#!/usr/bin/env python3
"""Independently audit all statistics in a strict, complete 15-job confirmation."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import math
from pathlib import Path

TERMS = ("Q", "T", "L", "P", "M", "E")
REL_TOL = ABS_TOL = 1e-10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def receipt(path):
    return {"path": str(Path(path).resolve()), "sha256": sha(path)}


def require(value, reason):
    if not value:
        raise ValueError(reason)


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def equal(actual, expected, field):
    """Reject missing/extra fields and compare every nested numerical value."""
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and set(actual) == set(expected), "Field roster differs: " + field)
        for key, value in expected.items():
            equal(actual[key], value, field + "." + key)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(actual) == len(expected), "List differs: " + field)
        for index, value in enumerate(expected):
            equal(actual[index], value, field + f"[{index}]")
    elif type(expected) is float:
        require(numeric(actual) and math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=ABS_TOL),
                "Numeric statistic differs: " + field)
    else:
        require(type(actual) is type(expected) and actual == expected, "Statistic differs: " + field)


def t_df2_quantile(probability):
    """Analytic inverse CDF for t(df=2), independent of scipy's t implementation.

    F(x)=1/2 + x/(2*sqrt(x*x+2)); invert with z=2*p-1.
    """
    require(0 < probability < 1, "Invalid t probability")
    z = 2 * probability - 1
    return math.copysign(math.sqrt(2 * z * z / (1 - z * z)), z)


def holm(raw_p_values):
    ordered = sorted(enumerate(raw_p_values), key=lambda pair: pair[1])
    adjusted = [None] * len(raw_p_values)
    previous = 0.
    for rank, (index, value) in enumerate(ordered):
        require(numeric(value) and 0 <= value <= 1, "Invalid p value")
        previous = max(previous, min(1., (len(ordered) - rank) * value))
        adjusted[index] = previous
    return adjusted


def recompute(rows, winner_key, candidate_keys, seeds):
    require(seeds == [43, 44, 45] and len(candidate_keys) == 5 and len(set(candidate_keys)) == 5,
            "Expected five unique plans and seeds 43/44/45")
    groups = {}
    for row in rows:
        key = row["plan_key"], row["seed"]
        require(key not in groups, "Duplicate plan/seed")
        require(row["plan_key"] in candidate_keys and row["seed"] in seeds, "Unexpected plan/seed")
        utility = row["utility"]
        require(utility.get("eligible") is True and all(numeric(utility.get(k + "_norm")) for k in TERMS)
                and numeric(utility.get("U")), "Incomplete seed-level utility")
        groups[key] = utility
    require(set(groups) == {(key, seed) for key in candidate_keys for seed in seeds},
            "Final statistical audit requires all 15 unique plan/seed blocks")
    pairs = []
    for rival in sorted(set(candidate_keys) - {winner_key}):
        differences = [groups[winner_key, s]["U"] - groups[rival, s]["U"] for s in seeds]
        mean = math.fsum(differences) / 3
        # Unbiased sample variance with n-1=2; SE = SD/sqrt(n).
        sample_variance = math.fsum((value - mean) ** 2 for value in differences) / 2
        se = math.sqrt(sample_variance / 3)
        half = t_df2_quantile(.975) * se
        observed = abs(math.fsum(differences))
        sign_sums = [abs(math.fsum(sign * value for sign, value in zip(signs, differences)))
                     for signs in itertools.product((-1, 1), repeat=3)]
        p_value = sum(value >= observed - 1e-12 for value in sign_sums) / 8
        pairs.append(dict(winner_key=winner_key, rival_key=rival, paired_seeds=seeds, missing_seeds=[],
            status="COMPLETE_PAIRED_BLOCKS", U_differences=differences, mean_U_difference=mean,
            standard_error=se, student_interval=[mean - half, mean + half], student_confidence_level=.95,
            exact_two_sided_sign_flip_p=p_value,
            all_observed_differences_positive=all(value > 0 for value in differences),
            mean_term_differences={term: math.fsum(groups[winner_key, s][term + "_norm"]
                - groups[rival, s][term + "_norm"] for s in seeds) / 3 for term in TERMS}))
    for pair, adjusted in zip(pairs, holm([pair["exact_two_sided_sign_flip_p"] for pair in pairs])):
        pair.update(holm_adjusted_sign_flip_p=adjusted, sign_flip_significant_after_holm=adjusted < .05)
    return pairs


def audit(report_path, verifier_path, *, cpu_fixture=False):
    report_path, verifier_path = Path(report_path).resolve(), Path(verifier_path).resolve()
    inputs = [receipt(report_path), receipt(verifier_path)]
    raw_report = json.loads(report_path.read_text())
    require(raw_report.get("status") == "ALL_FROZEN_JOBS_VERIFIED"
            and raw_report.get("completed_jobs") == raw_report.get("planned_jobs") == 15
            and raw_report.get("missing_job_ids") == [] and len(raw_report.get("runs", [])) == 15,
            "FINAL_AUDIT_REFUSED_INCOMPLETE: strict 15/15 report with no missing jobs required")
    fixture = raw_report.get("evidence_kind") == "CPU_FIXTURE_NOT_MEASURED"
    require(fixture == cpu_fixture, "CPU fixtures require --cpu-fixture; real reports must not use that flag")
    spec = importlib.util.spec_from_file_location("receipt_verifier_for_statistics_audit", verifier_path)
    verifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(verifier)
    report, packet = verifier.verify_report(report_path)
    seeds = packet["confirmation"]["seeds"]
    keys = [candidate["plan_key"] for candidate in packet["candidates"]]
    winner = next(role["plan_key"] for role in packet["candidate_roles"] if role["label"] == packet["winner_label"])
    require(winner in keys, "Winner missing from candidates")
    paired = report["paired"]
    require(paired.get("schema") == "gsdeploy.paired_measured_comparisons.v1"
            and paired.get("winner_key") == winner and paired.get("planned_seeds") == seeds
            and paired.get("all_planned_blocks_complete") is True
            and paired.get("physical_global_optimality_certified") is False,
            "Changed statistical report identity or completeness")
    equal(paired.get("smallest_possible_two_sided_sign_flip_p"), .25, "smallest_possible_p")
    expected = recompute(report["runs"], winner, keys, seeds)
    equal(paired.get("comparisons"), expected, "paired.comparisons")
    inputs.append(report["packet"])
    inputs.extend({"path": row["path"], "sha256": row["sha256"]} for row in report["runs"])
    require(all(sha(row["path"]) == row["sha256"] for row in inputs), "Input changed during audit")
    return dict(schema="gsdeploy.confirmation_statistics_independent_audit.v1",
        status="CPU_FIXTURE_STATISTICS_VERIFIED" if fixture else "REAL_15_OF_15_STATISTICS_VERIFIED",
        evidence_kind="CPU_FIXTURE_NOT_MEASURED" if fixture else "STRICT_UPSTREAM_NATIVE_CONFIRMATION_REPORT",
        input_receipts=inputs, confirmation_sha256=packet["confirmation_sha256"],
        verified_jobs=15, unique_candidates=5, paired_rivals=4, seeds=seeds,
        independent_comparisons=expected, analytic_t_df2_975=t_df2_quantile(.975),
        tolerance=dict(relative=REL_TOL, absolute=ABS_TOL),
        implementation="No gsdeploy.study_reporting or scipy statistics used; analytical t(df=2), explicit 8-sign enumeration, independent Holm step-down",
        upstream_native_import_rerun=False, full_statistical_audit_complete=not fixture,
        physical_global_optimality_certified=False, gpu_started=False,
        scope="Frozen five candidates and three seed blocks only; strict upstream report receipts verified; no raw native re-evaluation here.",
        limitations="Three seeds give minimum exact two-sided p=.25; no .05 significance claim; Student interval assumes approximately normal independent seed differences.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path, help="Frozen strict plot/report receipt verifier")
    parser.add_argument("--output", required=True, type=Path, help="New JSON file; never overwritten")
    parser.add_argument("--cpu-fixture", action="store_true")
    args = parser.parse_args()
    result = audit(args.report, args.verifier, cpu_fixture=args.cpu_fixture)
    result["auditor"] = receipt(__file__)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("status", "verified_jobs", "paired_rivals", "full_statistical_audit_complete", "gpu_started")}))


if __name__ == "__main__":
    main()
