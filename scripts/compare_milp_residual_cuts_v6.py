#!/usr/bin/env python3
"""Optional short CPU benchmark of old/new feedback on identical synthetic data.

Every invocation uses MILP and a fresh process. Only the identity-cut formula
changes. The default comparison is deliberately bounded; no speed advantage is
assumed. Does not import workers or GPU APIs.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def legacy_cut(base_row, theta, selected, blocks, residual):
    cost = residual - sum(base_row.get(i, 0) for i in selected)
    return {**{i: -cost for i in selected}, theta: 1.0}, -cost * (blocks - 1)


def run_worker(config_path, variant):
    from gsdeploy.static_study_benchmark import worker
    if variant == "legacy_full_cost":
        with patch("gsdeploy.milp.residual_identity_cut", legacy_cut):
            return worker(config_path, "milp")
    return worker(config_path, "milp")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/static-study-gpu-20261022/residual-cut-comparison"))
    parser.add_argument("--time-limit", type=float, default=2.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--variant", choices=["legacy_full_cost", "residual"])
    args = parser.parse_args()
    if args.worker:
        print(json.dumps(run_worker(args.worker, args.variant)))
        return
    from gsdeploy.static_study_benchmark import make_problem
    if args.repeats < 1 or args.time_limit <= 0:
        parser.error("positive repeats/time limit required")
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1] / "gsdeploy" / "milp.py"
    start_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    report = dict(evidence="synthetic CPU only; MILP retained in both variants",
                  protocol="fresh processes; alternating order; identical problem; includes model build; library import excluded",
                  repeats=args.repeats, time_limit_seconds=args.time_limit,
                  milp_source_sha256=start_hash, cases=[])
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    for case in [dict(name="tiny", models=2, tasks=1, replicas=1, K=2),
                 dict(name="small", models=3, tasks=2, replicas=2, K=3)]:
        config, count = make_problem(case, args.time_limit)
        config_path = args.output / f"{case['name']}.json"
        config_path.write_text(json.dumps(config.as_dict(), sort_keys=True, indent=2) + "\n")
        entry = dict(case=case, plans=count, config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(), runs=[])
        for repeat in range(args.repeats):
            order = ["legacy_full_cost", "residual"]
            if repeat % 2:
                order.reverse()
            for variant in order:
                start = time.perf_counter()
                child = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                                        "--worker", str(config_path.resolve()), "--variant", variant],
                                       check=True, capture_output=True, text=True, env=env,
                                       timeout=args.time_limit + 30)
                record = json.loads(child.stdout)
                entry["runs"].append(dict(variant=variant, repeat=repeat + 1,
                                          subprocess_wall_seconds=time.perf_counter() - start, **record))
        entry["summary"] = {}
        for variant in ["legacy_full_cost", "residual"]:
            rows = [r for r in entry["runs"] if r["variant"] == variant]
            entry["summary"][variant] = dict(
                median_wall_seconds=statistics.median(r["wall_seconds_including_model_build"] for r in rows),
                median_complete_evaluations=statistics.median(r["metrics"]["full_prediction_calls"] for r in rows),
                median_milp_calls=statistics.median(r["metrics"]["milp_calls"] for r in rows),
                certified_runs=sum(r["optimality_certified"] for r in rows),
                observed_U=[r["U"] for r in rows], gaps=[r["optimality_gap"] for r in rows],
            )
        report["cases"].append(entry)
    report["source_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == start_hash
    (args.output / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(output=str(args.output.resolve()), source_unchanged=report["source_unchanged"])))


if __name__ == "__main__":
    main()
