#!/usr/bin/env python3
"""Count original-pool task columns with only conservative memory screening.

No model execution, fit, utility/FIFO computation or GPU API is used. A count
of retained columns is not a count of feasible complete deployments.
"""

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.milp import compositions


def count(c, grid):
    if c.route_pair_denominators_by_task:
        raise ValueError("PAIR_ROUTE_DOMAIN_UNSUPPORTED_BY_LEGACY_COLUMN_COUNT: use shared fraction_rows")
    details = {}
    for task in sorted(c.workload):
        models = sorted(m for m, spec in c.catalog["models"].items() if task in spec["supported_tasks"])
        seen, total, pruned = set(), 0, 0
        for denominator in range(1, grid + 1):
            for size in range(1, min(denominator, len(models)) + 1):
                for active in itertools.combinations(models, size):
                    for extra in compositions(denominator - size, size):
                        fractions = tuple((m, (u + 1) / denominator) for m, u in zip(active, extra))
                        if fractions in seen:
                            continue
                        seen.add(fractions)
                        for counts in itertools.product(*(sorted(int(n) for n in c.catalog["capacity_factors"][m]
                                                                 if int(n) > 0) for m in active)):
                            total += 1
                            memory = c.catalog["safe_factor"] * math.fsum(
                                c.catalog["memory_table_mib"][m][str(n)] for m, n in zip(active, counts))
                            pruned += memory > c.gpu_memory_mib + 1e-6
        details[task] = dict(raw_columns=total, active_memory_pruned=pruned, retained_upper_bound=total-pruned)
    return dict(grid=grid, per_task=details,
                raw_columns=sum(row["raw_columns"] for row in details.values()),
                active_memory_pruned=sum(row["active_memory_pruned"] for row in details.values()),
                retained_upper_bound=sum(row["retained_upper_bound"] for row in details.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--grids", type=int, nargs="+", default=[4, 8])
    args = parser.parse_args()
    original = args.config.read_bytes()
    c = SystemConfig.load(args.config)
    if c.catalog.get("memory_mode") == "profiled":
        parser.error("Nonadditive profiled memory has no additive screening bound")
    if c.catalog.get("performance_mode") in {"calibrated_surrogate", "calibrated_replay_v2"}:
        parser.error("Use the unchanged original pool; this diagnostic never fits or fabricates calibration")
    started = time.perf_counter()
    report = dict(config_path=str(args.config.resolve()), config_sha256=hashlib.sha256(original).hexdigest(),
        loaded_problem_sha256=__import__("gsdeploy.quality", fromlist=["quality_digest"]).quality_digest(c.as_dict()),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), gpu_started=False,
        calibrated_model_created=False, memory_budget_mib=c.gpu_memory_mib, safe_factor=c.catalog["safe_factor"],
        performance_mode=c.catalog.get("performance_mode"), memory_mode=c.catalog.get("memory_mode", "standalone_proxy"),
        lower_bound="original additive safe residency; multiplier fixed to 1 for conservative counting only",
        scope="Task-column count only. No performance, utility or complete-deployment feasibility is inferred. Any validated v1/v2 multiplier>=1 can only reduce retained columns.",
        counts=[count(c, grid) for grid in args.grids])
    report["elapsed_seconds"] = time.perf_counter() - started
    report["original_config_unchanged"] = args.config.read_bytes() == original
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
