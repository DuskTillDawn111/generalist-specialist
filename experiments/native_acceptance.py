"""Prepare a six-task smoke run; execution requires an explicit --execute flag."""

import argparse
from dataclasses import replace
from pathlib import Path

from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.cli import read
from gsdeploy.io import save
from gsdeploy.runtime.preflight import preflight


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[1] / "configs/static_replicas.json"
        ),
    )
    parser.add_argument("--workers", required=True)
    parser.add_argument("--samples", required=True)
    parser.add_argument("--admission")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--warmup", type=float, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    output = Path(args.output_dir)
    # Reserve a fresh directory before writing any experiment files.
    output.mkdir(parents=True, exist_ok=False)
    system = DeploymentSystem(
        replace(
            SystemConfig.load(args.config),
            horizon_seconds=args.duration,
            random_seed=args.seed,
        )
    )
    plan = system.plan()
    save(output / "plan.json", plan)
    if not plan["winner"]:
        print("No feasible deployment")
        return 1
    workers = read(args.workers)
    checks = preflight(
        system.config,
        plan["winner"],
        args.samples,
        workers,
        args.admission,
        require_gpu=args.execute,
    )
    save(output / "preflight.json", checks)
    if not checks["ready"]:
        print("Preflight failed: " + "; ".join(checks["issues"]))
        return 1
    if not args.execute:
        print(f"Inputs checked; GPU and inference untested. Results: {output}")
        return 0
    raw = system.run(
        plan,
        args.samples,
        backend="native",
        workers=workers,
        admission=args.admission,
        duration=args.duration,
        warmup=args.warmup,
        seed=args.seed,
    )
    save(output / "run.json", raw)
    return 0 if raw["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
