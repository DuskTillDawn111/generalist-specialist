"""Unified command line for planning, prediction, execution and evaluation."""

import argparse
import json
import sys
from pathlib import Path
from .config import SystemConfig
from .evaluation import evaluate_run
from .io import dumps, save
from .system import DeploymentSystem
from .evidence import checkpoint
from .runtime.preflight import preflight


def read(path):
    return json.loads(Path(path).read_text())


def main(argv=None):
    parser = argparse.ArgumentParser(prog="gsdeploy")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "predict", "run", "preflight"):
        command = commands.add_parser(name)
        command.add_argument("--config", required=True)
        command.add_argument("--output")
        if name == "plan":
            command.add_argument(
                "--deployment", help="JSON model -> count for fixed-deployment routing"
            )
            command.add_argument("--route-grid", type=int)
        else:
            command.add_argument("--plan", required=True)
        if name in {"run", "preflight"}:
            command.add_argument("--samples", required=True)
            command.add_argument("--workers", help="Native worker registry JSON")
            command.add_argument(
                "--admission", help="Checkpoint and quality admission manifest"
            )
        if name == "preflight":
            command.add_argument("--require-gpu", action="store_true")
        if name == "run":
            command.add_argument(
                "--backend", choices=("mock", "native"), default="mock"
            )
            command.add_argument("--duration", type=float, default=30.0)
            command.add_argument("--warmup", type=float, default=1.0)
            command.add_argument("--seed", type=int, default=42)
            command.add_argument(
                "--arrivals",
                choices=("deterministic_periodic", "poisson"),
                default="deterministic_periodic",
            )
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--run", required=True)
    evaluate.add_argument("--output")
    freeze = commands.add_parser("checkpoint")
    freeze.add_argument("--path", required=True)
    freeze.add_argument("--output")
    args = parser.parse_args(argv)
    try:
        if args.output and Path(args.output).exists():
            raise FileExistsError(f"Output already exists: {args.output}")
        if args.command == "checkpoint":
            result = checkpoint(args.path)
        elif args.command == "evaluate":
            raw = read(args.run)
            result = evaluate_run(SystemConfig(**raw["config"]), raw)
        else:
            system = DeploymentSystem(SystemConfig.load(args.config))
            if args.command == "plan":
                result = system.plan(
                    read(args.deployment) if args.deployment else None, args.route_grid
                )
            elif args.command == "predict":
                winner = read(args.plan).get("winner")
                if not winner:
                    raise ValueError("Plan has no winner")
                result = system.predict(winner["deployment"], winner["routing"])
            elif args.command == "preflight":
                winner = read(args.plan).get("winner")
                if not winner:
                    raise ValueError("Plan has no winner")
                result = preflight(
                    system.config,
                    winner,
                    args.samples,
                    read(args.workers) if args.workers else {},
                    args.admission,
                    args.require_gpu,
                )
            else:
                result = system.run(
                    read(args.plan),
                    args.samples,
                    backend=args.backend,
                    workers=read(args.workers) if args.workers else None,
                    duration=args.duration,
                    warmup=args.warmup,
                    seed=args.seed,
                    arrival_process=args.arrivals,
                    admission=args.admission,
                )
        if args.output:
            save(args.output, result)
        else:
            print(dumps(result), end="")
        if args.command == "plan" and result.get("winner") is None:
            return 1
        if args.command == "run" and result["status"] != "completed":
            return 1
        if args.command == "preflight" and not result["ready"]:
            return 1
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"gsdeploy: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
