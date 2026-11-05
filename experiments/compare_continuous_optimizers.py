"""Run opt-in Phase 1 experiments without changing the historical CLI."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gsdeploy.continuous.benchmark import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithms", nargs="+")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--outer", type=int)
    parser.add_argument("--predictions", type=int)
    parser.add_argument("--seconds", type=float)
    args = parser.parse_args()
    summary = run_experiment(args.config, args.output, algorithms=args.algorithms,
        seeds=args.seeds, outer=args.outer, predictions=args.predictions, seconds=args.seconds)
    print(json.dumps(dict(runs=len(summary["records"]), total_wall_seconds=summary["total_wall_seconds"],
                         output=str(Path(args.output).resolve())), indent=2))


if __name__ == "__main__":
    main()
