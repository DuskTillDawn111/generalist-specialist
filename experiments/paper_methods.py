"""Bounded J3O/USHER adaptation using the common phase/FIFO evaluator."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.io import dumps, save
from gsdeploy.paper_algorithms import J3OInspiredOptimizer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--max-evaluations", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--seed-seconds", type=float, default=2.)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args(argv)
    if args.output and Path(args.output).exists():
        raise FileExistsError("Output already exists: " + args.output)
    result = J3OInspiredOptimizer(SystemConfig.load(args.config),
        max_evaluations=args.max_evaluations, time_limit_seconds=args.seconds,
        seed_seconds=args.seed_seconds, max_rounds=args.rounds).solve()
    if args.output:
        save(args.output, result)
    else:
        print(dumps(result), end="")
    return 0 if result["winner"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
