"""Run the target activation-aware BO/MADS + progressive deployment MILP."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.continuous import (ActivationAwareBilevelOptimizer, Budget,
    ContinuousRoutingEvaluator, ImmutableContinuousProblem, SessionPredictorOracle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--seconds", type=float, default=30.)
    parser.add_argument("--predictions", type=int, default=256)
    parser.add_argument("--routes", type=int, default=128)
    parser.add_argument("--milp-calls", type=int, default=1024)
    parser.add_argument("--inner-seconds", type=float, default=1.)
    parser.add_argument("--inner-predictions", type=int, default=8)
    parser.add_argument("--max-spaces", type=int, default=32)
    parser.add_argument("--joint-seed-k", type=int)
    parser.add_argument("--exact-inner", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output directory already exists")
    budget = Budget(args.seconds, args.predictions, args.routes, args.milp_calls)
    c = SystemConfig.load(args.config)
    # This constructor enforces ready normalization references. It does not
    # supply synthetic defaults to a pending real configuration.
    p = ImmutableContinuousProblem(c)
    with SessionPredictorOracle(p) as oracle:
        e = ContinuousRoutingEvaluator(p, oracle, budget=budget)
        raw = ActivationAwareBilevelOptimizer(e, seed=args.seed, max_spaces=args.max_spaces,
            inner_seconds=args.inner_seconds, inner_predictions=args.inner_predictions,
            joint_seed_k=args.joint_seed_k, exact_inner=args.exact_inner).solve()
        raw["engine_stats"] = oracle.engine_stats()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"result.json").write_text(json.dumps(raw, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    manifest = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (ROOT/"gsdeploy/continuous").glob("*.py")}
    manifest[str(Path(__file__).relative_to(ROOT))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/"run-manifest.json").write_text(json.dumps(dict(config=c.as_dict(),
        config_path=str(args.config.resolve()), config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
        arguments={k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        source_files=manifest, physical_qualification_granted=False), ensure_ascii=False, indent=2)+"\n")
    print(json.dumps(dict(status=raw["status"], winner=raw["winner"], budget=raw["budget"],
                         output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
