"""Freeze real retained-model inputs, resources and per-input quality on CPU."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.profiling.static_inputs import prepare_static_study


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/static-study-v6-20261021/inputs")
    args = parser.parse_args()
    contract = prepare_static_study(ROOT, args.output)
    print(json.dumps(dict(status=contract["phase"], output=str(args.output.resolve()),
        paths=contract["resource_grid"]["model_task_paths"], missing_quality=contract["quality_missing_paths"],
        workload=contract["workload"], formal_horizon_seconds=contract["formal_horizon_seconds"]), ensure_ascii=False))


if __name__ == "__main__":
    main()
