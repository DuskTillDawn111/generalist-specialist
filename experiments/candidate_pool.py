"""Compare candidate pools through the same public system interface."""

import argparse
from pathlib import Path
from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.io import save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if Path(args.output).exists():
        raise FileExistsError(args.output)
    baseline = SystemConfig.load(args.config)
    rows = []
    for pool in ("a", "b", "c"):
        import json

        data = baseline.as_dict()
        data["catalog"] = json.loads(
            (
                Path(args.config).resolve().parent / "catalogs" / f"{pool}.json"
            ).read_text()
        )
        fit = DeploymentSystem(SystemConfig(**data)).plan()
        rows.append({"pool": pool, "result": fit})
    save(args.output, rows)


if __name__ == "__main__":
    main()
