"""Explicit CPU-only v2 preparation/fitting; never auto-activate from v1 driver."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.calibration_v2 import declare_replay_protocol, fit_calibrated_replay, freeze_input_baselines
from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.io import save


def prepare(base_packet_path, legacy_protocol_path, holdout_packet_path, output):
    """Freeze standalone medians and role bindings; never inspect actual runs."""
    paths = [Path(p).resolve() for p in (base_packet_path, legacy_protocol_path, holdout_packet_path)]
    base_packet, legacy, new_packet = [json.loads(path.read_text()) for path in paths]
    base = SystemConfig(**base_packet["config"])
    if base.horizon_seconds != 1800 or base.random_seed != 42:
        raise ValueError("V2 base config must remain formal H1800/development seed42")
    legacy_ids = legacy["training_ids"] + legacy["validation_ids"]
    if not set(legacy_ids).issubset(job["id"] for job in base_packet["jobs"]):
        raise ValueError("Base historical packet must contain original eight conditions")
    development = [{"id": key, "packet_path": str(paths[0]), "packet_sha256": digest(paths[0]),
                    "purpose": "legacy_v1_development"} for key in legacy_ids]
    if len(new_packet["jobs"]) != 2:
        raise ValueError("V2 holdout packet must contain exactly two new conditions")
    holdouts = [{"id": job["id"], "packet_path": str(paths[2]), "packet_sha256": digest(paths[2]),
                 "purpose": "v2_independent_holdout"}
                for job in sorted(new_packet["jobs"], key=lambda job: job["seed"])]
    frozen = freeze_input_baselines(base, Path(base_packet["inputs"]) / "runtime-inputs.jsonl")
    protocol = declare_replay_protocol(base, frozen, development, holdouts, legacy)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Immutable v2 preparation output already exists: " + str(output))
    save(output / "frozen-input-baselines.json", frozen)
    save(output / "protocol.json", protocol)
    save(output / "base-config.json", base.as_dict())
    save(output / "manifest.json", {"schema": "gsdeploy.replay_v2_preparation.v1",
        "files": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "sources": [{"path": str(path), "sha256": digest(path)} for path in paths],
        "gpu_started": False, "fitted": False,
        "activation": "Explicit opt-in after preserved v1 validation failure; not read by v1 automatic driver",
        "missing_input_paths": frozen["missing_paths"], "status": "CPU_PREPARED_V2_PROTOCOL_ONLY"})
    return {"status": "CPU_PREPARED_V2_PROTOCOL_ONLY", "output": str(output), "fitted": False,
            "gpu_started": False, "development_conditions": len(development), "holdout_conditions": len(holdouts),
            "missing_input_paths": len(frozen["missing_paths"])}


def fit_from_completed(prepared, runs_root, output, explicit_runs=None):
    """Require all declared conditions, each with its own frozen source packet."""
    prepared = Path(prepared).resolve()
    manifest = json.loads((prepared / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if digest(prepared / name) != expected:
            raise ValueError("V2 prepared artifact changed: " + name)
    protocol = json.loads((prepared / "protocol.json").read_text())
    frozen = json.loads((prepared / "frozen-input-baselines.json").read_text())
    base = SystemConfig.load(prepared / "base-config.json")
    conditions = {condition["id"]: condition for condition in protocol["conditions"]}
    paths = list(map(Path, explicit_runs)) if explicit_runs else sorted(Path(runs_root).rglob("run.json"))
    matches = {key: [] for key in conditions}
    for path in paths:
        raw = json.loads(path.read_text())
        if (raw.get("experiment_job_id") in matches and raw.get("status") == "completed"
                and raw.get("backend") == "native"):
            matches[raw["experiment_job_id"]].append(path.resolve())
    missing = [key for key, paths in matches.items() if not paths]
    ambiguous = {key: list(map(str, paths)) for key, paths in matches.items() if len(paths) > 1}
    if missing or ambiguous:
        return {"status": "WAITING_FOR_V2_DECLARED_CONDITIONS" if missing else "AMBIGUOUS_COMPLETED_RUNS",
                "missing_ids": missing, "ambiguous": ambiguous,
                "available_conditions": sum(bool(paths) for paths in matches.values()),
                "required_conditions": len(matches), "fitted_config_created": False, "gpu_started": False}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Immutable v2 fit output already exists: " + str(output))
    sources = [{**condition, "path": str(matches[key][0]), "sha256": digest(matches[key][0])}
               for key, condition in conditions.items()]
    config, report = fit_calibrated_replay(base, frozen, sources, protocol)
    save(output / "sources.json", sources)
    save(output / "system-config.json", config)
    save(output / "calibration-report.json", report)
    save(output / "manifest.json", {"schema": "gsdeploy.replay_v2_fit_bundle.v1",
        "files": {path.name: digest(path) for path in output.iterdir() if path.is_file()},
        "prepared_manifest": {"path": str(prepared / "manifest.json"), "sha256": digest(prepared / "manifest.json")},
        "gpu_started": False, "fitted_config_created": True, "status": report["status"],
        "scope": "Frozen v2 mathematical predictor; failure preserved, never retuned on new holdouts"})
    return {"status": report["status"], "output": str(output), "fitted_config_created": True,
            "gpu_started": False, "version_sha256": report["version_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    declaration = commands.add_parser("prepare", help="Freeze standalone baselines/protocol; no run inspection or fit")
    declaration.add_argument("--base-packet", type=Path, required=True)
    declaration.add_argument("--legacy-v1-protocol", type=Path, required=True)
    declaration.add_argument("--holdout-packet", type=Path, required=True)
    declaration.add_argument("--output", type=Path, required=True)
    fit = commands.add_parser("fit", help="Explicitly opt in after all ten declared conditions finish")
    fit.add_argument("--prepared", type=Path, required=True)
    fit.add_argument("--runs-root", type=Path, required=True)
    fit.add_argument("--runs", type=Path, nargs="+")
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = (prepare(args.base_packet, args.legacy_v1_protocol, args.holdout_packet, args.output)
              if args.action == "prepare" else fit_from_completed(args.prepared, args.runs_root, args.output, args.runs))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
