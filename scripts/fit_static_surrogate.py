"""Fit a CPU-only frozen surrogate after every predeclared calibration finishes."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.calibration import fit_calibrated_surrogate
from gsdeploy.config import SystemConfig
from gsdeploy.evidence import digest
from gsdeploy.io import save


def fit_from_completed(protocol_path, packet_path, runs_root, output, explicit_runs=None):
    """Return WAITING without creating a fitted output if any condition is absent."""
    protocol_path, packet_path = Path(protocol_path).resolve(), Path(packet_path).resolve()
    protocol, packet = json.loads(protocol_path.read_text()), json.loads(packet_path.read_text())
    requested = protocol["training_ids"] + protocol["validation_ids"]
    jobs = {j["id"]: j for j in packet["jobs"]}
    if len(set(requested)) != len(requested) or not set(requested).issubset(jobs):
        raise ValueError("Declared protocol conditions do not match execution packet")
    provenance = protocol.get("execution_packet_provenance")
    if provenance and provenance["sha256"] != digest(packet_path):
        raise ValueError("Execution packet differs from predeclared protocol binding")
    candidates = list(map(Path, explicit_runs)) if explicit_runs else sorted(Path(runs_root).rglob("run.json"))
    matches = {key: [] for key in requested}
    ignored = []
    for path in candidates:
        raw = json.loads(path.read_text())
        identity = raw.get("experiment_job_id")
        if identity not in matches:
            continue
        if raw.get("status") != "completed" or raw.get("backend") != "native":
            ignored.append({"path": str(path.resolve()), "status": raw.get("status"), "id": identity})
            continue
        matches[identity].append(path.resolve())
    missing = [key for key, paths in matches.items() if not paths]
    ambiguous = {key: list(map(str, paths)) for key, paths in matches.items() if len(paths) > 1}
    if missing or ambiguous:
        return {"status": "WAITING_FOR_DECLARED_CALIBRATIONS" if missing else "AMBIGUOUS_COMPLETED_RUNS",
                "missing_ids": missing, "ambiguous": ambiguous, "ignored_incomplete": ignored,
                "available_conditions": sum(bool(paths) for paths in matches.values()),
                "required_conditions": len(requested), "fitted_config_created": False, "gpu_started": False}
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Immutable fit output already exists: " + str(output))
    packet_sha = digest(packet_path)
    sources = [{"id": key, "path": str(matches[key][0]), "sha256": digest(matches[key][0]),
                "packet_path": str(packet_path), "packet_sha256": packet_sha} for key in requested]
    versions = sorted({jobs[key]["identity"]["runtime_code_sha256"] for key in requested})
    config, report = fit_calibrated_surrogate(SystemConfig(**packet["config"]), sources,
        protocol["training_ids"], protocol["validation_ids"], versions,
        thresholds=protocol["thresholds"], protocol=protocol)
    save(output / "sources.json", sources)
    save(output / "system-config.json", config)
    save(output / "calibration-report.json", report)
    save(output / "manifest.json", {"schema": "gsdeploy.calibrated_fit_bundle.v1",
        "files": {p.name: digest(p) for p in output.iterdir() if p.is_file()},
        "protocol": {"path": str(protocol_path), "sha256": digest(protocol_path)},
        "execution_packet": {"path": str(packet_path), "sha256": packet_sha},
        "accepted_runtime_versions": versions, "gpu_started": False,
        "fitted_config_created": True, "status": report["status"],
        "scope": "Frozen mathematical predictor; validation failure is preserved, not retuned"})
    return {"status": report["status"], "output": str(output), "sources": len(sources),
            "fitted_config_created": True, "gpu_started": False,
            "version_sha256": report["version_sha256"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--execution-packet", type=Path, required=True,
                        help="Explicitly accept this bound historical execution revision and its runtime hashes")
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", help="Explicit run selection when retries create multiple completed runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit_from_completed(args.protocol, args.execution_packet, args.runs_root, args.output, args.runs)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
