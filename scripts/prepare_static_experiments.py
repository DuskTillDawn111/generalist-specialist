"""Prepare/validate static GPU packets on CPU; run requires explicit --execute."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from gsdeploy.static_study import prepare_experiments, validate_packet, execute_job, freeze_evaluator, queue_packet, _file, read
from gsdeploy.config import SystemConfig
from gsdeploy.io import save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    prepare = modes.add_parser("prepare")
    prepare.add_argument("--inputs", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--config", type=Path)
    validate = modes.add_parser("validate")
    validate.add_argument("--packet", type=Path, required=True)
    run = modes.add_parser("run")
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--job", required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    freeze = modes.add_parser("freeze-evaluator")
    freeze.add_argument("--packet", type=Path, required=True)
    freeze.add_argument("--runs", type=Path, nargs="+", required=True)
    freeze.add_argument("--output", type=Path, required=True)
    queue = modes.add_parser("queue")
    queue.add_argument("--packet", type=Path, required=True)
    queue.add_argument("--config", type=Path, required=True)
    queue.add_argument("--plans", type=Path, required=True)
    queue.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        packet = prepare_experiments(args.inputs, args.output, config_path=args.config)
        result = {"status": packet["status"], "calibration_jobs": packet["calibration"]["selected"],
                  "confirmation": packet["confirmation"]["status"], "gpu_started": False}
    elif args.mode == "validate":
        result = validate_packet(read(args.packet))
    elif args.mode == "freeze-evaluator":
        result = freeze_evaluator(read(args.packet), args.runs, args.output)
    elif args.mode == "queue":
        packet = read(args.packet)
        validation = validate_packet(packet)
        if not validation["valid"]:
            raise ValueError("Packet bindings changed")
        result = queue_packet(packet, SystemConfig.load(args.config), read(args.plans),
                              [_file(args.config), _file(args.plans)])
        save(args.output, result)
        result = {"queued_jobs": len(result["jobs"]), "gpu_started": False}
    else:
        raw = execute_job(read(args.packet), args.job, args.output, execute=args.execute)
        result = {"status": raw["status"], "job": args.job}
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
