"""Prepare/validate an independent paired confirmation; native runs are explicit."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy.confirmation import build_confirmation_packet, validate_confirmation_packet, run_confirmation_job
from gsdeploy.static_study import read


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--source-packet", type=Path, required=True)
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--search", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--checkpoint-bindings", type=Path,
                         help="CPU checkpoint report for candidate models absent from calibration runs")
    validate = commands.add_parser("validate")
    validate.add_argument("--packet", type=Path, required=True)
    run = commands.add_parser("run")
    run.add_argument("--packet", type=Path, required=True)
    run.add_argument("--job", required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.mode == "prepare":
        packet = build_confirmation_packet(args.source_packet, args.config, args.search, args.output,
                                          checkpoint_bindings_path=args.checkpoint_bindings)
        result = {"status": packet["status"], "jobs": len(packet["jobs"]),
                  "candidates": len(packet["candidates"]), "gpu_started": False}
    elif args.mode == "validate":
        result = validate_confirmation_packet(read(args.packet))
    else:
        result = run_confirmation_job(read(args.packet), args.job, args.output_root, execute=args.execute)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
