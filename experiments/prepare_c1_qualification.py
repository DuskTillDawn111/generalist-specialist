"""Prepare a targeted evidence packet. Does not collect or start workers."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsdeploy.config import SystemConfig
from gsdeploy.c1_qualification import build_packet, receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/system_reference_v2_20261103.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--historical-runs", nargs="*", type=Path)
    args = parser.parse_args()
    runs = args.historical_runs
    if runs is None:
        base = ROOT/"artifacts/static-study-gpu-20261022/v2-holdout-runs"
        runs = [base/f"{n}-attempt1/run.json" for n in
            ("v2_holdout_A_g2_image075_ocr100", "v2_holdout_B_g2_image100_ocr075")]
        runs = [p for p in runs if p.exists()]
    c = SystemConfig.load(args.config)
    packet = build_packet(c, historical_runs=runs)
    packet["sources"] = [receipt(args.config), receipt(__file__), receipt(ROOT/"gsdeploy/c1_qualification.py")]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"packet.json").write_text(json.dumps(packet, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(json.dumps(dict(status=packet["status"], scenarios=len(packet["scenarios"]),
                         blockers=packet["blockers"], output=str(args.output.resolve())), ensure_ascii=False))


if __name__ == "__main__":
    main()
