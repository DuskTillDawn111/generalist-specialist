"""Prepare, validate, explicitly run, or report contract-driven confirmation."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy.spillover_confirmation import prepare, validate, run, report
from gsdeploy.static_study import read


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    subs=parser.add_subparsers(dest="action",required=True)
    p=subs.add_parser("prepare")
    for key in ("source-packet","config","candidates","contract","runtime-root","output"):
        p.add_argument("--"+key,required=True,type=Path)
    p.add_argument("--checkpoint-report",type=Path)
    p=subs.add_parser("validate");p.add_argument("--packet",required=True,type=Path)
    p=subs.add_parser("run");p.add_argument("--packet",required=True,type=Path)
    p.add_argument("--job",required=True);p.add_argument("--output",required=True,type=Path)
    p.add_argument("--execute",action="store_true")
    p=subs.add_parser("report");p.add_argument("--packet",required=True,type=Path)
    p.add_argument("--runs",nargs="+",required=True,type=Path);p.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    if args.action=="prepare":
        value=prepare(args.source_packet,args.config,args.candidates,args.contract,args.runtime_root,args.output,
                      checkpoint_report=args.checkpoint_report)
        print(json.dumps(dict(status=value["status"],jobs=len(value["jobs"]),gpu_started=False)))
    elif args.action=="validate":
        value=validate(read(args.packet));print(json.dumps(value))
        if not value["valid"]:raise SystemExit(1)
    elif args.action=="run":
        _,utility=run(read(args.packet),args.job,args.output,execute=args.execute)
        print(json.dumps(dict(status="VERIFIED",utility=utility)))
    else:
        value=report(args.packet,args.runs,args.output)
        print(json.dumps({k:value[k] for k in ("status","completed_jobs","planned_jobs","gpu_started")}))


if __name__=="__main__":
    main()
