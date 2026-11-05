"""Fit explicitly normalized training observations; never collects GPU data."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.paper_training import fit_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--training',type=Path,required=True,help='JSON map: canonical deployment -> list of complete training documents')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--interpolate-contexts',action='store_true',help='Opt in to within-hull context interpolation, not extrapolation')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    config,report=fit_config(SystemConfig.load(args.config),json.loads(args.training.read_text()),
                             interpolate_contexts=args.interpolate_contexts)
    report['inputs']={str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.config,args.training)}
    for name,value in [('system-config.json',config),('report.json',report)]:
        (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
