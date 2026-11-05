"""Normalize a complete training capture without collecting or running GPU work."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.paper_capture import normalize_capture
from gsdeploy.paper_forecast import MODE,canonical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--captures',type=Path,required=True,help='Canonical deployment -> list of complete native capture wrappers')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();c=SystemConfig.load(args.config)
    entries={canonical(e['deployment']):e for e in c.catalog[MODE]['deployments']}
    captures=json.loads(args.captures.read_text());result={}
    for key,rows in captures.items():
        if key not in entries:raise ValueError('Capture references an undeclared deployment')
        result[key]=[normalize_capture(row,entries[key]) for row in rows]
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'training.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    report=dict(status='NORMALIZED_COMPLETE_TRAINING_CAPTURE',GPU_started=False,
        source_files={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.config,args.captures)},
        runs=sum(map(len,result.values())),physical_accuracy_qualified=False)
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
