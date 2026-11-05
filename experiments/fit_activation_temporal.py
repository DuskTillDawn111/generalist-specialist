"""Fit an unvalidated causal-age alternative without overwriting prior fits."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.forecasting.native_training import digest
from gsdeploy.forecasting.native_temporal import fit_temporal_progress


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fit',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--ridge',type=float,default=.01)
    p.add_argument('--timing-objective',action='store_true')
    p.add_argument('--age-descriptor',choices=('strict-v1','bounded-terminal-v1'),default='strict-v1')
    args=p.parse_args()
    fit=json.loads(args.fit.read_text())
    for source in fit['source_receipts']:
        path,sha=source.rsplit(':sha256:',1)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=sha:raise ValueError('Parent training raw receipt changed')
    if not 0<args.ridge<=10:raise ValueError('Explicit positive temporal regularization required')
    spec=fit_temporal_progress(fit,ridge=args.ridge,timing_objective=args.timing_objective,age_descriptor=args.age_descriptor)
    result=copy.deepcopy(fit);law=result['profiles']['native_activity_law']
    law['original_progress_diagnostics']=law['fit_diagnostics'];law['progress_groups']=[]
    law['temporal_progress']=spec;law['fit_diagnostics']=spec['fit_diagnostics']
    result['parent_fit_sha256']=digest(fit);result['parent_fit_source']=str(args.fit.resolve())
    result['temporal_progress_qualified']=False
    with args.output.open('x') as out:json.dump(result,out,ensure_ascii=False,indent=2,allow_nan=False);out.write('\n')
    print(json.dumps(dict(output=str(args.output),admission_enabled=False,
        diagnostics=spec['fit_diagnostics']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
