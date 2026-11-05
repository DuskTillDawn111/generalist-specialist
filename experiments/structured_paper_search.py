"""Independent paper/FIFO structured optimizer; no GPU execution."""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from gsdeploy.config import SystemConfig
from gsdeploy.continuous.paper_structured import PaperStructuredOptimizer
from gsdeploy.profiled_resource_search import ProfiledResourceConfigurationSearch


def main():
    p=argparse.ArgumentParser(description=__doc__)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--config',type=Path)
    group.add_argument('--resource-candidates',type=Path,help='List of id/config path/profile_bindings objects')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seconds',type=float,default=10.)
    p.add_argument('--max-predictions',type=int,default=96)
    p.add_argument('--max-routes',type=int,default=96)
    p.add_argument('--max-candidates',type=int)
    p.add_argument('--method',choices=['coordinate','de'],default='coordinate')
    p.add_argument('--seed',type=int,default=11)
    p.add_argument('--without-joint-seed',action='store_true')
    args=p.parse_args()
    options=dict(seconds=args.seconds,max_predictions=args.max_predictions,max_routes=args.max_routes,
                 method=args.method,seed=args.seed,joint_seed=not args.without_joint_seed)
    if args.config:
        solver=PaperStructuredOptimizer(SystemConfig.load(args.config),max_candidates=args.max_candidates,**options)
    else:
        if args.max_candidates is not None:p.error('--max-candidates applies only to a single configuration')
        rows=json.loads(args.resource_candidates.read_text())
        for row in rows:row['config']=SystemConfig.load(args.resource_candidates.resolve().parent/row['config'])
        solver=ProfiledResourceConfigurationSearch(rows,**options)
    # Reserve the output before doing potentially expensive work.
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    raw=solver.solve()
    (out/'result.json').write_text(json.dumps(raw,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    summary={k:raw.get(k) for k in ('algorithm','status','budget','counts','deployment_domain','fifo_stats',
        'deployment_milp_calls_after_seed','independent_accuracy_qualified','final_gain_eligible')}
    w=raw.get('winner')
    selected=w.get('winner',w) if w else None
    summary['winner']=({k:selected.get(k) for k in ('deployment','routing','utility')} if selected else None)
    if w and 'configuration_id' in w:summary['winner']['configuration_id']=w['configuration_id']
    summary['GPU_started']=False
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
