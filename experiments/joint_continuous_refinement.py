"""Experiment B, run only after Experiment A has produced its frozen summary."""

import argparse
import copy
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous import JointMILPContinuousRefinement
from experiments.joint_scalability import ROOT, DEFAULT_SOURCE, run, source_manifest, subset_config, save
from experiments.diagnose_continuous_phase1 import assignment


def off_grid(c,routing):
    """Membership diagnostic only; never used to quantize a search point."""
    distances={}
    for task,row in routing.items():
        distances[task]=min(max(abs(q/c.workload[task]*d-round(q/c.workload[task]*d))/d
                                for q in row.values()) for d in range(1,c.route_grid+1))
    return dict(outside_old_grid=any(v>1e-12 for v in distances.values()),
                distance_tolerance=1e-12,per_task_rational_distance=distances)


def routing_audit(c,result):
    initial=result['initial_winner']
    if initial is None: return dict(status='NO_VALID_INITIAL_WINNER')
    ref=assignment(c,initial['deployment'],initial['routing'])
    final=assignment(c,result['winner']['deployment'],result['winner']['routing'])
    refworkers=[r['worker'] for r in ref['requests']]
    rows=[]
    for stage in result['histories']:
        for row in stage['history']:
            fifo=assignment(c,stage['deployment'],row['routing_qps'])
            rows.append(dict(stage=stage['stage'],deployment=stage['deployment'],routing=row['routing_qps'],
                seconds=row['elapsed_seconds'],utility=row['best_feasible_utility'],
                constraint_state=row['constraint_state'],prediction_calls=row['full_prediction_calls'],
                assignment_sha256=fifo['assignment_sha256'],
                changed_requests=sum(a!=b['worker'] for a,b in zip(refworkers,fifo['requests'])),
                **off_grid(c,row['routing_qps'])))
    return dict(initial_fifo=ref,final_fifo=final,
        final_changed_requests=sum(a!=b['worker'] for a,b in zip(refworkers,final['requests'])),
        final_off_grid=off_grid(c,result['winner']['routing']),rows=rows,
        best_off_grid_utility=max((r['utility'] for r in rows if r['outside_old_grid'] and r['utility'] is not None),default=None))


def main(args):
    a=Path(args.experiment_a)
    if not (a/'summary.json').exists():
        raise ValueError('Complete Experiment A before Experiment B')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    before=source_manifest();save(out/'source-manifest.json',before)
    for path in list((ROOT/'gsdeploy').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'experiments/joint_scalability.py',
            ROOT/'tests/test_continuous_refinement.py']:
        target=out/'source-snapshot'/path.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(path,target)
    selected=['p2-n1-K2','p5-n1-K2','p5-n2-K2','p7-n1-K2','p7-n1-K8',
              'p5-n2-K4-trace43','p5-n2-K4-trace45']
    cases=[c for c in json.loads((a/'plan.json').read_text()) if c['name'] in (args.cases or selected)]
    save(out/'plan.json',dict(cases=cases,seconds=args.seconds,max_routes=args.routes,seeds=args.seeds,
        initial_fraction=args.initial_fraction,neighbor_limit=4,fixed_fraction_with_neighbors=.4,
        initial_K='same as corresponding joint baseline; not chosen after observing refinement',
        no_new_profiles=True,no_gpu=True,no_gp_bo=True,no_mads=True))
    base=SystemConfig.load(ROOT/args.source)
    summary=[];started=time.perf_counter()
    for case in cases:
        name,states=case['name'],case['states']
        c=subset_config(base,case['models'],case['K'],args.seconds,seed=case['trace_seed'])
        save(out/f'{name}-config.json',c.as_dict())
        save(out/f'{name}-domain.json',json.loads((a/f'{name}-domain.json').read_text()))
        jobs=[(algorithm,seed) for seed in args.seeds for algorithm in
              ('joint','de_milp','refine_local','refine_de','refine_de_neighbors')]
        random.Random(20261101).shuffle(jobs)
        for algorithm,seed in jobs:
            if algorithm in {'joint','de_milp'}:
                raw=run(c,states,algorithm,args.seconds,args.routes,seed)
                row=dict(case=name,algorithm=algorithm,seed=seed,seconds=raw['seconds'],utility=raw['utility'],
                    predictions=raw['metrics']['full_prediction_calls'],
                    milp_calls=raw['metrics'].get('milp_calls',raw['metrics'].get('milp_solve_calls',0)),
                    route_evaluations=raw['metrics'].get('outer_evaluations',0),
                    certified=raw['certified'],status=raw['status'],winner=raw['winner'],raw=raw,
                    convergence=raw['timing']['evaluations'])
                # The public run timers start after common setup. Preserve
                # that offset explicitly when computing time-to-target.
                row['convergence_offset_seconds']=raw['setup_seconds']
            else:
                raw=JointMILPContinuousRefinement(c,states,seconds=args.seconds,seed=seed,max_routes=args.routes,
                    initial_fraction=args.initial_fraction,
                    method='local' if algorithm=='refine_local' else 'de',
                    neighborhood=algorithm=='refine_de_neighbors',neighbor_limit=4).solve()
                row=dict(case=name,algorithm=algorithm,seed=seed,seconds=raw['total_seconds'],
                    utility=raw['winner']['utility'] if raw['winner'] else None,
                    predictions=raw['budget']['full_prediction_calls'],milp_calls=raw['budget']['milp_solve_calls'],
                    route_evaluations=raw['budget']['outer_evaluations'],certified=False,status=raw['status'],
                    winner=raw['winner'],raw=raw,convergence=raw['convergence'],convergence_offset_seconds=0.,
                    routing_audit=routing_audit(c,raw))
                if raw['initial_winner']:
                    assert row['utility']>=raw['initial_winner']['utility']
                assert all(h['milp_calls']==0 for h in raw['histories'])
                assert raw['budget']['milp_solve_calls']==raw['initial_counts']['milp']
                assert raw['budget']['full_prediction_calls']==raw['initial_counts']['predictions']+sum(h['predictions'] for h in raw['histories'])
                if not raw['neighborhood_enabled'] and raw['winner']:
                    assert raw['winner']['deployment']==raw['initial_winner']['deployment']
            row['input_sha256']=raw.get('input_sha256',None)
            if row['input_sha256'] is None:
                from gsdeploy.continuous.contracts import digest
                row['input_sha256']=digest(dict(config=c.as_dict(),states=states))
            row['time_limit_seconds']=args.seconds
            if row['winner']:
                row['winner_off_grid']=off_grid(c,row['winner']['routing'])
                row['fifo']=assignment(c,row['winner']['deployment'],row['winner']['routing'])
            save(out/f'{name}-{algorithm}-seed{seed}.json',row)
            summary.append({k:row[k] for k in ('case','algorithm','seed','seconds','utility','predictions','milp_calls',
                'route_evaluations','certified','status','input_sha256')} |
                dict(improvement=raw.get('utility_improvement'),
                     initial_seconds=raw.get('joint_seconds'),
                     changed_requests=row.get('routing_audit',{}).get('final_changed_requests'),
                     off_grid=row.get('winner_off_grid',{}).get('outside_old_grid')))
            print(name,algorithm,seed,round(row['seconds'],3),row['predictions'],row['utility'],flush=True)
    assert before==source_manifest()
    save(out/'summary.json',dict(runs=len(summary),seconds=time.perf_counter()-started,source_unchanged=True,rows=summary))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--experiment-a',default='artifacts/joint-scalability-A-v1-20261101')
    p.add_argument('--output',required=True)
    p.add_argument('--seconds',type=float,default=8.)
    p.add_argument('--routes',type=int,default=128)
    p.add_argument('--initial-fraction',type=float,default=1.)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,29,47])
    p.add_argument('--cases',nargs='+')
    main(p.parse_args())
