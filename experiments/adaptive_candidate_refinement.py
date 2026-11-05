"""Controlled candidate screening/scheduling ablations after correctness tests."""
import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import random
import shutil
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous.adaptive_candidate import AdaptiveCandidateRefinementOptimizer
from gsdeploy.continuous.multi_candidate import MultiCandidateContinuousRefinement
from gsdeploy.continuous.contracts import digest
from experiments.joint_scalability import ROOT,DEFAULT_SOURCE,source_manifest,subset_config,save
from experiments.multi_candidate_refinement import original_run,audit_routes

BUDGETS={'predictions':dict(seconds=60.,predictions=160,routes=512),
         'wall':dict(seconds=8.,predictions=1000,routes=512),
         'outer':dict(seconds=60.,predictions=1000,routes=128)}
METHODS=['joint','top1','uniform','safe_uniform','adaptive','safe_adaptive','uniform_restart','priority']


def execute(c,states,method,k,seed,limits):
    common=dict(seconds=limits['seconds'],max_predictions=limits['predictions'],max_routes=limits['routes'],seed=seed)
    if method=='joint':return original_run(c,states,'joint',limits['seconds'],limits['predictions'],limits['routes'],seed)
    if method in {'top1','uniform'}:
        return MultiCandidateContinuousRefinement(c,states,k=1 if method=='top1' else k,**common).solve()
    policy={'safe_uniform':'legacy_uniform','adaptive':'adaptive','safe_adaptive':'adaptive',
            'uniform_restart':'uniform','priority':'priority'}[method]
    return AdaptiveCandidateRefinementOptimizer(c,states,k=k,safe_filter=method.startswith('safe_'),policy=policy,**common).solve()


def main(args):
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    manifest=source_manifest();save(out/'source-manifest.json',manifest)
    for path in list((ROOT/'gsdeploy').rglob('*.py'))+[Path(__file__),ROOT/'tests/test_adaptive_candidate.py']:
        dest=out/'source-snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    base=SystemConfig.load(ROOT/args.source)
    save(out/'plan.json',dict(axes=args.axes,budgets=BUDGETS,traces=args.traces,seeds=args.seeds,ks=args.ks,
        methods=args.methods,screen_routes=16,min_screen_routes=8,screening_fraction=.25,
        chunk_routes=48,max_extra_routes=112,stagnation_rounds=2,exploration_period=4,
        adaptive_score='current_U+2*observed_gain+.02/(1+routes/max(1,dimension))',
        safe_bound='route-independent per-request relaxation',
        restart='fresh unchanged DE with best-route warm start and shared exact cache',
        old_uniform_per_candidate_routes=128,no_neighborhood=True,algorithm_sources=manifest))
    jobs=[]
    for axis in args.axes:
        for trace in args.traces:
            for seed in args.seeds:
                for method in args.methods:
                    for k in ([1] if method in {'joint','top1'} else args.ks):jobs.append((axis,trace,seed,method,k))
    random.Random(20261102).shuffle(jobs)
    summary=[];started=time.perf_counter()
    for axis,trace,seed,method,k in jobs:
        limits=BUDGETS[axis]
        case=f'p5-n1-K1-trace{trace}'
        c=replace(subset_config(base,['generalist_2','text','image','ocr','asr'],1,60.,seed=trace),milp_max_iterations=1000)
        states={m:[1] for m in c.catalog['models']}
        save(out/f'{case}-config.json',c.as_dict())
        raw=execute(c,states,method,k,seed,limits)
        if raw.get('initial_winner'):
            assert raw['winner']['utility']>=raw['initial_winner']['grid_utility']-1e-12
        assert raw['budget']['full_prediction_calls']<=limits['predictions']
        assert raw['budget']['outer_evaluations']<=limits['routes']
        assert all(s['counts']['milp']==0 for s in raw.get('stages',[]))
        row=dict(axis=axis,case=case,method=method,k=k,seed=seed,trace_seed=trace,states=states,
            config_sha256=digest(c.as_dict()),budget_limits=limits,seconds=raw['total_seconds'],
            utility=raw['winner']['utility'] if raw.get('winner') else None,
            predictions=raw['budget']['full_prediction_calls'],milp=raw['budget']['milp_solve_calls'],
            routes=raw['budget']['outer_evaluations'],status=raw['status'],raw=raw,
            routing_audit=audit_routes(c,raw))
        filename=f'{axis}-{case}-{method}-k{k}-s{seed}.json'
        save(out/filename,row)
        summary.append({key:value for key,value in row.items() if key not in {'raw','routing_audit'}}|dict(file=filename))
        save(out/'summary.json',dict(rows=summary,runs=len(summary),seconds=time.perf_counter()-started))
        print(axis,trace,seed,method,k,round(row['seconds'],3),row['predictions'],row['routes'],row['utility'],
              [(s['generation_index'],round(s['improvement'],7),s.get('state',s.get('status'))) for s in raw.get('stages',[])],flush=True)
        del row,raw,c
        gc.collect()
    assert source_manifest()==manifest
    save(out/'complete.json',dict(runs=len(summary),seconds=time.perf_counter()-started,
        optimizer_sources_unchanged=True,neighborhood_executed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--source',default=DEFAULT_SOURCE)
    p.add_argument('--axes',nargs='+',choices=list(BUDGETS),default=['predictions','wall'])
    p.add_argument('--traces',type=int,nargs='+',default=[43]);p.add_argument('--seeds',type=int,nargs='+',default=[11,29,47])
    p.add_argument('--ks',type=int,nargs='+',default=[3,5]);p.add_argument('--methods',nargs='+',choices=METHODS,default=METHODS)
    main(p.parse_args())
