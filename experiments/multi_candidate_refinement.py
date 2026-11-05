"""Step 1--3 study, with a later explicit decision before neighbourhood work."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil
import sys
import time
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.continuous import (Budget, ImmutableContinuousProblem, SessionPredictorOracle,
    ContinuousRoutingEvaluator, DERoutingOptimizer)
from gsdeploy.continuous.multi_candidate import MultiCandidateContinuousRefinement, JointDeploymentCandidates
from gsdeploy.continuous.contracts import digest
from experiments.joint_scalability import ROOT, DEFAULT_SOURCE, source_manifest, subset_config, save
from experiments.diagnose_continuous_phase1 import assignment


def original_run(c,states,algorithm,seconds,predictions,routes,seed):
    b=Budget(seconds,predictions,routes)
    problem=ImmutableContinuousProblem(c,replica_options=states)
    with SessionPredictorOracle(problem) as oracle:
        setup=time.perf_counter()-b.start
        ex=JointDeploymentCandidates(problem,oracle,b)
        if algorithm=='joint':
            candidates=ex.generate(1)
            return dict(algorithm='joint',winner=ex.best,candidates=candidates,events=ex.events,
                joint_solves=ex.solves,stages=[],budget=b.report(),setup_seconds=setup,
                initial_winner=candidates[0] if candidates else None,
                status=ex.solves[-1]['status'] if ex.solves else 'NOT_STARTED',
                grid_certified=ex.solves[-1]['optimality_certified'] if ex.solves else False,
                continuous_optimality=False,total_seconds=time.perf_counter()-b.start)
        ex.stage='de_milp'
        original=oracle.predict
        def observe(problem,deployment,routing,trace,budget,deadline=None):
            p=original(problem,deployment,routing,trace,budget,deadline)
            ex.observe(deployment,routing,p)
            return p
        e=ContinuousRoutingEvaluator(problem,oracle,'milp_bounds',b)
        with patch.object(oracle,'predict',observe):
            raw=DERoutingOptimizer(e,seed=seed,popsize=3,maxiter=100).solve()
        raw.update(algorithm='de_milp',winner=ex.best,events=ex.events,stages=[],
            history=e.history,setup_seconds=setup,budget=b.report(),total_seconds=time.perf_counter()-b.start)
        return raw


def audit_routes(c,raw):
    """Full request mappings are exported after the timed optimization."""
    known={};rows=[]
    for event in raw['events']:
        key=digest([event['deployment'],event['routing']])
        if key not in known:known[key]=assignment(c,event['deployment'],event['routing'])
        rows.append(dict(seconds=event['seconds'],predictions=event['predictions'],stage=event['stage'],
            utility=event['utility'],key=key,assignment_sha256=known[key]['assignment_sha256']))
    for stage in raw.get('stages',[]):
        for row in stage['history']:
            key=digest([stage['deployment'],row['routing_qps']])
            if key not in known:known[key]=assignment(c,stage['deployment'],row['routing_qps'])
    return dict(assignments=known,rows=rows)


def execute(c,states,algorithm,args,seed,k=1,strategy='utility'):
    if algorithm in {'joint','de_milp'}:
        return original_run(c,states,algorithm,args.seconds,args.predictions,args.routes,seed)
    return MultiCandidateContinuousRefinement(c,states,k=k,strategy=strategy,seed=seed,
        seconds=args.seconds,max_predictions=args.predictions,max_routes=args.routes,
        per_candidate_routes=args.per_candidate_routes).solve()


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=SystemConfig.load(ROOT/args.source)
    manifest=source_manifest();save(out/'source-manifest.json',manifest)
    paths=list((ROOT/'gsdeploy').rglob('*.py'))+[Path(__file__).resolve(),ROOT/'tests/test_multi_candidate.py']
    for path in paths:
        dest=out/'source-snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,dest)
    shutil.copy2(ROOT/args.source,out/'frozen-source-config.json')
    save(out/'plan.json',dict(mode=args.mode,seconds=args.seconds,predictions=args.predictions,routes=args.routes,
        per_candidate_routes=args.per_candidate_routes,seeds=args.seeds,traces=args.traces,
        K=args.K,ks=args.ks,strategies=args.strategies,
        joint_fresh_solver_per_exclusion=True,exact_input_cache=True,no_cross_route_cuts=True,
        no_neighborhood_before_explicit_step3_decision=True))
    summary=[];started=time.perf_counter()
    models=['generalist_2','text','image','ocr','asr']
    for n in args.replicas:
        for trace in args.traces:
            case=f'p5-n{n}-K{args.K}-trace{trace}'
            c=replace(subset_config(base,models,args.K,args.seconds,seed=trace),milp_max_iterations=1000)
            states={m:list(range(1,n+1)) for m in models}
            save(out/f'{case}-config.json',c.as_dict());save(out/f'{case}-states.json',states)
            jobs=[]
            for seed in args.seeds:
                if args.mode=='full':jobs.extend((a,1,'utility',seed) for a in ('joint','de_milp'))
                jobs.extend(('multi',k,strategy,seed) for strategy in args.strategies for k in args.ks)
            random.Random(20261102+n+trace).shuffle(jobs)
            for algorithm,k,strategy,seed in jobs:
                raw=execute(c,states,algorithm,args,seed,k,strategy)
                label=algorithm if algorithm!='multi' else f'{strategy}-k{k}'
                name=f'{case}-{label}-s{seed}'
                row=dict(case=case,algorithm=label,seed=seed,k=k,strategy=strategy,
                    trace_seed=trace,config_sha256=digest(c.as_dict()),states=states,
                    seconds=raw['total_seconds'],utility=raw['winner']['utility'] if raw['winner'] else None,
                    predictions=raw['budget']['full_prediction_calls'],milp_calls=raw['budget']['milp_solve_calls'],
                    route_evaluations=raw['budget']['outer_evaluations'],status=raw['status'],raw=raw,
                    budget_limits=dict(seconds=args.seconds,predictions=args.predictions,routes=args.routes),
                    routing_audit=audit_routes(c,raw))
                if raw.get('initial_winner'):
                    assert row['utility']>=raw['initial_winner']['grid_utility']
                assert row['predictions']<=args.predictions
                assert row['route_evaluations']<=args.routes
                assert all(stage['counts']['milp']==0 for stage in raw.get('stages',[]))
                save(out/f'{name}.json',row)
                summary.append({k:row[k] for k in row if k not in {'raw','routing_audit'}})
                save(out/'summary.json',dict(rows=summary,runs=len(summary),seconds=time.perf_counter()-started))
                print(name,round(row['seconds'],3),row['utility'],row['predictions'],
                    [(s['generation_index'],round(s['improvement'],8)) for s in raw.get('stages',[])],flush=True)
    assert source_manifest()==manifest
    save(out/'complete.json',dict(runs=len(summary),seconds=time.perf_counter()-started,
        source_unchanged=True,neighborhood_executed=False))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',default=DEFAULT_SOURCE);p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=['pilot','full'],default='pilot')
    p.add_argument('--seconds',type=float,default=16.)
    p.add_argument('--predictions',type=int,default=400);p.add_argument('--routes',type=int,default=384)
    p.add_argument('--per-candidate-routes',type=int,default=128)
    p.add_argument('--seeds',nargs='+',type=int,default=[11,29])
    p.add_argument('--traces',nargs='+',type=int,default=[42])
    p.add_argument('--replicas',nargs='+',type=int,default=[1,2])
    p.add_argument('--ks',nargs='+',type=int,default=[1,3,5])
    p.add_argument('--strategies',nargs='+',choices=['utility','diversity'],default=['utility'])
    p.add_argument('--K',type=int,default=1)
    main(p.parse_args())
