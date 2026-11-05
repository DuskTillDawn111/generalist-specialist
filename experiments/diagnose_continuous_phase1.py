"""Small performance study using an existing validated real-data replay model.

No model fitting, new measurements, synthetic profiles, or optimizer changes.
"""

import argparse
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from gsdeploy import DeploymentSystem, SystemConfig
from gsdeploy.calibration_v2 import validated_replay_spec
from gsdeploy.continuous import Budget, ContinuousProblem, ContinuousRouteDomain, ContinuousRoutingEvaluator, DERoutingOptimizer, LegacyPredictorOracle
from gsdeploy.continuous.contracts import digest
from gsdeploy.route_domain import fraction_rows
from gsdeploy.routing import Router
from gsdeploy.trace import expand_routes, make_trace, trace_digest
from experiments.continuous_diagnostic_timing import PhaseClock, diagnostics

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SOURCE='artifacts/static-study-gpu-20261022/search-v2-v5-K8/system-config.json'


def save(path,obj):
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False))


def subset(base,models,seconds,K):
    # Retain the entire quality document, calibrated spec and all state rows.
    # The original public scope gate verifies each retained baseline identity.
    data=dict(base.catalog)
    for key in ('models','capacity_factors','memory_table_mib'):
        data[key]={m:data[key][m] for m in models}
    c=replace(base,catalog=data,route_grid=K,milp_time_limit_seconds=seconds,
              milp_max_iterations=10000,search_max_states=1000000,milp_capacity_witness=False)
    validated_replay_spec(c.catalog,c.workload)
    return c


def domain_size(c,states,K):
    names=tuple(states)
    count=0
    for ns in itertools.product(*([0]+list(states[m]) for m in names)):
        x={m:n for m,n in zip(names,ns) if n}
        product=1
        for task in c.workload:
            eligible=tuple(m for m in x if task in c.catalog['models'][m]['supported_tasks'])
            product*=len(list(fraction_rows(eligible,K))) if eligible else 0
        count+=product
    edges={t:[m for m,s in c.catalog['models'].items() if t in s['supported_tasks']] for t in c.workload}
    return dict(tasks=len(c.workload),models=len(names),deployment_states=math.prod(1+len(v) for v in states.values()),
                routing_dimension=sum(len(v)-1 for v in edges.values()),grid_complete_identities=count,
                eligible_models=edges,replica_options=states)


def assignment(c,deployment,routing):
    trace=make_trace(c.workload,c.horizon_seconds,c.random_seed,c.arrival_process)
    aliases,rows=expand_routes(deployment,routing)
    router=Router(rows)
    result=[]
    counts={t:{} for t in c.workload}
    for i,(offset,task) in enumerate(trace):
        alias=router.choose(task)
        counts[task][alias]=counts[task].get(alias,0)+1
        result.append(dict(index=i,offset=offset,task=task,worker=alias,model=aliases[alias]))
    return dict(trace_sha256=trace_digest(trace),assignment_sha256=digest(result),counts=counts,requests=result)


def run_one(c,states,algorithm,seconds,outer,seed,*,route=None,profile=False,K=None):
    clock=PhaseClock() if profile else None
    context=diagnostics(clock) if profile else nullcontext()
    with context:
        if clock:
            clock.seconds={};clock.started=clock.last=time.perf_counter()
        started=time.perf_counter()
        if algorithm in ('dfs','joint_milp'):
            config=replace(c,solver='dfs' if algorithm=='dfs' else 'milp',route_grid=K or c.route_grid)
            raw=DeploymentSystem(config).plan(replica_options=states)
            w=raw['winner']
            winner=(dict(deployment=w['deployment'],routing=w['routing'],utility=w['score'],prediction=w['prediction'])
                    if w and w['prediction'].get('U') is not None else None)
            metrics=raw['metrics']
            predictions=metrics['full_prediction_calls']
            milp_calls=metrics.get('milp_calls',0)
            certified=raw['optimality_certified']
            history=[]
        else:
            problem=ContinuousProblem(c,replica_options=states)
            e=ContinuousRoutingEvaluator(problem,LegacyPredictorOracle(),
                  'enumeration' if algorithm=='fixed_enumeration' else 'milp',Budget(seconds,10000,outer))
            if algorithm.startswith('fixed_'):
                r=e.evaluate(route)
                raw=dict(status=r['status'],winner=r if r['feasible'] else None,budget=e.budget.report())
                certified=r['certified']
            else:
                raw=DERoutingOptimizer(e,seed=seed,popsize=3,maxiter=100,face_limit=128).solve()
                certified=False
            raw['evaluations']=e.history
            history=e.history
            w=raw['winner']
            winner=(dict(deployment=w['deployment'],routing=w['routing_qps'],utility=w['best_feasible_utility'],prediction=w['prediction']) if w else None)
            predictions=e.budget.counts['predictions']
            milp_calls=e.budget.counts['milp']
            metrics=e.budget.report()
        elapsed=time.perf_counter()-started
        phases=clock.report() if clock else None
    if algorithm in ('dfs','joint_milp'):
        raw.pop('problem',None) # One frozen config is saved per case, not per run.
    result=dict(algorithm=algorithm,profiled=profile,seconds=elapsed,predictions=predictions,
        milp_calls=milp_calls,status=raw['status'],certified=certified,
        utility=winner['utility'] if winner else None,winner=winner,metrics=metrics,
        timing=phases,raw=raw,outer_evaluations=len(history),
        problem_sha256=digest(dict(config=c.as_dict(),states=states)),
        trace_sha256=trace_digest(make_trace(c.workload,c.horizon_seconds,c.random_seed,c.arrival_process)))
    if winner:
        result['winner_routing_shares']={t:{m:q/c.workload[t] for m,q in row.items()} for t,row in winner['routing'].items()}
        result['fifo']=assignment(c,winner['deployment'],winner['routing'])
    return result


def route_scan(c,states,winner):
    """One-task-at-a-time frozen-deployment diagnostic, not a global proof."""
    if not winner:
        return dict(status='NO_REFERENCE_WINNER')
    problem=ContinuousProblem(c,replica_options=states)
    e=ContinuousRoutingEvaluator(problem,LegacyPredictorOracle(),budget=Budget(120,1000,1000))
    x=winner['deployment']; reference=winner['routing']
    rows=[]
    for t in c.workload:
        target=next(iter(reference[t])) if len(reference[t])==1 else None
        candidates=[m for m in x if t in c.catalog['models'][m]['supported_tasks'] and m!=target]
        for alt in candidates:
            for share in (0,1e-8,1e-6,.01,.125,.25,.5,.75,1):
                routing={k:dict(v) for k,v in reference.items()}
                if target is None:
                    continue
                routing[t]={m:c.workload[t]*v for m,v in ((target,1-share),(alt,share)) if v>0}
                p=e.candidate(x,routing)
                fifo=assignment(c,x,routing)
                rows.append(dict(task=t,alternative=alt,alternative_share=share,utility=p['utility'],state=p['constraint_state'],
                                 assignment_sha256=fifo['assignment_sha256'],counts=fifo['counts'],
                                 traces=p['traces']))
    return dict(scope='same winning deployment; one task perturbed; not continuous global optimality',rows=rows,budget=e.budget.report())


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source).resolve();base=SystemConfig.load(source)
    validated_replay_spec(base.catalog,base.workload)
    assert base.catalog['performance_mode']=='calibrated_replay_v2'
    assert not any(p.get('kind')=='synthetic_fixture' for p in base.catalog.get('colocation_profiles',{}).values())
    save(out/'source.json',dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        predictor_version=base.catalog['calibrated_replay_v2']['version_sha256'],status=base.catalog['calibrated_replay_v2']['status'],
        no_training=True,no_new_measurements=True,no_generated_profiles=True))
    protected={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'gsdeploy').rglob('*.py')}
    save(out/'source-manifest-before.json',protected)
    for path in list((ROOT/'gsdeploy').rglob('*.py'))+[Path(__file__).resolve(),
            ROOT/'experiments/continuous_diagnostic_timing.py',ROOT/'tests/test_continuous_diagnostics.py']:
        target=out/'source-snapshot'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(path.read_bytes())
    tasks=list(base.workload)
    gates=[]
    for n in range(1,len(tasks)+1):
        rates={t:base.workload[t] for t in tasks[:n]}
        try:
            validated_replay_spec(base.catalog,rates);reason='SUPPORTED'
        except ValueError as exc:
            reason=str(exc)
        gates.append(dict(tasks=n,workload=rates,status=reason))
    save(out/'task-scope-gates.json',gates)
    core=['generalist_2','text','image','ocr','asr']
    cases=[('pool2',['generalist_2','image'],1,2),
           ('pool3',['generalist_2','image','ocr'],1,2),
           ('pool5',core,1,2),
           ('pool7',list(base.catalog['models']),1,2),
           ('replicas2_pool5',core,2,2),
           ('replicas2_pool7',list(base.catalog['models']),2,2),
           ('grid4_pool5',core,1,4)]
    all_rows=[];details=[];references={}
    start=time.perf_counter()
    for case_index,(name,models,cap,K) in enumerate(cases):
        c=subset(base,models,args.seconds,K)
        states={m:[n for n in range(1,cap+1) if str(n) in c.catalog['capacity_factors'][m]] for m in models}
        save(out/(name+'-config.json'),dict(system=c.as_dict(),replica_options=states))
        size=domain_size(c,states,K);size['case']=name;details.append(size)
        save(out/'domains.json',details)
        for repeat in range(args.repeats):
            algorithms=['dfs','joint_milp','de_milp']
            shift=(case_index+repeat)%3;algorithms=algorithms[shift:]+algorithms[:shift]
            for algorithm in algorithms:
                row=run_one(c,states,algorithm,args.seconds,args.outer,11+18*repeat)
                row.update(case=name,repeat=repeat,domain=size)
                filename=f'{name}-{algorithm}-r{repeat}.json';save(out/filename,row)
                compact={k:row[k] for k in ('case','repeat','algorithm','seconds','predictions','milp_calls','status','certified','utility','outer_evaluations')}
                compact['raw_file']=filename
                all_rows.append(compact)
                with (out/'results.jsonl').open('a') as stream:stream.write(json.dumps(compact)+'\n')
                print(json.dumps(compact),flush=True)
                if row['winner'] and (name not in references or row['utility']>references[name]['utility']):
                    references[name]=row['winner']
        if name in ('pool2','pool5'):
            domain=ContinuousRouteDomain(ContinuousProblem(c,replica_options=states))
            routes=[{t:{'generalist_2':1.} for t in c.workload},
                    {t:{t if t in models else 'generalist_2':1.} for t in c.workload},
                    domain.decode([.3141592653589793]*domain.dimension)]
            if name=='pool5' and name in references:
                routes.append({t:{m:q/c.workload[t] for m,q in row.items()}
                               for t,row in references[name]['routing'].items()})
            fixed=[]
            for i,route in enumerate(routes):
                for algorithm in ('fixed_enumeration','fixed_milp'):
                    row=run_one(c,states,algorithm,args.seconds,args.outer,11,route=route,profile=True)
                    row.update(case=name,route_id=i)
                    save(out/f'{name}-route{i}-{algorithm}-profile.json',row)
                    fixed.append(dict(route=i,algorithm=algorithm,utility=row['utility'],certified=row['certified'],
                                      predictions=row['predictions'],seconds=row['seconds'],milp_calls=row['milp_calls']))
            save(out/(name+'-fixed-summary.json'),fixed)
            for algorithm in ('dfs','joint_milp','de_milp'):
                row=run_one(c,states,algorithm,args.seconds,args.outer,11,profile=True)
                row.update(case=name)
                save(out/f'{name}-{algorithm}-profile.json',row)
            if name=='pool5':
                save(out/'routing-gain-scan.json',route_scan(c,states,references.get(name)))
    after={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'gsdeploy').rglob('*.py')}
    assert protected==after,'OPTIMIZER_SOURCE_CHANGED'
    save(out/'summary.json',dict(records=all_rows,domains=details,total_wall_seconds=time.perf_counter()-start,
        source_unchanged=True,python=platform.python_version(),source_config=str(source),
        git_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        budget=dict(seconds=args.seconds,outer=args.outer,repeats=args.repeats),
        scope='existing validated replay predictor; no physical all-pool optimality claim'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--outer',type=int,default=16)
    parser.add_argument('--repeats',type=int,default=2)
    main(parser.parse_args())
