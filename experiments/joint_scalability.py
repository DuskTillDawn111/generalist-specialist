"""Experiment A: unchanged joint search, real validated replay, explicit scope.

Wrappers only measure API boundaries; original sources and expressions stay
unchanged. Common isolated digest sessions are available to every algorithm.
"""

import argparse
from contextlib import contextmanager, ExitStack
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
import time
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gsdeploy import SystemConfig
from gsdeploy.search_session import SearchSession
from gsdeploy.surrogate_bounds import surrogate_bounds
from gsdeploy.route_domain import fraction_rows
from gsdeploy.trace import make_trace, trace_digest
from gsdeploy.continuous import (Budget, ImmutableContinuousProblem, SessionPredictorOracle,
    ContinuousRoutingEvaluator, DERoutingOptimizer)
from gsdeploy.continuous.contracts import digest
from experiments.diagnose_continuous_phase1 import ROOT, DEFAULT_SOURCE, save, assignment


def source_manifest():
    return {str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT/'gsdeploy').rglob('*.py')}


@contextmanager
def probes(module, opt=None):
    """Low-overhead wall timers, with separately nested HiGHS backend time."""
    import scipy.optimize._milp as scipy_milp
    stats = dict(columns_seconds=0.,prediction_seconds=0.,master_seconds=0.,
                 highs_seconds=0.,prediction_calls=0,rounds=[],evaluations=[],columns_returned=None,
                 per_task_columns={},column_generation_complete=None)
    start = time.perf_counter()
    with ExitStack() as stack:
        original = module.predict_scenarios
        def prediction(*args, **kwargs):
            t = time.perf_counter()
            p = original(*args, **kwargs)
            elapsed = time.perf_counter()-t
            stats['prediction_seconds'] += elapsed
            stats['prediction_calls'] += 1
            stats['evaluations'].append(dict(seconds=time.perf_counter()-start,
                prediction_seconds=elapsed,deployment=args[1],routing=args[2],
                utility=p.get('U') if p.get('constraints_eligible') else None,
                reason=p.get('reason'),trace_sha256=p.get('trace_sha256')))
            return p
        stack.enter_context(patch.object(module,'predict_scenarios',prediction))
        if hasattr(module,'solve_master'):
            master = module.solve_master
            def solve(*args, **kwargs):
                t = time.perf_counter()
                fit = master(*args, **kwargs)
                dt = time.perf_counter()-t
                stats['master_seconds'] += dt
                con = kwargs['constraints']
                stats['rounds'].append(dict(seconds=dt,status=int(fit.status),
                    variables=len(args[0]),constraints=con.A.shape[0],
                    nonzeros=int(con.A.nnz),nodes=int(fit.mip_node_count)
                    if getattr(fit,'mip_node_count',None) is not None else None))
                return fit
            stack.enter_context(patch.object(module,'solve_master',solve))
        backend = scipy_milp._highs_wrapper
        def highs(*args, **kwargs):
            t = time.perf_counter()
            try:
                return backend(*args, **kwargs)
            finally:
                stats['highs_seconds'] += time.perf_counter()-t
        stack.enter_context(patch.object(scipy_milp,'_highs_wrapper',highs))
        if opt is not None and hasattr(opt,'columns'):
            original_columns = opt.columns
            def columns(*args, **kwargs):
                t = time.perf_counter()
                value = original_columns(*args, **kwargs)
                stats['columns_seconds'] += time.perf_counter()-t
                stats['column_generation_complete'] = value is not None
                if value is not None:
                    stats['columns_returned'] = len(value)
                    stats['per_task_columns'] = {task:sum(c['task']==task for c in value) for task in opt.rates}
                return value
            stack.enter_context(patch.object(opt,'columns',columns))
        yield stats


def context(c):
    return dict(horizon=c.horizon_seconds,energy_reference_watts=c.energy_reference_watts,
                startup_amortization_seconds=c.startup_amortization_seconds)


def subset_config(base, models, K, seconds, max_columns=50000, seed=42):
    data = dict(base.catalog)
    for field in ('models','capacity_factors','memory_table_mib'):
        data[field] = {m:data[field][m] for m in models}
    c = replace(base,catalog=data,route_grid=K,random_seed=seed,
        milp_time_limit_seconds=seconds,milp_max_iterations=10000,
        milp_max_columns=max_columns,milp_absolute_gap=1e-8,
        search_max_states=1000000,milp_capacity_witness=False)
    if surrogate_bounds(c.catalog,c.workload,**context(c)) is None:
        raise ValueError('NO_VALIDATED_MATHEMATICAL_BOUND_CONTRACT')
    return c


def domain_audit(c, states):
    """Independent raw-domain accounting, outside timed optimization.

    Memory filtering is a necessary mathematical condition, not a claim that
    all survivors satisfy coupled rho/FIFO/quality constraints.
    """
    t0=time.perf_counter()
    b=surrogate_bounds(c.catalog,c.workload,**context(c))
    names=sorted(states)
    choices={m:[0]+list(states[m]) for m in names}
    mem={(m,n):b.memory_multiplier*c.catalog['safe_factor']*c.catalog['memory_table_mib'][m][str(n)]
         for m in names for n in choices[m]}
    count=0; covered=0
    for ns in itertools.product(*(choices[m] for m in names)):
        if math.fsum(mem[m,n] for m,n in zip(names,ns))<=c.gpu_memory_mib+1e-6:
            count+=1
            x={m for m,n in zip(names,ns) if n}
            covered+=all(any(t in c.catalog['models'][m]['supported_tasks'] for m in x) for t in c.workload)
    tasks={}
    for task in c.workload:
        eligible=[m for m in names if task in c.catalog['models'][m]['supported_tasks']]
        rows=0; raw=0
        for fractions in fraction_rows(eligible,c.route_grid,c.route_pair_denominators_by_task.get(task,())):
            rows+=1
            raw+=math.prod(len(states[m]) for m,f in fractions)
        tasks[task]=dict(eligible_models=eligible,rational_rows=rows,raw_count_columns=raw)
    trace=make_trace(c.workload,c.horizon_seconds,c.random_seed,c.arrival_process)
    bins={}
    for offset,task in trace:
        index=int(offset//10)
        bins[index]=bins.get(index,0)+1
    return dict(raw_deployment_combinations=math.prod(len(v) for v in choices.values()),
        memory_bound_survivors=count,memory_and_task_cover_survivors=covered,
        memory_filter_scope='validated additive residency lower bound; survivors not guaranteed feasible',
        tasks=len(tasks),models=len(states),replica_options=states,per_task=tasks,
        routing_dimension=sum(len(v['eligible_models'])-1 for v in tasks.values()),
        splitting_tasks=sum(len(v['eligible_models'])>1 for v in tasks.values()),
        raw_task_count_columns=sum(v['raw_count_columns'] for v in tasks.values()),
        max_columns=c.milp_max_columns,iteration_limit=c.milp_max_iterations,
        trace_sha256=trace_digest(trace),trace_requests=len(trace),trace_seed=c.random_seed,
        burst_bins_10s=dict(mean=len(trace)/(c.horizon_seconds/10),maximum=max(bins.values(),default=0)),
        audit_seconds=time.perf_counter()-t0)


def scope_audit(base):
    rows=[]
    rates=base.workload
    variants={'original':rates,'low':{t:q*.5 for t,q in rates.items()},
              'high':{t:q*2 for t,q in rates.items()},'all_high':{t:q*4 for t,q in rates.items()},
              'asr_hot':{t:q*(3 if t=='asr' else 1) for t,q in rates.items()},
              'ocr_hot':{t:q*(3 if t=='ocr' else 1) for t,q in rates.items()}}
    for n in (1,2,3):
        variants[f'tasks{n}']={t:rates[t] for t in list(rates)[:n]}
    from gsdeploy.calibration_v2 import validated_replay_spec
    for name, workload in variants.items():
        try:
            validated_replay_spec(base.catalog,workload)
            status='SUPPORTED'
        except ValueError as exc:
            status=str(exc)
        rows.append(dict(case=name,workload=workload,status=status))
    for p in sorted((ROOT/'artifacts/static-spillover-study-20261022').glob('transfer*/*config.json')):
        try:
            c=SystemConfig.load(p)
            status='SUPPORTED' if surrogate_bounds(c.catalog,c.workload,**context(c)) else 'NO_VALIDATED_BOUND'
        except ValueError as exc:
            status=str(exc)
        rows.append(dict(case=str(p.relative_to(ROOT)),status=status))
    return rows


def run(c, states, algorithm, seconds=8., outer=32, seed=11):
    start=time.perf_counter()
    oracle=None;session=None
    try:
        if algorithm in {'joint','dfs'}:
            session=SearchSession(ROOT,c.as_dict())
            setup=time.perf_counter()-start
            opt=session.optimizer('milp' if algorithm=='joint' else 'dfs',
                milp_time_limit_seconds=max(.001,seconds-setup))
            module=session.module('milp' if algorithm=='joint' else 'optimizer')
            with probes(module,opt) as timing:
                raw=opt.solve(K=c.route_grid,replica_options=states)
            winner=raw['winner']
            w=(dict(deployment=winner['deployment'],routing=winner['routing'],utility=winner['score'],prediction=winner['prediction'])
               if winner and winner['prediction'].get('U') is not None else None)
            metrics=raw['metrics']
            certified=raw['optimality_certified']
            initial_time=(setup+raw['first_feasible_seconds']) if raw.get('first_feasible_seconds') is not None else None
            best_time=(setup+raw['best_found_seconds']) if raw.get('best_found_seconds') is not None else None
            raw.pop('problem',None)
        else:
            budget=Budget(seconds,10000,outer)
            p=ImmutableContinuousProblem(c,replica_options=states)
            oracle=SessionPredictorOracle(p)
            e=ContinuousRoutingEvaluator(p,oracle,'milp_bounds',budget)
            setup=time.perf_counter()-start
            with probes(oracle.session.module('prediction')) as timing:
                # This module's predict_scenarios is also the entry point,
                # so the wrapper records each complete candidate bundle.
                raw=DERoutingOptimizer(e,seed=seed,popsize=3,maxiter=100).solve()
            raw['evaluations']=e.history
            winner=raw['winner']
            w=(dict(deployment=winner['deployment'],routing=winner['routing_qps'],utility=winner['best_feasible_utility'],prediction=winner['prediction'])
               if winner else None)
            metrics=budget.report();certified=False
            initial_time=next((setup+v['seconds'] for v in timing['evaluations'] if v['utility'] is not None),None)
            best_time=None
        elapsed=time.perf_counter()-start
        stats=oracle.session.stats() if oracle else session.stats()
        return dict(algorithm=algorithm,K=c.route_grid,seconds=elapsed,setup_seconds=setup,
            timing=timing,metrics=metrics,status=raw['status'],certified=certified,
            utility=w['utility'] if w else None,winner=w,raw=raw,session_stats=stats,
            first_feasible_seconds=initial_time,best_found_seconds=best_time,
            fifo=assignment(c,w['deployment'],w['routing']) if w else None,
            input_sha256=digest(dict(config=c.as_dict(),states=states)),
            time_limit_seconds=seconds,outer_limit=outer,optimizer_seed=seed)
    finally:
        if oracle: oracle.close()
        if session: session.close()


def plan(base):
    pools={2:['generalist_2','image'],3:['generalist_2','image','ocr'],
           5:['generalist_2','text','image','ocr','asr'],7:sorted(base.catalog['models'])}
    cases=[]
    for pool,replicas,Ks in [(2,1,[1,2,8,16]),(3,1,[1,2,8,16]),
                            (5,1,[1,2,4,8,12,16]),(7,1,[1,2,4,8,12,16]),
                            (5,2,[2,4,8,16]),(7,2,[2,4,8]),(7,3,[2,4]),(7,99,[2])]:
        models=pools[pool]
        states={m:[int(n) for n in base.catalog['capacity_factors'][m]
                   if 0<int(n)<=replicas] for m in models}
        for K in Ks:
            cases.append(dict(name=f'p{pool}-n{replicas}-K{K}',models=models,states=states,K=K,trace_seed=42))
    # Natural Poisson trace variability is supported; no invented burst data.
    for seed in (43,45):
        cases.append(dict(name=f'p5-n2-K4-trace{seed}',models=pools[5],
                          states={m:[1,2] for m in pools[5]},K=4,trace_seed=seed))
    return cases


def main(args):
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    base=SystemConfig.load(ROOT/args.source)
    before=source_manifest();save(out/'source-manifest.json',before)
    save(out/'scope-gates.json',scope_audit(base))
    cases=plan(base)
    if args.cases:
        cases=[c for c in cases if c['name'] in args.cases]
    save(out/'plan.json',cases)
    results=[];start=time.perf_counter()
    for case in cases:
        name=case['name'];states=case['states']
        c=subset_config(base,case['models'],case['K'],args.seconds,seed=case['trace_seed'])
        save(out/f'{name}-config.json',c.as_dict())
        save(out/f'{name}-domain.json',domain_audit(c,states))
        algorithms=['joint']
        if (case['K'] in (2,8,16) and name.startswith(('p5-n1','p7-n1','p5-n2'))) or 'trace' in name:
            algorithms+=['dfs','de_milp']
        for repeat in range(args.repeats):
            order=list(algorithms);random.Random(111+repeat).shuffle(order)
            for algorithm in order:
                r=run(c,states,algorithm,args.seconds,args.outer,[11,29][repeat%2])
                r.update(case=name,repeat=repeat)
                save(out/f'{name}-{algorithm}-r{repeat}.json',r)
                results.append({k:r[k] for k in ('case','algorithm','repeat','K','seconds','setup_seconds','utility','certified','status','metrics','timing')})
                print(name,algorithm,repeat,round(r['seconds'],3),r['utility'],r['status'],flush=True)
    assert before==source_manifest()
    save(out/'summary.json',dict(runs=len(results),seconds=time.perf_counter()-start,
        source_unchanged=True,source=str(ROOT/args.source),rows=results))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    parser.add_argument('--output',required=True)
    parser.add_argument('--seconds',type=float,default=8.)
    parser.add_argument('--outer',type=int,default=32)
    parser.add_argument('--repeats',type=int,default=2)
    parser.add_argument('--cases',nargs='+')
    main(parser.parse_args())
