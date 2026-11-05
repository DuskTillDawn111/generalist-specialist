"""Actual SciPy optimizers under a common hard prediction budget.

Five-variable capability-repaired encoding is specific to the declared
two-task, at-most-two-replica benchmark. Library operators are not reimplemented.
No objective value, reference winner or gradient is used by encoding/repair.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import copy
import inspect
import itertools
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import scipy
from scipy.optimize import differential_evolution, dual_annealing

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from experiments.compare_crossdomain_strategies_20261105 import (
    ARCHIVE, LoggedOracle, DirectSearch, legal, initialize, audit, run_one as run_ours)
from experiments.bilevel_scalability_benchmark import build_problem, read, receipt, save
from experiments.verify_bilevel_scheduler import ScheduledControlledOptimizer
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, BudgetExhausted, digest

PROTOCOL=ROOT/'configs/classic_optimizer_comparison_20261105.json'


class CompactCodec:
    def __init__(self, problem, domain, seed):
        assert len(domain.edges)==2 and problem.config.gpu_memory_mib==2.4
        assert all(tuple(ns)==(0,1,2) for ns in problem.states.values())
        self.problem,self.domain=problem,domain
        self.tasks=list(domain.edges)
        self.names=list(problem.states)
        np.random.default_rng(seed).shuffle(self.names)
        self.n=len(self.names)
        self.all_task=[m for m in self.names if all(m in domain.edges[t] for t in self.tasks)]
        assert self.all_task

    @staticmethod
    def bin(value,count):return min(int(np.clip(value,0.,1.)*count),count-1)

    def decode(self, vector):
        z=np.clip(vector,0.,1.)
        a,b=[self.names[self.bin(z[i],self.n)] for i in (0,1)]
        mode=self.bin(z[2],3)
        if mode<2:
            if a not in self.all_task:a=self.all_task[0]
            deployment={a:mode+1}
        else:
            # Circular successor handles duplicate slots with no score query.
            if b==a:b=self.names[(self.names.index(a)+1)%self.n]
            deployment={a:1,b:1}
            if not legal(self.problem,deployment):
                b=next(m for m in self.names if m!=a and legal(self.problem,{a:1,m:1}))
                deployment={a:1,b:1}
        assert legal(self.problem,deployment)
        shares={}
        for i,t in enumerate(self.tasks):
            models=[m for m in deployment if m in self.domain.edges[t]]
            shares[t]=({models[0]:1.} if len(models)==1 else
                       {models[0]:float(z[3+i]),models[1]:float(1-z[3+i])})
        return deployment,self.domain.validate(shares)

    def encode(self,deployment,shares):
        assert legal(self.problem,deployment)
        ms=list(deployment);a=ms[0];b=ms[1] if len(ms)>1 else a
        mode=2 if len(ms)>1 else deployment[a]-1
        z=np.array([(self.names.index(a)+.5)/self.n,(self.names.index(b)+.5)/self.n,(mode+.5)/3,.5,.5])
        for i,t in enumerate(self.tasks):
            if len(ms)>1 and all(m in self.domain.edges[t] for m in ms):z[3+i]=shares[t].get(a,0.)
        return z


class LibraryObjective:
    def __init__(self,search,codec,method,cap):
        self.search,self.codec,self.method,self.cap=search,codec,method,cap
        self.attempts=0

    def __call__(self,z):
        self.search.e.budget.check('predictions')
        if self.attempts>=self.cap:raise BudgetExhausted('OBJECTIVE_ATTEMPT_LIMIT')
        self.attempts+=1
        row=self.search.evaluate(*self.codec.decode(z),self.method)
        # Synthetic feasible losses are nonnegative and well below this penalty.
        return -row['U'] if row['U'] is not None else 1e6


def optimize(method, objective, codec, opt, cfg, seed):
    rng=np.random.default_rng(seed)
    common=opt.inner.history[:4]
    if method=='SciPy-DE':
        settings=cfg['de'];pop=rng.random((settings['population'],5))
        for i,r in enumerate(common):pop[i]=codec.encode(r['deployment'],r['routing_shares'])
        result=differential_evolution(objective,[(0.,1.)]*5,
            strategy=settings['strategy'],init=pop,mutation=tuple(settings['mutation']),
            recombination=settings['recombination'],updating=settings['updating'],polish=settings['polish'],
            maxiter=settings['maxiter'],tol=settings['tol'],atol=settings['atol'],rng=rng,workers=1)
    elif method=='SciPy-DA':
        best=objective.search.incumbent
        result=dual_annealing(objective,[(0.,1.)]*5,rng=rng,
            x0=codec.encode(best['deployment'],best['shares']),maxfun=cfg['objective_attempt_cap'],**cfg['da'])
    elif method=='Random':
        while True:objective(rng.random(5))
    else:raise ValueError('UNKNOWN_METHOD')
    return dict(message=str(result.message),success=bool(result.success),nfev=int(result.nfev),nit=int(result.nit))


def run_one(n,method,seed,cfg,base):
    if method=='BO-MADS':return run_ours(n,method,seed,cfg,base)
    protocol=dict(base,budget=cfg['budget']);problem=build_problem(n,protocol)
    b=cfg['budget'];oracle=LoggedOracle()
    e=ContinuousRoutingEvaluator(problem,oracle,budget=Budget(b['seconds'],b['predictions'],b['routes'],b['milp']))
    opt=ScheduledControlledOptimizer(e,'BO-MADS',seed,protocol);initialize(opt)
    search=DirectSearch(e,opt,seed,cfg);codec=CompactCodec(problem,e.domain,seed)
    objective=LibraryObjective(search,codec,method,cfg['objective_attempt_cap'])
    result=None
    try:
        result=optimize(method,objective,codec,opt,cfg,seed);status='LIBRARY_TERMINATED'
    except BudgetExhausted as exc:status=str(exc)
    raw=dict(size=n,method=method,seed=seed,status=status,budget=e.budget.report(),
        calls=oracle.calls,initial_signature=opt.initial_signature,inner_evaluations=opt.inner.history,
        initial_U=opt.initial_signature[-1]['work']['best_feasible_utility'],
        initial_calls=opt.initial_signature[-1]['work']['full_prediction_calls'],
        winner_observed=max((c for c in oracle.calls if c['eligible']),key=lambda c:c['U']),
        problem_sha256=problem.identity,oracle_version=oracle.version,events=search.events,
        objective_attempts=objective.attempts,library_result=result,encoding_model_order=codec.names,
        scipy_version=scipy.__version__,evidence=cfg['evidence'],formal_optimality_claim=False)
    return raw,problem


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();cfg=read(PROTOCOL);base=read(ARCHIVE/'plan.json')['protocol']
    sources=[Path(__file__),PROTOCOL,ROOT/'tests/test_classic_optimizers.py',
        ROOT/'experiments/compare_crossdomain_strategies_20261105.py',ROOT/'experiments/bilevel_scalability_benchmark.py',
        ROOT/'experiments/verify_bilevel_scheduler.py',*sorted((ROOT/'gsdeploy/continuous').glob('*.py')),
        Path(inspect.getsourcefile(inspect.unwrap(differential_evolution))),
        Path(inspect.getsourcefile(inspect.unwrap(dual_annealing))),
        ROOT/'artifacts/classic-optimizer-references-20261105/references.json']
    receipts=[receipt(p) for p in sources]
    archived=[receipt(ARCHIVE/f'reference-{n}.json') for n in cfg['sizes']]
    archived +=[receipt(ROOT/f'artifacts/bilevel-prediction-budget-stop-20261105/{n}-BO-MADS-{s}.json')
                for n in cfg['sizes'] for s in cfg['seeds']]
    schedule=list(itertools.product(cfg['sizes'],cfg['methods'],cfg['seeds']));random.Random(6110513).shuffle(schedule)
    assert len(schedule)==cfg['maximum_searches']
    out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(protocol=cfg,base_protocol=base,sources=receipts,archived=archived,schedule=schedule,
        scipy=scipy.__version__,numpy=np.__version__,python=sys.version,blas_threads=1,frozen_before_search=True))
    rows=[];errors=[];checked=0
    for n,method,seed in schedule:
        raw,problem=run_one(n,method,seed,cfg,base)
        save(out/f'{n}-{method}-{seed}.json',raw)
        ref=read(ARCHIVE/f'reference-{n}.json')
        old=read(ROOT/f'artifacts/bilevel-prediction-budget-stop-20261105/{n}-BO-MADS-{seed}.json')
        try:
            assert raw['initial_signature']==old['initial_signature']
            assert raw['problem_sha256']==old['problem_sha256'] and raw['oracle_version']==old['oracle_version']
            checked+=audit(raw,problem,ref,dict(base,budget=cfg['budget']))
            if method=='BO-MADS':
                keys=('index','phase','deployment','routing_qps','U','eligible')
                assert [{k:r[k] for k in keys} for r in raw['calls']]==[{k:r[k] for k in keys} for r in old['calls']]
        except AssertionError as exc:errors.append(dict(size=n,method=method,seed=seed,error=str(exc) or 'AUDIT_FAILURE'))
        winner=raw['winner_observed'];target=raw['initial_U']+cfg['target_gain']
        reached=next((c for c in raw['calls'] if c['U'] is not None and c['U']>=target),None)
        row=dict(size=n,method=method,seed=seed,U=winner['U'],gain=winner['U']-raw['initial_U'],
            regret_upper=ref['upper_U']-winner['U'],budget=raw['budget'],status=raw['status'],
            target_calls=reached['index'] if reached else None,target_seconds=reached['elapsed_seconds'] if reached else None,
            objective_attempts=raw.get('objective_attempts'),winner_deployment=winner['deployment'],winner_routing=winner['routing_qps'])
        rows.append(row);print(json.dumps(row),flush=True)
    for r in receipts+archived:assert receipt(r['path'])==r
    save(out/'summary.json',dict(rows=rows,audit_errors=errors,checked_inner_results=checked,
        actual_calls=sum(r['budget']['full_prediction_calls'] for r in rows),searches=len(rows),new_gpu_runs=0,predictor_refits=0))
    save(out/'completion.json',dict(status='COMPLETE' if not errors else 'AUDIT_FAILURE',errors=errors,
        remaining_automatic_experiments=[]))
    if errors:raise SystemExit('Audit failures retained; no performance retries')


if __name__=='__main__':main()
