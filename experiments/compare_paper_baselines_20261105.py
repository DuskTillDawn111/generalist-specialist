"""Bounded, explicitly scoped CherryPick and Proteus mechanism adaptations.

Search cannot access reference scores, analytic gradients or archived winners.
Proteus is a one-shot profiled planner; unused prediction budget is not wasted.
"""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
os.environ['MKL_NUM_THREADS']='1'
import argparse
import itertools
from pathlib import Path
import random
import sys
import time
import json
import numpy as np
import scipy
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize_scalar, milp, Bounds, LinearConstraint
from scipy.special import ndtr
from scipy.spatial.distance import cdist
from scipy.stats import qmc

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from experiments.compare_classic_optimizers_20261105 import CompactCodec
from experiments.compare_crossdomain_strategies_20261105 import (
    ARCHIVE, LoggedOracle, DirectSearch, initialize, audit, run_one as run_ours)
from experiments.bilevel_scalability_benchmark import build_problem, read, receipt, save
from experiments.verify_bilevel_scheduler import ScheduledControlledOptimizer
from gsdeploy.continuous import ContinuousRoutingEvaluator
from gsdeploy.continuous.contracts import Budget, BudgetExhausted, digest

PROTOCOL=ROOT/'configs/paper_baseline_comparison_20261105.json'


def matern52(a,b,length):
    d=np.sqrt(5.)*cdist(a,b)/length
    return (1+d+d*d/3)*np.exp(-d)


def expected_improvement(mean,std,best):
    std=np.asarray(std);delta=best-np.asarray(mean)
    z=delta/np.maximum(std,1e-30)
    return np.where(std>0,delta*ndtr(z)+std*np.exp(-z*z/2)/np.sqrt(2*np.pi),0.)


def gp_predict(x,y,query,settings):
    offset=float(np.mean(y));scale=max(float(np.std(y)),1e-6)
    yn=(y-offset)/scale
    def factor(log_length):
        k=matern52(x,x,np.exp(log_length))+np.eye(len(x))*settings['noise']
        return cho_factor(k,lower=True,check_finite=False)
    def nll(log_length):
        f=factor(log_length);alpha=cho_solve(f,yn,check_finite=False)
        return .5*yn@alpha+np.log(np.diag(f[0])).sum()
    fit=minimize_scalar(nll,bounds=np.log(settings['length_bounds']),method='bounded',
                        options={'maxiter':30,'xatol':1e-4})
    length=float(np.exp(fit.x));f=factor(fit.x)
    k=matern52(x,query,length);alpha=cho_solve(f,yn,check_finite=False)
    v=solve_triangular(f[0],k,lower=True,check_finite=False)
    return offset+scale*(k.T@alpha),scale*np.sqrt(np.maximum(0.,1.-(v*v).sum(axis=0))),length


def capacity_feasible(problem,deployment,shares,margin=1e-6):
    c=problem.config
    for m,n in deployment.items():
        load=sum(c.workload[t]*r.get(m,0.) for t,r in shares.items())
        cap=1000./next(iter(c.catalog['models'][m]['service_time_ms'].values()))
        cap*=c.catalog['capacity_factors'][m][str(n)]
        if load>cap*(1-margin):return False
    return True


def cherry_pick(search,codec,cfg,seed,log=None):
    settings=cfg['cherrypick'];rng=np.random.default_rng(seed)
    sobol=qmc.Sobol(5,scramble=True,seed=seed)
    if log is None:log=[]
    for iteration in range(settings['maximum_steps']):
        search.e.budget.check('predictions')
        observed=[r for r in search.seen.values() if r['U'] is not None]
        x=np.array([codec.encode(r['deployment'],r['shares']) for r in observed]);y=np.array([-r['U'] for r in observed])
        center=codec.encode(search.incumbent['deployment'],search.incumbent['shares'])
        proposals=np.vstack([sobol.random(settings['sobol_candidates']),
            np.clip(center+rng.normal(0,settings['local_sigma'],(settings['local_candidates'],5)),0,1)])
        candidates={}
        for z in proposals:
            d,r=codec.decode(z);key=digest([d,r])
            if key not in search.seen and capacity_feasible(search.e.problem,d,r):
                candidates[key]=(d,r,codec.encode(d,r))
        if not candidates:return 'NO_UNSEEN_ACQUISITION_CANDIDATES',log
        candidates=list(candidates.values());query=np.array([c[2] for c in candidates])
        mean,std,length=gp_predict(x,y,query,settings)
        # All remaining candidates have analytically known feasibility probability 1.
        ei=expected_improvement(mean,std,float(y.min()));index=int(np.argmax(ei))
        threshold=settings['relative_ei_stop']*abs(float(y.min()))
        stop=len(observed)>=settings['minimum_observations'] and ei[index]<threshold
        log.append(dict(iteration=iteration,observations=len(y),max_ei=float(ei[index]),
            threshold=threshold,length=length,stop=bool(stop),candidate_count=len(candidates)))
        search.e.budget.check('predictions')
        if stop:return 'RELATIVE_EI_STOP',log
        search.evaluate(*candidates[index][:2],'cherrypick_gp_ei')
    return 'ITERATION_GUARD',log


class JointProfileMILP:
    """Disaggregate task flow by model/replica state to linearize assignment.

    One chosen state per model, shared memory, all demand served, no traffic
    to inactive states, mixed-task capacity. No nonlinear utility formula here.
    """
    def __init__(self,problem,domain,settings):
        self.problem,self.domain,self.settings=problem,domain,settings
        c=problem.config
        self.states=[(m,n) for m,ns in problem.states.items() for n in ns if n]
        self.flows=[(t,m,n) for m,n in self.states for t in domain.edges if m in domain.edges[t]]
        self.si={p:i for i,p in enumerate(self.states)}
        self.fi={p:i+len(self.states) for i,p in enumerate(self.flows)}
        dim=len(self.states)+len(self.flows);rows=[];lower=[];upper=[]
        def constraint(terms,lo,hi):
            row=np.zeros(dim)
            for i,value in terms:row[i]=value
            rows.append(row);lower.append(lo);upper.append(hi)
        for m in problem.states:
            constraint([(self.si[m,n],1.) for n in problem.states[m] if n],0,1)
        constraint([(self.si[m,n],c.catalog['memory_table_mib'][m][str(n)]) for m,n in self.states],0,c.gpu_memory_mib)
        for t in domain.edges:
            constraint([(idx,1.) for (tt,m,n),idx in self.fi.items() if tt==t],c.workload[t],c.workload[t])
        for m,n in self.states:
            cap=1000./next(iter(c.catalog['models'][m]['service_time_ms'].values()))
            cap*=c.catalog['capacity_factors'][m][str(n)]*(1-settings['capacity_margin'])
            constraint([(self.si[m,n],-cap)]+[(idx,1.) for (t,mm,nn),idx in self.fi.items() if (mm,nn)==(m,n)],-np.inf,0)
        self.constraints=LinearConstraint(np.array(rows),lower,upper)
        self.bounds=Bounds(np.zeros(dim),np.array([1.]*len(self.states)+[c.workload[t] for t,m,n in self.flows]))
        self.integrality=np.array([1]*len(self.states)+[0]*len(self.flows))

    def features(self,deployment,shares):
        x=np.zeros(len(self.integrality))
        for m,n in deployment.items():x[self.si[m,n]]=1
        for t,row in shares.items():
            for m,value in row.items():
                if value>0:x[self.fi[t,m,deployment[m]]]=self.problem.config.workload[t]*value
        return x

    def coefficients(self,method,observations):
        if method=='Proteus-Q':
            c=np.zeros(len(self.integrality));catalog=self.problem.config.catalog
            for (t,m,n),idx in self.fi.items():c[idx]=catalog['models'][m]['benchmark_quality'][t]
            return c,dict(kind='profiled_quality_loss',fit_samples=0)
        # Linear ridge regression of -U on allocation and flow, with intercept.
        # The minimum-norm regularized fit uses ONLY common paid observations.
        x=np.array([self.features(r['deployment'],r['shares']) for r in observations]);y=np.array([-r['U'] for r in observations])
        mx=x.mean(axis=0);my=y.mean();xc=x-mx;yc=y-my
        coef=xc.T@np.linalg.solve(xc@xc.T+self.settings['ridge']*np.eye(len(x)),yc)
        intercept=float(my-mx@coef)
        return coef,dict(kind='linear_utility_surrogate',fit_samples=len(x),intercept=intercept,
            training_rmse=float(np.sqrt(np.mean((x@coef+intercept-y)**2))),coefficients=coef.tolist(),
            source='common_initialization_only',ridge=self.settings['ridge'])

    def solve(self,coef,budget):
        budget.check('predictions');budget.consume('milp');started=time.perf_counter()
        result=milp(coef,integrality=self.integrality,bounds=self.bounds,constraints=self.constraints,
            options={'time_limit':max(.001,budget.deadline-time.perf_counter()),'mip_rel_gap':self.settings['milp_gap']})
        budget.times['milp']+=time.perf_counter()-started
        return result

    def decode(self,vector):
        d={m:n for (m,n),idx in self.si.items() if vector[idx]>.5};shares={}
        for t in self.domain.edges:
            row={m:max(0.,float(vector[self.fi[t,m,n]]))/self.problem.config.workload[t]
                 for m,n in d.items() if (t,m,n) in self.fi}
            total=sum(row.values());assert abs(total-1)<1e-6
            shares[t]={m:v/total for m,v in row.items()}
        return d,self.domain.validate(shares)


def proteus(search,method,cfg):
    planner=JointProfileMILP(search.e.problem,search.e.domain,cfg['proteus'])
    obs=[r for r in search.seen.values() if r['U'] is not None]
    coef,fit=planner.coefficients(method,obs);result=planner.solve(coef,search.e.budget)
    detail=dict(fit=fit,solver_status=int(result.status),message=result.message,
        solver_optimal=bool(result.success),milp_variables=len(coef),binary_variables=len(planner.states),
        returned_plan=None)
    if result.x is None:return 'MILP_NO_PLAN',detail
    d,r=planner.decode(result.x)
    assert capacity_feasible(search.e.problem,d,r,margin=0.)
    search.e.budget.check('predictions')
    row=search.evaluate(d,r,'proteus_joint_milp')
    detail.update(returned_plan=row,surrogate_objective=float(coef@result.x),
                  solver_gap=float(result.mip_gap),linear_constraint_max_violation=float(max(
                      0,np.max(planner.constraints.lb-planner.constraints.A@result.x),
                      np.max(planner.constraints.A@result.x-planner.constraints.ub))))
    return 'ONE_SHOT_PLAN_COMPLETE',detail


def run_one(n,method,seed,cfg,base):
    if method=='BO-MADS':return run_ours(n,method,seed,cfg,base)
    protocol=dict(base,budget=cfg['budget']);problem=build_problem(n,protocol);b=cfg['budget']
    oracle=LoggedOracle();e=ContinuousRoutingEvaluator(problem,oracle,budget=Budget(b['seconds'],b['predictions'],b['routes'],b['milp']))
    opt=ScheduledControlledOptimizer(e,'BO-MADS',seed,protocol);initialize(opt)
    search=DirectSearch(e,opt,seed,cfg);details=[] if method=='CherryPick-U' else None
    try:
        if method=='CherryPick-U':status,details=cherry_pick(search,CompactCodec(problem,e.domain,seed),cfg,seed,details)
        else:status,details=proteus(search,method,cfg)
    except BudgetExhausted as exc:status=str(exc)
    return dict(size=n,method=method,seed=seed,status=status,budget=e.budget.report(),
        calls=oracle.calls,initial_signature=opt.initial_signature,inner_evaluations=opt.inner.history,
        initial_U=opt.initial_signature[-1]['work']['best_feasible_utility'],
        initial_calls=opt.initial_signature[-1]['work']['full_prediction_calls'],
        winner_observed=max((r for r in oracle.calls if r['eligible']),key=lambda r:r['U']),
        problem_sha256=problem.identity,oracle_version=oracle.version,events=search.events,method_details=details,
        evidence=cfg['evidence'],formal_optimality_claim=False),problem


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    cfg=read(PROTOCOL);base=read(ARCHIVE/'plan.json')['protocol'];out=args.output.resolve()
    paths=[Path(__file__),PROTOCOL,ROOT/'tests/test_paper_baselines.py',
        ROOT/'docs/paper_baseline_mapping_20261105.md',ROOT/'experiments/compare_classic_optimizers_20261105.py',
        ROOT/'experiments/compare_crossdomain_strategies_20261105.py',ROOT/'experiments/bilevel_scalability_benchmark.py',
        ROOT/'experiments/verify_bilevel_scheduler.py',*sorted((ROOT/'gsdeploy/continuous').glob('*.py')),
        *sorted((ROOT/'artifacts/paper-baseline-references-20261105').glob('*'))]
    receipts=[receipt(p) for p in paths]
    archived=[receipt(ARCHIVE/f'reference-{n}.json') for n in cfg['sizes']]
    archived +=[receipt(ROOT/f'artifacts/bilevel-prediction-budget-stop-20261105/{n}-BO-MADS-{s}.json') for n in cfg['sizes'] for s in cfg['seeds']]
    schedule=list(itertools.product(cfg['sizes'],cfg['methods'],cfg['seeds']));random.Random(6110521).shuffle(schedule)
    assert len(schedule)==cfg['maximum_searches'];out.mkdir(parents=True,exist_ok=False)
    save(out/'plan.json',dict(protocol=cfg,base_protocol=base,sources=receipts,archived=archived,
        schedule=schedule,scipy=scipy.__version__,numpy=np.__version__,python=sys.version,blas_threads=1,frozen_before_search=True))
    rows=[];errors=[];checked=0
    for n,method,seed in schedule:
        raw,problem=run_one(n,method,seed,cfg,base);save(out/f'{n}-{method}-{seed}.json',raw)
        ref=read(ARCHIVE/f'reference-{n}.json');old=read(ROOT/f'artifacts/bilevel-prediction-budget-stop-20261105/{n}-BO-MADS-{seed}.json')
        try:
            assert raw['initial_signature']==old['initial_signature']
            assert raw['problem_sha256']==old['problem_sha256'] and raw['oracle_version']==old['oracle_version']
            checked+=audit(raw,problem,ref,dict(base,budget=cfg['budget']))
            if method=='BO-MADS':
                keys=('index','phase','deployment','routing_qps','U','eligible')
                assert [{k:r[k] for k in keys} for r in raw['calls']]==[{k:r[k] for k in keys} for r in old['calls']]
            if method.startswith('Proteus'):
                assert raw['method_details']['solver_optimal'] and raw['method_details']['linear_constraint_max_violation']<1e-7
        except AssertionError as exc:errors.append(dict(size=n,method=method,seed=seed,error=str(exc) or 'AUDIT_FAILURE'))
        w=raw['winner_observed'];target=raw['initial_U']+cfg['target_gain']
        reached=next((r for r in raw['calls'] if r['U'] is not None and r['U']>=target),None)
        row=dict(size=n,method=method,seed=seed,U=w['U'],gain=w['U']-raw['initial_U'],
            regret_upper=ref['upper_U']-w['U'],budget=raw['budget'],status=raw['status'],
            target_calls=reached['index'] if reached else None,target_seconds=reached['elapsed_seconds'] if reached else None,
            winner_deployment=w['deployment'],winner_routing=w['routing_qps'],
            returned_plan_U=(raw.get('method_details') or {}).get('returned_plan',{}).get('U') if method.startswith('Proteus') else w['U'])
        rows.append(row);print(json.dumps(row),flush=True)
    for r in receipts+archived:assert receipt(r['path'])==r
    save(out/'summary.json',dict(rows=rows,audit_errors=errors,checked_inner_results=checked,
        actual_calls=sum(r['budget']['full_prediction_calls'] for r in rows),searches=len(rows),new_gpu_runs=0,predictor_refits=0))
    save(out/'completion.json',dict(status='COMPLETE' if not errors else 'AUDIT_FAILURE',errors=errors,remaining_automatic_experiments=[]))
    if errors:raise SystemExit('Audit failures retained; no performance retries')


if __name__=='__main__':main()
