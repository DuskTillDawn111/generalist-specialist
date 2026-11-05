"""Joint MILP seed followed by fixed-residency continuous routing experiments.

The original joint/DFS sources are never edited. Fixed-deployment evaluation
uses no deployment solver. PairwiseTransferSearch is a simple coordinate
transfer baseline, not MADS. Neighbourhood changes are separately opt-in.
"""

import copy
from contextlib import ExitStack
import math
import time
from unittest.mock import patch

from .contracts import Budget, BudgetExhausted, digest
from .evaluator import ContinuousRoutingEvaluator
from .fixed_routing_enumeration import result
from .de_optimizer import DERoutingOptimizer
from .immutable import ImmutableContinuousProblem, SessionPredictorOracle
from .predictor_adapter import normalize_prediction


class FixedDeploymentRoutingEvaluator(ContinuousRoutingEvaluator):
    """Evaluate one unchanged deployment; zero flow never unloads a model."""

    def __init__(self, problem, oracle, deployment, budget=None, cache=True):
        super().__init__(problem, oracle, budget=budget, cache=cache)
        if not deployment or any(m not in problem.states or type(n) is not int
                or n <= 0 or n not in problem.states[m] for m,n in deployment.items()):
            raise ValueError("INVALID_FIXED_DEPLOYMENT")
        self.deployment = copy.deepcopy(deployment)
        self._deployment_identity = digest(self.deployment)
        self.inner = "fixed_deployment"
        self.domain.edges = {t:tuple(m for m in ms if m in deployment)
                             for t,ms in self.domain.edges.items()}
        if any(not ms for ms in self.domain.edges.values()):
            raise ValueError("UNCOVERED_FIXED_DEPLOYMENT")
        self.domain.dimension = sum(len(ms)-1 for ms in self.domain.edges.values())

    def check(self):
        super().check()
        if digest(self.deployment) != self._deployment_identity:
            raise ValueError("FIXED_DEPLOYMENT_CHANGED")

    def evaluate(self, routing_vector, evaluation_budget=None, *, face=None):
        self.check()
        shares = (self.domain.validate(routing_vector) if isinstance(routing_vector,dict)
                  else self.domain.decode(routing_vector,face))
        routing = self.domain.to_qps(shares)
        self.budget.consume('outer')
        started = time.perf_counter()
        before = self.budget.counts.copy()
        if evaluation_budget is not None and (not math.isfinite(evaluation_budget) or evaluation_budget <= 0):
            raise ValueError("INVALID_EVALUATION_BUDGET")
        deadline = min(self.budget.deadline, started+evaluation_budget) if evaluation_budget else self.budget.deadline
        candidate = self.candidate(self.deployment,routing,deadline)
        ledger=[dict(deployment=copy.deepcopy(self.deployment),state=candidate['constraint_state'],utility=candidate['utility'])]
        row=result(self,candidate if candidate['utility'] is not None else None,ledger,True,None,None,1)
        row.update(routing_shares=shares,routing_qps=routing,activation=[m for m in self.deployment if any(r.get(m,0)>0 for r in routing.values())],
            problem_sha256=self.problem.identity,oracle_version=self.oracle.version,objective_id=self.problem.objective_id,
            full_prediction_calls=self.budget.counts['predictions']-before['predictions'],
            predictor_leaf_calls=self.budget.counts['predictor_leaves']-before['predictor_leaves'],milp_solve_calls=0,
            evaluation_seconds=time.perf_counter()-started,cached=self.budget.counts['cache_hits']>before['cache_hits'],
            deployment_fixed=True,certificate_scope="single frozen deployment and route only",
            continuous_optimality=False,elapsed_seconds=time.perf_counter()-self.budget.start)
        self.history.append(copy.deepcopy(row))
        return row


class PairwiseTransferSearch:
    """Deterministic simplex mass transfers from a supplied legal route."""

    def __init__(self, evaluator, steps=(.25,.1,.03,.01,.003,.001,.0001,.000001)):
        if any(not math.isfinite(s) or s<=0 or s>1 for s in steps):
            raise ValueError("INVALID_TRANSFER_STEP")
        self.evaluator,self.steps=evaluator,tuple(steps)

    def solve(self, shares):
        e=self.evaluator
        best=None;status="LOCAL_STEP_SCAN_COMPLETED"
        try:
            row=e.evaluate(shares)
            if row['feasible']: best=row
            if not row['certified']: raise BudgetExhausted("INCOMPLETE_PREDICTOR_EVIDENCE")
            for step in self.steps:
                for sweep in range(32):
                    previous=best['best_feasible_utility'] if best else -math.inf
                    center=best['routing_shares'] if best else row['routing_shares']
                    for task,models in e.domain.edges.items():
                        for source in models:
                            for target in models:
                                if source==target or center[task][source]<=0: continue
                                candidate=copy.deepcopy(center)
                                amount=min(step,candidate[task][source])
                                candidate[task][source]-=amount
                                candidate[task][target]+=amount
                                row=e.evaluate(candidate)
                                if row['feasible'] and (best is None or row['best_feasible_utility']>best['best_feasible_utility']):
                                    best=row
                                if not row['certified']: raise BudgetExhausted("INCOMPLETE_PREDICTOR_EVIDENCE")
                    if best is None or best['best_feasible_utility']<=previous:
                        break
        except BudgetExhausted as exc:
            status=str(exc)
        return dict(algorithm="PairwiseTransferSearch",winner=best,status=status,steps=self.steps,
                    continuous_optimality=False,budget=e.budget.report())


def deployment_neighbors(problem, deployment, routing, limit=8):
    """A bounded, explicit one-change neighbourhood; all rows re-predicted."""
    seen={digest(deployment)}; candidates=[]
    def add(x,kind):
        if not x or any(not any(t in problem.config.catalog['models'][m]['supported_tasks'] for m in x)
                        for t in problem.config.workload): return
        key=digest(x)
        if key not in seen:
            seen.add(key);candidates.append(dict(deployment=x,kind=kind))
    for m,n in sorted(deployment.items()):
        if all(row.get(m,0)==0 for row in routing.values()):
            add({k:v for k,v in deployment.items() if k!=m},'remove_zero_flow')
        for other in sorted(problem.states[m],key=lambda v:abs(v-n)):
            if other and other!=n: add(dict(deployment,**{m:other}),'replica_change')
    specialists=[m for m in problem.states if m not in deployment and
                 problem.config.catalog['models'][m]['role']=='specialist']
    for m in specialists:
        counts=[n for n in problem.states[m] if n]
        if counts: add(dict(deployment,**{m:counts[0]}),'add_specialist')
    for old in sorted(deployment):
        if problem.config.catalog['models'][old]['role']=='specialist':
            for m in specialists:
                counts=[n for n in problem.states[m] if n]
                if counts: add({**{k:v for k,v in deployment.items() if k!=old},m:counts[0]},'replace_specialist')
    return candidates[:limit]


def project_shares(problem, deployment, routing):
    """Keep surviving flow; uncovered rows go to a deterministic legal model."""
    shares={}
    for t,q in problem.config.workload.items():
        eligible=[m for m in sorted(deployment) if t in problem.config.catalog['models'][m]['supported_tasks']]
        row={m:routing[t].get(m,0) for m in eligible}
        total=math.fsum(row.values())
        shares[t]=({m:v/total for m,v in row.items()} if total else {eligible[0]:1.0})
    return shares


class JointMILPContinuousRefinement:
    """Keep the best valid initial/fixed-route/optional-neighbour candidate.

    A fresh private session owns every input and digest cache. All stages share
    one absolute budget. The joint seed gets a configurable fraction of it;
    its complete raw prediction is reused only for the exact same identity.
    """

    def __init__(self, config, replica_options=None, *, seconds=8., seed=11,
                 max_predictions=10000, max_routes=128, initial_fraction=1.,
                 method='de', neighborhood=False, neighbor_limit=4, local_fraction=.4):
        if method not in {'de','local'} or not 0<initial_fraction<=1 or not 0<local_fraction<1:
            raise ValueError("INVALID_REFINEMENT_CONFIGURATION")
        if type(neighbor_limit) is not int or neighbor_limit<1:
            raise ValueError("INVALID_NEIGHBOR_LIMIT")
        self.config,self.replica_options=config,replica_options
        self.seconds,self.seed=seconds,seed
        self.max_predictions,self.max_routes=max_predictions,max_routes
        self.initial_fraction,self.method=initial_fraction,method
        self.neighborhood,self.neighbor_limit=neighborhood,neighbor_limit
        self.local_fraction=local_fraction

    def solve(self):
        budget=Budget(self.seconds,self.max_predictions,self.max_routes)
        p=ImmutableContinuousProblem(self.config,replica_options=self.replica_options)
        histories=[]; convergence=[]; neighbors=[]; initial=None; best=None
        with SessionPredictorOracle(p) as oracle:
            setup=time.perf_counter()-budget.start
            module=oracle.session.module('milp')
            limit=min(budget.deadline-time.perf_counter(),self.seconds*self.initial_fraction)
            opt=oracle.session.optimizer('milp',milp_time_limit_seconds=max(.001,limit))
            # result() also needs a start timestamp if setup already exhausted
            # the shared budget before the first call to original solve().
            opt.start=time.monotonic()
            opt.no_split=False
            t0=time.perf_counter()
            original_predict,original_master=module.predict_scenarios,module.solve_master
            def predict(*args,**kwargs):
                budget.consume('predictions');budget.counts['predictor_leaves']+=3
                t=time.perf_counter()
                try:
                    value=original_predict(*args,**kwargs)
                finally:
                    budget.times['prediction']+=time.perf_counter()-t
                state,u=normalize_prediction(value,p.config.search_policy)
                convergence.append(dict(stage='joint',seconds=time.perf_counter()-budget.start,
                    utility=u,constraint_state=state,deployment=copy.deepcopy(args[1]),routing=copy.deepcopy(args[2])))
                return value
            def master(*args,**kwargs):
                budget.consume('milp')
                kwargs['options']['time_limit']=min(kwargs['options']['time_limit'],max(.001,budget.deadline-time.perf_counter()))
                t=time.perf_counter()
                try: return original_master(*args,**kwargs)
                finally: budget.times['milp']+=time.perf_counter()-t
            with ExitStack() as stack:
                stack.enter_context(patch.object(module,'predict_scenarios',predict))
                stack.enter_context(patch.object(module,'solve_master',master))
                try:
                    budget.check()
                    initial=opt.solve(K=p.config.route_grid,replica_options=self.replica_options)
                except BudgetExhausted as exc:
                    initial=opt.result(p.config.route_grid,str(exc),False)
            joint_seconds=time.perf_counter()-t0
            w=initial.get('winner')
            if w:
                state,u=normalize_prediction(w['prediction'],p.config.search_policy)
                if state=='FEASIBLE':
                    best=dict(deployment=copy.deepcopy(w['deployment']),routing=copy.deepcopy(w['routing']),
                              utility=u,prediction=copy.deepcopy(w['prediction']),source='joint')
            initial_counts=budget.counts.copy()
            initial_best=copy.deepcopy(best)
            stage_status='NO_VALID_JOINT_INCUMBENT'
            shared_cache={}

            def refine(deployment,seed_routing,stage,stage_seconds,route_allowance):
                nonlocal best,stage_status
                e=FixedDeploymentRoutingEvaluator(p,oracle,deployment,budget)
                e.prediction_cache=shared_cache
                if stage=='fixed' and initial_best is not None:
                    trace=p.train_traces[0]
                    key=digest([e.identity,deployment,initial_best['routing'],trace.identity])
                    shared_cache[key]=copy.deepcopy(initial_best['prediction'])
                start_route_count=budget.counts['outer']
                start_counts=budget.counts.copy()
                old_deadline,old_outer_limit=budget.deadline,budget.limits['outer']
                budget.deadline=min(old_deadline,time.perf_counter()+max(.001,stage_seconds))
                budget.limits['outer']=min(old_outer_limit,start_route_count+route_allowance)
                t=time.perf_counter()
                try:
                    shares=project_shares(p,deployment,seed_routing)
                    if self.method=='local':
                        raw=PairwiseTransferSearch(e).solve(shares)
                    else:
                        raw=DERoutingOptimizer(e,seed=self.seed,popsize=3,maxiter=100,
                                               face_limit=128,initial_shares=shares).solve()
                    stage_status=raw['status']
                    for row in e.history:
                        if row['feasible']:
                            u=row['best_feasible_utility']
                            if best is None or u>best['utility']:
                                best=dict(deployment=copy.deepcopy(deployment),routing=row['routing_qps'],utility=u,
                                    prediction=row['prediction']['traces'][0]['prediction'],source=stage)
                        convergence.append(dict(stage=stage,seconds=row['elapsed_seconds'],
                            utility=row['best_feasible_utility'],constraint_state=row['constraint_state'],
                            deployment=copy.deepcopy(deployment),routing=row['routing_qps'],
                            assignment_sha256=(row['prediction']['traces'][0]['assignment_sha256'] if row['feasible'] else None)))
                    histories.append(dict(stage=stage,deployment=copy.deepcopy(deployment),seconds=time.perf_counter()-t,
                        predictions=budget.counts['predictions']-start_counts['predictions'],
                        milp_calls=budget.counts['milp']-start_counts['milp'],
                        routes=budget.counts['outer']-start_route_count,status=raw['status'],
                        coverage=raw.get('activation_coverage'),history=e.history))
                    return raw
                finally:
                    budget.deadline,budget.limits['outer']=old_deadline,old_outer_limit

            fixed_start=time.perf_counter()
            if best is not None:
                remaining=budget.deadline-time.perf_counter()
                available_routes=budget.limits['outer']-budget.counts['outer']
                fraction=self.local_fraction if self.neighborhood else 1.
                refine(best['deployment'],best['routing'],'fixed',remaining*fraction,
                       max(1,int(available_routes*fraction)))
            fixed_seconds=time.perf_counter()-fixed_start
            fixed_best=copy.deepcopy(best)
            if self.neighborhood and best is not None:
                neighbors=deployment_neighbors(p,fixed_best['deployment'],fixed_best['routing'],self.neighbor_limit)
                for i,neighbor in enumerate(neighbors):
                    if opt.surrogate is not None and math.fsum(
                            opt.state_memory(m,n) for m,n in neighbor['deployment'].items()) > p.config.gpu_memory_mib+1e-6:
                        neighbor['status']='PRUNED_PROVEN_MEMORY_BOUND'
                        continue
                    remaining=budget.deadline-time.perf_counter()
                    routes=budget.limits['outer']-budget.counts['outer']
                    if remaining<=0 or routes<=0: break
                    neighbor['status']='EVALUATED'
                    refine(neighbor['deployment'],fixed_best['routing'],f'neighbor{i}',
                           remaining/(len(neighbors)-i),max(1,routes//(len(neighbors)-i)))
            report=budget.report()
            engine_stats=None
            if p.config.catalog.get('performance_mode')=='paper_profiled_fifo_v1':
                engine=oracle.session.module('paper_forecast').session(oracle._config.catalog)
                engine_stats=engine.stats()
            return dict(algorithm='JointMILPContinuousRefinement',method=self.method,
                neighborhood_enabled=self.neighborhood,neighbors=neighbors,initial=initial,
                initial_winner=initial_best,fixed_winner=fixed_best,winner=best,
                utility_improvement=(best['utility']-initial_best['utility'] if best and initial_best else None),
                deployment_preserved=best['deployment']==initial_best['deployment'] if best and initial_best else None,
                setup_seconds=setup,joint_seconds=joint_seconds,fixed_seconds=fixed_seconds,
                initial_counts=initial_counts,budget=report,histories=histories,convergence=convergence,
                status=stage_status,continuous_optimality=False,optimality_certified=False,
                initial_certificate_scope=initial.get('certificate_scope'),seed=self.seed,
                prediction_mode=p.config.catalog.get('performance_mode','independent'),
                fixed_route_dimension=(sum(sum(t in p.config.catalog['models'][m]['supported_tasks']
                    for m in initial_best['deployment'])-1 for t in p.config.workload)
                    if initial_best else None),
                predictor_engine_stats=engine_stats,
                native_qualified=False,
                session_stats=oracle.session.stats(),problem_sha256=p.identity,
                total_seconds=time.perf_counter()-budget.start)
