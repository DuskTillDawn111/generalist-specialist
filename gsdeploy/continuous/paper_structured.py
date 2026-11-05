"""Joint-seeded structured deployment/routing search for stage FIFO models.

J3O motivates alternating discrete candidates and continuous routing proposals;
USHER resource demand orders candidates. All acceptance uses the original full
FIFO utility. No paper approximation ratio or continuous optimum is inherited.
Exactly one Joint seed is allowed; deployment exploration never re-solves MILP.
"""

import copy
import math
import time

from .contracts import Budget, BudgetExhausted, digest
from .immutable import ImmutableContinuousProblem, SessionPredictorOracle
from .multi_candidate import JointDeploymentCandidates, deployment_distance
from .refinement import FixedDeploymentRoutingEvaluator, project_shares
from .candidate_bounds import ContinuousDeploymentBounds
from .de_optimizer import DERoutingOptimizer
from ..paper_forecast import MODE, canonical


def covering_deployment_count(problem):
    """Count legal task covers by mask DP without materializing deployments."""
    tasks=list(problem.config.workload);full=(1<<len(tasks))-1
    counts={0:1}
    for model,states in problem.states.items():
        mask=sum(1<<i for i,t in enumerate(tasks) if t in problem.config.catalog['models'][model]['supported_tasks'])
        new={}
        for old,n in counts.items():
            for replicas in states:
                key=old|mask if replicas else old
                new[key]=new.get(key,0)+n
        counts=new
    return counts.get(full,0)


class CapacityAwareTransferSearch:
    """Simplex pair transfers clipped to the original shared-model rho room.

    This is a bounded coordinate heuristic, not MADS. Exact zero endpoints and
    the supplied non-grid LP route are retained; no old-grid quantization.
    """

    def __init__(self,evaluator,entry,*,steps=(.5,.25,.1,.03,.01,.003)):
        self.e,self.entry,self.steps=evaluator,entry,tuple(steps)
        self.best=None;self.seen=set();self.unknown=0

    def probe(self,shares):
        e=self.e;shares=e.domain.validate(shares);key=digest(shares)
        if key in self.seen:return
        self.seen.add(key)
        row=e.evaluate(shares)
        if row['constraint_state']=='UNKNOWN':self.unknown+=1
        if row['feasible'] and (self.best is None or row['best_feasible_utility']>self.best['best_feasible_utility']):
            self.best=row

    def transfer_limit(self,center,task,source,target,step):
        e=self.e;c=e.problem.config;service=self.entry['capacity_service_ms']
        value=service.get(target,{}).get(task)
        maximum=min(step,center[task][source])
        if value is not None:
            load=math.fsum(c.workload[t]*center[t].get(target,0.)*service.get(target,{}).get(t,0.)/1000
                           for t in c.workload)
            capacity=.8*c.catalog['capacity_factors'][target][str(e.deployment[target])]
            maximum=min(maximum,max(0.,capacity-load)*1000/(value*c.workload[task]))
        return maximum

    def solve(self,initial,lp=None):
        e=self.e;status='CAPACITY_TRANSFER_FINISHED'
        started=time.perf_counter();prediction_before=e.budget.times['prediction']
        endpoints_seen=0
        try:
            self.probe(initial)
            if lp is not None:
                self.probe(lp)
            # Pure routing supports precede interior sweeps. Budget truncation
            # is still explicit when the Cartesian endpoint set is too large.
            for point in e.domain.endpoints():
                self.probe(point);endpoints_seen+=1
            if lp is not None:
                for alpha in (.25,.5,.75):
                    self.probe({t:{m:(1-alpha)*initial[t].get(m,0)+alpha*lp[t].get(m,0)
                                   for m in e.domain.edges[t]} for t in e.domain.edges})
            self.probe({t:{m:1/len(ms) for m in ms} for t,ms in e.domain.edges.items()})
            for step in self.steps:
                for sweep in range(4):
                    previous=self.best['best_feasible_utility'] if self.best else -math.inf
                    center=copy.deepcopy(self.best['routing_shares'] if self.best else initial)
                    for task,models in e.domain.edges.items():
                        for source in models:
                            for target in models:
                                if source==target:continue
                                amount=self.transfer_limit(center,task,source,target,step)
                                if amount<=0:continue
                                row=copy.deepcopy(center)
                                row[task][source]-=amount;row[task][target]+=amount
                                self.probe(row)
                    if self.best is None or self.best['best_feasible_utility']<=previous:break
        except BudgetExhausted as exc:
            status=str(exc)
        e.budget.times['outer']+=max(0.,time.perf_counter()-started-
                                    (e.budget.times['prediction']-prediction_before))
        return dict(algorithm='CapacityAwareTransferSearch',status=status,winner=self.best,
                    unknown_routes=self.unknown,continuous_optimality=False,
                    endpoints_seen=endpoints_seen,endpoint_total=math.prod(len(ms) for ms in e.domain.edges.values()))


class PaperStructuredOptimizer:
    def __init__(self,config,replica_options=None,*,seconds=10.,max_predictions=96,max_routes=96,
                 max_candidates=None,seed=11,method='coordinate',joint_seed=True,seed_fraction=.35):
        if config.catalog.get('performance_mode')!=MODE or method not in {'coordinate','de'}:
            raise ValueError('Explicit stage predictor and coordinate/de routing method required')
        if type(joint_seed) is not bool or not 0<seed_fraction<1 or type(seed) is not int:
            raise ValueError('Invalid seed configuration')
        if max_candidates is not None and (type(max_candidates) is not int or max_candidates<1):
            raise ValueError('Positive candidate limit required')
        self.config,self.replica_options=config,replica_options
        self.seconds,self.max_predictions,self.max_routes=seconds,max_predictions,max_routes
        self.max_candidates,self.seed,self.method=max_candidates,seed,method
        self.joint_seed,self.seed_fraction=joint_seed,seed_fraction

    def solve(self):
        budget=Budget(self.seconds,self.max_predictions,self.max_routes)
        p=ImmutableContinuousProblem(self.config,replica_options=self.replica_options)
        decisions=[];stages=[];seed_result=None;status='BOUNDED_SEARCH_FINISHED'
        with SessionPredictorOracle(p) as oracle:
            extractor=JointDeploymentCandidates(p,oracle,budget)
            if self.joint_seed:
                absolute=budget.deadline
                budget.deadline=min(absolute,time.perf_counter()+max(0.,absolute-time.perf_counter())*self.seed_fraction)
                try:seed_result=extractor.solve_one(K=p.config.route_grid,excluded=[])
                finally:budget.deadline=absolute
            seed_best=copy.deepcopy(extractor.best)
            seed_counts=budget.counts.copy()
            bounder=ContinuousDeploymentBounds(p,oracle)
            paper=oracle.session.module('paper_forecast').session(oracle._config.catalog)
            ordering=oracle.session.module('paper_algorithms').UsherResourcePlanner.order_profiled_deployments(oracle._config)
            ranks={canonical(r['deployment']):i for i,r in enumerate(ordering)}
            # Domain accounting is independent of how many bounds fit in the
            # budget. Declared but unprocessed states remain unvisited, never
            # silently relabelled unsupported or infeasible.
            supported=[]
            for entry,engine in paper.entries.values():
                x=entry['deployment']
                if any(m not in p.states or n not in p.states[m] for m,n in x.items()):continue
                if any(not any(t in p.config.catalog['models'][m]['supported_tasks'] for m in x) for t in p.config.workload):continue
                supported.append(entry)
            pool=[];selected=[]
            bound_started=time.perf_counter()
            bound_seconds=0.
            try:
                for entry in supported:
                    budget.check()
                    x=entry['deployment']
                    proof=bounder.evaluate(x,deadline=budget.deadline)
                    pool.append(dict(deployment=x,entry=entry,bound=proof))
                bound_seconds=time.perf_counter()-bound_started
                center=seed_best['deployment'] if seed_best else {}
                pool.sort(key=lambda row:(0 if row['deployment']==center else 1,
                    -(row['bound']['upper_bound'] if row['bound']['upper_bound'] is not None else math.inf),
                    deployment_distance(center,row['deployment']),ranks.get(canonical(row['deployment']),len(ranks))))
                selected=pool if self.max_candidates is None else pool[:self.max_candidates]
                for index,row in enumerate(selected):
                    budget.check();x=row['deployment'];proof=row['bound']
                    incumbent=extractor.best['utility'] if extractor.best else None
                    if proof['safe_pruned'] or incumbent is not None and proof['upper_bound'] is not None and proof['upper_bound']<=incumbent:
                        decisions.append(dict(deployment=x,action='SAFE_PRUNE',bound=proof));continue
                    budget.check('outer');budget.check('predictions')
                    left=len(selected)-index
                    allowance=max(1,(budget.limits['outer']-budget.counts['outer'])//left)
                    old_deadline,old_limits=budget.deadline,budget.limits.copy()
                    budget.deadline=min(old_deadline,time.perf_counter()+max(0.,old_deadline-time.perf_counter())/left)
                    budget.limits['outer']=min(old_limits['outer'],budget.counts['outer']+allowance)
                    budget.limits['predictions']=min(old_limits['predictions'],budget.counts['predictions']+
                        max(1,(old_limits['predictions']-budget.counts['predictions'])//left))
                    e=FixedDeploymentRoutingEvaluator(p,oracle,x,budget)
                    e.prediction_cache=extractor.cache
                    reference=extractor.best['routing'] if extractor.best else {t:{} for t in p.config.workload}
                    initial=project_shares(p,x,reference)
                    before=budget.counts.copy();started=time.perf_counter();lp=None;lp_seconds=0.
                    try:
                        remaining=max(.001,min(.1,budget.deadline-time.perf_counter()))
                        lp=oracle.session.module('paper_algorithms').fixed_deployment_lp(oracle._config,x,seconds=remaining)
                        lp_seconds=time.perf_counter()-started
                        lp_shares=project_shares(p,x,lp['routing']) if lp else None
                        if self.method=='coordinate':
                            raw=CapacityAwareTransferSearch(e,row['entry']).solve(initial,lp_shares)
                        else:
                            # Probe the LP proposal too; DE still starts at the
                            # projected Joint incumbent and uses unrounded U.
                            if lp_shares is not None:e.evaluate(lp_shares)
                            raw=DERoutingOptimizer(e,seed=self.seed,popsize=3,maxiter=100,
                                initial_shares=initial).solve()
                    except BudgetExhausted as exc:
                        raw=dict(status=str(exc))
                    finally:
                        budget.deadline,budget.limits=old_deadline,old_limits
                    for history in e.history:
                        if history['feasible']:
                            record=history['prediction']['traces'][0]['prediction']
                            extractor.stage='structured_continuous'
                            extractor.observe(x,history['routing_qps'],record,cached=history['cached'])
                            extractor.events[-1]['seconds']=history['elapsed_seconds']
                    stages.append(dict(deployment=x,lp_proposal=lp,status=raw['status'],history=e.history,
                        routing_search={k:v for k,v in raw.items() if k not in {'winner','budget','convergence'}},
                        seconds=time.perf_counter()-started,lp_seconds=lp_seconds,
                        counts={k:budget.counts[k]-before[k] for k in before}))
                    decisions.append(dict(deployment=x,action='CONTINUOUS_REFINEMENT',bound=proof))
            except BudgetExhausted as exc:
                status=str(exc)
                if not bound_seconds:bound_seconds=time.perf_counter()-bound_started
            winner=copy.deepcopy(extractor.best)
            domain=covering_deployment_count(p)
            visited={canonical(d['deployment']) for d in decisions}
            return dict(algorithm='PaperStructuredOptimizer',method=self.method,winner=winner,
                initial_winner=seed_best,initial_joint_result=seed_result,
                seed_counts=seed_counts,stages=stages,decisions=decisions,events=extractor.events,resource_ordering=ordering,
                budget=budget.report(),status=status,seed=self.seed,
                bound_seconds=bound_seconds,
                deployment_domain=dict(legal_task_covers=domain,profiled_task_covers=len(supported),
                    unsupported_task_covers=max(0,domain-len(supported)),bounds_evaluated=len(pool),
                    bounds_unprocessed=len(supported)-len(pool),selected_candidates=len(selected),
                    candidate_limit_truncated=max(0,len(pool)-len(selected)),visited=len(visited),
                    unvisited_profiled=max(0,len(supported)-len(visited)),candidate_limit=self.max_candidates),
                fifo_stats=dict(engine_calls=paper.engine_calls,fifo_cache_hits=paper.fifo_hits,
                    fifo_cache_bytes=paper.fifo_cache_bytes,score_cache_hits=paper.hits),
                deployment_milp_calls_after_seed=budget.counts['milp']-seed_counts['milp'],
                continuous_optimality=False,global_optimality=False,native_eligible=False,
                independent_accuracy_qualified=False,final_gain_eligible=False,
                problem_sha256=p.identity,oracle_version=oracle.version,
                scope='one Joint seed; declared profiled deployments; full FIFO acceptance; no hierarchy approximation guarantee')
