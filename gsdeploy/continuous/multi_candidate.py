"""Opt-in Joint deployment candidates followed by unchanged continuous DE.

Each restricted Joint solve owns fresh feedback state. Only deployment
exclusions and exact whole-prediction inputs are shared. Neither a master
bound nor a finite-budget continuous result is called a global Top-K proof.
"""

import copy
from contextlib import ExitStack
import math
import time
from unittest.mock import patch

from .contracts import Budget, BudgetExhausted, digest
from .de_optimizer import DERoutingOptimizer
from .immutable import ImmutableContinuousProblem, SessionPredictorOracle
from .predictor_adapter import normalize_prediction
from .refinement import FixedDeploymentRoutingEvaluator, project_shares


def deployment_distance(a, b):
    """Activation Jaccard plus bounded replica L1, independently of routes."""
    models=set(a)|set(b)
    if not models: return 0.
    activation=1-len(set(a)&set(b))/len(models)
    replicas=math.fsum(abs(a.get(m,0)-b.get(m,0))/max(a.get(m,0),b.get(m,0),1) for m in models)/len(models)
    return .5*(activation+replicas)


def select_candidates(candidates, k, strategy='utility', quality_window=.15):
    if strategy not in {'utility','diversity'} or type(k) is not int or k<1 or quality_window<0:
        raise ValueError('INVALID_CANDIDATE_SELECTION')
    if not candidates:return []
    ordered=sorted(candidates,key=lambda c:(-c['grid_utility'],c['generation_index']))
    # The first unrestricted Joint solution is always retained, even when a
    # later uncertified restricted solve happens to improve its utility.
    first=next(c for c in candidates if c['generation_index']==0)
    selected=[first]
    while len(selected)<min(k,len(ordered)):
        remaining=[c for c in ordered if c not in selected]
        if strategy=='utility': choice=remaining[0]
        else:
            quality=[c for c in remaining if c['grid_utility']>=ordered[0]['grid_utility']-quality_window]
            pool=quality or remaining
            choice=max(pool,key=lambda c:(min(deployment_distance(c['deployment'],s['deployment']) for s in selected),
                                          c['grid_utility'],-c['generation_index']))
        selected.append(choice)
    return selected


class JointDeploymentCandidates:
    """Repeated original Joint calls excluding complete deployment vectors."""

    def __init__(self, problem, oracle, budget, cache=None, *, cache_columns=True):
        if oracle.problem is not problem:
            raise ValueError('CANDIDATE_ORACLE_PROBLEM_MISMATCH')
        self.problem,self.oracle,self.budget=problem,oracle,budget
        self.cache={} if cache is None else cache
        self.evaluator_identity=digest(dict(problem=problem.identity,oracle=oracle.version,objective=problem.objective_id))
        self.events=[];self.solves=[];self.best=None;self.stage='generation'
        from .column_cache import JointColumnCache
        self.column_cache = JointColumnCache(problem.identity,oracle.version) if cache_columns else None

    def cache_key(self, deployment, routing):
        if len(self.problem.train_traces)!=1:
            raise ValueError('CANDIDATE_GENERATION_REQUIRES_ONE_TRACE')
        return digest([self.evaluator_identity,deployment,routing,self.problem.train_traces[0].identity])

    def observe(self, deployment, routing, prediction, *, cached=False):
        state,u=normalize_prediction(prediction,self.problem.config.search_policy)
        event=dict(stage=self.stage,seconds=time.perf_counter()-self.budget.start,
            predictions=self.budget.counts['predictions'],milp=self.budget.counts['milp'],
            outer=self.budget.counts['outer'],deployment=copy.deepcopy(deployment),routing=copy.deepcopy(routing),
            state=state,utility=u,cached=cached,prediction=copy.deepcopy(prediction))
        self.events.append(event)
        if state=='FEASIBLE' and (self.best is None or u>self.best['utility']):
            self.best=dict(deployment=copy.deepcopy(deployment),routing=copy.deepcopy(routing),
                utility=u,prediction=copy.deepcopy(prediction),source=self.stage)

    def solve_one(self, *, K, excluded, deployment=None):
        budget=self.budget
        opt=self.oracle.session.optimizer('milp',milp_time_limit_seconds=max(.001,budget.deadline-time.perf_counter()))
        opt.start=time.monotonic();opt.no_split=False
        module=self.oracle.session.module('milp')
        predictor,master=module.predict_scenarios,module.solve_master
        start=time.perf_counter();before=budget.counts.copy()
        column_stats_before = self.column_cache.report() if self.column_cache else None
        original_columns = opt.columns
        column_seconds = 0.
        def columns(choices,K,*,no_split=False):
            nonlocal column_seconds
            budget.check();started=time.perf_counter()
            try:
                value = (self.column_cache.call(original_columns,opt,choices,K,no_split=no_split)
                         if self.column_cache else original_columns(choices,K,no_split=no_split))
            finally:
                column_seconds += time.perf_counter()-started
            budget.check()
            return value
        opt.columns = columns
        def predict(*args,**kwargs):
            budget.check()
            x,r=args[1],args[2];key=self.cache_key(x,r)
            hit=key in self.cache
            if hit:
                budget.counts['cache_hits']+=1;p=copy.deepcopy(self.cache[key])
            else:
                budget.consume('predictions');budget.counts['predictor_leaves']+=3
                t=time.perf_counter()
                try:p=predictor(*args,**kwargs)
                finally:budget.times['prediction']+=time.perf_counter()-t
                self.cache[key]=copy.deepcopy(p)
            self.observe(x,r,p,cached=hit)
            return p
        def solve_master(*args,**kwargs):
            budget.consume('milp')
            kwargs['options']['time_limit']=min(kwargs['options']['time_limit'],max(.001,budget.deadline-time.perf_counter()))
            t=time.perf_counter()
            try:return master(*args,**kwargs)
            finally:budget.times['milp']+=time.perf_counter()-t
        with ExitStack() as stack:
            stack.enter_context(patch.object(module,'predict_scenarios',predict))
            stack.enter_context(patch.object(module,'solve_master',solve_master))
            try:
                budget.check()
                result=opt.solve(K=K,replica_options={m:[n for n in ns if n] for m,ns in self.problem.states.items()},
                    exclude_deployments=copy.deepcopy(excluded),deployment=deployment)
            except BudgetExhausted as exc:
                result=opt.result(K,str(exc),False)
        result['wrapper_seconds']=time.perf_counter()-start
        result['column_cache'] = self.column_cache.report() if self.column_cache else None
        result['column_cache_before'] = column_stats_before
        result['column_seconds'] = column_seconds
        result['wrapper_counts']={k:budget.counts[k]-before[k] for k in before}
        result['excluded_deployments']=copy.deepcopy(excluded)
        result['fixed_deployment']=copy.deepcopy(deployment)
        self.solves.append(result)
        return result

    def generate(self, count, K=None):
        if type(count) is not int or count<1:raise ValueError('INVALID_CANDIDATE_COUNT')
        candidates=[];excluded=[];K=K or self.problem.config.route_grid
        for index in range(count):
            result=self.solve_one(K=K,excluded=excluded)
            w=result['winner']
            if w is None:break
            state,u=normalize_prediction(w['prediction'],self.problem.config.search_policy)
            if state!='FEASIBLE':break
            if digest(w['deployment']) in {digest(x) for x in excluded}:
                raise RuntimeError('DEPLOYMENT_EXCLUSION_FAILED')
            candidate=dict(generation_index=index,deployment=copy.deepcopy(w['deployment']),
                grid_routing=copy.deepcopy(w['routing']),grid_utility=u,grid_prediction=copy.deepcopy(w['prediction']),
                generation_status=result['status'],restricted_grid_certified=result['optimality_certified'],
                certificate_scope='this residual finite grid only; not continuous or global Top-K',
                seconds=result['wrapper_seconds'],counts=result['wrapper_counts'])
            candidates.append(candidate);excluded.append(w['deployment'])
            try:self.budget.check()
            except BudgetExhausted:break
        return candidates


class MultiCandidateContinuousRefinement:
    """Independent experiment; default planners and original DE are untouched."""

    def __init__(self, config, replica_options=None, *, k=5, strategy='utility',pool_factor=2,
                 seconds=16.,max_predictions=400,max_routes=384,per_candidate_routes=128,
                 generation_fraction=.5,seed=11,quality_window=.15):
        if (type(k) is not int or k<1 or strategy not in {'utility','diversity'}
                or type(pool_factor) is not int or pool_factor<1 or not 0<generation_fraction<=1
                or type(per_candidate_routes) is not int or per_candidate_routes<1
                or not math.isfinite(quality_window) or quality_window<0):
            raise ValueError('INVALID_MULTI_CANDIDATE_CONFIGURATION')
        self.config,self.replica_options=config,replica_options
        self.k,self.strategy,self.pool_factor=k,strategy,pool_factor
        self.seconds,self.max_predictions,self.max_routes=seconds,max_predictions,max_routes
        self.per_candidate_routes,self.generation_fraction=per_candidate_routes,generation_fraction
        self.seed,self.quality_window=seed,quality_window

    def solve(self):
        budget=Budget(self.seconds,self.max_predictions,self.max_routes)
        p=ImmutableContinuousProblem(self.config,replica_options=self.replica_options)
        stages=[];candidates=[];selected=[]
        with SessionPredictorOracle(p) as oracle:
            setup=time.perf_counter()-budget.start
            extractor=JointDeploymentCandidates(p,oracle,budget)
            original_deadline=budget.deadline
            generation_start=time.perf_counter()
            # Setup is included in the global wall budget, but does not
            # silently consume a fraction measured before setup completed.
            budget.deadline=min(original_deadline,generation_start+
                max(0.,original_deadline-generation_start)*self.generation_fraction)
            candidates=extractor.generate(self.k*(self.pool_factor if self.strategy=='diversity' else 1))
            generation_seconds=time.perf_counter()-generation_start
            budget.deadline=original_deadline
            selected=select_candidates(candidates,self.k,self.strategy,self.quality_window)
            generation_counts=budget.counts.copy()
            original_predict=oracle.predict
            def observe_predict(problem,deployment,routing,trace,budget,deadline=None):
                prediction=original_predict(problem,deployment,routing,trace,budget,deadline)
                extractor.observe(deployment,routing,prediction)
                return prediction
            with patch.object(oracle,'predict',observe_predict):
                for index,candidate in enumerate(selected):
                    try:budget.check()
                    except BudgetExhausted:break
                    remaining=len(selected)-index
                    available_predictions=budget.limits['predictions']-budget.counts['predictions']
                    available_routes=budget.limits['outer']-budget.counts['outer']
                    if available_predictions<=0 or available_routes<=0:break
                    allocation=min(self.per_candidate_routes,max(1,available_routes//remaining))
                    old_deadline=budget.deadline;old_limits=budget.limits.copy()
                    budget.deadline=min(old_deadline,time.perf_counter()+(old_deadline-time.perf_counter())/remaining)
                    budget.limits['outer']=min(old_limits['outer'],budget.counts['outer']+allocation)
                    budget.limits['predictions']=min(old_limits['predictions'],budget.counts['predictions']+max(1,available_predictions//remaining))
                    e=FixedDeploymentRoutingEvaluator(p,oracle,candidate['deployment'],budget)
                    e.prediction_cache=extractor.cache
                    extractor.stage=f'refine:{candidate["generation_index"]}'
                    before=budget.counts.copy();t=time.perf_counter()
                    try:
                        raw=DERoutingOptimizer(e,seed=self.seed,popsize=3,maxiter=100,
                            initial_shares=project_shares(p,candidate['deployment'],candidate['grid_routing'])).solve()
                    finally:
                        budget.deadline=old_deadline;budget.limits=old_limits
                    # Warm-start cache hits still constitute valid incumbent
                    # observations. Preserve the exact original grid route.
                    best=dict(deployment=candidate['deployment'],routing=candidate['grid_routing'],
                        utility=candidate['grid_utility'],prediction=candidate['grid_prediction'])
                    for row in e.history:
                        if row['feasible'] and row['best_feasible_utility']>best['utility']:
                            best=dict(deployment=candidate['deployment'],routing=row['routing_qps'],
                                utility=row['best_feasible_utility'],prediction=row['prediction']['traces'][0]['prediction'])
                    if extractor.best is None or best['utility']>extractor.best['utility']:
                        extractor.best=copy.deepcopy(best)
                    stages.append(dict(generation_index=candidate['generation_index'],deployment=candidate['deployment'],
                        grid_utility=candidate['grid_utility'],continuous_utility=best['utility'],
                        improvement=best['utility']-candidate['grid_utility'],winner=best,history=e.history,
                        seconds=time.perf_counter()-t,counts={k:budget.counts[k]-before[k] for k in before},
                        status=raw['status'],coverage=raw['activation_coverage']))
            winner=copy.deepcopy(extractor.best)
            return dict(algorithm='MultiCandidateContinuousRefinement',k=self.k,strategy=self.strategy,
                pool_factor=self.pool_factor,quality_window=self.quality_window,seed=self.seed,
                setup_seconds=setup,generation_seconds=generation_seconds,generation_counts=generation_counts,
                candidates=candidates,selected_indices=[c['generation_index'] for c in selected],stages=stages,
                winner=winner,initial_winner=copy.deepcopy(candidates[0]) if candidates else None,
                utility_improvement=winner['utility']-candidates[0]['grid_utility'] if winner and candidates else None,
                joint_solves=extractor.solves,events=extractor.events,budget=budget.report(),
                predictor_engine_stats=oracle.engine_stats(),
                status='REFINEMENT_FINISHED' if len(stages)==len(selected) and len(selected)==self.k else 'PARTIAL_SEARCH',
                generated_candidates=len(candidates),selected_candidates=len(selected),
                requested_pool_size=self.k*(self.pool_factor if self.strategy=='diversity' else 1),
                continuous_optimality=False,global_top_k_certified=False,
                problem_sha256=p.identity,objective_id=p.objective_id,total_seconds=time.perf_counter()-budget.start,
                neighborhood_enabled=False)
