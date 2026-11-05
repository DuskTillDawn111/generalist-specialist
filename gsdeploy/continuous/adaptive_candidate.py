"""Opt-in candidate scheduling; original Joint, FIFO and SciPy DE unchanged.

DE chunks are explicit restarts from the best route, not continued populations.
All stages share an exact-input prediction cache and one absolute budget.
"""

import copy
from contextlib import contextmanager
import math
import time
from unittest.mock import patch

from scipy.stats import qmc

from .candidate_bounds import ContinuousDeploymentBounds
from .contracts import Budget, BudgetExhausted
from .de_optimizer import DERoutingOptimizer
from .immutable import ImmutableContinuousProblem, SessionPredictorOracle
from .multi_candidate import JointDeploymentCandidates, select_candidates
from .refinement import FixedDeploymentRoutingEvaluator, project_shares


@contextmanager
def budget_slice(budget, routes, predictions, seconds):
    deadline, limits = budget.deadline, budget.limits.copy()
    budget.deadline = min(deadline, time.perf_counter()+max(0.,seconds))
    budget.limits['outer'] = min(limits['outer'],budget.counts['outer']+routes)
    budget.limits['predictions'] = min(limits['predictions'],budget.counts['predictions']+predictions)
    try:
        yield
    finally:
        budget.deadline, budget.limits = deadline, limits


def screen_points(evaluator, shares, seed):
    """Warm start, single-task midpoints, endpoints, then unrounded Sobol.

    Single-task probes preserve the other grid rows, addressing the previously
    observed support-face starvation without replacing continuous DE.
    """
    domain=evaluator.domain
    yield domain.validate(shares)
    for task, models in domain.edges.items():
        for index, source in enumerate(models):
            for target in models[index+1:]:
                row=copy.deepcopy(shares)
                row[task]={m:0. for m in models}
                row[task][source]=row[task][target]=.5
                yield domain.validate(row)
    yield from domain.endpoints()
    if domain.dimension:
        sampler=qmc.Sobol(domain.dimension,scramble=True,seed=seed)
        while True:
            # Powers of two avoid non-balanced Sobol sample-size warnings.
            for vector in sampler.random_base2(3) if sampler.num_generated==0 else sampler.random(8):
                yield domain.decode(vector)


class AdaptiveCandidateRefinementOptimizer:
    def __init__(self, config, replica_options=None, *, k=5, safe_filter=True,
                 policy='adaptive',candidate_strategy='utility',seed=11,
                 seconds=40.,max_predictions=400,max_routes=512,
                 generation_fraction=.5,min_screen_routes=8,screening_fraction=.25,
                 screen_routes=16,chunk_routes=48,max_extra_routes=112,
                 uniform_candidate_routes=128,stagnation_rounds=2,exploration_period=4):
        if (type(k) is not int or k<1 or type(seed) is not int or type(safe_filter) is not bool
                or policy not in {'legacy_uniform','uniform','priority','adaptive'}
                or candidate_strategy not in {'utility','diversity'}
                or not 0<generation_fraction<=1 or not 0<screening_fraction<=1):
            raise ValueError('INVALID_ADAPTIVE_CONFIGURATION')
        for value in (min_screen_routes,screen_routes,chunk_routes,max_extra_routes,
                      uniform_candidate_routes,stagnation_rounds,exploration_period):
            if type(value) is not int or value<1:
                raise ValueError('INVALID_ADAPTIVE_COUNT')
        if min_screen_routes>screen_routes:
            raise ValueError('SCREEN_MINIMUM_EXCEEDS_TARGET')
        self.config,self.replica_options=config,replica_options
        self.k,self.safe_filter,self.policy=k,safe_filter,policy
        self.candidate_strategy,self.seed=candidate_strategy,seed
        self.seconds,self.max_predictions,self.max_routes=seconds,max_predictions,max_routes
        self.generation_fraction,self.screening_fraction=generation_fraction,screening_fraction
        self.min_screen_routes,self.screen_routes=min_screen_routes,screen_routes
        self.chunk_routes,self.max_extra_routes=chunk_routes,max_extra_routes
        self.uniform_candidate_routes=uniform_candidate_routes
        self.stagnation_rounds,self.exploration_period=stagnation_rounds,exploration_period

    def solve(self):
        budget=Budget(self.seconds,self.max_predictions,self.max_routes)
        p=ImmutableContinuousProblem(self.config,replica_options=self.replica_options)
        decisions=[];segments=[];entries=[];termination='NO_CANDIDATES'
        with SessionPredictorOracle(p) as oracle:
            setup=time.perf_counter()-budget.start
            ex=JointDeploymentCandidates(p,oracle,budget)
            started=time.perf_counter();absolute=budget.deadline
            budget.deadline=min(absolute,started+max(0.,absolute-started)*self.generation_fraction)
            try:
                candidates=ex.generate(self.k*(2 if self.candidate_strategy=='diversity' else 1))
            finally:
                budget.deadline=absolute
            generation_seconds=time.perf_counter()-started
            generation_counts=budget.counts.copy()
            selected=select_candidates(candidates,self.k,self.candidate_strategy)
            bounder=ContinuousDeploymentBounds(p,oracle) if self.safe_filter else None
            def incumbent():
                return ex.best['utility'] if ex.best else None
            def decision(entry, action, reason, **extra):
                decisions.append(dict(index=entry['candidate']['generation_index'],action=action,reason=reason,
                    seconds=time.perf_counter()-budget.start,predictions=budget.counts['predictions'],
                    routes=budget.counts['outer'],incumbent=incumbent(),**extra))
            def prune(entry):
                if bounder is None:return False
                proof=bounder.evaluate(entry['candidate']['deployment'],incumbent())
                entry['bound']=proof
                if proof['safe_pruned']:
                    entry['state']='SAFE_PRUNED'
                    decision(entry,'safe_prune',proof['reason'],upper_bound=proof['upper_bound'])
                    return True
                return False
            for c in selected:
                entry=dict(candidate=c,evaluator=None,state='ACTIVE',best=copy.deepcopy(dict(
                    deployment=c['deployment'],routing=c['grid_routing'],utility=c['grid_utility'],
                    prediction=c['grid_prediction'])),counts=dict.fromkeys(budget.counts,0),seconds=0.,
                    extra_routes=0,stagnant=0,rounds=0,rescued=False,screened=False,screen_counts=None,
                    screen_best=None,bound=None)
                entries.append(entry)
                if not prune(entry):
                    e=FixedDeploymentRoutingEvaluator(p,oracle,c['deployment'],budget)
                    e.prediction_cache=ex.cache
                    entry['evaluator']=e
                    entry['dimension']=e.domain.dimension
                    if not e.domain.dimension:
                        entry['state']='SINGLE_ROUTE_EXHAUSTED'
                        decision(entry,'exact_stop','ZERO_DIMENSION_SINGLE_ROUTE')
                else:
                    entry['dimension']=sum(max(0,sum(t in p.config.catalog['models'][m]['supported_tasks']
                        for m in c['deployment'])-1) for t in p.config.workload)
            original_predict=oracle.predict
            def observed(problem,deployment,routing,trace,budget,deadline=None):
                prediction=original_predict(problem,deployment,routing,trace,budget,deadline)
                ex.observe(deployment,routing,prediction)
                return prediction
            def run(entry,stage,allowance,seconds):
                e=entry['evaluator'];before=budget.counts.copy();t=time.perf_counter()
                old_utility=entry['best']['utility'];history_before=len(e.history)
                ex.stage=f'{stage}:{entry["candidate"]["generation_index"]}'
                status='SEARCH_COMPLETED'
                with budget_slice(budget,allowance,allowance,seconds):
                    try:
                        shares=project_shares(p,entry['candidate']['deployment'],entry['best']['routing'])
                        if stage=='screen':
                            seen=set()
                            from .contracts import digest
                            for point in screen_points(e,shares,self.seed):
                                key=digest(point)
                                if key in seen:continue
                                seen.add(key);e.evaluate(point)
                        else:
                            restart_seed=self.seed+1009*entry['rounds']
                            status=DERoutingOptimizer(e,seed=restart_seed,popsize=3,maxiter=100,
                                initial_shares=shares).solve()['status']
                    except BudgetExhausted as exc:
                        status=str(exc)
                delta={k:budget.counts[k]-before[k] for k in before}
                elapsed=time.perf_counter()-t
                for h in e.history[history_before:]:
                    if h['feasible'] and h['best_feasible_utility']>entry['best']['utility']:
                        entry['best']=dict(deployment=entry['candidate']['deployment'],routing=h['routing_qps'],
                            utility=h['best_feasible_utility'],prediction=h['prediction']['traces'][0]['prediction'])
                for key,value in delta.items():entry['counts'][key]+=value
                entry['seconds']+=elapsed
                gain=entry['best']['utility']-old_utility
                if stage=='screen':
                    entry.update(screened=True,screen_counts=delta.copy(),screen_best=entry['best']['utility'])
                else:
                    entry['extra_routes']+=delta['outer'];entry['rounds']+=1
                    entry['stagnant']=0 if gain>1e-12 else entry['stagnant']+1
                segments.append(dict(index=entry['candidate']['generation_index'],stage=stage,
                    round=entry['rounds'],seconds=elapsed,counts=delta,gain=gain,status=status,
                    history_start=history_before,history_end=len(e.history),
                    search_state='fresh DE restart from best route; exact prediction cache retained'
                    if stage!='screen' else 'deterministic probes plus Sobol'))
                decision(entry,'evaluated',stage,gain=gain,counts=delta)
                if ex.best is None or entry['best']['utility']>ex.best['utility']:
                    ex.best=copy.deepcopy(entry['best'])
                return delta
            def global_check():
                for kind in ('predictions','outer'):budget.check(kind)
            with patch.object(oracle,'predict',observed):
                try:
                    global_check()
                    if self.policy=='legacy_uniform':
                        active=[e for e in entries if e['state']=='ACTIVE']
                        for i,entry in enumerate(active):
                            global_check()
                            if prune(entry):continue
                            n=len(active)-i
                            routes=min(self.uniform_candidate_routes,max(1,(budget.limits['outer']-budget.counts['outer'])//n))
                            predictions=max(1,(budget.limits['predictions']-budget.counts['predictions'])//n)
                            run(entry,'uniform_de',min(routes,predictions),(budget.deadline-time.perf_counter())/n)
                            entry['state']='UNIFORM_ALLOCATION_FINISHED'
                        termination='UNIFORM_FINISHED'
                    else:
                        active=[e for e in entries if e['state']=='ACTIVE']
                        n=len(active)
                        screen_deadline=time.perf_counter()+max(0.,budget.deadline-time.perf_counter())*self.screening_fraction
                        if n:
                            available=min(budget.limits['outer']-budget.counts['outer'],
                                          budget.limits['predictions']-budget.counts['predictions'])
                            target=min(self.screen_routes,max(self.min_screen_routes,int(available*self.screening_fraction)//n))
                            # If even all minima do not fit, use the available
                            # count honestly; no global budget is ever renewed.
                            target=min(target,max(1,available//n))
                            for i,entry in enumerate(active):
                                global_check()
                                if prune(entry):continue
                                delta=run(entry,'screen',target,max(0.,screen_deadline-time.perf_counter())/(n-i))
                                if delta['outer']<self.min_screen_routes:
                                    decision(entry,'partial_screen','GLOBAL_OR_STAGE_BUDGET_BEFORE_MINIMUM')
                        iteration=0
                        while True:
                            global_check()
                            for entry in entries:
                                if entry['state'] in {'ACTIVE','PAUSED_STAGNATION'}:prune(entry)
                            active=[e for e in entries if e['state']=='ACTIVE' and e['extra_routes']<self.max_extra_routes]
                            if not active:
                                termination='HEURISTIC_STOP_NO_ACTIVE_CANDIDATES'
                                break
                            if self.policy=='uniform':
                                entry=min(active,key=lambda e:(e['rounds'],e['candidate']['generation_index']))
                                reason='ROUND_ROBIN_SAME_PROBES_AND_RESTARTS'
                            elif self.policy=='priority':
                                entry=max(active,key=lambda e:(e['screen_best'] or e['best']['utility'],
                                    -e['candidate']['generation_index']))
                                reason='STATIC_SCREEN_PRIORITY'
                            else:
                                exploration=(iteration+1)%self.exploration_period==0
                                pool=active+[e for e in entries if exploration and e['state']=='PAUSED_STAGNATION'
                                    and not e['rescued'] and e['extra_routes']<self.max_extra_routes]
                                if exploration:
                                    entry=min(pool,key=lambda e:(e['counts']['outer']/max(1,e['dimension']),e['candidate']['generation_index']))
                                    reason='UNDEREXPLORED_CANDIDATE'
                                else:
                                    def score(e):
                                        gain=e['best']['utility']-e['candidate']['grid_utility']
                                        return e['best']['utility']+2*gain+.02/(1+e['counts']['outer']/max(1,e['dimension']))
                                    entry=max(pool,key=lambda e:(score(e),-e['candidate']['generation_index']))
                                    reason='CURRENT_UTILITY_PLUS_OBSERVED_GAIN_AND_EXPLORATION'
                                if entry['state']=='PAUSED_STAGNATION':
                                    entry.update(state='ACTIVE',rescued=True)
                                    decision(entry,'resume','ONE_EXPLORATION_RESTART')
                            decision(entry,'allocate',reason,heuristic_not_upper_bound=True)
                            allowance=min(self.chunk_routes,self.max_extra_routes-entry['extra_routes'])
                            allowance=min(allowance,budget.limits['outer']-budget.counts['outer'],
                                          budget.limits['predictions']-budget.counts['predictions'])
                            delta=run(entry,'de_restart',allowance,min(3.,max(0.,budget.deadline-time.perf_counter())))
                            if self.policy=='adaptive' and entry['stagnant']>=self.stagnation_rounds:
                                entry['state']='PAUSED_STAGNATION'
                                decision(entry,'pause','HEURISTIC_STAGNATION_NOT_SAFE_PRUNING')
                            if entry['extra_routes']>=self.max_extra_routes:
                                entry['state']='EXTRA_ALLOCATION_CAP'
                                decision(entry,'budget_stop','PER_CANDIDATE_EXTRA_ROUTE_CAP')
                            if not delta['outer']:
                                entry['state']='PAUSED_STAGNATION'
                                decision(entry,'pause','NO_PROGRESS_WITHIN_CHUNK_TIME')
                            iteration+=1
                except BudgetExhausted as exc:
                    termination=str(exc)
            stages=[]
            for entry in entries:
                c=entry['candidate'];e=entry['evaluator']
                stages.append(dict(generation_index=c['generation_index'],deployment=c['deployment'],
                    grid_utility=c['grid_utility'],continuous_utility=entry['best']['utility'],
                    improvement=entry['best']['utility']-c['grid_utility'],winner=entry['best'],
                    dimension=entry['dimension'],state=entry['state'],bound=entry['bound'],
                    history=e.history if e else [],counts=entry['counts'],seconds=entry['seconds'],
                    screened=entry['screened'],screen_counts=entry['screen_counts'],screen_best=entry['screen_best'],
                    extra_routes=entry['extra_routes'],restart_rounds=entry['rounds'],
                    status=entry['state'],safe_pruned=entry['state']=='SAFE_PRUNED'))
            return dict(algorithm='AdaptiveCandidateRefinementOptimizer',policy=self.policy,safe_filter=self.safe_filter,
                k=self.k,seed=self.seed,candidate_strategy=self.candidate_strategy,
                candidates=candidates,selected_indices=[c['generation_index'] for c in selected],
                stages=stages,segments=segments,decisions=decisions,events=ex.events,joint_solves=ex.solves,
                winner=copy.deepcopy(ex.best),initial_winner=copy.deepcopy(candidates[0]) if candidates else None,
                setup_seconds=setup,generation_seconds=generation_seconds,generation_counts=generation_counts,
                safe_bounds_available=bounder.available if bounder else None,
                screening_minimum_routes=self.min_screen_routes,screening_fraction=self.screening_fraction,
                chunk_routes=self.chunk_routes,max_extra_routes=self.max_extra_routes,
                budget=budget.report(),total_seconds=time.perf_counter()-budget.start,status=termination,
                predictor_engine_stats=oracle.engine_stats(),
                continuous_optimality=False,global_top_k_certified=False,neighborhood_enabled=False,
                problem_sha256=p.identity,objective_id=p.objective_id)
