"""Composable cumulative SAVE-hazard budget, never a physical execution gate.

At conditional observation entry, one remaining budget is drawn for the known
surviving restore. Unobserved preentry hazard is not integrated. Visible updates
consume the old frontier's
hazard through the update clock, and modify only future rates. Model extrapolation
remains an explicitly unvalidated approximation; opaque context groups have
already falsified the interpretation as a strict kernel availability gate.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
import math

from .multi_native_composite_v5 import checked
from .native_collect import receipt
from .profiling import digest
from .save_hazard_v13 import EDGES_NS, SaveHazardCandidate, frontier_features

SCHEMA='gsdeploy.cumulative_SAVE_hazard_budget_candidate.v14'
EMPTY_POLICIES=('UNKNOWN','NO_QUERY_PAUSE')


def _frontier(frontier):
    if not isinstance(frontier,(list,tuple)):
        raise ValueError('Explicit visible frontier, including an explicit empty list, required')
    return sorted([[r[0],r[1],r[2],sorted(r[3])] for r in frontier])


def _seal(state):
    state=copy.deepcopy(state)
    state['state_sha256']=digest(state)
    return state


class CumulativeSaveHazardController:
    def __init__(self,model_reference,*,allow_unvalidated_approximation=False,
                 budget_policy='MEDIAN_LOG2',seed=None,empty_frontier_policy='UNKNOWN'):
        if allow_unvalidated_approximation is not True:
            raise ValueError('Explicit unvalidated statistical approximation opt-in required')
        if empty_frontier_policy not in EMPTY_POLICIES:
            raise ValueError('Declared empty-frontier policy required')
        if budget_policy not in ('MEDIAN_LOG2','SEEDED_UNIFORM'):
            raise ValueError('Declared single-restore budget policy required')
        if (budget_policy=='SEEDED_UNIFORM') != (type(seed) is int):
            raise ValueError('Seeded uniform requires an explicit integer seed; median takes no seed')
        self.model_reference=copy.deepcopy(model_reference)
        model=checked(self.model_reference)
        if model.get('training_only') is not True:
            raise ValueError('Frozen training-only source hazard candidate required')
        self._candidate=SaveHazardCandidate(model)
        self._sources=[self.model_reference,receipt(__file__)]
        from . import save_hazard_v13
        self._sources.append(receipt(save_hazard_v13.__file__))
        self._configuration=dict(schema=SCHEMA,sources=self._sources,budget_policy=budget_policy,
            seed=seed,uniform_recipe='first_52_sha256_bits_plus_half_over_2_to_52; budget=-log(U)',
            empty_frontier_policy=empty_frontier_policy,frontier_constant_between_updates=True,
            all_rates_are_statistical_approximation=True,physical_execution_gate=False,
            entry_rule='conditional_on_survival_through_observation_ns; preentry hazard unobserved and not integrated')
        self._identity=digest(self._configuration)
        self._budget_policy=budget_policy;self._seed=seed;self._empty=empty_frontier_policy
        self.assert_sources_unchanged()

    @property
    def identity(self):
        return self._identity

    def provenance(self):
        return dict(identity=self._identity,configuration=copy.deepcopy(self._configuration))

    def assert_sources_unchanged(self):
        # The fitted model is a private in-memory snapshot. Explicit boundary
        # checks let a caller verify files without hashing the model per node.
        for reference in self._sources:
            if receipt(reference['path'])!=reference:
                raise ValueError('Frozen hazard model or solver implementation changed')

    def _check(self,state):
        if (state.get('schema')!=SCHEMA or state.get('controller_identity')!=self.identity
                or state.get('state_sha256')!=digest({k:v for k,v in state.items() if k!='state_sha256'})):
            raise ValueError('Sealed cumulative state bound to this fixed source/solver required')
        return copy.deepcopy({k:v for k,v in state.items() if k!='state_sha256'})

    def _budget(self,restore_key):
        if self._budget_policy=='MEDIAN_LOG2':return math.log(2.)
        encoded=digest(dict(seed=self._seed,restore_key=restore_key,controller_identity=self.identity))
        uniform=(int(encoded[:13],16)+.5)/2**52
        return -math.log(uniform)

    def _rate(self,holder,frontier,age,duration):
        query=self._candidate.query(holder,frontier,age,duration)
        reasons=set(query['OOD_reasons'])
        # Categorical age support remains a pooled hypothesis even when every
        # individual category was seen. This is not probability calibration.
        return query['hazard_per_ms'],reasons

    def _integrate(self,state,end_ns):
        """Integrate the old frontier; stop at the first integer-ns crossing."""
        start=state['last_update_ns'];frontier=state['frontier']
        if end_ns==start:return None
        if not frontier:
            reason='NO_QUERY_PAUSE_ASSUMPTION' if self._empty=='NO_QUERY_PAUSE' else 'EMPTY_FRONTIER_HAZARD_UNKNOWN'
            state['uncertainty_reasons']=sorted(set(state['uncertainty_reasons'])|{reason})
            state['empty_intervals'].append(dict(start_ns=start,end_ns=end_ns,policy=self._empty))
            if self._empty=='UNKNOWN':state['cumulative_hazard_history_known']=False
            state['last_update_ns']=end_ns
            return None
        clock=start
        while clock<end_ns:
            age=clock-state['restore_ns'];index=bisect_right(EDGES_NS,age)-1
            stop=min(end_ns,state['restore_ns']+EDGES_NS[index+1]) if index+1<len(EDGES_NS) else end_ns
            rate,reasons=self._rate(state['held_owner'],frontier,age,stop-clock)
            state['uncertainty_reasons']=sorted(set(state['uncertainty_reasons'])|reasons)
            remaining=state['hazard_budget']-state['consumed_hazard']
            increment=rate*(stop-clock)/1e6
            if state['cumulative_hazard_history_known'] and increment>=remaining:
                delta=max(0,math.ceil(remaining/rate*1e6))
                crossing=min(stop,clock+delta)
                state['consumed_hazard']=state['hazard_budget']
                state['last_update_ns']=crossing
                state['terminal_save_candidate_ns']=crossing
                return crossing
            # Compensated summation reduces dependence on host refresh count.
            y=increment-state['summation_compensation'];total=state['consumed_hazard']+y
            state['summation_compensation']=(total-state['consumed_hazard'])-y
            state['consumed_hazard']=total;clock=stop
        state['last_update_ns']=end_ns
        return None

    def _future(self,state):
        if state['terminal_save_candidate_ns'] is not None:
            return state['terminal_save_candidate_ns'],set()
        if not state['cumulative_hazard_history_known'] or not state['frontier']:
            return None,set()
        remaining=max(0.,state['hazard_budget']-state['consumed_hazard'])
        clock=state['last_update_ns'];reasons=set()
        while True:
            age=clock-state['restore_ns'];index=bisect_right(EDGES_NS,age)-1
            rate,flags=self._rate(state['held_owner'],state['frontier'],age,0)
            reasons.update(flags)
            stop=state['restore_ns']+EDGES_NS[index+1] if index+1<len(EDGES_NS) else None
            increment=rate*(stop-clock)/1e6 if stop is not None else float('inf')
            if increment>=remaining:
                delta=max(0,math.ceil(remaining/rate*1e6));candidate=clock+delta
                _,flags=self._rate(state['held_owner'],state['frontier'],age,delta)
                reasons.update(flags)
                return candidate,reasons
            _,flags=self._rate(state['held_owner'],state['frontier'],age,stop-clock)
            reasons.update(flags);remaining-=increment;clock=stop

    def _result(self,state,old_review=None):
        save_ns,future_reasons=self._future(state)
        now=state['last_update_ns']
        reviews=[t for t in (old_review,save_ns) if t is not None and t>now]
        next_review=now if save_ns is not None and save_ns<=now else min(reviews) if reviews else None
        state['next_review_ns']=next_review
        all_reasons=set(state['uncertainty_reasons'])|future_reasons
        if not state['frontier']:
            all_reasons.add('NO_QUERY_PAUSE_ASSUMPTION' if self._empty=='NO_QUERY_PAUSE' else 'EMPTY_FRONTIER_HAZARD_UNKNOWN')
        status='SAVE_BUDGET_REACHED' if state['terminal_save_candidate_ns'] is not None else 'UNKNOWN_HAZARD_HISTORY' if not state['cumulative_hazard_history_known'] else 'NO_QUERY_PAUSE' if not state['frontier'] and self._empty=='NO_QUERY_PAUSE' else 'EMPTY_FRONTIER_UNKNOWN' if not state['frontier'] else 'CUMULATIVE_SAVE_APPROXIMATION'
        return dict(schema=SCHEMA,status=status,state=_seal(state),absolute_save_candidate_ns=save_ns,
                    next_review_ns=next_review,restore_age_ns=now-state['restore_ns'],
                    origin_restore_ns=state['restore_ns'],observation_ns=state['observation_ns'],
                    conditional_entry_age_ns=state['observation_ns']-state['restore_ns'],
                    preentry_hazard_integrated=False,conditional_on_survival_to_observation=True,
                    hazard_budget=state['hazard_budget'],consumed_hazard=state['consumed_hazard'],
                    remaining_hazard=max(0.,state['hazard_budget']-state['consumed_hazard']),
                    OOD_or_assumption=bool(all_reasons),uncertainty_reasons=sorted(all_reasons),
                    source_query_support_flags_clear=not all_reasons,
                    candidate_empirical_support_claim=False,
                    candidate_probability_calibrated=False,complete_finite_distribution=False,
                    approximation=True,physical_execution_gate=False,controller_identity=self.identity,
                    source_model=copy.deepcopy(self.model_reference),budget_redrawn=False)

    def start_restore(self,held_owner,restore_ns,frontier,restore_key,observation_ns=None):
        if type(restore_ns) is not int or not isinstance(restore_key,str) or not restore_key:
            raise ValueError('Integer absolute restore clock and explicit unique restore key required')
        if observation_ns is None:observation_ns=restore_ns
        if type(observation_ns) is not int or observation_ns<restore_ns:
            raise ValueError('Integer observation clock at or after original restore required')
        frontier=_frontier(frontier)
        if frontier:frontier_features(held_owner,frontier)
        elif not isinstance(held_owner,str) or not held_owner:raise ValueError('Visible held owner required')
        state=dict(schema=SCHEMA,controller_identity=self.identity,restore_key=restore_key,
                   held_owner=held_owner,restore_ns=restore_ns,last_update_ns=observation_ns,
                   origin_restore_ns=restore_ns,observation_ns=observation_ns,
                   preentry_interval_ns=[restore_ns,observation_ns],
                   preentry_hazard=None,preentry_hazard_integrated=False,
                   conditional_on_survival_to_observation=True,
                   frontier=frontier,hazard_budget=self._budget(restore_key),consumed_hazard=0.,
                   summation_compensation=0.,cumulative_hazard_history_known=True,
                   terminal_save_candidate_ns=None,next_review_ns=None,uncertainty_reasons=[],empty_intervals=[])
        return self._result(state)

    def update(self,sealed_state,now_ns,frontier):
        state=self._check(sealed_state)
        if type(now_ns) is not int or now_ns<state['last_update_ns']:
            raise ValueError('Monotone integer absolute update clock required')
        frontier=_frontier(frontier)
        if frontier:frontier_features(state['held_owner'],frontier)
        old_review=state['next_review_ns']
        if state['terminal_save_candidate_ns'] is None:self._integrate(state,now_ns)
        if state['terminal_save_candidate_ns'] is None:state['frontier']=frontier
        return self._result(state,old_review)
