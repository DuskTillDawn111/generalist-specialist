"""One conditional SAVE budget with causal lifecycle and full empty risk.

Rates are statistical physical SAVE_END hazards, not GPU service or ownership
guarantees. Old lifecycle/frontier rates are integrated before a visible update.
No update, empty frontier, host wakeup or competing arrival redraws the budget.
"""
from __future__ import annotations

from bisect import bisect_right
import copy
import math

from .learned_availability_forecast_v20 import verify_sources
from .native_collect import receipt
from .owner_lifecycle_mark_v21 import OwnerLifecycle, FIELDS, RESIDENTS
from .profiling import digest

SCHEMA = 'gsdeploy.cumulative_causal_lifecycle_SAVE_budget.v24'


def encode_lifecycle(states):
    if set(states) != set(RESIDENTS) or any(not isinstance(s, OwnerLifecycle) for s in states.values()):
        raise ValueError('Complete typed resident lifecycle required')
    return {owner:{field:getattr(states[owner],field) for field in FIELDS} for owner in RESIDENTS}


def decode_lifecycle(states):
    return {owner:OwnerLifecycle(**values) for owner,values in states.items()}


def frontier(rows):
    return sorted([[a,t,p,sorted(phases)] for a,t,p,phases in rows])


class CumulativeLifecycleHazard:
    def __init__(self, candidate, *, age_edges_ns, sources, candidate_identity,
                 allow_unvalidated_approximation=False, budget_policy='MEDIAN_LOG2', seed=None):
        if allow_unvalidated_approximation is not True:
            raise ValueError('Explicit statistical approximation opt-in required')
        if (not age_edges_ns or age_edges_ns[0] != 0
                or any(type(v) is not int or v < 0 for v in age_edges_ns)
                or list(age_edges_ns) != sorted(set(age_edges_ns))):
            raise ValueError('Strictly increasing fixed integer age bins from zero required')
        if (budget_policy not in ('MEDIAN_LOG2','SEEDED_UNIFORM')
                or (budget_policy == 'SEEDED_UNIFORM') != (type(seed) is int)):
            raise ValueError('One declared median or explicitly seeded budget required')
        if not isinstance(candidate_identity,str) or len(candidate_identity) != 64:
            raise ValueError('Frozen candidate SHA256 identity required')
        if not sources:
            raise ValueError('Complete candidate/model/implementation evidence required')
        self._candidate = copy.deepcopy(candidate)
        self.edges = tuple(age_edges_ns)
        self._sources = tuple([*copy.deepcopy(sources),receipt(__file__)])
        self._policy, self._seed = budget_policy, seed
        self._configuration = dict(schema=SCHEMA, candidate_identity=candidate_identity,
            age_edges_ns=list(self.edges), sources=list(self._sources), budget_policy=budget_policy,
            seed=seed, entry_rule='Conditional survival to observation; no preentry integral',
            empty_frontier_rule='Query the learned full-risk lifecycle hazard',
            target='Physical SAVE_END age since RESTORE_START', physical_execution_gate=False)
        self.identity = digest(self._configuration)
        self.assert_sources_unchanged()

    def assert_sources_unchanged(self):
        verify_sources(self._sources)

    def provenance(self):
        return dict(identity=self.identity,configuration=copy.deepcopy(self._configuration))

    def _budget(self,key):
        if self._policy == 'MEDIAN_LOG2':
            return math.log(2.)
        encoded = digest(dict(seed=self._seed,restore_key=key,controller_identity=self.identity))
        return -math.log((int(encoded[:13],16)+.5)/2**52)

    def _rate(self,state,age,horizon):
        result = self._candidate.query(state['held_owner'],state['frontier'],
            decode_lifecycle(state['lifecycle']),age,horizon)
        rate = result['hazard_per_ms']
        if type(rate) not in (int,float) or not math.isfinite(rate) or rate < 0:
            raise ValueError('Finite nonnegative learned hazard required')
        return rate,set(result['OOD_reasons'])

    def _check(self,sealed):
        state = {k:v for k,v in sealed.items() if k != 'state_sha256'}
        if (state.get('schema') != SCHEMA or state.get('controller_identity') != self.identity
                or sealed.get('state_sha256') != digest(state)):
            raise ValueError('Sealed lifecycle state bound to this candidate required')
        return copy.deepcopy(state)

    def _integrate(self,state,end):
        clock = state['last_update_ns']
        while clock < end:
            age = clock-state['restore_ns']
            index = bisect_right(self.edges,age)-1
            stop = min(end,state['restore_ns']+self.edges[index+1]) if index+1 < len(self.edges) else end
            rate,flags = self._rate(state,age,stop-clock)
            state['uncertainty_reasons'] = sorted(set(state['uncertainty_reasons'])|flags)
            remaining = max(0.,state['hazard_budget']-state['consumed_hazard'])
            increment = rate*(stop-clock)/1e6
            if rate > 0 and increment >= remaining:
                crossing = min(stop,clock+max(0,math.ceil(remaining/rate*1e6)))
                state.update(consumed_hazard=state['hazard_budget'],last_update_ns=crossing,
                             terminal_save_candidate_ns=crossing)
                return
            y = increment-state['summation_compensation']
            total = state['consumed_hazard']+y
            state['summation_compensation'] = (total-state['consumed_hazard'])-y
            state['consumed_hazard'] = total
            clock = stop
        state['last_update_ns'] = end

    def _future(self,state):
        if state['terminal_save_candidate_ns'] is not None:
            return state['terminal_save_candidate_ns'],set()
        remaining = max(0.,state['hazard_budget']-state['consumed_hazard'])
        clock,flags = state['last_update_ns'],set()
        while True:
            age = clock-state['restore_ns']
            index = bisect_right(self.edges,age)-1
            rate,reasons = self._rate(state,age,0)
            flags.update(reasons)
            stop = state['restore_ns']+self.edges[index+1] if index+1 < len(self.edges) else None
            increment = rate*(stop-clock)/1e6 if stop is not None else math.inf if rate > 0 else 0.
            if rate > 0 and increment >= remaining:
                delay = max(0,math.ceil(remaining/rate*1e6))
                _,reasons = self._rate(state,age,delay)
                return clock+delay,flags|reasons
            if stop is None:
                return None,flags
            _,reasons = self._rate(state,age,stop-clock)
            flags.update(reasons)
            remaining -= increment
            clock = stop

    def _result(self,state,old_review=None):
        save,flags = self._future(state)
        now = state['last_update_ns']
        review = now if save is not None and save <= now else min(
            (t for t in (save,old_review) if t is not None and t > now),default=None)
        state['next_review_ns'] = review
        sealed = copy.deepcopy(state)
        sealed['state_sha256'] = digest(sealed)
        reasons = sorted(set(state['uncertainty_reasons'])|flags)
        return dict(schema=SCHEMA,state=sealed,
            status='SAVE_BUDGET_REACHED' if state['terminal_save_candidate_ns'] is not None else 'CUMULATIVE_SAVE_APPROXIMATION',
            absolute_save_candidate_ns=save,next_review_ns=review,
            hazard_budget=state['hazard_budget'],consumed_hazard=state['consumed_hazard'],
            remaining_hazard=max(0.,state['hazard_budget']-state['consumed_hazard']),
            restore_age_ns=now-state['restore_ns'],origin_restore_ns=state['restore_ns'],
            observation_ns=state['observation_ns'],preentry_hazard_integrated=False,
            conditional_on_survival_to_observation=True, budget_redrawn=False,
            uncertainty_reasons=reasons, candidate_probability_calibrated=False,
            empty_frontier_hazard_queried=True,physical_execution_gate=False,
            active_service_replacement_timing_identified=False,controller_identity=self.identity)

    def start_restore(self,held_owner,restore_ns,pending_frontier,restore_key,owner_lifecycle,*,observation_ns=None):
        if (held_owner not in RESIDENTS or type(restore_ns) is not int
                or not isinstance(restore_key,str) or not restore_key):
            raise ValueError('Known resident, absolute RESTORE and unique key required')
        if observation_ns is None:
            observation_ns = restore_ns
        if type(observation_ns) is not int or observation_ns < restore_ns:
            raise ValueError('Observation after known surviving RESTORE required')
        state = dict(schema=SCHEMA,controller_identity=self.identity,held_owner=held_owner,
            restore_ns=restore_ns,observation_ns=observation_ns,last_update_ns=observation_ns,
            restore_key=restore_key,frontier=frontier(pending_frontier),lifecycle=encode_lifecycle(owner_lifecycle),
            hazard_budget=self._budget(restore_key),consumed_hazard=0.,summation_compensation=0.,
            terminal_save_candidate_ns=None,next_review_ns=None,uncertainty_reasons=[],visible_update_count=0)
        return self._result(state)

    def update(self,sealed,now_ns,pending_frontier,owner_lifecycle):
        state = self._check(sealed)
        if type(now_ns) is not int or now_ns < state['last_update_ns']:
            raise ValueError('Monotone absolute update clock required')
        next_lifecycle,next_frontier = encode_lifecycle(owner_lifecycle),frontier(pending_frontier)
        review = state['next_review_ns']
        if state['terminal_save_candidate_ns'] is None:
            self._integrate(state,now_ns)
        if state['terminal_save_candidate_ns'] is None:
            state.update(lifecycle=next_lifecycle,frontier=next_frontier)
        state['visible_update_count'] += 1
        return self._result(state,review)
