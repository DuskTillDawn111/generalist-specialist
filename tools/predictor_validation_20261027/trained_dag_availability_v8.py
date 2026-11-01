"""Executable stateful availability model with fitted residual contention entry.

Solo dispatch, contention entry residual, grant expiry and early yield are
different trained states. Repeated decisions never reset an existing deadline.
Rates advance solo reference work only; no physical SM ownership is claimed.
"""
from __future__ import annotations

import copy

from gsdeploy.forecast_errors import UnknownEvidence
from .conditional_availability_scheduler_v8 import frontier,cell_key
from .dag_phase_scheduler_v8 import SchedulingDecision


class TrainedDagAvailabilityScheduler:
    def __init__(self,policy,*,allow_unvalidated_training_candidate=False):
        if allow_unvalidated_training_candidate is not True:
            raise ValueError('Explicit trained availability-hypothesis opt-in required')
        self.policy=copy.deepcopy(policy)
        if (policy.get('schema')!='gsdeploy.trained_DAG_availability_policy.v8'
                or policy.get('reference_work_domain')!='SOLO_ONLY_REFERENCE_NS'
                or policy.get('training_only') is not True):
            raise ValueError('Frozen solo-reference training availability policy required')
        self.cells={cell_key(r['current_owner'],r['reason'],r['frontier']):copy.deepcopy(r) for r in policy['cells']}
        self.solo={cell_key(None,'uncontended',r['frontier']):copy.deepcopy(r) for r in policy['uncontended_cells']}
        self.initial_solo={cell_key(None,'uncontended',r['frontier']):copy.deepcopy(r) for r in policy['initial_uncontended_cells']}
        if len(self.cells)!=len(policy['cells']) or len(self.solo)!=len(policy['uncontended_cells']):
            raise ValueError('Distinct exact policy cells required')
        for row in self.cells.values():
            if (row['reason'] not in ('initial','contention_entry','grant_expired','yield')
                    or type(row['grant_budget_ns']) is not int or row['grant_budget_ns']<=0
                    or type(row['handoff_ns']) is not int or row['handoff_ns']<0
                    or row['next_owner'] not in {r[0] for r in row['frontier']}):
                raise ValueError('Explicit positive residual/grant and nonnegative handoff required')
        self.current_owner=None;self.grant_end_ns=None;self.handoff_end_ns=None
        self.mode=None;self.last_clock_ns=None;self.transitions=[]

    def choose(self,view,state,reason):
        row=self.cells.get(cell_key(self.current_owner,reason,state))
        if row is None:
            raise UnknownEvidence('UNKNOWN_TRAINED_AVAILABILITY_STATE:'+cell_key(self.current_owner,reason,state))
        old=self.current_owner;self.current_owner=row['next_owner'];self.mode='contended'
        self.handoff_end_ns=view.clock_ns+row['handoff_ns'];self.grant_end_ns=self.handoff_end_ns+row['grant_budget_ns']
        self.transitions.append(dict(clock_ns=view.clock_ns,previous_owner=old,next_owner=self.current_owner,
            reason=reason,frontier=state,handoff_end_ns=self.handoff_end_ns,grant_end_ns=self.grant_end_ns))

    def decide(self,view):
        if type(view.clock_ns) is not int or self.last_clock_ns is not None and view.clock_ns<self.last_clock_ns:
            raise UnknownEvidence('MONOTONIC_TRAINED_AVAILABILITY_CLOCK_REQUIRED')
        self.last_clock_ns=view.clock_ns;state=frontier(view);owners={v[0] for v in state}
        if not owners:return SchedulingDecision({})
        if self.handoff_end_ns is not None and view.clock_ns<self.handoff_end_ns:
            return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},self.handoff_end_ns)
        if len(owners)==1:
            owner=next(iter(owners))
            if self.current_owner!=owner:
                table=self.initial_solo if self.current_owner is None else self.solo
                solo=table.get(cell_key(None,'uncontended',state))
                if solo is None:raise UnknownEvidence('UNKNOWN_EXACT_UNCONTENDED_AVAILABILITY_STATE:'+cell_key(None,'uncontended',state))
                old=self.current_owner;self.current_owner=owner;self.handoff_end_ns=view.clock_ns+solo['dispatch_wait_ns']
                self.grant_end_ns=None;self.transitions.append(dict(clock_ns=view.clock_ns,previous_owner=old,
                    next_owner=owner,reason='uncontended_dispatch',frontier=state,handoff_end_ns=self.handoff_end_ns,grant_end_ns=None))
            # Preserve a prior finite grant while uncontended. A returning peer
            # must consume its remainder rather than trigger another full grant.
            self.mode='uncontended'
            if view.clock_ns<self.handoff_end_ns:
                return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},self.handoff_end_ns)
            return SchedulingDecision({n.key:1. for n in view.active_gpu_nodes})
        if self.current_owner is None:self.choose(view,state,'initial')
        elif self.current_owner not in owners:self.choose(view,state,'yield')
        elif self.mode=='uncontended':
            if self.grant_end_ns is None or self.grant_end_ns<=view.clock_ns:
                self.choose(view,state,'contention_entry')
            else:self.mode='contended'
        elif view.clock_ns>=self.grant_end_ns:self.choose(view,state,'grant_expired')
        if view.clock_ns<self.handoff_end_ns:
            return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},self.handoff_end_ns)
        return SchedulingDecision({n.key:float(n.key[0]==self.current_owner) for n in view.active_gpu_nodes},self.grant_end_ns)

    def provenance(self):
        return dict(kind='trained_DAG_state_conditional_availability',policy=copy.deepcopy(self.policy),
            transition_count=len(self.transitions),transitions=copy.deepcopy(self.transitions),
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',physical_SM_service_identified=False,
            independent_per_node_progress_within_owner_hypothesis=True,
            remaining_grant_reset_by_host_events=False,checkpoint_supported=False,
            unknown_state_fallback=False,independent_prediction_accuracy_validated=False)
