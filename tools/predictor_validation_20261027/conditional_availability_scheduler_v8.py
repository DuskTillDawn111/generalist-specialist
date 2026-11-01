"""Stateful conditional context-availability candidate, with no default policy.

Each supported frontier/reason/current-owner cell explicitly chooses the next
owner, grant budget and handoff duration. This mechanism advances solo reference
work; it does not consume co-run kernel lifetimes as service. The input policy
must be fitted and frozen by a separate training producer before real use.
"""
from __future__ import annotations

import copy
import json

from gsdeploy.forecast_errors import UnknownEvidence
from .dag_phase_scheduler_v8 import SchedulingDecision


def frontier(view):
    owners={}
    for node in view.active_gpu_nodes:
        alias=node.key[0]
        value=owners.setdefault(alias,dict(task=node.task,pool_id=node.pool_id,phases=set()))
        if (value['task'],value['pool_id']) != (node.task,node.pool_id):
            raise UnknownEvidence('ONE_PUBLIC_REQUEST_PER_ACTIVE_OWNER_REQUIRED')
        if node.phase=='UNBOUND_PHASE':
            raise UnknownEvidence('STATIC_BOUND_FRONTIER_PHASE_REQUIRED')
        value['phases'].add(node.phase)
    return tuple((alias,v['task'],v['pool_id'],tuple(sorted(v['phases']))) for alias,v in sorted(owners.items()))


def cell_key(current_owner, reason, active_frontier):
    return json.dumps([current_owner,reason,active_frontier],sort_keys=True,separators=(',',':'))


class ConditionalAvailabilityScheduler:
    """Absolute grant deadlines survive host events, arrivals and phase changes.

    Calls at the same clock may contain new ready nodes, but do not re-grant,
    re-charge handoff, or advance work. The DAG engine alone advances work.
    Empty-frontier gaps preserve the previous grant's absolute deadline.
    """
    def __init__(self, policy, *, allow_unvalidated_training_candidate=False):
        if allow_unvalidated_training_candidate is not True:
            raise ValueError('Explicit unvalidated availability-candidate opt-in required')
        self.policy=copy.deepcopy(policy)
        if (self.policy.get('schema')!='gsdeploy.conditional_context_availability_policy.v8'
                or self.policy.get('reference_work_domain')!='SOLO_ONLY_REFERENCE_NS'
                or self.policy.get('training_only') is not True
                or self.policy.get('whole_corun_multiplier_used') is not False):
            raise ValueError('Explicit solo-reference training availability policy required')
        self.cells={}
        for row in self.policy['cells']:
            state=tuple((v[0],v[1],v[2],tuple(v[3])) for v in row['frontier'])
            if (not state or tuple(sorted(state))!=state
                    or row['reason'] not in ('initial','grant_expired','yield')
                    or row['next_owner'] not in {v[0] for v in state}
                    or type(row['grant_budget_ns']) is not int or row['grant_budget_ns']<=0
                    or type(row['handoff_ns']) is not int or row['handoff_ns']<0):
                raise ValueError('Complete exact conditional policy cell required')
            if row['current_owner']==row['next_owner'] and row['handoff_ns']!=0:
                raise ValueError('Same-context regrant cannot charge a context handoff')
            key=cell_key(row['current_owner'],row['reason'],state)
            if key in self.cells:raise ValueError('Duplicate exact conditional policy cell')
            self.cells[key]=copy.deepcopy(row)
        self.current_owner=None
        self.grant_end_ns=None
        self.handoff_end_ns=None
        self.pending_budget_ns=None
        self.last_clock_ns=None
        self.transitions=[]

    def _choose(self, view, state, reason):
        row=self.cells.get(cell_key(self.current_owner,reason,state))
        if row is None:
            raise UnknownEvidence('UNSUPPORTED_CONDITIONAL_AVAILABILITY_CELL:'+cell_key(self.current_owner,reason,state))
        previous=self.current_owner;self.current_owner=row['next_owner']
        self.handoff_end_ns=view.clock_ns+row['handoff_ns']
        self.pending_budget_ns=row['grant_budget_ns']
        self.grant_end_ns=self.handoff_end_ns+self.pending_budget_ns
        self.transitions.append(dict(clock_ns=view.clock_ns,previous_owner=previous,
            next_owner=self.current_owner,reason=reason,frontier=state,
            handoff_end_ns=self.handoff_end_ns,grant_end_ns=self.grant_end_ns))

    def decide(self, view):
        if type(view.clock_ns) is not int or (self.last_clock_ns is not None and view.clock_ns<self.last_clock_ns):
            raise UnknownEvidence('MONOTONIC_INTEGER_AVAILABILITY_CLOCK_REQUIRED')
        self.last_clock_ns=view.clock_ns
        state=frontier(view);owners={v[0] for v in state}
        if not owners:
            return SchedulingDecision({})
        # An in-flight handoff is never restarted by a new host/device event.
        if self.handoff_end_ns is not None and view.clock_ns<self.handoff_end_ns:
            return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},self.handoff_end_ns)
        if self.current_owner is None:
            self._choose(view,state,'initial')
        elif self.current_owner not in owners:
            self._choose(view,state,'yield')
        elif view.clock_ns>=self.grant_end_ns:
            self._choose(view,state,'grant_expired')
        if view.clock_ns<self.handoff_end_ns:
            return SchedulingDecision({n.key:0. for n in view.active_gpu_nodes},self.handoff_end_ns)
        return SchedulingDecision({n.key:float(n.key[0]==self.current_owner) for n in view.active_gpu_nodes},self.grant_end_ns)

    def provenance(self):
        return dict(kind='conditional_context_availability_candidate',
            reference_work_domain='SOLO_ONLY_REFERENCE_NS',
            rates='one for granted owner nodes, zero for all other owner nodes',
            same_owner_parallel_nodes='independent per-node progress hypothesis; not identified aggregate SM capacity',
            grant_deadline_reset_by_host_events=False,work_advanced_by_scheduler=False,
            checkpoint_supported=False,unknown_cell_fallback=False,
            physical_SM_service_identified=False,deployment_qualified=False,
            policy=copy.deepcopy(self.policy))
