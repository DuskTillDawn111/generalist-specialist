"""Frozen policy identity separated from stateful absolute scheduler deadlines."""
from __future__ import annotations

import copy

from .dag_frozen_model_v10 import freeze, plain, static_digest
from .trained_dag_availability_v8 import TrainedDagAvailabilityScheduler

STATE_FIELDS=('current_owner','grant_end_ns','handoff_end_ns','mode','last_clock_ns','transitions')


class CheckpointTrainedScheduler(TrainedDagAvailabilityScheduler):
    def __init__(self,policy):
        object.__setattr__(self,'_static_sealed',False)
        super().__init__(plain(policy),allow_unvalidated_training_candidate=True)
        self.policy_sha256=static_digest(self.policy)
        for name in ('policy','cells','solo','initial_solo'):
            setattr(self,name,freeze(getattr(self,name)))
        object.__setattr__(self,'_static_sealed',True)

    def __setattr__(self,name,value):
        if getattr(self,'_static_sealed',False) and name in (
                'policy_sha256','policy','cells','solo','initial_solo','_static_sealed'):
            raise AttributeError('Frozen scheduling policy cannot change')
        object.__setattr__(self,name,value)

    def __delattr__(self,name):
        raise AttributeError('Availability scheduler fields cannot be deleted')

    @classmethod
    def fresh_from(cls,scheduler):
        if type(scheduler) not in (cls,TrainedDagAvailabilityScheduler):
            raise ValueError('Explicit trained availability scheduler required')
        if any(getattr(scheduler,k) is not None for k in STATE_FIELDS[:-1]) or scheduler.transitions:
            raise ValueError('New complete forecast requires an initially empty scheduler')
        return cls(scheduler.policy)

    def checkpoint_configuration(self):
        return dict(policy_sha256=self.policy_sha256,reference_work_domain='SOLO_ONLY_REFERENCE_NS',
            dynamic_state_separate=True)

    def saved_state(self):
        return plain({name:getattr(self,name) for name in STATE_FIELDS})

    def restore_state(self,state):
        if type(state) is not dict or set(state)!=set(STATE_FIELDS):
            raise ValueError('Complete explicit availability scheduler state required')
        for name in STATE_FIELDS:
            value=copy.deepcopy(state[name])
            if name=='transitions':value=[dict(row) for row in value]
            setattr(self,name,value)

    def provenance(self):
        return dict(kind='trained_DAG_state_conditional_availability',policy=plain(self.policy),
            policy_sha256=self.policy_sha256,transition_count=len(self.transitions),
            transitions=copy.deepcopy(self.transitions),reference_work_domain='SOLO_ONLY_REFERENCE_NS',
            physical_SM_service_identified=False,independent_per_node_progress_within_owner_hypothesis=True,
            remaining_grant_reset_by_host_events=False,checkpoint_supported=True,
            checkpoint_state_verified_by_complete_prefix_replay=True,
            unknown_state_fallback=False,independent_prediction_accuracy_validated=False)
