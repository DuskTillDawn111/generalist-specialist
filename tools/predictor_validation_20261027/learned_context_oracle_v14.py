"""Compose learned initial state, cumulative SAVE, mark and handoff candidates.

This is a prospective statistical approximation. Context envelopes do not
identify an exclusive physical execution gate. The execution engine's binary
reference-work progress is an explicit approximation to be independently tested.
"""
from __future__ import annotations

import copy
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .context_initial_prior_v14 import realize_initial_scenario
from .held_context_scheduler_v12_1 import ContextAvailabilityDecision
from .held_context_scheduler_v14 import HeldContextScheduler, InitialContextDecision
from .dag_phase_scheduler_v8 import SchedulingDecision


def plain_frontier(query):
    return [[owner,task,pool,list(phases)] for owner,task,pool,phases in query.pending_frontier]


class LearnedContextOracle:
    def __init__(self, *, initial_prior, save_controller, next_owner, handoff,
                 public_setup, public_workload, source_identity,
                 initial_quantile, initial_readiness_rule, handoff_readiness_rule,
                 allow_initial_pooling, allow_observer_transfer,
                 allow_flagged_statistical_extrapolation):
        if (not isinstance(source_identity,str) or len(source_identity)!=64
                or any(c not in '0123456789abcdef' for c in source_identity)
                or type(initial_quantile) not in (float,int) or not 0<initial_quantile<1):
            raise ValueError('Frozen source identity and predeclared interior scenario quantile required')
        self.initial_prior=initial_prior;self.controller=save_controller
        self.next_owner=next_owner;self.handoff=handoff
        self.public_setup=copy.deepcopy(public_setup);self.public_workload=copy.deepcopy(public_workload)
        self.identity=source_identity;self.quantile=initial_quantile
        self.initial_rule=initial_readiness_rule;self.handoff_rule=handoff_readiness_rule
        self.allow_pooling=allow_initial_pooling;self.allow_transfer=allow_observer_transfer
        self.allow_extrapolation=allow_flagged_statistical_extrapolation
        self.initial_prediction=None;self.selected_initial=None
        self.restore_key=None;self.last_restored_owner=None;self.previous_held_owner=None
        self.hazard=None;self.last_frontier=None;self.uncertainty=set();self.decisions=[]

    def initialize(self, query):
        if self.initial_prediction is not None:
            raise UnknownEvidence('INITIAL_PRIOR_CANNOT_BE_RESELECTED')
        prior=self.initial_prior.predict(first_pending_frontier=plain_frontier(query),
            public_execution_setup=self.public_setup,
            allow_unvalidated_frontier_pooling=self.allow_pooling,
            allow_unvalidated_observer_transfer=self.allow_transfer)
        self.initial_prediction=copy.deepcopy(prior)
        # A point forecast cannot silently condition on known-holder rows and
        # renormalize away unidentified initial probability mass.
        if prior['unknown_holder_mass']>0:
            raise UnknownEvidence('UNRESOLVED_INITIAL_HOLDER_MASS_PROHIBITS_POINT_FORECAST')
        if not math.isclose(prior['probability_sum'],1.,abs_tol=1e-12):
            raise UnknownEvidence('COMPLETE_INITIAL_PRIOR_PROBABILITY_REQUIRED')
        total=0.;selected=None
        for scenario in prior['scenarios']:
            total+=scenario['probability']
            if self.quantile<total:
                selected=scenario;break
        if selected is None:
            raise UnknownEvidence('COMPLETE_INITIAL_PRIOR_PROBABILITY_REQUIRED')
        realized=realize_initial_scenario(selected,query_clock_ns=query.clock_ns,
                                        readiness_rule=self.initial_rule)
        self.selected_initial=dict(realized,probability=selected['probability'])
        if prior['observer_transfer_hypothesis']:self.uncertainty.add('INITIAL_OBSERVER_TRANSFER_HYPOTHESIS')
        if prior['conditioning_tier']!='EXACT_FIRST_PENDING_FRONTIER':
            self.uncertainty.add(prior['conditioning_tier'])
        if prior['unresolved_point_readiness_mass']>0:
            self.uncertainty.add('INITIAL_PRIOR_CONTAINS_NONPOINT_READINESS')
        return InitialContextDecision(realized['held_owner'],realized['context_age_ns'],
                                      realized['service_ready_delay_ns'],self.identity)

    def observe_query(self, query):
        if query.held_owner is None or query.restore_start_ns is None or not query.service_available:
            return
        frontier=plain_frontier(query);key=(query.held_owner,query.restore_start_ns)
        if key!=self.restore_key:
            self.previous_held_owner=self.last_restored_owner
            self.last_restored_owner=query.held_owner;self.restore_key=key
            self.hazard=self.controller.start_restore(query.held_owner,query.restore_start_ns,
                frontier,str(key),observation_ns=query.clock_ns)
            self.last_frontier=frontier
        elif (frontier!=self.last_frontier or self.hazard['next_review_ns'] is not None
                and query.clock_ns>=self.hazard['next_review_ns']):
            self.hazard=self.controller.update(self.hazard['state'],query.clock_ns,frontier)
            self.last_frontier=frontier
        if self.hazard is not None:
            self.uncertainty.update(self.hazard['uncertainty_reasons'])
            if self.hazard['status'] in ('UNKNOWN_HAZARD_HISTORY','EMPTY_FRONTIER_UNKNOWN'):
                raise UnknownEvidence('CUMULATIVE_SAVE_HISTORY_UNIDENTIFIED')
            if self.hazard['uncertainty_reasons'] and not self.allow_extrapolation:
                raise UnknownEvidence('FLAGGED_HAZARD_HYPOTHESIS_NOT_AUTHORIZED_BY_MODEL_CONFIGURATION')

    def save_due(self, query):
        return (self.hazard is not None and self.hazard['status']=='SAVE_BUDGET_REACHED'
                and self.restore_key==(query.held_owner,query.restore_start_ns))

    def decide(self, query):
        self.observe_query(query)
        if self.hazard is None:
            raise UnknownEvidence('CONDITIONAL_SAVE_MODEL_NOT_INITIALIZED')
        now=query.clock_ns
        if not self.save_due(query):
            review=self.hazard['next_review_ns']
            if review is None or review<=now:
                raise UnknownEvidence('FUTURE_CUMULATIVE_SAVE_REVIEW_UNAVAILABLE')
            # A switch also schedules a post-SERVICE review. A frontier
            # change at SERVICE can query the new restore's hazard before
            # that earlier review expires. Keep both clocks distinct: the
            # scheduler reviews earlier without renewing the hazard budget.
            earlier=query.existing_review_deadline_ns
            if earlier is not None and earlier>now:
                review=min(review,earlier)
            return ContextAvailabilityDecision('hold',review)
        save=self.hazard['absolute_save_candidate_ns']
        if save!=now:
            raise UnknownEvidence('ENGINE_SKIPPED_PREDICTED_SAVE_BOUNDARY_NO_REWIND')
        mark=self.next_owner.predict_given_save_now(held_owner=query.held_owner,
            context_age_ns=query.context_age_ns,pending_frontier=plain_frontier(query),
            public_workload=self.public_workload,previous_held_owner=self.previous_held_owner)
        target=mark['most_likely_next_owner']
        if target==query.held_owner:
            raise UnknownEvidence('UNSUPPORTED_SAME_CONTEXT_REACQUISITION_PREDICTION')
        handoff=self.handoff.query(held_owner=query.held_owner,next_owner=target,
            context_age_ns=query.context_age_ns,pending_frontier=plain_frontier(query),
            readiness_rule=self.handoff_rule)
        flags=[*mark.get('ood_flags',[]),*handoff.get('ood_flags',[])]
        self.uncertainty.update(flags)
        if flags and not self.allow_extrapolation:
            raise UnknownEvidence('FLAGGED_MARK_OR_HANDOFF_EXTRAPOLATION_PROHIBITS_FORECAST')
        gap=handoff['save_to_restore_gap_ns'];delay=handoff['service_ready_delay_after_restore_ns']
        if any(type(t) is not int or t<0 for t in (gap,delay)):
            raise UnknownEvidence('FINITE_NONNEGATIVE_HANDOFF_REFERENCE_TIME_REQUIRED')
        restore=now+gap;service=restore+delay
        self.decisions.append(dict(clock_ns=now,held_owner=query.held_owner,
            previous_held_owner=self.previous_held_owner,context_age_ns=query.context_age_ns,
            next_owner=target,mark=mark,handoff=handoff,
            cumulative_hazard_budget=self.hazard['hazard_budget']))
        # +1 is the engine's original integer time resolution, not a grant or
        # a global request admission limit.
        return ContextAvailabilityDecision('switch',service+1,target,now,restore,service)

    def provenance(self):
        return dict(kind='composed_learned_context_candidate.v14',source_identity=self.identity,
            initial_quantile=self.quantile,initial_readiness_rule=self.initial_rule,
            handoff_readiness_rule=self.handoff_rule,initial_prediction=self.initial_prediction,
            selected_initial=self.selected_initial,save_controller=self.controller.provenance(),
            uncertainty_flags=sorted(self.uncertainty),decision_count=len(self.decisions),
            decisions=copy.deepcopy(self.decisions),
            future_workload_used_by_queue_engine=True,future_workload_used_as_next_owner_feature=False,
            physical_exclusive_execution_gate=False,binary_reference_progress_is_approximation=True,
            independent_prediction_validation=False,global_request_cap_used=False)


class LearnedContextScheduler(HeldContextScheduler):
    def decide(self, view):
        # The main engine calls at every activity/host boundary. In particular,
        # record entry into an empty frontier before the base scheduler's
        # ordinary empty-ready early return.
        query=self.snapshot(view);self.oracle.observe_query(query)
        if not view.active_gpu_nodes and self.oracle.save_due(query):
            self._accept(self.oracle.decide(query),query)
            self.last_frontier=query.pending_frontier
            future=[t for t,_,_ in self.context_events if t>view.clock_ns]
            if self.review_deadline_ns is not None and self.review_deadline_ns>view.clock_ns:
                future.append(self.review_deadline_ns)
            return SchedulingDecision({},min(future) if future else None)
        return super().decide(view)
