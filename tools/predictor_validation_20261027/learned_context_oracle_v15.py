"""v15 learned initial transition composition; frozen v14 ordinary dynamics."""
import copy
import math

from gsdeploy.forecast_errors import UnknownEvidence
from .context_initial_prior_v15 import realize_joint_initial_scenario
from .dag_phase_scheduler_v8 import SchedulingDecision
from .held_context_scheduler_v14 import InitialContextDecision
from .held_context_scheduler_v15 import HeldContextScheduler,InitialTransitionDecision
from .learned_context_oracle_v14 import LearnedContextOracle as V14Oracle


class LearnedContextOracle(V14Oracle):
    def initialize(self,query):
        if self.initial_prediction is not None:raise UnknownEvidence('INITIAL_PRIOR_CANNOT_BE_RESELECTED')
        frontier=[[a,t,p,list(phases)] for a,t,p,phases in query.pending_frontier]
        prior=self.initial_prior.predict(first_pending_frontier=frontier,public_execution_setup=self.public_setup,
            allow_unvalidated_frontier_pooling=self.allow_pooling,
            allow_unvalidated_observer_transfer=self.allow_transfer)
        self.initial_prediction=copy.deepcopy(prior)
        if prior['unknown_state_mass']>0:raise UnknownEvidence('UNRESOLVED_INITIAL_STATE_MASS_PROHIBITS_POINT_FORECAST')
        if not math.isclose(prior['probability_sum'],1.,abs_tol=1e-12):
            raise UnknownEvidence('COMPLETE_INITIAL_PRIOR_PROBABILITY_REQUIRED')
        total=0.;selected=None
        for scenario in prior['scenarios']:
            total+=scenario['probability']
            if self.quantile<total:selected=scenario;break
        if selected is None:raise UnknownEvidence('COMPLETE_INITIAL_PRIOR_PROBABILITY_REQUIRED')
        realized=realize_joint_initial_scenario(selected,query_clock_ns=query.clock_ns,readiness_rule=self.initial_rule)
        self.selected_initial=dict(realized,probability=selected['probability'])
        if prior['observer_transfer_hypothesis']:self.uncertainty.add('INITIAL_OBSERVER_TRANSFER_HYPOTHESIS')
        if prior['conditioning_tier']!='EXACT_FIRST_PENDING_FRONTIER':self.uncertainty.add(prior['conditioning_tier'])
        if prior['unresolved_point_readiness_mass']>0:self.uncertainty.add('INITIAL_PRIOR_CONTAINS_NONPOINT_READINESS')
        if realized['service_ready_ns'] is None:
            raise UnknownEvidence('INITIAL_SERVICE_READINESS_BOUNDS_DO_NOT_IDENTIFY_A_POINT')
        if realized['numerical_readiness_is_hypothesis']:
            self.uncertainty.add('INITIAL_SERVICE_READINESS_'+self.initial_rule.upper())
        if realized['state_kind']=='HELD_CONTEXT':
            return InitialContextDecision(realized['held_owner'],realized['context_age_ns'],
                realized['service_ready_ns']-query.clock_ns,self.identity)
        # At the first future restoration, ordinary v14 observe_query records
        # this already-finished previous holder as history, not as current state.
        self.last_restored_owner=realized['previous_held_owner']
        self.uncertainty.add('INITIAL_SCHEDULER_IN_TRANSITION_STATISTICAL_SCENARIO')
        bounds=realized['service_ready_delay_after_restore_bounds_ns']
        return InitialTransitionDecision(realized['previous_held_owner'],realized['elapsed_gap_ns'],
            realized['next_owner'],realized['residual_restore_gap_ns'],
            realized['service_ready_ns']-realized['next_RESTORE_ns'],self.identity,
            realized['source_scenario_id'],self.initial_rule,bounds['upper_ns'],
            realized['numerical_readiness_is_hypothesis'])

    def provenance(self):
        result=super().provenance()
        result.update(kind='composed_learned_held_or_initial_transition_candidate.v15',
            initial_joint_state_kind=None if self.selected_initial is None else self.selected_initial['state_kind'],
            initial_transition_future_events_from_frozen_prior_only=True,
            prospective_source_event_lookup=False)
        return result


class LearnedContextScheduler(HeldContextScheduler):
    def decide(self,view):
        query=self.snapshot(view);self.oracle.observe_query(query)
        if not view.active_gpu_nodes and self.oracle.save_due(query):
            self._accept(self.oracle.decide(query),query)
            self.last_frontier=query.pending_frontier
            future=[t for t,_,_ in self.context_events if t>view.clock_ns]
            if self.review_deadline_ns is not None and self.review_deadline_ns>view.clock_ns:
                future.append(self.review_deadline_ns)
            return SchedulingDecision({},min(future) if future else None)
        return super().decide(view)
