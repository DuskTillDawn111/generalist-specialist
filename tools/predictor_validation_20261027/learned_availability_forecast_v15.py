"""Full original cohort and six terms, using composed learned context models.

The original source-verified work, host, lifecycle, resource and startup
providers are reused. The v15 candidate adds explicit initial transition states and compact hazard
history state. The work, host and score models are unchanged. This adapter
does not seal a protocol, read validation outcomes or submit GPU work.
"""
from __future__ import annotations

import copy

from .availability_forecast_v11 import AvailabilityForecast
from .context_initial_prior_v15 import JointInitialContextPrior
from .context_initial_prior_v14 import CORE_SETUP_FIELDS, OBSERVER_SETUP_FIELDS
from .context_next_owner_v14 import NextOwnerCandidate
from .cumulative_save_hazard_v15 import CumulativeSaveHazardController
from .dag_phase_scheduler_v8 import DagPhaseFullHorizonModel
from .handoff_candidate_v14 import HandoffCandidate
from .learned_context_oracle_v15 import LearnedContextOracle, LearnedContextScheduler
from .multi_native_composite_v5 import checked
from .profiling import digest
from .prospective_flow_contract_v11 import bind_public, original_flow_defaults

SCHEMA = 'gsdeploy.learned_context_full_cohort_forecast.v15'
OPTIONS = dict(initial_quantile=.5, initial_readiness_rule='lower_bound_hypothesis',
    handoff_readiness_rule='lower_bound_reference_availability_hypothesis',
    save_budget_policy='MEDIAN_LOG2',empty_frontier_policy='NO_QUERY_PAUSE',
    allow_initial_pooling=True,allow_observer_transfer=True,
    allow_flagged_statistical_extrapolation=True)


class LearnedAvailabilityForecast(AvailabilityForecast):
    def __init__(self, configuration, *, allow_unvalidated_training_hypothesis=False):
        required={'schema','base_model_configuration','initial_prior_model','next_owner_model',
                  'handoff_model','save_model','save_training_report','options'}
        if (set(configuration)!=required or configuration['schema']!=SCHEMA
                or configuration['options']!=OPTIONS or allow_unvalidated_training_hypothesis is not True):
            raise ValueError('Complete source-bound, predeclared learned candidate configuration required')
        self.learned_configuration=copy.deepcopy(configuration)
        self.learned_identity=digest(configuration)
        self.initial_prior=JointInitialContextPrior(checked(configuration['initial_prior_model']))
        self.mark=NextOwnerCandidate(checked(configuration['next_owner_model']))
        self.handoff=HandoffCandidate(checked(configuration['handoff_model']))
        report=checked(configuration['save_training_report'])
        if (report['models']['age_additive']!=configuration['save_model']
                or report.get('training_only') is not True or report.get('validation_labels_used') is not False):
            raise ValueError('Actual source-verified training SAVE model/report binding required')
        self.save_controller=CumulativeSaveHazardController(configuration['save_model'],
            allow_unvalidated_approximation=True,budget_policy=OPTIONS['save_budget_policy'],
            empty_frontier_policy=OPTIONS['empty_frontier_policy'])
        self._execution_binding=None
        super().__init__(configuration['base_model_configuration'],
                         allow_unvalidated_training_hypothesis=True)

    def engine(self, *, max_events):
        if self._execution_binding is None:
            raise ValueError('Whole current public cohort and actual target setup must bind the oracle')
        setup,workload=self._execution_binding
        oracle=LearnedContextOracle(initial_prior=self.initial_prior,save_controller=self.save_controller,
            next_owner=self.mark,handoff=self.handoff,public_setup=setup,public_workload=workload,
            source_identity=self.learned_identity,**{k:OPTIONS[k] for k in (
                'initial_quantile','initial_readiness_rule','handoff_readiness_rule',
                'allow_initial_pooling','allow_observer_transfer','allow_flagged_statistical_extrapolation')})
        scheduler=LearnedContextScheduler(self.aliases,oracle=oracle)
        return DagPhaseFullHorizonModel(self.aliases,self.plans,duration_provider=self.work,
            release_provider=self.release,phase_provider=self.phases,scheduler=scheduler,
            lifecycle_provider=self.lifecycle,power_provider=self.resources,memory_provider=self.resources,
            require_energy=True,require_memory=True,allow_independent_copy_hypothesis=True,
            allow_no_implicit_default_stream_hypothesis=True,max_events=max_events)

    def execute(self, plan_ref, candidate_id, public, routes, startup_model_ref, *, max_events=5_000_000):
        if self._execution_binding is not None:
            raise ValueError('One full-cohort forecast at a time on this source-verified provider')
        plan=checked(plan_ref);context=checked(plan['context'])
        defaults=original_flow_defaults(plan,self.defaults)
        rows=bind_public(plan,candidate_id,public,routes,defaults,plan['weights'])
        if any(k not in context for k in CORE_SETUP_FIELDS):
            raise ValueError('Actual target core execution setup required')
        setup={k:copy.deepcopy(context[k]) for k in CORE_SETUP_FIELDS}
        # A missing observer field is explicitly absent. It is never replaced
        # by the profiler's whole-lifetime value to manufacture setup equality.
        setup.update({k:copy.deepcopy(context.get(k)) for k in OBSERVER_SETUP_FIELDS})
        workload=[[r['alias'],r['task'],r['pool_id'],r['arrival_ns']-public['phase_start_ns']] for r in rows]
        self._execution_binding=(setup,workload)
        audit_cursor=self.save_controller.begin_audit_scope(digest(dict(
            plan=plan_ref,candidate_id=candidate_id,public=public,routes=routes,
            learned_configuration_identity=self.learned_identity)))
        try:
            result=super().execute(plan_ref,candidate_id,public,routes,startup_model_ref,max_events=max_events)
        finally:
            self._execution_binding=None
        result.update(schema=SCHEMA,configuration=self.learned_configuration,
            learned_scheduler_source_identity=self.learned_identity,
            actual_target_public_setup=setup,
            absent_target_observer_fields=[k for k in OBSERVER_SETUP_FIELDS if k not in context],
            original_six_term_and_full_cohort_contract_reused=True,
            old_exact_availability_policy_used_for_scheduling=False,
            joint_host_observation_bank_is_executable_release_provider=False,
            independent_context_accuracy_validated=False,
            binary_reference_progress_is_approximation=True,physical_exclusive_gate_claimed=False)
        result['complete_empty_hazard_interval_audit']=self.save_controller.export_audit_since(audit_cursor)
        self.save_controller.assert_sources_unchanged()
        for key in ('initial_prior_model','next_owner_model','handoff_model','save_model','save_training_report'):
            checked(self.learned_configuration[key])
        return result
