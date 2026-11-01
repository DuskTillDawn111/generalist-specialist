"""Full-cohort composition of lifecycle-conditioned hazard and mark candidates.

Reuses original frozen DAG/work/host/phase/lifecycle/startup and six-term score.
No routes, request cap, utility thresholds or drain windows are changed. All
new source checks hash bytes, including inner compiled training references.
"""
from __future__ import annotations

import copy

from . import (cumulative_lifecycle_hazard_v24, learned_lifecycle_oracle_v25,
               owner_lifecycle_engine_v21, owner_lifecycle_hazard_v24_fit,
               owner_lifecycle_hazard_query_v24, owner_lifecycle_mark_v21)
from .availability_forecast_v11 import AvailabilityForecast
from .context_initial_prior_v14 import CORE_SETUP_FIELDS, OBSERVER_SETUP_FIELDS
from .cumulative_lifecycle_hazard_v24 import CumulativeLifecycleHazard
from .learned_availability_forecast_v16 import LearnedAvailabilityForecast as V16Forecast
from .learned_availability_forecast_v19 import QueueDiagnosticEngine, implementation_sources
from .learned_availability_forecast_v20 import RetainReturnedForecast, attach_unscored_diagnostic, verify_sources
from .learned_availability_forecast_v15 import OPTIONS
from .learned_lifecycle_oracle_v25 import LearnedLifecycleOracle, LearnedLifecycleScheduler
from .multi_native_composite_v5 import checked
from .native_collect import receipt
from .native_queue_phase_busy_v10 import _references
from .owner_lifecycle_engine_v21 import LifecycleDagPhaseModel
from .owner_lifecycle_hazard_v24_fit import EDGES_NS
from .owner_lifecycle_hazard_query_v24 import LifecycleSaveHazard
from .owner_lifecycle_mark_v21 import LifecycleActiveServiceMark
from .profiling import digest
from .prospective_flow_contract_v11 import bind_public, original_flow_defaults

SCHEMA = 'gsdeploy.full_cohort_lifecycle_conditioned_forecast.v25'
OPTIONS_V25 = dict(budget_policy='MEDIAN_LOG2',
    allow_active_service_context_hypothesis=True,
    allow_flagged_statistical_extrapolation=True,
    empty_frontier_policy='LEARNED_FULL_RISK_LIFECYCLE_HAZARD',
    same_holder_reacquisition='UNKNOWN', physical_exclusive_gate_claimed=False)


class LearnedLifecycleForecast(V16Forecast):
    def __init__(self,configuration,*,allow_unvalidated_training_hypothesis=False):
        if (set(configuration) != {'schema','base_v15_configuration','lifecycle_mark_model',
                                  'lifecycle_hazard_model','options'}
                or configuration['schema'] != SCHEMA or configuration['options'] != OPTIONS_V25
                or allow_unvalidated_training_hypothesis is not True):
            raise ValueError('Complete predeclared lifecycle candidate configuration required')
        self.lifecycle_configuration = copy.deepcopy(configuration)
        self.lifecycle_identity = digest(configuration)
        # Reconstruct the original providers and immutable graph without editing
        # old configurations or changing their source identities.
        super().__init__(configuration['base_v15_configuration'],allow_unvalidated_training_hypothesis=True)
        mark = checked(configuration['lifecycle_mark_model'])
        hazard = checked(configuration['lifecycle_hazard_model'])
        statistics = checked(hazard['source_statistics'])
        if (hazard.get('training_only') is not True or statistics.get('missing_marked_SAVE_labels')
                or statistics.get('original_marked_SAVE_labels_preserved') != statistics.get('original_marked_SAVE_labels')):
            raise ValueError('Training-only complete physical SAVE risk evidence required')
        self._mark_model = copy.deepcopy(mark)
        self.mark = self._new_mark()
        candidate = LifecycleSaveHazard(hazard)
        sources = _references([configuration,mark,hazard,statistics,
            [dict(ref) for ref in self.compiled.sources],implementation_sources(),receipt(__file__),
            *[receipt(m.__file__) for m in (cumulative_lifecycle_hazard_v24,learned_lifecycle_oracle_v25,
                owner_lifecycle_engine_v21,owner_lifecycle_hazard_v24_fit,
                owner_lifecycle_hazard_query_v24,owner_lifecycle_mark_v21)]])
        self._lifecycle_sources = tuple(sources)
        verify_sources(self._lifecycle_sources)
        self.save_controller = CumulativeLifecycleHazard(candidate,age_edges_ns=EDGES_NS,
            sources=sources,candidate_identity=configuration['lifecycle_hazard_model']['sha256'],
            allow_unvalidated_approximation=True,budget_policy=OPTIONS_V25['budget_policy'])
        self._latest_engine = None

    def _new_mark(self):
        return LifecycleActiveServiceMark(copy.deepcopy(self._mark_model),
                                        allow_active_service_context_hypothesis=True)

    def engine(self,*,max_events):
        if self._execution_binding is None:
            raise ValueError('Actual target setup and complete public workload must bind the oracle')
        setup,workload = self._execution_binding
        # The lifecycle mark retains an observation clock. A fresh forecast
        # must not inherit the previous candidate's clock or handler state.
        self.mark = self._new_mark()
        oracle = LearnedLifecycleOracle(initial_prior=self.initial_prior,save_controller=self.save_controller,
            next_owner=self.mark,handoff=self.handoff,public_setup=setup,public_workload=workload,
            source_identity=self.lifecycle_identity,**{k:OPTIONS[k] for k in (
                'initial_quantile','initial_readiness_rule','handoff_readiness_rule',
                'allow_initial_pooling','allow_observer_transfer','allow_flagged_statistical_extrapolation')})
        scheduler = LearnedLifecycleScheduler(self.aliases,oracle=oracle)
        self.compiled.assert_sources_unchanged()
        engine = LifecycleDagPhaseModel(self.compiled.graph.aliases,self.compiled.graph.plans,
            frozen_model=self.compiled.graph,duration_provider=self.compiled.duration,
            release_provider=self.compiled.release,phase_provider=self.compiled.phase,scheduler=scheduler,
            lifecycle_provider=self.lifecycle,power_provider=self.resources,memory_provider=self.resources,
            require_energy=True,require_memory=True,allow_independent_copy_hypothesis=True,
            allow_no_implicit_default_stream_hypothesis=True,max_events=max_events,
            source_receipts=self._lifecycle_sources)
        queue = QueueDiagnosticEngine(engine,self.compiled,list(self._lifecycle_sources))
        self._latest_engine = RetainReturnedForecast(queue,self._lifecycle_sources)
        return self._latest_engine

    def execute(self,plan_ref,candidate_id,public,routes,startup_model_ref,*,max_events=5_000_000):
        if self._execution_binding is not None:
            raise ValueError('One source-bound full-cohort execution at a time')
        verify_sources(self._lifecycle_sources)
        plan = checked(plan_ref)
        context = checked(plan['context'])
        if any(k not in context for k in CORE_SETUP_FIELDS):
            raise ValueError('Actual target execution setup required')
        rows = bind_public(plan,candidate_id,public,routes,original_flow_defaults(plan,self.defaults),plan['weights'])
        setup = {k:copy.deepcopy(context[k]) for k in CORE_SETUP_FIELDS}
        setup.update({k:copy.deepcopy(context.get(k)) for k in OBSERVER_SETUP_FIELDS})
        workload = [[r['alias'],r['task'],r['pool_id'],r['arrival_ns']-public['phase_start_ns']] for r in rows]
        self._execution_binding = setup,workload
        self._latest_engine = None
        try:
            result = AvailabilityForecast.execute(self,plan_ref,candidate_id,public,routes,startup_model_ref,
                                                  max_events=max_events)
            engine = self._latest_engine
            if engine is None or engine.sidecar is None:
                raise ValueError('Actual lifecycle engine and full-public queue entry required')
            attach_unscored_diagnostic(result,engine)
            result.update(schema=SCHEMA,configuration=self.lifecycle_configuration,
                learned_scheduler_source_identity=self.lifecycle_identity,
                actual_target_public_setup=setup,static_provider_compilation=self.compiled.diagnostic(),
                CPU_representation_version=16,lifecycle_scheduler_version=25,
                queue_diagnostic_sidecar=engine.sidecar,
                original_six_term_and_full_cohort_contract_reused=True,
                global_request_concurrency_cap_used=False,physical_exclusive_gate_claimed=False,
                binary_reference_progress_is_approximation=True,
                independent_context_accuracy_validated=False,
                joint_host_observation_bank_is_executable_release_provider=False,
                rolling_feedback_correction_applied=False)
            return result
        finally:
            self._execution_binding = None
            verify_sources(self._lifecycle_sources)
