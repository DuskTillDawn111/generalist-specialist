"""Prospective full-cohort adapter for the actual trained phase DAG engine.

Unlike the old twelve-template diagnostic, repeated requests and the complete
declared future suffix execute here. Unsupported host/work/scheduling states
remain UNKNOWN. No protocol is frozen or GPU request submitted by this module.
"""
from __future__ import annotations

import json

from gsdeploy.forecast_errors import UnknownEvidence
from .component_generation_support_v7 import ComponentGenerationSupport
from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from .dag_availability_complete_fit_v8 import derive
from .dag_phase_scheduler_v8 import DagPhaseFullHorizonModel
from .dag_preparation_host_release_v8 import DagPreparationHostReleaseProvider, SCENARIO
from .multi_native_composite_v5 import MultiNativeProvider, checked
from .native_collect import receipt
from .native_execution_projection_v5 import native_execution_projection
from .natural_resources_v6 import NaturalResourceProvider
from .prospective_flow_contract_v11 import bind_public, score_complete_forecast, original_flow_defaults
from .solo_reference_work_v8 import SoloOrdinalReferenceWork
from .startup_prediction_v11 import prediction as startup_prediction
from .static_dag_phase_binding_v8 import StaticDagPhaseBinding
from .trained_dag_availability_v8 import TrainedDagAvailabilityScheduler
from .training_source_guard_v11 import availability_training_sources

SCHEMA = 'gsdeploy.prospective_availability_model.v11'


class AvailabilityForecast:
    """Verify expensive immutable training providers once; new scheduler per run."""
    def __init__(self, configuration, *, allow_unvalidated_training_hypothesis=False):
        if (set(configuration) != {'schema','parameters','availability_model','component_bundle'}
                or configuration.get('schema') != SCHEMA
                or allow_unvalidated_training_hypothesis is not True):
            raise ValueError('Explicit source-bound unvalidated availability hypothesis required')
        self.configuration = configuration
        self.parameters = checked(configuration['parameters'])
        self.model = checked(configuration['availability_model'])
        self.availability_training_audit = availability_training_sources(self.model['source_DAG_reports'])
        regenerated = derive([r['path'] for r in self.model['source_DAG_reports']])
        if json.loads(json.dumps(regenerated)) != self.model:
            raise ValueError('Availability model differs from original training rederivation')
        self.bundle = checked(configuration['component_bundle'])
        self.defaults = self.bundle['forecast_defaults']
        self.aliases = self.defaults['aliases']
        self.host = checked(self.parameters['host_model'])
        dag = checked(self.host['dag'])
        self.plans = {(p['alias'],p['task'],p['pool_id']):p for p in dag['profiles']}
        if len(self.plans) != len(dag['profiles']):
            raise ValueError('Distinct original exact DAG components required')
        self.lifecycle = MultiNativeProvider(checked(self.bundle['composite']),self.aliases,
                                            host_scenario=self.bundle['host_supply_scenario'])
        support = checked(self.bundle['generation']['component_support'])
        if (support['generation_model'] != self.bundle['generation']['generation_work']
                or support['composite'] != self.bundle['composite']):
            raise ValueError('Generation model and component receipts differ')
        self.generation = ComponentGenerationSupport(support,self.lifecycle,
                                                    allow_exact_content_generation_hypothesis=True)
        bindings = {(b['alias'],b['task'],b['pool_id']):b for b in self.host['public_bindings']}
        for key,binding in bindings.items():
            generated = self.generation.public_bindings.get(key)
            if generated is None or any(generated.get(k) != v for k,v in binding.items()):
                raise ValueError('Preparation public work/content differs from generation support')
        self.release = DagPreparationHostReleaseProvider(self.host,self.host['execution_context_sha256'],
            self.plans,self.aliases,scenario=SCENARIO,allow_preparation_alignment_hypothesis=True,
            public_bindings=bindings)
        copy_provider = ExactSourceCopyMemsetHypothesisResolver(self.bundle['copy_memset_fit']['path'])
        if copy_provider.fit_source != self.bundle['copy_memset_fit']:
            raise ValueError('Changed original copy/memset reference')
        def copy_work(node,row):
            result = copy_provider(node,self.plans[row['alias'],row['task'],row['pool_id']]['source_sqlite'])
            if result.get('numeric_hypothesis_available') is not True:
                raise UnknownEvidence('EXACT_SOURCE_COPY_REFERENCE_WORK_UNAVAILABLE')
            return result['duration_ns']
        self.work = SoloOrdinalReferenceWork(self.host['dag'],self.parameters['solo_light_reviews'],
                                            copy_duration_provider=copy_work)
        self.phases = StaticDagPhaseBinding(self.host['dag'],self.parameters['solo_light_reviews'][:1])
        composite = checked(self.bundle['composite'])
        self.source_target_context_ref = composite['target']['context']
        self.source_target_context = checked(self.source_target_context_ref)
        self.source_native = native_execution_projection(self.source_target_context)
        self.resources = NaturalResourceProvider(checked(self.bundle['natural_resources']),
            self.source_target_context,allow_conditional_resource_hypothesis=True)

    def engine(self, *, max_events):
        if type(max_events) is not int or max_events <= 0:
            raise ValueError('Explicit positive whole-cohort event budget required')
        scheduler = TrainedDagAvailabilityScheduler(self.model['policy'],allow_unvalidated_training_candidate=True)
        return DagPhaseFullHorizonModel(self.aliases,self.plans,duration_provider=self.work,
            release_provider=self.release,phase_provider=self.phases,scheduler=scheduler,
            lifecycle_provider=self.lifecycle,power_provider=self.resources,memory_provider=self.resources,
            require_energy=True,require_memory=True,allow_independent_copy_hypothesis=True,
            allow_no_implicit_default_stream_hypothesis=True,max_events=max_events)

    def execute(self, plan_ref, candidate_id, public, routes, startup_model_ref, *, max_events=5_000_000):
        plan = checked(plan_ref)
        defaults = original_flow_defaults(plan,self.defaults)
        if plan['aliases'] != self.aliases or plan['weights'] != self.bundle['weights']:
            raise ValueError('Original owners and six equal weights must match trained components')
        rows = bind_public(plan,candidate_id,public,routes,defaults,plan['weights'])
        target_context = checked(plan['context'])
        target_native = native_execution_projection(target_context)
        if (target_native['original_full_context_sha256'] != plan['execution_context_sha256']
                or target_native['projection_sha256'] != self.source_native['projection_sha256']):
            raise ValueError('Actual target native execution differs from original trained components')
        generation = self.generation.check(rows)
        # Target verification rederives original startup labels and checks the
        # actual new collector, media, weights, source closure and full priming.
        checked(startup_model_ref)
        startup = startup_prediction(startup_model_ref['path'],plan_ref['path'],candidate_id,
                                      allow_native_priming_transfer_hypothesis=True)
        engine = self.engine(max_events=max_events)
        result = dict(schema='gsdeploy.prospective_availability_result.v11',configuration=self.configuration,
            plan=plan_ref,candidate_id=candidate_id,full_public_cohort=rows,
            execution_context_sha256=plan['execution_context_sha256'],
            original_component_target_context=self.source_target_context_ref,
            target_native_projection=target_native,original_native_projection=self.source_native,
            full_context_equality_claimed=False,collector_transfer_is_hypothesis=True,
            generation_work_support=generation,startup_prediction=startup,GPU_started=False,
            independent_prediction_accuracy_validated=False,physical_gain_certified=False,
            equal_context_fallback_used=False,global_request_concurrency_cap_used=False)
        try:
            forecast = engine.forecast(rows,public['now_ns'],public['deadline_ns'],
                                       drain_deadline_ns=public['drain_deadline_ns'])
            metrics = score_complete_forecast(forecast,rows,plan,self.lifecycle.quality,startup,
                                              phase_start_ns=public['phase_start_ns'])
            result.update(status='COMPLETE_UNVALIDATED_PROSPECTIVE_FORECAST',forecast=forecast,metrics=metrics)
        except UnknownEvidence as exc:
            result.update(status='UNKNOWN',reason=str(exc))
        result['scheduler_diagnostic'] = engine.scheduler.provenance()
        checked(plan_ref)
        for ref in (self.configuration['parameters'],self.configuration['availability_model'],
                    self.configuration['component_bundle'],startup_model_ref):
            checked(ref)
        return result
