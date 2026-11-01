"""Opt-in profiler-lifetime response clocks in the existing horizon engine.

Static kernel covariates select a fitted statistical clock rate. They never
become an occupancy-to-throughput rule. Runtime collection is not changed.
"""
from __future__ import annotations

import copy
import math

from scipy.special import ndtri

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.resources import identity
from . import kernel_competition_features_v2 as features
from .implementation_source_closure_v27 import implementation_sources

SCHEMA = 'gsdeploy.kernel_competition_response_clock.v1'
DOMAIN = 'CUPTI_PROFILED_KERNEL_LIFETIME_NOT_SM_SERVICE'


def need(value, message):
    if not value:
        raise ValueError(message)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def training_bundle(kernel_provider, captures, *, allow_cpu_fixture=False):
    """Reuse the unchanged v2 training compiler and response-likelihood fit.

    Caller declarations do not gain native authentication here. A failed fit
    is preserved in the bundle and cannot produce a prospective service law.
    One profiler domain is required; no profiled/natural pooling is performed.
    """
    captures = list(captures)
    prepared = features.KernelCompetitionFeatureProvider(kernel_provider).training_rows(
        captures, allow_cpu_fixture=allow_cpu_fixture)
    profilers = {c['profiler_id'] for c in captures}
    need(len(profilers) == 1, 'One declared profiler response domain required')
    info, _, _, _ = features.fit_response(prepared)
    document = dict(schema=SCHEMA, split='training', response_domain=DOMAIN,
        execution_context_sha256=kernel_provider.execution_context_sha256,
        kernel_catalog_sha256=identity(kernel_provider.document),
        hardware_id=kernel_provider.hardware['hardware_id'], profiler_id=next(iter(profilers)),
        evidence_kind='cpu_fixture' if any(c['evidence_kind'] == 'cpu_fixture' for c in captures) else 'declared_measured',
        feature_schema=features.SCHEMA, features=list(features.FEATURES), response_fit=info,
        capture_identities=[dict(capture_id=c['capture_id'], sha256=identity(c)) for c in captures],
        offered_requests=prepared['offered_requests'], offered_kernel_count=prepared['offered_kernel_count'],
        accepted_kernel_count=prepared['accepted_kernel_count'],
        sources=implementation_sources(__file__), native_sources_authenticated=False,
        natural_timing_validated=False, independent_prediction_accuracy_evaluated=False,
        effective_SM_service_identified=False, joint_request_distribution_identified=False,
        physically_validated=False, deployment_qualified=False, GPU_started=False,
        validation_labels_read=False)
    document['bundle_sha256'] = identity(document)
    return document


class KernelCompetitionResponseProvider:
    """Thin FullHorizonModel adapter over catalog plans and v2 covariates.

    For each generated active set, d(reference_ms)/d(wall_ms)=exp(-beta*x).
    The original event engine conserves remaining reference work and reprices
    the rate at generated completions, segment changes and future arrivals.
    A scenario fixes a reference functional for each kernel; it is not a fitted
    joint request draw or the expected end-to-end response of a random system.
    """
    def __init__(self, kernel_provider, bundle, *, scenarios=None,
                 allow_unvalidated_profiled_response=False, allow_cpu_fixture=False,
                 non_gpu_provider=None):
        need(type(allow_unvalidated_profiled_response) is bool and type(allow_cpu_fixture) is bool,
             'Explicit boolean response/fixture opt-ins required')
        doc = copy.deepcopy(bundle)
        seal = doc.pop('bundle_sha256', None)
        need(seal == identity(doc), 'Response bundle identity changed')
        need(doc.get('schema') == SCHEMA and doc.get('split') == 'training'
             and doc.get('response_domain') == DOMAIN and doc.get('feature_schema') == features.SCHEMA
             and doc.get('features') == list(features.FEATURES)
             and doc.get('validation_labels_read') is False, 'Exact training kernel response bundle required')
        need(doc.get('kernel_catalog_sha256') == identity(kernel_provider.document)
             and doc.get('execution_context_sha256') == kernel_provider.execution_context_sha256
             and doc.get('hardware_id') == kernel_provider.hardware.get('hardware_id'),
             'Response fit catalog/context/hardware identity differs')
        need(doc.get('evidence_kind') in ('declared_measured', 'cpu_fixture')
             and (doc['evidence_kind'] != 'cpu_fixture' or allow_cpu_fixture),
             'CPU response fixture requires explicit opt-in')
        need(doc.get('sources') == implementation_sources(__file__),
             'Kernel response implementation source changed')
        need(type(doc.get('profiler_id')) is str and doc['profiler_id'], 'Explicit fitted profiler domain required')
        if non_gpu_provider is not None:
            need(getattr(non_gpu_provider, 'execution_context_sha256', None) == kernel_provider.execution_context_sha256,
                 'Non-GPU response execution identity differs')
        # Snapshot the catalog once. Forecast boundaries must not hash/copy a
        # whole tens-of-thousands-kernel plan for every active kernel event.
        self.catalog = copy.deepcopy(kernel_provider)
        self.catalog.power_provider = kernel_provider.power_provider
        self.catalog.memory_provider = kernel_provider.memory_provider
        self.catalog_sha256 = doc['kernel_catalog_sha256']
        self.features = features.KernelCompetitionFeatureProvider(self.catalog)
        self.execution_context_sha256 = kernel_provider.execution_context_sha256
        self.bundle_sha256 = seal
        self.profiler_id = doc['profiler_id']
        self.evidence_kind = doc['evidence_kind']
        self.allow_response = allow_unvalidated_profiled_response
        self.non_gpu_provider = non_gpu_provider
        self.fit = doc['response_fit']
        self.qualified = self.fit.get('numerical_qualification') is True
        self.scenarios = copy.deepcopy(scenarios if scenarios is not None else {
            'mean': dict(catalog_scenario='mean', reference_functional='mean')})
        need(type(self.scenarios) is dict and self.scenarios, 'Predeclared reference-clock scenarios required')
        for name, scenario in self.scenarios.items():
            need(type(name) is str and name and type(scenario) is dict
                 and type(scenario.get('catalog_scenario')) is str and scenario['catalog_scenario'],
                 'Named catalog/reference-clock scenario required')
            if scenario.get('reference_functional') == 'mean':
                need(set(scenario) == {'catalog_scenario', 'reference_functional'}, 'Closed mean-clock scenario required')
            else:
                need(set(scenario) == {'catalog_scenario', 'reference_functional', 'quantile'}
                     and scenario['reference_functional'] == 'quantile'
                     and finite(scenario['quantile']) and 0 < scenario['quantile'] < 1,
                     'Explicit reference-clock quantile scenario required')
        self.baselines = {}
        self.beta = None
        if self.qualified:
            need(self.fit.get('features') == list(features.FEATURES)
                 and self.fit.get('feature_schema') == features.SCHEMA
                 and self.fit.get('response_domain') == DOMAIN,
                 'Fitted clock feature/response domain differs')
            beta, sigma = self.fit.get('coefficients'), self.fit.get('lognormal_reference_sigma')
            need(type(beta) is list and len(beta) == len(features.FEATURES) and all(finite(b) for b in beta)
                 and finite(sigma) and sigma > 0, 'Finite complete fitted coefficients and positive sigma required')
            self.beta, self.sigma = beta, sigma
            for entry in self.fit.get('baselines', []):
                key = tuple(entry['group'])
                need(len(key) == 5 and all(type(k) is str and k for k in key)
                     and key[-2:] == (self.execution_context_sha256, self.profiler_id)
                     and key not in self.baselines and finite(entry['log_reference_ms']),
                     'Unique exact owner/profile/segment/context/profiler baseline required')
                self.baselines[key] = entry['log_reference_ms']
            need(self.baselines, 'Complete fitted kernel reference baselines required')

    @staticmethod
    def _unknown(reason):
        return dict(known=False, context_supported=False, reason=reason,
            physically_validated=False, deployment_qualified=False)

    def _scenario(self, scenario):
        if scenario not in self.scenarios:
            raise UnknownEvidence('UNKNOWN_KERNEL_REFERENCE_SCENARIO:' + scenario)
        return self.scenarios[scenario]

    def _reference(self, alias, profile_id, segment_id, scenario):
        if not self.allow_response:
            raise UnknownEvidence('UNVALIDATED_PROFILED_RESPONSE_OPT_IN_REQUIRED')
        if not self.qualified:
            raise UnknownEvidence('NUMERICALLY_QUALIFIED_KERNEL_RESPONSE_FIT_REQUIRED')
        key = alias, profile_id, segment_id, self.execution_context_sha256, self.profiler_id
        if key not in self.baselines:
            raise UnknownEvidence('UNKNOWN_OWNER_PROFILE_KERNEL_RESPONSE_BASELINE:' + repr(key))
        spec = self._scenario(scenario)
        try:
            shift = self.sigma**2 / 2 if spec['reference_functional'] == 'mean' else self.sigma * float(ndtri(spec['quantile']))
            log_ms = self.baselines[key] + shift
            ms = math.exp(log_ms)
        except OverflowError as exc:
            raise UnknownEvidence('NONFINITE_FITTED_KERNEL_REFERENCE') from exc
        if not math.isfinite(ms) or ms <= 0:
            raise UnknownEvidence('NONFINITE_FITTED_KERNEL_REFERENCE')
        return ms

    def plan(self, row, aliases, scenario):
        need(identity(self.catalog.document) == self.catalog_sha256
             and self.catalog.execution_context_sha256 == self.execution_context_sha256,
             'Kernel response catalog/context changed during prediction')
        spec = self._scenario(scenario)
        plan = self.catalog.plan(row, aliases, spec['catalog_scenario'])
        for segment in plan['segments']:
            if segment['kind'] == 'gpu':
                segment['duration_ms'] = self._reference(row['alias'], plan['profile_id'], segment['id'], scenario)
            elif self.non_gpu_provider is None:
                raise UnknownEvidence('EXPLICIT_NON_GPU_KERNEL_GAP_RESPONSE_PROVIDER_REQUIRED')
        return plan

    def service(self, row, active, aliases, scenario):
        # Validate the generated response-clock plan, then adapt only its
        # duration field back to the immutable catalog for the unchanged v2
        # static feature gate. The fitted clock duration is not a solo label.
        need(row['alias'] in active and row == active[row['alias']], 'Target must belong to generated active set')
        static_active = {}
        spec = self._scenario(scenario)
        for alias, current in active.items():
            source = self.catalog.segments.get((current.get('profile_id'), current.get('segment_id')))
            need(source is not None and current.get('segment_kind') == source['kind'],
                 'Active fitted response-clock segment differs from its generated plan')
            if source['kind'] == 'gpu':
                expected_ms = self._reference(alias, current['profile_id'], current['segment_id'], scenario)
            else:
                if self.non_gpu_provider is None:
                    return self._unknown('EXPLICIT_NON_GPU_KERNEL_GAP_RESPONSE_PROVIDER_REQUIRED')
                expected_ms = source['duration_ms'].get(spec['catalog_scenario'])
            need(current.get('segment_duration_ms') == expected_ms,
                 'Active fitted response-clock baseline differs from its generated plan')
            static_active[alias] = dict(current, segment_duration_ms=source['duration_ms'][spec['catalog_scenario']])
        if row['segment_kind'] != 'gpu':
            self.catalog._active_segments(static_active, aliases, spec['catalog_scenario'])
            return self.non_gpu_provider.service(static_active[row['alias']], static_active,
                                                 aliases, spec['catalog_scenario'])
        try:
            state = self.features.features(static_active[row['alias']], static_active, aliases, spec['catalog_scenario'])
            if not state['known']:
                return self._unknown(state['reason'])
            reference = self._reference(row['alias'], row['profile_id'], row['segment_id'], scenario)
            linear = sum(state['vector'][name] * b for name, b in zip(features.FEATURES, self.beta))
            ms = math.exp(math.log(reference) + linear)
            if not math.isfinite(ms) or ms <= 0:
                return self._unknown('NONFINITE_KERNEL_RESPONSE_CLOCK')
        except OverflowError:
            return self._unknown('NONFINITE_KERNEL_RESPONSE_CLOCK')
        return dict(known=True, context_supported=True, service_ms=ms,
            response_domain=DOMAIN, reference_clock_ms=reference, covariates=state,
            reference_functional=copy.deepcopy(spec), response_bundle_sha256=self.bundle_sha256,
            generated_active_kernel_set=True, target_future_END_read=False,
            static_occupancy_used_as_throughput=False, native_sources_authenticated=False,
            joint_request_distribution_identified=False, natural_timing_validated=False,
            physically_validated=False, deployment_qualified=False)

    def power(self, *args): return self.catalog.power(*args)
    def memory(self, *args): return self.catalog.memory(*args)
    def power_preparing(self, *args): return self.catalog.power_preparing(*args)
    def memory_preparing(self, *args): return self.catalog.memory_preparing(*args)
    def demand_summary(self, profile_id): return self.catalog.demand_summary(profile_id)
