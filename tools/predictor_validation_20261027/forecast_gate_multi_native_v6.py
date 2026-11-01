"""Original twelve-template CPU diagnostic with verified generation/startup.

Full arithmetic is training diagnostic only. This creates no prospective seal
and grants no physical accuracy or deployment qualification.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

from gsdeploy.forecast_errors import UnknownEvidence
from gsdeploy.forecasting.metrics import score_forecast
from gsdeploy.forecasting.workload import FutureWorkload
from gsdeploy.routing import Router
from .copy_memset_baseline_v5 import ExactSourceCopyMemsetHypothesisResolver
from .dag_queue_diagnostics_v6 import DagQueueDiagnostics
from .forecast_gate_v2 import INPUT_SCHEMA, MODEL_FIELDS_FORBIDDEN, _public_cohort
from .kernel_semantic_query_v5 import ExactSourceSemanticHypothesisResolver
from .kernel_two_capture_provisional_v5 import ExactSourceTwoCaptureProvisionalResolver
from .multi_native_composite_v5 import MultiNativeProvider, checked
from .native_collect import receipt, save
from .component_generation_support_v7 import ComponentGenerationSupport
from .startup_prediction_v6 import prediction as startup_prediction

BUNDLE_SCHEMA = 'gsdeploy.multi_native_forecast_gate_bundle.v6'


def verify_startup(bundle, public, adapter, rows):
    spec = bundle['startup']
    if (set(spec) != {'training_model', 'flow_plan', 'candidate_id', 'prediction'}
            or bundle.get('allow_native_priming_transfer_hypothesis') is not True):
        raise ValueError('Exact flow-bound startup inputs and transfer opt-in required')
    checked(spec['training_model']); plan = checked(spec['flow_plan'])
    result = startup_prediction(spec['training_model']['path'], spec['flow_plan']['path'],
        spec['candidate_id'], allow_native_priming_transfer_hypothesis=True)
    if checked(spec['prediction']) != result:
        raise ValueError('Stored startup prediction differs from full source reverification')
    if (result['execution_context_sha256'] != adapter.target_execution_context_sha256
            or result['flow_plan'] != spec['flow_plan'] or result['candidate_id'] != spec['candidate_id']
            or result['safety_factor_applied'] is not False or result['warmup_seconds'] != 0.
            or result['startup_amortization_seconds'] != 3600.
            or result['energy_includes_priming'] is not True):
        raise ValueError('Original target, full priming and unapplied startup safety factor required')
    bind_flow(plan, spec['candidate_id'], public, bundle['forecast_defaults'], bundle['weights'], rows)
    return result


def bind_flow(plan, candidate_id, public, defaults, weights, rows):
    candidates = [c for c in plan['candidates'] if c['id'] == candidate_id]
    if len(candidates) != 1:
        raise ValueError('Unique bound startup flow candidate required')
    candidate = candidates[0]; start = public['phase_start_ns']
    expected = {r['request_id']: dict(request_id=r['request_id'], task=r['task'], pool_id=r['pool_id'],
        alias=candidate['routes'][r['request_id']], arrival_ns=start + r['offset_ns']) for r in plan['trace']}
    actual = {r['request_id']: r for r in rows}
    if (len(expected) != len(plan['trace']) or len(actual) != len(rows) or actual != expected
            or set(candidate['routes']) != set(actual)):
        raise ValueError('Full public cohort, arrivals and fixed routes differ from startup-bound flow')
    for alias in defaults['aliases']:
        planned_order = [r['request_id'] for r in plan['trace']
                         if candidate['routes'][r['request_id']] == alias]
        if [r['request_id'] for r in rows if r['alias'] == alias] != planned_order:
            raise ValueError('Per-owner public queue order differs from startup-bound flow')
    workload = public['future_workload']
    pools = {task: sorted({r['pool_id'] for r in rows if r['task'] == task}) for task in workload['rates']}
    if (public['now_ns'] != start or workload['process'] != 'declared_schedule'
            or workload['declared_arrivals'] or workload['rates'] != plan['workload_rps']
            or workload['input_pools'] != pools
            or public['deadline_ns'] - start != round(plan['window_seconds'] * 1e9)
            or public['drain_deadline_ns'] - public['deadline_ns'] != round(plan['drain_seconds'] * 1e9)
            or defaults['route_initial_with_continuation_policy'] is not False):
        raise ValueError('Complete measurement/drain window and fixed training cohort required')
    for name in ('aliases', 'latency_reference_ms', 'memory_budget_mib',
                 'energy_reference_watts', 'memory_safe_factor'):
        if defaults[name] != plan[name]:
            raise ValueError('Startup flow scoring/defaults differ: ' + name)
    if weights != plan['weights'] or plan['warmup_seconds'] != 0. or candidate['allow_spill']:
        raise ValueError('Original no-warmup fixed-route scoring contract required')


def diagnostic_score(forecast, adapter, workload, defaults, bundle, public, startup):
    horizon = (public['deadline_ns'] - public['phase_start_ns']) / 1e9
    amortized = startup['energy_joules'] * horizon / startup['startup_amortization_seconds']
    metrics = score_forecast(forecast, [], defaults['aliases'], workload.rates,
        defaults['latency_reference_ms'], adapter.quality,
        phase_start_ns=public['phase_start_ns'], now_ns=public['now_ns'], deadline_ns=public['deadline_ns'],
        memory_budget_mib=defaults['memory_budget_mib'], energy_reference_watts=defaults['energy_reference_watts'],
        memory_safe_factor=defaults['memory_safe_factor'], weights=bundle['weights'],
        observed_peak_memory_mib=startup['peak_memory_mib'], observed_board_energy_joules=amortized)
    if metrics['full_U'] is None or not math.isfinite(metrics['full_U']):
        raise UnknownEvidence('COMPLETE_FINITE_TRAINING_DIAGNOSTIC_UTILITY_REQUIRED')
    metrics.update(E_scope='measurement board energy plus startup_J * measurement_seconds / 3600; drain energy excluded',
        M_scope='safety factor once times maximum of verified startup and complete active/drain peak',
        startup_energy_amortized_to_window_joules=amortized,
        physical_six_term_improvement_certified=False)
    return metrics


def execute_bundle(bundle, public, *, allow_trained_hypothesis=False,
                   allow_training_diagnostic=False, allow_cpu_fixture=False):
    if (bundle.get('schema') != BUNDLE_SCHEMA or bundle.get('kind') != 'trained_hypothesis'
            or not allow_trained_hypothesis):
        raise ValueError('Explicit multi-native trained-hypothesis bundle opt-in required')
    allowed = {'schema', 'kind', 'composite', 'semantic_fit', 'copy_memset_fit',
               'two_capture_provisional_fit', 'allow_two_capture_provisional_hypothesis',
               'allow_numeric_semantic_hypothesis', 'allow_copy_memset_numeric_hypothesis',
               'host_supply_scenario', 'forecast_defaults', 'dag_hypothesis_options',
               'weights', 'training_template_diagnostic', 'generation', 'natural_resources',
               'allow_conditional_resource_hypothesis', 'allow_exact_content_generation_hypothesis',
               'startup', 'allow_native_priming_transfer_hypothesis', 'host_supply'}
    if set(bundle) - allowed:
        raise ValueError('Unsupported multi-native model fields')
    if (bundle.get('allow_numeric_semantic_hypothesis') is not True
            or bundle.get('allow_copy_memset_numeric_hypothesis') is not True):
        raise ValueError('Explicit numeric kernel and copy/memset hypotheses required')
    defaults = bundle['forecast_defaults']
    allowed_defaults = {'aliases', 'continuation_routing_rps', 'resources', 'execution_contract',
        'service_scenario', 'route_initial_with_continuation_policy', 'max_events',
        'require_energy', 'require_memory', 'latency_reference_ms', 'memory_budget_mib',
        'energy_reference_watts', 'memory_safe_factor'}
    if (set(defaults) - allowed_defaults or MODEL_FIELDS_FORBIDDEN.intersection(defaults)
            or defaults.get('execution_contract') != 'strict_lifecycle_v2'):
        raise ValueError('Only explicit strict-lifecycle multi-native model defaults accepted')
    if defaults.get('resources') is not None:
        raise UnknownEvidence('MULTI_NATIVE_DAG_REQUEST_ADMISSION_NOT_IMPLEMENTED')
    if bundle['weights'] != {key: 1 / 6 for key in 'QTLPME'}:
        raise ValueError('Original six equal utility weights required')
    _public_cohort(dict(schema=INPUT_SCHEMA, forecast_inputs=public))
    aliases = defaults['aliases']
    adapter = MultiNativeProvider(checked(bundle['composite']), aliases,
        host_scenario=bundle['host_supply_scenario'], allow_cpu_fixture=allow_cpu_fixture)
    workload = FutureWorkload(**public['future_workload'])
    suffix = workload.suffix(public['phase_start_ns'], public['now_ns'], public['deadline_ns'])
    router = Router(defaults['continuation_routing_rps'])
    rows = [copy.deepcopy(row) for queue in public['queues'].values() for row in queue]
    if defaults.get('route_initial_with_continuation_policy', True):
        for row in sorted(rows, key=lambda r: (r['arrival_ns'], r['task'], r['request_id'])):
            if router.choose(row['task']) != row['alias']:
                raise ValueError('Initial route differs from declared continuation policy')
    rows.extend(dict(row, alias=router.choose(row['task'])) for row in suffix['arrivals'])
    for row in rows:
        adapter.component(row)  # Exact-input support, including lifecycle/host dispatch.
    diagnostic = bundle.get('training_template_diagnostic', False)
    if type(diagnostic) is not bool:
        raise ValueError('Explicit boolean diagnostic scope required')
    if (not diagnostic or not allow_training_diagnostic or suffix['arrivals']
            or len(rows) != len(adapter.plans)
            or {(r['alias'], r['task'], r['pool_id']) for r in rows} != set(adapter.plans)):
        raise ValueError('Diagnostic requires explicit opt-in and every exact training template once')
    if 'generation' not in bundle or set(bundle['generation']) != {'generation_work', 'component_support'}:
        raise UnknownEvidence('COMPLETE_FOUR_TASK_COMPONENT_GENERATION_SUPPORT_REQUIRED')
    support = checked(bundle['generation']['component_support'])
    if support['generation_model'] != bundle['generation']['generation_work'] or support['composite'] != bundle['composite']:
        raise ValueError('Component support must bind exact generation model and composite receipts')
    generation_provider = ComponentGenerationSupport(support, adapter,
        allow_exact_content_generation_hypothesis=bundle.get('allow_exact_content_generation_hypothesis') is True)
    generation = generation_provider.check(rows)
    startup = verify_startup(bundle, public, adapter, rows)
    release = adapter.release
    if 'host_supply' in bundle:
        from .dag_host_release_v7 import DagHostReleaseProvider, SCHEMA as HOST_SCHEMA
        host_spec = bundle['host_supply']
        if (set(host_spec) != {'model', 'allow_identification_hypothesis', 'allow_exact_input_conditioning_hypothesis'}
                or host_spec['allow_identification_hypothesis'] is not True
                or host_spec['allow_exact_input_conditioning_hypothesis'] is not True):
            raise ValueError('Explicit exact-input host identification hypotheses required')
        document = checked(host_spec['model'])
        if document.get('schema') != HOST_SCHEMA:
            raise ValueError('Exact v7 host schema dispatch required')
        release = DagHostReleaseProvider(document, adapter.target_execution_context_sha256,
            adapter.plans, aliases, scenario=bundle['host_supply_scenario'],
            allow_identification_hypothesis=True, allow_exact_input_conditioning_hypothesis=True,
            public_bindings=generation_provider.public_bindings)
    baseline = ExactSourceSemanticHypothesisResolver(bundle['semantic_fit']['path'])
    if baseline.fit_source != bundle['semantic_fit']:
        raise ValueError('Changed semantic fit receipt')
    if 'two_capture_provisional_fit' in bundle:
        if bundle.get('allow_two_capture_provisional_hypothesis') is not True:
            raise ValueError('Separate two-capture provisional opt-in required')
        ref = bundle['two_capture_provisional_fit']
        checked(ref)
        baseline = ExactSourceTwoCaptureProvisionalResolver(ref['path'],
            allow_two_capture_provisional_hypothesis=True, base_resolver=baseline)
    copy_provider = ExactSourceCopyMemsetHypothesisResolver(bundle['copy_memset_fit']['path'])
    if copy_provider.fit_source != bundle['copy_memset_fit']:
        raise ValueError('Changed copy/memset fit receipt')
    coverage = {kind: dict(queried_nodes=0, strict_qualified_nodes=0, numeric_hypothesis_nodes=0,
        unavailable_nodes=0, strict_failure_reason_counts={}, strict_fit_gate_changed=False,
        deployment_qualified=False,
        coverage_denominator='unique_used_template_nodes_cached_for_all_requests') for kind in ('gpu', 'copy_memset')}

    def duration(node, row):
        if node['kind'] == 'gpu':
            result = baseline(node['kernel']); counts = coverage['gpu']
        else:
            plan = adapter.plans[row['alias'], row['task'], row['pool_id']]
            result = copy_provider(node, plan['source_sqlite']); counts = coverage['copy_memset']
        counts['queried_nodes'] += 1
        if result.get('hypothesis_only') is not True or result.get('deployment_qualified') is not False:
            raise ValueError('Numerical duration cannot imply physical qualification')
        counts['strict_qualified_nodes'] += result.get('strict_known') is True
        for reason in result.get('strict_failure_reasons', []):
            reasons = counts['strict_failure_reason_counts']; reasons[reason] = reasons.get(reason, 0) + 1
        if result.get('numeric_hypothesis_available') is not True:
            counts['unavailable_nodes'] += 1
            raise UnknownEvidence('MULTI_NATIVE_NUMERIC_NODE_DURATION_UNAVAILABLE:' + node['id'])
        counts['numeric_hypothesis_nodes'] += 1
        return result['duration_ns']

    resource = None
    if 'natural_resources' in bundle:
        if bundle.get('allow_conditional_resource_hypothesis') is not True:
            raise ValueError('Conditional resource hypothesis opt-in required')
        from .natural_resources_v6 import NaturalResourceProvider
        contract = checked(bundle['composite'])
        resource = NaturalResourceProvider(checked(bundle['natural_resources']),
            checked(contract['target']['context']), allow_conditional_resource_hypothesis=True)
    options = bundle.get('dag_hypothesis_options', {})
    allowed_options = {'allow_independent_copy_hypothesis', 'allow_context_time_slice_hypothesis',
                       'allow_no_implicit_default_stream_hypothesis'}
    if set(options) - allowed_options or any(type(v) is not bool for v in options.values()):
        raise ValueError('Only explicit DAG scheduling hypothesis flags accepted')
    from .cuda_dag_simulator_v6 import DagFullHorizonModel
    diagnostics = DagQueueDiagnostics(aliases, adapter.plans, duration, now_ns=public['now_ns'])
    forecast = DagFullHorizonModel(aliases, adapter.plans, lifecycle_provider=adapter,
        duration_provider=diagnostics.duration, release_provider=release, queue_diagnostics=diagnostics,
        power_provider=resource, memory_provider=resource,
        require_energy=defaults.get('require_energy', False), require_memory=defaults.get('require_memory', False),
        scenario=defaults.get('service_scenario', 'profiled'), max_events=defaults.get('max_events', 5_000_000),
        **options).forecast(rows, public['now_ns'], public['deadline_ns'],
                           drain_deadline_ns=public['drain_deadline_ns'])
    if set(forecast['cohort']) != {r['request_id'] for r in rows}:
        raise ValueError('Multi-native engine dropped declared public cohort')
    metrics = diagnostic_score(forecast, adapter, workload, defaults, bundle, public, startup)
    return dict(schema='gsdeploy.multi_native_dag_forecast_result.v6', forecast=forecast,
        training_diagnostic_metrics=metrics, training_diagnostic_utility=metrics['full_U'],
        deployment_utility=None, startup_prediction_verified=True, startup_prediction=startup,
        target_execution_context_sha256=adapter.target_execution_context_sha256,
        model_composite_execution_sha256=adapter.execution_context_sha256,
        request_source_bindings={r['request_id']: adapter.provenance(r) for r in rows},
        baseline_coverage=coverage, generation_work_support=generation,
        training_template_diagnostic_only=True, prospective_prediction_seal_created=False,
        independent_validation_performed=False, startup_physically_qualified=False,
        unseen_input_generalization_claimed=False, physical_accuracy_validated=False,
        deployment_qualified=False, measured_target_fallback_used=False, GPU_started=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'public-inputs', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--allow-trained-hypothesis', action='store_true', required=True)
    parser.add_argument('--allow-training-diagnostic', action='store_true')
    args = parser.parse_args()
    bundle_ref, public_ref = receipt(args.bundle), receipt(args.public_inputs)
    bundle = checked(bundle_ref); public, _ = _public_cohort(checked(public_ref))
    result = execute_bundle(bundle, public, allow_trained_hypothesis=args.allow_trained_hypothesis,
                            allow_training_diagnostic=args.allow_training_diagnostic)
    if receipt(args.bundle) != bundle_ref or receipt(args.public_inputs) != public_ref:
        raise ValueError('Multi-native inputs changed during CPU forecast')
    save(Path(args.output), dict(result, inputs=dict(bundle=bundle_ref, public=public_ref)))


if __name__ == '__main__':
    main()
