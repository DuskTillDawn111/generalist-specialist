"""CPU-only, source-bound seed transfer without fabricated measured profiles.

The original resolver remains authoritative about its own evidence. This
adapter separately reproduces its rho arithmetic using matched training means.
It never turns a statistical transfer into physical capacity admission.
"""
from __future__ import annotations

import argparse
import collections
import copy
import math
from pathlib import Path
import random
import statistics
import time

from .flow_capacity_catalog import condition, verify, verify_catalog
from .flow_eligibility import _routing, original_rho, native_stability, RHO_MAX, RHO_TOLERANCE
from .probe_validation import read, receipt, save
from .profiling import digest

SCHEMA = 'gsdeploy.prospective_training_condition_capacity_transfer.v1'
GENERATION_SCHEMA = 'gsdeploy.periodic_seeded_uniform_input_contract.v1'


def _plan(path):
    plan = read(path)
    if (plan.get('schema') != 'gsdeploy.native_flow_plan.v1' or
            plan.get('plan_sha256') != digest({k: v for k, v in plan.items() if k != 'plan_sha256'})):
        raise ValueError('Native prospective plan self hash mismatch')
    if type(plan.get('seed')) is not int:
        raise ValueError('Explicit integer seed required')
    if plan.get('kind') == 'measured':
        raise ValueError('Prospective plan required, never a measured target')
    for key in ('runner_source', 'context', 'profile_plan', 'catalog'):
        verify(plan[key])
    if not plan.get('candidates') or len({c['id'] for c in plan['candidates']}) != len(plan['candidates']):
        raise ValueError('Nonempty unique declared candidates required')
    return plan


def _generation(plan, spec):
    """Validate the actual declared trace against its source-bound input law."""
    contract = plan.get('input_generation_contract')
    if contract is None:
        native_source = receipt(Path(__file__).with_name('native_flow.py'))
        if plan['runner_source'] != native_source:
            raise ValueError('Unsupported input generator requires explicit supported contract')
        contract = dict(schema=GENERATION_SCHEMA, task_order=['text', 'image', 'asr', 'ocr'],
            task_phase_fractions=dict(text=0., image=.19, asr=.47, ocr=.73),
            rng_namespace='full-flow-inputs:', request_id_prefix='measurement:')
    if (contract.get('schema') != GENERATION_SCHEMA or
            set(contract) != {'schema', 'task_order', 'task_phase_fractions', 'rng_namespace', 'request_id_prefix'} or
            len(contract['task_order']) != len(set(contract['task_order'])) or
            set(contract['task_order']) != set(plan['workload_rps']) or
            set(contract['task_phase_fractions']) != set(plan['workload_rps']) or
            any(type(x) not in (int, float) or not math.isfinite(x) or not 0 <= x < 1
                for x in contract['task_phase_fractions'].values()) or
            not isinstance(contract['rng_namespace'], str) or not isinstance(contract['request_id_prefix'], str)):
        raise ValueError('Unsupported periodic seeded-uniform input generation contract')
    rng = random.Random(contract['rng_namespace'] + str(plan['seed']))
    trace = []
    for task in contract['task_order']:
        rate = plan['workload_rps'][task]
        if type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Positive offered rates required')
        inputs = spec['known_input_domain'][task]
        offset = contract['task_phase_fractions'][task] / rate
        index = 0
        while offset + index / rate < plan['window_seconds']:
            trace.append(dict(offset_ns=round((offset + index / rate) * 1e9), task=task,
                pool_id=inputs[rng.randrange(len(inputs))]))
            index += 1
    trace.sort(key=lambda row: (row['offset_ns'], row['task']))
    for index, row in enumerate(trace):
        row['request_id'] = contract['request_id_prefix'] + str(index)
    if trace != plan['trace']:
        raise ValueError('Declared trace does not match frozen input generation contract and seed')
    return {'contract': copy.deepcopy(contract), 'runner_source': plan['runner_source'],
            'profile_plan': plan['profile_plan'], 'context': plan['context'], 'base_catalog': plan['catalog']}


def _training_conditions(manifest_path):
    manifest = read(manifest_path)
    if (manifest.get('schema') != 'gsdeploy.training_flow_capacity_catalog_manifest.v1' or
            manifest.get('heldout_observations_used') is not False or not manifest.get('conditions')):
        raise ValueError('Training-only capacity manifest required')
    for ref in manifest['source_receipts']:
        verify(ref)
    rows = []
    for item in manifest['conditions']:
        verify(item['catalog']); catalog = read(item['catalog']['path']); verify_catalog(catalog)
        audit = catalog['native_flow_capacity_audit']
        if digest(audit['condition']) != item['condition_sha256']:
            raise ValueError('Training condition identity mismatch')
        references = catalog['native_flow_capacity_binding']['source_receipts']
        if not references:
            raise ValueError('Training native evidence required')
        generations = []; evidence = []
        for ref in references:
            wrapper = read(ref['path']); verify(wrapper['plan']); plan = _plan(wrapper['plan']['path'])
            candidate = next(c for c in plan['candidates'] if c['id'] == wrapper['candidate_id'])
            if condition(plan, candidate) != audit['condition']:
                raise ValueError('Training wrapper full condition differs from capacity audit')
            generations.append(_generation(plan, audit['condition']))
            rho = original_rho(plan, candidate, catalog)
            if rho['status'] != 'ASSESSED':
                raise ValueError('Original training resolver/binding cannot reproduce service')
            expected = {m: {t: v['family_normalized_service_ms'] for t, v in tasks.items()}
                        for m, tasks in audit['service_estimates'].items()}
            if rho['service_time_ms'] != expected:
                raise ValueError('Original training service differs from verified audit')
            evidence.append({'plan': wrapper['plan'], 'original_rho': rho})
        if any(g != generations[0] for g in generations):
            raise ValueError('Training input-generation conditions differ')
        rows.append(dict(item=item, catalog=catalog, audit=audit, generation=generations[0], evidence=evidence))
    return rows


def _estimates(plan, conditions):
    results = []
    for candidate in plan['candidates']:
        routing = _routing(plan, candidate); spec = condition(plan, candidate)
        generation = _generation(plan, spec)
        matches = [r for r in conditions if r['audit']['condition'] == spec and r['generation'] == generation]
        if len(matches) != 1:
            raise ValueError('Exactly one full training condition must match; only seed/generated trace may change')
        matched = matches[0]; catalog = matched['catalog']; audit = matched['audit']
        if plan['seed'] in audit['training_seeds']:
            raise ValueError('Target seed must be disjoint from all source training seeds')
        original = original_rho(plan, candidate, catalog)
        if original.get('service_time_ms') is not None or original.get('rho_admission_passed') is not None:
            raise ValueError('New seed unexpectedly resolves a measured profile; no fabricated target keys allowed')
        deployment = dict(collections.Counter(v['family'] for v in plan['aliases'].values()))
        service = {m: {t: v['family_normalized_service_ms'] for t, v in tasks.items()}
                   for m, tasks in audit['service_estimates'].items()}
        capacity = {m: catalog['capacity_factors'][m][str(n)] for m, n in deployment.items()}
        demand = dict.fromkeys(deployment, 0.)
        for task, weights in routing.items():
            for family, rate in weights.items():
                if rate:
                    value = service.get(family, {}).get(task)
                    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                        raise ValueError('Missing positive supported training route service')
                    demand[family] += rate * value / 1000
        rho = {m: demand[m] / capacity[m] for m in deployment}
        threshold = max(rho.values()) <= RHO_MAX + RHO_TOLERANCE
        results.append(dict(candidate_id=candidate['id'], original_target_resolver=original,
            transfer=dict(status='PROSPECTIVE_TRANSFER_ESTIMATE', kind='prospective_training_condition_transfer',
                training_catalog=matched['item']['catalog'], condition_sha256=matched['item']['condition_sha256'],
                training_seeds=audit['training_seeds'], source_training_original_rho=matched['evidence'],
                input_generation=generation, service_estimates=copy.deepcopy(audit['service_estimates']),
                service_time_ms=service, deployment_by_family=deployment, routing_rps_by_family=routing,
                capacity_factors=capacity, rho_by_family=rho, max_rho=max(rho.values()),
                threshold=RHO_MAX, tolerance=RHO_TOLERANCE,
                formula='rho_m = sum_t planned_routing_rps[t,m] * training_family_normalized_service_ms[m,t] /1000 / original_capacity_factor[m,n_m]',
                raw_owned_sampled_peak_mib=audit['raw_owned_sampled_peak_mib'], safe_memory_mib=audit['safe_memory_mib'],
                memory_safety_factor_application_count=1, target_measured=False, target_outcomes_used=False,
                physical_capacity_bound=False, uncertainty_calibrated=False),
            transferred_rho_threshold_satisfied=threshold, provisional_transfer_planning_filter_passed=threshold,
            training_condition_match=True, target_applicability_validated=None, original_rho_admission_passed=None,
            original_capacity_and_stability_passed=None, final_gain_eligible=False))
    return results


def seal(target_plan_path, training_manifest_path, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    plan = _plan(target_plan_path); conditions = _training_conditions(training_manifest_path)
    candidates = _estimates(plan, conditions)
    import gsdeploy.performance, gsdeploy.routing, gsdeploy.stability
    sources = [receipt(__file__), receipt(target_plan_path), receipt(training_manifest_path),
        *[receipt(m.__file__) for m in (gsdeploy.performance, gsdeploy.routing, gsdeploy.stability)],
        *[receipt(Path(__file__).with_name(n)) for n in ('flow_capacity_catalog.py', 'flow_eligibility.py', 'profiling.py')]]
    sources += [plan[k] for k in ('runner_source', 'context', 'profile_plan', 'catalog')]
    sources += [c['item']['catalog'] for c in conditions]
    sources = list({(r['path'], r['sha256']): r for r in sources}.values())
    value = dict(schema=SCHEMA, kind='prospective_training_condition_transfer', created_ns=time.perf_counter_ns(),
        target_plan=receipt(target_plan_path), training_manifest=receipt(training_manifest_path), source_receipts=sources,
        target_seed=plan['seed'], target_trace_sha256=digest(plan['trace']), candidates=candidates, GPU_started=False,
        target_outcomes_used=False, target_measured=False, final_gain_eligible=False,
        acceptance_policy=None, target_applicability_validated=None,
        scope='Matched-condition seed transfer is a provisional planning estimate; original target resolver remains unchanged; no physical capacity or final gain admission')
    value['transfer_sha256'] = digest(value)
    save(output, value); verify_transfer(output); return value


def verify_transfer(path):
    sealed = read(path)
    if (sealed.get('schema') != SCHEMA or sealed.get('transfer_sha256') !=
            digest({k: v for k, v in sealed.items() if k != 'transfer_sha256'})):
        raise ValueError('Changed prospective capacity transfer artifact')
    for ref in sealed['source_receipts']:
        verify(ref)
    verify(sealed['target_plan']); verify(sealed['training_manifest'])
    plan = _plan(sealed['target_plan']['path'])
    if (sealed['target_seed'] != plan['seed'] or sealed['target_trace_sha256'] != digest(plan['trace']) or
            sealed['candidates'] != _estimates(plan, _training_conditions(sealed['training_manifest']['path']))):
        raise ValueError('Capacity transfer differs from independently recomputed training arithmetic')
    if (sealed.get('target_outcomes_used') is not False or sealed.get('target_measured') is not False or
            sealed.get('final_gain_eligible') is not False or sealed.get('target_applicability_validated') is not None or
            sealed.get('acceptance_policy') is not None or type(sealed.get('created_ns')) is not int):
        raise ValueError('Transfer evidence cannot claim validated or measured target admission')
    return True


def report(sealed_path, raw_paths, output):
    """Descriptive independent transfer check; no post-hoc acceptance threshold."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    verify_transfer(sealed_path); sealed = read(sealed_path); plan = _plan(sealed['target_plan']['path'])
    declared = {c['id']: c for c in plan['candidates']}
    predictions = {c['candidate_id']: c for c in sealed['candidates']}; seen = set(); results = []
    for path in raw_paths:
        raw = read(path); cid = raw['candidate_id']
        if cid not in declared or cid in seen:
            raise ValueError('Duplicate or foreign independent target candidate')
        seen.add(cid)
        if raw.get('plan') != sealed['target_plan']:
            raise ValueError('Actual target plan differs from frozen transfer plan')
        if raw.get('startup_begin_ns', 0) <= sealed['created_ns']:
            raise ValueError('Transfer must be frozen before actual target startup')
        candidate = declared[cid]; predicted = predictions[cid]['transfer']
        if (raw.get('resource_configuration') != candidate.get('resource_configuration') or
                raw.get('status') != 'completed' or raw.get('collection_complete') is not True or
                any(raw.get(k) for k in ('errors', 'runtime_errors', 'telemetry_errors', 'qualification_issues'))):
            raise ValueError('Complete error-free matched independent native target required')
        catalog = read(predicted['training_catalog']['path'])
        stability = native_stability(plan, candidate, raw, catalog)
        observed = collections.defaultdict(list)
        for row in raw['request_records']:
            start = row['observed_worker_start_time_ns']; response = row['worker_response']; end = response['end_perf_ns']
            if (response.get('success') is not True or response['start_perf_ns'] != start or
                    not row['scheduled_time_ns'] <= start < end <= row['completion_time_ns'] <= raw['observation_end_ns']):
                raise ValueError('Invalid independent native service clocks or inference failure')
            family = plan['aliases'][row['model']]['family']
            observed[(family, row['task'])].append((end - start) / 1e6)
        service = {}
        for (family, task), values in observed.items():
            n = predicted['deployment_by_family'][family]; g = predicted['capacity_factors'][family]
            expected = predicted['service_time_ms'][family][task] * n / g
            mean = statistics.mean(values)
            service.setdefault(family, {})[task] = dict(request_count=len(values),
                predicted_worker_service_ms=expected, observed_mean_worker_service_ms=mean,
                request_service_mare=statistics.mean(abs(v - expected) / v for v in values),
                run_mean_relative_error=abs(mean - expected) / mean)
        results.append(dict(candidate_id=cid, raw_source=receipt(path),
            transfer_prediction=predictions[cid], independent_service_comparison=service, **stability,
            target_applicability_validated=None, transferred_rho_admission_passed=None,
            original_capacity_and_stability_passed=None, final_gain_eligible=False))
    for cid in sorted(set(declared) - seen):
        results.append(dict(candidate_id=cid, status='NOT_RUN', target_applicability_validated=None, final_gain_eligible=False))
    value = dict(schema='gsdeploy.independent_capacity_transfer_validation.v1',
        status='ASSESSED' if len(seen) == len(declared) else 'INCOMPLETE', sealed_source=receipt(sealed_path),
        report_source=receipt(__file__), candidates=results, GPU_started=False, target_applicability_validated=None,
        acceptance_policy=None, final_gain_eligible=False,
        scope='Independent per-run descriptive service error and unchanged native stability; no predeclared transfer acceptance gate, physical capacity guarantee, or final gain qualification')
    save(output, value); return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='op', required=True)
    p = sub.add_parser('seal'); p.add_argument('--plan', required=True); p.add_argument('--training-manifest', required=True); p.add_argument('--output', required=True)
    p = sub.add_parser('verify'); p.add_argument('--sealed', required=True)
    p = sub.add_parser('report'); p.add_argument('--sealed', required=True); p.add_argument('--raw', action='append', required=True); p.add_argument('--output', required=True)
    a = parser.parse_args(argv)
    if a.op == 'seal':
        value = seal(a.plan, a.training_manifest, a.output); print(len(value['candidates']), 'prospective transfer estimates; GPU_started=False')
    elif a.op == 'verify':
        verify_transfer(a.sealed); print('VERIFIED; GPU_started=False')
    else:
        value = report(a.sealed, a.raw, a.output); print(value['status'], '; GPU_started=False')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
