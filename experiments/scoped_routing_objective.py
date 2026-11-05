"""Fixed-residency routing objective using the sealed aggregate predictor.

This adapter reuses all six numerical references. Its experiment binding is
new and explicit: it is not the old 1800-second reference workload. No model
fit, GPU collection, or reference measurement is performed here.
"""
from collections import defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.audit_activation_aggregate_capture import baseline_outputs
from experiments.fit_activation_native import load_bundle
from experiments.replay_activation_aggregate import predict_plan
from gsdeploy import normalization as norm
from gsdeploy.prediction import components
from gsdeploy.routing import Router


def file_receipt(path):
    path = Path(path).resolve()
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def prepare_evidence(solo_base, training_base, reference):
    """Only training captures supply output replay, capacity and memory estimates."""
    system = json.loads(Path(reference).read_text())
    bundles = [load_bundle(Path(solo_base).resolve()/f'solo-round{i}') for i in (1, 2)]
    outputs = baseline_outputs(bundles, system['catalog'])
    services = defaultdict(list)
    for b in bundles:
        for row in b['raw']['request_records']:
            family = b['context']['resident_aliases'][row['model']]
            services[family, row['task']].append(row['worker_response']['worker_latency_ms'])
    audits = [Path(training_base)/f'capture-audit-v1/aggregate-training-round{i}.json' for i in (1, 2)]
    peaks = []
    for path in audits:
        run = json.loads(path.read_text())['runs'][0]
        if run['issues'] or not run['quality_cohort_complete'] or not run['energy']['coverage_valid']:
            raise ValueError('Clean completed training audit required')
        peaks.append(run['observed_owned_peak_mib'])
    return dict(schema='gsdeploy.scoped_routing_evidence.v1',
        execution_context_sha256=bundles[0]['plan']['execution_context_sha256'],
        resident_aliases=bundles[0]['context']['resident_aliases'],
        quality=[dict(family=f, task=t, pool_id=i, **v) for (f, t, i), v in sorted(outputs.items())],
        capacity_service_ms={f: {t: statistics.mean(v) for (ff, t), v in services.items() if ff == f}
                             for f in sorted({f for f, _ in services})},
        capacity_contract='Original family-normalized rho=sum(route_rps*solo_mean_handler_seconds)/g(1); g(1)=1; not measured dynamic utilization',
        raw_peak_estimate_mib=max(peaks),
        memory_contract='Common maximum observed whole-episode training peak at fixed residency; estimate, not a proven bound; confirm candidate peaks in measured comparison',
        sources=[file_receipt(reference), *[file_receipt(p) for p in audits]],
        solo_sources=[b['source'] for b in bundles],
        independent_memory_accuracy_qualified=False)


def scoped_reference(system, plan):
    """Preserve numerical anchors, disclose reuse and rebind experiment metadata."""
    if len(plan['jobs']) != 1:
        raise ValueError('Each candidate must have its own complete one-job episode')
    job = plan['jobs'][0]
    horizon = job['duration_ms']/1000.
    counts = {t: sum(r['task'] == t for r in job['requests']) for t in system['workload']}
    if not horizon > 0 or any(n <= 0 for n in counts.values()):
        raise ValueError('Positive complete task cohort required')
    rates = {t: n/horizon for t, n in counts.items()}
    data = copy.deepcopy(system['catalog'])
    previous = copy.deepcopy(norm.contract(data))
    binding = data[norm.KEY]
    binding.update(workload_rps=rates, horizon_seconds=horizon,
        episode_protocol=dict(scope='scoped_aggregate_launch_prime_one_job_drain_tail_v1',
                              boundary='startup_begin_through_observation_end',
                              execution_context_sha256=plan['execution_context_sha256']))
    binding['provenance']['scoped_reference_reuse'] = dict(
        original_reference_sha256=norm.identity(system['catalog']),
        original_workload_rps=previous['workload_rps'], original_horizon_seconds=previous['horizon_seconds'],
        original_episode_protocol=previous['episode_protocol'], numerical_anchors_unchanged=True,
        new_reference_measurement=False,
        interpretation='Fixed historical energy scale and quality/deadline anchors; not a matched-load energy baseline or percent energy saving')
    norm.validate(data, rates, system['latency_reference_ms'], horizon, system['gpu_memory_mib'])
    return data, rates


def candidate_plan(template, routing, rates):
    if len(template['jobs']) != 1 or set(routing) != set(rates):
        raise ValueError('One episode and exactly the declared tasks required')
    plan = copy.deepcopy(template)
    families = plan['aliases']
    aliases = {family: alias for alias, family in families.items()}
    if len(aliases) != len(families):
        raise ValueError('Only one replica per family is supported')
    for task, route in routing.items():
        if (any(m not in aliases or m not in (task, 'generalist_2')
                or not math.isfinite(q) or q < 0 for m, q in route.items())
                or not math.isclose(sum(route.values()), rates[task], rel_tol=1e-10)):
            raise ValueError('Unsupported or nonconserving routing')
    router = Router(routing)
    for request in sorted(plan['jobs'][0]['requests'], key=lambda r: r['offset_ms']):
        request['alias'] = aliases[router.choose(request['task'])]
    plan['jobs'][0]['design'].pop('generalist_share', None)
    plan['jobs'][0]['design']['routing_rps'] = copy.deepcopy(routing)
    plan['routing_assignment'] = 'original smooth weighted round robin; discrete finite-cohort realization'
    return plan


def predict_candidate(fit, template, routing, evidence, system):
    if (fit['profiles']['execution_context_sha256'] != evidence['execution_context_sha256']
            or template['execution_context_sha256'] != evidence['execution_context_sha256']
            or template['aliases'] != evidence['resident_aliases']):
        raise ValueError('Evidence/runtime/residency mismatch')
    data, rates = scoped_reference(system, template)
    plan = candidate_plan(template, routing, rates)
    prediction = predict_plan(fit, plan)
    if not prediction['complete'] or prediction['predicted_episode_board_joules'] is None:
        return dict(U=None, constraints_eligible=None, performance_unknown=True,
                    reason='INCOMPLETE_SCOPED_PREDICTION', prediction=prediction)
    rows = prediction['jobs'][0]['requests']
    quality = {(r['family'], r['task'], r['pool_id']): r['quality_totals'] for r in evidence['quality']}
    stats, native_quality, loss = {}, {}, {}
    for task in rates:
        task_rows = [r for r in rows if r['task'] == task]
        latencies = [r['predicted_e2e_ms'] for r in task_rows]
        totals = [quality[plan['aliases'][r['alias']], task, r['pool_id']] for r in task_rows]
        numerator, denominator = (sum(v[k] for v in totals) for k in (0, 1))
        if denominator <= 0:
            raise ValueError('Positive quality reference denominator required')
        native_quality[task] = numerator/denominator
        loss[task] = norm.quality_loss(data, task, native_quality[task])
        stats[task] = dict(mean_ms=float(np.mean(latencies)), p95_ms=float(np.percentile(latencies, 95)),
                           request_count=len(task_rows))
    Q = sum(rates[t]*loss[t] for t in rates)/sum(rates.values())
    T = norm.deadline_loss([(r['task'], r['predicted_e2e_ms'], True) for r in rows], system['latency_reference_ms'])
    episode = dict(kind='predicted_episode', scope='whole_gpu_board', episode_coverage_valid=True,
        episode_protocol=norm.contract(data)['episode_protocol'], episode_joules=prediction['predicted_episode_board_joules'])
    E = norm.episode_cost(data, episode)
    peak = evidence['raw_peak_estimate_mib']
    if not math.isfinite(peak) or peak <= 0:
        raise ValueError('Positive measured memory estimate required')
    safe_memory = peak*data['safe_factor']
    rho = {m: sum(routing[t].get(m, 0)*ms/1000 for t, ms in services.items())
           for m, services in evidence['capacity_service_ms'].items()}
    eligible = safe_memory <= system['gpu_memory_mib']+1e-6 and max(rho.values()) <= .8+1e-8
    terms = components(stats, rates, system['latency_reference_ms'], T, peak/system['gpu_memory_mib'],
                       system['weights'], E, Q, data=data)
    return dict(**terms, constraints_eligible=eligible, performance_unknown=False,
        reason='OK' if eligible else 'CONSTRAINT_FAILURE', rho=rho, max_rho=max(rho.values()),
        safe_memory_mib=safe_memory, raw_peak_estimate_mib=peak, per_task=stats,
        native_quality=native_quality, episode_energy_evidence=episode,
        scenario_delay_costs=[-sum(terms[t+'_contribution'] for t in ('T', 'L', 'P'))],
        candidate_plan=plan, prediction=prediction, reference_reuse=norm.contract(data)['provenance']['scoped_reference_reuse'],
        physical_predictor_qualified=False, candidate_ranking_qualified=False,
        memory_estimate_requires_measured_confirmation=True)
