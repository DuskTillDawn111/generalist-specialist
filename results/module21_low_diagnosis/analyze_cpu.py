"""Module 21D: read-only CPU diagnosis, no serving/runtime/CUDA imports.

Run with envs/data_env/bin/python results/module21_low_diagnosis/analyze_cpu.py.
All writes are confined to this script's directory. Delta = proposed - control.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
import numpy as np
from scipy.stats import t as student_t, chi2, nct

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
M21 = ROOT / 'results/module21_positive_utility'
M20 = ROOT / 'results/module20_utility_optimizer'
M14 = ROOT / 'results/module14f_memory_ocr_x3'

def read(p):
    return json.loads(Path(p).read_text())

def digest(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def csvwrite(name, rows):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in r.items()})

def stats(values):
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean, sd = float(a.mean()), float(a.std(ddof=1))
    se = sd / math.sqrt(n)
    radius = float(student_t.ppf(.975, n - 1)) * se
    return {'paired_values': a.tolist(), 'n_independent_pairs': n, 'mean': mean,
            'sample_variance': sd * sd, 'sample_sd': sd, 'standard_error': se,
            'nominal_paired_t_95CI': [mean - radius, mean + radius],
            'positive_count': int(sum(a > 0)), 'negative_count': int(sum(a < 0))}

def covariance_share(values, total):
    return float(np.cov(values, total, ddof=1)[0, 1] / np.var(total, ddof=1))

def metric(records):
    done = [r for r in records if r['status'] == 'completed']
    fields = {'mean_E2E_ms': ('e2e_latency_ms', np.mean),
              'P95_E2E_ms': ('e2e_latency_ms', lambda x: np.percentile(x, 95)),
              'mean_service_ms': ('worker_latency_ms', np.mean),
              'P95_service_ms': ('worker_latency_ms', lambda x: np.percentile(x, 95)),
              'mean_queue_ms': ('worker_queue_wait_ms', np.mean),
              'P95_queue_ms': ('worker_queue_wait_ms', lambda x: np.percentile(x, 95)),
              'mean_schedule_delay_ms': ('schedule_delay_ms', np.mean)}
    out = {k: float(fn([r[col] for r in done])) for k, (col, fn) in fields.items()}
    out.update(offered=len(records), accepted_submitted=sum(r.get('actual_submit_time_ns') is not None for r in records),
               completed=len(done), unique_samples=len({r['pool_id'] for r in records}))
    out['mean_other_overhead_ms'] = out['mean_E2E_ms'] - out['mean_service_ms'] - out['mean_queue_ms']
    return out

def tail_details(records, raw):
    rr = sorted(records, key=lambda r: r['e2e_latency_ms'])
    h = .95 * (len(rr) - 1)
    lo, hi = math.floor(h), math.ceil(h)
    details = []
    all_r = raw['phases']['measurement']['request_records']
    for j in sorted({lo, hi}):
        r = rr[j]
        previous = [v for v in all_r if v['model'] == r['model'] and v['arrival_index'] < r['arrival_index']]
        prev = previous[-1] if previous else None
        details.append({'order_statistic_1based': j + 1, 'request_id': r['request_id'], 'sample_id': r['sample_id'],
                        'offset_seconds': r['frozen_offset_seconds'], 'e2e_ms': r['e2e_latency_ms'],
                        'service_ms': r['worker_latency_ms'], 'queue_ms': r['worker_queue_wait_ms'],
                        'previous_same_worker_request_id': prev['request_id'] if prev else None,
                        'previous_same_worker_task': prev['task'] if prev else None,
                        'previous_worker_end_minus_current_submit_ms':
                        (prev['worker_response']['end_ns'] - r['actual_submit_unix_ns']) / 1e6 if prev else None})
    return {'n_requests': len(rr), 'expected_count_above_population_P95': .05 * len(rr),
            'linear_interpolation_index_0based': h, 'upper_order_statistic_weight': h - lo,
            'support_requests': details,
            'probability_of_no_observation_above_population_P95_if_iid': .95 ** len(rr),
            'iid_probability_is_diagnostic_only_not_deployment_inference': True}

def main():
    protected = sorted(p for base in [M21, M20, M14] for p in base.rglob('*') if p.is_file())
    frozen = read(M21 / 'frozen_manifest.json')
    extra = [Path(frozen[k]['path']) for k in ['sample_pool', 'quality_policy']]
    protected = sorted(set(protected + extra))
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    definition = read(M20 / 'utility_definition.json')
    refs = read(M20 / 'utility_reference_manifest.json')['latency_reference_ms']
    profiles = read(M14 / 'heavy_empirical_x3_reconciled.json')
    catalog = read(M21 / 'five_model_catalog.json')
    weights = definition['weights']
    assert weights == {'Q': 0., 'T': .25, 'L': .30, 'P': .30, 'M': .15}
    assert profiles['safe_factor'] == catalog['safe_factor']
    history = read(M21 / 'real_utility_analysis.json')
    original = {r['run_id']: r for r in history['runs']}
    cpu = read(M21 / 'cpu_fullpool_optimization.json')['stages']['LOW']
    predicted = {'proposed': cpu['winner']['winner'], 'control': cpu['control']['winner']}
    raw = {(i, a): read(M21 / f'raw_runs/low_{a}_rep{i}.json') for i in range(1, 4) for a in ['proposed', 'control']}
    computed, run_audits, task_data, tails = {}, [], {}, []
    reference_checks = []
    references = [frozen[k] for k in ['catalog', 'sample_pool', 'utility_definition', 'utility_references', 'quality_policy']]
    references += frozen['implementation'] + frozen['reused_implementation']
    references += [read(M20 / 'utility_reference_manifest.json')['reference_source']]
    references += [r['raw'] for r in original.values()]
    for r in references:
        actual = digest(r['path'])
        reference_checks.append({'path': r['path'], 'expected_sha256': r['sha256'], 'actual_sha256': actual,
                                 'matches': actual == r['sha256']})
    for (rep, arm), d in raw.items():
        slot, ph = d['slot'], d['phases']['measurement']
        rates, w = slot['lambda_rps'], ph['window']
        total_rate, budget = sum(rates.values()), slot['safe_memory_budget_mib']
        assert budget == 32019.2544
        all_r = ph['request_records']
        done = [r for r in all_r if r['status'] == 'completed']
        within = sum(r['completion_time_ns'] <= w['end_time_ns'] for r in done)
        norms = {'T_norm': 1 - min(1, len(done) / len(all_r), within / (total_rate * w['planned_duration_seconds'])),
                 'L_norm': 0., 'P_norm': 0.}
        for task, q in rates.items():
            records = [r for r in all_r if r['task'] == task]
            m = metric(records)
            model = d['replica_to_family'][records[0]['model']]
            s0 = profiles['models'][model]['service_time_ms'][task]
            assert s0 == catalog['models'][model]['service_time_ms'][task]
            m.update(model=model, offered_QPS=q, offered_weight=q / total_rate, reference_ms=refs[task],
                     within_window_completed=sum(r['completion_time_ns'] <= w['end_time_ns'] for r in records),
                     L_norm_contribution=q / total_rate * m['mean_E2E_ms'] / refs[task],
                     P_norm_contribution=q / total_rate * m['P95_E2E_ms'] / refs[task],
                     frozen_service_ms=s0, service_ratio_to_frozen=m['mean_service_ms'] / s0,
                     CPU_predicted_mean_ms=predicted[arm]['per_task'][task]['mean_ms'],
                     CPU_predicted_P95_ms=predicted[arm]['per_task'][task]['p95_ms'])
            m['U_mean_contribution'] = -.30 * m['L_norm_contribution']
            m['U_P95_contribution'] = -.30 * m['P_norm_contribution']
            m['service_error_ms'] = m['mean_service_ms'] - s0
            m['service_error_percent'] = 100 * (m['service_ratio_to_frozen'] - 1)
            task_data[rep, arm, task] = m
            norms['L_norm'] += m['L_norm_contribution']
            norms['P_norm'] += m['P_norm_contribution']
            tails.append({'run_id': slot['run_id'], 'task': task, **tail_details(records, d)})
        idle = {phase: float(np.mean([v['owned_used_mib'] for v in d[phase]])) for phase in ['postwarm_idle', 'postmeasurement_idle']}
        safe = profiles['safe_factor'] * max(idle.values())
        norms['M_norm'] = safe / budget
        norms.update(U_throughput=-.25 * norms['T_norm'], U_mean=-.30 * norms['L_norm'],
                     U_P95=-.30 * norms['P_norm'], U_memory=-.15 * norms['M_norm'])
        norms['U_total'] = sum(norms[k] for k in ['U_throughput', 'U_mean', 'U_P95', 'U_memory'])
        norms['safe_memory_mib'] = safe
        computed[rep, arm] = norms
        old = original[slot['run_id']]
        errors = {k: norms[k] - old[k] for k in ['T_norm', 'L_norm', 'P_norm', 'M_norm']}
        errors['U_total'] = norms['U_total'] - old['U']
        max_err = max(abs(v) for v in errors.values())
        assert max_err < 1e-12
        equations = {'e2e_ms': [], 'worker_service_ms': [], 'worker_queue_ms': [], 'runtime_ms': []}
        for r in all_r:
            response = r['worker_response']
            equations['e2e_ms'].append(abs(r['e2e_latency_ms'] - (r['completion_time_ns'] - r['scheduled_time_ns']) / 1e6))
            equations['runtime_ms'].append(abs(r['runtime_latency_ms'] - (r['completion_time_ns'] - r['actual_submit_time_ns']) / 1e6))
            equations['worker_service_ms'].append(abs(r['worker_latency_ms'] - (response['end_ns'] - response['start_ns']) / 1e6))
            equations['worker_queue_ms'].append(abs(r['worker_queue_wait_ms'] - (response['start_ns'] - response['submit_ns']) / 1e6))
        phase_audits = {}
        for label in ['warmup', 'measurement']:
            phase = d['warmup'] if label == 'warmup' else ph
            trajectory = read(slot['trajectories'][label]['path'])
            records = phase['request_records']
            tw = phase['window']
            match = len(records) == len(trajectory['events']) and all(
                r['task'] == e['task'] and r['pool_id'] == e['item']['pool_id'] and
                r['sample_id'] == e['item']['sample']['id'] and r['frozen_offset_seconds'] == e['offset_seconds'] and
                r['routing_uniform'] == e['routing_uniform'] and r['replica_uniform'] == e['replica_uniform'] and
                r['scheduled_time_ns'] - tw['start_time_ns'] == round(e['offset_seconds'] * 1e9)
                for r, e in zip(records, trajectory['events']))
            assert match
            phase_audits[label] = {'trajectory_exact_match': match, 'trajectory_sha256_matches': digest(slot['trajectories'][label]['path']) == slot['trajectories'][label]['sha256'],
                                   'planned_duration_seconds': tw['planned_duration_seconds'], 'actual_duration_seconds': tw['duration_seconds'],
                                   'completed': sum(r['status'] == 'completed' for r in records), 'offered': len(records),
                                   'drain_seconds': tw['drain_seconds'], 'stop': tw['stop']}
        tele = d['telemetry']['samples']
        measuring = [v for v in tele if v['phase'] == 'measurement']
        text_rr = [r for r in all_r if r['task'] == 'text']
        quartiles = [float(np.mean([r['worker_latency_ms'] for r in text_rr if b * 30 <= r['frozen_offset_seconds'] < (b + 1) * 30])) for b in range(4)]
        quality = all(r['task'] in catalog['models'][d['replica_to_family'][r['model']]]['supported_tasks'] for r in all_r)
        quality = quality and not any(r['worker_response'].get('valid_closed_set_output') is False for r in all_r)
        run_audits.append({'run_id': slot['run_id'], 'seed': slot['seed'], 'phases': phase_audits,
                           'offered': len(all_r), 'accepted_submitted': sum(r.get('actual_submit_time_ns') is not None for r in all_r),
                           'completed': len(done), 'completed_in_window': within, 'unfinished': len(all_r) - len(done),
                           'completed_after_window': len(done) - within, 'quality_admission_matches_original': quality == old['quality_admission'],
                           'safe_memory_mib': safe, 'idle_owned_means_mib': idle, 'frozen_predicted_safe_memory_mib': slot['CPU_safe_memory_mib'],
                           'memory_prediction_error_mib': safe - slot['CPU_safe_memory_mib'], 'sampled_owned_peak_mib': max(v['owned_used_mib'] for v in tele),
                           'budget_mib': budget, 'raw_valid_session': d['valid_session'], 'errors': d['errors'],
                           'system_state': ph['stability']['stability_state_v2'], 'system_stability_detail': ph['stability'],
                           'foreign_pids_observed': sorted({p for v in tele for p in v['foreign_pids']}),
                           'warmup_cleared_before_measurement': all(r['status'] == 'completed' and r['completion_time_ns'] < w['start_time_ns'] for r in d['warmup']['request_records']),
                           'probe_cleared_before_warmup': all(r['status'] == 'completed' and r['completion_time_ns'] < d['warmup']['window']['start_time_ns'] for r in d['path_probe_records']),
                           'probe_count': len(d['path_probe_records']), 'probe_counts_by_task': dict(Counter(r['task'] for r in d['path_probe_records'])),
                           'worker_pids': [v['pid'] for v in d['worker_identities'].values()], 'GPU_uuid': d['baseline_GPU']['uuid'],
                           'GPU_driver': d['baseline_GPU']['driver'], 'gpu_release_verified': d['gpu_release']['verified'],
                           'max_time_equation_errors_ms': {k: max(v) for k, v in equations.items()}, 'utility_recompute_errors': errors,
                           'text_service_mean_by_30s_quartile_ms': quartiles,
                           'measurement_sampled_power_mean_w': float(np.mean([v['power_w'] for v in measuring])),
                           'measurement_sampled_host_available_RAM_min_bytes': min(v['host_available_RAM_bytes'] for v in measuring)})
        assert max(max(v) for v in equations.values()) < 1e-8
    components = ['memory', 'mean', 'P95', 'throughput', 'total']
    decomposition, per_task = [], []
    for rep in range(1, 4):
        a, b = computed[rep, 'proposed'], computed[rep, 'control']
        row = {'repetition': rep, 'seed': raw[rep, 'proposed']['slot']['seed'], 'delta_direction': 'proposed_minus_control'}
        for name in components:
            row[f'proposed_U_{name}'] = a[f'U_{name}']
            row[f'control_U_{name}'] = b[f'U_{name}']
            row[f'delta_U_{name}'] = a[f'U_{name}'] - b[f'U_{name}']
        row['component_sum'] = sum(row[f'delta_U_{name}'] for name in components[:-1])
        row['original_delta_U'] = original[f'low_proposed_rep{rep}']['U'] - original[f'low_control_rep{rep}']['U']
        row['sum_minus_original'] = row['component_sum'] - row['original_delta_U']
        row['proposed_safe_memory_mib'], row['control_safe_memory_mib'] = a['safe_memory_mib'], b['safe_memory_mib']
        row['safe_memory_saving_mib'] = b['safe_memory_mib'] - a['safe_memory_mib']
        assert abs(row['sum_minus_original']) < 1e-12
        decomposition.append(row)
        for task in refs:
            ma, mb = task_data[rep, 'proposed', task], task_data[rep, 'control', task]
            tr = {'repetition': rep, 'task': task, 'delta_direction': 'proposed_minus_control'}
            for prefix, m in [('proposed', ma), ('control', mb)]:
                tr.update({prefix + '_' + k: v for k, v in m.items()})
            for k in ma:
                if isinstance(ma[k], (int, float)):
                    tr['delta_' + k] = ma[k] - mb[k]
            tr['delta_U_latency_total'] = tr['delta_U_mean_contribution'] + tr['delta_U_P95_contribution']
            tr['relative_mean_E2E_change_percent'] = 100 * (ma['mean_E2E_ms'] / mb['mean_E2E_ms'] - 1)
            tr['relative_P95_E2E_change_percent'] = 100 * (ma['P95_E2E_ms'] / mb['P95_E2E_ms'] - 1)
            coefficient = -.30 * ma['offered_weight'] / ma['reference_ms']
            for source in ['service', 'queue', 'other_overhead']:
                tr[f'delta_U_mean_from_{source}'] = coefficient * tr[f'delta_mean_{source}_ms']
            assert abs(sum(tr[f'delta_U_mean_from_{source}'] for source in ['service', 'queue', 'other_overhead']) - tr['delta_U_mean_contribution']) < 1e-12
            per_task.append(tr)
        assert abs(sum(r['delta_U_mean_contribution'] for r in per_task if r['repetition'] == rep) - row['delta_U_mean']) < 1e-12
        assert abs(sum(r['delta_U_P95_contribution'] for r in per_task if r['repetition'] == rep) - row['delta_U_P95']) < 1e-12
    total = [r['delta_U_total'] for r in decomposition]
    compstats = {name: stats([r['delta_U_' + name] for r in decomposition]) for name in components}
    for name in components[:-1]:
        compstats[name]['covariance_with_total_over_total_variance'] = covariance_share(compstats[name]['paired_values'], total)
    taskstats = {}
    for task in refs:
        rows = [r for r in per_task if r['task'] == task]
        taskstats[task] = {k: stats([r['delta_' + k] for r in rows]) for k in ['U_mean_contribution', 'U_P95_contribution', 'U_latency_total', 'mean_E2E_ms', 'P95_E2E_ms', 'mean_service_ms', 'mean_queue_ms']}
        taskstats[task]['covariance_share_of_total_variance'] = covariance_share([r['delta_U_latency_total'] for r in rows], total)
        taskstats[task]['run_mean_service_ratios_by_arm'] = {arm: [task_data[i, arm, task]['service_ratio_to_frozen'] for i in range(1, 4)] for arm in ['proposed', 'control']}
        taskstats[task]['run_mean_service_ms_by_arm'] = {arm: [task_data[i, arm, task]['mean_service_ms'] for i in range(1, 4)] for arm in ['proposed', 'control']}
        taskstats[task]['request_counts_per_repetition'] = [task_data[i, 'proposed', task]['offered'] for i in range(1, 4)]
        coefficient = -.30 * task_data[1, 'proposed', task]['offered_weight'] / refs[task]
        taskstats[task]['delta_U_mean_additive_sources'] = {source: stats([r[f'delta_U_mean_from_{source}'] for r in rows]) for source in ['service', 'queue', 'other_overhead']}
        taskstats[task]['one_ms_latency_utility_sensitivity_each_mean_or_P95'] = coefficient
        taskstats[task]['CPU_predicted_delta_U_latency'] = coefficient * sum(predicted['proposed']['per_task'][task][k] - predicted['control']['per_task'][task][k] for k in ['mean_ms', 'p95_ms'])
        taskstats[task]['actual_minus_CPU_predicted_delta_U_latency'] = taskstats[task]['U_latency_total']['mean'] - taskstats[task]['CPU_predicted_delta_U_latency']
    s = compstats['total']['sample_sd']
    sigma_ci = [math.sqrt(2 * s * s / chi2.ppf(p, 2)) for p in [.975, .025]]
    planning = []
    for window in [120., 1200., 4000.]:
        sigma = s * math.sqrt(120. / window)
        n_power = next(n for n in range(3, 20000) if nct.sf(student_t.ppf(.95, n - 1), n - 1, .01 * math.sqrt(n) / sigma) >= .8)
        n_ci = next(n for n in range(3, 20000) if student_t.ppf(.975, n - 1) * sigma / math.sqrt(n) <= .01)
        planning.append({'window_seconds_per_arm': window, 'expected_ASR_requests': .05 * window, 'expected_LID_requests': .1 * window,
                         'assumed_pair_sd': sigma, 'fixed_pairs_for_80percent_power_H0_mean_le_001_true_mean_002_one_sided_alpha_005': n_power,
                         'fixed_pairs_for_nominal_95CI_halfwidth_001': n_ci,
                         'measurement_only_GPU_hours_at_power_sample_size': 2 * window * n_power / 3600})
    lid_mean = taskstats['lid']['U_latency_total']['mean']
    mem_mean = compstats['memory']['mean']
    memory = {'theoretical_safe_saving_mib': predicted['control']['safe_memory_mib'] - predicted['proposed']['safe_memory_mib'],
              'theoretical_delta_U_memory': predicted['proposed']['M_contribution'] - predicted['control']['M_contribution'],
              'observed_safe_savings_mib': [r['safe_memory_saving_mib'] for r in decomposition],
              'observed_delta_U_memory': compstats['memory'],
              'interpretation': 'Same saving direction, observed saving smaller; NVML owned postwarm/postmeasurement idle means recompute exactly. Per-process MMS/context and other process allocations differ from standalone table. Safe memory is a frozen idle safety surrogate, not an unsampled instantaneous peak guarantee.'}
    memory['mean_observed_safe_saving_mib'] = float(np.mean(memory['observed_safe_savings_mib']))
    memory['saving_relative_error_percent'] = 100 * (memory['mean_observed_safe_saving_mib'] / memory['theoretical_safe_saving_mib'] - 1)
    memory['per_run_idle_allocation_by_family_mib'] = []
    for d in raw.values():
        means = {phase: np.mean([v['owned_used_mib'] for v in d[phase]]) for phase in ['postwarm_idle', 'postmeasurement_idle']}
        phase = max(means, key=means.get)
        alloc = {v['family']: float(np.mean([next((p['used_mib'] for p in sample['processes'] if p['pid'] == v['pid']), 0.) for sample in d[phase]])) for v in d['worker_identities'].values()}
        memory['per_run_idle_allocation_by_family_mib'].append({'run_id': d['slot']['run_id'], 'phase_selected': phase, 'family_idle_mib': alloc})
    pair_audits = []
    for rep in range(1, 4):
        aa, bb = raw[rep, 'proposed'], raw[rep, 'control']
        signatures = lambda d, label: [(r['task'], r['pool_id'], r['sample_id'], r['frozen_offset_seconds'], r['routing_uniform'], r['replica_uniform']) for r in (d['warmup'] if label == 'warmup' else d['phases']['measurement'])['request_records']]
        pair_audits.append({'repetition': rep, 'same_offered_measurement_trace_task_sample_uniforms': signatures(aa, 'measurement') == signatures(bb, 'measurement'),
                            'same_warmup_trace_task_sample_uniforms': signatures(aa, 'warmup') == signatures(bb, 'warmup'),
                            'same_rates': aa['slot']['lambda_rps'] == bb['slot']['lambda_rps'], 'same_frozen_manifest_hash': aa['frozen_manifest']['sha256'] == bb['frozen_manifest']['sha256'] == digest(M21 / 'frozen_manifest.json'),
                            'same_budget': aa['slot']['safe_memory_budget_mib'] == bb['slot']['safe_memory_budget_mib'],
                            'same_transport_limits': all(aa['transport_config'][k] == bb['transport_config'][k] for k in ['warmup_seconds', 'measurement_seconds', 'drain_timeout_seconds', 'max_pending_per_model', 'startup_timeout_seconds', 'shutdown_timeout_seconds']),
                            'observed_measurement_order': sorted(['proposed', 'control'], key=lambda arm: raw[rep, arm]['phases']['measurement']['window']['start_unix_ns'])})
    after = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert before == after
    assert all(r['matches'] for r in reference_checks)
    consistency = {'module': '21D', 'CPU_only': True, 'GPU_commands_or_runtime_imports_executed': False,
                   'delta_definition': 'G-only proposed minus MMS control', 'formula': 'U=-0.25*T_loss-0.30*L_norm-0.30*P_norm-0.15*M_norm',
                   'weights': weights, 'latency_reference_ms': refs, 'offered_weights_source': 'frozen offered QPS, total 8.35; never observed/completed-count weights',
                   'safe_factor': profiles['safe_factor'], 'memory_budget_mib': 32019.2544, 'run_audits': run_audits, 'pair_audits': pair_audits,
                   'reference_hash_checks': reference_checks, 'memory_diagnosis': memory,
                   'historical_input_integrity': {'all_unchanged': before == after, 'protected_file_count': len(before), 'sha256_before_equal_after': before},
                   'accepted_count_semantics': 'accepted_submitted is actual_submit_time_ns present; no independent worker admission ACK exists. Every submitted request has a completion response here.',
                   'input_distribution_evidence': 'Exact task/sample/order/offset/uniform match to frozen event payloads. Source sample-pool hash verified and enqueue implementation verified. Logs do not serialize a second copy of media bytes as received by workers.',
                   'warmup_rule': 'Same 10s offered warmup, drain up to 60s, then 2s idle; completed queues before formal measurement verified. Sparse-task warmup offers no common minimum per-task count.',
                   'probe_rule_caveat': 'Same three sequential probes per admitted resident model/task path; 24 proposed vs 27 control, because control additionally primes MMS LID. Exact pre-measurement probe workload differs by deployment; this is the frozen rule, not a trace mismatch in formal measurement. Cache/power state equality is unverified.',
                   'completion_and_censoring': 'All 6066 formal requests submitted and completed before each 120s end; zero unfinished or after-window completions. No censoring or drain throughput masking in these six runs. Future incomplete cohorts must retain U=null.',
                   'throughput_semantics_caveat': 'Frozen T_loss uses planned QPS*window rather than realized Poisson offered count. Counts 1016/1008/1009 exceed expected 1002, so clamp gives zero; extra arrivals can mask a few boundary completions in other runs. This does not explain these pairs: none finished late.',
                   'quality': {'same_frozen_hard_admission': True, 'quality_policy_hash_verified': True, 'admitted_paths_and_closed_set_outputs_verified': True,
                               'no_quality_reweighting_or_new_quality_gate': True, 'caveat': 'Per-run accuracy/WER are descriptive, not new pool admission estimates; e.g. observed Text accuracy below historical 0.925 on this repeated small pool does not change frozen full-dataset admission.'},
                   'missing_causal_evidence': ['GPU clocks and temperature not logged', 'No same-input sparse standalone control under Module21 arrival/warmup conditions', 'No per-request kernel/CPU preprocessing timing', 'No simultaneous randomized residency-only contrast; routing and MMS residency change together'],
                   'worker_states_from_original_cpu_analysis': history['workers'],
                   'effective_Module21_acceptance': read(M21 / 'acceptance_clarification.json')['effective_rule']}
    statistical = {'module': '21D', 'primary_statistical_unit': 'independent fresh GPU process repetition, paired by frozen trace; n=3 pairs, not 6066 deployment samples',
                   'component_paired_statistics': compstats, 'task_paired_statistics': taskstats,
                   'absolute_U_by_arm': {arm: {'run_values': [computed[i, arm]['U_total'] for i in range(1, 4)],
                                              'n_independent_runs': 3,
                                              'mean': float(np.mean([computed[i, arm]['U_total'] for i in range(1, 4)])),
                                              'sample_sd': float(np.std([computed[i, arm]['U_total'] for i in range(1, 4)], ddof=1))}
                                        for arm in ['proposed', 'control']},
                   'variance_accounting': 'Cov(component,total)/Var(total) sums to one including covariance. Negative shares are covariance cancellation, not negative variances. With n=3 shares are descriptive, not stable population estimates.',
                   'component_covariance_matrix': {'order': components[:-1], 'matrix': np.cov(np.array([compstats[c]['paired_values'] for c in components[:-1]]), ddof=1).tolist()},
                   'tail_order_statistics': tails,
                   'normal_iid_pair_sd_nominal_95_interval': sigma_ci,
                   'can_reliably_distinguish_mean_delta_above_001': False,
                   'paired_t_one_sided_p_H0_mean_le_001': float(student_t.sf((compstats['total']['mean'] - .01) / compstats['total']['standard_error'], 2)),
                   'sign_evidence': 'Two positive, one negative; three pairs cannot establish universal/systematic loss. All background paired CIs include zero. LID service/E2E penalty is present in all pairs but n=3 and multiple task exploration do not justify a general deployment-loss claim.',
                   'LID_mean_latency_utility_loss': lid_mean, 'LID_plus_memory_mean_delta': lid_mean + mem_mean,
                   'profile_mismatch': 'Observed handler service differs markedly from original standalone constants. Text/Image/OCR and Qwen LID ratios persist above one in all runs; those changes are too large to represent only a quantile sample-count issue. Conditions and sample distributions differ from historical profiles, so causal co-location inflation is unproven.',
                   'sparse_tail_interpretation': 'LID n=10/9/9 and ASR n=12/6/7: P95 uses the largest two observations. They are unstable tail estimates; ASR has small paired utility contribution after the original normalization, while LID remains material. Text accounts for most total paired variance.',
                   'request_samples_not_independent_population_draws': 'Frozen sample pool cycles within task; Text has 30 unique inputs per run, repeated 16-17 times. Different repetitions have independent arrival seeds/fresh workers but partly shared input identities. Inference is conditional on this fixed pool and sequential runs; no workload-population claim.',
                   'hypothetical_sample_size_sensitivity': {'not_a_recommended_GPU_extension': True, 'planning_rows': planning,
                       'hypothesis': 'One-sided H0 mean delta<=0.01, target true mean=0.02, alpha=0.05, power=0.8; also nominal two-sided 95% precision +/-0.01.',
                       'assumptions': 'Normal independent pair deltas, observed n=3 SD equals population SD, all variance decreases as 1/window, fixed unchanged workload/model/utility, no time drift or persistent co-location floor. These strong assumptions are unverified.',
                       'tail_count_target': '4000s yields expected ASR=200 (10 observations above population P95) and LID=400 (20); Poisson counts are random and input cycling remains. This is a count heuristic, not precision proof.',
                       'no_window_scaling_case': 'If Text/run-level state variance persists, use the 120s-row repetition requirement even with longer windows; longer measurement alone is insufficient.',
                       'uncertainty': 'Three pairs cannot estimate variance-vs-window scaling or select an efficient fixed sample size confidently. SD nominal interval assumes normal iid pairs; nonnormal/drift effects may make it misleading. More data cannot guarantee positive gain.'}}
    decision = {'module': '21D', 'decision': 'LOW_COLOCATION_OR_MODEL_MISMATCH',
                'reason': 'Large persistent service-profile mismatch is directly observed; paired Text state changes dominate variance, while doubled Qwen LID service and its queue tails consume the predicted memory margin. n=3 is underpowered for utility sign, but this is not only a sample-size problem.',
                'recommended_action': 'Suspend current LOW candidate and do not extend its GPU efficacy experiment as-is. First diagnose sparse-serving/profile conditions using the existing logs; any future controlled remeasurement requires a separately frozen protocol and explicit new authorization. Do not modify these frozen inputs.',
                'GPU_extension_recommended_now': False, 'GPU_executed_in_this_module': False,
                'systematic_total_utility_loss_established': False, 'causal_MMS_removal_background_degradation_established': False,
                'transferable_corrected_model_established': False,
                'quantified_errors': {'service_ratios_by_task_and_arm': {task: v['run_mean_service_ratios_by_arm'] for task, v in taskstats.items()},
                                      'CPU_predicted_delta_U': cpu['predicted_delta_U'], 'actual_mean_delta_U': compstats['total']['mean'],
                                      'actual_minus_predicted_delta_U': compstats['total']['mean'] - cpu['predicted_delta_U'],
                                      'memory_saving_relative_error_percent': memory['saving_relative_error_percent']},
                'required_evidence_before_any_new_efficacy_test': ['Matched sample/arrival-condition single-path service evidence, including sparse idle gaps and power/clock state',
                    'Independent counterbalanced repetitions separating residency-only effects from added LID contention on Qwen',
                    'Fixed sample size and window preregistered before outcomes; preserve original weights, reference values, hard admission and null utility for incomplete/unstable cohorts'],
                'diagnostic_GPU_cost_if_later_authorized': {'not_scheduled': True, 'example_isolation_design': '4 isolated paths (Text/Image/OCR/Qwen LID), 3 fresh repetitions each, original low task rate and matched input stream, 120s each',
                    'isolated_measurement_only_minutes': 24, 'limits': 'Loading/probes/idle/drain add cost. Sparse LID still only ~12 requests per run; causal residency-only plus routed-LID factorial contrasts add at least 6 more 120s runs (12 measurement minutes) for three pairs. This is a cost illustration, not an executable protocol or sufficient tail-power design.'},
                'MID_HIGH': {'independent_exploration_still_reasonable': True, 'currently_validated': False,
                             'MID': 'Predicted benefit +0.595139 at LID=3 QPS; structurally larger than LOW margin, but service/queue mismatch must be considered before independent validation.',
                             'HIGH': 'Predicted benefit +0.043004 at LID=12 QPS; narrow margin and much greater contention/capacity uncertainty; validate independently only after profile-condition diagnosis.',
                             'cannot_inherit_LOW_outcomes_or_claim_three_phase_success': True},
                'stop_after_analysis': True}
    csvwrite('utility_component_decomposition.csv', decomposition)
    csvwrite('per_task_paired_differences.csv', per_task)
    write('measurement_consistency.json', consistency)
    write('statistical_diagnosis.json', statistical)
    write('next_experiment_decision.json', decision)
    write('diagnosis_manifest.json', {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'CPU_only': True,
                                    'script_sha256': digest(__file__), 'python': 'envs/data_env/bin/python',
                                    'protected_input_files': len(before), 'input_hashes_verified_unchanged': before == after})
    print(json.dumps({'decision': decision['decision'], 'components': compstats,
                      'task_latency_means_and_variance_shares': {task: [v['U_latency_total']['mean'], v['covariance_share_of_total_variance']] for task, v in taskstats.items()},
                      'memory': {k: v for k, v in memory.items() if 'per_run' not in k}, 'planning': planning,
                      'sigma_ci': sigma_ci, 'integrity': len(before)}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
