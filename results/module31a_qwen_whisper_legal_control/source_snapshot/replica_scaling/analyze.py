"""Analyze preserved runs, including unstable and failed two-replica conditions."""
from collections import Counter
import math
import statistics
from .common import OUT, frozen_plan, read, write, reference

METRICS = ('offered_load_rps', 'actual_offered_load_rps', 'completed_throughput_rps',
    'final_half_window_throughput_rps', 'completion_ratio', 'within_window_completion_ratio',
    'cohort_throughput_including_drain_rps', 'mean_e2e_ms', 'p95_e2e_ms', 'mean_queue_wait_ms',
    'p95_queue_wait_ms', 'mean_worker_latency_ms', 'p95_worker_latency_ms', 'mean_schedule_delay_ms')


def stats(values):
    valid = [v for v in values if v is not None and math.isfinite(v)]
    return dict(values=values, count=len(valid), missing_count=len(values) - len(valid),
                mean=statistics.mean(valid) if valid else None,
                sample_std=statistics.stdev(valid) if len(valid) > 1 else None,
                minimum=min(valid) if valid else None, maximum=max(valid) if valid else None)


def ratio(a, b):
    return a / b if a is not None and b is not None and b > 0 else None


def memory_metrics(raw):
    mean = lambda xs: statistics.mean(xs) if xs else None
    maximum = lambda xs: max(xs) if xs else None
    extract = lambda values, key: [v[key] for v in values if v.get(key) is not None]
    load = raw.get('idle_after_load', [])
    idle = raw.get('idle_before_workload', [])
    samples = raw.get('telemetry', {}).get('samples', [])
    workload = [s for s in samples if s['phase'] == 'measurement']
    active = [s for s in samples if s['phase'] in ('measurement', 'drain')]
    return dict(idle_after_load_owned_mib=mean(extract(load, 'owned_used_mib')),
        idle_before_workload_owned_mib=mean(extract(idle, 'owned_used_mib')),
        idle_before_workload_device_mib=mean(extract(idle, 'used_mib')),
        peak_owned_mib=maximum(extract(active, 'owned_used_mib')), peak_device_mib=maximum(extract(active, 'used_mib')),
        mean_gpu_util_percent=mean(extract(workload, 'gpu_util_percent')), peak_gpu_util_percent=maximum(extract(workload, 'gpu_util_percent')),
        mean_memory_bandwidth_util_percent=mean(extract(workload, 'memory_util_percent')),
        mean_power_w=mean(extract(workload, 'power_w')), peak_power_w=maximum(extract(active, 'power_w')),
        telemetry_count=len(active), telemetry_errors=raw.get('telemetry', {}).get('errors', []),
        release_verified=raw.get('gpu_release', {}).get('verified', False))


def analyze():
    plan = frozen_plan()
    ledger = read(OUT / 'campaign_ledger.json')
    raw_by_id = {s['run_id']: read(s['output']['path']) for s in ledger['slots'] if s.get('output')}
    conditions = []
    condition_map = {}
    memories = []
    pairs = []
    model_results = []
    observations = []
    for model in plan['models']:
        name = model['model_id']
        for level in ('low', 'medium', 'high'):
            for count in (1, 2):
                slots = [s for s in ledger['slots'] if (s['model_id'], s['replica_count'], s['load_level']) == (name, count, level)]
                raws = [raw_by_id[s['run_id']] for s in slots if s['run_id'] in raw_by_id]
                valid = [r for r in raws if r['valid_full_repetition']]
                memory = [memory_metrics(r) for r in raws]
                metrics = {key: stats([(r.get('metrics') or {}).get(key) for r in raws]) for key in METRICS}
                evidence_keys = ('peak_backlog_requests', 'end_backlog_requests', 'tail_backlog_slope', 'tail_backlog_growth', 'drain_duration')
                evidence = {key: stats([(r.get('stability_evidence') or {}).get(key) for r in raws]) for key in evidence_keys}
                evidence['offered_end_backlog_requests'] = stats([(r.get('stability_evidence') or {}).get('offered_backlog_diagnostic', {}).get('end_backlog_requests') for r in raws])
                evidence['offered_peak_backlog_requests'] = stats([(r.get('stability_evidence') or {}).get('offered_backlog_diagnostic', {}).get('peak_backlog_requests') for r in raws])
                per_replica = {}
                for i in range(count):
                    worker_id = f'{name}#{i}'
                    per_replica[worker_id] = {key: stats([((r.get('metrics') or {}).get('per_replica', {}).get(worker_id) or {}).get(key) for r in raws])
                        for key in ('offered_requests', 'completed_requests', 'within_window_completions', 'completed_throughput_rps', 'completion_ratio')}
                condition = dict(model_id=name, task=model['task'], replica_count=count, load_level=level,
                    offered_load_rps=model['loads'][level], attempted_repetitions=len(raws), valid_full_repetitions=len(valid),
                    skipped_repetitions=len(slots) - len(raws), all_three_valid=len(valid) == 3,
                    execution_outcomes=dict(Counter(r['execution_status'] for r in raws)),
                    stability_outcomes=dict(Counter((r.get('stability') or {}).get('stability_state_v2', 'unclassified_execution_failure') for r in raws)),
                    metrics=metrics, backlog_metrics=evidence, per_replica=per_replica,
                    raw_paths=[str(OUT / 'raw_runs' / (r['slot']['run_id'] + '.json')) for r in raws],
                    aggregation='Equal arithmetic weighting across preserved attempted repetitions; no outcome/stability filtering. Missing measurements remain null.',
                    ratios_eligible_for_capacity_claim=len(valid) == 3 and all(r.get('status') == 'completed' for r in raws))
                conditions.append(condition)
                condition_map[(name, count, level)] = condition
                mem_keys = ('idle_after_load_owned_mib', 'idle_before_workload_owned_mib', 'idle_before_workload_device_mib',
                            'peak_owned_mib', 'peak_device_mib', 'mean_gpu_util_percent', 'peak_gpu_util_percent',
                            'mean_memory_bandwidth_util_percent', 'mean_power_w', 'peak_power_w')
                memories.append(dict(model_id=name, replica_count=count, load_level=level,
                    measurements={key: stats([m[key] for m in memory]) for key in mem_keys},
                    release_verified=all(m['release_verified'] for m in memory) if memory else False,
                    measurements_include_all_attempts=True))
            paired = []
            for repetition in (1, 2, 3):
                a = raw_by_id.get(f'{name}_r1_{level}_rep{repetition}')
                b = raw_by_id.get(f'{name}_r2_{level}_rep{repetition}')
                if not a or not b:
                    paired.append(dict(repetition=repetition, missing_pair=True, eligible=False))
                    continue
                am, bm = a.get('metrics') or {}, b.get('metrics') or {}
                av, bv = memory_metrics(a), memory_metrics(b)
                gain = ratio(bm.get('completed_throughput_rps'), am.get('completed_throughput_rps'))
                paired.append(dict(repetition=repetition, missing_pair=False,
                    eligible=a['valid_full_repetition'] and b['valid_full_repetition'] and a['status'] == b['status'] == 'completed',
                    throughput_gain=gain, efficiency=ratio(gain, 2.),
                    final_half_throughput_gain=ratio(bm.get('final_half_window_throughput_rps'), am.get('final_half_window_throughput_rps')),
                    latency_ratio=ratio(bm.get('mean_e2e_ms'), am.get('mean_e2e_ms')),
                    p95_latency_ratio=ratio(bm.get('p95_e2e_ms'), am.get('p95_e2e_ms')),
                    queue_wait_ratio=ratio(bm.get('mean_queue_wait_ms'), am.get('mean_queue_wait_ms')),
                    worker_latency_interference_ratio=ratio(bm.get('mean_worker_latency_ms'), am.get('mean_worker_latency_ms')),
                    memory_ratio=ratio(bv['idle_before_workload_owned_mib'], av['idle_before_workload_owned_mib']),
                    idle_memory_increase_mib=(bv['idle_before_workload_owned_mib'] - av['idle_before_workload_owned_mib']) if None not in (av['idle_before_workload_owned_mib'], bv['idle_before_workload_owned_mib']) else None,
                    peak_memory_ratio=ratio(bv['peak_owned_mib'], av['peak_owned_mib']),
                    execution_status_1=a['execution_status'], execution_status_2=b['execution_status'],
                    stability_1=(a.get('stability') or {}).get('stability_state_v2'), stability_2=(b.get('stability') or {}).get('stability_state_v2')))
            fields = ('throughput_gain', 'efficiency', 'final_half_throughput_gain', 'latency_ratio', 'p95_latency_ratio',
                      'queue_wait_ratio', 'worker_latency_interference_ratio', 'memory_ratio', 'idle_memory_increase_mib', 'peak_memory_ratio')
            a = condition_map[(name, 1, level)]
            b = condition_map[(name, 2, level)]
            pairs.append(dict(model_id=name, load_level=level, offered_load_rps=model['loads'][level],
                paired_repetitions=paired, paired_ratios={key: stats([p.get(key) for p in paired]) for key in fields},
                all_three_pairs_eligible=all(p['eligible'] for p in paired),
                throughput_gain_ratio_of_means=ratio(b['metrics']['completed_throughput_rps']['mean'], a['metrics']['completed_throughput_rps']['mean']),
                low_load_demand_censored=level == 'low',
                note='Ratios from partial/stopped/failed observations remain descriptive, never silently omitted or interpreted as sustainable capacity.'))
        model_pairs = [p for p in pairs if p['model_id'] == name]
        pressure_pairs = []
        usable = []
        for pair in model_pairs:
            level = pair['load_level']
            a = condition_map[(name, 1, level)]
            baseline_queue = a['backlog_metrics']['peak_backlog_requests']['mean']
            constrained = bool(a['stability_outcomes'].get('unstable') or a['stability_outcomes'].get('recoverable_congested') or
                (baseline_queue is not None and baseline_queue >= 5 and (a['metrics']['within_window_completion_ratio']['mean'] or 0) < .98))
            pair['baseline_capacity_pressure_observed'] = constrained
            if level != 'low' and pair['all_three_pairs_eligible'] and constrained:
                pressure_pairs.append(pair)
                gains = pair['paired_ratios']['throughput_gain']
                if gains['minimum'] > plan['interpretation']['clear_usable_gain_min']:
                    usable.append(pair)
        # Highest frozen load is the primary capacity evidence. Do not select a
        # more favorable lower-load result after seeing two-replica outcomes.
        primary = next(p for p in model_pairs if p['load_level'] == 'high')
        baseline = condition_map[(name, 1, 'high')]
        full = primary['all_three_pairs_eligible'] and primary['baseline_capacity_pressure_observed']
        gain = primary['paired_ratios']['throughput_gain']['mean']
        lo, hi = plan['interpretation']['approximate_linear_gain_interval']
        if full and gain is not None:
            conclusion = ('approximately supported' if lo <= gain <= hi else 'under-estimates real scaling' if gain > hi
                          else 'over-estimates real scaling' if gain > plan['interpretation']['clear_usable_gain_min'] else 'not supported')
            assessment = 'observed_high_load_window_throughput_under_baseline_pressure'
        else:
            conclusion = 'not supported'
            assessment = 'insufficient_comparable_high_load_evidence; no linear-scaling claim can be supported'
        observed_gain = gain if full else None
        model_results.append(dict(model_id=name, role=model['role'], task=model['task'], linear_surrogate_assessment=conclusion,
            assessment_basis=assessment, primary_load_level='high', exploratory_g2=observed_gain,
            exploratory_gamma=ratio(observed_gain, 2.), usable_capacity_gain_observed=bool(usable),
            usable_gain_load_levels=[p['load_level'] for p in usable], high_load_all_three_pairs_eligible=primary['all_three_pairs_eligible'],
            exact_max_capacity_claim=False, formula_changed=False,
            caution='g2 describes within-window completed-throughput gain at a fixed stressed load; it is not an identified exact capacity ratio. Low-load ratios are arrival-censored; 30 s tests do not establish long-duration sustainability.'))
        regions = []
        for count in (1, 2):
            strict, recovered, unstable, unavailable = [], [], [], []
            for level in ('low', 'medium', 'high'):
                c = condition_map[(name, count, level)]
                row = dict(load_level=level, offered_load_rps=c['offered_load_rps'],
                           observed_throughput_rps=c['metrics']['completed_throughput_rps'], stability=c['stability_outcomes'],
                           offered_end_backlog_requests=c['backlog_metrics']['offered_end_backlog_requests'],
                           mean_host_schedule_delay_ms=c['metrics']['mean_schedule_delay_ms'],
                           sustainable_capacity_proven_by_recovery=False)
                if not c['all_three_valid'] or not c['ratios_eligible_for_capacity_claim']:
                    unavailable.append(row)
                elif c['stability_outcomes'].get('unstable'):
                    unstable.append(row)
                elif c['stability_outcomes'].get('steady') == 3:
                    strict.append(row)
                else:
                    recovered.append(row)
            regions.append(dict(replica_count=count, tested_steady_loads=strict,
                tested_recoverable_congested_loads=recovered, tested_unstable_loads=unstable,
                recovery_note='Recoverable means this finite burst drained, not that its offered rate is sustainable. Host pipe backpressure can flatten submitted-backlog growth while offered backlog continues growing; retain original DQA label and inspect offered backlog, schedule delay and within-window throughput.',
                incomplete_or_failed_condition_evidence=unavailable,
                highest_tested_steady_load_rps=max((r['offered_load_rps'] for r in strict), default=None),
                observed_range_note='Only the explicitly tested points, not an exact maximum or proof of every intermediate load.'))
        observations.append(dict(model_id=name, task=model['task'], regions=regions))
    gain_models = [m for m in model_results if m['usable_capacity_gain_observed']]
    qwen_gain = any(m['model_id'] == 'qwen' for m in gain_models)
    specialist_gain = any(m['role'] != 'generalist' for m in gain_models)
    if qwen_gain and specialist_gain and len(gain_models) == len(plan['models']):
        gate = 'SUPPORTED'
    elif gain_models:
        gate = 'PARTIAL'
    else:
        gate = 'NOT SUPPORTED'
    validation = dict(module='7C-1', decision_gate=gate, model_assessments=model_results,
        gate_reason='Clear repeated capacity-pressure throughput gain exists for ' + (', '.join(m['model_id'] for m in gain_models) or 'no selected model') + '.',
        decision_gate_scope='Repeated usable capacity-pressure throughput gain direction, not uniform 2x magnitude.',
        linear_scaling_support_across_models={m['model_id']: m['linear_surrogate_assessment'] for m in model_results},
        requirement_interpretation='SUPPORTED requires Generalist plus specialist gain and all selected models meeting the frozen >10% gain rule; PARTIAL applies when selected models differ in meeting that rule. Linear magnitude is assessed separately for each model.',
        old_planning_surrogate='D_m <= rho_max x_m', unchanged=True,
        formula_note='Integer formulation supports replica counts under a linear analytical surrogate; real same-GPU scaling is model/task/load dependent. No 7B constraint or predictor has been updated.',
        experiment_limit='Three offered loads, three 30-second repetitions, one representative task per model, independent FP/native-precision workers on one GPU. No generalization to other tasks, all models, batching or more replicas.',
        unstable_and_failure_outcomes_retained=True)
    write(OUT / 'replica_scaling_metrics.json', dict(schema_version=1, frozen_plan_sha256=reference(OUT / 'replica_scaling_plan.json')['sha256'],
        conditions=conditions, scaling_comparisons=pairs, physical_runs=ledger.get('physical_runs'), valid_full_runs=ledger.get('valid_full_runs'),
        aggregation='Per-repetition arithmetic means and sample std; paired ratios and ratio of means both saved. Missing/failed outcomes not assigned zero or synthetic values.',
        throughput_definition=plan['metrics_definitions']['primary_throughput']))
    memory_diagnostics = []
    for model in plan['models']:
        name = model['model_id']
        for level in ('low', 'medium', 'high'):
            memory_a = next(m for m in memories if (m['model_id'], m['replica_count'], m['load_level']) == (name, 1, level))
            memory_b = next(m for m in memories if (m['model_id'], m['replica_count'], m['load_level']) == (name, 2, level))
            v1 = memory_a['measurements']['idle_before_workload_owned_mib']['mean']
            v2 = memory_b['measurements']['idle_before_workload_owned_mib']['mean']
            peak2 = memory_b['measurements']['peak_owned_mib']['mean']
            worst_idle2 = memory_b['measurements']['idle_before_workload_owned_mib']['maximum']
            worst_peak2 = memory_b['measurements']['peak_owned_mib']['maximum']
            ratio_idle = ratio(v2, v1)
            additive = 2 * model['measured_idle_vram_mib']
            memory_diagnostics.append(dict(model_id=name, load_level=level, actual_idle_v1_mib=v1, actual_idle_v2_mib=v2,
                actual_idle_v2_minus_2v1_mib=v2-2*v1 if None not in (v1,v2) else None,
                deviation_from_2v1_fraction=ratio(v2,2*v1)-1 if None not in (v1,v2) and v1>0 else None,
                memory_classification='pending' if ratio_idle is None else 'approximately additive (within 5%)' if 1.9 <= ratio_idle <= 2.1 else 'sublinear observed residency' if ratio_idle<1.9 else 'superlinear observed residency',
                worst_observed_idle2_mib=worst_idle2, worst_observed_peak2_mib=worst_peak2,
                worst_observed_idle2_minus_frozen_estimate_mib=worst_idle2-additive if worst_idle2 is not None else None,
                worst_observed_peak2_minus_frozen_estimate_mib=worst_peak2-additive if worst_peak2 is not None else None,
                frozen_additive_idle_estimate_mib=additive, actual_idle_minus_frozen_estimate_mib=v2-additive if v2 is not None else None,
                observed_peak2_minus_frozen_idle_estimate_mib=peak2-additive if peak2 is not None else None,
                no_shared_weight_or_library_cause_inferred=True))
    write(OUT / 'memory_scaling.json', dict(conditions=memories, diagnostics=memory_diagnostics,
        paired_scaling=[dict(model_id=p['model_id'], load_level=p['load_level'],
            memory_ratio=p['paired_ratios']['memory_ratio'], idle_memory_increase_mib=p['paired_ratios']['idle_memory_increase_mib'],
            peak_memory_ratio=p['paired_ratios']['peak_memory_ratio']) for p in pairs],
        interpretation='Independent process CUDA residency; ratio below two alone does not prove shared model weights or libraries. Runtime caches, algorithm workspaces and input history may differ. Peak measured at 0.5 s resolution.',
        additive_planning_estimate_mib={m['model_id']: {'one': m['measured_idle_vram_mib'], 'two': 2 * m['measured_idle_vram_mib']} for m in plan['models']}))
    write(OUT / 'capacity_observations.json', dict(models=observations, exact_maximum_estimated=False,
        sustainable_definition=plan['metrics_definitions']['sustainable']))
    write(OUT / 'planning_surrogate_validation.json', validation)
    print(__import__('json').dumps(validation), flush=True)

if __name__ == '__main__':
    analyze()
