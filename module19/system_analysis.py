"""Request-level system analysis of the existing frozen residency ablations.

This post-measurement extension never changes serving or fits capacity. System
P95 is a percentile of individual completed request E2Es, not task percentiles.
"""
from collections import Counter
import numpy as np
from .common import *
from .metrics import summarize
from .completion_audit import signature
from .finalize import effect

TASKS = ['text', 'image', 'ocr', 'asr', 'lid', 'visual_qa']
METRICS = ['target_mean_ms', 'target_P95_ms', 'system_request_pooled_mean_ms',
           'system_request_pooled_P95_ms', 'offered_QPS_weighted_task_mean_ms',
           'system_within_window_completed_QPS', 'observed_safe_memory_mib',
           'GPU_peak_mib']


def distribution(records):
    done = [r for r in records if r['status'] == 'completed']
    values = [r['e2e_latency_ms'] for r in done]
    for r in done:
        assert abs(r['e2e_latency_ms'] -
                   (r['completion_time_ns'] - r['scheduled_time_ns']) / 1e6) < 1e-6
    return (done, statistics.mean(values) if values else None,
            float(np.percentile(values, 95)) if values else None)


def quality(records):
    responses = [r['worker_response'] for r in records if r['status'] == 'completed']
    scored = [v['correct'] for v in responses if isinstance(v.get('correct'), bool)]
    word_scored = [v for v in responses if 'reference_words' in v and 'word_errors' in v]
    words = sum(v['reference_words'] for v in word_scored)
    return {'quality_scored_requests': len(scored), 'quality_correct_requests': sum(scored),
            'observed_accuracy_completed_sample': statistics.mean(scored) if scored else None,
            'word_scored_requests': len(word_scored), 'reference_words': words,
            'word_errors': sum(v['word_errors'] for v in word_scored),
            'observed_corpus_WER_completed_sample': sum(v['word_errors'] for v in word_scored) / words if words else None,
            'invalid_closed_set_outputs': sum(v.get('valid_closed_set_output') is False for v in responses),
            'quality_interpretation': 'Frozen quality admission is audited separately. Small serving cohorts are descriptive quality checks, not a replacement admission dataset or a new SLO.'}


def run_statistics(raw, config, case, arm):
    row, _, workers, _, memory = summarize(raw, config, case)
    phase = raw['phases']['measurement']
    records = phase['request_records']
    window = phase['window']
    duration, end = window['duration_seconds'], window['end_time_ns']
    rates = raw['slot']['lambda_rps']
    catalog = read(case['catalog']['path'])
    admitted = all(r['task'] in catalog['models'][raw['replica_to_family'][r['model']]]['supported_tasks'] for r in records)
    done, mean, p95 = distribution(records)
    per_task = []
    for task in TASKS:
        cohort = [r for r in records if r['task'] == task]
        td, tm, tp95 = distribution(cohort)
        per_task.append({'case_id': case['case_id'], 'run_id': row['run_id'],
                         'arm': arm, 'repetition': raw['slot']['repetition'], 'task': task,
                         'source_task_present': task in rates,
                         'frozen_offered_QPS': rates.get(task, 0.), 'offered': len(cohort),
                         'host_send_start_count_NOT_worker_ACK': sum(r.get('actual_submit_time_ns') is not None for r in cohort),
                         'completed': len(td), 'outcome_counts': dict(Counter(r['status'] for r in cohort)),
                         'completed_within_window': sum(r['completion_time_ns'] <= end for r in td),
                         'completed_within_window_QPS': sum(r['completion_time_ns'] <= end for r in td) / duration,
                         'mean_E2E_ms': tm, 'P95_E2E_ms': tp95,
                         'mean_execution_service_ms': statistics.mean(r['worker_latency_ms'] for r in td) if td else None,
                         'mean_queue_ms': statistics.mean(r['worker_queue_wait_ms'] for r in td) if td else None,
                         'censored_by_incomplete_requests': len(td) < len(cohort),
                         **quality(cohort)})
    by_task = {r['task']: r for r in per_task}
    weighted = math.fsum(rate * by_task[task]['mean_E2E_ms'] for task, rate in rates.items()) / math.fsum(rates.values()) if all(by_task[task]['mean_E2E_ms'] is not None for task in rates) else None
    quality_faults = sum(v['invalid_closed_set_outputs'] for v in per_task)
    good = bool(row['valid_session'] and row['memory_passed'] and len(done) == len(records)
                and row['system_stability'] != 'unstable'
                and not any(v['worker_stability'] == 'unstable' for v in workers)
                and admitted and quality_faults == 0)
    target = by_task[case['target_task']]
    result = {'case_id': case['case_id'], 'run_id': row['run_id'], 'arm': arm,
              'repetition': raw['slot']['repetition'], 'seed': raw['slot']['seed'],
              'source_module': case['source_module'], 'task_count': len(rates),
              'frozen_all_task_QPS': rates, 'actual_deployment': row['deployment_actual'],
              'theoretically_legal': config['operationally_legal'],
              'diagnostic_only': not config['operationally_legal'],
              'offered': len(records), 'completed': len(done),
              'completion_ratio': len(done) / len(records) if records else None,
              'system_request_pooled_mean_ms': mean, 'system_request_pooled_P95_ms': p95,
              'offered_QPS_weighted_task_mean_ms': weighted,
              'target_mean_ms': target['mean_E2E_ms'], 'target_P95_ms': target['P95_E2E_ms'],
              'system_within_window_completed_QPS': sum(r['completion_time_ns'] <= end for r in done) / duration,
              'system_cohort_completed_QPS_including_drain': len(done) / (duration + window['drain_seconds']),
              'completed_cohort_divided_by_window_QPS': len(done) / duration,
              'window_seconds': duration, 'drain_seconds': window['drain_seconds'],
              'system_stability': row['system_stability'], 'worker_nonsteady': row['worker_nonsteady'],
              'end_backlog_requests': row['end_backlog_requests'],
              'final_backlog_after_drain': row['final_backlog_after_drain'],
              'tail_backlog_slope_rps': row['tail_backlog_slope_rps'],
              'observed_safe_memory_mib': row['observed_safe_memory_mib'],
              'GPU_peak_mib': row['GPU_peak_mib'], 'memory_budget_mib': case['budget_mib'],
              'memory_passed': row['memory_passed'],
              'CPU_predicted_full_system_J_ms': raw['slot']['CPU_J_seconds'] * 1000,
              'predicted_safe_memory_mib': raw['slot']['CPU_safe_memory_mib'],
              'quality_admission_all_observed_paths': admitted,
              'invalid_closed_set_outputs': quality_faults, 'performance_certificate_eligible': good,
              'all_workers_steady': not row['worker_nonsteady'],
              'censored_by_incomplete_requests': len(done) < len(records),
              'E2E_population': 'All completed requests scheduled during this run measurement window, including their drain completions; unfinished requests explicitly retained as censoring.',
              'weighted_mean_population': 'Sum(frozen offered QPS_k * completed-cohort task mean_k) / sum(frozen offered QPS). Missing task mean yields null; incomplete tasks flagged, weights never renormalized by completions.',
              'safe_memory_semantics': 'max(safe_factor * postwarm idle mean, safe_factor * postmeasurement idle mean); peak GPU reported separately.',
              'worker_backlog_scope': 'Host-send-start pending cohorts; not exact worker queue acceptance telemetry.',
              'raw': ref(OUT / 'raw_runs' / (row['run_id'] + '.json'))}
    for w in workers:
        w['arm'] = arm
        w['analysis_scope'] = 'FIXED_RESIDENCY_SYSTEM_ABLATION'
    return result, per_task, workers


def positive(stats):
    return stats['evidence'] == 'CONSISTENT_POSITIVE_NOMINAL_95CI'


def selected_comparisons(f):
    """Include independently solved arms only when residency/background match.

    Preserve their deployment-policy experiment label. In the frozen LID MID
    case, independent solutions happen to realize a pure target route contrast.
    Duplicate physical arms are analyzed once, never as independent evidence.
    """
    configs = {c['config_id']: c for c in f['configurations']}
    selected = [dict(c, analysis_selection='EXPLICIT_FROZEN_RESIDENCY_ABLATION')
                for c in f['comparisons'] if c['kind'] == 'FIXED_RESIDENCY_ROUTING_ABLATION']
    identities = {(c['case_id'], c['proposed_id'], c['control_id']) for c in selected}
    for c in f['comparisons']:
        if c['kind'] != 'DEPLOYMENT_STRATEGY' or c['control_label'] != 'BASELINE_B_S_ORIENTED' or not c.get('control_id'):
            continue
        identity = (c['case_id'], c['proposed_id'], c['control_id'])
        if identity in identities:
            continue
        p, b = configs[c['proposed_id']], configs[c['control_id']]
        target = p['target_task']
        catalog = read(p['catalog']['path']) if 'catalog' in p else read(next(x for x in f['cases'] if x['case_id'] == c['case_id'])['catalog']['path'])
        roles = catalog['models']
        target_roles = {roles[m]['role'] for m, v in p['routing'][target].items() if v > 0}
        control_roles = {roles[m]['role'] for m, v in b['routing'][target].items() if v > 0}
        if p['deployment'] != b['deployment'] or p['rates'] != b['rates'] or target_roles != {'generalist', 'specialist'} or control_roles != {'specialist'}:
            continue
        if any(p['routing'][t] != b['routing'][t] for t in p['rates'] if t != target):
            continue
        selected.append({**c, 'same_residency': True, 'non_target_routes_unchanged': True,
                         'diagnostic_only': not c['control_operationally_legal'],
                         'analysis_selection': 'INDEPENDENT_POLICIES_REALIZED_IDENTICAL_RESIDENCY_AND_BACKGROUND; original deployment-strategy experiment kind retained'})
        identities.add(identity)
    return selected


def main():
    f = read(OUT / 'frozen_benchmark_manifest.json')
    verify(f['implementation'] + f['reused_implementation'] + [f['CPU'], f['input_protocol']])
    cases = {c['case_id']: c for c in f['cases']}
    configs = {c['config_id']: c for c in f['configurations']}
    runs, tasks, workers, summaries, pair_rows, task_pairs = [], [], [], [], [], []
    for comparison in selected_comparisons(f):
        case = cases[comparison['case_id']]
        pairs, observed_pairs = [], []
        for rep in [1, 2, 3]:
            arms = {}
            for arm, key in [('GS', 'proposed_id'), ('S_ONLY', 'control_id')]:
                ident = comparison[key]
                path = OUT / 'raw_runs' / (ident + '_rep' + str(rep) + '.json')
                if not path.exists():
                    continue
                raw = read(path)
                r, tr, wr = run_statistics(raw, configs[ident], case, arm)
                runs.append(r); tasks += tr; workers += wr
                arms[arm] = (r, tr, raw)
            if len(arms) != 2:
                continue
            p, pt, pr = arms['GS']; b, bt, br = arms['S_ONLY']
            assert pr['slot']['deployment'] == br['slot']['deployment']
            assert p['actual_deployment'] == b['actual_deployment']
            assert pr['slot']['lambda_rps'] == br['slot']['lambda_rps']
            assert all(pr['slot']['routing'][t] == br['slot']['routing'][t]
                       for t in case['rates'] if t != case['target_task'])
            same_samples = signature(pr) == signature(br)
            eligible = p['performance_certificate_eligible'] and b['performance_certificate_eligible'] and same_samples
            observed_pairs.append((p, b))
            if eligible:
                pairs.append((p, b))
            delta = {metric: (p[metric] - b[metric]) if metric == 'system_within_window_completed_QPS'
                     else (b[metric] - p[metric]) for metric in METRICS
                     if p[metric] is not None and b[metric] is not None}
            prediction = b['CPU_predicted_full_system_J_ms'] - p['CPU_predicted_full_system_J_ms']
            pair_rows.append({'case_id': case['case_id'], 'repetition': rep,
                              'GS_run': p['run_id'], 'S_only_run': b['run_id'],
                              'same_actual_deployment': True, 'same_all_task_sample_schedule': same_samples,
                              'non_target_frozen_routes_unchanged': True,
                              'performance_certificate_eligible': eligible,
                              'legal_baseline_pair': comparison['control_operationally_legal'],
                              'CPU_predicted_J_improvement_ms_positive_GS_better': prediction,
                              'predicted_vs_observed_mean_direction_agree': prediction * delta['system_request_pooled_mean_ms'] > 0 if eligible else None,
                              'predicted_vs_QPS_weighted_mean_direction_agree': prediction * delta['offered_QPS_weighted_task_mean_ms'] > 0 if eligible else None,
                              'GS_system_stability': p['system_stability'], 'S_only_system_stability': b['system_stability'],
                              'GS_worker_nonsteady': p['worker_nonsteady'], 'S_only_worker_nonsteady': b['worker_nonsteady'],
                              'GS_end_backlog': p['end_backlog_requests'], 'S_only_end_backlog': b['end_backlog_requests'],
                              'observed_effects_positive_GS_better': delta})
            for task in TASKS:
                a = next(r for r in pt if r['task'] == task); z = next(r for r in bt if r['task'] == task)
                task_pairs.append({'case_id': case['case_id'], 'repetition': rep, 'task': task,
                                   'source_task_present': a['source_task_present'],
                                   'performance_certificate_eligible': eligible,
                                   'GS_offered': a['offered'], 'S_only_offered': z['offered'],
                                   'GS_completed': a['completed'], 'S_only_completed': z['completed'],
                                   'GS_mean_ms': a['mean_E2E_ms'], 'S_only_mean_ms': z['mean_E2E_ms'],
                                   'GS_P95_ms': a['P95_E2E_ms'], 'S_only_P95_ms': z['P95_E2E_ms'],
                                   'mean_improvement_ms': z['mean_E2E_ms'] - a['mean_E2E_ms'] if a['mean_E2E_ms'] is not None and z['mean_E2E_ms'] is not None else None,
                                   'P95_improvement_ms': z['P95_E2E_ms'] - a['P95_E2E_ms'] if a['P95_E2E_ms'] is not None and z['P95_E2E_ms'] is not None else None})
        stats = {metric: effect([(p[metric] - b[metric]) if metric == 'system_within_window_completed_QPS'
                                 else (b[metric] - p[metric]) for p, b in pairs]) for metric in METRICS}
        task_stats = {}
        for task in case['rates']:
            rr = [r for r in task_pairs if r['case_id'] == case['case_id'] and r['task'] == task and r['performance_certificate_eligible']]
            task_stats[task] = {k: effect([r[k] for r in rr]) for k in ['mean_improvement_ms', 'P95_improvement_ms']}
        target_benefit = any(positive(stats[k]) for k in ['target_mean_ms', 'target_P95_ms'])
        system_benefit = any(positive(stats[k]) for k in ['system_request_pooled_mean_ms', 'system_request_pooled_P95_ms', 'offered_QPS_weighted_task_mean_ms'])
        recovery = effect([len(b['worker_nonsteady']) - len(p['worker_nonsteady']) for p, b in pairs])
        capacity_benefit = positive(stats['system_within_window_completed_QPS']) or positive(recovery)
        labels = ([name for name, yes in [('TARGET_TASK_BENEFIT', target_benefit),
                                         ('SYSTEM_LEVEL_BENEFIT', system_benefit),
                                         ('CAPACITY_OR_STABILITY_BENEFIT', capacity_benefit)] if yes]
                  if len(pairs) == 3 else ['INSUFFICIENT_EVIDENCE'])
        if not labels:
            labels = ['NO_MEASURABLE_BENEFIT']
        severe = [r['run_id'] for r in runs if r['case_id'] == case['case_id'] and
                  (r['system_stability'] == 'unstable' or any(w['status'] == 'unstable' for w in r['worker_nonsteady'])
                   or r['censored_by_incomplete_requests'] or not r['quality_admission_all_observed_paths']
                   or r['invalid_closed_set_outputs'])]
        summaries.append({**comparison, 'task_count': len(case['rates']), 'source_module': case['source_module'],
                          'classification': labels, 'three_valid_same_seed_pairs': len(pairs) == 3,
                          'observed_pair_count': len(observed_pairs), 'eligible_pair_count': len(pairs),
                          'paired_effects': stats, 'per_task_effects': task_stats,
                          'reduction_in_nonsteady_worker_count': recovery,
                          'negative_latency_metrics': [k for k, v in stats.items() if v['evidence'] == 'CONSISTENT_NEGATIVE_NOMINAL_95CI'],
                          'tasks_with_consistent_latency_degradation': [t for t, z in task_stats.items() if any(v['evidence'] == 'CONSISTENT_NEGATIVE_NOMINAL_95CI' for v in z.values())],
                          'CPU_predicted_system_J_improvement_ms': statistics.mean(b['CPU_predicted_full_system_J_ms'] - p['CPU_predicted_full_system_J_ms'] for p, b in observed_pairs) if observed_pairs else None,
                          'prediction_agree_with_observed_mean_repetitions': sum(p['predicted_vs_observed_mean_direction_agree'] is True for p in pair_rows if p['case_id'] == case['case_id']),
                          'prediction_agree_with_weighted_mean_repetitions': sum(p['predicted_vs_QPS_weighted_mean_direction_agree'] is True for p in pair_rows if p['case_id'] == case['case_id']),
                          'benefit_flags': {'TARGET_TASK_BENEFIT': target_benefit and len(pairs) == 3,
                                            'SYSTEM_LEVEL_BENEFIT': system_benefit and len(pairs) == 3,
                                            'CAPACITY_OR_STABILITY_BENEFIT': capacity_benefit and len(pairs) == 3,
                                            'NO_MEASURABLE_BENEFIT': labels == ['NO_MEASURABLE_BENEFIT'],
                                            'INSUFFICIENT_EVIDENCE': len(pairs) != 3},
                          'unstable_censored_or_quality_fault_runs': severe,
                          'interpretation': 'System benefit and target benefit are separate, potentially conflicting dimensions. A gain in any endpoint is qualified by all losses and worker warnings. Diagnostic illegal controls cannot establish an advantage against a feasible baseline. No defined latency SLO permits a latency-violation claim.'})
    csvsave('routing_system_runs.csv', runs)
    csvsave('routing_system_task_metrics.csv', tasks)
    csvsave('routing_system_worker_backlog.csv', workers)
    csvsave('routing_system_pair_differences.csv', pair_rows)
    csvsave('routing_system_task_pair_differences.csv', task_pairs)
    csvsave('routing_system_comparison.csv', [
        {**{k: v for k, v in s.items() if k not in ['paired_effects', 'per_task_effects']},
         **{metric + '_' + key: z[key] for metric, z in s['paired_effects'].items()
            for key in ['n', 'mean', 'values', 'CI95', 'evidence']}} for s in summaries])
    save('routing_system_analysis.json', {'at_utc': now(), 'post_measurement_user_requested_extension': True,
                                         'no_new_GPU': True, 'no_changes_to_QPS_routes_models_quality_or_history': True,
                                         'classification_rule': 'Existing nominal paired n=3 tCI and all-three-sign rule, extended to system endpoints; unadjusted exploratory comparisons. Incomplete/unstable pairs cannot certify completed-cohort benefits. Metric conflicts remain visible.',
                                         'predicted_J_semantics': 'Frozen Module9 surrogate sum(D_m/(1-rho_m))/sum(offered QPS), seconds per offered request; compare direction to offered-QPS-weighted empirical means, not to an average of task P95s. This surrogate is not an exact E2E or P95 predictor.',
                                         'analyses': summaries, 'source': ref(ROOT / 'module19/system_analysis.py')})
    gate = read(OUT / 'module19_gate.json')
    gate['system_level_routing_analysis'] = ref(OUT / 'routing_system_analysis.json')
    gate['routing_benefit_classifications'] = {s['case_id']: s['classification'] for s in summaries}
    gate['legal_S_only_system_benefit_tasks'] = [s['case_id'] for s in summaries if s['control_operationally_legal'] and 'SYSTEM_LEVEL_BENEFIT' in s['classification']]
    gate['legal_S_only_capacity_or_stability_benefit_tasks'] = [s['case_id'] for s in summaries if s['control_operationally_legal'] and 'CAPACITY_OR_STABILITY_BENEFIT' in s['classification']]
    gate['target_latency_loss_not_used_as_system_benefit_decision'] = True
    save('module19_gate.json', gate)
    report = '\n系统级固定驻留路由补充（按用户追加要求，只重新分析已有日志）：\n\n'
    report += '所有完成请求按实际数量合并得到系统mean和P95；P95来自原始请求分布，绝不平均任务P95。第二种mean按冻结offered QPS加权每任务完成队列mean；不按完成比例重新归一化。完整请求流两者接近/相等，但失败截尾时均须结合未完成数，不能据低延迟宣称收益。ASR原实例是ASR/VQA两任务，另外三例为六任务，未虚构ASR背景。\n\n'
    report += 'LID MID的Proposed与独立求解B恰好得到同一驻留且所有背景路由完全一致，仅目标G/S对比S-only不同，因此也纳入这项系统统计；原实验类型仍记部署策略比较，不伪称为另一个预先强制固定驻留实验。重复物理控制只统计一次。\n\n'
    report += '全系统P95按实际请求数定义，会由高QPS任务主导，不能代表每类背景任务都满足某个tail约束；各任务P95和拥塞必须同时检查。六任务背景中的低QPS任务每run通常只有12或24条请求，其P95/quality估计粒度有限。这里不额外赋予任务权重、不修改原J，也不声称未定义的latency SLO通过。\n\n'
    report += '|case|valid pairs|结论|system mean effect ms|system P95 effect ms|QPS-weighted mean effect ms|throughput effect QPS|\n|---|---:|---|---:|---:|---:|---:|\n'
    def show(v): return '—' if v is None else f'{v:.6f}'
    for s in summaries:
        z = s['paired_effects']
        report += f"|{s['case_id']}|{s['eligible_pair_count']}|{', '.join(s['classification'])}{'（理论非法S-only诊断）' if s['diagnostic_only'] else ''}|{show(z['system_request_pooled_mean_ms']['mean'])}|{show(z['system_request_pooled_P95_ms']['mean'])}|{show(z['offered_QPS_weighted_task_mean_ms']['mean'])}|{show(z['system_within_window_completed_QPS']['mean'])}|\n"
    report += '\n效果正值表示G/S更好；延迟为S-only−GS，吞吐为GS−S-only。下面完整保留每次运行、每任务代价、预测差异与worker风险，系统收益不要求每任务改善。若多个系统指标方向冲突，不能声称全面提升。n3 CI未经多重比较修正且此系统级扩展为追加分析；NO_MEASURABLE_BENEFIT表示未检出收益，并非证明效果恰为零。\n\n'
    for s in summaries:
        cid = s['case_id']; report += f"{cid}：分类 {s['classification']}；S-only理论合法={s['control_operationally_legal']}。\n\n"
        report += '|rep / arm|offered / completed|system mean / P95 ms|offered-QPS mean ms|system window QPS|CPU J ms|safe MiB|system / worker warnings|\n|---|---|---:|---:|---:|---:|---:|---|\n'
        for r in [r for r in runs if r['case_id'] == cid]:
            report += f"|{r['repetition']} / {r['arm']}|{r['offered']} / {r['completed']}|{show(r['system_request_pooled_mean_ms'])} / {show(r['system_request_pooled_P95_ms'])}|{show(r['offered_QPS_weighted_task_mean_ms'])}|{show(r['system_within_window_completed_QPS'])}|{show(r['CPU_predicted_full_system_J_ms'])}|{show(r['observed_safe_memory_mib'])}|{r['system_stability']} / {json.dumps(r['worker_nonsteady'],ensure_ascii=False)}|\n"
        report += '\n|task|3-pair mean effect ms [nominal CI95]|P95 effect ms [nominal CI95]|方向证据|\n|---|---|---|---|\n'
        for t, z in s['per_task_effects'].items():
            a,b=z['mean_improvement_ms'],z['P95_improvement_ms']
            report += f"|{t}|{show(a['mean'])} {a['CI95']}|{show(b['mean'])} {b['CI95']}|{a['evidence']} / {b['evidence']}|\n"
        report += '\n|rep|CPU J预测mean improvement ms|实测system mean improvement ms|实测system P95 improvement ms|实测QPS-weighted mean improvement ms|预测/实测mean方向相符|valid pair|\n|---|---:|---:|---:|---:|---|---|\n'
        for p in [p for p in pair_rows if p['case_id'] == cid]:
            z = p['observed_effects_positive_GS_better']
            report += f"|{p['repetition']}|{show(p['CPU_predicted_J_improvement_ms_positive_GS_better'])}|{show(z.get('system_request_pooled_mean_ms'))}|{show(z.get('system_request_pooled_P95_ms'))}|{show(z.get('offered_QPS_weighted_task_mean_ms'))}|{p['predicted_vs_observed_mean_direction_agree']}|{p['performance_certificate_eligible']}|\n"
        report += '\n三重复有效证据flag：'+str(s['benefit_flags'])+'；CPU J与实测mean方向相符次数='+str(s['prediction_agree_with_observed_mean_repetitions'])+'/'+str(s['eligible_pair_count'])+'，全任务样本/非目标路由/实际驻留同一性见routing_system_pair_differences.csv。\n\n'
        report += '持续积压/截尾/质量fault runs：'+str(s['unstable_censored_or_quality_fault_runs'])+'；一致性延迟退化任务：'+str(s['tasks_with_consistent_latency_degradation'])+'。严重拥塞由原worker/system分类与backlog记录识别；没有新增或倒推延迟SLO。\n\n'
    report += 'quality/capability admission按冻结目录逐请求path核验；各任务correct数、accuracy、ASR corpus WER及无效closed-set输出见routing_system_task_metrics.csv。背景每task仅约12或24请求，不能将小样本accuracy波动直接认定为原始质量SLO失效；也不能以准入合格替代报告实测错误。逐worker队列、end/tail/drain backlog和service/queue严格分开见routing_system_worker_backlog.csv。\n\n'
    report += 'CPU J是原始全系统预测，不是目标任务J或真实延迟上界；逐rep报告预测/实测方向是否相符，方向错误完整保留，没有根据结果修路由或加干扰系数。系统级结论优先于仅凭目标任务延迟的整体判断，目标代价和其他任务收益/损失均独立可见。\n'
    path = OUT / 'module19_summary.md'
    path.write_text(path.read_text().split('\n系统级固定驻留路由补充（按用户追加要求')[0] + report)
    print('SYSTEM_ROUTING_ANALYSIS', {s['case_id']: s['classification'] for s in summaries})


if __name__ == '__main__':
    main()
