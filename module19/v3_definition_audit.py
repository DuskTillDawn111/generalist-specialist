"""Correct the dry-run initializer's label without changing any observation."""
from .common import *

OLD = 'BASELINE_C_ORIGINAL_V3'
DIAG = 'DIAGNOSTIC_V3_INITIALIZATION'


def csvread(path):
    # Worker evidence includes full time-series JSON in a single CSV field.
    csv.field_size_limit(100_000_000)
    with path.open() as stream:
        return list(csv.DictReader(stream))


def main():
    correction = read(OUT / 'benchmark_protocol_correction.json')
    verify(correction['controller_implementation'])
    original = csvread(OUT / 'cpu_baseline_comparison.csv')
    save('audit/initial_cpu_comparison_before_V3_definition_correction.json', original)
    cpu = []
    for r in original:
        if r['strategy'] != OLD:
            cpu.append(r)
            continue
        diag = {**r, 'strategy': DIAG,
                'algorithm_comparison_eligibility': 'INITIALIZATION_DIAGNOSTIC_ONLY',
                'full_V3_search_executed': False}
        cpu.append(diag)
        full = {**r, 'strategy': OLD, 'status': 'UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED',
                'feasibility_scope': 'NO_FINAL_FULL_V3_DEPLOYMENT_SOLVED_IN_CPU_ONLY_PROTOCOL',
                'solver_status': 'DRY_RUN_INITIALIZATION_IS_NOT_FULL_SEARCH',
                'initialization_diagnostic_status': r['status'],
                'initialization_diagnostic_strategy': DIAG,
                'algorithm_comparison_eligibility': 'NOT_AVAILABLE',
                'full_V3_search_executed': False}
        for k in ['deployment', 'safe_memory_mib', 'routing', 'rho', 'max_rho',
                  'predicted_J_seconds', 'solve_seconds', 'audit_seconds',
                  'original_gate_rejection', 'audit']:
            full[k] = None
        cpu.append(full)
    csvsave('cpu_baseline_comparison.csv', cpu)
    original_gpu = csvread(OUT / 'gpu_strategy_comparison.csv')
    save('audit/initial_gpu_comparison_before_V3_definition_correction.json', original_gpu)
    gpu = []
    for r in original_gpu:
        if r['strategy'] != OLD:
            gpu.append(r)
            continue
        gpu.append({**r, 'strategy': DIAG,
                    'comparison_eligibility': 'INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3',
                    'full_V3_search_executed': False})
        gpu.append({'case_id': r['case_id'], 'target_task': r['target_task'],
                    'stage': r['stage'], 'scope': r['scope'], 'strategy': OLD,
                    'repetition': r['repetition'], 'measurement_source': 'NOT_RUN',
                    'CPU_status': 'UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED',
                    'comparison_eligibility': 'UNAVAILABLE_FULL_V3',
                    'skip_reason': 'Original V3 requires measured GPU feedback; dry-run exits after initialization.',
                    'full_V3_search_executed': False})
    csvsave('gpu_strategy_comparison.csv', gpu)
    tradeoff = csvread(OUT / 'resource_latency_tradeoff.csv')
    for r in tradeoff:
        if r['control_strategy'] == OLD:
            r['control_strategy'] = DIAG
            r['status'] = 'INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON'
    csvsave('resource_latency_tradeoff.csv', tradeoff)
    analyses = read(OUT / 'paired_comparison_analysis.json')
    save('audit/initial_paired_analyses_before_V3_definition_correction.json', analyses)
    for a in analyses:
        if a['control_label'] == OLD:
            a['control_label'] = DIAG
            a['frozen_original_label'] = OLD
            a['initialization_diagnostic_status'] = a['status']
            a['status'] = 'INITIALIZATION_DIAGNOSTIC_ONLY_NOT_FULL_V3_COMPARISON'
            a['kind'] = 'INITIALIZATION_DIAGNOSTIC_COMPARISON'
            a['full_V3_algorithm_comparison_available'] = False
    save('paired_comparison_analysis.json', analyses)
    csvsave('paired_comparison_summary.csv', [
        {**{k: v for k, v in a.items() if k != 'paired_effects'},
         **{metric + '_' + key: stats[key] for metric, stats in a['paired_effects'].items()
            for key in ['n', 'mean', 'sd', 'CI95', 'evidence']}}
        for a in analyses])
    gate = read(OUT / 'module19_gate.json')
    gate['gate'] = 'BENCHMARK_COMPLETED_WITH_FAILURES_AND_UNVERIFIED_V3'
    gate['V3_definition_correction'] = ref(OUT / 'benchmark_protocol_correction.json')
    gate['full_Original_V3_comparison_available'] = False
    gate['full_Original_V3_CPU_status'] = 'UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED'
    gate['initialization_diagnostic_positive_cases'] = gate.pop('independent_V3_positive_cases')
    gate['initialization_diagnostic_negative_cases'] = gate.pop('independent_V3_negative_cases')
    gate.pop('LOW_vs_V3_identical_realizations', None)
    gate['LOW_vs_V3_initialization_identical_realizations'] = True
    gate['independent_algorithm_contribution'] = 'INSUFFICIENT_EVIDENCE: full Original V3 not executed; seed-only diagnostics cannot benchmark the complete historical optimizer. A/B/Proposed and fixed-residency ablations remain valid, with all negative outcomes retained.'
    gate['full_requested_V3_benchmark_completed'] = False
    gate['user_selected_static_scope_resolution'] = ref(OUT / 'user_v3_scope_resolution.json')
    gate['user_selected_static_benchmark_completed'] = True
    gate['CPU_infeasible_and_unresolved'] = [r for r in cpu if r['strategy'] != DIAG and r['status'] != 'FEASIBLE']
    physical = read(OUT / 'physical_gpu_rows.json')
    excluded = set(correction['no_further_seed_only_configurations'])
    for r in physical:
        if r['config_id'] in excluded:
            r['frozen_configuration_role'] = r['configuration_role']
            r['configuration_role'] = 'V3_INITIALIZATION_DIAGNOSTIC_ONLY'
    save('physical_gpu_rows.json', physical)
    gate['actual_new_configuration_count'] = len({r['config_id'] for r in physical if r['measurement_source'] == 'NEW_MODULE19'})
    gate['fresh_V3_initialization_diagnostic_runs'] = sum(r['measurement_source'] == 'NEW_MODULE19' and r['config_id'] in excluded for r in physical)
    for name in ['physical_gpu_runs.csv', 'per_task_e2e.csv', 'worker_stability.csv',
                 'worker_task_service.csv', 'memory_observations.csv']:
        rows = csvread(OUT / name)
        for r in rows:
            r['algorithm_scope'] = 'V3_INITIALIZATION_DIAGNOSTIC_ONLY' if r['config_id'] in excluded else 'STATIC_FROZEN_BENCHMARK_OR_HISTORICAL_DESCRIPTION'
        csvsave(name, rows)
    save('module19_gate.json', gate)
    summary = (OUT / 'module19_summary.md').read_text()
    summary = summary.replace('原V3', 'V3初始化诊断').replace('相对V3', '相对V3初始化诊断')
    summary = summary.replace('LOW四任务V3/G-only/Proposed', 'LOW四任务V3初始化/G-only/Proposed')
    summary = summary.replace('Baseline C', 'V3初始化诊断')
    summary = summary.replace('CPU48策略独立求解', 'CPU36项正式A/B/Proposed策略独立求解，另12项仅V3初始化诊断')
    summary = summary.replace('BENCHMARK_COMPLETED_WITH_FAILURES_AND_LIMITATIONS', gate['gate'])
    summary = summary.replace('BENCHMARK_COMPLETED_WITH_MIXED_OR_NEGATIVE_EVIDENCE', gate['gate'])
    notice = ('**V3定义修正：原Module19包装错误使用dry_run=True，原engine在初始化/排序后退出，'
              '没有执行原V3的接受/回滚GPU反馈搜索。已测数据仅为初始化诊断，'
              '不能证明Proposed优于或劣于完整原V3。完整Baseline C标为'
              'UNRESOLVED_RUNTIME_FEEDBACK_REQUIRED；不虚构CPU P95或GPU反馈。'
              '其余A/B/Proposed和固定驻留路由消融不受影响。**\n\n'
              '发现后在完整run保存和GPU释放边界停止旧控制器，新控制器仅跳过尚未执行的'
              'V3初始化诊断。所有真实配置/QPS/预算/路由保持原冻结，已测数据未删除或重试。\n\n')
    notice += '用户明确选择保持静态实验范围：完整V3未验证，保留初始化诊断，不开展GPU反馈搜索。见user_v3_scope_resolution.json。\n\n'
    summary = summary.replace('\n\n', '\n\n' + notice, 1)
    (OUT / 'module19_summary.md').write_text(summary)
    save('v3_definition_audit.json', {'at_utc': now(), 'correction': correction,
                                    'full_V3_not_benchmarked': True,
                                    'no_CPU_P95_or_runtime_oracle_invented': True,
                                    'no_historical_source_modified': True,
                                    'source': ref(ROOT / 'module19/v3_definition_audit.py')})
    print('V3_SCOPE_CORRECTED_INITIALIZATION_DIAGNOSTIC_ONLY')


if __name__ == '__main__':
    main()
