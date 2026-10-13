"""Descriptive completion tables; does not alter frozen tests or decision rules."""
from .common import *


def number(value):
    return None if value is None or value == '' else float(value)


def display(value):
    return '—' if value is None else f'{value:.3f}'


def main():
    frozen = read(OUT / 'frozen_benchmark_manifest.json')
    verify(frozen['implementation'] + frozen['reused_implementation'])
    analyses = read(OUT / 'paired_comparison_analysis.json')
    with (OUT / 'gpu_strategy_comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    with (OUT / 'resource_latency_tradeoff.csv').open() as stream:
        low = list(csv.DictReader(stream))
    physical = read(OUT / 'physical_gpu_rows.json')
    by_run = {r['run_id']: r for r in physical}
    additional = '\n补充量化（沿用GPU前冻结的配对规则；无新配置、阈值或实验）：\n\n'
    for row in low:
        pair = next(a for a in analyses if a['case_id'] == row['case_id'] and
                    a['control_label'] == row['control_strategy'] and
                    a['kind'] in ['DEPLOYMENT_STRATEGY', 'INITIALIZATION_DIAGNOSTIC_COMPARISON'])
        effects = pair['paired_effects']
        row['GPU_peak_saving_mib_mean'] = effects['GPU_peak_mib']['mean']
        row['GPU_peak_saving_pairs_mib'] = effects['GPU_peak_mib']['values']
        row['GPU_peak_saving_nominal_CI95_mib'] = effects['GPU_peak_mib']['CI95']
        row['GPU_peak_saving_consistency'] = effects['GPU_peak_mib']['evidence']
        peak_percent = []
        for rep in [1, 2, 3]:
            p = next((r for r in physical if r['config_id'] == pair['proposed_id'] and r['repetition'] == rep), None)
            b = next((r for r in physical if r['config_id'] == pair['control_id'] and r['repetition'] == rep), None)
            if p and b and all(r['valid_session'] and r['memory_passed'] and
                              r['completion_ratio'] == 1. and r['system_stability'] != 'unstable' and
                              not any(w['status'] == 'unstable' for w in r['worker_nonsteady'])
                              for r in [p, b]):
                peak_percent.append((b['GPU_peak_mib'] - p['GPU_peak_mib']) / b['GPU_peak_mib'] * 100.)
        row['GPU_peak_saving_percent_mean'] = statistics.mean(peak_percent) if peak_percent else None
        row['mean_delay_effect_consistency'] = effects['target_mean_E2E_ms']['evidence']
        row['P95_delay_effect_consistency'] = effects['target_P95_E2E_ms']['evidence']
    csvsave('resource_latency_tradeoff.csv', low)
    additional += '|LOW目标|safe saving MiB / %|GPU peak saving MiB / %|mean cost ms|P95 cost ms|\n|---|---:|---:|---:|---:|\n'
    for row in low:
        if row['control_strategy'] == 'BASELINE_B_S_ORIENTED':
            additional += f"|{row['target_task']}|{display(number(row['observed_safe_saving_mib_mean']))} / {display(number(row['observed_safe_saving_percent_mean']))}|{display(row['GPU_peak_saving_mib_mean'])} / {display(row['GPU_peak_saving_percent_mean'])}|{display(number(row['target_mean_E2E_cost_ms']))}|{display(number(row['target_P95_E2E_cost_ms']))}|\n"
    additional += '\nLID LOW的B还将ASR背景从Qwen转至Medium，这是独立部署优化的真实结果，不能将该差值解释成只改变LID路由。safe memory与采样GPU peak分别报告，未把safe factor乘peak。\n\nMID实测（目标任务；每run mean/P95等权平均，未将三run请求合并）：\n\n|目标|策略|已测重复 / 源|系统steady次数|全worker steady次数|窗口完成QPS|mean ms|P95 ms|\n|---|---|---|---:|---:|---:|---:|---:|\n'
    mid = []
    for task in ['asr', 'text', 'image', 'lid']:
        for strategy in ['PROPOSED', 'BASELINE_B_S_ORIENTED', 'DIAGNOSTIC_V3_INITIALIZATION']:
            group = [r for r in rows if r['case_id'] == task + '_mid' and
                     r['strategy'] == strategy and r['measurement_source'] != 'NOT_RUN']
            if not group:
                continue
            record = {'target_task': task, 'strategy': strategy, 'n': len(group),
                      'measurement_sources': sorted({r['measurement_source'] for r in group}),
                      'system_steady': sum(r['system_stability'] == 'steady' for r in group),
                      'all_workers_steady': sum(not json.loads(r['worker_nonsteady']) for r in group),
                      'configuration_ids': sorted({r['config_id'] for r in group}),
                      'latency_scope': 'Completed requests; failed runs are censored descriptive observations, never an improvement certificate.'}
            for key in ['target_within_window_throughput_rps', 'target_mean_E2E_ms', 'target_P95_E2E_ms']:
                values = [number(r[key]) for r in group if number(r[key]) is not None]
                record[key] = statistics.mean(values) if values else None
            mid.append(record)
            additional += f"|{task}|{strategy}|{len(group)} / {','.join(record['measurement_sources'])}|{record['system_steady']}|{record['all_workers_steady']}|{display(record['target_within_window_throughput_rps'])}|{display(record['target_mean_E2E_ms'])}|{display(record['target_P95_E2E_ms'])}|\n"
    csvsave('mid_descriptive_summary.csv', mid)
    additional += '\n同一个config_id表示共享等价配置，不是额外独立GPU重复。ASR/Image MID历史三次只作单臂描述，不用于证明当前运行时收益。四任务G-only MID/HIGH均无合法CPU计划，因而没有合法GPU G-only变化百分比；这是可行性边界，不是测得的延迟提升。\n\nV3初始化诊断配对（非完整V3算法比较；正差值=初始化诊断−Proposed；吞吐正差值=Proposed−诊断；单位MiB/ms/QPS）：\n\n|case|comparison status|memory effect|mean effect|P95 effect|throughput effect|\n|---|---|---:|---:|---:|---:|\n'
    for a in analyses:
        if a['control_label'] != 'DIAGNOSTIC_V3_INITIALIZATION':
            continue
        z = a['paired_effects']
        additional += f"|{a['case_id']}|{a['status']}|{display(z['observed_safe_memory_mib']['mean'])}|{display(z['target_mean_E2E_ms']['mean'])}|{display(z['target_P95_E2E_ms']['mean'])}|{display(z['target_within_window_throughput_rps']['mean'])}|\n"
    additional += '\n只有预冻结规则给出3次同方向且nominal CI排除0，才标记一致性支持；上表单个均值差本身不构成显著性。全背景任务指标及任何回退见per_task_e2e.csv。\n\nHIGH实际运行状态（不排除失败配置）：\n\n|case / config|run|CPU合法|系统 / worker|目标窗口QPS|offered / completed|end backlog|\n|---|---|---|---|---:|---|---:|\n'
    for row in physical:
        if row['stage'] != 'HIGH':
            continue
        additional += f"|{row['case_id']} / {row['config_id']}|{row['run_id']}|{row['configuration_role']}|{row['system_stability']} / {json.dumps(row['worker_nonsteady'],ensure_ascii=False)}|{display(row['target_within_window_throughput_rps'])}|{row['target_offered']} / {row['target_completed']}|{row['end_backlog_requests']}|\n"
    additional += '\n未提交/accepted host-send-start/完成/窗口吞吐/drain计时见request_accounting.csv；worker真实完成路由见actual_task_worker_routing.csv。host-send-start不是worker ACK，日志不足时不声称精确worker acceptance。未完成请求的执行忙碌比例只具下界意义，GPU utilization另由NVML采样报告。失败run只报告被截尾的完成请求延迟，不能用其较小均值证明性能胜利。\n\n'
    additional += '历史运行时变化见historical_runtime_drift.csv：必须结合same_scheduled_samples及same_task_family_assignment判断，不能把样本差异当纯服务时间变化。ASR LOW新/历史同种子服务时间约3.60×、4.12×、1.30×，没有拟合固定惩罚，也未由失败吞吐推出精确容量。当前benchmark不能证明跨运行状态可迁移的容量模型。\n\n'
    additional += 'Text/Image/LID结论的候选池边界是Module18预定义COMMON_G17，未开展完整3-Generalist池新GPU对照；ASR为Module15原24模型两任务池。没有把两种池/预算/背景直接混合排名，没有增加模型，原V3与Module9–18未修改。\n'
    summary = OUT / 'module19_summary.md'
    base = summary.read_text().split('\n补充量化（沿用GPU前冻结的配对规则；无新配置、阈值或实验）：')[0]
    summary.write_text(base + additional)
    artifact = read(OUT / 'artifact_manifest.json')
    artifact['artifacts'] = [ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json']
    artifact['implementation'] = [ref(p) for p in sorted((ROOT / 'module19').glob('*.py'))]
    artifact['post_analysis_source'] = ref(ROOT / 'module19/report_addendum.py')
    save('artifact_manifest.json', artifact)
    print('DESCRIPTIVE_ADDENDUM_COMPLETE', len(mid), 'MID strategy summaries')


if __name__ == '__main__':
    main()
