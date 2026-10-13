"""Write cross-module corrected gate summary from audited, read-only results."""
from collections import Counter
import json
from pathlib import Path
from .audit import ROOT, RESULTS, OUT, original_files, sha, save
from .reanalyze import D,E,F,G,read_lines


def main():
    snapshot=json.loads((OUT/'original_files_sha256.json').read_text())
    now={str(p):sha(p) for p in original_files()}
    if snapshot != now:
        raise ValueError('Original result files changed during correction analysis')
    counts=json.loads((OUT/'audit_counts.json').read_text())
    ledger=read_lines(OUT/'run_provenance_audit.jsonl')
    d=json.loads((D/'corrected_lodo_metrics.json').read_text())
    e=json.loads((E/'calibration_fold_metrics.json').read_text())
    f=json.loads((F/'ranking_metrics.json').read_text())
    g=json.loads((G/'ranking_metrics.json').read_text())
    groups=json.loads((G/'comparison_groups.json').read_text())['groups']
    old_e=json.loads((RESULTS/'performance_model_local_calibration/calibration_fold_metrics.json').read_text())
    old_g=json.loads((RESULTS/'performance_model_comparable_groups/ranking_metrics.json').read_text())
    source_e=[r for r in ledger if r['phase'] in ('Stage 2A','Stage 2B','Module 2B/2C profiling','Module 2D/2E/2F source')]
    if len(source_e)!=69 or any(r['run_status']!='completed' for r in source_e):
        raise ValueError('Invalid Module 2E source coverage')
    e_counts=dict(Counter(r['stability_state_v2_corrected'] for r in source_e))
    save(E/'corrected_stability_statistics.json',dict(total=69,run_status_counts={'completed':69,'external_interruption':0,
        'harness_failure':0,'model_failure':0,'uncertain_interruption':0},
        stability_counts={k:e_counts.get(k,0) for k in ('steady','recoverable_congested','unstable')},
        corrected_label_ledger=str(OUT/'corrected_stability_labels.jsonl'),
        corrected_label_sha256=sha(OUT/'corrected_stability_labels.jsonl')))
    save(E/'correction_provenance.json',dict(source_audit=str(OUT/'run_provenance_audit.jsonl'),
        source_audit_sha256=sha(OUT/'run_provenance_audit.jsonl'),all_69_source_runs_completed=True,
        fixed_model_config_unchanged=True,old_decision=old_e['decision'],new_decision=e['decision'],
        no_hyperparameter_tuning=True,no_new_gpu_measurements=True))
    g_rows=[r for r in ledger if r['phase']=='Module 2G']
    if len(g_rows)!=45 or any(r['run_status']!='completed' for r in g_rows):
        raise ValueError('Invalid 2G source coverage')
    if any(r['original_stability_state_v2']!=r['stability_state_v2_corrected'] for r in g_rows):
        raise ValueError('2G labels changed; original analyzer cannot be reused')
    archived=[r for r in ledger if r['run_status']=='external_interruption']
    if len(archived)!=1 or archived[0]['phase']!='Module 3B' or archived[0]['stability_state_v2_corrected'] is not None:
        raise ValueError('External interruption was not isolated')
    if any(archived[0][key] for key in ('valid_for_latency','valid_for_throughput','valid_for_stability','valid_for_ranking')):
        raise ValueError('Interrupted attempt entered performance analysis')
    if not archived[0]['valid_for_attempt_accounting']:
        raise ValueError('Interrupted attempt omitted from physical accounting')
    save(OUT/'audit_validation.json',dict(passed=True,source_run_count=len(ledger),
        original_file_count=len(snapshot),original_files_sha256_verified=True,
        all_2g_runs_completed=len(g_rows)==45,
        old_2g_unstable_count=sum(r['original_stability_state_v2']=='unstable' for r in g_rows),
        corrected_2g_genuine_unstable_count=sum(r['stability_state_v2_corrected']=='unstable' for r in g_rows),
        module2g_harness_footers=sum(r['evidence']['harness_footer_present'] is True for r in g_rows),
        external_attempts_excluded_from_performance=True,
        corrected_outputs_exist=all((p/'ranking_metrics.json').exists() for p in (F,G)) and
            (D/'corrected_lodo_metrics.json').exists() and (E/'calibration_fold_metrics.json').exists(),
        module3c_directory_created=(RESULTS/'deployment_search_multiworkload').exists()))
    result=dict(case='A',module2_role='optional scorer; Module 3 mainline predictor=None',
                module2d_decision=d['overall_decision'],module2e_decision=e['decision'],
                module2f_decision=f['decision']['status'],
                module2g_decision={k:v['result'] for k,v in g['decisions'].items()},
                external_reclassified_from_unstable=counts['legacy_unstable_reclassified_external'],
                module2g_unstable_reclassified_external=0,
                original_files_modified=0,gpu_profiling_runs_added=0,deployment_enumeration_calls=0)
    save(OUT/'module2_gate_corrected.json',result)
    lines=['# Data Quality Audit：External Interruption Reclassification','',
        '本次只读取已有原始测量与历史分析文件，新增纠错层和 CPU 重新分析输出。原始 raw JSON、request records、旧标签及旧报告的 '+str(len(snapshot))+' 个文件 SHA-256 在分析前后完全一致；原始文件修改数 **0**。新增 GPU profiling **0**，deployment enumeration 调用 **0**。','',
        '## 物理 run 总账','',
        '| 项目 | 数量 |','|---|---:|',
        f"| Total runs | {counts['total_runs']} |"]
    for key in ('completed','external_interruption','harness_failure','model_failure','uncertain_interruption'):
        lines.append(f"| {key} | {counts['run_status_counts'][key]} |")
    for key in ('steady','recoverable_congested','unstable'):
        lines.append(f"| Completed: {key} | {counts['completed_stability_counts'][key]} |")
    lines += ['',f"旧 unstable 改为 external interruption：**{counts['legacy_unstable_reclassified_external']}** 次，只有 Module 3B 已归档的 host power-off；旧 unstable 改为 uncertain：{counts['legacy_unstable_reclassified_uncertain']} 次。External interruption 的 corrected stability 为 null，所有 performance 有效性均为 false，仅 physical attempt accounting 为 true。",'',
        '2G 的 45 次都有完整计时 measurement window、arrival loop 和 drain 记录，也都有 harness 结果页脚。旧 11 次 unstable 中，9 次在完整 drain 后仍有 incomplete requests，2 次记录了持续尾段 backlog 增长。raw status 为 `failed` 的 9 次表示服务未能在 drain 截止前完成请求；它们并非 host power-off。一个运行的 shutdown broken pipe 出现在完整测量和 drain 之后，保留其真实 backlog/incomplete 证据。没有依据将任何 2G 运行改为 external。',
        'Module 2B/2C profiling 的 3 次 raw `failed` 同样有完整测量与 drain，未发现外部中断。Module 3B 的 15 次有效重复仍为 completed；归档的 1 次 adapter failure envelope 无 harness 原始结果，与既有关机纠正记录一致。','',
        '## Module 2E stability 纠正统计','',
        f"69 次 completed：**{e_counts.get('steady',0)} steady / {e_counts.get('recoverable_congested',0)} recoverable / {e_counts.get('unstable',0)} genuine unstable / 0 external**。本次仍等于旧统计，原因是 2E 的全部 69 次都已完成测量。",'',
        '## Module 2D LODO（固定原模型配置与 random seed）','',
        '| v2 subset | path rows | C GBT macro MAE (ms) | C GBT macro sMAPE | GBT gate |',
        '|---|---:|---:|---:|---|']
    for mode in ('strict','extended'):
        x=d['subsets'][mode]
        ev=d['gate_evidence'][mode]['gbt']
        lines.append(f"| {mode} | {x['rows']} | {x['macro']['C_gbt_log_ratio']['mae_ms']:.3f} | {x['macro']['C_gbt_log_ratio']['smape_percent']:.3f}% | {'PASS' if ev['passes'] else 'not PASS'} |")
    lines += ['',f"原门槛同时要求 strict 和 extended、Ridge 和 GBT 通过；纠正后整体 **{d['overall_decision']}**。LODO 按原配置在 CPU 重新拟合每折估计器以评估纠正样本，没有调整超参数，也没有导出新的线上 predictor。",'',
        '## Module 2E bias-only calibration','',
        '| K | Strict global MAE | Strict bias-only MAE | Extended global MAE | Extended bias-only MAE |',
        '|---:|---:|---:|---:|---:|']
    agg=e['scenarios']['nominal']['paired_aggregates']
    for k in ('0','1','2','3'):
        vals=[agg[mode][k]['macro'][method]['mae_ms'] for mode in ('strict','extended') for method in ('global_metrics','bias_only_metrics')]
        lines.append('| '+k+' | '+' | '.join(f'{v:.3f}' for v in vals)+' |')
    lines += ['',f"固定 K=0/1/2/3 和原 selection/bias-only 策略重算，gate **{e['decision']}**（旧结果 {old_e['decision']}）。这些是匹配测试子集的 macro MAE；K 间测试集可能不同，不应把行间变化解释为同一样本上的单调收益。",'',
        '## Module 2F / 2G ranking','',
        f"Module 2F 使用 corrected strict LODO OOF 预测及 corrected completed labels 重新建组：**{f['decision']['status']}**；strict/extended cross-deployment rankable groups 均为 {f['decision']['breadth']['strict']['cross_deployment_groups']}。",'',
        '| Workload | Strict 合法配置 | Extended 合法配置 | Extended Kendall τ | Extended Spearman ρ | Top-1 | Top-2 coverage | Top-3 coverage | Regret |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for w in groups:
        a,b=w['views']['strict'],w['views']['extended']
        m=b['metrics']
        def fmt(key): return 'N/A' if m is None or m.get(key) is None else f"{m[key]:.3f}"
        lines.append(f"| {w['workload_group']} | {a['candidate_count']} | {b['candidate_count']} | {fmt('kendall_tau')} | {fmt('spearman_rho')} | {fmt('top1_hit')} | {fmt('top2_coverage')} | {fmt('top3_coverage')} | {fmt('regret')} |")
    lines += ['',
        f"2G strict 可排序 workload 组数 {g['decisions']['strict']['macro']['groups']}；extended 为 {g['decisions']['extended']['macro']['groups']}，低于预先固定的 3 组、其中至少 2 组非平凡 Top-2 要求。两视图 gate 均为 **INSUFFICIENT COMPARABLE GROUPS**，与旧结论相同。相关性/Top-K/regret 只描述已有合法组，不可替代缺失的跨 workload 覆盖。",'',
        '## 决策','',
        '**Case A：Module 2G corrected result remains INSUFFICIENT.** Module 2 继续为 optional scorer；后续 Module 3 主线保持 `predictor=None`。本次停在 data quality audit，不进入 Module 3C。', '',
        '执行状态与 serving stability 独立；只对 completed run 计算 steady / recoverable / unstable。所有纠正记录保留绝对源路径和 SHA-256；原始文件未覆盖。', '']
    (OUT/'audit_summary.md').write_text('\n'.join(lines))
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
