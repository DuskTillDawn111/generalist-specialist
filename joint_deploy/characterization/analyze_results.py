"""Descriptive contrasts and plots of completed measurements; no fitted model."""

import argparse
import json
from pathlib import Path
import statistics

from .routing_sweep import OUT, WORKLOAD, SHARES, write_json


def analyze(output):
    summary = json.loads((output/'routing_characterization_summary.json').read_text())
    if summary['num_runs'] != 15 or not summary['integrity_passed']:
        raise ValueError('Analysis requires all 15 independently audited runs')
    by_share = {s['specialist_share']:s for s in summary['settings']}
    runs = summary['runs']
    endpoints = {share:{r['seed']:r for r in runs if r['specialist_share']==share} for share in (0.0,1.0)}
    findings = {'num_runs':15, 'overloaded_or_unstable':summary['overloaded_or_unstable'],
                'interpretation':'descriptive paired seed contrasts; no inference of optimality or statistical significance',
                'tasks':{}, 'vqa_long_queue_examples':[], 'conditional_worker_observations':[],
                'paired_adjacent_changes':[]}
    for seed in sorted(endpoints[0.0]):
        paired = {r['specialist_share']:r for r in runs if r['seed']==seed}
        for lower, higher in zip(SHARES, SHARES[1:]):
            for task in WORKLOAD:
                a = paired[lower]['per_task_metrics'][task]
                b = paired[higher]['per_task_metrics'][task]
                findings['paired_adjacent_changes'].append(dict(seed=seed, task=task,
                    lower_share=lower, higher_share=higher,
                    lower_e2e_mean_ms=a['mean_e2e_latency_ms'], higher_e2e_mean_ms=b['mean_e2e_latency_ms'],
                    e2e_mean_delta_ms=b['mean_e2e_latency_ms']-a['mean_e2e_latency_ms']))
    for task in WORKLOAD:
        lo, hi = by_share[0.0]['per_task_metrics'][task], by_share[1.0]['per_task_metrics'][task]
        mean_series = [by_share[s]['per_task_metrics'][task]['mean_e2e_latency_ms']['mean'] for s in SHARES]
        contrasts = []
        for seed in sorted(endpoints[0.0]):
            a, b = (endpoints[share][seed]['per_task_metrics'][task] for share in (0.0,1.0))
            contrasts.append({'seed':seed,
                              'e2e_mean_delta_ms':b['mean_e2e_latency_ms']-a['mean_e2e_latency_ms'],
                              'e2e_p95_delta_ms':b['p95_e2e_latency_ms']-a['p95_e2e_latency_ms']})
        fields = ('mean_e2e_latency_ms','p95_e2e_latency_ms','mean_worker_latency_ms',
                  'mean_schedule_delay_ms','mean_runtime_minus_worker_ms','mean_worker_queue_wait_ms')
        changes = {key:hi[key]['mean']-lo[key]['mean'] for key in fields}
        findings['tasks'][task] = {
            'generalist_first':{k:lo[k] for k in fields}, 'specialist_first':{k:hi[k] for k in fields},
            'endpoint_delta_ms':changes, 'paired_seed_contrasts':contrasts,
            'mean_e2e_reduction_percent':100*(1-hi['mean_e2e_latency_ms']['mean']/lo['mean_e2e_latency_ms']['mean']),
            'mean_e2e_max_min_factor_across_routings':max(mean_series)/min(mean_series),
            'adjacent_share_step_e2e_deltas_ms':[b-a for a,b in zip(mean_series,mean_series[1:])],
            'p95_over_p50_by_share':{str(s):by_share[s]['per_task_metrics'][task]['p95_e2e_latency_ms']['mean']/
                                    by_share[s]['per_task_metrics'][task]['p50_e2e_latency_ms']['mean'] for s in SHARES},
        }
    # Concrete serial-queue evidence: host submission happens during the previous handler.
    for run in runs:
        raw = json.loads(Path(run['result']).read_text())
        conditional = {}
        for task in WORKLOAD:
            for model in raw['deployment']:
                values = [r['worker_latency_ms'] for r in raw['request_records']
                          if r['task']==task and r['model']==model and r['status']=='completed'
                          and r.get('worker_latency_ms') is not None]
                if values:
                    conditional[f'{task}/{model}'] = dict(n=len(values), mean_ms=statistics.mean(values),
                                                         min_ms=min(values), max_ms=max(values))
        findings['conditional_worker_observations'].append(dict(run_id=run['run_id'],
                    specialist_share=run['specialist_share'], service_paths=conditional,
                    qwen_observed_handler_seconds=raw['models']['qwen']['observed_worker_handler_time_ms']/1000))
        qwen = sorted((r for r in raw['request_records'] if r['model']=='qwen' and r['status']=='completed'),
                      key=lambda r:r['worker_response']['start_ns'])
        examples=[]
        for previous, current in zip(qwen,qwen[1:]):
            if current['task'] != 'visual_qa' or current.get('worker_queue_wait_ms') is None:
                continue
            submit=current['actual_submit_unix_ns']
            prev=previous['worker_response']
            if prev['start_ns'] <= submit < prev['end_ns']:
                examples.append({'run_id':run['run_id'],'request_id':current['request_id'],
                                 'preceding_task':previous['task'], 'preceding_request_id':previous['request_id'],
                                 'preceding_worker_latency_ms':previous['worker_latency_ms'],
                                 'vqa_worker_queue_wait_ms':current['worker_queue_wait_ms'],
                                 'vqa_e2e_latency_ms':current['e2e_latency_ms'],
                                 'vqa_worker_latency_ms':current['worker_latency_ms']})
        if examples:
            findings['vqa_long_queue_examples'].append(max(examples,key=lambda e:e['vqa_worker_queue_wait_ms']))
    findings['vqa_long_queue_examples'].sort(key=lambda e:e['vqa_worker_queue_wait_ms'],reverse=True)
    write_json(output/'routing_characterization_findings.json', findings)
    plot(output,summary)
    write_findings(output, summary, findings)
    return findings


def write_findings(output, summary, findings):
    blocks = summary['settings']
    cohort = [s['system_metrics']['achieved_total_throughput_rps']['mean'] for s in blocks]
    in_window = [s['system_metrics']['within_window_completion_throughput_rps']['mean'] for s in blocks]
    lines = ['# Routing characterization: research observations', '',
             '以下均为当前 x_full、固定 5.8 req/s workload、periodic arrivals 和 deterministic routing 下的观测。',
             '三次重复间 std 描述离散程度，不是置信区间；不作统计显著性或系统最优性声明。', '',
             '## 1. 仅改变 routing 是否改变 latency / throughput？', '',
             f'各 routing 的 cohort completion throughput 均值范围为 {min(cohort):.4f}–{max(cohort):.4f} req/s；'
             f'窗口内完成吞吐均值范围为 {min(in_window):.4f}–{max(in_window):.4f} req/s。',
             '当请求全部在 drain 期限前完成时，固定 offered load 会使 cohort throughput 相同，'
             '不能据此推断各 routing 的最大承载能力相同。', '',
             '| Task | Generalist-first E2E mean ± std ms | Specialist-first E2E mean ± std ms | Endpoint reduction | 3 paired seeds with lower E2E |',
             '| --- | ---: | ---: | ---: | ---: |']
    for task,item in findings['tasks'].items():
        a=item['generalist_first']['mean_e2e_latency_ms']
        b=item['specialist_first']['mean_e2e_latency_ms']
        lower=sum(r['e2e_mean_delta_ms']<0 for r in item['paired_seed_contrasts'])
        lines.append(f"| {task} | {a['mean']:.2f} ± {a['std']:.2f} | {b['mean']:.2f} ± {b['std']:.2f} | "
                     f"{item['mean_e2e_reduction_percent']:.1f}% | {lower}/3 |")
    lines += ['', '## 2. 各 task 是否一样敏感？', '',
              '| Task | Across-routing max/min E2E mean factor | Generalist-first P95 mean ± std ms | Specialist-first P95 mean ± std ms |',
              '| --- | ---: | ---: | ---: |']
    for task,item in findings['tasks'].items():
        a=item['generalist_first']['p95_e2e_latency_ms']
        b=item['specialist_first']['p95_e2e_latency_ms']
        lines.append(f"| {task} | {item['mean_e2e_max_min_factor_across_routings']:.2f}× | "
                     f"{a['mean']:.2f} ± {a['std']:.2f} | {b['mean']:.2f} ± {b['std']:.2f} |")
    lines += ['', '## 3. Specialist-first 是否一定更好？', '',
              '上述表格仅比较本次测量的两个端点，并保留三个 paired-seed 的方向。'
              '即使 specialist-first 在某指标上为本实验 measured lowest，也不能称为全系统最优 routing。'
              '吞吐、均值、tail 也不能自动视为同一个目标。', '',
              '## 4. 哪个 latency 分量变化最大？', '',
              '下面是 specialist-first − generalist-first 的均值差；负值表示下降。'
              'runtime−worker 包含 pipe/worker queue、传输与返回开销，原始 worker_queue_wait_ms 可交叉核对。', '',
              '| Task | E2E Δ ms | Worker Δ ms | Schedule Δ ms | Runtime−worker Δ ms | Largest absolute component change |',
              '| --- | ---: | ---: | ---: | ---: | --- |']
    components = {'mean_worker_latency_ms':'worker', 'mean_schedule_delay_ms':'schedule',
                  'mean_runtime_minus_worker_ms':'runtime−worker'}
    for task,item in findings['tasks'].items():
        delta=item['endpoint_delta_ms']
        largest=max(components,key=lambda k:abs(delta[k]))
        lines.append(f"| {task} | {delta['mean_e2e_latency_ms']:.2f} | {delta['mean_worker_latency_ms']:.2f} | "
                     f"{delta['mean_schedule_delay_ms']:.3f} | {delta['mean_runtime_minus_worker_ms']:.2f} | {components[largest]} |")
    lines += ['', '## 5. 非单调、tail 与跨任务影响', '',
              '对相邻 specialist_share 的实测均值作直接比较，以下条目出现了上升（不拟合函数）：', '']
    increases=[]
    for task,item in findings['tasks'].items():
        for i,delta in enumerate(item['adjacent_share_step_e2e_deltas_ms']):
            if delta>0:
                increases.append(f'- {task}: share {SHARES[i]:.2f} → {SHARES[i+1]:.2f}, E2E mean +{delta:.2f} ms。')
    lines += increases or ['- 本次各相邻点均未观察到 E2E mean 上升；这不证明任意 workload 下单调。']
    lines += ['', '三次平均可能掩盖 seed 间不同方向。VQA 的 share 0.50 → 0.75 配对结果：', '',
              '| Seed | E2E mean at 0.50 (ms) | E2E mean at 0.75 (ms) | Delta (ms) |',
              '| --- | ---: | ---: | ---: |']
    for row in findings['paired_adjacent_changes']:
        if row['task']=='visual_qa' and row['lower_share']==0.5:
            lines.append(f"| {row['seed']} | {row['lower_e2e_mean_ms']:.2f} | "
                         f"{row['higher_e2e_mean_ms']:.2f} | {row['e2e_mean_delta_ms']:+.2f} |")
    lines += ['', '变化幅度并不均匀：不能把这五个测量点解释为恒定比例的 latency 改善。'
              'generalist-first 各任务 P95 约 2 s，明显高于均值；specialist-first 的 LID P95 '
              '跨重复 std 较大，tail 仍依赖请求相位和样本。具体数值见上述表和完整汇总。']
    lines += ['', 'VQA 始终全部交给 Qwen，所以其曲线变化不能由“VQA 改用了 specialist”解释。'
              '确定性周期到达、路由节奏、分配给 Qwen 的 ASR 样本长度等都可能影响它遇到的排队。'
              '条件 service-path 样本数/均值及 Qwen handler 总时间另存 findings JSON。', '',
              '原始时间线中的具体例子（VQA 提交时，其前序 Qwen handler 仍在执行）：', '']
    for example in findings['vqa_long_queue_examples'][:3]:
        lines.append(f"- {example['run_id']} / {example['request_id']}: 前序 {example['preceding_task']} "
                     f"handler={example['preceding_worker_latency_ms']:.1f} ms；VQA queue wait="
                     f"{example['vqa_worker_queue_wait_ms']:.1f} ms，handler={example['vqa_worker_latency_ms']:.1f} ms，"
                     f"E2E={example['vqa_e2e_latency_ms']:.1f} ms。")
    lines += ['', '这些时间线支持同一 worker 内任务之间排队影响的解释，不足以识别 GPU 微观干扰机制。'
              'ASR 每轮 n=30、VQA n=50，P99 对极少数请求敏感；完整 P50/P95/P99、std/min/max 均保留，'
              '本次结果不外推到 Poisson arrivals、其他 workload 或其他 deployment。', '',
              '![Measured latency](routing_latency.png)', '',
              '![Measured decomposition](routing_latency_components.png)', '']
    text='\n'.join(lines)
    (output/'routing_characterization_findings.md').write_text(text,encoding='utf-8')
    summary_path=output/'routing_characterization_summary.md'
    marker='<!-- ROUTING_FINDINGS -->'
    original=summary_path.read_text(encoding='utf-8').split(marker)[0].rstrip()
    summary_path.write_text(original+'\n\n'+marker+'\n\n---\n\n'+text,encoding='utf-8')


def plot(output, summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    settings=summary['settings']
    for ax,task in zip(axes.flat,WORKLOAD):
        for metric,label,marker in [('mean_e2e_latency_ms','E2E mean','o'),('p95_e2e_latency_ms','E2E P95','s')]:
            blocks=[s['per_task_metrics'][task][metric] for s in settings]
            ax.errorbar(SHARES,[b['mean'] for b in blocks],yerr=[b['std'] for b in blocks],
                        marker=marker,capsize=3,label=label)
        ax.set_title(task)
        ax.set_xlabel('specialist_share (matrix-generation label)')
        ax.set_ylabel('Latency (ms)')
        ax.set_xticks(SHARES)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Fixed deployment and workload: measured routing characterization\n'
                 '3 repetitions per point; error bars = sample standard deviation')
    fig.savefig(output/'routing_latency.png',dpi=180)
    fig.savefig(output/'routing_latency.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(14,8),constrained_layout=True)
    for ax,task in zip(axes.flat,WORKLOAD):
        for metric,label in [('mean_worker_latency_ms','Worker handler'),
                             ('mean_runtime_minus_worker_ms','Runtime minus worker'),
                             ('mean_schedule_delay_ms','Host schedule delay')]:
            ax.plot(SHARES,[s['per_task_metrics'][task][metric]['mean'] for s in settings],marker='o',label=label)
        ax.set_title(task)
        ax.set_xlabel('specialist_share')
        ax.set_ylabel('Mean component (ms)')
        ax.set_xticks(SHARES)
        ax.grid(alpha=.25)
        ax.legend()
    fig.suptitle('Measured latency decomposition (no prediction model)')
    fig.savefig(output/'routing_latency_components.png',dpi=180)
    fig.savefig(output/'routing_latency_components.pdf')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    analyze(args.output_dir)


if __name__=='__main__':
    main()
