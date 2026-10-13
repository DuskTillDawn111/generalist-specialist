"""Describe the completed negative exploration without changing selection."""
from .common import *


def main():
    result = read(OUT / 'exploratory_result.json')
    assert result['status'] == 'NO_POSITIVE_EXPLORATORY_CANDIDATE'
    assert not result['Stage_C_permitted'] and result['selected'] is None
    runs = {}
    for name in result['completed_runs']:
        raw = read(OUT / 'raw_runs' / (name + '.json'))
        value, observation = arithmetic(raw)
        records = raw['phases']['measurement']['request_records']
        paths = []
        for model in raw['slot']['deployment']:
            for task in raw['slot']['lambda_rps']:
                requests = [r for r in records if r['status'] == 'completed'
                            and r['task'] == task and raw['replica_to_family'][r['model']] == model]
                if not requests:
                    continue
                scored = [r['worker_response']['correct'] for r in requests
                          if isinstance(r['worker_response'].get('correct'), bool)]
                paths.append(dict(model=model, task=task, completed=len(requests),
                    service=stats([r['worker_latency_ms'] for r in requests]),
                    queue=stats([r['worker_queue_wait_ms'] for r in requests]),
                    E2E=stats([r['e2e_latency_ms'] for r in requests]),
                    observed_accuracy=sum(scored) / len(scored) if scored else None,
                    accuracy_scored_requests=len(scored), missing_quality_not_imputed=True))
        runs[name] = dict(raw=ref(OUT / 'raw_runs' / (name + '.json')), U=observation['U'],
            offered=observation['offered'], completed=observation['completed'],
            all_offered_completed=observation['offered'] == observation['completed'],
            original_hard_constraint_eligible=observation['eligible'],
            hard_unstable=observation['hard_unstable'], worker_states=observation['workers'],
            safe_memory_mib=observation['safe_memory_mib'], sampled_peak_mib=observation['sampled_peak_mib'],
            per_task=observation['per_task'], paths=paths)
    save('negative_split_attribution.json', dict(at_utc=now(), runs=runs, pairs=result['pairs'],
        status=result['status'], development_only=True, independent_sample_count=0,
        inference='Same-input service and queue observations; not a uniquely identified physical interference mechanism',
        uncertainty='Block sensitivity is conditional on a single matched pair, not an independent confidence interval',
        interpretation='S waiting was not relieved enough to offset G slow paths and background waiting; all three measured delta U are negative'))
    table = ['|LID QPS|G份额|ΔU|mean贡献差|P95贡献差|吞吐贡献差|显存贡献差|条件block范围|',
             '|---:|---:|---:|---:|---:|---:|---:|---|']
    for pair in result['pairs']:
        ci = pair['conditional_block_sensitivity']
        table.append('|{}|{:.0%}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|{:.6f}|[{:.6f}, {:.6f}]|'.format(
            pair['point_id'].split('_')[-1], pair['G_fraction'], pair['delta_U'],
            pair['L_delta'], pair['P_delta'], pair['T_delta'], pair['M_delta'], ci['low'], ci['high']))
    marker = '\n## 完整实测负收益归因与验收\n'
    summary = (OUT / 'module27_summary.md').read_text().split(marker)[0]
    start = summary.index('3. 原因必须')
    end = summary.index('\n4. ', start)
    summary = summary[:start] + ('3. 三项分流均未改善全系统U：S路径服务时间增加，S等待未得到有效缓解，'
        'G慢路径及VQA等待增加；损失主要来自mean/P95，详见下方实测归因。') + summary[end:]
    summary += marker + '\n' + '\n'.join(table) + '\n\n'
    summary += ('三个区间均只是60s block的单次配对条件敏感性；10.5QPS/10%跨零，'
        '其他两项条件范围为负，不能把它们当独立复现CI。全部五次测量完整完成35,343个请求，'
        '每臂600s，全部排空成功且通过原硬稳定性/显存判据；'
        '10.5QPS/20%的Qwen与11.5QPS的Whisper有recoverable_congested记录，不能称为完全无拥塞。\n\n'
        '10.5QPS/10%：同输入Whisper LID服务mean 69.20→74.13ms，等待181.72→179.42ms；'
        'VQA等待6.76→79.67ms，LID整体E2E mean/P95由249.78/797.84→270.96/820.41ms。'
        '卸载几乎未减少S等待，却增加G慢路径与背景等待。\n\n'
        '10.5QPS/20%：同输入S LID服务69.16→78.71ms，等待183.31→180.15ms；'
        'VQA等待6.76→1205.85ms，VQA P95 700.45→4019.95ms，LID P95 797.84→1989.14ms。'
        '效用损失主要是P95；Qwen可恢复拥塞且最终清空，不截尾计算。\n\n'
        '11.5QPS/10%：同输入S LID服务69.16→74.60ms，等待293.84→318.23ms；'
        'VQA等待20.49→161.16ms。S少接请求后的服务及等待仍变慢，不能用固定服务时间假设宣称容量获益。'
        '这些是匹配输入的运行观测，尚不能唯一归因到物理GPU干扰、热状态或CPU竞争中的某一种机制。\n\n'
        '三项实际G份额分别为9.962%、19.811%、9.477%，G/S均实际承担目标请求。'
        '最高safe memory为26960.43MiB，NVML采样峰值24270MiB，均低于B4。'
        '六任务完整mean/P95与条件区间见per_task_latency.csv；'
        '逐模型任务服务/等待/质量输出另见negative_split_attribution.json。'
        '背景任务每臂约102–135请求，P95仍有采样与相关性不确定性。\n\n'
        '原冻结CPU验收脚本的样本字段路径错误及失败日志保留；'
        'post_run_verify.py按原生item.sample.id逐条核验，不修改GPU原始记录或冻结配置，不重跑GPU。'
        'analysis_acceptance_amendment.json记录此CPU修正；最终验收见final_verification.json。'
        '没有正开发候选，独立复现与正式优化器阶段未执行，不能宣称正分流不存在于所有场景，'
        '也不能宣称优化器已成功或漏选。缺少新的、同驻留合法稳定且能同时改善S等待并限制G代价的正收益配对证据。\n')
    (OUT / 'module27_summary.md').write_text(summary)
    print('MODULE27_NEGATIVE_EVIDENCE_REPORTED', flush=True)


if __name__ == '__main__':
    main()
