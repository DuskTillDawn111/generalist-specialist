"""Self-contained evidence summaries; no measurements or deployment optimization."""
from .common import OUT, frozen_plan, read


def f(value, digits=3):
    return 'pending' if value is None else f'{value:.{digits}f}'


def report():
    plan = frozen_plan()
    metrics = read(OUT / 'replica_scaling_metrics.json')
    memory = read(OUT / 'memory_scaling.json')
    validation = read(OUT / 'planning_surrogate_validation.json')
    audit = read(OUT / 'measurement_validation.json')
    cpu = read(OUT / 'cpu_regression_validation.json')
    regions = read(OUT / 'capacity_observations.json')
    conditions = {(c['model_id'], c['replica_count'], c['load_level']): c for c in metrics['conditions']}
    memories = {(c['model_id'], c['replica_count'], c['load_level']): c for c in memory['conditions']}
    comparisons = {(p['model_id'], p['load_level']): p for p in metrics['scaling_comparisons']}
    value = lambda c, key: c['metrics'][key]['mean']
    mem = lambda c, key: c['measurements'][key]['mean']
    peak = lambda c: c['measurements']['peak_owned_mib']['maximum']
    ratio = lambda p, key: p['paired_ratios'][key]['mean']
    stability = lambda c: ', '.join(f'{key}×{n}' for key, n in c['stability_outcomes'].items())
    lines = ['# Module 7C-1：Real Multi-replica Scaling Validation', '',
        f'**Decision gate: {validation["decision_gate"]}。** {validation["gate_reason"]}', '',
        'SUPPORTED 表示预冻结阈值下的可用 capacity-pressure throughput 增益方向得到支持。2× 线性幅度的证据按模型分开判断：Qwen / Whisper-small 被高估，ResNet50 近似成立。', '',
        f'固定计划完成 {metrics["physical_runs"]} 个真实 runs，其中 {metrics["valid_full_runs"]} 个完整有效执行；每个 condition 的重复/保护停止情况见 measurement_validation.json。',
        f'GPU: {plan["gpu"]["name"]}, usable ledger capacity 49140 MiB, UUID `{plan["gpu"]["uuid"]}`。',
        f'计划 SHA-256: `{audit["frozen_plan_sha256"]}`；首次 worker 启动之前冻结，运行中没有调整负载、任务或预算。', '',
        'Qwen 固定 ASR 单任务，Whisper-small 固定 ASR，ResNet50 固定 Imagenette closed-set image。使用原 Stage-1 各任务前 30 个样本，保留 frozen native adapter 的输入预处理、prompt、输出和 generation limits，按 seed shuffle 后循环。',
        '低/中/高 offered load = 0.45/0.95/2.2 × (1000 / frozen standalone service-demand ms)。每次新建 1 或 2 个独立 CUDA worker；priming 每 replica 一条，固定 low load warmup 5 s，measurement 30 s，bounded drain 60 s，然后卸载并验证释放。两副本 family stream 均匀确定性 round-robin。', '',
        '下表 throughput/idle 是三轮均值，gain/efficiency/ratio 是 paired ratios 的均值，峰值列是三轮最大 sampled peak。独立重复值、sample std、min/max 保存在 JSON。Throughput 为 measurement window 内成功完成请求/s；cohort 全部 drain 完成不等于可持续容量。Mean/P95 E2E 和 queue 是成功 cohort 请求（包括 drain 完成）的统计，始终附带 completion ratio。低负载吞吐受到达量约束，不能用低负载约 1× gain 判断最大容量。', '',
        '| Model/task | Load | Offered/s | T1 | T2 | Gain mean±sample std | Efficiency | E2E mean 1→2 ms | Queue mean 1→2 ms | Completion 1/2 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for model in plan['models']:
        name = model['model_id']
        for level in ('low', 'medium', 'high'):
            a, b = conditions[name, 1, level], conditions[name, 2, level]
            p = comparisons[name, level]
            gain = p['paired_ratios']['throughput_gain']
            lines.append(f'| {name}/{model["task"]} | {level} | {f(model["loads"][level])} | {f(value(a,"completed_throughput_rps"))} | {f(value(b,"completed_throughput_rps"))} | {f(gain["mean"])} ± {f(gain["sample_std"])} | {f(ratio(p,"efficiency"))} | {f(value(a,"mean_e2e_ms"),1)} → {f(value(b,"mean_e2e_ms"),1)} | {f(value(a,"mean_queue_wait_ms"),1)} → {f(value(b,"mean_queue_wait_ms"),1)} | {f(value(a,"completion_ratio"))}/{f(value(b,"completion_ratio"))} |')
    lines += ['', '| Model | Load | P95 E2E 1→2 ms | Worker mean 1→2 ms | Worker interference ratio | Stability 1 / 2 | Valid repeats 1 / 2 |',
              '|---|---|---:|---:|---:|---|---:|']
    for model in plan['models']:
        name = model['model_id']
        for level in ('low', 'medium', 'high'):
            a, b = conditions[name, 1, level], conditions[name, 2, level]
            p = comparisons[name, level]
            lines.append(f'| {name} | {level} | {f(value(a,"p95_e2e_ms"),1)} → {f(value(b,"p95_e2e_ms"),1)} | {f(value(a,"mean_worker_latency_ms"),1)} → {f(value(b,"mean_worker_latency_ms"),1)} | {f(ratio(p,"worker_latency_interference_ratio"))} | {stability(a)} / {stability(b)} | {a["valid_full_repetitions"]}/{b["valid_full_repetitions"]} |')
    decreases = []
    for model in plan['models']:
        for level in ('medium', 'high'):
            a, b = conditions[model['model_id'], 1, level], conditions[model['model_id'], 2, level]
            t1, t2 = value(a, 'completed_throughput_rps'), value(b, 'completed_throughput_rps')
            if t1 is not None and t2 is not None and t1 > 0 and t2 < t1:
                decreases.append(f'{model["model_id"]}/{level}: throughput 下降 {(1-t2/t1)*100:.1f}%')
    lines += ['', '中/高负载三轮均值 throughput 下降：' + ('；'.join(decreases) if decreases else '本次未观察到。')]
    lines += ['', '同卡两个 CUDA 进程的 worker latency 上升是 interference 的直接服务时间证据；还包含输入读取、preprocessing、generation 与输出处理，不能解释为纯 GPU kernel 时间。GPU utilization、memory bandwidth utilization、power 和 per-replica request/throughput 的独立重复值及 std 完整保存在 JSON/raw 中。', '',
              '| Model | Load | Post-load idle 1→2 MiB | Post-warm idle 1→2 MiB | Idle ratio | Idle increment MiB | Worst sampled workload/drain peak 1→2 MiB | Additive frozen estimate for 2 MiB |',
              '|---|---|---:|---:|---:|---:|---:|---:|']
    for model in plan['models']:
        name = model['model_id']
        for level in ('low', 'medium', 'high'):
            a, b = memories[name, 1, level], memories[name, 2, level]
            p = comparisons[name, level]
            lines.append(f'| {name} | {level} | {f(mem(a,"idle_after_load_owned_mib"),0)} → {f(mem(b,"idle_after_load_owned_mib"),0)} | {f(mem(a,"idle_before_workload_owned_mib"),0)} → {f(mem(b,"idle_before_workload_owned_mib"),0)} | {f(ratio(p,"memory_ratio"))} | {f(ratio(p,"idle_memory_increase_mib"),0)} | {f(peak(a),0)} → {f(peak(b),0)} | {f(2*model["measured_idle_vram_mib"],0)} |')
    lines += ['', '与 additive estimate 的差额（使用 high condition，MiB）：', '']
    for item in memory['diagnostics']:
        if item['load_level'] == 'high':
            lines.append(f'- {item["model_id"]}: {item["memory_classification"]}; actual idle V2−2V1={f(item["actual_idle_v2_minus_2v1_mib"],0)}; actual idle V2−frozen 2Vm={f(item["actual_idle_minus_frozen_estimate_mib"],0)}; worst observed idle V2−frozen 2Vm={f(item["worst_observed_idle2_minus_frozen_estimate_mib"],0)}; worst observed peak V2−frozen idle 2Vm={f(item["worst_observed_peak2_minus_frozen_estimate_mib"],0)}.')
    lines += ['', '内存数值是 NVML owned CUDA PID residency 总和；device-wide 值同时保存。Peak 是 0.5 s 采样 high-water mark，可能遗漏更短暂峰值。Idle ratio 小于 2 不自动证明共享模型权重或库，allocator cache / workspace / input history 也会影响；大于 frozen idle additive estimate 的差额也不是更改 7B ledger 的理由。本阶段未修改 planning memory surrogate。', '',
              'High-load backlog / host backpressure evidence（保持原 DQA label）：', '',
              '| Model | Replicas | Submitted backlog at end | Offered backlog at end | Host schedule delay mean ms | Drain mean s |',
              '|---|---:|---:|---:|---:|---:|']
    for model in plan['models']:
        for count in (1, 2):
            c = conditions[model['model_id'], count, 'high']
            lines.append(f'| {model["model_id"]} | {count} | {f(c["backlog_metrics"]["end_backlog_requests"]["mean"],1)} | {f(c["backlog_metrics"]["offered_end_backlog_requests"]["mean"],1)} | {f(value(c,"mean_schedule_delay_ms"),1)} | {f(c["backlog_metrics"]["drain_duration"]["mean"],1)} |')
    lines += ['', '| Model | 7B linear surrogate assessment | Exploratory g(2) | gamma=g(2)/2 | Usable gain observed |', '|---|---|---:|---:|---|']
    for model in validation['model_assessments']:
        lines.append(f'| {model["model_id"]} | {model["linear_surrogate_assessment"]} | {f(model["exploratory_g2"])} | {f(model["exploratory_gamma"])} | {model["usable_capacity_gain_observed"]}: {", ".join(model["usable_gain_load_levels"]) or "none"} |')
    lines += ['', 'g(2)/gamma 是固定高负载测量窗口的探索性 throughput characterization，不是精确容量估计。判定约 2× 的预冻结区间为 1.8–2.2；clear usable gain 需三个可比完整 pairs 全部 >1.10，并观察到单实例 queue/capacity pressure。保护停止、执行失败的 ratio 仅描述，不用于容量证明。', '', 'Observed sustainable region 仅列实际测试点；不声称精确最大 capacity，也不证明长时间稳定性。Recovered congestion 只证明有限 burst 能 drain，不能证明该 offered rate 可持续；stdin backpressure 可能使 submitted backlog 的 tail 变平，offered backlog / host schedule delay 仍在增加。DQA labels 完全沿用，两个 backlog 定义分别报告：', '']
    for model in regions['models']:
        for region in model['regions']:
            rates = lambda field: ', '.join(f(row['offered_load_rps']) for row in region[field]) or 'none'
            lines.append(f'- {model["model_id"]} ×{region["replica_count"]}: steady tested req/s = {rates("tested_steady_loads")}; recovered congestion = {rates("tested_recoverable_congested_loads")}; unstable = {rates("tested_unstable_loads")}; incomplete/failed = {rates("incomplete_or_failed_condition_evidence")}.')
    lines += ['', f'验证：{audit["status"]}；full SHA checks {audit["source_files_full_hash_checked"]} 文件，旧 7B、candidate pool、真实 service-demand catalog、worker/model/input bytes 保持不变；environment matches frozen v2 = {audit["environments_unchanged"]}；CPU regression {cpu["tests_run"]} passed = {cpu["passed"]}。保留四个未重跑的原 synthetic/scalability wrapper 证据，没有重跑旧 campaigns。',
        '未进行 12-model integer deployment search、Generalist-vs-Specialist allocation optimization、routing search、dynamic adaptation、predictor retraining 或 3+ replicas。Module 7B 的 D_m ≤ rho_max x_m 仍是 analytical planning surrogate，未据此次 characterization 自动改写。完成此报告后停止。', '',
        'Artifacts: replica_scaling_plan.json / plan_freeze.json / raw_runs / campaign_ledger.json / replica_scaling_metrics.json / memory_scaling.json / capacity_observations.json / planning_surrogate_validation.json / measurement_validation.json / source_hashes*.json / environment_manifest.json / cpu_regression*.json|log / qwen_replica_analysis.md。']
    (OUT / 'module7c1_summary.md').write_text('\n'.join(lines) + '\n')
    q = ['# Qwen=2 vs Qwen=1：固定 ASR 单任务', '',
         '此结果适用于 frozen Qwen ASR adapter 和 Stage-1 前 30 个 ASR 样本的固定输入流；不能扩展为所有 Qwen tasks/mixes 的容量结论。中/高负载没有因双副本结果而调整。', '']
    for level in ('medium', 'high'):
        a, b = conditions['qwen', 1, level], conditions['qwen', 2, level]
        p = comparisons['qwen', level]
        am, bm = memories['qwen', 1, level], memories['qwen', 2, level]
        direction = lambda left, right, increase: 'pending' if left is None or right is None else ('是' if (right > left if increase else right < left) else '否')
        q += [f'## {level}: offered {f(a["offered_load_rps"])} req/s', '',
            f'三轮均值：吞吐提高 **{direction(value(a,"completed_throughput_rps"),value(b,"completed_throughput_rps"),True)}**；queue 下降 **{direction(value(a,"mean_queue_wait_ms"),value(b,"mean_queue_wait_ms"),False)}**；mean E2E latency 改善 **{direction(value(a,"mean_e2e_ms"),value(b,"mean_e2e_ms"),False)}**。', '',
            f'- Throughput: {f(value(a,"completed_throughput_rps"))} → {f(value(b,"completed_throughput_rps"))} req/s；paired gain {f(ratio(p,"throughput_gain"))}；efficiency {f(ratio(p,"efficiency"))}。',
            f'- Queue mean: {f(value(a,"mean_queue_wait_ms"),1)} → {f(value(b,"mean_queue_wait_ms"),1)} ms；paired queue ratio {f(ratio(p,"queue_wait_ratio"))}。',
            f'- Mean E2E: {f(value(a,"mean_e2e_ms"),1)} → {f(value(b,"mean_e2e_ms"),1)} ms；latency ratio {f(ratio(p,"latency_ratio"))}；P95 E2E {f(value(a,"p95_e2e_ms"),1)} → {f(value(b,"p95_e2e_ms"),1)} ms。',
            f'- Worker latency: {f(value(a,"mean_worker_latency_ms"),1)} → {f(value(b,"mean_worker_latency_ms"),1)} ms；interference ratio {f(ratio(p,"worker_latency_interference_ratio"))}。',
            f'- Post-warm idle VRAM: {f(mem(am,"idle_before_workload_owned_mib"),0)} → {f(mem(bm,"idle_before_workload_owned_mib"),0)} MiB；increment {f(ratio(p,"idle_memory_increase_mib"),0)} MiB；memory ratio {f(ratio(p,"memory_ratio"))}。Observed workload/drain peak: {f(peak(am),0)} → {f(peak(bm),0)} MiB。',
            f'- Completion ratio: {f(value(a,"completion_ratio"))}/{f(value(b,"completion_ratio"))}；stability: {stability(a)} / {stability(b)}；three pairs eligible: {p["all_three_pairs_eligible"]}。', '',
            '吞吐/queue/latency 改变可直接由以上方向判断；低于 offered load 的 within-window throughput 和窗口结束 backlog 不能被 drain 后 100% completion 掩盖。', '']
    assessment = next(m for m in validation['model_assessments'] if m['model_id'] == 'qwen')
    q += [f'Frozen high-load capacity surrogate assessment: **{assessment["linear_surrogate_assessment"]}**；usable capacity gain observed = **{assessment["usable_capacity_gain_observed"]}**。', '',
          '不声称两个 Qwen replicas 提供 2× real capacity；保留 worker latency interference、backlog/stability 与独立重复值。当前未修改 7B formulation，也未据结果执行 Generalist-vs-Specialist allocation search。']
    (OUT / 'qwen_replica_analysis.md').write_text('\n'.join(q) + '\n')
    print(str(OUT / 'module7c1_summary.md'))

if __name__ == '__main__':
    report()
