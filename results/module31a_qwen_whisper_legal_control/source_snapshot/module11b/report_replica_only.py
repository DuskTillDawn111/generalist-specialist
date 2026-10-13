"""Replica-only closeout requested by the user; no fixed-cost campaign."""
from collections import Counter, defaultdict
from .common import *
from .analyze_replica import calibration
from .replay import gate
from module11.envelope import fixed_lp


def main():
    selected_gate = gate()
    c, hw, roles, base = inputs()
    cal = calibration()
    scaling = read(OUT / 'replica_scaling.json')
    original = {(r['budget'], r['set_id']): r for r in read(OLD / 'windows.json')['rows']}
    windows = read(OUT / 'updated_windows.json')['rows']
    fits = {(r['budget'], r['set_id'], r['mode']): r for r in read(OUT / 'updated_capacity_witnesses.json')['rows']}
    freeze = read(OLD / 'input_freeze.json')
    comparisons = []
    usages = Counter()
    replaced_old_window_usages = Counter()
    used_counts = defaultdict(set)
    for new in windows:
        key = (new['budget'], new['set_id'])
        old = original[key]
        row = {'budget': key[0], 'set_id': key[1]}
        for field in ['alpha_S', 'alpha_1G', 'alpha_2G', 'W_1G', 'W_2G']:
            row['old_' + field] = old[field]
            row['new_' + field] = new[field]
            row['change_' + field] = new[field] - old[field] if new[field] is not None and old[field] is not None else None
        one = fits[key + ('1G',)]
        two = fits[key + ('2G',)]
        new_S = {m: n for m, n in one.get('deployment', {}).items()
                 if roles[m] == 'specialist' and n > base.evidence_bounds[m]}
        new_G = {m: n for m, n in two.get('deployment', {}).items()
                 if roles[m] == 'generalist' and n > base.evidence_bounds[m]}
        usages.update(new_S.keys())
        for m, n in new_S.items():
            used_counts[m].add(n)
        row['newly_measured_S_counts_in_1G_maximizer'] = new_S
        row['newly_measured_G_counts_in_2G_maximizer'] = new_G
        row['new_1G_deployment'] = {m: n for m, n in one.get('deployment', {}).items() if n}
        row['new_2G_deployment'] = {m: n for m, n in two.get('deployment', {}).items() if n}
        old_mid = (old['alpha_1G'] + old['alpha_2G']) / 2 if old['W_2G'] and old['W_2G'] > 0 else None
        mid_witness = fixed_lp(one['deployment'], freeze['background'], one['headroom'], c, roles, cal, '1G', old_mid) if old_mid is not None else None
        row['old_conditional_2G_midpoint_alpha'] = old_mid
        row['old_midpoint_now_1G_feasible'] = mid_witness is not None if old_mid is not None else None
        row['old_midpoint_1G_witness'] = mid_witness
        row['new_SECOND_GENERALIST_CAPACITY_NECESSARY'] = new['SECOND_GENERALIST_CAPACITY_NECESSARY']
        if old_mid is not None and mid_witness is not None:
            replaced_old_window_usages.update(new_S.keys())
        comparisons.append(row)
    csvsave('replica_scaling_impact.csv', comparisons)
    save('replica_scaling_impact.json', {'rows': comparisons,
        'interpretation': 'Measured higher Specialist counts used jointly by updated 1G witnesses; this is not an isolated per-model causal ablation.',
        'contexts_using_new_S_count_in_1G_maximizers': dict(usages),
        'new_S_counts_used': {m: sorted(v) for m, v in used_counts.items()},
        'contexts_using_new_S_count_in_1G_witnesses_covering_old_2G_midpoints': dict(replaced_old_window_usages)})
    positive1 = [w for w in windows if w['W_1G'] is not None and w['W_1G'] > 0]
    positive2 = [w for w in windows if w['SECOND_GENERALIST_CAPACITY_NECESSARY']]
    candidate = next(w for w in windows if w['budget'] == 'B6' and w['set_id'] == 'asr+image+lid')
    max1 = max(positive1, key=lambda w: w['W_1G']) if positive1 else None
    max2 = max(positive2, key=lambda w: w['W_2G']) if positive2 else None
    old_2G_midpoints_covered = sum(r['old_midpoint_now_1G_feasible'] is True for r in comparisons)
    summary = {'scope': 'REPLICA_EVIDENCE_ONLY', 'replica_gate': selected_gate,
        'W1_positive_contexts': len(positive1), 'W2_positive_certified_contexts': len(positive2),
        'max_W1_context': max1, 'max_W2_context': max2,
        'diagnostic_B6_ASR_Image_LID': candidate,
        'old_conditional_2G_midpoints_now_1G_feasible': old_2G_midpoints_covered,
        'new_S_counts_used_in_1G_maximizers': {m: sorted(v) for m, v in used_counts.items()},
        'contexts_using_new_S_count_in_1G_maximizers': dict(usages),
        'terminal_frontiers': {m: {'reason': r['reason'], 'x': r['x']} for m, r in scaling['terminal_frontiers'].items()},
        'fixed_cost_status': 'DEFERRED_BY_USER', 'fixed_cost_GPU_runs': 0,
        'formal_hotspot_GPU_runs': 0, 'Task_Rich_started': False, 'sensitivity_audit_started': False,
        'Module9_unchanged': True, 'old_periodic_and_closed_loop_observations_preserved': True}
    save('module11b_replica_summary.json', summary)
    lines = ['# Module 11B replica evidence 收尾', '',
        f'最终 replica evidence gate：`{selected_gate}`。', '',
        '本次仅完成 Replica Evidence Completion / Closed-loop Capacity Frontier 与 CPU structural capacity audit。fixed-cost / low-QPS GPU campaign 按用户最新指示暂缓，未启动任何 session；14 resident sets、42 次测量的协议、脚本与 CPU checks 均保留，准备文件另存于 `fixed_cost_deferred_archive/`。没有开始正式 multi-hotspot GPU、Task-Rich 或 sensitivity audit。', '',
        '## 实测与终止', '',
        'Closed-loop 只测 replica service capacity；periodic 继续提供 arrival、E2E、latency、stability evidence，旧 raw 未重跑或覆盖。已验证当前 worker 同步串行处理单请求、无跨请求 batching，因此每 replica 最多 1 个 outstanding，完成后立即补发。5 秒 warmup 后，固定 30 秒 completed/elapsed 窗口，drain 不计容量，每个相邻 count 使用 3 组独立配对。', '',
        '新 g(x) = 历史/此前实测 g(x−1) × 三组相邻 count throughput ratio 的 median。所有历史 g/M 点保留，新点只存入本模块 calibration；未外推未知 counts，也未更改 s0 或 Module 9 loader。代表任务的 family scalar 向其它 tasks/混合 runtime 迁移仍需未来真实 workload 验证。', '',
        '| Model | 原缺失 count | 最终实测 bound | 首次真实终止 |', '|---|---:|---:|---|']
    for required in read(OUT / 'measurement_plan.json')['required_initial_frontier']:
        m = required['model']; terminal = scaling['terminal_frontiers'].get(m)
        reason = f"{terminal['reason']} @ {terminal['x']}" if terminal else 'Generalist：仅补齐≤2G所需 point'
        lines.append(f"| {m} | {required['required_x']} | {scaling['evidence_bounds'][m]} | {reason} |")
    lines += ['', 'Specialist 仅沿原 breakpoint-relevant frontier 逐级增加。容量终止要求三组配对均无正增量；mixed-sign 标记为 inconclusive 并继续，未因单次下降停止。低负载终止要求三次均 unstable；memory 终止来自实测 count memory 与最小 background capability cover 的 safe-memory 不可行。达到首次真实终止后，不再将更高 counts 视作人工缺失证据。', '',
        '## 原 11A 与更新后的结构窗口', '',
        '使用原 20 个 hotspot sets × 6 budgets × S/1G/2G，保留原 background、normalized headrooms、quality/capability、safe-memory factor 与全 worker rho≤0.8。没有重新定义 directions，也没有根据 GPU 结果移动热点。', '',
        f'W1>0：{len(positive1)}/120 contexts；max W1={max1["W_1G"] if max1 else 0:.12g}。',
        f'证据闭合且 W2>0：{len(positive2)}/120 contexts；max W2={max2["W_2G"] if max2 else 0:.12g}。']
    if max1:
        lines.append(f'最大 W1：{max1["budget"]} / {max1["set_id"]}，alphaS={max1["alpha_S"]:.12g}，alpha1={max1["alpha_1G"]:.12g}。')
    if max2:
        lines.append(f'最大 W2：{max2["budget"]} / {max2["set_id"]}，alpha1={max2["alpha_1G"]:.12g}，alpha2={max2["alpha_2G"]:.12g}。')
    lines += ['', f'真正 1G infeasible / 2G feasible：{"存在" if positive2 else "没有确认"}。正 W2 区间的 midpoint 必须高于全局 1G MILP upper bound，且存在合法 2G fixed-LP witness；证书记录在 `updated_second_generalist_regions.csv`，完整 deployment/routing/rho/memory 在 `updated_capacity_witnesses.json`。这是实测 g/M 支持的 planning feasibility，不是新的混合 workload 稳定性 GPU 结论。', '',
        f'原 conditional 2G 窗口中有 {old_2G_midpoints_covered} 个 midpoint 现在由合法 1G witness 承载。更新后 1G 最优状态使用的新增 Specialist counts：{dict((m, sorted(v)) for m, v in used_counts.items())}。各 context 的 old/new alpha、W、具体新 counts 与旧 midpoint witness 见 `replica_scaling_impact.csv/json`。这些 counts 是共同参与结构变化的实测证据，不声称某单模型的独立因果贡献。', '',
        f'B6 / ASR+Image+LID：alphaS={candidate["alpha_S"]:.12g}，alpha1={candidate["alpha_1G"]:.12g}，alpha2={candidate["alpha_2G"]:.12g}；W1={candidate["W_1G"]:.12g}，W2={candidate["W_2G"]:.12g}。',
        '该 context 仍有证据闭合的 2G capacity window，可作为未来 GPU 候选。' if candidate['SECOND_GENERALIST_CAPACITY_NECESSARY'] else '不能继续以该 context 的旧 conditional 2G window 为正式 2G capacity benchmark 依据。', '',
        '## 停止边界', '',
        '本次 gate 仅依据 replica evidence 与 capacity structural replay；不对低-QPS Specialist deployment 是否值得作结论。fixed-cost/low-QPS 准备输出为未执行的准备材料，不是测量结果。下一任务 Specialist Efficiency Sensitivity Audit 尚未执行；Module 9、worker/runtime、MPS 状态、历史结果均保留。']
    (OUT / 'module11b_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
