"""Freeze the current efficient-Specialist regime without completing its frontier."""
import shutil
import subprocess
from collections import Counter, defaultdict
from .common import *
from replica_scaling.telemetry import snapshot


def main():
    assert not (OUT / 'module11b_closeout_verification.json').exists()
    gpu = snapshot()
    assert not gpu['processes'], 'Finish any active single measurement before closing'
    assert not subprocess.run(['pgrep', '-af', '[n]vidia-cuda-mps'], capture_output=True, text=True).stdout.strip()
    scaling = read(OUT / 'replica_scaling.json')
    plan = read(OUT / 'measurement_plan.json')
    c, hw, roles, base = inputs()
    raw_refs = []
    groups = defaultdict(list)
    measured = defaultdict(lambda: defaultdict(list))
    low_states = defaultdict(Counter)
    kind_counts = Counter()
    for p in sorted((OUT / 'raw_runs').glob('*.json')):
        raw = read(p); slot = raw['slot']; kind = slot['kind']
        assert raw['gpu_release']['verified'] and raw['valid_session']
        assert slot == read(OUT / 'slots' / f"{slot['run_id']}.json")
        kind_counts[kind] += 1
        if kind in ['replica_profile', 'closed_loop_profile']:
            n = slot['deployment'][slot['model']]
            measured[slot['model']][kind].append(n)
            groups[(slot['model'], slot['required_x'], n, kind)].append(slot['seed'])
            low = raw.get('phases', {}).get('low_load')
            if low:
                low_states[kind][low['stability']['stability_state_v2']] += 1
        if kind == 'closed_loop_profile':
            phase = raw['phases']['closed_loop_capacity']
            assert phase['max_outstanding_per_replica'] == 1
            assert 'stability' not in phase and 'mean_E2E_ms' not in phase
            assert raw['closed_loop_warmup']['all_requests_completed_after_drain']
        raw_refs.append(ref(p))
    assert kind_counts.get('fixed_cost', 0) == 0
    ledger = read(OUT / 'execution_ledger.json')['attempts']
    assert all(r['status'] == 'RETURNED' and r['returncode'] == 0 for r in ledger)
    assert len(ledger) == len(raw_refs)
    seeds = sorted(plan['seeds'])
    complete_groups = []
    incomplete_groups = []
    for (m, frontier, n, kind), observed in sorted(groups.items()):
        entry = {'model': m, 'frontier': frontier, 'replica_count': n,
            'protocol': kind, 'completed_seeds': sorted(observed)}
        if sorted(observed) == seeds:
            complete_groups.append(entry)
        else:
            entry['unrun_seeds'] = [s for s in seeds if s not in observed]
            entry['status'] = 'NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE'
            incomplete_groups.append(entry)
    inventories = []
    for m in c:
        added = [p for p in scaling['new_points'] if p['model'] == m and p['measured_point_added']]
        row = {'model': m, 'role': roles[m],
            'historical_empirical_counts_preserved': sorted(base.capacity_factors[m]),
            'current_periodic_counts_measured': sorted(set(measured[m]['replica_profile'])),
            'current_closed_loop_counts_measured': sorted(set(measured[m]['closed_loop_profile'])),
            'new_g_points': {p['x']: p['g_new'] for p in added},
            'new_g_point_classifications': {p['x']: p['classification'] for p in added},
            'highest_empirical_calibration_count': scaling['evidence_bounds'][m],
            'first_unmeasured_higher_count': scaling['evidence_bounds'][m] + 1,
            'unmeasured_higher_counts_status': 'NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE',
            'absolute_physical_frontier_complete': False,
            'breakpoint_evidence_sufficiency_claimed': False}
        inventories.append(row)
    for m in c:
        for n, value in base.capacity_factors[m].items():
            assert scaling['capacity_factors'][m][str(n)] == value
        for n, value in base.memory_table_mib[m].items():
            assert scaling['memory_table_mib'][m][str(n)] == value
    print(json.dumps({'stage': 'raw_inventory_verified', 'runs': dict(kind_counts),
        'new_g_points': len(scaling['new_points']), 'partial_groups': incomplete_groups}), flush=True)

    # Capture every existing result/diagnostic before creating the closeout.
    frozen_existing = [ref(p) for p in sorted(OUT.rglob('*')) if p.is_file()]
    save('frozen_existing_evidence_manifest.json', {'at_utc': now(),
        'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
        'policy': 'preserve current periodic, closed-loop, calibration and stability data without overwrite or rerun',
        'files': frozen_existing})
    baseline = OUT / 'efficiency_optimal_baseline'
    baseline.mkdir(exist_ok=False)
    copies = []
    for name in ['replica_scaling.json', 'replica_profile_results.csv',
                 'periodic_profile_results.csv', 'closed_loop_profile_results.csv',
                 'replica_point_status.json', 'measurement_plan.json', 'closed_loop_protocol.json']:
        shutil.copy2(OUT / name, baseline / name)
        assert ref(OUT / name)['sha256'] == ref(baseline / name)['sha256']
        copies.append(ref(baseline / name))
    csvsave('replica_evidence_closeout.csv', inventories)
    save('replica_evidence_closeout.json', {'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
        'models': inventories, 'protocol_session_counts': dict(kind_counts),
        'complete_repetition_groups': complete_groups, 'incomplete_repetition_groups': incomplete_groups,
        'low_load_periodic_stability_by_capacity_protocol': {k: dict(v) for k, v in low_states.items()},
        'all_completed_observations_preserved': True,
        'physical_saturation_not_established': True, 'breakpoint_evidence_not_claimed_complete': True})
    save('replica_evidence_gate.json', {'gate': 'UNDETERMINED_REPLICA_EVIDENCE_REQUIRED',
        'execution_status': 'CLOSED_BY_USER_WITH_PARTIAL_FRONTIER',
        'blocking_current_next_stage': False,
        'unmeasured_counts_status': 'NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE',
        'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
        'absolute_frontier_complete': False, 'BREAKPOINT_EVIDENCE_SUFFICIENT_claimed': False,
        'new_W1G_W2G_certification_claimed': False,
        'next_stage': 'CPU-only Specialist Efficiency Sensitivity Audit, not started',
        'formal_GPU_hotspot_not_started': True, 'Module9_unchanged': True})
    logs = [Path('/tmp/module11b_incremental_handoff.log'),
            Path('/tmp/module11b_incremental_handoff_numerical.log')]
    diagnostic_dir = OUT / 'cpu_audit_diagnostics'
    diagnostic_dir.mkdir(exist_ok=True)
    diagnostic_refs = []
    for p in logs:
        if p.exists():
            target = diagnostic_dir / p.name
            assert not target.exists()
            shutil.copy2(p, target)
            diagnostic_refs.append(ref(target))
    save('cpu_audit_closeout_status.json', {'status': 'NOT_COMPLETED_TIE_BREAK_NUMERICAL_FAILURE',
        'new_capacity_window_certification': False,
        'detail': 'Same-formulation CPU replay passed alpha and memory stages but failed replica-count tie-break; tightening HiGHS numerical feasibility checks also did not resolve it. No physical infeasibility conclusion was drawn.',
        'diagnostics': diagnostic_refs, 'Module11A_original_sources_and_results_unchanged': True})
    summary = {'execution_status': 'MODULE11B_CLOSED_BY_USER',
        'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME',
        'periodic_sessions': kind_counts['replica_profile'],
        'closed_loop_sessions': kind_counts['closed_loop_profile'],
        'new_g_points': len(scaling['new_points']),
        'absolute_frontier_complete': False, 'unmeasured_status': 'NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE',
        'partial_repetition_groups': incomplete_groups,
        'fixed_cost_GPU_runs': 0, 'fixed_cost_status': 'DEFERRED_BY_USER',
        'formal_hotspot_GPU_runs': 0, 'Task_Rich_started': False,
        'sensitivity_audit_started': False, 'Module9_unchanged': True,
        'baseline_copies': copies, 'GPU_released': gpu}
    save('efficiency_optimal_specialist_regime_frozen.json', summary)
    lines = ['# Module 11B 简短收尾', '',
        '状态：`MODULE11B_CLOSED_BY_USER`。当前 efficient Specialist 池冻结为 `EFFICIENCY_OPTIMAL_SPECIALIST_REGIME`，作为 efficiency-favorable baseline 保留。', '',
        f'保留 {kind_counts["replica_profile"]} 次 periodic、{kind_counts["closed_loop_profile"]} 次 closed-loop 独立 session，共 {len(scaling["new_points"])} 个新增实测 g(x) 点。所有 session 已正常完成并释放 GPU；未重跑或覆盖已完成数据。Closed-loop 仅用于 service capacity，periodic 保留 arrival/E2E/latency/stability 职责。历史 g/M、s0、Module 9 与 worker/runtime 未修改，MPS 未开启。', '',
        '| Model | 本轮 closed-loop 已测 counts | 新增 g(x) | 尚未测的更高 counts |',
        '|---|---|---|---|']
    for r in inventories:
        points = ', '.join(f'g({x})={value:.6f}' for x, value in r['new_g_points'].items()) or '仅保留历史 points'
        missing = f'{r["first_unmeasured_higher_count"]}及以上'
        lines.append(f'| {r["model"]} | {r["current_closed_loop_counts_measured"]} | {points} | {missing} |')
    lines += ['', '全部未测 higher counts 标记为 `NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE`，不是“不可行”或“已饱和”。完整物理 replica frontier 与 breakpoint evidence 均未假装完成，也不宣称 `BREAKPOINT_EVIDENCE_SUFFICIENT`。Qwen 的既有 count 1/2 points 保留，本轮未新增其 closed-loop 测量。', '',
        'RoBERTa 3-replica frontier 仅完成了对应 2-replica 基线的一次 session；3 replicas 尚未测，未加入 g(3)。这份已完成基线 raw 保留；未执行的 repetitions 同样标为 `NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE`。详见 `replica_evidence_closeout.csv/json`。', '',
        '本次未获得更新 W1G/W2G 的完整 CPU 认证：增量 replay 的 replica-count tie-break 出现数值 infeasible，原 formulation 与更严格数值配置均未完成。日志已归档，未将该问题解释为物理 capacity infeasible；原 Module 11A conditional 结果保持不变。replica evidence gate 保留未定状态，但不阻塞当前下一阶段。', '',
        '完整旧 Specialist frontier 不再继续追踪，因为下一阶段将先用 CPU-only Specialist Efficiency Sensitivity Audit 定位 transition region，再选择真实、更大/更慢且 quality-admitted 的同任务 Specialist，对新模型重新实测 Q、s0、M、g(x)。Counterfactual sensitivity 只定位搜索区间，最终性能结论仍以真实模型为依据。该下一阶段本轮尚未启动。', '',
        'fixed-cost / low-QPS campaign 继续暂停，GPU runs=0；14 resident-set manifest、42 次计划测量的协议、脚本、CPU checks 与准备输出全部归档保留，不作为最终低-QPS deployment 证据。正式 multi-hotspot GPU、Task-Rich、low-QPS campaign 均未启动。', '',
        '现有数据 SHA manifest：`frozen_existing_evidence_manifest.json`；baseline 快照：`efficiency_optimal_baseline/`。']
    (OUT / 'module11b_summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'stage': 'report_written', 'protected_freeze_verification': 'starting'}), flush=True)

    freeze = read(OUT / 'input_freeze.json')
    verify(freeze['protected_references'])
    verify([freeze[k] for k in ['measurement_plan', 'resident_set_manifest', 'samples']])
    verify(read(OUT / 'closed_loop_source_revision_v2.json')['effective_harness'])
    verify(read(OUT / 'periodic_completion_manifest.json')['immutable_raw_refs'])
    verify(frozen_existing)
    verify(copies)
    gpu_final = snapshot()
    assert not gpu_final['processes']
    save('module11b_closeout_verification.json', {'status': 'PASS', 'at_utc': now(),
        'existing_evidence_files_unchanged': len(frozen_existing),
        'protected_files_verified': len(freeze['protected_references']),
        'valid_completed_sessions': len(raw_refs), 'new_g_points': len(scaling['new_points']),
        'historical_g_M_unchanged': True, 'Module9_unchanged': True,
        'periodic_and_closed_loop_metric_roles_preserved': True,
        'unmeasured_counts_honestly_marked': True, 'frontier_completion_not_claimed': True,
        'GPU_released': gpu_final, 'MPS_off': True, 'fixed_cost_GPU_runs': 0,
        'formal_hotspot_GPU_runs': 0, 'Task_Rich_started': False, 'sensitivity_audit_started': False})
    save('module11b_closeout_manifest.json', {'at_utc': now(), 'regime': summary['regime'],
        'existing_evidence_manifest': ref(OUT / 'frozen_existing_evidence_manifest.json'),
        'closeout_outputs': [ref(OUT / name) for name in [
            'replica_evidence_closeout.csv', 'replica_evidence_closeout.json', 'replica_evidence_gate.json',
            'efficiency_optimal_specialist_regime_frozen.json', 'cpu_audit_closeout_status.json',
            'module11b_summary.md', 'module11b_closeout_verification.json']],
        'baseline_copies': copies})
    print(json.dumps({'verification': 'PASS', **summary}), flush=True)


if __name__ == '__main__':
    main()
