"""CPU-only closeout after the frozen campaign process exits; never launches GPU work."""
import csv
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

P = Path(__file__).resolve().parent
ROOT = P.parent.parent
F = ROOT / 'results/module8f1b_intensity_grid_freeze'

def load(p):
    return json.loads(Path(p).read_text())

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save(name, value):
    path = P / name
    tmp = path.with_name(path.name + '.closeout.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)

def csvrows(name):
    with (P / name).open() as f:
        return list(csv.DictReader(f))

def deployment(value):
    if not value:
        return '未完成'
    x = json.loads(value) if isinstance(value, str) else value
    return ', '.join(f'{m}×{n}' for m, n in x.items() if n)

def number(value, digits=3):
    return f'{float(value):.{digits}f}' if value not in (None, '') else '—'

def replica_evidence(cells_by_id, result_cells):
    """Join measured states to the pre-GPU complete capacity catalog; no solver/run."""
    sys.path.insert(0, str(ROOT))
    from module8f2_resource_static.common import inputs
    _, _, _, cal = inputs()
    def key(x):
        return tuple(sorted((m, n) for m, n in x.items() if n))
    with (F / 'deployment_capacity_limits.csv').open() as source:
        limits = {(r['mixture'], key(json.loads(r['replica_vector']))): float(r['alpha_limit']) for r in csv.DictReader(source)}
    final = {c['cell_id']: c for c in result_cells['v2']}
    evidence = []
    for cid, cell in cells_by_id.items():
        path = P / cid / 'static_v2/search_trace.json'
        if not path.exists():
            continue
        trace = load(path)
        aggs = []
        for p in (P / cid / 'static_v2/search_cache/raw_runs').glob('*/aggregate.json'):
            a = load(p)
            aggs.append((key(a['replica_count_by_model']), a, str(p)))
        for iteration in trace.get('iterations', []):
            for index, candidate in enumerate(iteration['candidates']):
                x = candidate['mutation']['deployment']
                runtime = candidate.get('runtime')
                for model in ['qwen', 'whisper-small', 'resnet50']:
                    if x.get(model) != 2 or runtime is None:
                        continue
                    paired = dict(x)
                    paired[model] = 1
                    a1 = limits.get((cell['mixture'], key(paired)))
                    a2 = limits.get((cell['mixture'], key(x)))
                    canonical = candidate.get('canonical') or {}
                    routing = canonical.get('routing', {})
                    rates = {k.split('::')[0]: v for k, v in routing.items() if k.split('::')[1] == model and v > 0}
                    measured = cal.model_evidence[model]
                    matched = next(((a, path) for k, a, path in aggs if k == key(paired)), None)
                    observed_x1 = matched[0] if matched else None
                    source_x1 = matched[1] if matched else None
                    def pressure(a):
                        if a is None:
                            return None
                        return {'family_metrics': a.get('aggregate_family_metrics', {}).get(model), 'worker_queue_ms': {k: v for k, v in a.get('per_replica_queue_wait', {}).items() if k.split('#')[0] == model}, 'worker_throughput_rps': {k: v for k, v in a.get('per_replica_throughput', {}).items() if k.split('#')[0] == model}, 'stability': a['stability_state_v2'], 'weighted_mean_e2e_ms': a['weighted_mean_e2e'], 'raw_result_paths': a.get('raw_result_paths', [])}
                    final_cell = final[cid]
                    final_agg = final_cell.get('confirmations', {}).get('final')
                    final_has = (final_cell.get('deployment') or {}).get(model) == 2
                    evidence.append({'cell_id': cid, 'mixture': cell['mixture'], 'intensity': cell['intensity_level'], 'frozen_alpha': cell['alpha'], 'budget_mib': cell['memory_budget_mib'], 'iteration': iteration['iteration'] + 1, 'candidate_index': index, 'model': model, 'x2_deployment': x, 'paired_x1_deployment': paired, 'accepted_in_search': candidate['status'] == 'ACCEPTED', 'candidate_status': candidate['status'], 'candidate_reason': candidate['reason'], 'present_in_final_deployment': final_has, 'empirical_g1': cal.capacity_factors[model][1], 'empirical_g2': cal.capacity_factors[model][2], 'measured_scaling_task': measured['task'], 'capacity_scope': measured['capacity_scope'], 'active_family_routing_rps': rates, 'task_transfer_present': any(t != measured['task'] for t in rates), 'paired_alpha_max_x1': a1, 'paired_alpha_max_x2': a2, 'capacity_useful_interval_open_closed': [a1, a2] if a1 is not None and a2 is not None and a2 > a1 else None, 'frozen_alpha_in_capacity_useful_interval': a1 < cell['alpha'] <= a2 if a1 is not None and a2 is not None and a2 > a1 else False, 'observed_x2_search': pressure(runtime), 'observed_paired_x1_search': pressure(observed_x1), 'paired_x1_aggregate_source': source_x1, 'observed_final_confirmations': pressure(final_agg) if final_has and final_agg else None, 'throughput_interpretation': 'Arrival-limited served throughput is not a saturation capacity estimate. Paired canonical routing may change family task rates/mix; no direct g(2) ratio claim or linear doubling assumption.'})
    save('replica_empirical_consistency_audit.json', {'frozen_H5_test': load(F / 'success_hypotheses.json')['H5'], 'frozen_state_capacity_catalog': {'path': str(F / 'deployment_capacity_limits.csv'), 'sha256': sha(F / 'deployment_capacity_limits.csv')}, 'evaluated_x2_candidates': evidence, 'accepted_x2_candidates': sum(r['accepted_in_search'] for r in evidence), 'new_GPU_runs': 0, 'new_capacity_fits': 0, 'new_deployment_enumeration': 0, 'new_routing_optimization': 0})

def closeout():
    status = load(P / 'campaign_status.json')
    complete = status['status'] == 'COMPLETE'
    checks = []
    def check(name, ok, details=None):
        checks.append({'check': name, 'pass': bool(ok), 'details': details})
    freeze = load(P / 'execution_freeze.json')
    protected = freeze['protected_references'] + list(freeze['execution_sources'].values())
    bad = [r['path'] for r in protected if sha(r['path']) != r['sha256']]
    check('frozen_inputs_and_execution_source_hashes', not bad, {'checked': len(protected), 'mismatches': bad})
    assets = load(P / 'hash_verification.json')['model_assets']
    asset_bad = [r['path'] for r in assets if sha(r['path']) != r['expected']]
    check('model_asset_hashes_at_campaign_end', not asset_bad, {'checked': len(assets), 'mismatches': asset_bad})
    bank = load(P / 'physical_measurements.json')['runs']
    cells_by_id = {c['cell_id']: c for c in load(F / 'campaign_cells.json')['cells']}
    contexts = load(P / 'campaign_manifest.json')['contexts']
    protocol = load(ROOT / 'results/module8e2_heldout_freeze/method_comparison_freeze.json')
    for r in bank:
        c = cells_by_id[r['workload']]
        identity = r['identity']
        config = identity['config']
        ok = config['workload_rps'] == c['lambda_rps'] and identity['runtime_context_sha256'] == contexts[r['workload']]['sha256']
        ok = ok and all(config[k] == v for k, v in protocol['measurement_settings'].items())
        check('frozen_workload_context_protocol_' + r['physical_raw_id'], ok)
    raw_with_hash = [r for r in bank if r.get('raw_sha256')]
    check('all_saved_raw_hashes', all(sha(r['raw_path']) == r['raw_sha256'] for r in raw_with_hash), len(raw_with_hash))
    check('planning_safe_admission_for_all_launched_states', all(r.get('planning_safe', True) for r in bank))
    check('completed_campaign_has_no_invalid_or_active_physical_runs', not complete or all(r.get('valid') and r['status'] != 'started' and not r.get('physical_budget_violation') for r in bank))
    sharing = load(P / 'measurement_sharing_log.json')['logical_run_references']
    donors = {r['physical_raw_id']: r for r in bank}
    reuse = [r for r in sharing if r.get('cache_hit')]
    for r in reuse:
        d = donors[r['physical_raw_id']]
        ok = r['phase'] == d['phase'] == 'search'
        ok = ok and all(r[k] == d[k] for k in ['workload', 'workload_sha256', 'deployment', 'routing_hash', 'repetition_seed', 'measurement_identity_hash', 'raw_sha256'])
        ok = ok and sha(r['raw_path']) == d['raw_sha256']
        check('exact_reuse_' + r['workload'] + '_' + r['method'] + '_' + str(r['repetition_seed']), ok)
    check('confirmation_and_historical_reuse_forbidden', not any(r.get('cache_hit') and r['phase'] != 'search' for r in sharing) and load(P / 'reuse_audit.json')['historical_reuse'] == 0)
    accounting = load(P / 'run_accounting.json')
    check('frozen_global_and_structural_adjusted_run_caps', accounting['within_budget'] and accounting['logical_search_evaluations'] <= 84 and accounting['unique_physical_states'] <= 84 and accounting['new_physical_attempts'] <= 324 and accounting['new_timed_runs'] <= 324 and accounting['retries'] == 0)
    check('all_54_potential_reuse_references_audited_before_GPU', len(csvrows('reuse_manifest.csv')) == 54 and load(P / 'reuse_audit.json')['completed_before_first_GPU'])
    tables = {name: csvrows(name) for name in ['memory_sweep_results.csv', 'intensity_sweep_results.csv', 'deployment_composition.csv', 'resource_feasibility.csv']}
    check('all_frozen_cells_and_methods_retained', len(tables['memory_sweep_results.csv']) == 6 and len(tables['intensity_sweep_results.csv']) == 9 and len(tables['deployment_composition.csv']) == 36)
    gc = load(P / 'generalist_centric_results.json')
    check('GC_structural_failure_rows_retained_without_GPU', len(gc['cells']) == 12 and gc['MEMORY_INFEASIBLE'] == 3 and gc['CAPACITY_INFEASIBLE'] == 9 and gc['new_GPU_runs'] == 0)
    with (F / 'intensity_breakpoint_curve.csv').open() as source:
        curve = list(csv.DictReader(source))
    joint = []
    for c in cells_by_id.values():
        prior = max((r for r in curve if r['mixture'] == c['mixture'] and float(r['budget_mib']) == c['memory_budget_mib'] and float(r['alpha_breakpoint']) <= c['alpha']), key=lambda r: float(r['alpha_breakpoint']))
        joint.append({'cell_id': c['cell_id'], 'alpha': c['alpha'], 'preceding_frozen_breakpoint': float(prior['alpha_breakpoint']), 'generalist_only_count_at_preceding_breakpoint': int(prior['generalist_only_feasible_count']), 'specialist_only_count_at_preceding_breakpoint': int(prior['specialist_only_feasible_count']), 'zero_remains_zero_under_increasing_scalar_load': True})
    check('joint_deployment_structural_evidence_from_frozen_curve', all(r['generalist_only_count_at_preceding_breakpoint'] == r['specialist_only_count_at_preceding_breakpoint'] == 0 for r in joint))
    save('joint_deployment_structural_evidence.json', {'frozen_curve': {'path': str(F / 'intensity_breakpoint_curve.csv'), 'sha256': sha(F / 'intensity_breakpoint_curve.csv')}, 'cells': joint, 'runtime_advantage_implied': False, 'no_new_deployment_enumeration': True})
    result_cells = {m: load(P / f)['cells'] for m, f in [('v2', 'v2_results.json'), ('RC', 'rc_sg_results.json')]}
    bottleneck_rows = []
    for method, cells in result_cells.items():
        check(method + '_12_result_rows_retained', len(cells) == 12)
        for cell in cells:
            if cell['status'] != 'COMPLETE':
                continue
            agg = cell['confirmations']['final' if method == 'v2' else 'fixed']
            queues = {worker: metrics['mean']['mean'] for worker, metrics in agg.get('per_replica_queue_wait', {}).items() if metrics.get('mean', {}).get('mean') is not None}
            lam = cells_by_id[cell['cell_id']]['lambda_rps']
            total = sum(lam.values())
            task_contributions = {task: lam[task] / total * metrics['mean'] for task, metrics in agg.get('per_task_mean', {}).items() if metrics.get('mean') is not None}
            bottleneck_rows.append({'cell_id': cell['cell_id'], 'method': 'static_v2' if method == 'v2' else 'RC_SG', 'intensity': cells_by_id[cell['cell_id']]['intensity_level'], 'budget_mib': cells_by_id[cell['cell_id']]['memory_budget_mib'], 'workers_with_largest_mean_queue': sorted(k for k, v in queues.items() if v == max(queues.values())), 'largest_mean_worker_queue_ms': max(queues.values()) if queues else None, 'worker_mean_queues_ms': queues, 'tasks_with_largest_weighted_mean_contribution': sorted(k for k, v in task_contributions.items() if v == max(task_contributions.values())), 'task_weighted_mean_contributions_ms': task_contributions, 'stability': agg['stability_state_v2'], 'source': 'independent final confirmation aggregate; descriptive observed pressure, not a causal attribution or selection input'})
            paths = []
            for role, agg in cell['confirmations'].items():
                expected = P / cell['cell_id'] / ('static_v2' if method == 'v2' else 'RC_SG') / 'confirmation'
                raws = [expected / f'{role}_rep_{i}.json' for i in [1, 2]]
                check(f'{cell["cell_id"]}_{method}_{role}_independent_confirmations', all(p.exists() for p in raws) and len(set(map(str, raws))) == 2)
                paths.extend(map(str, raws))
            check(f'{cell["cell_id"]}_{method}_no_seed_final_confirmation_sharing', len(paths) == len(set(paths)))
    save('runtime_bottleneck_audit.json', {'cells': bottleneck_rows, 'planning_bottleneck_fields_in_main_tables': 'canonical frozen D/g pressure; distinct from these observed worker queue and task latency contribution fields', 'new_GPU_runs': 0, 'routing_modified': False})
    if bottleneck_rows:
        with (P / 'runtime_bottlenecks.csv').open('w', newline='') as out:
            writer = csv.DictWriter(out, fieldnames=list(bottleneck_rows[0]))
            writer.writeheader()
            for r in bottleneck_rows:
                writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in r.items()})
    replica_evidence(cells_by_id, result_cells)
    check('full_campaign_completion', not complete or all(c['status'] == 'COMPLETE' for cells in result_cells.values() for c in cells))
    valid = all(c['pass'] for c in checks)
    validation = {'status': ('PASS' if complete else 'PARTIAL_EVIDENCE_VALID') if valid else 'FAIL', 'campaign_status': status['status'], 'checked_at_utc': datetime.now(timezone.utc).isoformat(), 'checks': checks, 'new_GPU_runs_by_closeout': 0, 'frozen_parameters_modified': False, 'full_grid_completed': complete}
    save('validation_results.json', validation)
    save('actual_reuse_audit.json', {'actual_reused_repetition_references': len(reuse), 'historical_reused_measurements': 0, 'confirmation_reuses': 0, 'checks': [c for c in checks if 'reuse_' in c['check']], 'frozen_54_reference_preaudit': {'path': str(P / 'reuse_audit.json'), 'sha256': sha(P / 'reuse_audit.json')}, 'additional_exact_same_cell_candidate_sharing': 'Permitted by the pre-GPU frozen phase contract only when independent RC-SG commitment and every measurement identity field match.'})
    memory = tables['memory_sweep_results.csv']
    intensity = tables['intensity_sweep_results.csv']
    composition = tables['deployment_composition.csv']
    analysis = load(P / 'campaign_analysis.json')
    replicas = load(P / 'replica_activation_summary.json')['models']
    h = load(P / 'hypothesis_audit.json')['hypotheses']
    hypothesis_audit = load(P / 'hypothesis_audit.json')
    hypothesis_audit['hypotheses']['H5']['empirical_consistency_evidence'] = {'path': str(P / 'replica_empirical_consistency_audit.json'), 'sha256': sha(P / 'replica_empirical_consistency_audit.json')}
    save('hypothesis_audit.json', hypothesis_audit)
    failures = load(P / 'runtime_failure_log.json')['events']
    categories = Counter(r['category'] for r in failures)
    stability = Counter(r.get('stability', 'missing') for r in bank if r.get('valid'))
    regimes = [r['v2_regime'] for r in memory]
    memory_complete = all(r['v2_status'] == r['RC_SG_status'] == 'COMPLETE' for r in memory)
    chain = any(regimes[i] == 'Generalist-heavy' and regimes[j] == 'Hybrid' and regimes[k] == 'Specialist-heavy' for i in range(len(regimes)) for j in range(i + 1, len(regimes)) for k in range(j + 1, len(regimes))) if memory_complete else None
    loose = load(P / 'loose_resource_consistency.json')['cells']
    tight = [r for r in memory if r['memory_budget_id'] in ['M1', 'M2', 'M3']]
    tight_outcomes = {r['memory_budget_id']: r['outcome'] for r in tight}
    tight_sharing = {r['memory_budget_id']: json.loads(r['v2_generalist_task_sharing']) if r['v2_generalist_task_sharing'] else None for r in tight}
    intensity_changes = {r['memory_budget_id']: r['composition_changed'] for r in load(P / 'intensity_effect.json')['budgets']}
    same_vector = [r['cell_id'] for r in memory + intensity if r['outcome'] == 'WIN' and r['deployment_L1_distance'] == '0']
    same_vector = sorted(set(same_vector))
    paragraphs = [
        '# Module 8F-2 Resource-Constrained Static Real GPU Campaign', '',
        f'状态：**{status["status"]}**；CPU 完成审计：**{validation["status"]}**。12 个冻结单元和 3 个方法全部保留。缺失测量为缺失值，不填造延迟。', '',
        'GPU UUID 差异按用户明确授权放行；GPU 型号、驱动、49140 MiB 物理容量及冻结环境已经核验。原始 8F-1A/1B 输入与 static v2 源码未修改；旧 UUID 的历史测量未复用。', '',
        '## 16 项最终回答', '',
        f'1. 新增 timed runs：**{accounting["new_timed_runs"]}**；有效完成 {accounting["completed_valid_timed_runs"]}；物理启动 {accounting["new_physical_attempts"]}；独立确认 {accounting["confirmation_runs"]}；重试 {accounting["retries"]}。',
        f'2. 实际复用：**{len(reuse)}** 次搜索重复测量引用；历史复用 0，确认复用 0。54 个提前标记机会均已在 GPU 前审计，45 个跨预算引用禁止、9 个同单元引用条件允许；后续相同候选共享也必须满足同一冻结身份规则。',
        f'3. Memory Sweep 的 v2 vs RC-SG：**{dict(Counter(r["outcome"] for r in memory))}**。这是 6 个 HIGH 单元的比较，不混入 LOW 单元胜负。',
        f'4. Feasibility win budgets：{[r["memory_budget_id"] for r in memory if r["outcome"] == "FEASIBILITY_WIN"]}。结构不可行没有虚构延迟。',
        '5. M1→M6 的部署组成见下表，所有数值来自独立 final 确认；未完成的点显式保留。',
        f'6. Generalist-heavy→Hybrid→Specialist-heavy 链是否出现：**{chain}**（None 表示实验未齐）。M1→M6 实际显示类别：{regimes}。Frozen predicates 有重叠：Hybrid = N_G>0 且 N_S>0；Specialist-heavy = N_S>N_G；原始两个标记全部保留，不重新定义阈值。',
        f'7. M6 是否接近 RC-SG：{ {r["cell_id"]: r["close_under_frozen_rule"] for r in loose} }；按冻结 same-stability/max(1ms,5%) 规则判断，latency gap、部署 L1 distance 与 safe memory gap 的数值附于下文。',
        f'8. Tight-resource 正式结果：{tight_outcomes}；Generalist task sharing：{tight_sharing}。只有正式 WIN/FEASIBILITY_WIN 才列为 objective 优势；相同部署的测量 WIN 不证明部署选择优势，双方不稳定时 TIE 也不意味着均值相等。完整 mean/P95/内存差见 tight_resource_value.json 和 Table A。',
        f'9. Offered load 是否改变部署：{intensity_changes}（None 表示未齐）。下文展示各固定预算 I1→I2→I3 的实际向量；向量不变时如实记录延迟/稳定性变化。',
        f'10. x=2 evaluated / accepted 候选次数：{ {m: {k: r[k] for k in ["evaluated", "accepted"]} for m, r in replicas.items()} }。这是候选 occurrence，不是物理重复次数。',
        f'11. 未接受原因：{ {m: r["nonacceptance_reasons"] for m, r in replicas.items()} }。未生成、显存不符、Top-B 剪枝与 runtime 回滚分开记录；不得增加强度强行激活。',
        f'12. Planning-safe 但 actual peak 超预算：**{accounting["planning_safe_peak_budget_violations"]}**。每份真实测量的 plan/safe/idle/peak/B 在 physical_memory_audit.csv；任何违规均保留为 safety-model failure，预算不扩大。',
        f'13. 有效运行稳定性计数：{dict(stability)}；执行事件分类：{dict(categories)}。campaign 停机错误：{status.get("error", "无")}。OOM、外部中断和真实 unstable 不合并。',
        f'14. H1–H5：**{ {k: v["status"] for k, v in h.items()} }**；证据与限制见下文。假设不要求全部成立。',
        '15. 联合部署的必要性与 v2 的优势分别判断：已冻结 8F-1B 规划曲线中，这 12 个点的纯 Generalist / 纯 Specialist 可行数均为 0；runtime 结果进一步检验具体联合部署的稳定性和表现。此结构证据不证明 v2 全局最优或普遍优于 RC-SG。',
        f'16. 预定 Static main GPU 数据采集是否完成：**{complete and valid}**。' + ('本冻结 campaign 可结束；论文性能主张必须服从本次正面和负面证据。' if complete and valid else 'campaign 未完整完成或审计未通过，不能宣告 Static main experiments 结束。'), '',
        '## Table A — Memory Sweep', '',
        '| Budget / MiB | v2 deployment | RC-SG deployment | v2 / RC mean ms | v2 / RC P95 ms | v2 / RC safe MiB | Outcome |',
        '|---|---|---|---|---|---|---|',
    ]
    for r in memory:
        paragraphs.append(f'| {r["memory_budget_id"]} / {number(r["budget_mib"],9)} | {deployment(r["v2_deployment"])} | {deployment(r["RC_SG_deployment"])} | {number(r["v2_mean_e2e_ms"])} / {number(r["RC_SG_mean_e2e_ms"])} | {number(r["v2_weighted_p95_ms"])} / {number(r["RC_SG_weighted_p95_ms"])} | {number(r["v2_M_safe_mib"])} / {number(r["RC_SG_M_safe_mib"])} | {r["outcome"]} |')
    paragraphs += ['', '## Table B — Intensity Sweep', '', '| Budget / Intensity | v2 deployment | RC-SG deployment | v2 / RC mean ms | v2 / RC stability | Outcome |', '|---|---|---|---|---|---|']
    for r in intensity:
        paragraphs.append(f'| {r["memory_budget_id"]} / {r["intensity"]} | {deployment(r["v2_deployment"])} | {deployment(r["RC_SG_deployment"])} | {number(r["v2_mean_e2e_ms"])} / {number(r["RC_SG_mean_e2e_ms"])} | {r["v2_stability"]} / {r["RC_SG_stability"]} | {r["outcome"]} |')
    paragraphs += ['', '## Table C — Deployment Composition', '', '| Budget | N_G | N_S | Total replicas | Additional replicas | Frozen regime |', '|---|---|---|---|---|---|']
    for r in composition:
        if r['experiment_A'] == 'True' and r['method'] == 'static_v2':
            paragraphs.append(f'| {r["memory_budget_id"]} | {r["N_G"]} | {r["N_S"]} | {r["total_replicas"]} | {r["additional_replicas"]} | {r["regime"]}; hybrid={r["hybrid"]}, specialist_heavy={r["specialist_heavy"]} |')
    paragraphs += ['', '## Table D — Feasibility', '', '| Cell | v2 | RC-SG | Generalist-centric |', '|---|---|---|---|']
    for c in load(F / 'campaign_cells.json')['cells']:
        rows = {r['method']: r for r in composition if r['cell_id'] == c['cell_id']}
        paragraphs.append(f'| {c["cell_id"]} | {rows["static_v2"]["status"]} | {rows["RC_SG"]["status"]} | {rows["Generalist_centric"]["status"]} |')
    paragraphs += ['', '## Table E — Replica Audit', '', '| Model | Generated | Memory feasible | Capacity useful | Top-B | Evaluated | Accepted | Rollback |', '|---|---|---|---|---|---|---|---|']
    for m, r in replicas.items():
        paragraphs.append('| ' + m + ' | ' + ' | '.join(str(r[k]) for k in ['generated', 'memory_feasible', 'capacity_useful', 'Top_B', 'evaluated', 'accepted', 'rollback']) + ' |')
    paragraphs += ['', 'Memory-feasible x=2 只表示 M_safe(x2)<=B。Capacity-useful 是相同 λ、其余部署不变时 x2 的 canonical capacity gate 可行而配对 x1 不可行；两者不等同，runtime 是否改善另行判断。', '', '每个实测 x=2 的 paired planning interval、frozen g(2)、代表测量任务、task-transfer 限制以及实际 throughput/queue 见 replica_empirical_consistency_audit.json。未饱和的 offered-load throughput 比值不能作为 capacity gain 估计；不假设线性翻倍。', '', '主表 bottleneck_models 是冻结 canonical D/g 的规划压力；实测排队最长的 worker 与加权延迟贡献最大的 task 另见 runtime_bottlenecks.csv / runtime_bottleneck_audit.json。这是描述性运行压力，不用来修改路由或选择。规划可行不等于实际 steady。', '', '## Loose-resource and Intensity Evidence', '', '```json', json.dumps(load(P / 'loose_resource_consistency.json'), indent=2, ensure_ascii=False), '```', '', '```json', json.dumps([{k: r[k] for k in ['memory_budget_id', 'complete', 'distinct_deployment_count', 'composition_changed']} for r in load(P / 'intensity_effect.json')['budgets']], indent=2, ensure_ascii=False), '```', '', f'相同部署出现正式 WIN 的单元：{same_vector}。这些测量差异不能归因为不同部署选择。', '', '## Frozen H1–H5', '', '| Hypothesis | Status | Evidence |', '|---|---|---|']
    for name, r in h.items():
        paragraphs.append(f'| {name} | {r["status"]} | {r["reason"]} |')
    paragraphs += ['', f'Logical evaluations {accounting["logical_search_evaluations"]} / 96（结构调整后 84）；physical states {accounting["unique_physical_states"]} / 96（调整后 84）；new timed {accounting["new_timed_runs"]} / 384（调整后启动/计时上限 324）。', '', '全部原始结果、trace、独立确认、复用核查、预算检查和可直接绘图的 CSV 保留。没有新资源点、新强度、方法/路由修订、dynamic、第二硬件对比或额外 GPU 运行。']
    (P / 'module8f2_summary.md').write_text('\n'.join(paragraphs) + '\n')
    save('final_closeout.json', {'campaign_status': status['status'], 'validation_status': validation['status'], 'STATIC_MAIN_EXPERIMENTS_COMPLETE': complete and valid, 'finished_at_utc': datetime.now(timezone.utc).isoformat(), 'new_GPU_runs': 0})
    files = sorted(p for p in P.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json' and p.name not in ['finalization.log', '.campaign.lock'] and '__pycache__' not in p.parts)
    save('artifact_hashes.json', {str(p.relative_to(P)): {'path': str(p), 'sha256': sha(p)} for p in files})
    print(json.dumps({'event': 'CPU_CLOSEOUT_COMPLETE', 'campaign_status': status['status'], 'validation_status': validation['status']}, ensure_ascii=False), flush=True)

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--wait-pid':
        command = Path('/proc') / sys.argv[2] / 'cmdline'
        while command.exists():
            try:
                if b'module8f2_resource_static.campaign' not in command.read_bytes():
                    break
            except FileNotFoundError:
                break
            time.sleep(30)
    status = load(P / 'campaign_status.json')
    if status['status'] not in ['COMPLETE', 'INCOMPLETE']:
        save('finalization_blocked.json', {'reason': 'Campaign process ended without a terminal status; preserve partial evidence and require explicit investigation.', 'status': status, 'new_GPU_runs': 0})
        raise SystemExit(2)
    closeout()
