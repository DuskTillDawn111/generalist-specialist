"""Audit immutable runtime records and compare observed workload allocations."""
import csv
import json
from collections import defaultdict
import statistics
from candidate_pool_service_demand.dry_run import load_completed_catalog
from candidate_pool_audit.provenance import environment_snapshot
from deployment_core import Workload
from deployment_core.replica_calibration import load_replica_calibration, ReplicaPlanningPolicy
from deployment_core.replicas import replica_key, replica_statistics
from deployment_search import IntegerGuidedDeploymentSearch, SearchConfig
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.runtime_cache import save_json, file_hash, digest
from deployment_search.result import jsonable
from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluation
from .prepare import OUT, ROOT, PILOT, ALL, read, ref, verify_freeze, verify_references


def category(role, old, new):
    if new < old:
        return 'scale_down'
    if old:
        return role + '_replica_up'
    return 'new_' + role + '_type'


def percent_gain(seed, final):
    return 100 * (seed - final) / seed if seed is not None and seed > 0 and final is not None else None


def decision_gate(rows, audit_errors):
    complete = len(rows) == 3 and all(r['complete'] for r in rows)
    improvement = complete and all(r['mean_improvement_percent'] is not None and r['mean_improvement_percent'] > 0 and r['runtime_objective_improved'] for r in rows)
    confirmed = complete and all(r['confirmation_objective_direction_maintained'] for r in rows)
    bounded = complete and all(r['runtime_evaluations'] <= 6 and r['enumeration_calls'] == 0 for r in rows) and not audit_errors
    distinct = complete and len({replica_key(r['final']) for r in rows}) > 1
    steady = complete and all(r['final_stability'] == 'steady' for r in rows)
    status = ('STRONG SUPPORT' if improvement and confirmed and bounded and steady and distinct else
              'SUPPORT' if improvement and confirmed and bounded and steady else 'PARTIAL')
    return {'status': status, 'three_complete': complete, 'all_runtime_improved': improvement,
        'all_confirmation_maintained': confirmed, 'bounded_and_audited': bounded,
        'all_final_steady': steady, 'some_final_deployments_differ': distinct,
        'scope': 'Observed frozen W1/W2/W3, v2 pool, RTX 4090, evidence-bounded space and fixed Top-B/budget; no global or universal optimality claim.'}


def audit_candidates(name, trace, policy, roles, errors):
    rows = []
    objective = NoiseAwareRuntimeObjective(1., .05)
    for t in trace['iterations']:
        incumbent = ReplicaRuntimeEvaluation(**t['current_runtime'])
        memory = policy.deployment_memory(t['deployment'])
        ranked = sorted((a for a in t['candidates'] if a['rank'] is not None), key=lambda a: a['rank'])
        if [a for a in ranked if a['selected_top_b']] != ranked[:2]:
            errors.append(name + ': Top-B mismatch')
        if t['num_candidates_generated'] > 2 * trace['num_models']:
            errors.append(name + ': neighborhood >2M')
        best = None
        for a in ranked:
            if a['runtime'] is None:
                continue
            rt = ReplicaRuntimeEvaluation(**a['runtime'])
            better, _ = objective.compare(rt, incumbent, a['memory_mib'], memory)
            if better and (best is None or objective.compare(rt, ReplicaRuntimeEvaluation(**best['runtime']), a['memory_mib'], best['memory_mib'])[0]):
                best = a
        expected = jsonable(replica_key(best['mutation']['deployment'])) if best else None
        if expected != t['accepted']:
            errors.append(name + ': accept/rollback replay mismatch')
        for a in t['candidates']:
            model = a['mutation']['model']; old = t['deployment'][model]; new = a['mutation']['deployment'][model]
            rt = a['runtime']; better, reason = (None, None)
            if rt is not None:
                better, reason = objective.compare(ReplicaRuntimeEvaluation(**rt), incumbent, a['memory_mib'], memory)
                if not a['selected_top_b']:
                    errors.append(name + ': non-Top-B runtime evaluation')
            z = t['canonical']['max_pressure']; candidate_z = a['canonical']['max_pressure'] if a['canonical'] else None
            rows.append({'workload': name, 'iteration': t['iteration'], 'candidate': model,
                'candidate_category': category(roles[model], old, new), 'old_count': old, 'new_count': new,
                'incumbent': t['deployment'], 'deployment': a['mutation']['deployment'],
                'memory_delta_mib': policy.memory(model, new) - policy.memory(model, old),
                'empirical_capacity_delta': policy.capacity(model, new) - policy.capacity(model, old),
                'z_current': z, 'z_candidate': candidate_z, 'delta_z': a['delta_z'], 'analytical_rank': a['rank'],
                'selected_top_b': a['selected_top_b'], 'runtime_evaluated': rt is not None,
                'real_mean_e2e_ms': rt['weighted_mean_e2e'] if rt else None,
                'real_weighted_task_p95_ms': rt['weighted_p95_e2e'] if rt else None,
                'stability': rt['stability_state_v2'] if rt else None, 'decision': a['status'], 'reason': a['reason'],
                'runtime_better_than_incumbent': better, 'runtime_objective_reason': reason})
    return rows


def counterfactual(directory, trace, protocol, catalog, hardware, roles, cal):
    if directory == PILOT:
        # The existing W2 CPU analysis already matches prior visited history.
        return read(PILOT / 'linear_counterfactual_audit.json')['comparisons']
    comparisons = []; visited = {replica_key(trace['seed']['deployment'])}
    for t in trace['iterations']:
        config = SearchConfig(**dict(protocol['search_config'], dry_run=True))
        result = IntegerGuidedDeploymentSearch().search(Workload(protocol['workload']['arrival_rates_rps']), catalog, hardware,
            config, initial_deployment=t['deployment'], phase_visited_deployments=set(visited), roles=roles,
            capacity_mode='linear', memory_mode='empirical', calibration=cal)
        linear = [a.mutation.deployment for a in result.iterations[0]['candidates'] if a.selected_top_b]
        empirical = [a['mutation']['deployment'] for a in t['candidates'] if a['selected_top_b']]
        comparisons.append({'iteration': t['iteration'], 'fixed_incumbent': t['deployment'],
            'prior_visited_states': [dict(k) for k in sorted(visited)], 'linear_Top_B': linear, 'empirical_Top_B': empirical,
            'selection_changed': linear != empirical,
            'scope': 'CPU-only counterfactual at same actual incumbent with same prior visited history and empirical memory/evidence; no linear GPU search.'})
        visited.update(replica_key(a['mutation']['deployment']) for a in t['candidates'] if a['runtime'] is not None)
    return comparisons


def audit_runtime(directory, name, trace, confirmation, hardware, errors):
    cache = read(directory / 'search_cache/runtime_cache_manifest.json')
    entries = []; memory = []
    for key, entry in cache['configurations'].items():
        if digest(entry['identity']) != key:
            errors.append(name + ': cache identity mismatch')
        row = {'cache_key': key, 'identity': entry['identity'], 'status': entry['status'], 'repetitions': entry['repetitions']}
        if entry['status'] == 'complete':
            if file_hash(entry['aggregate']) != entry['aggregate_sha256'] or len(entry['repetitions']) != 3:
                errors.append(name + ': aggregate hash/repetition mismatch')
            row['runtime_result'] = read(entry['aggregate'])
        entries.append(row)
    reps = [('search', r['runtime']) for e in entries for r in e['repetitions'] if r.get('runtime')]
    reps += [('confirmation', r) for rows in confirmation.get('repetitions', {}).values() for r in rows]
    for kind, r in reps:
        raw_path = r['raw_result_path']
        if file_hash(raw_path) != r['raw_sha256']:
            errors.append(name + ': raw hash mismatch')
        raw = read(raw_path); o = raw['memory_observations']
        window = raw['measurement_window']
        samples = [s for s in raw['telemetry']['samples'] if window['start_time_ns'] <= s['time_ns'] < window['end_time_ns']]
        gpu_stats = {}
        for field in ('gpu_util_percent', 'memory_util_percent', 'power_w'):
            values = [s[field] for s in samples if s.get(field) is not None]
            gpu_stats[field] = {'mean': statistics.mean(values), 'min': min(values), 'max': max(values), 'measurement_count': len(values)} if values else None
        deviation = (o['actual_idle_mib'] - o['planning_idle_mib']) / o['planning_idle_mib'] if o['actual_idle_mib'] is not None else None
        memory.append({'category': kind, 'raw': ref(raw_path), 'deployment': raw['deployment'],
            **o, 'idle_relative_difference': deviation,
            'within_prefrozen_5percent_idle_tolerance': abs(deviation) <= .05 if deviation is not None else None,
            'runtime_resource_status': raw['runtime_resource_status'], 'release': raw['gpu_release'],
            'worker_identities': raw['worker_identities'], 'measurement_window_GPU_statistics': gpu_stats})
        if not raw['gpu_release']['verified']:
            errors.append(name + ': GPU release unverified')
        if o['peak_device_mib'] is not None and o['peak_device_mib'] > hardware.gpu_memory_mib:
            errors.append(name + ': actual peak above hard budget')
    complete = sum(e['status'] == 'complete' for e in entries)
    if complete != trace['runtime_evaluation_count'] or len(entries) > 6 or len(cache['run_attempts']) > 18:
        errors.append(name + ': runtime budget/count mismatch')
    if not confirmation.get('complete') or confirmation.get('physical_attempts') != 4:
        errors.append(name + ': confirmation incomplete')
    if len({(a['cache_key'], a['repetition']) for a in cache['run_attempts']}) != len(cache['run_attempts']):
        errors.append(name + ': duplicate physical attempt')
    return {'valid_unique_runtime_states': complete, 'reserved_states': len(entries),
        'search_physical_attempts': len(cache['run_attempts']), 'evaluations': entries}, {
        'GPU_capacity_mib': hardware.gpu_memory_mib, 'warmup_seconds': 10, 'runs': memory,
        'post_warmup_idle_within_frozen_5percent_tolerance': all(r['within_prefrozen_5percent_idle_tolerance'] is True for r in memory),
        'max_idle_relative_difference': max((abs(r['idle_relative_difference']) for r in memory if r['idle_relative_difference'] is not None), default=None),
        'maximum_sampled_device_peak_mib': max((r['peak_device_mib'] for r in memory if r['peak_device_mib'] is not None), default=None),
        'runtime_resource_infeasible_count': sum(r['runtime_resource_status'] == 'RUNTIME_RESOURCE_INFEASIBLE' for r in memory),
        'note': 'Idle-table deviation is diagnostic; it does not guarantee workload peak. All raw telemetry and per-family/per-replica statistics remain authenticated.'}


def csv_write(path, rows, fields):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore'); writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def report(incomplete=False):
    verify_freeze()
    catalog, hardware, roles, _, _ = load_completed_catalog(); cal = load_replica_calibration(catalog)
    policy = ReplicaPlanningPolicy(catalog, 'empirical', calibration=cal)
    errors = []; workloads = []; competition = []; activations = {}; confirmations = {}; comparisons = {}
    new_attempts = {}
    for short in ('w1', 'w3'):
        cache_path = OUT / short / 'search_cache/runtime_cache_manifest.json'
        cache = read(cache_path) if cache_path.exists() else {'run_attempts': [], 'configurations': {}}
        confirmation_path = OUT / short / 'confirmation_attempts.json'
        attempts = read(confirmation_path) if confirmation_path.exists() else []
        new_attempts[short] = {'search': len(cache['run_attempts']), 'confirmation': len(attempts), 'reserved_states': len(cache['configurations'])}
        if len(cache['run_attempts']) > 18 or len(attempts) > 4 or len(cache['configurations']) > 6:
            errors.append(short + ': reserved/physical budget exceeded')
    for short, name in ALL:
        directory = PILOT if short == 'w2' else OUT / short
        if not (directory / 'search_complete.json').exists() or not (directory / 'confirmation_results.json').exists() or not read(directory / 'confirmation_results.json').get('complete'):
            workloads.append({'workload': name, 'short': short, 'complete': False, 'reason': 'search/confirmation pending or failed'})
            continue
        trace = read(directory / 'search_trace.json'); confirmation = read(directory / 'confirmation_results.json')
        protocol = read(directory / ('pilot_protocol.json' if short == 'w2' else 'protocol.json'))
        if file_hash(directory / 'search_trace.json') != read(directory / 'post_search_freeze.json')['search_trace_sha256']:
            errors.append(name + ': confirmation changed search trace')
        if trace['enumeration_calls'] or trace['search_iterations'] > 3:
            errors.append(name + ': enumeration/iteration bound')
        decision_rows = audit_candidates(name, trace, policy, roles, errors); competition += decision_rows
        runtime, memory = audit_runtime(directory, name, trace, confirmation, hardware, errors)
        seed, final = trace['seed']['deployment'], trace['deployment']; a, b = trace['seed_runtime'], trace['runtime']
        stats = replica_statistics(final, roles)
        better, reason = NoiseAwareRuntimeObjective(1., .05).compare(ReplicaRuntimeEvaluation(**b), ReplicaRuntimeEvaluation(**a), policy.deployment_memory(final), policy.deployment_memory(seed))
        selected_two = any(n == 2 for n in seed.values()) or any(any(n == 2 for n in d['deployment'].values()) for d in decision_rows if d['decision'] == 'ACCEPTED')
        row = {'workload': name, 'short': short, 'complete': True, 'seed': seed, 'final': final,
            'seed_counts': replica_statistics(seed, roles), **stats,
            'active_generalist_model_ids': [m for m, n in final.items() if n and roles[m] == 'generalist'],
            'active_specialist_model_ids': [m for m, n in final.items() if n and roles[m] == 'specialist'],
            'total_replicas': sum(final.values()), 'accepted_x2': selected_two, 'final_has_x2': any(n == 2 for n in final.values()),
            'seed_memory_mib': policy.deployment_memory(seed), 'final_memory_mib': policy.deployment_memory(final),
            'seed_mean_e2e_ms': a['weighted_mean_e2e'], 'final_mean_e2e_ms': b['weighted_mean_e2e'],
            'mean_improvement_percent': percent_gain(a['weighted_mean_e2e'], b['weighted_mean_e2e']),
            'seed_weighted_task_p95_ms': a['weighted_p95_e2e'], 'final_weighted_task_p95_ms': b['weighted_p95_e2e'],
            'p95_improvement_percent': percent_gain(a['weighted_p95_e2e'], b['weighted_p95_e2e']),
            'seed_stability': a['stability_state_v2'], 'final_stability': b['stability_state_v2'],
            'runtime_objective_improved': better, 'runtime_objective_reason': reason,
            'runtime_evaluations': trace['runtime_evaluation_count'], 'search_physical_attempts': runtime['search_physical_attempts'],
            'search_status': trace['status'], 'enumeration_calls': trace['enumeration_calls'],
            'confirmation_complete': confirmation['complete'], 'confirmation_objective_direction_maintained': confirmation['objective_direction_maintained'],
            'confirmation_mean_improvement_percent': 100 * confirmation['relative_mean_gain'] if confirmation['relative_mean_gain'] is not None else None,
            'mandatory_capability_coverage': protocol['mandatory_presence'], 'source_trace': ref(directory / 'search_trace.json'),
            'source_confirmation': ref(directory / 'confirmation_results.json'), 'reused_W2_result': short == 'w2',
            'new_GPU_search_attempts': 0 if short == 'w2' else runtime['search_physical_attempts'],
            'new_GPU_confirmation_attempts': 0 if short == 'w2' else 4, 'memory_validation': memory}
        workloads.append(row)
        activations[name] = {'seed_x2_models': [m for m, n in seed.items() if n == 2], 'selected_or_accepted_x2': selected_two,
            'final_x2_models': [m for m, n in final.items() if n == 2], 'candidates': [d for d in decision_rows if d['new_count'] == 2],
            'evidence_bounds': policy.bounds(hardware)[0]}
        confirmations[name] = {'complete': True, 'reused_W2': short == 'w2', 'influences_search': False,
            'objective_direction_maintained': confirmation['objective_direction_maintained'], 'objective_reason': confirmation['objective_reason'],
            'relative_mean_gain': confirmation['relative_mean_gain'],
            'seed_mean_e2e_ms': confirmation['aggregates']['seed']['weighted_mean_e2e'],
            'final_mean_e2e_ms': confirmation['aggregates']['final']['weighted_mean_e2e'],
            'seed_weighted_task_p95_ms': confirmation['aggregates']['seed']['weighted_p95_e2e'],
            'final_weighted_task_p95_ms': confirmation['aggregates']['final']['weighted_p95_e2e'],
            'source': row['source_confirmation']}
        comparisons[name] = counterfactual(directory, trace, protocol, catalog, hardware, roles, cal)
        target = OUT / short
        save_json(target / 'result_summary.json', row)
        save_json(target / 'integer_replica_decisions.json', {'decisions': decision_rows, 'source_trace': row['source_trace']})
        save_json(target / 'linear_counterfactual.json', {'comparisons': comparisons[name], 'new_GPU_runs': 0})
        save_json(target / 'memory_validation.json', memory)
        save_json(target / 'runtime_evaluations.json', dict(runtime, reused_W2=short == 'w2'))
    before = read(OUT / 'before_hashes.json')
    verify_references(before['frozen_files'])
    native_changed = [name for name, value in before['native_models_and_adapters'].items() if file_hash(ROOT / name) != value['sha256']]
    environment_unchanged = environment_snapshot() == read(OUT / 'environment_manifest.json')['environments']
    if native_changed or not environment_unchanged:
        errors.append('native model/adapter/environment drift')
    save_json(OUT / 'source_validation.json', {'frozen_files_checked': len(before['frozen_files']), 'changed_frozen_files': [],
        'native_model_files_checked': len(before['native_models_and_adapters']), 'changed_native_files': native_changed,
        'environments_unchanged': environment_unchanged, 'W2_results_unchanged': True,
        'experiment_freeze_sha256': file_hash(OUT / 'experiment_freeze.json')})
    gate = decision_gate(workloads, errors)
    groups = defaultdict(list)
    for d in competition:
        if d['runtime_evaluated']:
            groups[replica_key(d['deployment'])].append(d)
    same = [{'deployment': dict(key), 'outcomes': value,
             'acceptance_decisions_differ': len({d['decision'] for d in value}) > 1,
             'objective_improvement_statuses_differ': len({d['runtime_better_than_incumbent'] for d in value}) > 1}
            for key, value in groups.items() if len({d['workload'] for d in value}) > 1]
    full = [r for r in workloads if r['complete']]
    cross = {'final_deployments_differ': len({replica_key(r['final']) for r in full}) > 1,
        'N_G_differs': len({r['N_G'] for r in full}) > 1, 'N_S_differs': len({r['N_S'] for r in full}) > 1,
        'workload_dependent_allocation_observed': len({replica_key(r['final']) for r in full}) > 1,
        'any_selected_x2': any(r['accepted_x2'] for r in full),
        'calibration_changed_Top_B': {name: any(c['selection_changed'] for c in cs) for name, cs in comparisons.items()},
        'linear_vs_empirical': comparisons,
        'analytical_Top_B_runtime_rollbacks': [d for d in competition if d['decision'] == 'ROLLBACK'],
        'same_evaluated_deployment_across_workloads': same,
        'memory_rejections': [d for d in competition if d['decision'] == 'REJECT_INFEASIBLE' and d['reason'] == 'memory'],
        'replica_vs_diversity_scope': 'Only bounded observed transitions and Top-B evaluations. Pruned x2 states have no workload runtime measurement; no universal comparison.'}
    summary = {'module': '7C-4', 'complete': not incomplete and len(full) == 3 and not errors,
        'decision_gate': gate, 'workloads': workloads, 'cross_workload': cross, 'audit_errors': errors,
        'new_GPU_attempts_by_workload': new_attempts,
        'total_new_GPU_search_attempts': sum(r['search'] for r in new_attempts.values()),
        'total_new_GPU_confirmation_attempts': sum(r['confirmation'] for r in new_attempts.values()),
        'new_GPU_physical_attempt_cap': 44, 'W2_new_GPU_runs': 0, 'new_adapter_smoke_runs': 0,
        'enumeration_calls': sum(r['enumeration_calls'] for r in full)}
    save_json(OUT / 'multiworkload_summary.json', summary)
    save_json(OUT / 'replica_activation.json', activations)
    save_json(OUT / 'confirmation_summary.json', confirmations)
    csv_write(OUT / 'deployment_count_table.csv', full, ['workload', 'seed', 'final', 'N_G', 'N_S',
        'active_generalist_types', 'active_specialist_types', 'total_replicas', 'accepted_x2',
        'seed_mean_e2e_ms', 'final_mean_e2e_ms', 'mean_improvement_percent', 'seed_weighted_task_p95_ms',
        'final_weighted_task_p95_ms', 'p95_improvement_percent', 'seed_stability', 'final_stability',
        'seed_memory_mib', 'final_memory_mib', 'runtime_evaluations', 'enumeration_calls', 'reused_W2_result'])
    csv_write(OUT / 'generalist_specialist_competition.csv', competition, ['workload', 'iteration', 'candidate',
        'candidate_category', 'old_count', 'new_count', 'memory_delta_mib', 'empirical_capacity_delta',
        'z_current', 'z_candidate', 'delta_z', 'analytical_rank', 'selected_top_b', 'runtime_evaluated',
        'real_mean_e2e_ms', 'real_weighted_task_p95_ms', 'stability', 'decision', 'reason', 'incumbent', 'deployment'])
    markdown(summary)
    save_json(OUT / 'validation.json', {'status': 'passed' if summary['complete'] else 'incomplete' if incomplete else 'failed',
        'audit_errors': errors, 'W2_new_GPU_runs': 0, 'enumeration_calls': summary['enumeration_calls'],
        'new_GPU_attempts': summary['total_new_GPU_search_attempts'] + summary['total_new_GPU_confirmation_attempts'], 'cap': 44})
    artifact_manifest()
    if errors and not incomplete:
        raise AssertionError('Module 7C-4 audit failed: ' + str(errors))


def markdown(summary):
    complete = [r for r in summary['workloads'] if r['complete']]; cross = summary['cross_workload']; gate = summary['decision_gate']
    sparse = lambda x: ', '.join(f'{m}={n}' for m, n in x.items() if n)
    lines = ['# Module 7C-4 Multi-workload Real Integer Deployment Validation', '',
        f'Decision gate：**{gate["status"]}**。固定 W1/W2/W3，未修改 rates、mix、10 s warmup / 210 s window 或 duration。',
        'W2 直接引用 Module 7C-3 search、confirmation、raw traces；新 GPU runs=0。W1/W3 独立 MILP seed；'
        'empirical capacity/memory、predictor=None、Top-B=2、最多 3 rounds / 6 unique states / 3 repetitions、patience=1、5% gate。',
        '接受规则与 7C-3 同一 implementation：stability 优先；same-steady mean→weighted task P95→memory；same-nonsteady 保留 incumbent。',
        '', '| Workload | Seed | Final | N_G | N_S | x=2 accepted? | Mean ms seed→final | Mean gain | Weighted task P95 ms seed→final | P95 gain | Stability | Runtime states | Enum |',
        '|---|---|---|---:|---:|---|---|---:|---|---:|---|---:|---:|']
    for r in complete:
        lines.append(f'| {r["workload"]} | {sparse(r["seed"])} | {sparse(r["final"])} | {r["N_G"]} | {r["N_S"]} | {r["accepted_x2"]} '
            f'| {r["seed_mean_e2e_ms"]:.3f}→{r["final_mean_e2e_ms"]:.3f} | {r["mean_improvement_percent"]:.2f}% '
            f'| {r["seed_weighted_task_p95_ms"]:.3f}→{r["final_weighted_task_p95_ms"]:.3f} | {r["p95_improvement_percent"]:.2f}% '
            f'| {r["seed_stability"]}→{r["final_stability"]} | {r["runtime_evaluations"]}/6 | {r["enumeration_calls"]} |')
    lines += ['', '完整 12-type replica vectors（包括所有 zeros）、active types、replica counts、memory、mandatory coverage 在 JSON/CSV 中保存。',
        '', '## Cross-workload observations', '',
        f'1. Final deployment 是否不同：{cross["final_deployments_differ"]}。',
        f'2. N_G 是否不同：{cross["N_G_differs"]}；观测值={[r["N_G"] for r in complete]}。',
        f'3. N_S 是否不同：{cross["N_S_differs"]}；观测值={[r["N_S"] for r in complete]}。',
        f'4. Workload-dependent G/S allocation：{cross["workload_dependent_allocation_observed"]}；只描述本 frozen space 的观测结果。',
        f'5. 是否选择 replica=2：{cross["any_selected_x2"]}；全部 x=2 proposal 的 rank、evaluated/accepted 状态见 replica_activation.json。',
        '6. '+('在当前三个 workload 和单 GPU 资源条件下，搜索倾向于增加 model diversity，而不是复制同模型。'
                if len(complete) == 3 and not cross['any_selected_x2'] and any(sum(r['final'].values()) > sum(r['seed'].values()) for r in complete) else
                'Replica expansion 与 model diversity 按实际 accepted transitions 分别记录，不从未测的状态推断普遍优劣。'),
        f'7. Empirical calibration 改变 Top-B：{cross["calibration_changed_Top_B"]}；CPU counterfactual 匹配实际 incumbent/visited history/evidence/memory，没有 linear GPU campaign。',
        f'8. Runtime feedback rollback analytical Top-B：{len(cross["analytical_Top_B_runtime_rollbacks"])} 次；包括不改善 incumbent，以及其他 Top-B candidate 更好两种原因，全部保留。',
        f'9. 同一 evaluated deployment 跨 workload groups={len(cross["same_evaluated_deployment_across_workloads"])}；'
        f'接受/回滚 decision 不同的 groups={sum(g["acceptance_decisions_differ"] for g in cross["same_evaluated_deployment_across_workloads"])}。'
        '不能把仅 mean/P95 数字不同当作相反的接受方向；每组 workload/runtime/outcome 完整列在 summary JSON。',
        f'10. Memory constraint 实际排除 candidate：{len(cross["memory_rejections"])} 次；只统计 memory rejection，tabu/coverage 不混入。',
        '', '## Independent confirmation', '', '| Workload | Seed mean ms | Final mean ms | Mean gain | Objective direction |', '|---|---:|---:|---:|---|']
    confirmations = read(OUT / 'confirmation_summary.json')
    for name, r in confirmations.items():
        lines.append(f'| {name} | {r["seed_mean_e2e_ms"]:.3f} | {r["final_mean_e2e_ms"]:.3f} | {100*r["relative_mean_gain"]:.2f}% | {r["objective_direction_maintained"]} |')
    lines += ['', '每个 workload 的 seed/final 各 2 次独立 confirmation，不影响搜索决策；W2 复用原有 2+2。',
        '', f'新 search physical attempts={summary["total_new_GPU_search_attempts"]}；新 confirmation attempts={summary["total_new_GPU_confirmation_attempts"]}；合计上限=44。'
        f'Enumeration calls={summary["enumeration_calls"]}；audit errors={summary["audit_errors"]}。',
        'Qwen presence 由唯一 verified VQA capability 强制，不描述成算法基于性能主动选择。'
        'Qwen/Whisper 双副本因子来自 ASR；mixed-task transfer 和异构 GPU interference 仍是 planning surrogate，runtime feedback 用于检验候选。',
        '没有 global optimum / universal G/S count / 2 replicas always worse / Specialists always better 的结论。'
        '只报告 frozen W1/W2/W3、candidate pool v2、当前 RTX 4090、evidence-bounded replica space、Top-B 和预算下的观测 allocation。',
        'Evidence bounds：qwen / whisper-small / resnet50≤2，其余≤1。没有 new workloads、dynamic adaptation、new scaling campaign 或 final paper benchmark。完成后停止。']
    (OUT / 'module7c4_summary.md').write_text('\n'.join(lines) + '\n')


def artifact_manifest():
    paths = [p for p in OUT.rglob('*') if p.is_file() and p.suffix in ('.json', '.md', '.csv', '.sha256') and p.name != 'artifact_manifest.json']
    save_json(OUT / 'artifact_manifest.json', {'module': '7C-4', 'artifacts': {str(p.relative_to(OUT)): ref(p) for p in sorted(paths)}})


if __name__ == '__main__':
    report()
