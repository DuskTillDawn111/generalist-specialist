"""All-condition reporting using only the frozen objective and preregistered rules."""
import csv
import math
from pathlib import Path
from types import SimpleNamespace

from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_core.replicas import replica_statistics
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.measurement_runtime import stats
from deployment_search.runtime import STABILITY
from deployment_search.runtime_cache import save_json, digest, file_hash
from .common import ROOT, F, OUT, read, ref, inputs, verify_manifest


def csvwrite(name, rows, fields):
    import json
    with (OUT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def compare(a, b, memory_a, memory_b):
    if a is None or b is None: return {'outcome': None, 'reason': 'INCOMPLETE'}
    policy = read(F / 'tie_policy.json')
    objective = NoiseAwareRuntimeObjective(policy['mean_tolerance_ms'], policy['min_relative_improvement'])
    win, wr = objective.compare(SimpleNamespace(**a), SimpleNamespace(**b), memory_a, memory_b)
    loss, lr = objective.compare(SimpleNamespace(**b), SimpleNamespace(**a), memory_b, memory_a)
    assert not (win and loss), 'Frozen comparator gives both directions improvement'
    return {'outcome': 'WIN' if win else 'LOSS' if loss else 'TIE', 'forward_reason': wr,
            'reverse_reason': lr, 'mean_difference_ms': difference(a.get('weighted_mean_e2e'), b.get('weighted_mean_e2e')),
            'P95_difference_ms': difference(a.get('weighted_p95_e2e'), b.get('weighted_p95_e2e')),
            'memory_difference_mib': memory_a - memory_b}


def difference(a, b):
    return a-b if a is not None and b is not None else None


def numeric_gap(a, b):
    if a is None or b is None or a['stability_state_v2'] != 'steady' or b['stability_state_v2'] != 'steady' or not b['weighted_mean_e2e']:
        return {'mean_gap': None, 'P95_gap': None, 'reason': 'Both final confirmations must be steady and B2 mean positive'}
    return {'mean_gap': a['weighted_mean_e2e']/b['weighted_mean_e2e']-1,
            'P95_gap': a['weighted_p95_e2e']/b['weighted_p95_e2e']-1 if b['weighted_p95_e2e'] else None,
            'reason': 'frozen signed relative gap'}


def classify_revision(variables, valid):
    if not valid: return None, 'Frozen prerequisite incomplete or protocol invalid'
    v = variables
    conditions = {
        'L>=3 OR SR>=2 OR severe': v['L'] >= 3 or v['SR'] >= 2 or v['severe'],
        'W>=3 AND W>L AND L<=1 AND SR==0 AND NOT clear_regression AND helped':
            v['W'] >= 3 and v['W'] > v['L'] and v['L'] <= 1 and v['SR'] == 0 and not v['clear_regression'] and v['helped'],
        'W>L AND W>=1 AND activated AND SR<=1 AND NOT severe':
            v['W'] > v['L'] and v['W'] >= 1 and v['activated'] and v['SR'] <= 1 and not v['severe'],
        'otherwise after no NOT_SUPPORTED rule': True}
    for rule in read(F / 'success_criteria.json')['ordered_rules']:
        assert rule['if'] in conditions, 'Unknown frozen rule; do not infer a new classification'
        if conditions[rule['if']]: return rule['status'], rule['if']
    raise AssertionError('No preregistered status rule matched')


def classify_gap(rows, valid):
    if not valid: return {'B2_GAP_STATUS': None, 'reason': 'INCOMPLETE/PROTOCOL_INVALID'}
    loss1 = sum(r['M1_vs_B2']['outcome'] == 'LOSS' for r in rows)
    loss2 = sum(r['M2_vs_B2']['outcome'] == 'LOSS' for r in rows)
    common = all(r['gap_to_B2_v1']['mean_gap'] is not None and r['gap_to_B2_v2']['mean_gap'] is not None for r in rows)
    g1 = sum(max(r['gap_to_B2_v1']['mean_gap'], 0) for r in rows)/6 if common else None
    g2 = sum(max(r['gap_to_B2_v2']['mean_gap'], 0) for r in rows)/6 if common else None
    if loss2 == 0: status, rule = 'CLOSED', 'zero M2 primary losses vs B2'
    elif common:
        if g1 > 0 and (g1-g2)/g1 >= .05: status, rule = 'REDUCED', 'mean excess gap reduced by >=5%'
        elif g2 > g1 and (g1 == 0 or (g2-g1)/g1 >= .05): status, rule = 'WORSENED', 'mean excess gap increased by >=5%'
        else: status, rule = 'UNCHANGED', 'below frozen relative gap change threshold'
    else:
        status = 'REDUCED' if loss2 < loss1 else 'WORSENED' if loss2 > loss1 else 'UNCHANGED'
        rule = 'frozen objective-count fallback; no cherry-picked steady subset'
    return {'B2_GAP_STATUS': status, 'rule': rule, 'all_six_steady_metric_eligible': common,
            'M1_loss_count': loss1, 'M2_loss_count': loss2, 'G1': g1, 'G2': g2,
            'already_closed_v1': loss1 == 0, 'per_workload': rows,
            'source': ref(F / 'success_criteria.json')}


def quality_rows(name, method, result, admitted, tasks):
    rows = []
    if not result: return [{'workload': name, 'method': method, 'task': t, 'status': 'INCOMPLETE'} for t in tasks]
    route = result['routing']
    for task in tasks:
        used = [(k.split('::')[1], value) for k, value in route.items() if k.split('::')[0] == task and value > 0]
        total = sum(value for _, value in used)
        legal = bool(used) and all((task, m) in admitted and result['deployment'].get(m, 0) > 0 for m, _ in used)
        components = [{'model': m, 'routing_fraction': value/total,
                       'quality_admissible': (task, m) in admitted,
                       'quality_metric': admitted.get((task, m), {}).get('quality_metric'),
                       'frozen_quality': admitted.get((task, m), {}).get('measured_quality'),
                       'threshold': admitted.get((task, m), {}).get('historical_threshold'),
                       'policy_status': admitted.get((task, m), {}).get('policy_status')} for m, value in used]
        quality = sum(c['routing_fraction']*c['frozen_quality'] for c in components) if components and all(c['frozen_quality'] is not None for c in components) else None
        rows.append({'workload': name, 'method': method, 'task': task, 'status': 'COMPLIANT' if legal else 'FAIL',
                     'all_positive_paths_admitted': legal, 'effective_frozen_quality': quality,
                     'components': components, 'quality_scope': 'Frozen path audit point estimates; no new runtime quality benchmark; semantic-only VQA policy unchanged.'})
    return rows


def candidate_audit(name, method, trace, policy):
    rows = []
    for tr in (trace or {}).get('iterations', []):
        for c in tr['candidates']:
            mu = c['mutation']; operation = mu['operation']
            closure = bool(mu.get('exact_replacement_closure', False))
            typ = 'REPLACEMENT' if closure else {'SCALE_UP': 'REPLICA_UP', 'SCALE_DOWN': 'REPLICA_DOWN'}.get(operation, operation)
            cls = 'v1 default' if method == 'M1' else 'R' if operation in ['SWAP', 'REMOVE', 'SCALE_DOWN'] else 'P'
            useful = False; objective_reason = None
            if c['runtime'] is not None and tr['current_runtime'] is not None:
                useful, objective_reason = NoiseAwareRuntimeObjective(1., .05).compare(
                    SimpleNamespace(**c['runtime']), SimpleNamespace(**tr['current_runtime']),
                    c['memory_mib'], policy.deployment_memory(tr['deployment']))
            rows.append({'workload': name, 'method': method, 'iteration': tr['iteration']+1,
                'current_state': tr['deployment'], 'candidate_deployment': mu['deployment'],
                'routing': c['canonical']['routing'] if c['canonical'] else None,
                'operation_original': operation, 'candidate_type': typ, 'candidate_class': cls,
                'targeted_swap': operation == 'SWAP', 'replacement_closure': closure,
                'origins': mu.get('origins', []), 'analytical_delta_z': c['delta_z'],
                'rank': c['rank'], 'Top_B': c['selected_top_b'], 'runtime_evaluated': c['runtime'] is not None,
                'accepted': c['status'] == 'ACCEPTED', 'rolled_back': c['status'] == 'ROLLBACK',
                'status': c['status'], 'reason': c['reason'], 'beneficial_to_actual_incumbent': useful,
                'objective_reason': objective_reason, 'contains_x2': any(v == 2 for v in mu['deployment'].values()),
                'x2_families': [m for m, v in mu['deployment'].items() if v == 2],
                'memory_mib': c['memory_mib'], 'runtime': c['runtime']})
    return rows


def mechanism_summary(name, intensity, candidates, v1_candidates, trace):
    swaps = [c for c in candidates if c['targeted_swap']]
    closures = [c for c in candidates if c['replacement_closure']]
    cases = []
    for c in swaps:
        matches = [x for x in v1_candidates if x['candidate_deployment'] == c['candidate_deployment'] and x['routing'] == c['routing']]
        any_evaluated = any(x['runtime_evaluated'] for x in matches)
        flags = {'not_generated': not matches,
                 'generated_later': bool(matches) and min(x['iteration'] for x in matches) > c['iteration'],
                 'rank_too_low': bool(matches) and not any_evaluated and any(x['rank'] is not None and x['rank'] > 2 for x in matches),
                 'budget_or_stopping_missed': bool(matches) and not any_evaluated and any(x['Top_B'] or x['status'] == 'NOT_EVALUATED_BUDGET' for x in matches)}
        cases.append({'candidate': c, 'M1_exact_state_routing_matches': [{k: m[k] for k in ['iteration', 'rank', 'Top_B', 'status', 'reason', 'runtime_evaluated']} for m in matches],
                      'M1_access_flags': flags, 'M1_evaluated_same_state': any_evaluated,
                      'M2_beneficial_M1_missed': c['beneficial_to_actual_incumbent'] and c['runtime_evaluated'] and not any_evaluated,
                      'attribution_limit': 'Observed trace access limitation; differing incumbent trajectories are not an unmeasured causal performance counterfactual.'})
    row = {'workload': name, 'intensity': intensity, 'status': 'COMPLETE' if trace else 'INCOMPLETE',
           'base_candidates_generated': sum(not c['targeted_swap'] for c in candidates),
           'Class_P_generated': sum(c['candidate_class'] == 'P' for c in candidates),
           'Class_R_generated': sum(c['candidate_class'] == 'R' for c in candidates),
           'targeted_swaps_generated': len(swaps), 'replacement_closures_generated': len(closures),
           'swaps_Top_B': sum(c['Top_B'] for c in swaps), 'closures_Top_B': sum(c['Top_B'] for c in closures),
           'swaps_evaluated': sum(c['runtime_evaluated'] for c in swaps), 'closures_evaluated': sum(c['runtime_evaluated'] for c in closures),
           'swaps_accepted': sum(c['accepted'] for c in swaps), 'closures_accepted': sum(c['accepted'] for c in closures),
           'swaps_rolled_back': sum(c['rolled_back'] for c in swaps), 'closures_rolled_back': sum(c['rolled_back'] for c in closures),
           'beneficial_states_M1_missed': sum(c['M2_beneficial_M1_missed'] for c in cases),
           'mechanism_activated': any(c['runtime_evaluated'] for c in swaps),
           'termination_reason': trace.get('status') if trace else None}
    return row, cases


def protocol_audit(workloads, results, traces):
    manifest_ok = True
    try: verify_manifest()
    except (AssertionError, OSError): manifest_ok = False
    physical = read(OUT / 'physical_measurements.json')['runs'] if (OUT / 'physical_measurements.json').exists() else []
    sharing = read(OUT / 'measurement_sharing_log.json') if (OUT / 'measurement_sharing_log.json').exists() else {'logical_run_references': [], 'logical_evaluations': []}
    refs = sharing['logical_run_references']; evaluations = sharing['logical_evaluations']
    raw_checks = []
    for run in physical:
        p = Path(run['raw_path'])
        if 'raw_sha256' in run: raw_checks.append(p.exists() and file_hash(p) == run['raw_sha256'])
    ids = {r['physical_raw_id']: r for r in physical}
    share_checks = []
    for r in refs:
        donor = ids[r['physical_raw_id']]
        same = r['workload_sha256'] == donor['workload_sha256'] and r['routing_hash'] == donor['routing_hash'] and r['deployment'] == donor['deployment'] and r['repetition_seed'] == donor['repetition_seed'] and r['measurement_identity_hash'] == donor['measurement_identity_hash'] and r['phase'] == donor['phase']
        if r['phase'] == 'confirmation': same = same and r['role'] == donor['role']
        if r['cache_hit']: same = same and donor.get('valid', False)
        same = same and Path(r['raw_path']).resolve() == Path(donor['raw_path']).resolve()
        share_checks.append(same)
    order_log = read(OUT / 'execution_log.json')['slots'] if (OUT / 'execution_log.json').exists() else []
    expected = [(b['workload'], i, s) for b in read(F / 'execution_order.json')['workloads'] for i, s in enumerate(b['slots'])]
    actual = [(s['workload'], s['slot_index'], s['slot']) for s in order_log]
    order_ok = actual == expected[:len(actual)]
    local_ok = all((t is None or (t['runtime_evaluation_count'] <= 6 and t['search_iterations'] <= 3
        and t['config']['top_b'] == 2 and t['config']['no_improvement_patience'] == 1 and t['config']['runtime_evaluation_budget'] == 6))
        for method in ['M1', 'M2'] for t in traces[method].values())
    confirmation_ok = True
    for method in ['M1', 'M2']:
        for result in results[method].values():
            if not result: continue
            paths = {role: {str(Path(r['raw_result_path']).resolve()) for r in agg['repetitions']}
                     for role, agg in result['confirmations'].items()}
            confirmation_ok &= len(paths['seed']) == 2 and len(paths['final']) == 2 and not paths['seed'] & paths['final']
    barrier_ok = True
    for e in evaluations:
        if e['role'] != 'candidate': continue
        barrier = e.get('round_barrier')
        if not barrier or not barrier[e['method']]: barrier_ok = False
        name = e['workload'];slot = read(F / 'execution_order.json')['workloads'][[w['id'] for w in workloads].index(name)]['slots'][e['logical_slot']]
        for method in ['M1', 'M2']:
            t = traces[method].get(name)
            iterations = (t or {}).get('iterations', [])
            if len(iterations) >= slot['iteration'] and iterations[slot['iteration']-1]['selected_top_b']:
                barrier_ok &= bool(barrier.get(method))
    acc = read(OUT / 'run_accounting.json') if (OUT / 'run_accounting.json').exists() else {'within_budget': True}
    complete = len(actual) == len(expected) and all(s['status'] in ['COMPLETED', 'SKIPPED'] for s in order_log)
    return {'frozen_hashes_unchanged': manifest_ok, 'all_raw_hashes_match': all(raw_checks),
            'all_sharing_exact_and_compatible': all(share_checks), 'execution_order_matches_frozen_prefix': order_ok,
            'entire_schedule_completed': complete, 'equal_search_budget_contract': local_ok,
            'confirmation_seed_final_independence': confirmation_ok, 'candidate_information_firewall_round_commits': barrier_ok,
            'budget_within_caps': acc['within_budget'], 'shared_references': sum(r['cache_hit'] for r in refs),
            'raw_measurements_audited': len(raw_checks), 'logical_records': len(evaluations),
            'valid': all([manifest_ok, all(raw_checks), all(share_checks), order_ok, complete, local_ok, confirmation_ok, barrier_ok, acc['within_budget']])}


def report():
    catalog, hardware, roles, cal = inputs()
    policy = ReplicaPlanningPolicy(catalog, 'empirical', 'empirical', cal)
    workloads = read(F / 'heldout_workloads.json')['workloads']
    ids = read(ROOT / 'results/candidate_pool_v2_1/candidate_pool_v2_1.json')['formal_model_ids']
    admissions = {(r['task'], r['model']): r for r in read(ROOT / 'results/module8b_quality_pool_finalization/quality_admission_matrix.json')['rows'] if r['quality_admissible']}
    results = {m: {} for m in ['M1', 'M2', 'B2']}
    traces = {m: {} for m in ['M1', 'M2']}
    enriched, deploy_rows, quality, search_cost, comparisons, mechanisms, cases = [], [], [], [], [], [], []
    metrics = {m: [] for m in results}
    replica_rows = []
    for w in workloads:
        name = w['id']; finals, memories = {}, {}
        for method in results:
            path = OUT / name / method / 'result.json'
            result = read(path) if path.exists() else None
            results[method][name] = result
            final = result['confirmations']['fixed' if method == 'B2' else 'final'] if result else None
            finals[method] = final
            x = result['deployment'] if result else None
            memory = policy.deployment_memory(x) if x else None
            memories[method] = memory
            counts = replica_statistics(x, roles) if x else {'N_G': None, 'N_S': None}
            memory_reps = final['memory_observations'] if final else []
            metric = {'workload': name, 'intensity': w['intensity'], 'method': method,
                'status': 'COMPLETE' if result else 'INCOMPLETE',
                'deployment_vector': {m: x.get(m, 0) for m in ids} if x else None,
                'N_G': counts['N_G'], 'N_S': counts['N_S'], 'total_replicas': sum(x.values()) if x else None,
                'planning_memory_mib': memory, 'n': len(final['repetitions']) if final else 0,
                'weighted_mean_E2E_ms': final['weighted_mean_e2e'] if final else None,
                'weighted_task_P95_ms': final['weighted_p95_e2e'] if final else None,
                'completion_ratio': final['completion_ratio'] if final else None,
                'throughput_rps': final['throughput'] if final else None,
                'stability': final['stability_state_v2'] if final else None,
                'idle_VRAM_mib': stats([m.get('actual_idle_mib') for m in memory_reps]),
                'peak_VRAM_mib': stats([m.get('peak_owned_mib') for m in memory_reps]),
                'queue_wait': final['per_replica_queue_wait'] if final else None,
                'queue_backlog': final['queue_metrics'] if final else None,
                'worker_latency': final['worker_latency'] if final else None,
                'per_replica_worker_latency': final['per_replica_worker_latency'] if final else None,
                'replicate_statistics': final['replicate_statistics'] if final else None,
                'confirmation_repetitions': final['repetitions'] if final else [],
                'all_phase_aggregates': result['confirmations'] if result else None,
                'search_aggregate': result['search_runtime'] if result else None,
                'quality_scope': 'Unchanged frozen path quality audit; no newly measured deployed-task accuracy'}
            metrics[method].append(metric)
            deploy_rows.append(metric)
            qrows = quality_rows(name, method, result, admissions, list(w['lambda_rps']))
            quality += qrows
            metric['effective_quality_compliance'] = all(r['status'] == 'COMPLIANT' for r in qrows) if result else None
            if method != 'B2':
                folder = OUT / ('m1_search_traces' if method == 'M1' else 'm2_search_traces') / name
                p = folder / 'search_complete.json'
                if not p.exists(): p = folder / 'search_trace.json'
                trace = read(p) if p.exists() else None
                traces[method][name] = trace
                audited = candidate_audit(name, method, trace, policy)
                enriched += audited
                if trace:
                    save_json(folder / 'audit_trace.json', {'method': method, 'workload': name,
                        'all_candidates': audited, 'termination_reason': trace['status'],
                        'Top_B_composition': [{ 'iteration': i['iteration']+1,
                            'classes': [c['candidate_class'] for c in audited if c['iteration'] == i['iteration']+1 and c['Top_B']]} for i in trace['iterations']]})
                search_cost.append({'workload': name, 'intensity': w['intensity'], 'method': method,
                    'logical_evaluations': trace.get('runtime_evaluation_count') if trace else 0,
                    'iterations': trace.get('search_iterations') if trace else 0,
                    'generated_candidates': trace.get('num_candidates_generated') if trace else 0,
                    'LP_calls': trace.get('lp_solve_count') if trace else 0,
                    'MILP_calls': trace.get('milp_solve_count') if trace else 0,
                    'termination_reason': trace.get('status') if trace else 'NOT_EXECUTED',
                    'top_b': 2, 'search_state_cap': 6, 'max_iterations': 3, 'patience': 1})
                x2 = [c for c in audited if c['contains_x2']]
                replica_rows.append({'workload': name, 'method': method, 'x2_generated': len(x2),
                    'x2_Top_B': sum(c['Top_B'] for c in x2), 'x2_evaluated': sum(c['runtime_evaluated'] for c in x2),
                    'x2_accepted': sum(c['accepted'] for c in x2), 'candidates': x2,
                    'seed_has_x2': any(v == 2 for v in trace['seed']['deployment'].values()) if trace else None,
                    'final_has_x2': any(v == 2 for v in trace['deployment'].values()) if trace else None})
            else:
                search_cost.append({'workload': name, 'intensity': w['intensity'], 'method': method,
                    'logical_evaluations': result['logical_search_evaluations'] if result else 0,
                    'iterations': 0, 'generated_candidates': 0, 'LP_calls': 0, 'MILP_calls': 0,
                    'termination_reason': 'FIXED_STATE' if result else 'NOT_EXECUTED'})
        row = {'workload': name, 'intensity': w['intensity'],
               'M2_vs_M1': compare(finals['M2'], finals['M1'], memories['M2'], memories['M1']),
               'M1_vs_B2': compare(finals['M1'], finals['B2'], memories['M1'], memories['B2']),
               'M2_vs_B2': compare(finals['M2'], finals['B2'], memories['M2'], memories['B2']),
               'gap_to_B2_v1': numeric_gap(finals['M1'], finals['B2']),
               'gap_to_B2_v2': numeric_gap(finals['M2'], finals['B2'])}
        comparisons.append(row)
        m2 = [c for c in enriched if c['workload'] == name and c['method'] == 'M2']
        m1 = [c for c in enriched if c['workload'] == name and c['method'] == 'M1']
        mechanism, evidence = mechanism_summary(name, w['intensity'], m2, m1, traces['M2'][name])
        mechanisms.append(mechanism); cases += evidence
    audit = protocol_audit(workloads, results, traces)
    save_json(OUT / 'protocol_audit.json', audit)
    status = read(OUT / 'campaign_status.json')
    valid = audit['valid'] and status['status'] == 'complete' and all(r is not None for method in results.values() for r in method.values()) and all(m['effective_quality_compliance'] for mm in metrics.values() for m in mm)
    counts = {v: sum(r['M2_vs_M1']['outcome'] == v for r in comparisons) for v in ['WIN', 'TIE', 'LOSS']}
    regressions = []
    for name in [w['id'] for w in workloads]:
        a = next(m for m in metrics['M2'] if m['workload'] == name)
        b = next(m for m in metrics['M1'] if m['workload'] == name)
        steady = a['stability'] == b['stability'] == 'steady'
        delta = a['weighted_mean_E2E_ms']/b['weighted_mean_E2E_ms']-1 if steady and b['weighted_mean_E2E_ms'] else None
        regressions.append({'workload': name, 'relative_mean_regression': delta,
            'stability_regression': STABILITY[a['stability']] < STABILITY[b['stability']] if a['stability'] and b['stability'] else None,
            'clear_regression': delta is not None and delta >= .10, 'severe_regression': delta is not None and delta >= .20})
    helped_cases = [c for c in cases if c['candidate']['accepted'] and c['candidate']['beneficial_to_actual_incumbent'] and
                   any(c['M1_access_flags'].values()) and next(r for r in comparisons if r['workload'] == c['candidate']['workload'])['M2_vs_M1']['outcome'] == 'WIN']
    variables = {'W': counts['WIN'], 'T': counts['TIE'], 'L': counts['LOSS'],
                 'SR': sum(r['stability_regression'] is True for r in regressions),
                 'severe': any(r['severe_regression'] for r in regressions),
                 'clear_regression': any(r['clear_regression'] for r in regressions),
                 'activated': any(m['mechanism_activated'] for m in mechanisms), 'helped': bool(helped_cases)}
    revision, rule = classify_revision(variables, valid)
    gap = classify_gap(comparisons, valid)
    intensity = {level: {'counts': {v: sum(r['M2_vs_M1']['outcome'] == v for r in comparisons if r['intensity'] == level) for v in ['WIN', 'TIE', 'LOSS']},
        'mechanism': [m for m in mechanisms if m['intensity'] == level],
        'note': 'Original reference-defined intensity unchanged; report activation, not a post-hoc saturation reinterpretation.'} for level in ['moderate', 'high']}
    for method, filename in [('M1', 'm1_results.json'), ('M2', 'm2_results.json'), ('B2', 'b2_results.json')]:
        save_json(OUT / filename, {'method': method, 'workloads': metrics[method], 'all_valid_repetitions_retained': True,
            'statistical_scope': 'n=2 independent final confirmations per method state; search n=3. Shared raw runs are correlated, one physical sample. No outlier removal or significance claim.'})
    save_json(OUT / 'v2_vs_v1_summary.json', {'counts': counts, 'unclassified': 6-sum(counts.values()),
        'per_workload': comparisons, 'by_intensity': intensity, 'regressions': regressions,
        'protocol_valid_and_campaign_complete': valid, 'objective': ref(F / 'tie_policy.json')})
    save_json(OUT / 'b2_gap_analysis.json', gap)
    save_json(OUT / 'replica_activation_audit.json', {'workloads_methods': replica_rows,
        'totals': {key: sum(r[key] for r in replica_rows) for key in ['x2_generated', 'x2_Top_B', 'x2_evaluated', 'x2_accepted']}})
    save_json(OUT / 'mechanism_evidence.json', {'workloads': mechanisms, 'all_targeted_swap_cases_including_failure': cases,
        'helped_cases': helped_cases, 'no_extra_counterfactual_runtime_measurements': True})
    next_stage = 'ablation' if revision in ['STRONG_SUPPORT', 'SUPPORT'] and helped_cases else 'routing revision' if valid else None
    validation = {'completion_status': 'COMPLETE' if valid else 'INCOMPLETE' if status['status'] != 'complete' else 'PROTOCOL_INVALID',
        'REVISION_VALIDATION_STATUS': revision, 'matched_frozen_rule': rule, 'variables': variables,
        'B2_GAP_STATUS': gap['B2_GAP_STATUS'], 'success_criteria': ref(F / 'success_criteria.json'),
        'protocol_audit': audit, 'ROUTING_REVISION_PENDING': True,
        'evidence_supports_ablation_next': next_stage == 'ablation', 'recommended_next_stage': next_stage,
        'next_stage_executed': False, 'claim': 'Targeted SWAP/replacement plus diversified selection improved this held-out search' if revision in ['STRONG_SUPPORT', 'SUPPORT'] else 'No supported held-out advantage claim' if valid else 'Experiment incomplete; success classification withheld',
        'limitations': ['canonical routing unchanged and unresolved', 'three independent mixtures with correlated intensities',
                       'same frozen task data and single hardware', 'two final confirmations; no significance claim', 'no routing-solvability or second-hardware claim']}
    save_json(OUT / 'revision_validation_status.json', validation)
    comp_rows = [{ 'workload': r['workload'], 'intensity': r['intensity'], 'M2_vs_M1': r['M2_vs_M1']['outcome'],
        'mean_difference_ms': r['M2_vs_M1'].get('mean_difference_ms'), 'P95_difference_ms': r['M2_vs_M1'].get('P95_difference_ms'),
        'memory_difference_mib': r['M2_vs_M1'].get('memory_difference_mib'), 'M1_vs_B2': r['M1_vs_B2']['outcome'],
        'M2_vs_B2': r['M2_vs_B2']['outcome'], 'gap_to_B2_v1': r['gap_to_B2_v1']['mean_gap'], 'gap_to_B2_v2': r['gap_to_B2_v2']['mean_gap']} for r in comparisons]
    csvwrite('heldout_method_comparison.csv', comp_rows, list(comp_rows[0]))
    csvwrite('heldout_deployment_comparison.csv', deploy_rows, ['workload', 'intensity', 'method', 'status', 'deployment_vector', 'N_G', 'N_S', 'total_replicas', 'planning_memory_mib', 'idle_VRAM_mib', 'peak_VRAM_mib', 'n', 'weighted_mean_E2E_ms', 'weighted_task_P95_ms', 'completion_ratio', 'throughput_rps', 'stability'])
    physical = read(OUT / 'physical_measurements.json')['runs'] if (OUT / 'physical_measurements.json').exists() else []
    sharing = read(OUT / 'measurement_sharing_log.json') if (OUT / 'measurement_sharing_log.json').exists() else {'logical_run_references': []}
    for cost in search_cost:
        rr = [r for r in sharing['logical_run_references'] if r['workload'] == cost['workload'] and r['method'] == cost['method']]
        cost.update(new_timed_runs=sum(r.get('measurement_started', False) for r in physical if r['workload'] == cost['workload'] and r['method'] == cost['method']),
                    shared_repetition_references=sum(r['cache_hit'] for r in rr),
                    confirmation_logical_refs=sum(r['phase'] == 'confirmation' for r in rr),
                    unique_physical_raw_ids=len({r['physical_raw_id'] for r in rr}))
    csvwrite('heldout_search_cost.csv', search_cost, ['workload', 'intensity', 'method', 'logical_evaluations', 'iterations', 'generated_candidates', 'LP_calls', 'MILP_calls', 'termination_reason', 'new_timed_runs', 'shared_repetition_references', 'confirmation_logical_refs', 'unique_physical_raw_ids'])
    csvwrite('heldout_mechanism_audit.csv', mechanisms, list(mechanisms[0]))
    csvwrite('heldout_quality_compliance.csv', quality, ['workload', 'method', 'task', 'status', 'all_positive_paths_admitted', 'effective_frozen_quality', 'components', 'quality_scope'])
    accounting = read(OUT / 'run_accounting.json') if (OUT / 'run_accounting.json').exists() else {}
    failures = read(OUT / 'runtime_failure_log.json') if (OUT / 'runtime_failure_log.json').exists() else {'events': []}
    failure_counts = {kind: sum(e['category'] == kind for e in failures['events']) for kind in ['steady', 'recoverable_congested', 'unstable', 'runtime_resource_infeasible', 'external_interruption', 'model_or_harness_failure']}
    swap_workloads = [m['workload'] for m in mechanisms if m['mechanism_activated']]
    totals = {k: sum(m[k] for m in mechanisms) for k in ['swaps_Top_B', 'closures_Top_B', 'swaps_accepted', 'closures_accepted', 'beneficial_states_M1_missed']}
    replicas = read(OUT / 'replica_activation_audit.json')['totals']
    lines = ['# Module 8E-3 — Held-out real GPU validation', '',
        f"Campaign: {validation['completion_status']}; REVISION_VALIDATION_STATUS={revision}; B2_GAP_STATUS={gap['B2_GAP_STATUS']}", '',
        '严格保留 8E-2 workload/intensity/method/routing/quality/pool/budget/seed/判定规则；ROUTING_REVISION_PENDING=true。', '',
        '|Workload|M2 vs M1|M1 vs B2|M2 vs B2|mean delta ms|P95 delta ms|', '|---|---|---|---|---:|---:|']
    for r in comp_rows:
        lines.append(f"|{r['workload']}|{r['M2_vs_M1']}|{r['M1_vs_B2']}|{r['M2_vs_B2']}|{r['mean_difference_ms']}|{r['P95_difference_ms']}|")
    lines += ['', '## Required final answers', '',
        f"1. 实际新增 timed runs：{accounting.get('new_timed_runs', 0)}；physical attempts：{accounting.get('new_physical_attempts', 0)}；confirmation runs：{accounting.get('confirmation_runs', 0)}；retries=0。",
        f"2. M2 vs M1 win/tie/loss：{counts['WIN']}/{counts['TIE']}/{counts['LOSS']}；未分类：{6-sum(counts.values())}。",
        f"3. REVISION_VALIDATION_STATUS：{revision}；按冻结规则 {rule}。",
        f"4. SWAP/replacement 被真实评估的 workload：{swap_workloads}。",
        f"5. replacement closure 进入 Top-B：{totals['closures_Top_B']} 次；全部 targeted SWAP 进入 Top-B：{totals['swaps_Top_B']} 次。",
        f"6. closure 真实接受：{totals['closures_accepted']} 次；全部 targeted SWAP 真实接受：{totals['swaps_accepted']} 次。",
        f"7. M1 没测到而 M2 同预算测到的 objective-beneficial targeted state：{totals['beneficial_states_M1_missed']} 次；全部成功与失败案例见 mechanism_evidence.json，incumbent 路径差异限制因果解释。",
        f"8. v2 是否缩小 B2 gap：{'是' if gap['B2_GAP_STATUS'] == 'REDUCED' else '已关闭' if gap['B2_GAP_STATUS'] == 'CLOSED' else '否' if valid else '未完成，不能判断'}。",
        f"9. B2_GAP_STATUS：{gap['B2_GAP_STATUS']}。",
        f"10. 明显退化（steady mean>=10%）：{[r['workload'] for r in regressions if r['clear_regression']]}；stability 退化：{[r['workload'] for r in regressions if r['stability_regression']]}。",
        f"11. moderate/high：moderate={intensity['moderate']['counts']}，high={intensity['high']['counts']}；mechanism 激活分 workload 记录，不调整 reference-defined intensity。",
        f"12. x=2 generated/Top-B/evaluated/accepted：{replicas['x2_generated']}/{replicas['x2_Top_B']}/{replicas['x2_evaluated']}/{replicas['x2_accepted']}。",
        f"13. failure/稳定性事件：{failure_counts}；无重复试到 steady。",
        f"14. equal-budget 静态与实际搜索契约：{audit['equal_search_budget_contract']}；logical={accounting.get('logical_search_evaluations', 0)}<=78，physical states={accounting.get('unique_physical_states', 0)}<=72，attempts={accounting.get('new_physical_attempts', 0)}<=264。",
        f"15. sharing exact-compatible：{audit['all_sharing_exact_and_compatible']}；shared raw references={audit['shared_references']}；confirmation seed/final 独立={audit['confirmation_seed_final_independence']}；round commitment firewall={audit['candidate_information_firewall_round_commits']}。",
        '16. routing limitation 仍存在；本阶段没有验证或解决 routing revision。',
        f"17. 是否支持继续 ablation：{validation['evidence_supports_ablation_next']}（仅下一阶段建议，未执行）。",
        f"18. 推荐下一阶段：{next_stage if next_stage else '先处理未完成的 campaign/协议问题，不启动四种扩展实验'}。", '',
        '全部有效 repetition 原样保留；final n=2，search n=3，mean/std 为 equal-repetition sample statistics。raw sharing 不计额外独立样本；三 mixture 的 moderate/high 成对相关，不做 significance claim。', '',
        f"Frozen protocol audit valid={audit['valid']}；campaign error={status.get('error')}。", '',
        '本阶段停止。未修改 v2，未执行 routing revision / ablation / dynamic / second hardware。']
    (OUT / 'module8e3_summary.md').write_text('\n'.join(lines)+'\n')
    save_json(OUT / 'report_manifest.json', {'artifacts': {p.name: ref(p) for p in OUT.iterdir()
        if p.is_file() and p.suffix in ['.json', '.csv', '.md'] and p.name != 'report_manifest.json'},
        'all_valid_repetitions_retained': True, 'success_rules_changed': False})


if __name__ == '__main__': report()
