"""Explain the completed frozen screen without any new optimization or cost tuning."""
import collections
import math
from .common import *
from .replay import PRIMARY, MATCHED

EPS = 1e-9


def positive_residents(row):
    if not row['feasible']:
        return []
    return [m for m in row['replicas'] if math.fsum(v for p, v in row['routing'].items() if p.split('::')[1] == m) > EPS]


def pairs(rows, scope):
    result = []
    for a in [r for r in rows if r['count_scope'] == scope and r['Specialist_regime'] == E]:
        z = next(r for r in rows if r['count_scope'] == scope and r['Specialist_regime'] == H and r['workload'] == a['workload'] and r['budget'] == a['budget'])
        both = a['feasible'] and z['feasible']
        v = {'workload': a['workload'], 'budget': a['budget'], 'comparable_feasible': both,
             'E_feasible': a['feasible'], 'H_feasible': z['feasible'],
             'deployment_or_feasibility_different': a['feasible'] != z['feasible'] or (both and a['replicas'] != z['replicas']),
             'T1': False, 'T2': False, 'T3': False, 'T4': False, 'T5': False, 'T6': False}
        if both:
            new_splits = sorted(set(z['split_tasks']) - set(a['split_tasks']))
            consolidated = [m for m, ts in z['G_served_task_sets'].items() if len(ts) >= 2 and any(t not in a['G_served_task_sets'].get(m, []) for t in ts)]
            ap = positive_residents(a); zp = positive_residents(z)
            effective_a = sum(a['replicas'][m] for m in ap if m in a['Specialist_identities'])
            effective_z = sum(z['replicas'][m] for m in zp if m in z['Specialist_identities'])
            v.update(T1=z['N_S'] < a['N_S'], T2=z['Generalist_routed_fraction'] > a['Generalist_routed_fraction'] + EPS,
                     T3=bool(new_splits), T4=bool(consolidated), T5=z['N_G'] > a['N_G'], T6=a['N_G'] < 2 <= z['N_G'],
                     new_split_tasks=new_splits, new_consolidated_G_ids=consolidated,
                     new_second_G_identity=a['N_G_identities'] < 2 <= z['N_G_identities'],
                     E_NG=a['N_G'], H_NG=z['N_G'], E_NS=a['N_S'], H_NS=z['N_S'],
                     delta_NG=z['N_G'] - a['N_G'], delta_NS=z['N_S'] - a['N_S'],
                     E_G_share=a['Generalist_routed_fraction'], H_G_share=z['Generalist_routed_fraction'],
                     delta_G_share=z['Generalist_routed_fraction'] - a['Generalist_routed_fraction'],
                     delta_J_seconds=z['J_star_seconds'] - a['J_star_seconds'],
                     delta_safe_memory_mib=z['safe_memory_mib'] - a['safe_memory_mib'],
                     G_identity_changed=a['Generalist_identities'] != z['Generalist_identities'],
                     effective_E_NS=effective_a, effective_H_NS=effective_z,
                     effective_positive_flow_T1=effective_z < effective_a,
                     primary_effective_state_different=a['N_G'] != z['N_G'] or effective_z != effective_a or a['Generalist_identities'] != z['Generalist_identities'],
                     certificate_gap_seconds=max(a['global_gap_seconds'], z['global_gap_seconds']))
        result.append(v)
    return sorted(result, key=lambda p: (p['workload'], p['budget']))


def minimum_real_memory_cover(catalog_data, required_tasks, only_specialists=False):
    """Exact tiny set-cover DP over actual measured residency costs, ignoring capacity.

    This is a necessary memory bound, not a new structural/counterfactual replay.
    All numbers come directly from frozen measured memory tables.
    """
    tasks = sorted(required_tasks); full = (1 << len(tasks)) - 1
    dp = {0: (0., [])}
    for m, p in sorted(catalog_data['models'].items()):
        if only_specialists and p['role'] != 'specialist':
            continue
        mask = sum(1 << i for i, t in enumerate(tasks) if t in p['supported_tasks'])
        if not mask:
            continue
        costs = [float(v) for n, v in catalog_data['memory_table_mib'][m].items() if int(n) > 0]
        cost = catalog_data['safe_factor'] * min(costs)
        for covered, (current, names) in list(dp.items()):
            new = covered | mask
            if new not in dp or current + cost < dp[new][0]:
                dp[new] = (current + cost, names + [m])
    return {'required_tasks': tasks, 'minimum_safe_memory_mib': dp[full][0], 'cover_models': dp[full][1], 'ignores_capacity_so_is_necessary_lower_bound': True}


def attribution(rows, comparisons):
    frozen = read(OUT / 'screen_protocol_freeze.json')
    e = read(OUT / 'regime_e.json'); h = read(OUT / 'regime_h.json')
    existing = read(ROOT / 'results/module12_specialist_sensitivity/speed_vs_memory_attribution.json')
    details = []
    for p in comparisons:
        a = next(r for r in rows if r['count_scope'] == PRIMARY and r['Specialist_regime'] == E and r['workload'] == p['workload'] and r['budget'] == p['budget'])
        z = next(r for r in rows if r['count_scope'] == PRIMARY and r['Specialist_regime'] == H and r['workload'] == p['workload'] and r['budget'] == p['budget'])
        work = next(w for w in frozen['workloads'] if w['id'] == p['workload'])
        cover = minimum_real_memory_cover(h, [k for k, v in work['rates'].items() if v > 0])
        specialists_tasks = {path.split('::')[0] for path, v in a['routing'].items() if v > EPS and path.split('::')[1] in a['Specialist_identities']}
        min_h_s = minimum_real_memory_cover(h, specialists_tasks, only_specialists=True)
        e_g_mem = e['safe_factor'] * math.fsum(e['memory_table_mib'][m][str(a['replicas'][m])] for m in a['Generalist_identities'])
        pure_replacement_lower = e_g_mem + min_h_s['minimum_safe_memory_mib']
        record = {**p, 'absolute_real_task_cover_memory_bound': cover,
                  'absolute_memory_cover_exceeds_budget': cover['minimum_safe_memory_mib'] > a['memory_budget_mib'] + 1e-6,
                  'retain_E_G_counts_plus_min_H_S_task_cover_memory_mib': pure_replacement_lower,
                  'retain_E_G_and_S_task_coverage_exceeds_budget': pure_replacement_lower > a['memory_budget_mib'] + 1e-6,
                  'counterfactual_cost_scaling_or_axis_replays_run': False}
        if not z['feasible']:
            record['primary_source'] = 'MEMORY' if record['absolute_memory_cover_exceeds_budget'] else 'UNRESOLVED_CAPACITY_OR_MEMORY'
            record['explanation'] = 'Even a minimum measured residency cover, before imposing latency/capacity constraints, exceeds this budget. Additional replicas cannot repair this memory lower bound.'
        else:
            record['active_H_S_rho'] = {m: z['rho'][m] for m in z['Specialist_identities']}
            record['H_binding_capacity_models'] = [m for m, rho in z['rho'].items() if rho >= .8 - 1e-6]
            record['primary_source'] = 'BOTH_WITH_EMPIRICAL_COUNT_SPACE_EFFECT'
            record['explanation'] = 'Real heavier residency changes affordable counts; real slower service increases J and favors spreading load across affordable real Specialist queues. Existing versus new empirical count spaces also differ. No isolated latency-only or memory-only causal claim is identified.'
            if record['retain_E_G_and_S_task_coverage_exceeds_budget']:
                record['necessary_memory_effect'] = 'The E Generalist counts plus even minimum H Specialist task cover cannot fit; preserving both is impossible.'
            if p.get('delta_NG', 0) < 0:
                record['explanation'] += ' Generalist count decreases here; this is not a new Generalist consolidation/2G success.'
        details.append(record)
    matched = pairs(rows, MATCHED)
    return {'at_utc': now(), 'kind': 'MEASURED_REAL_COST_MECHANISTIC_ATTRIBUTION_NOT_CAUSAL_AXIS_IDENTIFICATION',
            'source_Module12': ref(ROOT / 'results/module12_specialist_sensitivity/speed_vs_memory_attribution.json'),
            'Module12_global_explanatory_evidence': {'latency_only': existing['latency_only'], 'memory_only': existing['memory_only'],
               'note': 'Module12 T names differ from14C. Those sensitivity results are explanatory planning evidence, not these models measured GPU effects.'},
            'context_attribution': details,
            'memory_proven_contexts': [p['workload'] + '/' + p['budget'] for p in details if p['absolute_memory_cover_exceeds_budget']],
            'memory_prevents_preserving_E_G_and_H_S_coverage_contexts': [p['workload'] + '/' + p['budget'] for p in details if p['retain_E_G_and_S_task_coverage_exceeds_budget']],
            'latency_only_proven_contexts': [],
            'latency_associated_examples': 'mix_03_I2/I3 uses all four real OCR identities and separate Whisper-medium LID plus MMS. Higher actual service demands explain queue parallelism; memory simultaneously changes G identity qwen->phi4mm at B3. No G routing increase or G/S split is observed.',
            'matched_count_raw_NS_reduction_contexts': [p['workload'] + '/' + p['budget'] for p in matched if p['T1']],
            'matched_count_positive_flow_NS_reduction_contexts': [p['workload'] + '/' + p['budget'] for p in matched if p.get('effective_positive_flow_T1')],
            'count_space_warning': 'Most primary NS reductions are not robust to matching all Specialist counts0/1. One raw matched reduction includes numerical near-zero residents; report positive-flow counts separately without editing the frozen solver or routes.',
            'no_new_counterfactual_optimization': True}


def profiling(rows):
    pool = read(OUT / 'heavy_candidate_pool.json')
    h = read(OUT / 'regime_h.json'); e = read(OUT / 'regime_e.json')
    paths = [p for task in pool['tasks'].values() for p in task]
    decisions = []
    for m, model in sorted(h['models'].items()):
        if model['role'] != 'specialist':
            continue
        primary = [r for r in rows if r['Specialist_regime'] == H and r['count_scope'] == PRIMARY and r['feasible'] and m in r['Specialist_identities']]
        matched_positive = [r for r in rows if r['Specialist_regime'] == H and r['count_scope'] == MATCHED and r['feasible'] and m in positive_residents(r)]
        numerical_only = [r for r in rows if r['Specialist_regime'] == H and r['feasible'] and m in r['Specialist_identities'] and m not in positive_residents(r)]
        active = bool(primary)
        checkpoint = next(p for p in paths if p['planning_id'] == m)
        known = sorted(int(n) for n in h['capacity_factors'][m] if int(n) > 0)
        # The secondary count restriction itself cannot make a new checkpoint
        # primary-relevant. Retain that observation without expanding the pool.
        need_new = active and 2 not in known
        decisions.append({'planning_id': m, 'model_id': checkpoint['model_id'], 'revision': checkpoint['revision'],
                          'tasks': model['supported_tasks'], 'decision': 'REPLICA_PROFILING_REQUIRED' if active else 'NO_REPLICA_PROFILING_REQUIRED',
                          'requires_new_measurement': need_new, 'first_missing_count_requested': 2 if need_new else None,
                          'known_empirical_counts': known, 'criterion_1_primary_optimum_residency': active,
                          'criterion_2_new_x2_boundary_witness': False,
                          'criterion_3_G_S_split_or_NG_witness': False,
                          'criterion_2_note': 'Unknown g(2)/M(2) cannot be converted into a numeric boundary effect. Criterion1 already suffices for every required model; no extra hypothetical count2 solve is used.',
                          'primary_optimum_contexts': [r['workload'] + '/' + r['budget'] for r in primary],
                          'matched_only_positive_flow_contexts': [r['workload'] + '/' + r['budget'] for r in matched_positive if not primary],
                          'numerical_near_zero_tie_contexts': [r['count_scope'] + '/' + r['workload'] + '/' + r['budget'] for r in numerical_only],
                          'reason': 'Appears in primary optimum; measure g(2) and true M(2) before allowing count2. Subsequent counts require new boundary evidence.' if need_new else
                                    'Appears in primary optimum, but empirical g(2) and memory(2) already exist. Reuse unchanged data; no duplicate GPU measurement requested.' if active else
                                    'No primary residency or direct active-boundary witness. A supplementary artificial all-S count cap or a numerical near-zero tie does not establish primary profiling need.',
                          'profiling_executed': False})
    return {'at_utc': now(), 'decisions': decisions,
            'new_measurement_checkpoints': [d for d in decisions if d['requires_new_measurement']],
            'reuse_existing_empirical_checkpoints': [d for d in decisions if d['decision'] == 'REPLICA_PROFILING_REQUIRED' and not d['requires_new_measurement']],
            'next_stage': 'Only first missing count2 for the six active new checkpoints. Do not start a full frontier; any count>=3 needs measured count2/boundary justification.',
            'unknown_unselected_g_gt1_does_not_prove_dominance': True,
            'no_GPU_profiling_run': True}


def main():
    rows = read(OUT / 'screen_rows.json'); primary = pairs(rows, PRIMARY); matched = pairs(rows, MATCHED)
    counts = {k: sum(p[k] for p in primary) for k in ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']}
    residents = {}
    for r in rows:
        if r['count_scope'] == PRIMARY and r['Specialist_regime'] == H and r['feasible']:
            residents[r['workload'] + '/' + r['budget']] = r['Specialist_identities']
    identity_sets = {tuple(v) for v in residents.values()}
    summary = {'at_utc': now(), 'screen': 'SINGLE_REPLICA_STRUCTURAL_SCREEN',
               'contexts': 27, 'E_feasible_contexts': sum(p['E_feasible'] for p in primary),
               'H_feasible_contexts': sum(p['H_feasible'] for p in primary),
               'both_feasible_contexts': sum(p['comparable_feasible'] for p in primary),
               'deployment_or_feasibility_different_contexts': sum(p['deployment_or_feasibility_different'] for p in primary),
               'different_deployment_both_feasible_contexts': sum(p['deployment_or_feasibility_different'] and p['comparable_feasible'] for p in primary),
               'event_context_counts': counts, 'NG_decrease_contexts': sum(p.get('delta_NG', 0) < 0 for p in primary),
               'NS_increase_contexts': sum(p.get('delta_NS', 0) > 0 for p in primary),
               'new_second_G_identity_contexts': sum(p.get('new_second_G_identity', False) for p in primary),
               'T7_observed': len(identity_sets) > 1, 'T7_distinct_H_identity_sets': len(identity_sets),
               'H_identity_sets_by_context': residents, 'comparisons': primary,
               'matched_count_comparisons': matched,
               'matched_count_raw_event_counts': {k: sum(p[k] for p in matched) for k in counts},
               'matched_count_effective_positive_flow_NS_reduction_contexts': sum(p.get('effective_positive_flow_T1', False) for p in matched),
               'frozen_Module9_numerical_failures': [{'scope': r['count_scope'], 'regime': r['Specialist_regime'], 'workload': r['workload'], 'budget': r['budget'], 'status': r['frozen_Module9_solver']['status']} for r in rows if r['feasible'] and not r['frozen_Module9_solver']['converged']],
               'primary_frozen_Module9_numerical_failures': sum(r['count_scope'] == PRIMARY and r['feasible'] and not r['frozen_Module9_solver']['converged'] for r in rows),
               'routing_increase_not_inferred_from_infeasibility': True,
               'not_final_deployment_or_multi_replica_optimality_evidence': True}
    save('transition_summary.json', summary)
    save('transition_attribution.json', attribution(rows, primary))
    prof = profiling(rows); save('replica_profiling_required.json', prof)
    structural_contexts = [p for p in primary if any(p[k] for k in ['T1', 'T2', 'T3', 'T5'])]
    gate = 'HEAVY_STRUCTURAL_TRANSITION_SUPPORTED' if len(structural_contexts) >= 2 else 'HEAVY_STRUCTURAL_TRANSITION_PARTIAL' if structural_contexts else 'HEAVY_STRUCTURAL_TRANSITION_NOT_OBSERVED'
    save('module14c_gate.json', {'at_utc': now(), 'gate': gate, 'complete': True,
        'frozen_gate_rule': read(OUT / 'screen_protocol_freeze.json')['gate_rule'],
        'supported_primary_structural_contexts': len(structural_contexts), 'events': counts,
        'interpretation': 'Supports Heavy-regime resource/composition sensitivity, principally NS changes and memory infeasibility. Does NOT support increased Generalist routing, new G/S split or a new second Generalist.',
        'count_space_caveat': 'Matched single-count comparison retains only one positive-flow NS reduction; most primary NS changes cannot be isolated from evidence-bound count differences.',
        'proceed_to_targeted_replica_measurement_worthwhile': True,
        'formal_GPU_deployment_campaign_ready': False,
        'formal_GPU_campaign_precondition': 'Measure missing g(2)/M(2) of six primary-active new checkpoints, freeze extended empirical catalog, then repeat CPU screen; no promise of future Generalist gains.',
        'GPU_runs': 0, 'g_gt1_runs': 0, 'new_model_searches': 0, 'deployment_outcome_used_for_pool_membership': False})
    print('GATE', gate, counts, 'new profiles', len(prof['new_measurement_checkpoints']), flush=True)


if __name__ == '__main__':
    main()
