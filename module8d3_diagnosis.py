"""Module 8D-3: offline diagnosis and separately frozen fixed-state crossover."""
import argparse
import copy
import csv
import itertools
import json
import math
import os
import statistics
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.stats import spearmanr, kendalltau
from deployment_core import Workload
from deployment_core.replica_calibration import ReplicaPlanningPolicy
from deployment_search.feedback import bottleneck_feedback
from deployment_search.neighborhood import generate_replica_neighborhood, missing_coverage
from deployment_search.routing import solve_canonical_routing
from deployment_search.scoring import rank_candidates
from deployment_search.real_objective import NoiseAwareRuntimeObjective
from deployment_search.result import CandidateEvaluation
from deployment_search.runtime_cache import save_json, digest, file_hash
from module8d_real_baselines.common import ROOT, C, F, OUT as D2, B, P, read, ref, inputs, verify_inputs, full_vector, validate_raw

OUT = ROOT / 'results/module8d3_failure_diagnosis'
CAT, HW, ROLES, CAL = inputs()
POLICY = ReplicaPlanningPolicy(CAT, 'empirical', 'empirical', CAL)
BOUNDS = dict(CAL.evidence_bounds)
NAMES = [j['workload'] for j in read(D2 / 'campaign_manifest.json')['phase_A'] if j['baseline'] == 'B0']
OBJ = NoiseAwareRuntimeObjective(1., .05)


def csvwrite(filename, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT / filename).open('w', newline='') as f:
        w = csv.DictWriter(f, keys); w.writeheader()
        for row in rows:
            w.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list, tuple)) else v for k, v in row.items()})


def counts(x): return {m: x.get(m, 0) for m in CAT}
def key(x): return tuple(x.get(m, 0) for m in sorted(CAT))
def distance(a, b): return sum(abs(a.get(m, 0) - b.get(m, 0)) for m in CAT)
def rates(n): return read(C / n / 'protocol.json')['workload']['arrival_rates_rps']
def routing(n, x): return solve_canonical_routing(x, Workload(rates(n)), CAT, .8, BOUNDS, 'empirical', CAL)
def nested(route): return {t: {m: float(route[t, m]) or 0. for m in sorted(CAT)} for t in sorted({t for t, _ in route})}
def flat(route): return {(t, m): v for t, ms in route.items() for m, v in ms.items()}
def flatten_trace_route(route): return {tuple(pair.split('::')): v for pair, v in route.items()}


def identity(n, x, route):
    base = copy.deepcopy(read(D2 / 'b2' / n / 'result.json')['identity'])
    base['deployment'] = counts(x); base['routing'] = nested(route)
    return base


def better(candidate, current, cx, px):
    def rt(d): return SimpleNamespace(stability_state_v2=d['stability_state_v2'], weighted_mean_e2e=d['weighted_mean_e2e'], weighted_p95_e2e=d['weighted_p95_e2e'])
    return OBJ.compare(rt(candidate), rt(current), POLICY.deployment_memory(counts(cx)), POLICY.deployment_memory(counts(px)))


class Offline:
    def __init__(self):
        self.planning = {}; self.index = {}; self.traces = []
        for n in NAMES:
            self.traces.append(('Proposed', None, n, C / n / 'search_complete.json', read(C / n / 'search_complete.json')))
            for s in (1103, 2207, 47):
                p = D2 / f'b4_seed_{s}' / n / 'search_complete.json'
                self.traces.append(('Random', s, n, p, read(p)))
        for root in (C, D2):
            for p in sorted(root.glob('**/search_cache/runtime_cache_manifest.json')):
                for k, e in read(p)['configurations'].items():
                    if e['status'] == 'complete':
                        assert digest(e['identity']) == k
                        if k not in self.index:
                            assert file_hash(e['aggregate']) == e['aggregate_sha256']
                            self.index[k] = {'entry': e, 'runtime': read(e['aggregate']), 'manifest': ref(p)}

    def canonical(self, n, x):
        k = (n, key(x))
        if k not in self.planning:
            self.planning[k] = routing(n, x) if POLICY.deployment_memory(x) <= HW.gpu_memory_mib and not missing_coverage(x, CAT, Workload(rates(n))) else None
        return self.planning[k]

    def evidence(self, n, x, route):
        ident = identity(n, x, route); d = self.index.get(digest(ident))
        if not d: return None
        assert d['entry']['identity'] == ident
        return {'identity': ident, 'runtime': d['runtime'], 'source': d['manifest'], 'raw_sources': [ref(p) for p in d['runtime']['raw_result_paths']]}

    def neighbors(self, n, x):
        c = self.canonical(n, x)
        if c is None: return []
        return generate_replica_neighborhood(x, CAT, Workload(rates(n)), bottleneck_feedback(c, x), BOUNDS, .2)

    def shortest(self, n, start, target):
        start, target = counts(start), counts(target)
        base = [(m, 1 if target[m] > start[m] else -1) for m in sorted(CAT) for _ in range(abs(target[m] - start[m]))]
        if not base: return {'structural_l1': 0, 'pressure_gated_min_length': 0, 'path': [], 'reachable': True}
        def try_ops(ops):
            for order in sorted(set(itertools.permutations(ops))):
                x = dict(start); path = []
                for m, sign in order:
                    want = dict(x); want[m] += sign
                    if not any(mu.model == m and mu.deployment == want for mu in self.neighbors(n, x)) or self.canonical(n, want) is None: break
                    path.append({'model': m, 'delta': sign, 'deployment': full_vector(want), 'memory_mib': POLICY.deployment_memory(want), 'canonical_z': self.canonical(n, want).max_pressure})
                    x = want
                else:
                    if x == target: return path
            return None
        path = try_ops(base)
        if path is not None: return {'structural_l1': len(base), 'pressure_gated_min_length': len(path), 'path': path, 'reachable': True, 'minimality_proof': 'L1 lower bound attained'}
        for m in sorted(CAT):
            path = try_ops(base + [(m, 1), (m, -1)])
            if path is not None:
                return {'structural_l1': len(base), 'pressure_gated_min_length': len(path), 'path': path, 'reachable': True, 'minimality_proof': 'All L1-length paths fail generation/feasibility; lattice parity excludes L1+1; L1+2 witness found', 'detour_family': m}
        return {'structural_l1': len(base), 'pressure_gated_min_length': None, 'reachable': None, 'status': 'No path within L1+2; not a proof of global unreachability'}


def freeze_inputs():
    OUT.mkdir(parents=True, exist_ok=True); verify_inputs()
    assert read(D2 / 'campaign_status.json')['status'] == 'complete'
    assert file_hash(D2 / 'artifact_manifest.json') == (D2 / 'artifact_manifest.json.sha256').read_text().strip()
    for r in read(D2 / 'artifact_manifest.json')['artifacts'].values(): assert file_hash(r['path']) == r['sha256']
    document = {'module': '8D-3', 'protected_8D2_artifacts': ref(D2 / 'artifact_manifest.json'), 'protected_8D1': ref(F / 'baseline_freeze_hashes.json'), 'new_predictor': False, 'algorithm_patch': False,
        'correlation_units': 'Keep all evaluated transition events; also report exact-transition deduplicated sensitivity because shared raw events are not independent.',
        'correlation_minimum_n': 3, 'n_below_5_warning': 'Very small exploratory sample; no inferential significance claim',
        'pairwise_ordering': 'Within same workload/current canonical routing identity only; strict delta_z and strict observed delta_mean pairs, ties separately reported',
        'hit_rates': 'Analytical Top-1/Top-2 capture of objective-nondominated measured candidates within incumbent choice sets. Global rates only for complete measured feasible choice sets; incomplete coverage explicitly reported.',
        'reachability': 'Structural x±1 L1 versus actual pressure-gated generated, coverage/capacity/memory-feasible graph; shortest goal-directed permutations, no full replica enumeration and no RuntimeEvaluator.',
        'crossover_order': ['reasoning_vqa_heavy', 'speech_heavy', 'no_vqa_phase'], 'crossover_policy': 'Only missing fixed B2 deployment/canonical-routing states in first two priority workloads; no-VQA reuse if exact. A/B/C chain yields two conditional contrasts; no claim of interaction identification without D.',
        'frozen_objective': 'NoiseAwareRuntimeObjective(1ms,.05)', 'no_ablation_dynamic_second_hardware': True}
    p = OUT / 'diagnostic_analysis_protocol.json'
    if p.exists(): assert read(p) == document
    else: save_json(p, document); (OUT / (p.name + '.sha256')).write_text(file_hash(p) + '\n')


def offline():
    freeze_inputs(); audit = Offline(); distances = []; paths = {}; trace_rows = []; budget = []; crossover = []; random = []
    for n in NAMES:
        p = read(C / n / 'search_complete.json'); b = read(D2 / 'b2' / n / 'result.json'); target = counts(b['state']['deployment'])
        paths[n] = {}
        for source, x in [('seed', p['seed']['deployment']), ('final', p['deployment'])]:
            path = audit.shortest(n, x, target); paths[n][source] = path
            operations = [{'model': m, 'delta': target[m] - x.get(m, 0), 'operation': ('ADD' if not x.get(m, 0) else 'REPLICA_UP') if target[m] > x.get(m, 0) else ('REMOVE' if not target[m] else 'REPLICA_DOWN')} for m in sorted(CAT) if target[m] != x.get(m, 0)]
            distances.append({'workload': n, 'source': source, 'L1_distance': distance(x, target), 'required_operations': operations, **{k: v for k, v in path.items() if k != 'path'}})
        needed = {(m, 1 if target[m] > p['seed']['deployment'].get(m, 0) else -1) for m in CAT if target[m] != p['seed']['deployment'].get(m, 0)}
        for method, seed, name, path, t in audit.traces:
            if name != n: continue
            for it in t['iterations']:
                present = {}
                for c in it['candidates']:
                    m = c['mutation']['model']; diff = c['mutation']['deployment'][m] - it['deployment'][m]; present[m, diff] = c
                for m, sign in sorted(needed):
                    c = present.get((m, sign)); candidate_x = dict(it['deployment']); candidate_x[m] += sign
                    can = audit.canonical(n, candidate_x) if 0 <= candidate_x[m] <= BOUNDS[m] else None
                    ev = audit.evidence(n, candidate_x, can.routing) if can is not None else None
                    recorded = c.get('runtime') if c else None
                    status = c.get('status') if c else 'NOT_GENERATED'
                    why = ('not generated' if c is None else 'infeasible' if status == 'REJECT_INFEASIBLE' else 'cycle avoidance' if status == 'SKIP_TABU' else 'rank too low' if status == 'PRUNED_TOP_B' else 'budget exhausted' if status == 'NOT_EVALUATED_BUDGET' else '')
                    row = {'workload': n, 'method': method, 'random_seed': seed, 'iteration': it['iteration'], 'required_mutation_model': m, 'delta': sign, 'current_vector': full_vector(it['deployment']), 'generated': c is not None, 'feasibility_PASS': can is not None, 'trace_status': status, 'reason_not_evaluated': why, 'analytical_rank': c.get('rank') if c and method == 'Proposed' else None, 'random_selection_order': c.get('rank') if c and method == 'Random' else None, 'delta_z': c.get('delta_z') if c else None, 'Top_B': bool(c and c.get('selected_top_b')), 'runtime_evaluated_in_this_trace': recorded is not None, 'accept_rollback': status if recorded else None, 'trace_source': ref(path), 'historical_exact_runtime_available': ev is not None}
                    rt = recorded or (ev['runtime'] if ev else None)
                    if rt:
                        rb, reason = better(rt, it['current_runtime'], candidate_x, it['deployment'])
                        row.update(real_mean=rt['weighted_mean_e2e'], real_P95=rt['weighted_p95_e2e'], stability=rt['stability_state_v2'], runtime_better=rb, runtime_better_reason=reason, raw_source_paths=rt['raw_result_paths'])
                    trace_rows.append(row)
            if method == 'Random':
                pi = read(C / n / 'search_cache/runtime_cache_manifest.json')['configurations']
                ri = read(path.parent / 'search_cache/runtime_cache_manifest.json')['configurations']
                ps = {digest(e['identity']) for e in pi.values() if e['status'] == 'complete'}
                rs = {digest(e['identity']) for e in ri.values() if e['status'] == 'complete'}
                candidates = lambda tr: {key(c['mutation']['deployment']) for c in tr['iterations'][0]['candidates'] if c.get('runtime')}
                firstp, firstr = candidates(p), candidates(t)
                random.append({'workload': n, 'random_seed': seed, 'Proposed_evaluated_states': sorted(ps), 'Random_evaluated_states': sorted(rs), 'overlap': sorted(ps & rs), 'unique_to_random': sorted(rs - ps), 'unique_to_proposed': sorted(ps - rs), 'first_TopB_overlap': len(firstp & firstr), 'first_feasible_choice_count': sum(c.get('canonical') is not None for c in p['iterations'][0]['candidates']), 'same_final_deployment': key(p['deployment']) == key(t['deployment']), 'same_final_runtime_cache_key': p['runtime']['cache_key'] == t['runtime']['cache_key'], 'depth': t['search_iterations'], 'termination': t['status'], 'trace': ref(path)})
        final = counts(p['deployment']); can = audit.canonical(n, final); mus = audit.neighbors(n, final)
        visited = {tuple(tuple(a) for a in v) for v in p['visited_deployments']}
        rejected = {(tuple(tuple(a) for a in parent), tuple(tuple(a) for a in child)) for parent, child in p['rejected_mutations']}
        replica_key = lambda x: tuple(sorted((m, v) for m, v in x.items() if v))
        feasible = []; all_final = []
        for mu in mus:
            cc = audit.canonical(n, mu.deployment); tabu = replica_key(mu.deployment) in visited or (replica_key(final), replica_key(mu.deployment)) in rejected
            all_final.append({'model': mu.model, 'operation': mu.operation, 'feasible': cc is not None, 'tabu': tabu})
            if cc is not None and not tabu:
                c = CandidateEvaluation(mu, memory_mib=POLICY.deployment_memory(mu.deployment)); c.canonical = cc; c.delta_z = can.max_pressure - cc.max_pressure; feasible.append(c)
        ranked = rank_candidates(feasible)
        budget.append({'workload': n, 'max_states': 6, 'actual_evaluated_states': p['runtime_evaluation_count'], 'max_iterations': 3, 'actual_iterations': p['search_iterations'], 'patience': 1, 'termination_reason': p['status'], 'remaining_state_slots': 6 - p['runtime_evaluation_count'], 'remaining_iteration_slots': 3 - p['search_iterations'], 'remaining_feasible_unvisited_candidates': [{'rank': i, 'model': c.mutation.model, 'operation': c.mutation.operation, 'delta_z': c.delta_z, 'deployment': full_vector(c.mutation.deployment)} for i, c in enumerate(ranked, 1)], 'counterfactual_one_more_iteration_top2': [{'model': c.mutation.model, 'deployment': full_vector(c.mutation.deployment)} for c in ranked[:2]], 'runtime_continuation_executed': False, 'all_generated_at_final': all_final})
        bcan = audit.canonical(n, target)
        greedy = {}
        for task, rate in rates(n).items():
            available = [m for m in CAT if final[m] and task in CAT[m].supported_tasks]
            specialists = [m for m in available if ROLES[m] == 'specialist']
            selected = min(specialists or available, key=lambda m: (CAT[m].service_time_ms[task], m)) if rate else None
            for m in CAT: greedy[task, m] = rate if m == selected else 0.
        variants = {'A': (target, flat(b['identity']['routing'])), 'B': (target, bcan.routing), 'C': (final, flatten_trace_route(p['canonical']['routing'])), 'D': (final, greedy)}
        for label, (x, route) in variants.items():
            ev = audit.evidence(n, x, route)
            crossover.append({'workload': n, 'variant': label, 'meaning': {'A': 'B2 deployment + frozen B2 greedy routing', 'B': 'B2 deployment + unchanged canonical routing', 'C': 'Proposed final deployment + frozen canonical routing', 'D': 'Proposed final deployment + best available specialist else generalist greedy-compatible routing'}[label], 'identity': identity(n, x, route), 'identity_sha256': digest(identity(n, x, route)), 'deployment': full_vector(x), 'status': 'EXACT_REUSE_AVAILABLE' if ev else 'CROSSOVER_MEASUREMENT_MISSING', 'evidence': ev})
    csvwrite('deployment_distance.csv', distances); save_json(OUT / 'reachability_paths.json', paths); csvwrite('reachability_trace.csv', trace_rows)
    save_json(OUT / 'search_budget_audit.json', {'workloads': budget, 'no_actual_search_continuation': True})
    save_json(OUT / 'random_path_analysis.json', {'workloads': random, 'exact_state_identity_includes_routing': True, 'evaluated_deployment_vectors': [{'method': m, 'random_seed': s, 'workload': n, 'seed': full_vector(t['seed']['deployment']), 'candidates': [full_vector(c['mutation']['deployment']) for it in t['iterations'] for c in it['candidates'] if c.get('runtime')]} for m, s, n, p, t in audit.traces]})
    save_json(OUT / 'routing_crossover_audit.json', {'variants': crossover, 'identity_checks': 'Exact lambda, replica vector, fractions, protocol, revision/environment/backend/input sources, admission; no name-only reuse', 'greedy_D_contract': 'On deployed models, same B2 rule: fastest admitted specialist else fastest admitted generalist, active tasks only', 'new_GPU_measurements': 0})
    alignment(audit)
    strength = []
    for n in NAMES:
        for task, rate in rates(n).items():
            sp = [m for m in CAT if ROLES[m] == 'specialist' and task in CAT[m].supported_tasks]
            gs = [m for m in CAT if ROLES[m] == 'generalist' and task in CAT[m].supported_tasks]
            ss = min(sp, key=lambda m: (CAT[m].service_time_ms[task], m)) if sp else None
            gg = min(gs, key=lambda m: (CAT[m].service_time_ms[task], m)) if gs else None
            a = CAT[gg].service_time_ms[task] if gg else None; b = CAT[ss].service_time_ms[task] if ss else None
            strength.append({'workload': n, 'task': task, 'lambda_rps': rate, 'best_specialist': ss or 'NO_SPECIALIST', 'best_generalist': gg or 'NO_GENERALIST', 'specialist_s0_ms': b, 'generalist_s0_ms': a, 'difference_ms': a-b if a is not None and b is not None else None, 'speedup_specialist_vs_generalist': a/b if a is not None and b is not None else None, 'potential_offload_saving_service_seconds_per_second': rate*(a-b)/1000 if a is not None and b is not None else None})
    csvwrite('greedy_strength_analysis.csv', strength)
    resource = []
    # Random best is a descriptive diagnostic only, never a selected baseline score.
    baseline_rows = read(D2 / 'all_baseline_results.json')['results']
    for n in NAMES:
        rr = [r for r in baseline_rows if r['workload'] == n and r['baseline'] == 'B4']; best = min(rr, key=lambda r: (r['metrics']['mean_e2e_ms'], r['random_seed']))
        selected = [r for r in baseline_rows if r['workload'] == n and r['baseline'] in ('Proposed', 'B2', 'B3')] + [best]
        for r in selected:
            mt = r['metrics']; resource.append({'workload': n, 'method': r['baseline'] if r['baseline'] != 'B4' else 'Random_descriptive_lowest_mean', 'random_seed': r['random_seed'], 'planning_memory_mib': r['planning_memory_mib'], 'actual_idle_VRAM_mib': mt['actual_idle_VRAM_mib_mean'], 'actual_sampled_peak_VRAM_mib': mt['sampled_peak_device_VRAM_mib'], 'GPU_capacity_mib': 49140, 'memory_headroom_mib': 49140 - mt['sampled_peak_device_VRAM_mib'], 'peak_fraction': mt['sampled_peak_device_VRAM_mib']/49140, 'deployment': r['deployment'], 'selection_is_diagnostic_not_best_of_baseline': True})
    csvwrite('resource_constraint_audit.csv', resource)
    pending = [r for r in crossover if r['variant'] == 'B' and r['workload'] in ('reasoning_vqa_heavy', 'speech_heavy') and r['evidence'] is None]
    pending.sort(key=lambda r: ['reasoning_vqa_heavy', 'speech_heavy'].index(r['workload']))
    plan = {'status': 'FROZEN' if pending else 'NOT_RUN', 'gate': 'Existing exact evidence cannot form A/B/C decomposition for two priority workloads; no-VQA A/B/C/D fully available offline.', 'states': pending, 'repetitions': 3, 'seeds': [7101,7202,7303], 'settings': read(C / NAMES[0] / 'protocol.json')['measurement_settings'], 'maximum_new_timed_runs': 3*len(pending), 'search': False, 'candidate_ranking': False, 'D_new_measurements': 'NOT_PLANNED: A/B/C conditional contrast sufficient to measure routing difference on B2 deployment and deployment change under canonical policy; D interaction remains explicitly unidentified.', 'all_six_default_run': False, 'protected_inputs': ref(OUT / 'diagnostic_analysis_protocol.json'), 'implementation': ref(Path(__file__).resolve()), 'preexisting_measurements_used_to_define_workload_or_method': False}
    path = OUT / 'crossover_plan.json'
    if path.exists(): assert read(path) == plan
    else: save_json(path, plan); (OUT / 'crossover_plan.json.sha256').write_text(file_hash(path)+'\n')
    save_json(OUT / 'crossover_results.json', {'status': 'NOT_RUN', 'plan': ref(path), 'existing_variants': [{'workload': r['workload'], 'variant': r['variant'], 'status': r['status']} for r in crossover]})
    print(json.dumps({'event': 'offline_complete', 'distances': distances, 'crossover_missing': [(r['workload'], r['variant']) for r in crossover if not r['evidence']], 'planned_new_runs': plan['maximum_new_timed_runs']}, ensure_ascii=False), flush=True)


def alignment(audit):
    events = []; contexts = {}
    for method, seed, n, path, t in audit.traces:
        for it in t['iterations']:
            current = it['current_runtime']; context = digest(identity(n, it['deployment'], flatten_trace_route(it['canonical']['routing'])))
            group = contexts.setdefault((n, context), {'workload': n, 'context': context, 'candidates': {}, 'current': current, 'current_x': it['deployment']})
            for c in it['candidates']:
                if c.get('delta_z') is None: continue
                x = c['mutation']['deployment']; can_route = flatten_trace_route(c['canonical']['routing']); ev = audit.evidence(n, x, can_route)
                if ev:
                    rt = ev['runtime']; ident = digest(ev['identity']); rb, reason = better(rt, current, x, it['deployment'])
                    record = {'workload': n, 'method': method, 'random_seed': seed, 'iteration': it['iteration'], 'model': c['mutation']['model'], 'family_role': ROLES[c['mutation']['model']], 'operation': c['mutation']['operation'], 'context_sha256': context, 'candidate_identity_sha256': ident, 'delta_z': c['delta_z'], 'delta_mean_ms': current['weighted_mean_e2e']-rt['weighted_mean_e2e'], 'delta_p95_ms': current['weighted_p95_e2e']-rt['weighted_p95_e2e'], 'runtime_better': rb, 'runtime_reason': reason, 'stability': rt['stability_state_v2'], 'runtime_evaluated_in_this_trace': c.get('runtime') is not None, 'analytical_rank': c.get('rank') if method == 'Proposed' else None, 'selection_order': c.get('rank'), 'selected_top_b': c['selected_top_b'], 'trace_status': c['status'], 'current_deployment': full_vector(it['deployment']), 'candidate_deployment': full_vector(x), 'raw_source_paths': rt['raw_result_paths'], 'trace_source': ref(path)}
                    group['candidates'][ident] = record
                    if c.get('runtime') is not None: events.append(record)
            # Full candidate set generated for this incumbent, with fresh
            # unchanged LP ranking only for offline diagnostic scoring.
            group.setdefault('feasible_identities', set())
            for c in it['candidates']:
                if c.get('canonical') is not None:
                    ident = identity(n, c['mutation']['deployment'], flatten_trace_route(c['canonical']['routing']))
                    group['feasible_identities'].add(digest(ident))
    csvwrite('lp_runtime_alignment.csv', events)
    unique = list({(r['workload'],r['context_sha256'],r['candidate_identity_sha256']):r for r in events}.values())
    def summarize(rs):
        n = len(rs); x = [r['delta_z'] for r in rs]; y = [r['delta_mean_ms'] for r in rs]; z = [r['delta_p95_ms'] for r in rs]
        valid = n >= 3 and len(set(x)) > 1 and len(set(y)) > 1
        return {'n': n, 'status': 'EXPLORATORY' if valid else 'INSUFFICIENT', 'Spearman_delta_z_delta_mean': float(spearmanr(x,y).statistic) if valid else None, 'Kendall_tau_delta_z_delta_mean': float(kendalltau(x,y).statistic) if valid else None, 'Spearman_delta_z_delta_p95': float(spearmanr(x,z).statistic) if valid and len(set(z)) > 1 else None, 'no_pvalue_significance_claim': True, 'small_sample_warning': n < 5}
    def scopes(rs):
        return {'pooled': summarize(rs), 'per_workload': {n:summarize([r for r in rs if r['workload']==n]) for n in NAMES}, 'by_family_role': {role:summarize([r for r in rs if r['family_role']==role]) for role in ('generalist','specialist')}}
    paired = Counter(); hit = []; missed = []
    for (n, context), g in contexts.items():
        cs = list(g['candidates'].values()); ranked = sorted(cs, key=lambda r: (-r['delta_z'], r['model']))
        for a,b in itertools.combinations(cs,2):
            dx,dy=a['delta_z']-b['delta_z'],a['delta_mean_ms']-b['delta_mean_ms']
            if dx == 0 or dy == 0: paired['ties']+=1
            else: paired['eligible']+=1;paired['correct']+=int(dx*dy>0)
        # Objective-nondominated observed candidates (ties admitted), no
        # order-dependent invented best candidate or post-hoc latency gate.
        best=[]
        for a in cs:
            ra=audit.index[a['candidate_identity_sha256']]['runtime']
            if not any(better(audit.index[b['candidate_identity_sha256']]['runtime'],ra,counts(b['candidate_deployment']),counts(a['candidate_deployment']))[0] for b in cs if b is not a): best.append(a['candidate_identity_sha256'])
        all_ids=g['feasible_identities']; measured_ids=set(g['candidates']); complete=all_ids<=measured_ids
        original_ranked=[]
        for cid in all_ids:
            if cid in measured_ids: original_ranked.append(g['candidates'][cid])
        # Rank can be reconstructed across complete generated feasible sets
        # from cached/known canonical identities, independent of RNG order.
        scores=[]
        for method, seed, name, path, trace in audit.traces:
            if name!=n:continue
            for it in trace['iterations']:
                if digest(identity(n,it['deployment'],flatten_trace_route(it['canonical']['routing'])))!=context:continue
                for c in it['candidates']:
                    if c.get('delta_z') is not None:
                        cid=digest(identity(n,c['mutation']['deployment'],flatten_trace_route(c['canonical']['routing'])))
                        scores.append((cid,c['delta_z'],c['mutation']['model']))
        actual_order=sorted({a[0]:a for a in scores}.values(), key=lambda a:(-a[1],a[2]))
        hit.append({'workload':n,'context':context,'measured_candidates':len(cs),'feasible_generated_candidates':len(all_ids),'complete_choice_set':complete,'objective_best_measured':best,'Top1_hit':bool(actual_order and actual_order[0][0] in best),'Top2_hit':bool(set(a[0] for a in actual_order[:2])&set(best)),'at_least_one_runtime_better':any(c['runtime_better'] for c in cs),'coverage':len(measured_ids&all_ids)/len(all_ids) if all_ids else None})
        for c in cs:
            pr=[r for r in events if r['method']=='Proposed' and r['context_sha256']==context and r['candidate_identity_sha256']==c['candidate_identity_sha256']]
            lp_rank=next((i for i,a in enumerate(actual_order,1) if a[0]==c['candidate_identity_sha256']),None)
            if c['runtime_better'] and lp_rank and lp_rank>2:missed.append({'workload':n,'context':context,'model':c['model'],'analytical_rank':lp_rank,'delta_z':c['delta_z'],'delta_mean_ms':c['delta_mean_ms'],'runtime_evaluated_by_Proposed':bool(pr),'raw_sources':c['raw_source_paths']})
    def rates_hit(groups):
        return {'n_contexts':len(groups),'status':'EXPLORATORY' if len(groups)>=3 else 'INSUFFICIENT','Top1_hit_rate':sum(g['Top1_hit'] for g in groups)/len(groups) if groups else None,'Top2_hit_rate':sum(g['Top2_hit'] for g in groups)/len(groups) if groups else None}
    summary={'evaluated_event_count':len(events),'exact_unique_transition_count':len(unique),'all_events_retained':True,'all_event_correlations':scopes(events),'deduplicated_sensitivity':scopes(unique),'pairwise_ordering':{'unit':'Unique measured candidate pairs within same incumbent identity','counts':dict(paired),'accuracy':paired['correct']/paired['eligible'] if paired['eligible'] else None,'status':'EXPLORATORY' if paired['eligible']>=3 else 'INSUFFICIENT'},'hit_rates_complete_sets':rates_hit([g for g in hit if g['complete_choice_set']]),'hit_rates_observed_partial_sets':rates_hit(hit),'choice_set_coverage':hit,'measured_better_candidates_below_Top2':missed,'unmeasured_candidates_cannot_be_classified':True,'shared_raw_not_independent':True}
    save_json(OUT/'lp_runtime_alignment_summary.json',summary)


def crossover():
    verify_inputs();plan=read(OUT/'crossover_plan.json')
    assert file_hash(OUT/'crossover_plan.json')==(OUT/'crossover_plan.json.sha256').read_text().strip()
    assert file_hash(plan['implementation']['path'])==plan['implementation']['sha256']
    for k,v in read(F/'baseline_definitions.json')['measurement_protocol']['execution_environment_overrides'].items():os.environ[k]=v
    from candidate_pool_audit.provenance import environment_snapshot
    assert environment_snapshot()==read(C/'environment_manifest.json')
    for path,sha in read(D2/'source_preflight.json')['native_hashes'].items():assert file_hash(path)==sha
    from replica_scaling.telemetry import snapshot
    gpu=snapshot();assert not gpu['processes'] and gpu['used_mib']<=128
    from integer_replica_pilot.evaluator import ReplicaRuntimeEvaluator
    from module8c_runtime_compat.runner import runner as compatible
    from integer_replica_pilot.evaluator import runner
    results=[]
    for state in plan['states']:
        verify_inputs();n=state['workload'];directory=OUT/'crossover_raw'/n
        ev=ReplicaRuntimeEvaluator(directory,CAT,C/n/'runtime_context.json',plan['settings'],plan['seeds'],1,compatible if n=='no_vqa_phase' else runner)
        x=counts(state['deployment']);route=flat(state['identity']['routing']);w=Workload(rates(n))
        assert ev.identity(x,w,route)==state['identity']
        rt=ev.evaluate(x,w,route);runtime=asdict(rt)
        for path,seed in zip(rt.raw_result_paths,plan['seeds']):validate_raw(path,state['identity'],seed,n)
        results.append({'workload':n,'variant':'B','identity':state['identity'],'runtime':runtime,'raw_sources':[ref(p) for p in rt.raw_result_paths],'measurement_source':'newly_measured','search':False})
        save_json(OUT/'crossover_results.json',{'status':'RUNNING','plan':ref(OUT/'crossover_plan.json'),'new_states':results,'new_timed_runs':3*len(results)})
    save_json(OUT/'crossover_results.json',{'status':'COMPLETE' if results else 'NOT_RUN','plan':ref(OUT/'crossover_plan.json'),'new_states':results,'new_timed_runs':3*len(results),'no_search_or_ranking':True})
    print(json.dumps({'event':'crossover_complete','new_runs':3*len(results)}),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=('offline','crossover'));args=parser.parse_args()
    (offline if args.action=='offline' else crossover)()
