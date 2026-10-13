"""Mechanically audit measured task paths, then freeze all inputs before screening."""
import copy
import math
import sys
from .common import *
from module12.common import baseline
from module10.common import factor


def serialize(c, roles, cal, name):
    return {'regime': name, 'safe_factor': factor(),
            'models': {m: {'supported_tasks': sorted(p.supported_tasks), 'idle_memory_mib': p.memory_mib, 'service_time_ms': dict(p.service_time_ms), 'role': roles[m], 'sources': dict(p.sources)} for m, p in c.items()},
            'capacity_factors': cal.capacity_factors, 'memory_table_mib': cal.memory_table_mib,
            'evidence_bounds': cal.evidence_bounds, 'model_evidence': cal.model_evidence, 'source_hashes': cal.source_hashes}


def main():
    assert not (OUT / 'screen_protocol_freeze.json').exists(), 'Freeze already exists; do not rewrite it after observing results'
    OUT.mkdir(parents=True, exist_ok=True)
    # Historical manifests include raw measurements, frozen source, and artifacts.
    refs = read(PREVIOUS / 'module14b_protected_before.json')['files'] + read(PREVIOUS / 'module14b_artifact_manifest.json')['files']
    refs += [ref(PREVIOUS / 'module14b_artifact_manifest.json')]
    refs = sorted({r['path']: r for r in refs}.values(), key=lambda r: r['path'])
    verify(refs)
    save('protected_before.json', {'at_utc': now(), 'files': refs, 'verified_before': True})
    print('PROTECTED', len(refs), flush=True)
    c, hw, roles, cal = baseline()
    old = read(PREVIOUS / 'baseline_references.json')['tasks']
    policy = read(ROOT / 'results/module8b_quality_pool_finalization/quality_policy.json')
    policies = {p['task']: p for p in policy['tasks']}
    admission = read(ROOT / 'results/module8b_quality_pool_finalization/quality_admission_matrix.json')['rows']
    metadata = {m['id']: m for m in read(ROOT / 'results/candidate_pool_v2_1/candidate_pool_v2_1.json')['models']}
    sources = [ROOT / 'results/module13_real_specialists/real_operating_points.csv',
               PREVIOUS / 'image_quality_latency_memory.csv', PREVIOUS / 'ocr_quality_latency_memory.csv',
               PREVIOUS / 'ocr_gapfill_measurements.csv']
    paths = {}
    for source in sources:
        for row in table(source):
            p = {k: row.get(k) for k in ['task', 'model_id', 'revision', 'origin', 'quality_metric', 'implementation_commit', 'checkpoint', 'quality_raw']}
            p.update({k: float(row[k]) for k in ['quality', 's0_ms', 'idle_memory_mib', 'safe_memory_mib']})
            p['measurement_source'] = ref(source)
            p['legacy_planning_id'] = row.get('legacy_planning_id') or None
            p['local_and_semantic_validity'] = row.get('admission_pass', row.get('local_deployment_success', row.get('local_run_success'))) == 'True'
            paths[(p['task'], p['model_id'], p['revision'])] = p
    # Audit ALL existing measured Specialist paths, including ConvNeXt-tiny.
    # Larger/smaller model names and previous Pareto labels do not determine eligibility.
    for m, profile in c.items():
        if roles[m] != 'specialist':
            continue
        meta = metadata[m]
        for task in profile.supported_tasks:
            key = (task, meta['source'], meta['source_revision'])
            if key in paths:
                paths[key]['legacy_planning_id'] = m
                continue
            q = next(r for r in admission if r['task'] == task and r['model'] == m)
            paths[key] = {'task': task, 'model_id': meta['source'], 'revision': meta['source_revision'],
                          'origin': 'UNCHANGED_REAL_LEGACY_PATH', 'quality_metric': q['quality_metric'],
                          'quality': q['measured_quality'], 's0_ms': profile.service_time_ms[task],
                          'idle_memory_mib': profile.memory_mib, 'safe_memory_mib': factor() * profile.memory_mib,
                          'legacy_planning_id': m, 'local_and_semantic_validity': q['planning_admissible'],
                          'measurement_source': ref(ROOT / 'results/candidate_pool_v2_1/service_demand_v2_1.json'),
                          'quality_source': ref(ROOT / 'results/module8b_quality_pool_finalization/quality_admission_matrix.json')}
    audit = []
    for p in paths.values():
        task = p['task']; b = old[task]; rule = policies[task]
        assert math.isclose(p['safe_memory_mib'], factor() * p['idle_memory_mib'], rel_tol=1e-12)
        numeric = True if rule['policy_status'] == 'SEMANTIC_VALIDITY_ONLY' else (p['quality'] <= rule['historical_threshold'] if rule['direction'] == '<=' else p['quality'] >= rule['historical_threshold'])
        p['original_admission_pass'] = p['local_and_semantic_validity'] and numeric
        p['original_SLO'] = {'status': rule['policy_status'], 'metric': rule['metric'], 'threshold': rule.get('historical_threshold'), 'direction': rule.get('direction')}
        p['latency_increased'] = p['s0_ms'] > b['s0_ms']
        p['memory_increased'] = p['safe_memory_mib'] > b['safe_memory_mib']
        p['heavy_eligible'] = p['original_admission_pass'] and p['latency_increased'] and p['memory_increased']
        p['quality_delta'] = p['quality'] - b['quality']
        p['latency_ratio'] = p['s0_ms'] / b['s0_ms']
        p['safe_memory_ratio'] = p['safe_memory_mib'] / b['safe_memory_mib']
        p['failed_criteria'] = [k for k in ['original_admission_pass', 'latency_increased', 'memory_increased'] if not p[k]]
        p['planning_id'] = p['legacy_planning_id'] or p['model_id'] + '@' + p['revision']
        audit.append(p)
    audit.sort(key=lambda p: (p['task'], p['model_id'], p['revision'] or ''))
    csvsave('heavy_eligibility_audit.csv', audit)
    eligible = [p for p in audit if p['heavy_eligible']]
    tasks = {task: [p for p in eligible if p['task'] == task] for task in sorted(old)}
    assert len(tasks['asr']) == 3 and {p['model_id'] for p in tasks['asr']} == {'openai/whisper-large-v3-turbo', 'openai/whisper-medium', 'openai/whisper-large-v3'}
    assert len(tasks['ocr']) == 4
    assert all(tasks.values())
    save('heavy_candidate_pool.json', {'at_utc': now(), 'regime': H, 'eligibility': 'original frozen quality/capability admission AND strictly greater measured s0 AND strictly greater measured safe memory',
         'higher_quality_SLO_ignored': True, 'prior_Pareto_dominated_not_exclusion': True,
         'selection_used_deployment_or_V3_outcome': False, 'all_eligible_retained': True,
         'old_reference_anchors': old, 'original_quality_policy': ref(ROOT / 'results/module8b_quality_pool_finalization/quality_policy.json'),
         'task_model_paths': len(eligible), 'unique_specialist_checkpoints': len({p['planning_id'] for p in eligible}),
         'tasks': tasks, 'excluded_historical_paths': [p for p in audit if not p['heavy_eligible']],
         'convnext_tiny_note': 'Existing measured ConvNeXt-tiny also strictly passes the mechanical rule (5.847736 ms; idle610 vs ResNet50 idle604). It is retained; a name/scale exclusion would contradict the unified rule.'})
    e = serialize(c, roles, cal, E)
    e['pool_policy'] = 'Unchanged entire efficiency catalog and empirical count space; the five canonical lightweight references are not redefined.'
    save('regime_e.json', e)
    h = copy.deepcopy(e); h['regime'] = H
    for key in ['models', 'capacity_factors', 'memory_table_mib', 'evidence_bounds', 'model_evidence']:
        h[key] = {m: v for m, v in h[key].items() if roles[m] == 'generalist'}
    groups = {}
    for p in eligible:
        groups.setdefault(p['planning_id'], []).append(p)
    for m, ps in sorted(groups.items()):
        idle = max(p['idle_memory_mib'] for p in ps)
        h['models'][m] = {'role': 'specialist', 'supported_tasks': sorted(p['task'] for p in ps),
            'idle_memory_mib': idle, 'service_time_ms': {p['task']: p['s0_ms'] for p in ps},
            'sources': {'measurement_sources': [p['measurement_source'] for p in ps],
                        'checkpoint': {'model_id': ps[0]['model_id'], 'revision': ps[0]['revision']},
                        'shared_checkpoint_memory': 'max measured per-path idle, one residency; independent measured path-specific s0; no inferred capability'}}
        if m in c:
            assert idle == c[m].memory_mib
            for key in ['capacity_factors', 'memory_table_mib', 'evidence_bounds', 'model_evidence']:
                h[key][m] = copy.deepcopy(e[key][m])
        else:
            h['capacity_factors'][m] = {0: 0., 1: 1.}
            h['memory_table_mib'][m] = {0: 0., 1: idle}
            h['evidence_bounds'][m] = 1
            h['model_evidence'][m] = {'measured_replica_counts': [1], 'unknown_gt1': True, 'source': 'real local Module13B/14A/14B standalone measurements'}
    h['pool_policy'] = 'All eligible measured Heavy paths, no canonical lightweight baseline. New checkpoints limited to counts0/1; existing heavier checkpoints and Generalists retain unchanged empirical tables and bounds.'
    h['shared_checkpoint_memory_policy'] = 'Whisper-medium ASR/LID max idle3734 MiB => safe4147.928668921726 MiB, not sum, and not LID-only safe3890.2105513025936; explicit conservative joint residency.'
    catalog(e); catalog(h)
    assert not (set(b['planning_id'] for b in old.values()) & set(h['models']))
    for m in c:
        if roles[m] == 'generalist':
            for key in ['models', 'capacity_factors', 'memory_table_mib', 'evidence_bounds', 'model_evidence']:
                assert e[key][m] == h[key][m]
    save('regime_h.json', h)
    freeze = {'at_utc': now(), 'screen': 'SINGLE_REPLICA_STRUCTURAL_SCREEN', 'CPU_ONLY': True, 'GPU_runs_authorized': False,
       'primary_scope': 'New Heavy checkpoints xS<=1; all existing empirical data and Generalists unchanged. E uses original empirical count space.',
       'supplementary_scope': 'MATCHED_ALL_SPECIALIST_COUNTS_0_1: restrict BOTH regimes all Specialist counts0/1; Generalist empirical counts unchanged. Diagnostic control for search-space differences, not a replacement primary result.',
       'workloads': read(PLAN)['workloads'], 'budgets': read(PLAN)['budgets'], 'context_count': 27,
       'plan_reference': ref(PLAN), 'rho_max': .8, 'objective': 'Unchanged frozen J-routing; quality is original binary admission, no quality bonus or cost retuning.',
       'numerical_tolerance': 1e-8, 'routing_event_tolerance': 1e-9,
       'gate_rule': {'HEAVY_STRUCTURAL_TRANSITION_SUPPORTED': '>=2 comparable development contexts with T1/T2/T3/T5', 'HEAVY_STRUCTURAL_TRANSITION_PARTIAL': 'exactly1 comparable context with T1/T2/T3/T5', 'HEAVY_STRUCTURAL_TRANSITION_NOT_OBSERVED': 'zero comparable contexts with T1/T2/T3/T5; identity-only changes do not qualify'},
       'T_definitions': {'T1': 'Specialist replica count decreases', 'T2': 'rate-weighted Generalist share increases', 'T3': 'new positive G/S split task', 'T4': 'Generalist newly consolidates >=2 tasks', 'T5': 'Generalist replica count increases', 'T6': 'new second-Generalist state: old NG<2 <= new NG, report identities separately', 'T7': 'Heavy Specialist identity varies across frozen workloads/budgets'},
       'attribution_rule': 'No new counterfactual optimizations. Use observed real costs, active memory/capacity witnesses, matched-count diagnostic, and existing Module12 axis evidence. Mechanistic explanations are not isolated causal effects; report both/count-cap ambiguity explicitly.',
       'profiling_rule': 'Required if resident in any H primary or matched optimum; otherwise only with explicit active capacity/memory or G/S/NG decision witness. No outcome-driven pool changes; unknown higher counts stay unknown.',
       'immutable_inputs': [ref(OUT / name) for name in ['heavy_candidate_pool.json', 'regime_e.json', 'regime_h.json']],
       'implementation': [ref(p) for folder in ['module14c', 'module12', 'module11b', 'deployment_core', 'deployment_search', 'module9bc'] for p in sorted((ROOT / folder).rglob('*.py'))],
       'environment': {'python': sys.version, 'executable': sys.executable, 'CUDA_VISIBLE_DEVICES': '', 'numpy': __import__('numpy').__version__, 'scipy': __import__('scipy').__version__},
       'no_final_multi_replica_optimality_claim': True, 'no_deployment_or_V3_outcome_for_membership': True}
    save('screen_protocol_freeze.json', freeze)
    print('FROZEN', len(eligible), len(groups), {k: len(v) for k, v in tasks.items()}, flush=True)


if __name__ == '__main__':
    main()
