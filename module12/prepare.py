"""Freeze scope/labels/selection rules before observing sensitivity outcomes."""
from .common import *


def main():
    assert not (OUT / 'experiment_freeze.json').exists()
    c, hw, roles, cal = baseline()
    dev = read(ROOT / 'results/module8e1_method_revision/development_workload_designation.json')
    workloads = [{'id': name, 'rates': rates, 'regime': 'low_load',
        'source': ref(ROOT / 'results/module8e1_method_revision/development_workload_designation.json')}
        for name, rates in dev['lambda_vectors'].items()]
    cells_path = ROOT / 'results/module8f1b_intensity_grid_freeze/campaign_cells.json'
    cells = read(cells_path)['cells']
    for level, regime in [('I1', 'moderate_load'), ('I2', 'pressured'), ('I3', 'high_pressure_development')]:
        cell = next(r for r in cells if r['intensity_level'] == level)
        workloads.append({'id': 'mix_03_' + level, 'rates': cell['lambda_rps'],
            'regime': regime, 'source': ref(cells_path), 'intensity': level})
    values = sorted({r['memory_budget_mib'] for r in cells})
    budgets = [{'id': f'B{i+1}', 'safe_memory_budget_mib': values[i]} for i in [0, 2, 5]]
    families = []
    for task in ['ocr', 'image', 'text', 'asr', 'lid']:
        gs = sorted((p.service_time_ms[task], m) for m, p in c.items() if roles[m] == 'generalist' and task in p.supported_tasks)
        ss = sorted((p.service_time_ms[task], m) for m, p in c.items() if roles[m] == 'specialist' and task in p.supported_tasks)
        families.append({'task': task, 'models': sorted(m for _, m in ss),
            'best_admitted_G': gs[0][1], 'best_G_s0_ms': gs[0][0],
            'fastest_admitted_S': ss[0][1], 'fastest_S_s0_ms': ss[0][0],
            'baseline_speedup': gs[0][0] / ss[0][0]})
    families.sort(key=lambda r: (-r['baseline_speedup'], r['task']))
    refs = read(BASE / 'frozen_existing_evidence_manifest.json')['files']
    refs += read(BASE / 'module11b_closeout_manifest.json')['closeout_outputs']
    refs += read(BASE / 'input_freeze.json')['protected_references']
    refs += [ref(p) for d in ['module9bc', 'deployment_core/routing', 'deployment_core/search', 'results/module9bc_e2e_routing_ranking']
             for p in (ROOT / d).rglob('*') if p.is_file() and p.suffix != '.pyc']
    refs = list({r['path']: r for r in refs}.values())
    plan = {'at_utc': now(), 'kind': 'COUNTERFACTUAL_CPU_ONLY',
        'regime': 'EFFICIENCY_OPTIMAL_SPECIALIST_REGIME', 'workloads': workloads, 'budgets': budgets,
        'gamma_grid': GAMMAS, 'mu_grid': MUS, 'families_in_speedup_order': families,
        'global_grid': 'full frozen gamma x mu', 'per_task_grid': 'latency-only and memory-only axes; no 2D per-task interaction grid',
        'global_queries': len(workloads)*len(budgets)*len(GAMMAS)*len(MUS),
        'per_task_nonbaseline_queries': len(workloads)*len(budgets)*len(families)*(len(GAMMAS)+len(MUS)-2),
        'structural_optimizer': 'global minimum of the unchanged parameter-free J over the measured count space, hard rho<=.8, safe memory, admitted paths; CPU design oracle, not the V3 bounded GPU search',
        'Generalist_upper_count': 'all frozen empirical counts subject to memory; no artificial total-G cap of 2',
        'g_unchanged': True, 'unknown_counts': 'NOT_REQUIRED_FOR_CURRENT_NEXT_STAGE; not extrapolated',
        'per_family_scaling': 'whole model: all supported service paths and all measured replica memory points; ASR/LID share some models and are not independent categories',
        'numerical_J_certificate_seconds': 1e-8, 'numerical_routing_fraction_zero': 1e-9,
        'phase_definitions': {
            'base_phase': 'G_ONLY if no S resident; otherwise G_DOMINANT if every active G/S-overlap task has G majority, S_DOMINANT if every overlap task has S majority, HYBRID otherwise',
            'primary_priority': ['INFEASIBLE', 'MULTI_G', 'S_PLUS_G_OVERFLOW', 'G_ONLY', 'G_DOMINANT', 'S_DOMINANT', 'HYBRID'],
            'MULTI_G': 'at least two Generalist replicas (identities and replicas both recorded)',
            'S_PLUS_G_OVERFLOW': 'positive S and G routing on at least one overlap task; descriptive split only, not a physical exhaustion claim',
            'dominance_boundary': 'majority=1/2 per-task routing fraction; labels do not enter optimization'},
        'T6_definition': 'a previously S-served active task loses all resident dedicated S paths and is served by admitted G; planning removal, not a measured low-QPS economic claim',
        'refinement': {'scope': 'global axes only; earliest ANY T1..T6 coarse bracket on each axis',
            'method': 'deterministic midpoint bisection inside first coarse bracket',
            'relative_interval_width': '(high-low)/high <= .05', 'max_iterations': 6,
            'no_per_task_refinement': True},
        'protected_references': refs, 'baseline_calibration': ref(BASE/'efficiency_optimal_baseline/replica_scaling.json'),
        'development_source_refs': [ref(ROOT/'results/module8e1_method_revision/development_workload_designation.json'), ref(cells_path)],
        'GPU_runs': 0, 'heldout_data_used': False, 'model_search_downloads': False}
    save('experiment_freeze.json', plan)
    save('specialist_selection_rule.json', {'frozen_before_sensitivity_results': True, 'at_utc': now(),
        'steps': ['public real same-task Specialist candidates', 'original frozen quality admission',
            'measure real latency and memory', 'quality-max / latency-min / memory-min Pareto frontier',
            'on that frontier choose minimum distance to sensitivity transition region',
            'ties: higher frozen-task quality utility, then lower measured memory, then deterministic model ID'],
        'distance': 'unweighted L-infinity distance in (log latency ratio, log memory ratio) to the closed target region; ratios referenced to the frozen fastest admitted Specialist for that task',
        'distance_to_union': 'minimum distance to any reported target rectangle; no weights',
        'right_censored_targets': 'nearest frontier boundary outside scanned range is a search lower bound, not evidence that any farther point transitions',
        'before_model_choice': 'freeze candidate metadata, quality, latency, memory and Pareto/distance choice before viewing final V3 outcome',
        'this_module_searches_candidates': False})
    print(json.dumps({'frozen': True, 'workloads': len(workloads), 'budgets': budgets,
        'global_queries': plan['global_queries'], 'per_task_nonbaseline_queries': plan['per_task_nonbaseline_queries'],
        'family_speedups': families}), flush=True)


if __name__ == '__main__': main()
