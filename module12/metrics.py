from .common import *

EPS = 1e-9


def describe(fit, workload, roles, catalog):
    if fit['J_seconds'] is None:
        return {'feasible': False, 'phase': 'INFEASIBLE', 'base_phase': 'INFEASIBLE',
            'phase_flags': ['INFEASIBLE'], 'deployment': None, 'N_G': None, 'N_S': None,
            'Specialist_identities': [], 'Generalist_identities': [], 'routing': None,
            'task_G_fraction': {}, 'split_tasks': [], 'max_rho': None, 'J_star_seconds': None,
            'safe_memory_mib': None, 'solver_status': fit['status']}
    x = fit['deployment']; routing = fit['routing']; rates = workload['rates']
    ss = sorted(m for m, n in x.items() if n and roles[m] == 'specialist')
    gs = sorted(m for m, n in x.items() if n and roles[m] == 'generalist')
    overlap = [k for k, v in rates.items() if v > 0 and any(roles[m] == 'specialist' and k in p.supported_tasks for m, p in catalog.items())]
    fg = {k: math.fsum(v for path, v in routing.items() if path.split('::')[0] == k and roles[path.split('::')[1]] == 'generalist') / rates[k] for k in rates if rates[k] > 0}
    split = [k for k in overlap if EPS < fg[k] < 1-EPS]
    if not ss: base = 'G_ONLY'
    elif all(fg[k] < .5 for k in overlap): base = 'S_DOMINANT'
    elif all(fg[k] > .5 for k in overlap): base = 'G_DOMINANT'
    else: base = 'HYBRID'
    NG = sum(x[m] for m in gs); NS = sum(x[m] for m in ss)
    flags = [base] + (['S_PLUS_G_OVERFLOW'] if split else []) + (['MULTI_G'] if NG >= 2 else [])
    phase = 'MULTI_G' if NG >= 2 else 'S_PLUS_G_OVERFLOW' if split else base
    return {'feasible': True, 'phase': phase, 'base_phase': base, 'phase_flags': flags,
        'deployment': x, 'N_G': NG, 'N_S': NS, 'N_G_identities': len(gs), 'N_S_identities': len(ss),
        'Specialist_identities': ss, 'Generalist_identities': gs, 'replicas': {m: n for m, n in x.items() if n},
        'routing': routing, 'rho': fit['rho'], 'max_rho': fit['max_rho'],
        'J_star_seconds': fit['J_seconds'], 'safe_memory_mib': fit['safe_memory_mib'],
        'task_G_fraction': fg, 'split_tasks': split,
        'S_fraction_overlap_requests': math.fsum(rates[k]*(1-fg[k]) for k in overlap)/math.fsum(rates[k] for k in overlap) if overlap else None,
        'solver_status': fit['status'], 'global_lower_bound_seconds': fit['global_lower_bound_seconds'],
        'global_gap_seconds': fit['global_gap_seconds'], 'solve_seconds': fit.get('seconds'),
        'exact_zero_demand_residents_removed': fit.get('exact_zero_demand_residents_removed', [])}


def events(base, row, workload, catalog):
    if not base['feasible']: return {'events': [], 'status': 'BASELINE_INFEASIBLE'}
    if not row['feasible']: return {'events': [], 'status': 'BECAME_INFEASIBLE'}
    removed = sorted(set(base['Specialist_identities']) - set(row['Specialist_identities']))
    replaced = []
    for m in removed:
        tasks = [k for k in catalog[m].supported_tasks if workload['rates'][k] > 0]
        if any(row['task_G_fraction'][k] > base['task_G_fraction'][k] + EPS for k in tasks): replaced.append(m)
    new_split = sorted(set(row['split_tasks']) - set(base['split_tasks']))
    lost_paths = [k for k, v in workload['rates'].items() if v > 0 and
        any(k in catalog[m].supported_tasks for m in base['Specialist_identities']) and
        not any(k in catalog[m].supported_tasks for m in row['Specialist_identities']) and row['task_G_fraction'][k] > EPS]
    flags = {'T1': row['N_S'] < base['N_S'], 'T2': row['N_G'] > base['N_G'],
        'T3': base['N_G'] < 2 <= row['N_G'], 'T4': bool(replaced), 'T5': bool(new_split),
        'T6': workload['regime'] == 'low_load' and bool(lost_paths)}
    return {'events': [k for k, v in flags.items() if v], 'status': 'COMPARABLE_FEASIBLE',
        'removed_Specialists': removed, 'G_replaced_Specialists': replaced,
        'new_split_tasks': new_split, 'low_load_tasks_without_S': lost_paths}
