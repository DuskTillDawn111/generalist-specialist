"""Deterministic CPU context workers; cache complete mathematical witnesses."""
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from .common import *
from .solver import StructuralSolver
from .metrics import describe, events


def solve_query(workload, budget, gamma, mu, family, solver, c, roles, cal):
    key = query_key(workload['id'], budget['id'], gamma, mu, family)
    path = OUT / 'query_cache' / f'{key}.json'
    if path.exists(): return read(path)
    cc, ac = scaled(c, roles, cal, gamma, mu, family)
    fit = solver.solve(workload['rates'], cc, roles, ac, budget['safe_memory_budget_mib'])
    row = {'workload': workload['id'], 'regime': workload['regime'], 'budget': budget['id'],
        'memory_budget_mib': budget['safe_memory_budget_mib'], 'gamma': gamma, 'mu': mu,
        'scaled_S_models': sorted(family) if family is not None else sorted(m for m in c if roles[m]=='specialist'),
        'kind': 'COUNTERFACTUAL_CPU_ONLY', **describe(fit, workload, roles, c)}
    result = {'row': row, 'fit': fit}
    path.parent.mkdir(exist_ok=True); save_json(path, result)
    return result


def context(stage, wi, bi):
    plan = read(OUT / 'experiment_freeze.json'); workload = plan['workloads'][wi]; budget = plan['budgets'][bi]
    c, hw, roles, cal = baseline(); solver = StructuralSolver(); rows = []
    base = solve_query(workload, budget, 1., 1., None, solver, c, roles, cal)['row']
    if stage == 'global':
        for gamma in GAMMAS:
            for mu in MUS:
                row = solve_query(workload, budget, gamma, mu, None, solver, c, roles, cal)['row']
                rows.append({**row, **events(base, row, workload, c)})
    else:
        for family in plan['families_in_speedup_order']:
            for axis, values in [('latency', GAMMAS), ('memory', MUS)]:
                for value in values:
                    if axis == 'memory' and value == 1.: continue
                    gamma, mu = (value, 1.) if axis == 'latency' else (1., value)
                    # Real (1,1) is shared; never introduce family-specific fake baselines.
                    row = base if value == 1. else solve_query(workload, budget, gamma, mu, family['models'], solver, c, roles, cal)['row']
                    rows.append({**row, 'task_family': family['task'], 'axis': axis,
                        'baseline_speedup': family['baseline_speedup'], **events(base, row, workload, c)})
    name = f'{stage}_contexts/{workload["id"]}_{budget["id"]}.json'; save(name, {'rows': rows})
    return {'context': workload['id']+'/'+budget['id'], 'stage': stage, 'rows': len(rows)}


def main(stage):
    plan = read(OUT / 'experiment_freeze.json'); begin = time.perf_counter()
    jobs = [(stage, wi, bi) for wi in range(len(plan['workloads'])) for bi in range(len(plan['budgets']))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(context, *job) for job in jobs]
        for f in as_completed(futures): print(json.dumps(f.result()), flush=True)
    rows = []
    for _, wi, bi in jobs:
        name = f'{stage}_contexts/{plan["workloads"][wi]["id"]}_{plan["budgets"][bi]["id"]}.json'
        rows += read(OUT / name)['rows']
    csvsave('global_sensitivity_grid.csv' if stage=='global' else 'per_task_sensitivity.csv', rows)
    if stage=='global':
        csvsave('phase_map.csv', [{k:v for k,v in r.items() if k not in ['routing','rho']} for r in rows])
    save(f'{stage}_stage_status.json', {'status':'COMPLETE', 'rows':len(rows),'seconds':time.perf_counter()-begin,'GPU_runs':0})
    print(json.dumps({'stage':stage,'complete_rows':len(rows),'seconds':time.perf_counter()-begin}),flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['global','per_task']);main(p.parse_args().stage)
