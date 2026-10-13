"""Run only the unchanged CPU optimizer on frozen real operating points."""
import concurrent.futures
import copy
import math
from .common import *
from module12.solver import StructuralSolver
from module12.metrics import describe
from deployment_core.replica_calibration import ReplicaCalibration


PRIMARY = 'PRIMARY_NEW_CHECKPOINTS_SINGLE_REPLICA'
MATCHED = 'MATCHED_ALL_SPECIALIST_COUNTS_0_1'


def restrict(c, roles, cal):
    bounds = {m: 1 if roles[m] == 'specialist' else cal.evidence_bounds[m] for m in c}
    return ReplicaCalibration(
        {m: {n: v for n, v in cal.capacity_factors[m].items() if n <= bounds[m]} for m in c},
        {m: {n: v for n, v in cal.memory_table_mib[m].items() if n <= bounds[m]} for m in c},
        bounds, copy.deepcopy(cal.model_evidence), dict(cal.source_hashes)).validate(c)


def context(job):
    work, budget = job
    catalogs = {E: catalog(read(OUT / 'regime_e.json')), H: catalog(read(OUT / 'regime_h.json'))}
    solver = StructuralSolver(); rows = []; fits = {}
    for scope in [PRIMARY, MATCHED]:
        for regime in [E, H]:
            c, roles, cal = catalogs[regime]
            if scope == MATCHED:
                cal = restrict(c, roles, cal)
            fit = solver.solve(work['rates'], c, roles, cal, budget['safe_memory_budget_mib'])
            fit['evidence_scope'] = 'REAL_MEASURED_CPU_STRUCTURAL_DIAGNOSTIC_NOT_FINAL_DEPLOYMENT'
            fits[scope + '::' + regime] = fit
            row = {'workload': work['id'], 'workload_regime': work['regime'], 'budget': budget['id'],
                   'memory_budget_mib': budget['safe_memory_budget_mib'], 'Specialist_regime': regime,
                   'count_scope': scope, **describe(fit, work, roles, c),
                   'frozen_Module9_solver': fit.get('frozen_Module9_solver'),
                   'evidence_scope': fit['evidence_scope']}
            if row['feasible']:
                row['Generalist_routed_fraction'] = math.fsum(work['rates'][k] * v for k, v in row['task_G_fraction'].items()) / math.fsum(work['rates'].values())
                row['G_served_task_sets'] = {m: sorted({path.split('::')[0] for path, v in row['routing'].items() if v > 1e-9 and path.split('::')[1] == m}) for m in row['Generalist_identities']}
                row['memory_slack_mib'] = row['memory_budget_mib'] - row['safe_memory_mib']
                row['certificate_passed'] = fit['certificate']['passed']
            rows.append(row)
    path = OUT / 'witnesses' / (work['id'] + '__' + budget['id'] + '.json')
    path.write_text(json.dumps({'rows': rows, 'fits': fits}, indent=2) + '\n')
    return rows


def main():
    freeze = read(OUT / 'screen_protocol_freeze.json')
    verify(freeze['immutable_inputs'] + freeze['implementation'] + [freeze['plan_reference']])
    assert __import__('os').environ.get('CUDA_VISIBLE_DEVICES') == ''
    (OUT / 'witnesses').mkdir(exist_ok=True)
    jobs = [(w, b) for w in freeze['workloads'] for b in freeze['budgets']]
    assert len(jobs) == 27
    rows = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(context, j) for j in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result(); rows.extend(result)
            print('SCREEN', result[0]['workload'], result[0]['budget'], len(rows), flush=True)
    rows.sort(key=lambda r: (r['workload'], r['budget'], r['count_scope'], r['Specialist_regime']))
    csvsave('single_replica_structural_replay.csv', [r for r in rows if r['count_scope'] == PRIMARY])
    csvsave('single_replica_structural_replay_matched_counts.csv', [r for r in rows if r['count_scope'] == MATCHED])
    save('screen_rows.json', rows)
    save('screen_execution_status.json', {'at_utc': now(), 'status': 'COMPLETE', 'primary_rows': 54, 'matched_rows': 54, 'contexts': 27, 'GPU_runs': 0, 'new_g_gt1_values_created': 0})
    print('COMPLETE', len(rows), flush=True)


if __name__ == '__main__':
    main()
