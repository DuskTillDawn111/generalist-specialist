"""Final read-only evidence checks and cross-module closure accounting."""
from .common import *
from module14c.analyze import minimum_real_memory_cover

def main():
    out = read(OUT/'structural_transition_comparison.json')
    old_c = read(C/'transition_summary.json')['comparisons']
    current = {(p['workload'], p['budget']): p for p in out['comparisons']}
    originals = [p for p in old_c if p['T1']]
    persist = [p['workload']+'/'+p['budget'] for p in originals if current[p['workload'], p['budget']]['T1']]
    vanish = [p['workload']+'/'+p['budget'] for p in originals if not current[p['workload'], p['budget']]['T1']]
    old_bad = [p['workload']+'/'+p['budget'] for p in old_c if not p['H_feasible']]
    now_bad = [p['workload']+'/'+p['budget'] for p in out['comparisons'] if not p['H_feasible']]
    assert len(originals) == 14 and len(old_bad) == 8
    assert len(current) == 27
    rows = read(OUT/'structural_replay_rows.json')
    e = read(OUT/'regime_e_reconciled.json'); h = read(OUT/'heavy_empirical_x3_reconciled.json')
    freeze = read(C/'screen_protocol_freeze.json')
    budgets = {b['id']: b['safe_memory_budget_mib'] for b in freeze['budgets']}
    for row in rows:
        assert row['memory_budget_mib'] == budgets[row['budget']]
        data = e if row['Specialist_regime'] == E else h
        if row['feasible']:
            assert all(str(n) in data['capacity_factors'][m] and str(n) in data['memory_table_mib'][m] for m,n in row['replicas'].items())
            assert row['safe_memory_mib'] <= row['memory_budget_mib'] + 1e-6
    newly_infeasible_e = []
    for row in rows:
        if row['Specialist_regime'] == E and not row['feasible']:
            w = next(w for w in freeze['workloads'] if w['id'] == row['workload'])
            bound = minimum_real_memory_cover(e, [k for k,v in w['rates'].items() if v > 0])
            assert bound['minimum_safe_memory_mib'] > row['memory_budget_mib'] + 1e-6
            newly_infeasible_e.append({'context':row['workload']+'/'+row['budget'], 'minimum_measured_memory_cover':bound, 'budget_mib':row['memory_budget_mib']})
    points = read(OUT/'ocr_x3_empirical_points.json')['points']
    for p in points:
        assert p['valid_all_three'] and p['x3_stability'] == ['steady']*3, p
        assert h['capacity_factors'][p['family']]['3'] == p['g3']
        assert h['memory_table_mib'][p['family']]['3'] == p['M3_postwarm_idle_mib']
        assert '4' not in h['capacity_factors'][p['family']]
    runs = read(OUT/'ocr_x3_measurement_rows.json')
    assert len(runs) == 24 and all(r['worker_count'] == r['count'] and r['gpu_release_verified'] for r in runs)
    assert {r['count'] for r in runs} == {2,3}
    assert {r['family'] for r in runs} == {p['family'] for p in points}
    gate = read(OUT/'module14f_gate.json')
    stopped_no_shift = gate['gate'] != 'HEAVY_GENERALIST_SHIFT_SUPPORTED'
    if stopped_no_shift:
        for name in ['HEAVY_MULTI_HOTSPOT','HEAVY_LOW_QPS_FIXED_COST']:
            proposal = read(OUT/('next_stage_'+name+'_manifest.json'))
            assert proposal['status'] == 'PROPOSED_NOT_EXECUTED' and proposal['new_experiments_performed'] == 0
    extra = {'at_utc': now(), '14C_original_NS_reductions':14, 'persistent_original_NS_reductions':len(persist), 'persistent_contexts':persist, 'no_longer_reduced_contexts':vanish, '14C_original_H_infeasible':old_bad, '14F_H_infeasible':now_bad, 'original_H_infeasible_recovered':sorted(set(old_bad)-set(now_bad)), 'new_H_infeasible':sorted(set(now_bad)-set(old_bad)), 'all_requested_x3_three_independent_valid_steady':True, 'measured_state_budget_checks_passed':True, 'new_x_ge4_sessions':0, 'two_independent_proposals_not_executed':stopped_no_shift, 'original_27_contexts_closed':True, 'E_feasible_14C':sum(p['E_feasible'] for p in old_c), 'E_feasible_14F':out['E_feasible'], 'new_E_memory_infeasibility_witnesses':newly_infeasible_e}
    save('final_closure_audit.json',extra)
    summary = OUT/'module14f_summary.md'
    text = summary.read_text().split('\n## 与14C的闭环对照')[0]
    text += '\n## 与14C的闭环对照\n\n原14个N_S reduction保留'+str(len(persist))+'个。保留：'+', '.join(persist)+'。\n\n原8个H infeasible恢复：'+str(extra['original_H_infeasible_recovered'])+'；新H infeasible：'+str(extra['new_H_infeasible'])+'。\n'
    if newly_infeasible_e:
        text += '\n统一的empirical postwarm-memory修正也作用于E：E可行数由27降为19，8个含VQA的B1 context都不再可行。它们即使忽略capacity，最小实测safe-memory task cover='+str(newly_infeasible_e[0]['minimum_measured_memory_cover']['minimum_safe_memory_mib'])+'MiB也超过冻结B1='+str(newly_infeasible_e[0]['budget_mib'])+'MiB；不是只提高Heavy成本或修改budget。\n'
    text += '\n重新优化的mix_03_I3/B6 Heavy在CPU修正表下可以选取另一合法部署；其joint GPU稳定性尚未重新验证，本模块没有把新CPU最优当作GPU通过。\n'
    summary.write_text(text)
    artifacts = [ref(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json']
    implementation = [ref(p) for p in sorted((ROOT/'module14f').glob('*.py'))]
    save('artifact_manifest.json', {'at_utc':now(), 'module':'14F', 'status':'COMPLETE_STOPPED', 'artifacts':artifacts, 'implementation':implementation, 'historical_integrity_check':ref(OUT/'completion_verification.json'), 'final_gate':ref(OUT/'module14f_gate.json')})
    print(json.dumps(extra,ensure_ascii=False))
if __name__ == '__main__': main()
