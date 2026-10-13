"""Independent mathematical checks; no serving/GPU import or execution."""
from .common import *
from .counterfactual import select
import numpy as np
from dataclasses import replace
from deployment_core import Workload
from deployment_search.routing import solve_pressure_relaxation

def main():
    cache=read(OUT/'lp_states_cache.json');states=cache['states'];M=np.array([s['M_safe_mib'] for s in states]);z=np.array([s['z_base'] for s in states]);ng=np.array([s['N_G'] for s in states]);sid=np.array([s['state_id'] for s in states]);brute=[]
    for j,s in enumerate(states):
        dominated=np.any((M<=M[j]+TOL)&(z<=z[j]+TOL)&((M<M[j]-TOL)|(z<z[j]-TOL)))
        if not dominated:brute.append(s['state_id'])
    assert sorted(brute)==read(OUT/'structural_stage_results.json')['frontier_state_ids']
    cells=read(F/'campaign_cells.json')['cells'];results=read(OUT/'structural_stage_results.json')['oracle_cells'];counts=[]
    for c,r in zip(cells,results):
        scale=c['alpha']/cache['base_cell']['alpha'];valid=(M<=c['memory_budget_mib'])&(z*scale<=cache['rho_max']+TOL)
        assert sum(valid)==r['feasible_states']==c['capacity_feasible_space_count']
        counts.append({'cell_id':c['cell_id'],'feasible_states':int(sum(valid)),'matches_8F1B':True})
        assert abs(z[valid].min()*scale-r['oracle_z'])<1e-8
    # Counterfactual matrix implementation must reduce exactly to the existing
    # frozen LP when used with ephemeral, explicitly diagnostic service profiles.
    catalog,roles,cal=inputs();physical=[s for s in states if s['physical_memory_feasible']];G=np.array([s['G_safe_mib'] for s in physical]);S=np.array([s['S_safe_mib'] for s in physical]);screen=np.load(OUT/'speed_alpha_screen.npz');checks=[]
    for a in [0.,.5,.96875,1.]:
        iz=np.flatnonzero(abs(screen['alpha']-a)<1e-12)[0];za=screen['z'][iz];changed={}
        for m,p in catalog.items():
            demands={}
            for k,v in p.service_time_ms.items():
                ref=catalog[cache['fastest_G'][k]].service_time_ms[k]
                demands[k]=ref-a*(ref-v) if roles[m]=='specialist' else v
            changed[m]=replace(p,service_time_ms=demands)
        for B in sorted(set(c['memory_budget_mib'] for c in cells)):
            idx=select(physical,za,G,S,B);s=physical[idx]
            route=solve_pressure_relaxation({m:s['deployment'][m] for m in catalog},Workload(cache['base_cell']['lambda_rps']),changed,cache['rho_max'],replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
            assert route and abs(route.max_pressure-za[idx])<1e-8
            checks.append({'alpha_speed':a,'B_mib':B,'state_id':s['state_id'],'matrix_and_existing_canonical_LP_match':True})
    # Every scaled witness remains a conservation- and capacity-valid optimum.
    max_residual=0
    for s in states:
        for k,lam in cache['base_cell']['lambda_rps'].items():
            value=math.fsum(v for key,v in s['routing_base'].items() if key.split('::')[0]==k)
            max_residual=max(max_residual,abs(value-lam));assert math.isclose(value,lam,rel_tol=1e-7,abs_tol=0)
    rows=0;pairs=set()
    with (OUT/'pareto_all_states.csv').open() as f:
        for r in csv.DictReader(f):
            rows+=1;pairs.add((r['cell_id'],r['state_id']))
    assert rows==len(pairs)==13824*12
    assert read(R/'module9a1_task_conditioned_pilot/pilot_gate.json')['status']=='GATE_PARTIAL'
    assert read(R/'module9a2_repeatability/repeatability_gate.json')['status']=='REPEATABILITY_STRONG_SUPPORT'
    verify(read(OUT/'audit_manifest.json')['frozen_refs'])
    save('independent_verification.json',{'status':'PASS','GPU_RUNS':0,'Pareto_full_pairwise_check':True,'feasible_counts_matches_all_12_frozen_cells':counts,'counterfactual_existing_LP_reduction_checks':checks,'base_routing_conservation_max_absolute_residual_rps':max_residual,'all_cell_state_rows_and_unique_pairs':rows,'old_references_and_gates_unchanged':True,'no_heldout_validation_or_full_Module9A_started':True})
    save('execution_source_hashes.json',{str(f.resolve()):sha(f) for f in Path(__file__).parent.glob('*.py')})
    save('artifact_hashes.json',{str(f.relative_to(OUT)):{'path':str(f.resolve()),'sha256':sha(f)} for f in sorted(OUT.rglob('*')) if f.is_file() and f.name not in ['artifact_hashes.json','verification_execution.log']})
    print('INDEPENDENT VERIFICATION PASS',flush=True)

if __name__=='__main__':main()
