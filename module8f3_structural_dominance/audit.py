"""Frozen standalone LP enumeration and causal-boundary diagnostic, CPU only."""
import itertools, sys
from collections import Counter
import numpy as np
from deployment_core import Workload
from deployment_search.routing import solve_pressure_relaxation, solve_canonical_routing
from .common import *

def prepare():
    assert not (OUT/'audit_manifest.json').exists(), 'Do not regenerate audit freeze'
    OUT.mkdir(exist_ok=True)
    refs={}
    for directory in (A,F,DEV,R/'module9a1_task_conditioned_pilot',R/'module9a2_repeatability'):
        for r in read(directory/'artifact_hashes.json').values(): refs[r['path']]=r
    for r in read(F/'resource_campaign_gate.json')['frozen_source_hashes'].values(): refs[r['path']]=r
    for r in read(R/'module8e1_method_revision/method_v2_source_hashes.json')['sources'].values(): refs[r['path']]=r
    # Verify original evidence; no frozen module's write-capable verifier is called.
    verify(list(refs.values()))
    assert read(DEV/'final_closeout.json')['STATIC_MAIN_EXPERIMENTS_COMPLETE']
    assert read(R/'module9a1_task_conditioned_pilot/pilot_gate.json')['status']=='GATE_PARTIAL'
    assert read(R/'module9a2_repeatability/repeatability_gate.json')['status']=='REPEATABILITY_STRONG_SUPPORT'
    rule={
      'STRUCTURAL_AUDIT_ONLY':True,'GPU_RUNS':0,'frozen_at_utc':now(),
      'enumeration':'Exactly 8F-1A state_id/vector rows, including empty and quality-disabled states; no new replicas.',
      'oracle':'Frozen relaxed canonical min-max pressure LP; capacity-feasible serving witnesses separately checked at frozen rho_max. Not an E2E oracle.',
      'homogeneity':'All 12 cells share mix_03 composition. Solve each bound-valid covered vector once at C01 lambda; scale routing and z by exact intensity ratio, verify conservation and pressure. Independently solve each reported cell oracle and baseline with existing LP.',
      'Pareto':'Memory/pressure weak dominance with one strict inequality. Numerical equality uses existing LP_TOLERANCE=1e-9, not a scientific near-optimal gap cutoff. All bound-valid covered states included; expose global, physical and cell-budget/capacity flags.',
      'counts':'N_G/N_S count resident model identities; replica count recorded separately. Frozen regime: Specialist-heavy if N_S>N_G; Hybrid if both positive; Generalist-heavy display applies to Generalist-only. Preserve overlapping hybrid flag.',
      'sharing':'A resident Generalist receives positive flow for >=2 active tasks in its exact LP witness. Also report feasibility of forced multi-task Generalist sharing; never rename Specialist-heavy based on story.',
      'removal':'Primary reference is frozen RC-SG base Generalist plus unique task-fastest admitted Specialists (one each), irrespective of cell budget, with reference budget feasibility explicit. Additional references for each admitted Generalist reveal available fallback edges. Never invent an inadmissible Generalist fallback; uncovered removal is COVERAGE_INFEASIBLE.',
      'all_specialist_condition':'Pure 0G six-task coverage is explicitly tested. Separately audit 1G+all unique fastest task-Specialists residency, and all admitted Specialist identities. These are different conditions.',
      'memory_counterfactual':'M(c)=safe_G+c*safe_S, c>=1. Exact state budget exit events (B-safe_G)/safe_S and affine Pareto-memory envelope crossings. Feasible at equality; changes refer to immediately above event. No coarse multiplier grid and no benchmark edits.',
      'speed_counterfactual':'For every admitted Specialist-task edge, s(alpha)=s_fastest_admitted_G-alpha*(s_fastest_admitted_G-s_actual_S), alpha in [0,1]. All other edges/g/memory fixed. Screen 257 equally spaced points (resolution 1/256), refine every observed deployment/composition transition by bisection to width <=1e-8. State explicitly that sub-grid islands are not exhaustively certified.',
      'combined_map':'Use memory events and speed transitions plus their interval midpoints; evaluate representative tight/intermediate/loose budgets. Planning-only counterfactual map, no new official resource point.',
      'diagnosis':'Report each user criterion, exact gaps/distributions and counts. No percent near-optimal threshold; distinguish Specialist residency advantage from RC-SG exact z optimality and from runtime E2E. Final classification follows evidence, with unmet criteria explicit.',
      'frozen_refs':list(refs.values())}
    save('audit_manifest.json',rule)
    print('FROZEN INPUT VERIFICATION PASS',len(refs),flush=True)

def frontier(states):
    # Retain all numerically identical nondominated objective points.
    order=sorted(states,key=lambda s:(s['M_safe_mib'],s['z_base'],s['vector']))
    best=float('inf');front=[];i=0
    while i<len(order):
        j=i+1
        while j<len(order) and abs(order[j]['M_safe_mib']-order[i]['M_safe_mib'])<=TOL: j+=1
        zmin=min(s['z_base'] for s in order[i:j])
        if zmin<best-TOL:
            front.extend(s for s in order[i:j] if abs(s['z_base']-zmin)<=TOL)
            best=zmin
        i=j
    return front

def main():
    if not (OUT/'audit_manifest.json').exists(): prepare()
    catalog,roles,cal=inputs();manifest=read(A/'deployment_space_manifest.json');ids=manifest['model_order']
    cells=read(F/'campaign_cells.json')['cells'];rho=read(R/'module8e2_heldout_freeze/method_comparison_freeze.json')['search_config']['rho_max'];factor=read(A/'memory_safety_factor.json')['safety_factor'];base=cells[0]['lambda_rps'];work=Workload(base)
    assert all(c['mixture']==cells[0]['mixture'] and all(math.isclose(c['lambda_rps'][t]/base[t],c['alpha']/cells[0]['alpha'],rel_tol=1e-12) for t in base) for c in cells)
    pool=read(R/'candidate_pool_v2_1/candidate_pool_v2_1.json');allroles={r['id']:r['role'] for r in pool['models'] if r['id'] in ids};service=read(R/'candidate_pool_v2_1/service_demand_v2_1.json')['models'];memory=read(A/'memory_catalog_frozen.json')['models'];g=read(R/'replica_capacity_calibration/replica_capacity_factors.json')['models']
    poolrows=[]
    for m in ids:
        for n in range(manifest['replica_bounds'][m]+1):
            poolrows.append({'model':m,'role':allroles[m],'quality_admitted_tasks':sorted(catalog[m].supported_tasks) if m in catalog else [],'verified_capability_tasks':service[m]['capabilities_verified'],'admitted_service_time_ms':catalog[m].service_time_ms if m in catalog else {},'measured_service_times_all_ms':service[m]['service_time_ms'],'replica_option':n,'M_plan_mib':memory[m]['values'][str(n)]['value'],'M_safe_mib':factor*memory[m]['values'][str(n)]['value'],'g_m':g[m]['g'][str(n)],'memory_provenance':memory[m]['values'][str(n)],'quality_disabled_fixed_zero':m not in catalog})
    csvsave('model_pool_audit.csv',poolrows)
    dominance=[];fastS={};fastG={}
    for t in base:
        gs=[m for m in catalog if roles[m]=='generalist' and t in catalog[m].supported_tasks];ss=[m for m in catalog if roles[m]=='specialist' and t in catalog[m].supported_tasks]
        gm=min(gs,key=lambda m:(catalog[m].service_time_ms[t],m)) if gs else None;sm=min(ss,key=lambda m:(catalog[m].service_time_ms[t],m)) if ss else None;fastG[t]=gm;fastS[t]=sm
        for c in cells:
            a=catalog[gm].service_time_ms[t] if gm else None;b=catalog[sm].service_time_ms[t] if sm else None;mem=factor*cal.memory_table_mib[sm][1] if sm else None;saving=c['lambda_rps'][t]*(a-b)/1000 if gm and sm else None
            dominance.append({'cell_id':c['cell_id'],'task':t,'lambda_rps':c['lambda_rps'][t],'fastest_G':gm,'fastest_S':sm,'G_s0_ms':a,'S_s0_ms':b,'speedup':a/b if a and b else None,'extra_S_safe_mib':mem,'worker_demand_saving_seconds_per_second':saving,'value_per_safe_mib':saving/mem if saving is not None else None,'status':'NO_ADMITTED_SPECIALIST' if not sm else 'AVAILABLE'})
    csvsave('task_level_dominance.csv',dominance)
    frozenrows=list(csv.DictReader((A/'deployment_memory_states.csv').open()));states=[];allrows=[]
    for j,row in enumerate(frozenrows):
        x={m:int(row['x_'+m]) for m in ids};vec=tuple(x[m] for m in ids)
        plan=math.fsum(memory[m]['values'][str(x[m])]['value'] for m in ids);safe=plan*factor
        assert math.isclose(safe,float(row['M_safe_mib']),rel_tol=0,abs_tol=1e-8)
        enabled=all(x[m]==0 for m in ids if m not in catalog);covered=enabled and all(any(x[m] and t in catalog[m].supported_tasks for m in catalog) for t in base)
        s={'state_id':int(row['state_id']),'vector':vec,'deployment':x,'N_G':sum(x[m]>0 for m in ids if allroles[m]=='generalist'),'N_S':sum(x[m]>0 for m in ids if allroles[m]=='specialist'),'total_replicas':sum(x.values()),'M_plan_mib':plan,'M_safe_mib':safe,'quality_coverage':covered,'quality_disabled_bound_valid':enabled,'physical_memory_feasible':safe<=49140,'z_base':None,'routing_base':{},'model_pressure_base':{}}
        if covered:
            route=solve_pressure_relaxation({m:x[m] for m in catalog},work,catalog,rho,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal);assert route
            s.update(z_base=route.max_pressure,routing_base=route_dict(route),model_pressure_base=route.model_pressure)
            s['G_safe_mib']=factor*math.fsum(memory[m]['values'][str(x[m])]['value'] for m in ids if allroles[m]=='generalist');s['S_safe_mib']=safe-s['G_safe_mib'];states.append(s)
        allrows.append({k:v for k,v in s.items() if k not in ['routing_base','model_pressure_base']})
        if (j+1)%1500==0: print('ENUMERATED',j+1,'LP',len(states),flush=True)
    assert len(frozenrows)==manifest['actual_enumerated_state_count']==13824
    assert [tuple(int(row['x_'+m]) for m in ids) for row in frozenrows]==list(itertools.product(*(range(manifest['replica_bounds'][m]+1) for m in ids)))
    csvsave('all_deployments_structural.csv',allrows);save('lp_states_cache.json',{'base_cell':cells[0],'model_order':ids,'states':states,'rho_max':rho,'fastest_S':fastS,'fastest_G':fastG})
    front=frontier(states);fids={s['state_id'] for s in front};fullrows=[];frontrows=[];compositions=[];oracles=[];lookup={s['vector']:s for s in states}
    v2={c['cell_id']:c for c in read(DEV/'v2_results.json')['cells']};rc={c['cell_id']:c for c in read(DEV/'rc_sg_results.json')['cells']}
    for c in cells:
        scale=c['alpha']/cells[0]['alpha'];B=c['memory_budget_mib'];population=[s for s in states if s['M_safe_mib']<=B];feas=[s for s in population if s['z_base']*scale<=rho+TOL]
        assert feas
        best=min(feas,key=lambda s:(s['z_base'],s['M_safe_mib'],s['vector']))
        route=solve_canonical_routing({m:best['deployment'][m] for m in catalog},Workload(c['lambda_rps']),catalog,rho,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal);assert route and math.isclose(route.max_pressure,best['z_base']*scale,abs_tol=1e-8)
        record={'cell_id':c['cell_id'],'mixture':c['mixture'],'intensity':c['intensity_level'],'alpha':c['alpha'],'B_mib':B,'oracle_state_id':best['state_id'],'oracle_deployment':best['deployment'],'oracle_routing':route_dict(route),'oracle_z':route.max_pressure,'oracle_safe_mib':best['M_safe_mib'],'oracle_N_G':best['N_G'],'oracle_N_S':best['N_S'],'oracle_replicas':best['total_replicas'],'oracle_regime':counts(best['deployment'],roles)[2],'feasible_states':len(feas)}
        for method,raw in [('RC_SG',rc[c['cell_id']]),('v2',v2[c['cell_id']])]:
            x={m:raw['deployment'].get(m,0) for m in ids};s=lookup[tuple(x[m] for m in ids)];r=solve_pressure_relaxation({m:x[m] for m in catalog},Workload(c['lambda_rps']),catalog,rho,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal);assert math.isclose(r.max_pressure,s['z_base']*scale,abs_tol=1e-8)
            record.update({method+'_deployment':x,method+'_z':r.max_pressure,method+'_safe_mib':s['M_safe_mib'],method+'_gap':(r.max_pressure-route.max_pressure)/route.max_pressure,method+'_equals_oracle':s['vector']==best['vector'],method+'_z_equals_oracle':abs(r.max_pressure-route.max_pressure)<=TOL,method+'_Pareto':s['state_id'] in fids,method+'_routing':route_dict(r),method+'_frozen_runtime_routing':raw['routing']})
        oracles.append(record)
        for s in states:
            z=s['z_base']*scale;mem=s['M_safe_mib']<=B;capacity=z<=rho+TOL;ng,ns,regime=counts(s['deployment'],roles)
            row={'cell_id':c['cell_id'],'state_id':s['state_id'],'deployment_vector':s['vector'],'deployment':s['deployment'],'N_G':ng,'N_S':ns,'total_replicas':s['total_replicas'],'M_safe_mib':s['M_safe_mib'],'z_star':z,'routing':{k:v*scale for k,v in s['routing_base'].items()},'quality_capability_covered':True,'memory_feasible':mem,'capacity_feasible':capacity,'cell_feasible':mem and capacity,'Pareto_global':s['state_id'] in fids,'physical_memory_feasible':s['physical_memory_feasible'],'regime':regime,'hybrid_flag':bool(ng and ns)}
            fullrows.append(row)
            if row['Pareto_global']:frontrows.append(row)
        cf=[s for s in front if s['M_safe_mib']<=B and s['z_base']*scale<=rho+TOL]
        compositions.append({'cell_id':c['cell_id'],'B_mib':B,'frontier_all_bound_valid':len(front),'frontier_physical':sum(s['physical_memory_feasible'] for s in front),'frontier_cell_feasible':len(cf),'Generalist_count_distribution':dict(Counter(s['N_G'] for s in cf)),'Specialist_count_distribution':dict(Counter(s['N_S'] for s in cf)),'0G_nS':sum(s['N_G']==0 for s in cf),'1G_nS':sum(s['N_G']==1 for s in cf),'2G_nS':sum(s['N_G']==2 for s in cf),'3G_nS':sum(s['N_G']==3 for s in cf),'Specialist_heavy_count':sum(s['N_S']>s['N_G'] for s in cf),'Generalist_only_count':sum(s['N_G']>0 and s['N_S']==0 for s in cf),'Generalists_more_than_Specialists_count':sum(s['N_G']>s['N_S'] for s in cf),'RC_SG_Pareto':record['RC_SG_Pareto'],'v2_Pareto':record['v2_Pareto']})
    csvsave('pareto_all_states.csv',fullrows);csvsave('pareto_front_by_cell.csv',frontrows);csvsave('pareto_composition_summary.csv',compositions);csvsave('structural_oracle_12cells.csv',oracles)
    save('structural_stage_results.json',{'status':'PASS','STRUCTURAL_AUDIT_ONLY':True,'GPU_RUNS':0,'enumerated_vectors':len(frozenrows),'covered_LP_states':len(states),'canonical_LP_calls':len(states)+36,'global_frontier_states':len(front),'physical_frontier_states':sum(s['physical_memory_feasible'] for s in front),'oracle_cells':oracles,'frontier_state_ids':sorted(fids),'complete_enumeration_claim_is_algorithm_claim':False})
    print('STRUCTURAL LP COMPLETE',len(states),'PARETO',len(front),flush=True)

if __name__=='__main__':main()
