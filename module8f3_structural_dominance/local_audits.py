"""Specialist removal, frozen greedy choices, and intensity decomposition."""
from .common import *
import numpy as np
from deployment_core import Workload
from deployment_search.routing import solve_pressure_relaxation
from module8f2_resource_static.planning import rc_sg

def main():
    catalog,roles,cal=inputs();cache=read(OUT/'lp_states_cache.json');ids=cache['model_order'];rho=cache['rho_max'];cells=read(F/'campaign_cells.json')['cells'];factor=read(A/'memory_safety_factor.json')['safety_factor'];states={tuple(s['vector']):s for s in cache['states']};oracles=read(OUT/'structural_stage_results.json')['oracle_cells']
    def safe(x):return factor*math.fsum(cal.memory_table_mib[m][x.get(m,0)] for m in catalog)
    def solve(x,c):return solve_pressure_relaxation({m:x.get(m,0) for m in catalog},Workload(c['lambda_rps']),catalog,rho,replica_bounds=cal.evidence_bounds,capacity_mode='empirical',calibration=cal)
    def demand(r):return math.fsum(r.task_contributions.values()) if r else None
    removal=[];greedy=[];decomposition=[]
    v2={c['cell_id']:c for c in read(DEV/'v2_results.json')['cells']};rc={c['cell_id']:c for c in read(DEV/'rc_sg_results.json')['cells']}
    specialists=sorted(set(m for m in cache['fastest_S'].values() if m))
    for cell,oracle in zip(cells,oracles):
        actual=rc_sg(cell['lambda_rps'],cell['memory_budget_mib'],catalog,roles,cal,factor,ids)
        assert actual==cell['preflight']['RC_SG'],cell['cell_id']
        primary=actual['trace'][0]['generalist']
        for g in sorted(m for m in catalog if roles[m]=='generalist'):
            x={m:int(m==g or m in specialists) for m in ids};before=solve(x,cell);assert before
            for specialist in specialists:
                afterx=dict(x);afterx[specialist]=0;after=solve(afterx,cell)
                affected=sorted(catalog[specialist].supported_tasks)
                fallback={t:{m:r for (k,m),r in after.routing.items() if k==t and r>0} for t in affected} if after else {}
                uncovered=[t for t in cell['lambda_rps'] if not any(afterx[m] and t in catalog[m].supported_tasks for m in catalog)]
                generalists={t:[m for m in fallback.get(t,{}) if roles[m]=='generalist'] for t in affected}
                removal.append({'cell_id':cell['cell_id'],'reference_G':g,'primary_reference':g==primary,'reference_deployment':x,'reference_memory_feasible':safe(x)<=cell['memory_budget_mib'],'removed_specialist':specialist,'memory_saved_mib':safe(x)-safe(afterx),'M_safe_before_mib':safe(x),'M_safe_after_mib':safe(afterx),'status':'COVERAGE_INFEASIBLE' if after is None else 'CAPACITY_INFEASIBLE' if after.max_pressure>rho+TOL else 'FEASIBLE','uncovered_tasks':uncovered,'affected_tasks':affected,'fallback_generalists':generalists,'fallback_all_models':fallback,'routing_before':route_dict(before),'routing_after':route_dict(after) if after else None,'z_before':before.max_pressure,'z_after':after.max_pressure if after else None,'delta_z':after.max_pressure-before.max_pressure if after else None,'total_standalone_demand_before':demand(before),'total_standalone_demand_after':demand(after),'demand_multiplier':demand(after)/demand(before) if after else None,'delta_demand':demand(after)-demand(before) if after else None,'affected_task_demand_before':{t:sum(v for (k,m),v in before.task_contributions.items() if k==t) for t in affected},'affected_task_demand_after':{t:sum(v for (k,m),v in after.task_contributions.items() if k==t) for t in affected} if after else None})
        # Freeze-compatible exact trace; candidate marginal z is an audit, not a change to RC-SG.
        trace=actual['trace'];current={m:int(m==primary or m in trace[0]['assignment'].values()) for m in ids};paths=dict(trace[0]['assignment'])
        for step in range(len(trace)):
            chosen=trace[step+1]['model'] if step+1<len(trace) else None;route=solve(current,cell)
            for s in sorted(m for m in catalog if roles[m]=='specialist' and not current[m]):
                improved=[t for t in cell['lambda_rps'] if t in catalog[s].supported_tasks and catalog[s].service_time_ms[t]<catalog[paths[t]].service_time_ms[t]]
                gain=math.fsum(cell['lambda_rps'][t]*(catalog[paths[t]].service_time_ms[t]-catalog[s].service_time_ms[t])/1000 for t in improved);proposal=dict(current);proposal[s]=1;added=solve(proposal,cell);delta=factor*cal.memory_table_mib[s][1]
                greedy.append({'cell_id':cell['cell_id'],'step':step,'base_G':primary,'current_deployment':current.copy(),'current_assignment':paths.copy(),'candidate_S':s,'selected_next':s==chosen,'next_greedy_choice':chosen,'memory_fits':safe(proposal)<=cell['memory_budget_mib'],'demand_saving':gain,'additional_safe_mib':delta,'value_per_safe_mib':gain/delta,'improved_tasks':improved,'z_current':route.max_pressure if route else None,'z_after_candidate':added.max_pressure if added else None,'marginal_z_benefit':route.max_pressure-added.max_pressure if route and added else None,'oracle_contains_candidate':oracle['oracle_deployment'][s]>0,'oracle_final_specialist_set':[m for m in ids if roles.get(m)=='specialist' and oracle['oracle_deployment'][m]],'mandatory_base_specialists':[m for m in set(trace[0]['assignment'].values()) if roles[m]=='specialist'],'terminal_step':chosen is None})
            if chosen:
                current[chosen]=1
                for t in trace[step+1]['improved_tasks']:paths[t]=chosen
        assert current==actual['deployment']
        generalist_states=[s for s in cache['states'] if s['N_S']==0]
        within_budget=[s for s in generalist_states if s['M_safe_mib']<=cell['memory_budget_mib']]
        if within_budget:
            generalist_x=min(within_budget,key=lambda s:(s['z_base'],s['M_safe_mib'],s['vector']))['deployment']
        else:
            # Explicitly memory-infeasible minimum-residency coverage reference.
            generalist_x=min(generalist_states,key=lambda s:(s['M_safe_mib'],s['z_base'],s['vector']))['deployment']
        gc=read(DEV/'generalist_centric_results.json')['cells'];frozen_gc=next(r for r in gc if r['cell_id']==cell['cell_id'])
        gcx=frozen_gc.get('deployment') or cell['preflight']['Generalist_centric'].get('deployment')
        methods=[('Generalist_only_fallback',generalist_x),('Frozen_Generalist_centric',gcx),('RC_SG',rc[cell['cell_id']]['deployment']),('structural_oracle',oracle['oracle_deployment']),('v2',v2[cell['cell_id']]['deployment'])]
        for method,x in methods:
            if x is None:continue
            x={m:x.get(m,0) for m in ids};route=solve(x,cell)
            for m in ids:
                d=sum(v for (k,model),v in route.task_contributions.items() if model==m) if route and m in catalog else None
                cap=cal.capacity_factors[m][x[m]] if m in catalog else 0
                decomposition.append({'cell_id':cell['cell_id'],'intensity':cell['intensity_level'],'alpha':cell['alpha'],'budget_mib':cell['memory_budget_mib'],'method':method,'deployment':x,'model':m,'role':roles.get(m,'specialist'),'replicas':x[m],'standalone_D_seconds_per_second':d,'g_m':cap,'rho_m':d/cap if cap and d is not None else 0 if route else None,'max_pressure':route.max_pressure if route else None,'coverage':route is not None,'memory_feasible':safe(x)<=cell['memory_budget_mib'],'capacity_feasible':bool(route and route.max_pressure<=rho+TOL),'capacity_scale_limit_relative_to_cell':rho/route.max_pressure if route else None,'routing':route_dict(route) if route else None,'task_contributions':{k:v for (k,model),v in route.task_contributions.items() if model==m} if route and m in catalog else {}})
    csvsave('specialist_removal_cost.csv',removal);csvsave('rcsg_greedy_optimality.csv',greedy);csvsave('intensity_capacity_decomposition.csv',decomposition)
    save('local_audit_results.json',{'status':'PASS','GPU_RUNS':0,'primary_reference_G':primary,'fastest_unique_specialists':specialists,'primary_removal_rows':sum(r['primary_reference'] for r in removal),'primary_coverage_infeasible_removals':sum(r['primary_reference'] and r['status']=='COVERAGE_INFEASIBLE' for r in removal),'greedy_candidate_rows':len(greedy),'frozen_RC_SG_reconstruction_exact_all_12':True,'per_cell_trace':{c['cell_id']:c['preflight']['RC_SG']['trace'] for c in cells}})
    print('LOCAL AUDITS PASS',flush=True)

if __name__=='__main__':main()
