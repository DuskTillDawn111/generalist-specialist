"""Counterfactual diagnostics, never formal benchmark input revisions."""
from .common import *
from .audit import frontier
from deployment_core import Workload
from collections import Counter
import multiprocessing as mp
import warnings
import numpy as np
from scipy.optimize import linprog

def select(states,z,G,S,B,c=1):
    valid=np.flatnonzero(G+c*S<=B+TOL)
    if not len(valid):return None
    minimum=z[valid].min();tie=valid[z[valid]<=minimum+TOL]
    return min(tie,key=lambda i:(G[i]+c*S[i],states[i]['vector']))

def memory_audit(cache):
    states=cache['states'];z=np.array([s['z_base'] for s in states]);G=np.array([s['G_safe_mib'] for s in states]);S=np.array([s['S_safe_mib'] for s in states]);rho=cache['rho_max'];cells=read(F/'campaign_cells.json')['cells'];base=cache['base_cell'];intervals=[]
    # Solve every state's affine Pareto membership interval exactly: it must have
    # lower memory than every lower-pressure state and minimum memory among ties.
    for j,s in enumerate(states):
        rivals=np.flatnonzero((z<=z[j]+TOL)&(np.arange(len(z))!=j));db=S[j]-S[rivals];da=G[rivals]-G[j];zero=abs(db)<1e-12
        impossible=np.any(zero & ((da<-TOL)|((abs(da)<=TOL)&(z[rivals]<z[j]-TOL))))
        if impossible:continue
        upper=min(da[db>1e-12]/db[db>1e-12],default=float('inf'));lower=max(1.,max(da[db<-1e-12]/db[db<-1e-12],default=1.))
        if upper<=lower+1e-10:continue
        intervals.append({'state_id':s['state_id'],'deployment':s['deployment'],'N_G':s['N_G'],'N_S':s['N_S'],'c_lower':lower,'c_upper':upper if np.isfinite(upper) else None,'lower_endpoint_semantics':'immediately above; pressure strictness can exclude equality','upper_endpoint_semantics':'immediately below; pressure strictness can exclude equality','G_safe_mib':G[j],'S_safe_mib':S[j],'z_base':z[j]})
    csvsave('counterfactual_pareto_memory_intervals.csv',intervals)
    interval_events=[v for r in intervals for v in [r['c_lower'],r['c_upper']] if v is not None and v>1]
    result=[];pathrows=[];fast=sorted(set(m for m in cache['fastest_S'].values() if m));catalog,roles,cal=inputs();factor=read(A/'memory_safety_factor.json')['safety_factor'];allS=[m for m in catalog if roles[m]=='specialist'];gmin=min(factor*cal.memory_table_mib[m][1] for m in catalog if roles[m]=='generalist')
    for cell in cells:
        B=cell['memory_budget_mib'];scale=cell['alpha']/base['alpha'];exits=np.divide(B-G,S,out=np.full_like(S,np.inf),where=S>0);events=sorted(set([1.]+[float(v) for v in exits if np.isfinite(v) and v>=1]+interval_events));current=select(states,z,G,S,B);initial=current;initial_front={s['state_id'] for s in frontier([s for s in states if s['M_safe_mib']<=B])};initial_composition=Counter((s['N_G'],s['N_S']) for s in states if s['state_id'] in initial_front)
        first_reduction=None;first_sharing=None;first_front=None;previous=current
        for k,event in enumerate(events):
            next_event=events[k+1] if k+1<len(events) else event+max(1.,event)
            probe=(event+next_event)/2
            idx=select(states,z,G,S,B,probe)
            if idx!=previous:
                old=states[previous] if previous is not None else None;new=states[idx] if idx is not None else None
                pathrows.append({'cell_id':cell['cell_id'],'c_M_breakpoint':event,'probe_c_M':probe,'before_state':old['state_id'] if old else None,'after_state':new['state_id'] if new else None,'before_deployment':old['deployment'] if old else None,'after_deployment':new['deployment'] if new else None,'before_N_G':old['N_G'] if old else None,'after_N_G':new['N_G'] if new else None,'before_N_S':old['N_S'] if old else None,'after_N_S':new['N_S'] if new else None,'after_z':new['z_base']*scale if new else None,'capacity_feasible_after':bool(new and new['z_base']*scale<=rho+TOL),'semantics':'candidate residency feasible at equality; after state valid in next open event interval'})
                if new and new['N_S']<states[initial]['N_S'] and first_reduction is None:first_reduction=event
                if new:
                    r=new['routing_base'];shared={m:[t for t in cell['lambda_rps'] if r.get(t+'::'+m,0)>TOL] for m in catalog if roles[m]=='generalist' and new['deployment'][m]}
                    lostS=[t for t in cell['lambda_rps'] if cache['fastest_S'][t] and not any(new['deployment'][m] and roles[m]=='specialist' and t in catalog[m].supported_tasks for m in catalog)]
                    forced_shared=any(len(ts)>=2 and any(t in lostS for t in ts) for ts in shared.values())
                    if forced_shared and first_sharing is None:first_sharing=event
                previous=idx
            if first_front is None:
                active=[r for r in intervals if r['c_lower']<probe and (r['c_upper'] is None or probe<r['c_upper']) and r['G_safe_mib']+probe*r['S_safe_mib']<=B+TOL]
                composition=Counter((r['N_G'],r['N_S']) for r in active)
                if composition!=initial_composition:first_front=event
        def reference(condition,models):
            mem=factor*math.fsum(cal.memory_table_mib[m][1] for m in models);critical=(B-gmin)/mem
            return {'cell_id':cell['cell_id'],'event':condition,'c_M':critical if critical>=1 else 1.,'status':'ALREADY_FALSE_AT_REAL_POINT' if critical<1 else 'LOSS_IMMEDIATELY_ABOVE_BREAKPOINT','raw_residency_critical_c':critical,'real_point_feasible':critical>=1,'definition':'Exists one admitted Generalist resident plus all listed unique one-replica Specialists; memory condition only','specialists':models,'deployment_before':None,'deployment_after':None}
        result.extend([reference('FIRST_LOSS_1G_ALL_FASTEST_SPECIALISTS_RESIDENCY',fast),reference('FIRST_LOSS_1G_ALL_ADMITTED_SPECIALISTS_RESIDENCY',allS)])
        for name,value in [('FIRST_PARETO_COMPOSITION_DISTRIBUTION_CHANGE',first_front),('FIRST_ORACLE_SPECIALIST_COUNT_REDUCTION',first_reduction),('FIRST_FORCED_GENERALIST_MULTI_TASK_SHARING_REGION',first_sharing)]:
            result.append({'cell_id':cell['cell_id'],'event':name,'c_M':value,'status':'OBSERVED' if value is not None else 'NO_EVENT_BEFORE_COVERAGE_LOSS_OR_INFINITY','raw_residency_critical_c':None,'real_point_feasible':None,'definition':'Full event enumeration; Generalist sharing requires >=2 routed tasks including a task lacking every resident admitted Specialist','specialists':None,'deployment_before':None,'deployment_after':None})
    csvsave('counterfactual_memory_breakpoints.csv',result);csvsave('counterfactual_memory_oracle_path.csv',pathrows)
    save('memory_counterfactual_results.json',{'status':'PASS','GPU_RUNS':0,'pure_0G_all_task_specialist_coverage':False,'reason':'VQA has no admitted Specialist','pareto_intervals':len(intervals),'events':result,'oracle_path':pathrows})
    print('MEMORY COUNTERFACTUAL PASS',len(intervals),'intervals',len(pathrows),'oracle changes',flush=True)

MATRICES=[]
def matrices(cache):
    catalog,roles,cal=inputs();base=cache['base_cell']['lambda_rps'];tasks=Workload(base).active_tasks;models=sorted(catalog);result=[];states=[s for s in cache['states'] if s['physical_memory_feasible']]
    for s in states:
        x=s['deployment'];pairs=[(k,m) for k in tasks for m in models if x[m] and k in catalog[m].supported_tasks];zi=len(pairs);eq=np.zeros((len(tasks),zi+1));ub=np.zeros((len(models),zi+1));slope=np.zeros_like(ub)
        for j,(k,m) in enumerate(pairs):
            eq[tasks.index(k),j]=1/base[k]
            actual=catalog[m].service_time_ms[k]/1000
            if roles[m]=='specialist':
                ref=catalog[cache['fastest_G'][k]].service_time_ms[k]/1000
                assert ref>=actual
                ub[models.index(m),j]=ref;slope[models.index(m),j]=actual-ref
            else:ub[models.index(m),j]=actual
        ub[:,zi]=[-cal.capacity_factors[m][x[m]] for m in models]
        result.append((np.r_[np.zeros(zi),1.],eq,ub,slope,[(0.,base[k]) for k,m in pairs]+[(0.,None)]))
    return states,result

def alpha_eval(alpha):
    z=[]
    for obj,eq,ub,slope,bounds in MATRICES:
        r=linprog(obj,A_eq=eq,b_eq=np.ones(eq.shape[0]),A_ub=ub+alpha*slope,b_ub=np.zeros(ub.shape[0]),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':TOL,'dual_feasibility_tolerance':TOL})
        assert r.success and r.x is not None
        z.append(float(r.fun))
    return float(alpha),np.array(z)

def speed_audit(cache):
    global MATRICES
    states,MATRICES=matrices(cache);G=np.array([s['G_safe_mib'] for s in states]);S=np.array([s['S_safe_mib'] for s in states]);cells=read(F/'campaign_cells.json')['cells'];budgets=sorted(set(c['memory_budget_mib'] for c in cells));sample_cache={}
    fname=OUT/'speed_alpha_screen.npz'
    if fname.exists():
        old=np.load(fname);sample_cache={float(a):v for a,v in zip(old['alpha'],old['z'])};print('REUSE CPU SPEED CACHE',len(sample_cache),flush=True)
    else:
        # Independent CPU LP batches; this is not agent delegation or GPU use.
        with mp.get_context('fork').Pool(4) as pool:
            for k,(a,z) in enumerate(pool.imap_unordered(alpha_eval,np.linspace(0,1,257)),1):
                sample_cache[a]=z
                if k%8==0:print('SPEED SCREEN',k,'/257',flush=True)
        alphas=sorted(sample_cache);np.savez_compressed(fname,alpha=alphas,z=np.array([sample_cache[a] for a in alphas]))
    def evaluate(a):
        a=float(a)
        if a not in sample_cache:sample_cache[a]=alpha_eval(a)[1]
        return sample_cache[a]
    # Check exact alpha=1 reduction against canonical frozen LP cache.
    assert np.max(abs(evaluate(1)-np.array([s['z_base'] for s in states])))<1e-8
    rows=[];screen=[];base=cache['base_cell'];rho=cache['rho_max']
    for B in budgets:
        ordered=sorted(np.linspace(0,1,257),reverse=True);previous=select(states,evaluate(1),G,S,B);changes=[]
        for a,b in zip(ordered,ordered[1:]):
            idx=select(states,evaluate(b),G,S,B)
            if idx!=previous:
                lo=float(b);hi=float(a);old=previous
                while hi-lo>1e-8:
                    mid=(hi+lo)/2;im=select(states,evaluate(mid),G,S,B)
                    if im==old:hi=mid
                    else:lo=mid
                after=select(states,evaluate(lo),G,S,B);changes.append((lo,hi,old,after));previous=idx
        for a in ordered:
            z=evaluate(a);idx=select(states,z,G,S,B);s=states[idx]
            screen.append({'B_mib':B,'alpha_speed':float(a),'oracle_state_id':s['state_id'],'oracle_deployment':s['deployment'],'N_G':s['N_G'],'N_S':s['N_S'],'total_replicas':s['total_replicas'],'z_base':float(z[idx]),'M_safe_mib':s['M_safe_mib']})
        matching=[c for c in cells if c['memory_budget_mib']==B]
        for cell in matching:
            scale=cell['alpha']/base['alpha']
            for lo,hi,old,new in changes:
                before=states[old];after=states[new]
                rows.append({'cell_id':cell['cell_id'],'B_mib':B,'alpha_speed_lower':lo,'alpha_speed_upper':hi,'alpha_speed_midpoint':(lo+hi)/2,'remaining_speed_advantage_definition':'alpha multiplies actual difference from fastest admitted Generalist; not a speedup multiplier','before_deployment':before['deployment'],'after_deployment':after['deployment'],'before_N_G':before['N_G'],'after_N_G':after['N_G'],'before_N_S':before['N_S'],'after_N_S':after['N_S'],'before_replicas':before['total_replicas'],'after_replicas':after['total_replicas'],'identity_changed':True,'composition_changed':(before['N_G'],before['N_S'])!=(after['N_G'],after['N_S']),'after_z':float(evaluate(lo)[new]*scale),'after_capacity_feasible':bool(evaluate(lo)[new]*scale<=rho+TOL),'screen_resolution':1/256,'refinement_bracket_width':hi-lo})
            if not changes:
                rows.append({'cell_id':cell['cell_id'],'B_mib':B,'alpha_speed_lower':None,'alpha_speed_upper':None,'alpha_speed_midpoint':None,'remaining_speed_advantage_definition':'alpha multiplies service-time difference','before_deployment':states[previous]['deployment'],'after_deployment':states[previous]['deployment'],'before_N_G':states[previous]['N_G'],'after_N_G':states[previous]['N_G'],'before_N_S':states[previous]['N_S'],'after_N_S':states[previous]['N_S'],'before_replicas':states[previous]['total_replicas'],'after_replicas':states[previous]['total_replicas'],'identity_changed':False,'composition_changed':False,'after_z':None,'after_capacity_feasible':None,'screen_resolution':1/256,'refinement_bracket_width':None})
    csvsave('counterfactual_speed_breakpoints.csv',rows);csvsave('counterfactual_speed_screen.csv',screen)
    # Combined map at observed memory events, their midpoints, speed transition
    # representatives, and endpoint regimes. No official workload/budget is added.
    memory=read(OUT/'memory_counterfactual_results.json')['events'];memory_values=sorted(set([1.]+[r['c_M'] for r in memory if r['c_M'] is not None and r['c_M']>=1]));cs=sorted(set(memory_values+[(a+b)/2 for a,b in zip(memory_values,memory_values[1:])]))
    av=sorted(set([0.,1.]+[r['alpha_speed_midpoint'] for r in rows if r['alpha_speed_midpoint'] is not None]));av=sorted(set(av+[(a+b)/2 for a,b in zip(av,av[1:])]))
    maps=[]
    for B in (budgets[0],budgets[3],budgets[-1]):
        cell=next(c for c in cells if c['memory_budget_mib']==B and c['intensity_level']=='I3');scale=cell['alpha']/base['alpha']
        for a in av:
            z=evaluate(a)
            for c in cs:
                idx=select(states,z,G,S,B,c)
                s=states[idx] if idx is not None else None
                maps.append({'cell_id':cell['cell_id'],'B_mib':B,'c_M':c,'alpha_speed':a,'oracle_state_id':s['state_id'] if s else None,'deployment':s['deployment'] if s else None,'N_G':s['N_G'] if s else None,'N_S':s['N_S'] if s else None,'regime':counts(s['deployment'],{m:('generalist' if m in ['phi4mm','qwen','minicpm'] else 'specialist') for m in s['deployment']})[2] if s else 'COVERAGE_INFEASIBLE','z_star':float(z[idx]*scale) if s else None,'capacity_feasible':bool(s and z[idx]*scale<=rho+TOL),'counterfactual_only':True})
    csvsave('counterfactual_tradeoff_map.csv',maps)
    alphas=sorted(sample_cache);np.savez_compressed(OUT/'speed_alpha_all_evaluations.npz',alpha=alphas,z=np.array([sample_cache[a] for a in alphas]))
    save('speed_counterfactual_results.json',{'status':'PASS','GPU_RUNS':0,'screen_points':257,'alpha_evaluations_total':len(sample_cache),'LP_states_per_alpha':len(states),'alpha_1_exact_frozen_reduction_verified':True,'transition_bracket_max_width':1e-8,'sub_grid_nonmonotone_islands_certified_absent':False,'limitation':'Dense screen + bisection certifies brackets of observed transitions; no claim of exhaustive continuous parametric LP enumeration.','transitions':rows,'combined_map_cells':len(maps)})
    print('SPEED COUNTERFACTUAL PASS',len(rows),'transitions',len(maps),'map cells',flush=True)

if __name__=='__main__':
    from collections import Counter
    cache=read(OUT/'lp_states_cache.json')
    memory_audit(cache)
    speed_audit(cache)
