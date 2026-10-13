"""First frozen brackets, permitted global refinement, and transition witnesses."""
from .common import *
from .run import solve_query
from .solver import StructuralSolver
from .metrics import events
from module11b.convex_audit import reference_route
from deployment_core.routing.e2e_surrogate import evaluate_surrogate


def first_axis(rows, axis):
    other = 'mu' if axis == 'gamma' else 'gamma'
    rr = sorted([r for r in rows if r[other] == 1], key=lambda r:r[axis])
    base = rr[0]; result = {'baseline_N_G':base['N_G'], 'baseline_N_S':base['N_S'], 'events':{}}
    for event in ['ANY','T1','T2','T3','T4','T5','T6']:
        hit = next((i for i,r in enumerate(rr) if i and (r['events'] if event=='ANY' else event in r['events'])),None)
        if hit is None:
            status = 'ALREADY_PRESENT_AT_BASELINE' if event=='T3' and base['N_G']>=2 else 'NOT_OBSERVED_IN_FROZEN_GRID'
            result['events'][event]={'status':status,'search_upper':rr[-1][axis], 'infeasible_points':[r[axis] for r in rr if not r['feasible']]}
        else:
            result['events'][event]={'status':'COARSE_BRACKET','low':rr[hit-1][axis], 'high':rr[hit][axis], 'high_events':rr[hit]['events']}
    return result


def witness(row, base, workload, budget, family, c, roles, cal):
    name=query_key(workload['id'],budget['id'],row['gamma'],row['mu'],family)
    path=OUT/'transition_witnesses'/f'{name}.json'
    if path.exists(): return read(path)
    cc,ac=scaled(c,roles,cal,row['gamma'],row['mu'],family)
    x=base['deployment']; memory=factor()*math.fsum(ac.memory_table_mib[m][n] for m,n in x.items())
    out={'workload':workload['id'],'budget':budget['id'],'gamma':row['gamma'],'mu':row['mu'],'family':family}
    if memory>budget['safe_memory_budget_mib']+1e-6:
        out.update(status='BASELINE_DEPLOYMENT_MEMORY_INFEASIBLE',structural_support=True)
    else:
        local=reference_route(x,{k:v for k,v in workload['rates'].items() if v>0},cc,ac)
        if local is None:
            # Independent LP distinguishes capacity infeasibility from numerical routing failure.
            from deployment_search.routing import solve_canonical_routing
            from deployment_core import Workload
            canon=solve_canonical_routing(x,Workload({k:v for k,v in workload['rates'].items() if v>0}),cc,replica_bounds=ac.evidence_bounds,capacity_mode='empirical',calibration=ac)
            genuine=canon is None or canon.z > .8 + 1e-8
            out.update(status='BASELINE_DEPLOYMENT_CAPACITY_INFEASIBLE' if genuine else 'NUMERICAL_REFERENCE_UNRESOLVED',structural_support=genuine)
        else:
            lower=local['evaluation']['J_seconds']-(local['gap_seconds'] or 0.)
            guard=64*math.ulp(max(abs(lower),abs(row['J_star_seconds'])))
            supported=local['gap_seconds'] is not None and lower-row['J_star_seconds']>guard
            out.update(status='STRICT_BASELINE_DEPLOYMENT_OBJECTIVE_LOSS' if supported else 'NUMERICAL_TIE_OR_UNRESOLVED',structural_support=supported,
                baseline_deployment_J_lower=lower,new_deployment_J_upper=row['J_star_seconds'],baseline_local_gap_seconds=local['gap_seconds'],roundoff_guard_seconds=guard)
    # Same deployment routing transitions have a separate certificate.
    route={(p.split('::')[0],p.split('::')[1]):v for p,v in base['routing'].items()}
    try:
        ev=evaluate_surrogate(x,route,workload['rates'],cc,ac)
        out['baseline_route_J_seconds']=ev['J_seconds']
        out['route_support']=ev['z']>.8+1e-8 or ev['J_seconds']-row['J_star_seconds']>64*math.ulp(ev['J_seconds'])
    except ValueError:
        out['route_support']=True
    save_json(path,out);return out


def main():
    plan=read(OUT/'experiment_freeze.json');c,hw,roles,cal=baseline();ctx=[];refinements=[]
    for w in plan['workloads']:
        for b in plan['budgets']:
            rows=read(OUT/f"global_contexts/{w['id']}_{b['id']}.json")['rows'];base=rows[0];solver=StructuralSolver()
            item={'workload':w['id'],'budget':b['id'],'baseline':base,'axes':{}}
            for axis in ['gamma','mu']:
                info=first_axis(rows,axis);bracket=info['events']['ANY']
                if bracket['status']=='COARSE_BRACKET':
                    lo,hi=bracket['low'],bracket['high'];initial=[lo,hi];count=0
                    while (hi-lo)/hi>.05 and count<6:
                        mid=(lo+hi)/2;g,m=(mid,1.) if axis=='gamma' else (1.,mid)
                        row=solve_query(w,b,g,m,None,solver,c,roles,cal)['row'];row={**row,**events(base,row,w,c)}
                        refinements.append({'workload':w['id'],'budget':b['id'],'axis':axis,'iteration':count+1,'initial_bracket':initial,'low_before':lo,'high_before':hi,'mid':mid,**row})
                        if row['events']:hi=mid
                        elif row['feasible']:lo=mid
                        else:raise AssertionError('First structural bracket refinement encountered infeasibility; do not invent a transition')
                        count+=1
                    g,m=(hi,1.) if axis=='gamma' else (1.,hi)
                    high=solve_query(w,b,g,m,None,solver,c,roles,cal)['row'];high={**high,**events(base,high,w,c)}
                    info['first_any_refined']={'low':lo,'high':hi,'relative_width':(hi-lo)/hi,'iterations':count,
                        'high_events':high['events'],'high_N_G':high['N_G'],'high_N_S':high['N_S'],'witness':witness(high,base,w,b,None,c,roles,cal)}
                item['axes'][axis]=info
            pts=[r for r in rows if r['events']]
            item['two_dimensional']={}
            for e in ['ANY','T1','T2','T3','T4','T5','T6']:
                points=[r for r in pts if e=='ANY' or e in r['events']]
                minimal=[r for r in points if not any(s['gamma']<=r['gamma'] and s['mu']<=r['mu'] and (s['gamma'],s['mu'])!=(r['gamma'],r['mu']) for s in points)]
                item['two_dimensional'][e]=[{'gamma':r['gamma'],'mu':r['mu'],'events':r['events'],'N_G':r['N_G'],'N_S':r['N_S'],'witness':witness(r,base,w,b,None,c,roles,cal)} for r in minimal]
            item['nonmonotone_reentries']=[{'gamma':r['gamma'],'mu':r['mu']} for r in rows if r['feasible'] and not r['events'] and any(s['gamma']<=r['gamma'] and s['mu']<=r['mu'] for s in pts)]
            ctx.append(item)
            print(w['id'],b['id'],'refined',flush=True)
    csvsave('transition_refinement.csv',refinements)
    save('transition_frontiers.json',{'kind':'COUNTERFACTUAL_CPU_ONLY','contexts':ctx,'two_dimensional_boundary':'Pareto-minimal observed coarse event points; nonmonotone reentries retained, not an assumed monotone threshold','refinement_scope':'global first ANY event only, no per-task refinement'})

if __name__=='__main__':main()
