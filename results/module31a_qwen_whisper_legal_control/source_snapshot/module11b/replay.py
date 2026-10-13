"""Same frozen hotspot sets, directions, budgets, empirical-count MILP."""
from collections import defaultdict
from .common import *
from .analyze_replica import calibration,minimum_cover_memory
from .envelope_numerical import solve,fixed_lp
from module11.common import rates

def closure(c,roles,cal,scaling,bg,B):
    rows=[];terminal=scaling.get('terminal_frontiers',{})
    initial={r['model'] for r in read(OUT/'measurement_plan.json')['required_initial_frontier']}
    for m in sorted(initial):
        if roles[m]!='specialist':continue
        if m in terminal:
            rows.append({'model':m,'closed':True,'reason':terminal[m]['reason'],'terminal_x':terminal[m]['x']})
        else:
            rows.append({'model':m,'closed':False,'reason':'NEXT_BREAKPOINT_RELEVANT_COUNT_NOT_YET_EMPIRICALLY_TERMINATED','first_missing_count':cal.evidence_bounds[m]+1})
    # Only the next count in the witnessed frontier is relevant; no extrapolated g/M.
    return {'specialist_evidence_closed':all(r['closed'] for r in rows),'models':rows,
        'higher_counts_after_true_terminal_not_artificial':True,
        'Generalist_count_scope':'1G/2G states use at most two total G; S-only backgrounds need no third Phi because all admitted background demand is <.8*g_Phi(1); unmeasured M/G points never extrapolated'}

def capacity(stage='final'):
    c,hw,roles,base=inputs();cal=calibration();freeze=read(OLD/'input_freeze.json');directions=read(OLD/'normalized_directions.json')
    bg=freeze['background'];scaling=read(OUT/'replica_scaling.json');allfits=[];windows=[]
    for i,context in enumerate(freeze['contexts'],1):
        bid=f'B{i}';B=context['memory_budget_mib'];closed=closure(c,roles,cal,scaling,bg,B)
        for tasks in freeze['sets']:
            sid='+'.join(tasks);heads={k:directions['heads_by_budget'][bid][k] for k in tasks};fits={}
            for mode in ['S','1G','2G']:
                f=solve(c,roles,cal,B,bg,heads,mode,factor());fits[mode]=f
                allfits.append({'budget':bid,'set_id':sid,'tasks':tasks,'mode':mode,'headroom':heads,
                    'memory_budget_mib':B,'replica_certificate':closed['specialist_evidence_closed'],**f})
            s,a,b=[fits[m]['alpha'] for m in ['S','1G','2G']]
            w1=a-s if a is not None and s is not None else None;w2=b-a if b is not None and a is not None else None
            if w1 is not None and abs(w1)<1e-9:w1=0.
            if w2 is not None and abs(w2)<1e-9:w2=0.
            assert w2 is None or w2>=-1e-7
            mid=(a+b)/2 if w2 and w2>0 else None
            one= fixed_lp(fits['1G']['deployment'],bg,heads,c,roles,cal,'1G',mid) if mid else None
            two= fixed_lp(fits['2G']['deployment'],bg,heads,c,roles,cal,'2G',mid) if mid else None
            one_global_upper=fits['1G'].get('milp_alpha_upper_bound')
            if one_global_upper is not None:
                one_global_upper*=1.+(fits['1G']['stages'][0].get('gap') or 0.)
            midpoint_certified=bool(mid is not None and one_global_upper is not None and mid>one_global_upper and two is not None)
            windows.append({'budget':bid,'set_id':sid,'tasks':tasks,'alpha_S':s,'alpha_1G':a,'alpha_2G':b,
                'W_1G':w1,'W_2G':w2,'R_1G':w1/s if s and w1 is not None else None,'R_2G':w2/a if a and w2 is not None else None,
                'replica_certificate':closed['specialist_evidence_closed'],'evidence_closure':closed,
                'SECOND_GENERALIST_CAPACITY_NECESSARY':bool(closed['specialist_evidence_closed'] and w2 and w2>0 and midpoint_certified),
                'oneG_global_MILP_upper_bound_alpha':one_global_upper,'midpoint_capacity_certificate_passed':midpoint_certified,
                'midpoint_alpha':mid,'oneG_maximizing_state_at_midpoint_feasible':one is not None if mid else None,
                'twoG_midpoint_feasible':two is not None if mid else None,
                'all_1G_infeasible_certificate':'global MILP alpha_1G upper bound < midpoint' if mid else None,
                'count_scope':'all historical/new empirical prefixes only; higher true-terminal counts closed by user-authorized measured stopping rule'})
        print(json.dumps({'stage':stage,'budget':bid,'capacity_rows':len(allfits),'evidence_closed':closed['specialist_evidence_closed']}),flush=True)
    if stage=='final':
        save('updated_capacity_witnesses.json',{'rows':allfits,'calibration':ref(OUT/'replica_scaling.json'),'directions':ref(OLD/'normalized_directions.json')})
        csvsave('updated_multihotspot_capacity_envelopes.csv',allfits);csvsave('updated_second_generalist_regions.csv',windows)
        save('updated_windows.json',{'rows':windows})
    else:
        save(f'frontier_replay/{stage}.json',{'rows':allfits,'windows':windows,'calibration':ref(OUT/'replica_scaling.json')})
    return allfits,windows

def latency():
    from .convex_audit import solve_frozen_J
    c,hw,roles,base=inputs();cal=calibration();freeze=read(OLD/'input_freeze.json');direction=read(OLD/'normalized_directions.json');bg=freeze['background']
    wins=read(OUT/'updated_windows.json')['rows'];rows=[]
    for w in wins:
        s,a=w['alpha_S'],w['alpha_1G'];W=w['W_1G']
        if s is None or a is None:continue
        points=sorted(set(v for v in [.8*s,.98*s,s+.3*W,s+.7*W,.8*a,.98*a] if v>=0 and v<a-1e-9))
        bid=w['budget'];B=freeze['contexts'][int(bid[1:])-1]['memory_budget_mib'];heads={k:direction['heads_by_budget'][bid][k] for k in w['tasks']}
        for alpha in points:
            lam=rates(bg,heads,alpha);results={}
            for count in [1,2]:
                key=digest({'lambda':lam,'B':B,'NG':count,'calibration':ref(OUT/'replica_scaling.json'),'scope':'unchanged_frozen_router_exact_bound_pruning'})
                p=OUT/'J_reference_cache'/f'{key}.json'
                if p.exists():r=read(p)
                else:
                    try:r=solve_frozen_J(lam,c,roles,cal,B,count)
                    except (AssertionError,ValueError) as exc:r={'status':'NUMERICAL_REFERENCE_UNRESOLVED','J_seconds':None,'error':repr(exc)}
                    save_json(p,r)
                results[count]=r
            j1,j2=[results[n].get('mathematical_reference',{}).get('J_seconds') for n in [1,2]]
            f1,f2=[results[n].get('best',{}).get('J_seconds') if results[n].get('best') else None for n in [1,2]]
            rows.append({'budget':bid,'set_id':w['set_id'],'alpha':alpha,'lambda':lam,'J_1G_reference_seconds':j1,'J_2G_reference_seconds':j2,
                'DeltaJ_reference_seconds':j1-j2 if j1 is not None and j2 is not None else None,
                'J_1G_frozen_seconds':f1,'J_2G_frozen_seconds':f2,
                'DeltaJ_frozen_seconds':f1-f2 if f1 is not None and f2 is not None else None,
                'reference_1G':results[1],'reference_2G':results[2],
                'reference_scope':'same mathematical J objective and rho<=.8 convex certificate; CPU design audit only',
                'frozen_Module9_1G':results[1]['best']['frozen_solver'] if results[1].get('best') else None,
                'frozen_Module9_2G':results[2]['best']['frozen_solver'] if results[2].get('best') else None,
                'no_frozen_router_repair':True,'replica_certificate':w['replica_certificate']})
            save('updated_latency_witnesses.json',{'rows':rows});csvsave('updated_oneG_vs_twoG_j.csv',rows)
        print(json.dumps({'J_context':bid+'/'+w['set_id'],'comparisons':len(rows)}),flush=True)
    save('updated_latency_status.json',{'status':'COMPLETE','comparisons':len(rows),'formal_hotspot_GPU_runs':0})

def gate():
    windows=read(OUT/'updated_windows.json')['rows'];closed=all(w['replica_certificate'] for w in windows)
    positives=[w for w in windows if w['SECOND_GENERALIST_CAPACITY_NECESSARY']]
    if positives:g='MULTIHOTSPOT_2G_EVIDENCE_SUPPORTED'
    elif not closed:g='UNDETERMINED_REPLICA_EVIDENCE_REQUIRED'
    elif any(w['W_1G'] and w['W_1G']>0 for w in windows):g='MULTIHOTSPOT_1G_ONLY_CONFIRMED'
    else:g='MULTIHOTSPOT_NO_EXTENSION'
    primary=min(positives,key=lambda w:(-w['R_2G'],-w['W_2G'],int(w['budget'][1:]),tuple(w['tasks']))) if positives else None
    save('replica_evidence_gate.json',{'gate':g,'closed':closed,'positive_2G_certified_contexts':len(positives),'primary_candidate':primary,
         'formal_GPU_hotspot_not_started':True,'Module9_unchanged':True,'historical_replica_points_unchanged':True,
         'scope':'planning evidence with empirical representative-task family scalar; mixed-runtime stability still requires future authorized validation'})
    return g
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['capacity','latency','gate']);a=p.parse_args()
    {'capacity':capacity,'latency':latency,'gate':gate}[a.stage]()
