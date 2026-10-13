from .common import *
from .envelope import solve_envelope

def bound_audit(fit,target,catalog,roles,cal,budget):
    if fit['status']!='OPTIMAL':return {'status':'INFEASIBLE','REPLICA_BOUND_ARTIFICIAL':None,'models':[]}
    rows=[]
    for m in sorted(catalog):
        if roles[m]!='specialist' or target not in catalog[m].supported_tasks or not fit['deployment'][m]:continue
        n=fit['deployment'][m];bound=cal.evidence_bounds[m];at_bound=n==bound
        # Missing-count residency screen only. It does not add a memory or g
        # table entry; any actually used new count requires real profiling.
        next_memory=cal.memory_table_mib[m].get(n+1,cal.memory_table_mib[m][n]+catalog[m].memory_mib)
        next_safe=fit['safe_memory_mib']+factor()*(next_memory-cal.memory_table_mib[m][n])
        missing=n+1 not in cal.capacity_factors[m]
        artificial=at_bound and missing and next_safe<=budget+1e-6
        rows.append({'model':m,'replicas':n,'evidence_bound':bound,'at_evidence_bound':at_bound,
                     'next_count':n+1,'next_count_g_missing':missing,'next_count_memory_measured':n+1 in cal.memory_table_mib[m],
                     'next_count_memory_screen_mib':next_safe,'next_count_memory_screen_fits':next_safe<=budget+1e-6,
                     'memory_screen':'existing measured prefix + frozen one-replica idle memory; missing count stays unmeasured',
                     'REPLICA_BOUND_ARTIFICIAL':artificial,'target_pressure':fit['rho'][m]})
    artificial=any(r['REPLICA_BOUND_ARTIFICIAL'] for r in rows)
    return {'status':'REPLICA_EVIDENCE_REQUIRED' if artificial else 'PASS','REPLICA_BOUND_ARTIFICIAL':artificial,'models':rows}

def main():
    assert not (OUT/'primary_hotspot_selection.json').exists(),'Primary selection already frozen; do not redesign'
    c,h,roles,cal=inputs();refs=protected();verify(refs)
    selected9=read(ROOT/'results/module9bc_e2e_routing_ranking/v3_minimal_frozen.json')
    assert selected9['routing']=='J_ROUTING' and selected9['candidate_ranking']=='DELTA_J'
    tasks=sorted(set(k for m in c for k in c[m].supported_tasks))
    overlap={k:{role:sorted(m for m in c if roles[m]==role and k in c[m].supported_tasks) for role in ['specialist','generalist']} for k in tasks}
    eligible={k:v for k,v in overlap.items() if v['specialist'] and v['generalist']}
    if (OUT/'input_freeze.json').exists():verify(read(OUT/'input_freeze.json')['protected_references'])
    else:save('input_freeze.json',{'at_utc':now(),'module':'10A','protected_references':refs,'eligible_tasks':eligible,'excluded_tasks':{k:v for k,v in overlap.items() if k not in eligible},
                           'contexts':contexts(),'capacity_ceiling':'D <= frozen empirical g(x); methods retain rho_max=0.8','no_new_GPU_before_this_freeze':True,
                           'selection_rule':'among legally audited W>0 contexts: prioritize C_ALL maximizing witness with N_G>=2, then R, W, lower H3 target rate, lower B, task ID, background/context ID',
                           'replica_extension_rule':'Only one mechanically chosen target task; no missing-count g extrapolation; profile only if needed for its final audit; no Module9 edits'})
    rows=[];audits=[];windows=[]
    for context in contexts():
        for k in eligible:
            print(json.dumps({'event':'envelope_start','context':context['id'],'target':k}),flush=True)
            fits={}
            for only in [True,False]:
                fit=solve_envelope(c,roles,cal,context['memory_budget_mib'],context['background_lambda'],k,only,factor());fits[only]=fit
                rows.append({'context':context['id'],'target':k,'mode':'S_ONLY' if only else 'ALL',**context,**fit})
            audit=bound_audit(fits[True],k,c,roles,cal,context['memory_budget_mib']);audits.append({'context':context['id'],'target':k,**audit})
            cs,ca=fits[True]['capacity_rps'],fits[False]['capacity_rps'];W=ca-cs if cs is not None and ca is not None else None
            if W is not None and abs(W)<1e-7:W=0.
            windows.append({'context':context['id'],'target':k,'background':context['background'],'memory_budget_mib':context['memory_budget_mib'],
                            'C_Sonly':cs,'C_ALL':ca,'W':W,'R':W/cs if cs and W is not None else None,'C_Sonly_audit':audit['status'],
                            'REPLICA_BOUND_ARTIFICIAL':audit['REPLICA_BOUND_ARTIFICIAL'],'C_ALL_N_G':fits[False].get('N_G'),
                            'C_ALL_G_types':fits[False].get('G_types'),'C_ALL_target_G_fraction':fits[False].get('target_G_fraction'),
                            'H3_target_rate':cs+.7*W if W is not None else None,
                            'envelope_scope':'evidence-supported count space; provisional if replica audit does not PASS'})
            csvsave('capacity_envelope.csv',rows);csvsave('overflow_windows.csv',windows)
            save('capacity_envelope.json',{'rows':rows});save('replica_capacity_audit.json',{'contexts':audits,'new_replica_profiles':0})
    save('cpu_audit_complete.json',{'status':'CPU_AUDIT_COMPLETE','at_utc':now(),'rows':len(rows),'contexts':len(windows),'new_GPU_runs':0})
    print(json.dumps({'CPU_AUDIT_COMPLETE':True,'positive_windows':sum(r['W'] is not None and r['W']>0 for r in windows),
                      'legal_positive_windows':sum(r['W'] is not None and r['W']>0 and r['C_Sonly_audit']=='PASS' for r in windows)}),flush=True)
if __name__=='__main__':main()
