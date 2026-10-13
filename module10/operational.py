"""CPU-only operational envelope audit; never launches or authorizes GPU work."""
import csv
import itertools
import json
import math
from .common import ROOT, OUT, inputs, read, ref, now, factor, verify
from .envelope import solve_envelope
from .audit import bound_audit
from .select import missing_count_screen
from .planning import fixed_capacity

ARCHIVE = OUT / 'capacity_definition_mismatch_audit'
AUDIT = OUT / 'operational_envelope_audit'
RHO_MAX = .8

def save(name, value):
    AUDIT.mkdir(parents=True, exist_ok=True)
    (AUDIT/name).write_text(json.dumps(value, indent=2, sort_keys=True))

def csvsave(name, rows):
    columns = list(dict.fromkeys(k for r in rows for k in r))
    with (AUDIT/name).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns); w.writeheader()
        for r in rows:
            w.writerow({k:json.dumps(v, sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in r.items()})

def check_archive():
    manifest = read(ARCHIVE/'archive_manifest.json')
    for r in manifest['original_files']:
        assert ref(ARCHIVE/r['relative_path'])['sha256'] == r['sha256'], r
    return len(manifest['original_files'])

def main():
    assert not (AUDIT/'primary_hotspot_selection.json').exists(), 'Operational primary already frozen'
    freeze=read(ARCHIVE/'input_freeze.json'); c,h,roles,cal=inputs()
    verify(freeze['protected_references']); check_archive()
    save('input_freeze.json', {'at_utc':now(), 'capacity_ceiling_rho':RHO_MAX,
         'contexts':freeze['contexts'], 'eligible_tasks':freeze['eligible_tasks'],
         'selection_rule':freeze['selection_rule'], 'old_input_freeze':ref(ARCHIVE/'input_freeze.json'),
         'protected_references':freeze['protected_references'], 'new_GPU_runs':0,
         'Module9_modified':False, 'new_replica_profiles':0})
    rows=[]; windows=[]; audits=[]; fits={}
    for context in freeze['contexts']:
        for k in sorted(freeze['eligible_tasks']):
            pair={}
            for only in [True,False]:
                f=solve_envelope(c,roles,cal,context['memory_budget_mib'],context['background_lambda'],k,only,factor(),capacity_ceiling=RHO_MAX)
                pair[only]=f; fits[context['id'],k,only]=f
            a=bound_audit(pair[True],k,c,roles,cal,context['memory_budget_mib'])
            screen=missing_count_screen(c,roles,cal,k,context['background_lambda'],context['memory_budget_mib'])
            a['local_maximizer_REPLICA_BOUND_ARTIFICIAL']=a['REPLICA_BOUND_ARTIFICIAL']
            a['global_missing_count_memory_screen']=screen
            possible=any(s['missing_count_can_fit_optimistic_background_cover'] for s in screen)
            if a['status']!='INFEASIBLE':
                a['REPLICA_BOUND_ARTIFICIAL']=bool(a['REPLICA_BOUND_ARTIFICIAL'] or possible)
                a['status']='REPLICA_EVIDENCE_REQUIRED' if a['REPLICA_BOUND_ARTIFICIAL'] else 'PASS'
            audits.append({'context':context['id'],'target':k,**a})
            for only,f in pair.items():
                rows.append({**context,'context':context['id'],'target':k,'mode':'S_ONLY' if only else 'ALL',**f,
                             'replica_audit_status':a['status'],'envelope_certified':a['status']=='PASS',
                             'certified_capacity_rps':f['capacity_rps'] if a['status']=='PASS' else None})
            cs,ca=pair[True]['capacity_rps'],pair[False]['capacity_rps']
            W=ca-cs if cs is not None and ca is not None else None
            if W is not None and abs(W)<1e-7:W=0.
            assert W is None or W>=-1e-7
            windows.append({'context':context['id'],'target':k,'background':context['background'],
                'memory_budget_mib':context['memory_budget_mib'],'C_S_op':cs,'C_A_op':ca,'W_op':W,
                'R_op':W/cs if cs and W is not None else None,'C_Sonly_audit':a['status'],
                'REPLICA_BOUND_ARTIFICIAL':a['REPLICA_BOUND_ARTIFICIAL'],
                'C_ALL_N_G':pair[False].get('N_G'),'H3_target_rate':cs+.7*W if W is not None else None,
                'scope':'evidence-supported counts; uncertified bounds are not exhaustion evidence'})
            save('capacity_envelope_operational.json',{'rows':rows})
            csvsave('capacity_envelope_operational.csv',rows); csvsave('overflow_windows_operational.csv',windows)
            save('replica_capacity_audit.json',{'contexts':audits,'new_replica_profiles':0})
            print(json.dumps({'event':'operational_envelope','context':context['id'],'target':k,'C_S_op':cs,'C_A_op':ca,'audit':a['status']}),flush=True)
    legal=[w for w in windows if w['W_op'] is not None and w['W_op']>1e-7 and w['C_Sonly_audit']=='PASS' and w['C_S_op']>0]
    multi=[w for w in legal if w['C_ALL_N_G']>=2]
    if not legal:
        save('primary_hotspot_selection.json',{'status':'NO_CERTIFIED_OPERATIONAL_PRIMARY','new_GPU_runs':0})
        return
    s=min(multi or legal,key=lambda r:(-r['R_op'],-r['W_op'],r['H3_target_rate'],r['memory_budget_mib'],r['target'],r['background'],r['context']))
    con=next(t for t in freeze['contexts'] if t['id']==s['context']); k=s['target']
    old=read(ARCHIVE/'primary_hotspot_selection.json')
    primary={'status':'OPERATIONAL_PRIMARY_FROZEN','at_utc':now(),**s,'context':con,
        'specialist_only_witness':fits[con['id'],k,True], 'all_model_witness':fits[con['id'],k,False],
        'primary_unchanged':old['context']['id']==con['id'] and old['target']==k,
        'old_primary':{'context':old['context']['id'],'target':old['target']},
        'certified_positive_contexts':legal,'certified_multi_G_context_exists':bool(multi),
        'selection_rule':freeze['selection_rule'],'capacity_ceiling_rho':RHO_MAX,
        'new_GPU_results_used_for_selection':False,'new_GPU_runs':0,'requires_user_confirmation_before_new_GPU':True}
    save('primary_hotspot_selection.json',primary)
    cs,ca,W=s['C_S_op'],s['C_A_op'],s['W_op']; points=[]
    for name,rate in [('H0',.8*cs),('H1',.98*cs),('H2',cs+.3*W),('H3',cs+.7*W),('H4',1.02*ca)]:
        lam=dict(con['background_lambda']);lam[k]=rate
        points.append({'point':name,'target_rate':rate,'lambda_rps':lam,'formal_GPU_point':name!='H4',
                       'S_only_operational_feasible':rate<=cs,'ALL_operational_feasible':rate<=ca})
    save('hotspot_points.json',{'frozen':True,'at_utc':now(),'context':con,'target':k,'C_S_op':cs,'C_A_op':ca,
        'W_op':W,'capacity_ceiling_rho':RHO_MAX,'points':points,'movable_after_GPU':False,
        'new_GPU_runs_authorized':False,'requires_user_confirmation':True})
    models=sorted(c); maxima={True:None,False:None}; feasible=0
    for counts in itertools.product(*(range(cal.evidence_bounds[m]+1) for m in models)):
        x=dict(zip(models,counts))
        if factor()*sum(cal.memory_table_mib[m][n] for m,n in x.items())>con['memory_budget_mib']+1e-6:continue
        if any(not any(x[m] and t in c[m].supported_tasks for m in models) for t,v in con['background_lambda'].items() if v>0):continue
        feasible+=1
        for only in maxima:
            v=fixed_capacity(x,con['background_lambda'],k,c,roles,cal,only,RHO_MAX)
            if v is not None:maxima[only]=max(maxima[only] or 0,v)
    assert abs(maxima[True]-cs)<1e-7 and abs(maxima[False]-ca)<1e-7
    for only in maxima:
        repeat=solve_envelope(c,roles,cal,con['memory_budget_mib'],con['background_lambda'],k,only,factor(),capacity_ceiling=RHO_MAX)
        assert repeat['deployment']==fits[con['id'],k,only]['deployment']
        assert abs(repeat['capacity_rps']-fits[con['id'],k,only]['capacity_rps'])<1e-7
    save('cpu_validation.json',{'status':'PASS','envelopes':len(rows),'contexts':len(windows),
        'independent_enumeration_count_space':math.prod(cal.evidence_bounds[m]+1 for m in models),
        'primary_memory_coverage_feasible_deployments':feasible,'enumeration_C_S_op':maxima[True],
        'enumeration_C_A_op':maxima[False],'repeat_MILP_deployment_and_capacity_match':True,
        'archive_hashes_verified':check_archive(),'new_GPU_runs':0})
    print(json.dumps({'OPERATIONAL_PRIMARY':con['id'],'target':k,'C_S_op':cs,'C_A_op':ca,'W_op':W,'unchanged':primary['primary_unchanged'],'rates':{p['point']:p['target_rate'] for p in points}}),flush=True)

if __name__=='__main__':main()
