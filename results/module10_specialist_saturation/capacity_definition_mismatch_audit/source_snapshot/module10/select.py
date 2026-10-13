"""Certify replica bounds, freeze the one primary context and fixed rates."""
import itertools
from .common import *

def missing_count_screen(catalog,roles,cal,target,background,budget):
    required={k for k,v in background.items() if k!=target and v>0};models=sorted(catalog);records=[]
    for m in models:
        if roles[m]!='specialist' or target not in catalog[m].supported_tasks:continue
        bound=cal.evidence_bounds[m];next_memory=cal.memory_table_mib[m][bound]+catalog[m].memory_mib
        others=[n for n in models if n!=m];best=None
        for bits in itertools.product([0,1],repeat=len(others)):
            active=[n for n,b in zip(others,bits) if b]
            covered=set(catalog[m].supported_tasks)
            for n in active:covered.update(catalog[n].supported_tasks)
            if not required<=covered:continue
            memory=factor()*(next_memory+sum(catalog[n].memory_mib for n in active))
            key=(memory,len(active),tuple(active))
            if best is None or key<best[0]:best=(key,active)
        records.append({'model':m,'first_missing_count':bound+1,'next_memory_is_unmeasured':True,
                        'minimum_background_coverage_safe_memory_screen_mib':best[0][0],
                        'optimistic_background_cover':best[1],
                        'missing_count_can_fit_optimistic_background_cover':best[0][0]<=budget+1e-6,
                        'screen_scope':'minimum admitted capability cover, ignoring capacity; all legal identities. Existing prefix residency plus unchanged one-replica residency only; no g or memory table is extended.'})
    return records

def main():
    assert not (OUT/'primary_hotspot_selection.json').exists(),'Primary already frozen'
    c,h,roles,cal=inputs();freeze=read(OUT/'input_freeze.json');verify(freeze['protected_references'])
    rows=read(OUT/'capacity_envelope.json')['rows'];ctx={s['id']:s for s in contexts()};fits={(r['context'],r['target'],r['mode']):r for r in rows}
    audit=read(OUT/'replica_capacity_audit.json');windows=list(__import__('csv').DictReader((OUT/'overflow_windows.csv').open()))
    for a in audit['contexts']:
        con=ctx[a['context']]
        screen=missing_count_screen(c,roles,cal,a['target'],con['background_lambda'],con['memory_budget_mib'])
        a['global_missing_count_memory_screen']=screen
        possible=any(r['missing_count_can_fit_optimistic_background_cover'] for r in screen)
        a['local_maximizer_REPLICA_BOUND_ARTIFICIAL']=a['REPLICA_BOUND_ARTIFICIAL']
        a['REPLICA_BOUND_ARTIFICIAL']=bool(a['REPLICA_BOUND_ARTIFICIAL'] or possible)
        a['status']='REPLICA_EVIDENCE_REQUIRED' if a['REPLICA_BOUND_ARTIFICIAL'] else 'PASS'
        a['additional_check']='A full maximizing state is insufficient to rule out a different deployment reallocating memory to an unprofiled Specialist count.'
    checks={(a['context'],a['target']):a for a in audit['contexts']}
    actual=[]
    for r in windows:
        a=checks[(r['context'],r['target'])]
        actual.append({**r,'C_Sonly_audit':a['status'],'REPLICA_BOUND_ARTIFICIAL':a['REPLICA_BOUND_ARTIFICIAL']})
    csvsave('overflow_windows.csv',actual);save('replica_capacity_audit.json',audit)
    candidates=[r for r in actual if float(r['W'])>1e-7 and r['C_Sonly_audit']=='PASS']
    if not candidates:
        save('primary_hotspot_selection.json',{'status':'NO_CERTIFIED_PRIMARY_CONTEXT','candidate_audit':actual,'new_hotspot_GPU_runs':0})
        return
    multi=[r for r in candidates if int(r['C_ALL_N_G'])>=2];pool=multi or candidates
    selected=min(pool,key=lambda r:(-float(r['R']),-float(r['W']),float(r['H3_target_rate']),float(r['memory_budget_mib']),r['target'],r['background'],r['context']))
    con=ctx[selected['context']];cs=float(selected['C_Sonly']);ca=float(selected['C_ALL']);W=ca-cs
    frozen={'status':'PRIMARY_FROZEN','at_utc':now(),'context':con,'target':selected['target'],'C_Sonly':cs,'C_ALL':ca,'W':W,'R':W/cs,
            'specialist_only_witness':fits[(selected['context'],selected['target'],'S_ONLY')],
            'all_model_witness':fits[(selected['context'],selected['target'],'ALL')],
            'replica_audit':checks[(selected['context'],selected['target'])],
            'certified_positive_contexts':[{k:r[k] for k in ['context','target','R','W','C_ALL_N_G']} for r in candidates],
            'certified_multi_G_context_exists':bool(multi),'uncertified_multi_G_contexts_not_used':True,
            'selection_rule':freeze['selection_rule'],'new_replica_capacity_profiles':0,'new_hotspot_GPU_results_seen':0,
            'why_no_replica_profiles':'The mechanically selected certified context has no memory-feasible missing target-Specialist count, even under an optimistic minimum-background capability cover. Other contexts remain explicitly uncertified; they are not silently called exhausted.'}
    save('primary_hotspot_selection.json',frozen)
    points=[]
    for name,rate in [('H0',.8*cs),('H1',.98*cs),('H2',cs+.3*W),('H3',cs+.7*W),('H4',1.02*ca)]:
        lam=dict(con['background_lambda']);lam[selected['target']]=rate
        points.append({'point':name,'target_rate':rate,'lambda_rps':lam,'formal_GPU_point':name!='H4','above_C_Sonly':rate>cs,'above_C_ALL':rate>ca})
    save('hotspot_points.json',{'frozen':True,'at_utc':now(),'target':selected['target'],'context':con,'C_Sonly':cs,'C_ALL':ca,'W':W,'points':points,'movable_after_GPU':False})
    print(json.dumps({'PRIMARY':con['id'],'target':selected['target'],'C_Sonly':cs,'C_ALL':ca,'R':W/cs,'multi_G':bool(multi),'hotspot_rates':{r['point']:r['target_rate'] for r in points}}),flush=True)
if __name__=='__main__':main()
